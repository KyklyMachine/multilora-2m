"""
Multi-LoRA shared-backbone inference for a Qwen3-VL embedder. (FlashAttention-2)

Idea:
  - Run layers [0 .. split) once  -> heavy shared backbone.
  - For each LoRA adapter, run layers [split .. end] + final norm + pool
    with that adapter active. Cost grows with N, but only over the cheap tail.

This is the original single-model version with FlashAttention-2 enabled.
The only substantive changes vs the first cut:
  * `attn_implementation` is passed at load time (default "flash_attention_2").
  * pooling now uses `attention_mask` to pick the last *real* token, so it is
    correct under batching + padding (FA2 path relies on the 2D padding mask).

No cross-adapter batching yet — that is the next optimization.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


class _StopAtSplit(Exception):
    """Short-circuit signal raised by the layer-pre-hook in the backbone pass."""


class MultiLoRAEmbedder:
    def __init__(
        self,
        base_model: str,
        adapter_paths: dict[str, str],
        split_layer: int = 28,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
    ):
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        # FA2 requires fp16/bf16 weights. `attn_implementation` is applied to the
        # whole model (vision + text). Fall back to "sdpa" if flash-attn is not
        # installed / not supported on your GPU.
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        names = list(adapter_paths)
        self.model = PeftModel.from_pretrained(
            base, adapter_paths[names[0]], adapter_name=names[0]
        )
        for n in names[1:]:
            self.model.load_adapter(adapter_paths[n], adapter_name=n)
        self.model.eval()
        self.adapter_names = names
        self.split = split_layer

        decoder = self._find_decoder(self.model)
        self.layers = decoder.layers
        self.final_norm = decoder.norm
        assert len(self.layers) > self.split, "split_layer is beyond model depth"

    # ---------- model-tree navigation ----------

    @staticmethod
    def _find_decoder(m):
        """Recursively find the module that owns `.layers`, `.norm`, `.embed_tokens`."""

        def _search(module, depth=0):
            if depth > 20:  # защита от бесконечной рекурсии
                return None
            if all(hasattr(module, a) for a in ("layers", "norm", "embed_tokens")):
                return module
            for attr in ("base_model", "model", "language_model", "text_model", "transformer", "decoder"):
                if hasattr(module, attr):
                    result = _search(getattr(module, attr), depth + 1)
                    if result is not None:
                        return result
            return None

        result = _search(m)
        if result is None:
            raise AttributeError("cannot locate decoder module with .layers")
        return result

    # ---------- two-stage forward ----------

    @torch.inference_mode()
    def _run_backbone(self, model_inputs: dict) -> tuple[torch.Tensor, dict]:
        """
        Run the full model up to layer[split] by short-circuiting with a hook.
        HF handles all multimodal fusion + mask/position-id prep for us;
        we just capture what layer[split] would have received as input.

        Under FA2 the captured `attention_mask` is whatever HF actually passes
        the decoder layers (2D padding mask or None) — so the head replay stays
        consistent with the attention backend automatically.
        """
        captured: dict = {}

        def pre_hook(module, args, kwargs):
            captured["hidden"] = args[0] if args else kwargs["hidden_states"]
            captured["kwargs"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            raise _StopAtSplit()

        handle = self.layers[self.split].register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )
        try:
            with self.model.disable_adapter():
                try:
                    self.model(**model_inputs, use_cache=False)
                except _StopAtSplit:
                    pass
        finally:
            handle.remove()
        return captured["hidden"], captured["kwargs"]

    @torch.inference_mode()
    def _run_head(
        self,
        hidden: torch.Tensor,
        layer_kwargs: dict,
        attention_mask: torch.Tensor,
        adapter: str,
    ) -> torch.Tensor:
        self.model.set_adapter(adapter)
        h = hidden
        for layer in self.layers[self.split:]:
            out = layer(h, **layer_kwargs)
            h = out[0] if isinstance(out, tuple) else out
        h = self.final_norm(h)
        return self._pool(h, attention_mask)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last *real* token per row.

        Assumes right padding (the Qwen-VL processor default) or batch=1, where
        the last non-pad token is at index `sum(mask) - 1`. This is the correct
        replacement for `hidden[:, -1, :]` once you batch padded inputs.
        """
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_idx]

    # ---------- public API ----------

    @torch.inference_mode()
    def embed(
        self,
        messages,
        images=None,
        **processor_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        One backbone pass, then one head pass per loaded adapter.

        messages: OpenAI-style chat messages — single conversation (list of
                  dicts) or a batch (list of such lists).
        images:   PIL image or list of PIL images, forwarded to the processor.
        processor_kwargs: extra args forwarded to the processor call
                  (e.g. videos=..., padding=...).

        Returns {adapter_name: tensor[B, hidden]}.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if isinstance(text, str):
            text = [text]
        model_inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
            **processor_kwargs,
        ).to(self.model.device)
        attention_mask = model_inputs["attention_mask"]

        hidden, layer_kwargs = self._run_backbone(model_inputs)

        return {
            name: self._run_head(hidden, layer_kwargs, attention_mask, name)
            for name in self.adapter_names
        }
