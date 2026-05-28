"""
Multi-LoRA shared-backbone inference for a Qwen3-VL embedder.

Idea:
  - Run layers [0 .. split) once  -> heavy shared backbone.
  - For each LoRA adapter, run layers [split .. end] + final norm + pool
    with that adapter active. Cost grows with N, but only over the cheap tail.

First-cut implementation, meant to be debugged end-to-end against a real
model. No cross-adapter batching yet — that is the next optimization.
"""

from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
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
    ):
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
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
        self, hidden: torch.Tensor, layer_kwargs: dict, adapter: str
    ) -> torch.Tensor:
        self.model.set_adapter(adapter)
        h = hidden
        for layer in self.layers[self.split:]:
            h = layer(h, **layer_kwargs)
        h = self.final_norm(h)
        return self._pool(h)

    @staticmethod
    def _pool(hidden: torch.Tensor) -> torch.Tensor:
        """Last-token pooling. OK for batch=1 or right-padded inputs.
        For left-padded / mixed-length batches we'd need attention_mask here."""
        return hidden[:, -1, :]

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

        hidden, layer_kwargs = self._run_backbone(model_inputs)

        return {
            name: self._run_head(hidden, layer_kwargs, name)
            for name in self.adapter_names
        }
