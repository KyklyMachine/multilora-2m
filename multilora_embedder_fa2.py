"""
Multi-LoRA shared-backbone inference for a Qwen3-VL embedder.

Idea:
  - Run layers [0 .. split) once  -> heavy shared backbone.
  - Run layers [split .. end] + final norm + pool for every LoRA adapter.

Two head strategies live here:
  * `_run_head`          — sequential: set_adapter -> 8 layers, once per adapter.
                           Kept as the numerical reference.
  * `_run_head_batched`  — fan-out: replicate the backbone hidden state to a
                           batch of N*B and run the tail ONCE, routing each
                           sub-batch to its adapter via PEFT mixed-batch
                           (`_enable_peft_forward_hooks(adapter_names=...)`).
                           The shared base GEMMs run once over the whole N*B
                           batch; only the tiny rank-16 LoRA deltas are
                           per-adapter. Same FLOPs as the loop, but one big
                           batched kernel instead of N launch-bound passes.

Tiling note (verified against the modeling source):
  hidden [B, seq, h], attention_mask [B, 1, q, kv] (sdpa) and position_ids
  [B, seq] all carry batch on dim 0; rotary cos/sin are [B, seq, head_dim]
  (the mrope "3" axis is collapsed by apply_interleaved_mrope) — also dim 0.
  cache_position is [seq] with no batch axis. So: tile dim 0 for every tensor
  with ndim >= 2, pass 1-D tensors through unchanged.
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
        attn_implementation: str = "sdpa",
    ):
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        # `attn_implementation` is applied to the whole model (vision + text).
        # "sdpa" dispatches to a flash/mem-efficient kernel internally, without
        # the transformers FA2 unpad/repad overhead — for prefill-only embedding
        # over short/medium sequences it tends to match or beat "flash_attention_2",
        # and it composes with torch.compile. Use "flash_attention_2" only if you
        # move to long sequences + large padded batches.
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

    # ---------- preprocessing ----------

    def _prepare(self, messages, images=None, **processor_kwargs):
        """messages -> (model_inputs on device, attention_mask)."""
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
        return model_inputs, model_inputs["attention_mask"]

    # ---------- backbone ----------

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

    # ---------- head: sequential (reference) ----------

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

    # ---------- head: batched fan-out ----------

    @staticmethod
    def _tile_batch(t, n: int):
        """Tile dim-0 n times for tensors with a batch axis; pass others through."""
        if torch.is_tensor(t):
            return t.repeat(n, *([1] * (t.ndim - 1))) if t.ndim >= 2 else t
        if isinstance(t, tuple):
            return tuple(MultiLoRAEmbedder._tile_batch(x, n) for x in t)
        return t

    def _tile_kwargs(self, layer_kwargs: dict, n: int) -> dict:
        return {k: self._tile_batch(v, n) for k, v in layer_kwargs.items()}

    @torch.inference_mode()
    def _run_head_batched(
        self,
        hidden: torch.Tensor,
        layer_kwargs: dict,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """All adapters in a single N*B pass. Returns {name: tensor[B, hidden]}."""
        names = self.adapter_names
        n, b = len(names), hidden.shape[0]

        # Block layout: rows [k*B : (k+1)*B] are copy k -> adapter k.
        h = hidden.repeat(n, *([1] * (hidden.ndim - 1)))
        tiled = self._tile_kwargs(layer_kwargs, n)
        attn = attention_mask.repeat(n, *([1] * (attention_mask.ndim - 1)))
        adapter_names = [name for name in names for _ in range(b)]

        # Mixed-batch: base GEMM once over N*B, per-adapter LoRA delta per sub-batch.
        with self.model.base_model._enable_peft_forward_hooks(adapter_names=adapter_names):
            for layer in self.layers[self.split:]:
                out = layer(h, **tiled)
                h = out[0] if isinstance(out, tuple) else out
            h = self.final_norm(h)

        pooled = self._pool(h, attn)  # [N*B, hidden]
        return {name: pooled[i * b:(i + 1) * b] for i, name in enumerate(names)}

    # ---------- pooling ----------

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last *real* token per row.

        Assumes right padding (the Qwen-VL processor default) or batch=1, where
        the last non-pad token is at index `sum(mask) - 1`.
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
        batched: bool = True,
        **processor_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        One backbone pass, then the head for every loaded adapter.

        batched=True  -> single N*B fan-out pass (fast, default).
        batched=False -> sequential per-adapter loop (reference).

        Returns {adapter_name: tensor[B, hidden]}.
        """
        model_inputs, attention_mask = self._prepare(messages, images, **processor_kwargs)
        hidden, layer_kwargs = self._run_backbone(model_inputs)

        if batched:
            return self._run_head_batched(hidden, layer_kwargs, attention_mask)
        return {
            name: self._run_head(hidden, layer_kwargs, attention_mask, name)
            for name in self.adapter_names
        }
