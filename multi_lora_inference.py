"""
multi_lora_inference.py
=======================

Efficient multi-LoRA inference: single shared backbone pass + fan-out to N adapters.

Designed for Qwen3-VL-8B with LoRA on layers 28-35, but works with any
transformers model where LoRA targets only the tail layers.

Architecture
------------
                              ┌──► [Adapter 0: layers 28-35] ──► embed_0
  Input ──► [Layers 0-27] ───┼──► [Adapter 1: layers 28-35] ──► embed_1
          (computed once)     │         ...
                              └──► [Adapter N: layers 28-35] ──► embed_N

Speedup (N=30, split=28, total=36)
-----------------------------------
  Naive:   36 × 30 = 1 080 layer-passes
  Shared:  28 × 1  + 8 × 30 = 268 layer-passes  →  ~4× speedup

Key technique: two PyTorch forward hooks
  1. Pre-hook on layer[0]        → capture kwargs (4-D attn mask, pos embeddings, etc.)
  2. Post-hook on layer[split-1] → capture hidden states at the split point

This way we never duplicate model-internal logic (M-ROPE, visual token merging, …).

Usage
-----
    from multi_lora_inference import MultiLoRAEmbedder
    from transformers import Qwen2VLForConditionalGeneration

    embedder = MultiLoRAEmbedder.from_pretrained(
        base_model_path="Qwen/Qwen3-VL-8B",
        adapter_paths={
            "style_v1": "/checkpoints/style_v1",
            "domain_v2": "/checkpoints/domain_v2",
            # ... up to 30 adapters
        },
        split_layer=28,
        model_class=Qwen2VLForConditionalGeneration,
    )

    embeddings = embedder.embed_all(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,         # optional
        image_grid_thw=image_grid_thw,     # Qwen2VL specific, optional
    )
    # embeddings: {"style_v1": Tensor[batch, 4096], "domain_v2": Tensor[batch, 4096], ...}
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import PeftModel


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MultiLoRAEmbedder:
    """
    Shared-backbone multi-LoRA embedder.

    Parameters
    ----------
    model       : PeftModel wrapping any HuggingFace transformer.
    adapter_names: Ordered list of adapter names (must all be loaded).
    split_layer : Index of the first "adapter-only" layer.
                  Layers [0, split_layer) are shared; [split_layer, end) are per-adapter.
    pool        : Pooling strategy for the final hidden state.
                  "last"  – last non-padding token  (default, good for causal LMs)
                  "mean"  – mean over non-padding tokens
                  "first" – first token (CLS-style)
    """

    def __init__(
        self,
        model: PeftModel,
        adapter_names: List[str],
        split_layer: int = 28,
        pool: str = "last",
    ) -> None:
        assert pool in ("last", "mean", "first"), f"Unknown pool: {pool}"
        self.model = model
        self.adapter_names = list(adapter_names)
        self.split_layer = split_layer
        self.pool = pool
        self.model.eval()

        # Populated by hooks during _run_shared_pass
        self._split_hidden: Optional[torch.Tensor] = None
        self._tail_kwargs: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        adapter_paths: Dict[str, str],
        split_layer: int = 28,
        pool: str = "last",
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        model_class=None,
    ) -> "MultiLoRAEmbedder":
        """
        Load base model + all LoRA adapters from disk.

        Parameters
        ----------
        base_model_path : HF hub name or local path to base model.
        adapter_paths   : {adapter_name: path_or_hub_id}
        split_layer     : First layer that has LoRA weights.
        pool            : Pooling strategy ("last" | "mean" | "first").
        torch_dtype     : Dtype for model weights (float16 recommended for GPU).
        device_map      : Passed to from_pretrained ("auto", "cuda:0", …).
        model_class     : Model class (e.g. Qwen2VLForConditionalGeneration).
                          Falls back to AutoModel if None.
        """
        if model_class is None:
            from transformers import AutoModel
            model_class = AutoModel

        print(f"Loading base model: {base_model_path}")
        base = model_class.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        names = list(adapter_paths.keys())
        print(f"Loading {len(names)} LoRA adapters …")

        # First adapter creates the PeftModel wrapper
        model = PeftModel.from_pretrained(
            base,
            adapter_paths[names[0]],
            adapter_name=names[0],
        )

        # Remaining adapters are loaded into the same wrapper
        for name in names[1:]:
            model.load_adapter(adapter_paths[name], adapter_name=name)
            print(f"  ✓ {name}")

        print(f"All adapters loaded. split_layer={split_layer}, pool={pool}")
        return cls(model, names, split_layer, pool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layers(self) -> nn.ModuleList:
        """Navigate to transformer decoder layers (Qwen2/Qwen3 family)."""
        m = self.model.base_model.model       # Qwen2VLForConditionalGeneration
        try:
            return m.model.layers            # Qwen2VLModel → layers
        except AttributeError:
            try:
                return m.layers              # Some models expose it directly
            except AttributeError:
                raise RuntimeError(
                    "Cannot find transformer layers. "
                    "Override _get_layers() for your model architecture."
                )

    def _get_final_norm(self) -> nn.Module:
        """Return the final layer norm (applied after all transformer layers)."""
        m = self.model.base_model.model
        try:
            return m.model.norm
        except AttributeError:
            return m.norm

    def _pool(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pool [batch, seq, hidden] → [batch, hidden]."""
        if self.pool == "first":
            return hidden[:, 0, :]

        if self.pool == "last":
            if attention_mask is None:
                return hidden[:, -1, :]
            # Find last non-padding position per sample
            seq_lens = attention_mask.sum(dim=1) - 1          # [batch]
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, seq_lens]

        if self.pool == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> List[Any]:
        """
        Register two hooks and return their handles.

        Hook 1 – pre-hook on layer[0]:
            Captures kwargs passed to all decoder layers
            (4-D attention mask, position IDs, rotary embeddings, …).
            These are identical for every layer and every adapter,
            so we compute them once and reuse.

        Hook 2 – post-hook on layer[split_layer - 1]:
            Captures hidden states at the end of the shared backbone.
        """
        layers = self._get_layers()
        handles: List[Any] = []

        # --- Hook 1: capture layer kwargs from first layer ---
        def _pre_hook_layer0(module, args, kwargs):
            # Filter to only the kwargs relevant for tail layers.
            # We drop: past_key_value (None anyway), use_cache, output_attentions
            # to let each layer use its own defaults.
            safe = {}
            _skip = {"past_key_value", "use_cache", "output_attentions"}
            for k, v in kwargs.items():
                if k not in _skip:
                    safe[k] = v
            safe["use_cache"] = False
            safe["output_attentions"] = False
            self._tail_kwargs = safe
            return args, kwargs   # pass through unchanged

        handles.append(
            layers[0].register_forward_pre_hook(_pre_hook_layer0, with_kwargs=True)
        )

        # --- Hook 2: capture hidden state after last shared layer ---
        def _post_hook_split(module, args, output):
            # output[0] is hidden_states for all standard decoder layers
            self._split_hidden = output[0].detach().clone()

        handles.append(
            layers[self.split_layer - 1].register_forward_hook(_post_hook_split)
        )

        return handles

    # ------------------------------------------------------------------
    # Two-phase forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_shared_pass(self, inputs: Dict[str, Any]) -> None:
        """
        Phase 1 – run the full model once (adapters disabled).

        Even though LoRA layers only exist in [split_layer, end), we disable
        adapters explicitly to be safe and avoid any overhead.

        Side effects: populates self._split_hidden and self._tail_kwargs.
        """
        handles = self._register_hooks()
        try:
            with self.model.disable_adapter():
                self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        assert self._split_hidden is not None, "Hook failed to capture split hidden states."
        assert self._tail_kwargs is not None,  "Hook failed to capture layer kwargs."

    @torch.no_grad()
    def _run_adapter_head(
        self,
        adapter_name: str,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Phase 2 – run layers [split_layer, end) with one specific adapter.

        Starts from the shared hidden states captured in Phase 1.
        Returns pooled embedding: [batch, hidden_size].
        """
        self.model.set_adapter(adapter_name)

        layers   = self._get_layers()
        hidden   = self._split_hidden.clone()
        kwargs   = self._tail_kwargs          # shared tensors, no copy needed

        for layer in layers[self.split_layer:]:
            out    = layer(hidden, **kwargs)
            hidden = out[0]

        # Apply final layer norm
        hidden = self._get_final_norm()(hidden)

        # Pool to single embedding vector
        return self._pool(hidden, attention_mask)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_all(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute embeddings for all adapters with shared backbone.

        Parameters
        ----------
        input_ids      : [batch, seq_len]
        attention_mask : [batch, seq_len]  (optional but recommended)
        pixel_values   : Visual inputs for VL models (optional)
        **kwargs       : Extra model inputs (e.g. image_grid_thw for Qwen2VL)

        Returns
        -------
        {adapter_name: Tensor[batch, hidden_size]}
        """
        # Build input dict
        model_inputs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        model_inputs.update(kwargs)

        # ── Phase 1: shared backbone ─────────────────────────────────
        self._run_shared_pass(model_inputs)

        # ── Phase 2: adapter fan-out ──────────────────────────────────
        embeddings: Dict[str, torch.Tensor] = {}
        for name in self.adapter_names:
            embeddings[name] = self._run_adapter_head(name, attention_mask)

        return embeddings

    @torch.no_grad()
    def embed_single(
        self,
        adapter_name: str,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Embed with a single specific adapter (useful for testing / validation).
        Runs the full model normally (no split optimisation).
        """
        self.model.set_adapter(adapter_name)
        model_inputs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        model_inputs.update(kwargs)

        out    = self.model(**model_inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1]                     # after last layer
        hidden = self._get_final_norm()(hidden)
        return self._pool(hidden, attention_mask)

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        atol: float = 1e-3,
        **kwargs,
    ) -> bool:
        """
        Sanity-check: compare embed_all() vs embed_single() for every adapter.
        Returns True if all embeddings match within tolerance.
        """
        fast = self.embed_all(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )

        all_ok = True
        for name in self.adapter_names:
            ref   = self.embed_single(name, input_ids, attention_mask, pixel_values, **kwargs)
            delta = (fast[name] - ref).abs().max().item()
            ok    = delta < atol
            status = "✓" if ok else "✗"
            print(f"  [{status}] {name}  max_abs_diff={delta:.2e}  (tol={atol})")
            if not ok:
                all_ok = False

        return all_ok

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    @torch.no_grad()
    def benchmark(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        n_warmup: int = 3,
        n_runs: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare latency: naive (N full passes) vs. shared backbone.

        Returns dict with timing stats and speedup metrics.
        """
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ── Warmup ───────────────────────────────────────────────────
        for _ in range(n_warmup):
            self.embed_all(**inputs)

        # ── Shared backbone (our approach) ───────────────────────────
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.embed_all(**inputs)
            _sync()
        ms_shared = (time.perf_counter() - t0) / n_runs * 1_000

        # ── Naive (one full forward per adapter) ─────────────────────
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            for name in self.adapter_names:
                self.embed_single(name, **inputs)
            _sync()
        ms_naive = (time.perf_counter() - t0) / n_runs * 1_000

        # ── Theoretical speedup ──────────────────────────────────────
        n      = len(self.adapter_names)
        total  = len(self._get_layers())
        tail   = total - self.split_layer
        theory = (total * n) / (self.split_layer + tail * n)

        result = {
            "n_adapters":           n,
            "total_layers":         total,
            "split_layer":          self.split_layer,
            "tail_layers":          tail,
            "ms_naive":             round(ms_naive,  1),
            "ms_shared":            round(ms_shared, 1),
            "speedup_measured":     round(ms_naive / ms_shared, 2),
            "speedup_theoretical":  round(theory, 2),
        }

        print("\n── Benchmark results ──────────────────────────────────")
        print(f"  Adapters        : {n}")
        print(f"  Layers          : {total}  (shared 0-{self.split_layer-1}, "
              f"tail {self.split_layer}-{total-1})")
        print(f"  Naive latency   : {ms_naive:.1f} ms")
        print(f"  Shared latency  : {ms_shared:.1f} ms")
        print(f"  Speedup         : {ms_naive/ms_shared:.2f}×  "
              f"(theory: {theory:.2f}×)")
        print("────────────────────────────────────────────────────────\n")

        return result


# ---------------------------------------------------------------------------
# Convenience: build from already-loaded PEFT model
# ---------------------------------------------------------------------------

def wrap_peft_model(
    model: PeftModel,
    split_layer: int = 28,
    pool: str = "last",
) -> MultiLoRAEmbedder:
    """
    Wrap an already-loaded PeftModel (with all adapters loaded).

    Example
    -------
        peft_model = PeftModel.from_pretrained(base, "adapter_0", adapter_name="a0")
        peft_model.load_adapter("adapter_1", adapter_name="a1")
        embedder = wrap_peft_model(peft_model, split_layer=28)
    """
    # Discover adapter names from PEFT's internal state
    names = list(model.peft_config.keys())
    return MultiLoRAEmbedder(model, names, split_layer, pool)
