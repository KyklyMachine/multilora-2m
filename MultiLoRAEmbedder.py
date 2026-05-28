"""
Multi-LoRA shared-backbone inference for a Qwen3-VL embedder.

Architecture:
  Backbone (layers 0..split-1):  shared, no LoRA, runs once per request.
  Head     (layers split..end):  fan-out across N LoRA adapters.

The two stages live in two separate on-disk checkpoints — see `split_model.py`
for the one-time conversion. This keeps the runtime code free of hooks-as-
control-flow: the backbone is just a truncated `Qwen3VLModel`, and the head
is a small `nn.Module` that takes the hidden state plus the per-layer kwargs
HF prepared inside the original `Qwen3VLTextModel.forward` (causal mask,
text position ids, rotary cos/sin, cache_position).

The head's submodule paths mirror the original full-model layout
(`model.language_model.layers.{i}.*`, with `nn.Identity` stubs at `[0, split)`
so PEFT adapter checkpoints — keyed against the un-split paths — load
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
)
from peft import PeftModel
from safetensors.torch import load_file, save_file


HEAD_WEIGHTS = "head.safetensors"
HEAD_CONFIG = "head_config.json"


class HeadModel(nn.Module):
    """Tail layers + final norm, exposed at the same submodule paths the full
    Qwen3-VL had — so LoRA adapter checkpoints load unchanged.

    `model.language_model.layers` is a length-`num_total_layers` ModuleList:
    slots `[0, split)` are `nn.Identity` (no parameters, contribute nothing
    to `state_dict`), slots `[split, num_total_layers)` hold the real tail
    decoder layers.
    """

    def __init__(self, text_config: Qwen3VLTextConfig, split: int, num_total_layers: int):
        super().__init__()
        assert 0 < split < num_total_layers
        self.split = split
        self.num_total_layers = num_total_layers
        # Named `config` (not `text_config`) so PEFT's `model.config` introspection
        # finds a real Qwen3VLTextConfig when it wraps us.
        self.config = text_config

        text = nn.Module()
        stubs = [nn.Identity() for _ in range(split)]
        tail = [
            Qwen3VLTextDecoderLayer(text_config, layer_idx=i)
            for i in range(split, num_total_layers)
        ]
        text.layers = nn.ModuleList(stubs + tail)
        text.norm = Qwen3VLTextRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        outer = nn.Module()
        outer.language_model = text
        self.model = outer

    def forward(self, hidden_states: torch.Tensor, **layer_kwargs) -> torch.Tensor:
        """Run the tail decoder layers + final norm.

        `layer_kwargs` are the kwargs `Qwen3VLTextModel.forward` would have
        passed to each decoder layer: `attention_mask` (4D causal mask),
        `position_ids` (2D text mrope), `position_embeddings` ((cos, sin)),
        `cache_position`, ...
        """
        for layer in self.model.language_model.layers[self.split:]:
            out = layer(hidden_states, **layer_kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return self.model.language_model.norm(hidden_states)

    # ---------- persistence ----------

    def save_pretrained(self, dst: str | Path) -> None:
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), dst / HEAD_WEIGHTS)
        (dst / HEAD_CONFIG).write_text(json.dumps({
            "split": self.split,
            "num_total_layers": self.num_total_layers,
            "text_config": self.config.to_dict(),
        }, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        src: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> "HeadModel":
        src = Path(src)
        meta = json.loads((src / HEAD_CONFIG).read_text())
        config = Qwen3VLTextConfig(**meta["text_config"])
        head = cls(config, split=meta["split"], num_total_layers=meta["num_total_layers"])
        head.load_state_dict(load_file(src / HEAD_WEIGHTS), strict=True)
        return head.to(device=device, dtype=dtype).eval()


class MultiLoRAEmbedder:
    """Single backbone pass, then one head pass per loaded LoRA adapter."""

    def __init__(
        self,
        backbone_path: str | Path,
        head_path: str | Path,
        adapter_paths: dict[str, str],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.processor = AutoProcessor.from_pretrained(backbone_path, trust_remote_code=True)
        self.backbone = Qwen3VLModel.from_pretrained(
            backbone_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        head_base = HeadModel.from_pretrained(head_path, dtype=dtype, device=device)
        names = list(adapter_paths)
        self.head = PeftModel.from_pretrained(
            head_base, adapter_paths[names[0]], adapter_name=names[0],
        )
        for n in names[1:]:
            self.head.load_adapter(adapter_paths[n], adapter_name=n)
        self.head.eval()
        self.adapter_names = names

        # Capture point: the last backbone decoder layer. Its output IS the
        # state we want to feed into the head, and the kwargs it receives are
        # the very kwargs each head layer expects.
        self._capture_layer = self.backbone.language_model.layers[-1]

    # ---------- two-stage forward ----------

    @torch.inference_mode()
    def _run_backbone(self, model_inputs: dict) -> tuple[torch.Tensor, dict]:
        captured: dict = {}

        def hook(module, args, kwargs, output):
            captured["hidden"] = output[0] if isinstance(output, tuple) else output
            captured["kwargs"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}

        handle = self._capture_layer.register_forward_hook(hook, with_kwargs=True)
        try:
            self.backbone(**model_inputs, use_cache=False)
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
        self.head.set_adapter(adapter)
        h = self.head(hidden, **layer_kwargs)
        return self._pool_last(h, attention_mask)

    @staticmethod
    def _pool_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last *real* token per row — robust to left/right padding."""
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_idx]

    # ---------- public API ----------

    @torch.inference_mode()
    def embed(self, messages, images=None, **processor_kwargs) -> dict[str, torch.Tensor]:
        """One backbone pass, then one head pass per loaded adapter.

        Returns {adapter_name: tensor[B, hidden]}.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if isinstance(text, str):
            text = [text]
        model_inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True,
            **processor_kwargs,
        ).to(self.backbone.device)
        attention_mask = model_inputs["attention_mask"]

        hidden, layer_kwargs = self._run_backbone(model_inputs)

        return {
            name: self._run_head(hidden, layer_kwargs, attention_mask, name)
            for name in self.adapter_names
        }
