"""
One-time conversion: split a Qwen3-VL embedder into backbone + head checkpoints.

  Backbone:  Qwen3VLModel with `language_model.layers[:split]` and
             `text_config.num_hidden_layers = split`. Saved via
             `save_pretrained` — loads with `Qwen3VLModel.from_pretrained`.
             The original final `norm` stays in the checkpoint (so the
             default loader is happy); at inference time it is bypassed by
             capturing the last layer's output via a forward-hook.

  Head:      `MultiLoRAEmbedder.HeadModel` — the same tail layers + final
             norm, exposed at the same submodule paths the full model had
             so LoRA adapter checkpoints load unchanged.

Usage:
    python split_model.py --src <hub_id_or_path> --out split/ --split 28
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from MultiLoRAEmbedder import HeadModel


def split_model(src: str, out: Path, split: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    backbone_dst = out / "backbone"
    head_dst = out / "head"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    inner = model.model                      # Qwen3VLModel
    text = inner.language_model              # Qwen3VLTextModel
    num_total = len(text.layers)
    assert 0 < split < num_total, f"split must be in 1..{num_total - 1}"

    # ---------- head ----------
    # Build the head shell, then graft the real tail layers + norm onto it.
    # Sharing module instances means `head.state_dict()` carries the actual
    # pretrained tail weights, keyed at the original paths.
    head = HeadModel(text.config, split=split, num_total_layers=num_total)
    for i in range(split, num_total):
        head.model.language_model.layers[i] = text.layers[i]
    head.model.language_model.norm = text.norm
    head.save_pretrained(head_dst)

    # ---------- backbone ----------
    text.layers = nn.ModuleList(text.layers[:split])
    text.config.num_hidden_layers = split
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = split
    inner.save_pretrained(backbone_dst, safe_serialization=True)
    AutoProcessor.from_pretrained(src, trust_remote_code=True).save_pretrained(backbone_dst)

    print(f"Backbone -> {backbone_dst}  ({split} layers)")
    print(f"Head     -> {head_dst}      ({num_total - split} layers)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source Qwen3-VL model path or hub id")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--split", type=int, default=28, help="Layer split index (default: 28)")
    args = p.parse_args()
    split_model(args.src, Path(args.out), args.split)
