"""
Benchmark + numerical check: sequential head loop vs batched fan-out.

Verifies that `_run_head_batched` (PEFT mixed-batch) produces the same
embeddings as the per-adapter loop, and times both.

Usage:
    python bench_head.py \
        --base /path/to/qwen3vl \
        --adapters /path/to/adapters_dir \
        [--image /path/to/img.jpg] \
        [--prompt "describe the image"] \
        [--split 28] [--iters 20]

--adapters may be:
  * a directory whose subdirectories each contain an adapter_config.json, or
  * a path to a JSON file mapping {name: path}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from multilora_embedder_fa2 import MultiLoRAEmbedder


def discover_adapters(spec: str) -> dict[str, str]:
    p = Path(spec)
    if p.is_file() and p.suffix == ".json":
        return json.loads(p.read_text())
    if p.is_dir():
        found = {
            sub.name: str(sub)
            for sub in sorted(p.iterdir())
            if (sub / "adapter_config.json").exists()
        }
        if found:
            return found
    raise SystemExit(f"No adapters found at {spec!r}")


def build_messages(prompt: str, image_path: str | None):
    content = []
    images = None
    if image_path:
        from PIL import Image

        content.append({"type": "image"})
        images = [Image.open(image_path).convert("RGB")]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], images


@torch.inference_mode()
def time_call(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="Describe this in one word.")
    ap.add_argument("--split", type=int, default=28)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    adapters = discover_adapters(args.adapters)
    print(f"Loaded {len(adapters)} adapters: {list(adapters)}")

    emb = MultiLoRAEmbedder(
        base_model=args.base,
        adapter_paths=adapters,
        split_layer=args.split,
        attn_implementation=args.attn,
    )

    messages, images = build_messages(args.prompt, args.image)
    model_inputs, attn_mask = emb._prepare(messages, images)
    hidden, layer_kwargs = emb._run_backbone(model_inputs)
    print(f"backbone hidden: {tuple(hidden.shape)}  | adapters N={len(adapters)}")

    # ---------- numerical equivalence ----------
    seq_out = {
        n: emb._run_head(hidden, layer_kwargs, attn_mask, n) for n in emb.adapter_names
    }
    bat_out = emb._run_head_batched(hidden, layer_kwargs, attn_mask)

    print("\nnumerical check (sequential vs batched):")
    worst_cos, worst_abs = 1.0, 0.0
    for n in emb.adapter_names:
        a, b = seq_out[n].float(), bat_out[n].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
        mad = (a - b).abs().max().item()
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, mad)
    print(f"  worst cosine sim : {worst_cos:.6f}  (want ~1.0)")
    print(f"  worst max-abs-dif: {worst_abs:.4e}  (bf16 noise ~1e-2)")

    # ---------- timing ----------
    def run_seq():
        for n in emb.adapter_names:
            emb._run_head(hidden, layer_kwargs, attn_mask, n)

    def run_bat():
        emb._run_head_batched(hidden, layer_kwargs, attn_mask)

    ms_seq = time_call(run_seq, args.iters)
    ms_bat = time_call(run_bat, args.iters)

    # full pipeline incl. backbone, for context
    def run_full_seq():
        emb.embed(messages, images, batched=False)

    def run_full_bat():
        emb.embed(messages, images, batched=True)

    ms_full_seq = time_call(run_full_seq, args.iters)
    ms_full_bat = time_call(run_full_bat, args.iters)

    print("\ntiming (ms/iter):")
    print(f"  head sequential : {ms_seq:8.2f}")
    print(f"  head batched    : {ms_bat:8.2f}   speedup x{ms_seq / ms_bat:.2f}")
    print(f"  full sequential : {ms_full_seq:8.2f}")
    print(f"  full batched    : {ms_full_bat:8.2f}   speedup x{ms_full_seq / ms_full_bat:.2f}")
    print(f"\n  peak GPU mem    : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
