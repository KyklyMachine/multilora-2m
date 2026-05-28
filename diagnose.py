"""
Compare v1 (full PEFT model) vs v2 (split backbone + head) on identical inputs.

Run with the *same* base model + adapters loaded into both. Reports
max/mean abs diff at three stages:
  1. backbone hidden state (= input to layer `split` in the original).
  2. captured layer kwargs (attention_mask, position_ids, position_embeddings, ...).
  3. final embeddings, per adapter.

A diff at stage 1 means the backbone passes disagree (visual fusion, embed,
layers 0..split-1). A diff at stage 2 means HF prepared different mask/RoPE
for the two paths. A diff only at stage 3 means the head (LoRA loading,
adapter switching) is the culprit.

Numerical noise from bf16 lives around 1e-2..1e-3 max-abs on hidden states
and ~1e-3 on pooled embeddings — anything substantially larger is a bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from MultiLoRAEmbedder import MultiLoRAEmbedder  # the new (split) version


# ---------------------------------------------------------------- v1 ----

def load_v1(base_model: str, adapter_paths: dict[str, str], split: int,
            dtype=torch.bfloat16, device="cuda"):
    """Reproduce the original full-model embedder in one place so this
    diagnostic is self-contained — no dependency on the old file."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    names = list(adapter_paths)
    model = PeftModel.from_pretrained(base, adapter_paths[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(adapter_paths[n], adapter_name=n)
    model.eval()

    decoder = model.base_model.model.model.language_model
    return processor, model, decoder, names


class _StopAtSplit(Exception):
    pass


@torch.inference_mode()
def v1_run_backbone(model, decoder, split: int, model_inputs: dict):
    """Mirror the original v1 hook-trick: hook layer[split].pre, capture
    hidden_states and layer_kwargs, short-circuit via exception."""
    captured: dict = {}

    def pre_hook(module, args, kwargs):
        captured["hidden"] = args[0] if args else kwargs["hidden_states"]
        captured["kwargs"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _StopAtSplit()

    h = decoder.layers[split].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        try:
            model(**model_inputs, use_cache=False)
        except _StopAtSplit:
            pass
    finally:
        h.remove()
    return captured["hidden"], captured["kwargs"]


@torch.inference_mode()
def v1_run_head(model, decoder, split, hidden, layer_kwargs, attention_mask, adapter):
    model.set_adapter(adapter)
    h = hidden
    for layer in decoder.layers[split:]:
        out = layer(h, **layer_kwargs)
        h = out[0] if isinstance(out, tuple) else out
    h = decoder.norm(h)
    last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
    rows = torch.arange(h.shape[0], device=h.device)
    return h[rows, last_idx]


# ---------------------------------------------------------------- diff --

def _diff(a, b, name: str):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        ok = a == b
        print(f"  {name:30s}  equal={ok}  v1={a!r}  v2={b!r}")
        return
    if a.shape != b.shape:
        print(f"  {name:30s}  SHAPE MISMATCH  v1={tuple(a.shape)}  v2={tuple(b.shape)}")
        return
    d = (a.float() - b.float()).abs()
    print(f"  {name:30s}  shape={tuple(a.shape)}  max={d.max().item():.3e}  mean={d.mean().item():.3e}")


def compare(base_model: str, backbone_path: str, head_path: str,
            adapter_paths: dict[str, str], split: int,
            text: str = "describe the picture"):
    # v2
    e2 = MultiLoRAEmbedder(
        backbone_path=backbone_path, head_path=head_path,
        adapter_paths=adapter_paths,
    )
    # v1
    processor, v1_model, v1_decoder, names = load_v1(
        base_model, adapter_paths, split,
    )

    # same processor input on both sides
    msgs = [[{"role": "user", "content": [{"type": "text", "text": text}]}]]
    rendered = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if isinstance(rendered, str):
        rendered = [rendered]
    inputs = processor(text=rendered, return_tensors="pt", padding=True).to(v1_model.device)

    print("\n=== Stage 1+2: backbone hidden + layer kwargs ===")
    h1, k1 = v1_run_backbone(v1_model, v1_decoder, split, inputs)
    h2, k2 = e2._run_backbone(inputs)
    _diff(h1, h2, "hidden")
    common = sorted(set(k1) | set(k2))
    for key in common:
        if key not in k1: print(f"  {key:30s}  MISSING in v1"); continue
        if key not in k2: print(f"  {key:30s}  MISSING in v2"); continue
        v1, v2 = k1[key], k2[key]
        if isinstance(v1, tuple) and isinstance(v2, tuple):
            for i, (a, b) in enumerate(zip(v1, v2)):
                _diff(a, b, f"{key}[{i}]")
        else:
            _diff(v1, v2, key)

    print("\n=== Stage 3: final pooled embeddings, per adapter ===")
    attn = inputs["attention_mask"]
    for name in names:
        emb1 = v1_run_head(v1_model, v1_decoder, split, h1, k1, attn, name)
        emb2 = e2._run_head(h2, k2, attn, name)
        _diff(emb1, emb2, f"emb[{name}]")
        # Also cosine, since pooled embeddings are often compared by direction.
        cos = torch.nn.functional.cosine_similarity(
            emb1.float().flatten(1), emb2.float().flatten(1), dim=1,
        ).mean().item()
        print(f"  emb[{name}] mean cos-sim = {cos:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Original Qwen3-VL base model path")
    p.add_argument("--backbone", required=True, help="Split backbone dir")
    p.add_argument("--head", required=True, help="Split head dir")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="Adapter paths, e.g. /path/lora_adapter_0 /path/lora_adapter_1 ...")
    p.add_argument("--split", type=int, default=28)
    args = p.parse_args()
    adapter_paths = {f"adapter_{i}": pth for i, pth in enumerate(args.adapters)}
    compare(args.base, args.backbone, args.head, adapter_paths, args.split)
