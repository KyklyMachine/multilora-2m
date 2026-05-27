"""
example_usage.py
================

Demonstrates MultiLoRAEmbedder with:
  1. Real Qwen2VL + PEFT adapters
  2. Synthetic (CPU) smoke-test — runs without any GPU / real checkpoints
  3. Benchmark comparison

Run the smoke-test right now:
    python example_usage.py --smoke

Run with real model (needs GPU + checkpoints):
    python example_usage.py --real
"""

import argparse
import torch
import torch.nn as nn
from typing import Dict


# ── Smoke-test: tiny synthetic model ────────────────────────────────────────

class _FakeLMOutput:
    """Mimics HF model output so hooks work identically."""
    def __init__(self, hidden_states=None):
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1] if hidden_states else None


class _FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states),)


class _FakeLoRALayer(nn.Module):
    """Fake layer with a named 'lora_weight' so it looks adapter-specific."""
    def __init__(self, d):
        super().__init__()
        self.proj       = nn.Linear(d, d, bias=False)
        self.lora_down  = nn.Linear(d, 4, bias=False)
        self.lora_up    = nn.Linear(4, d, bias=False)

    def forward(self, hidden_states, **kwargs):
        out = self.proj(hidden_states) + self.lora_up(self.lora_down(hidden_states)) * 0.1
        return (out,)


class _FakeModel(nn.Module):
    """
    Minimal transformer-like model: embed → N shared layers → M lora layers → norm.
    Structured to match the path MultiLoRAEmbedder._get_layers() navigates.
    """
    def __init__(self, vocab=100, d=64, n_shared=4, n_lora=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            [_FakeLayer(d) for _ in range(n_shared)]
            + [_FakeLoRALayer(d) for _ in range(n_lora)]
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)[0]
        h = self.norm(h)
        return _FakeLMOutput()


class _FakeTopModel(nn.Module):
    """Wraps _FakeModel under .model.layers  (mirrors Qwen2VL path)."""
    def __init__(self, **kw):
        super().__init__()
        self.model = _FakeModel(**kw)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids, attention_mask, **kwargs)


class _FakePeftModel(nn.Module):
    """
    Minimal PeftModel shim:
      - .base_model.model  →  _FakeTopModel
      - .set_adapter()     →  changes active_adapter
      - .disable_adapter() →  context manager (no-op here)
    """
    def __init__(self, n_shared=4, n_lora=2, d=64):
        super().__init__()
        self._top        = _FakeTopModel(n_shared=n_shared, n_lora=n_lora, d=d)
        self.active_adapter = "default"
        # mirror PeftModel structure so _get_layers() works
        self.base_model      = self
        self.model           = self._top

    def set_adapter(self, name: str):
        self.active_adapter = name

    from contextlib import contextmanager

    @contextmanager
    def disable_adapter(self):
        yield   # fake: no-op

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self._top(input_ids, attention_mask, **kwargs)


def _smoke_test():
    """
    CPU smoke-test: verify that MultiLoRAEmbedder produces outputs and
    that the split+fan-out matches a naive sequential pass.
    """
    from multi_lora_inference import MultiLoRAEmbedder

    print("=" * 60)
    print("Smoke-test (synthetic model, CPU)")
    print("=" * 60)

    D        = 64
    N_SHARED = 4
    N_LORA   = 2
    SPLIT    = N_SHARED   # LoRA starts here

    fake_model   = _FakePeftModel(n_shared=N_SHARED, n_lora=N_LORA, d=D)
    adapter_names = ["adapter_a", "adapter_b", "adapter_c"]

    embedder = MultiLoRAEmbedder(
        model         = fake_model,
        adapter_names = adapter_names,
        split_layer   = SPLIT,
        pool          = "last",
    )

    batch, seq = 2, 8
    input_ids      = torch.randint(0, 100, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)

    print("\nRunning embed_all() …")
    embeddings = embedder.embed_all(
        input_ids      = input_ids,
        attention_mask = attention_mask,
    )

    print(f"\nResults:")
    for name, emb in embeddings.items():
        print(f"  {name}: shape={tuple(emb.shape)}, "
              f"norm={emb.norm(dim=-1).mean().item():.4f}")

    assert len(embeddings) == len(adapter_names)
    for name in adapter_names:
        assert embeddings[name].shape == (batch, D), \
            f"Wrong shape for {name}: {embeddings[name].shape}"

    print("\n✓ Smoke-test passed\n")


# ── Real model example ───────────────────────────────────────────────────────

def _real_example():
    """
    Full example with Qwen2VL-8B + PEFT adapters.
    Requires: GPU, Qwen2VL checkpoint, and trained LoRA adapters.
    """
    import os
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from multi_lora_inference import MultiLoRAEmbedder

    # ── Paths — edit these ──────────────────────────────────────────
    BASE_MODEL   = "Qwen/Qwen2-VL-7B-Instruct"          # or local path
    ADAPTER_ROOT = "./checkpoints"
    ADAPTER_NAMES = [f"adapter_{i:02d}" for i in range(30)]
    ADAPTER_PATHS = {name: os.path.join(ADAPTER_ROOT, name) for name in ADAPTER_NAMES}
    SPLIT_LAYER   = 28
    # ────────────────────────────────────────────────────────────────

    # Load processor for tokenisation
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # Build embedder (loads all adapters once)
    embedder = MultiLoRAEmbedder.from_pretrained(
        base_model_path = BASE_MODEL,
        adapter_paths   = ADAPTER_PATHS,
        split_layer     = SPLIT_LAYER,
        pool            = "last",
        torch_dtype     = torch.float16,
        device_map      = "auto",
        model_class     = Qwen2VLForConditionalGeneration,
    )

    # ── Prepare a batch ─────────────────────────────────────────────
    # Example: text-only query (drop pixel_values for text-only)
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc  = processor(text=[text], return_tensors="pt").to("cuda")

    # ── Embed ────────────────────────────────────────────────────────
    embeddings: Dict[str, torch.Tensor] = embedder.embed_all(**enc)

    print(f"\nEmbeddings for {len(embeddings)} adapters:")
    for name, emb in list(embeddings.items())[:3]:
        print(f"  {name}: shape={tuple(emb.shape)}")

    # ── Verify correctness ───────────────────────────────────────────
    print("\nRunning correctness check (comparing vs. naive per-adapter pass) …")
    ok = embedder.verify(**enc, atol=1e-2)
    print("Correctness:", "PASS ✓" if ok else "FAIL ✗")

    # ── Benchmark ────────────────────────────────────────────────────
    print("\nBenchmarking …")
    stats = embedder.benchmark(**enc, n_warmup=3, n_runs=20)
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run CPU smoke-test")
    parser.add_argument("--real",  action="store_true", help="Run real Qwen2VL example")
    args = parser.parse_args()

    if args.smoke or not (args.smoke or args.real):
        _smoke_test()

    if args.real:
        _real_example()
