"""
In-process LoRA benchmark for the transformers MultiLoRAEmbedder (split-backbone).

Mirrors `lora_benchmark.py` (vLLM/SGLang) so the resulting DataFrames are
directly comparable. Key differences vs the HTTP benchmark:

  * No async / network. Work is synchronous on a single GPU, so the
    vLLM `concurrency` knob is replaced by `batch_size`.
  * A "request" is one embedding for one (sample, adapter) pair. With a batch
    of B samples and N adapters, one `embed()` call produces B*N embeddings,
    so `rps` here means embeddings/sec — the same unit as the vLLM bench.
  * Latency is the wall time of the `embed()` call that produced an embedding
    (the moment it becomes available); every (sample, adapter) pair from a
    given call inherits that call's latency, matching the per-request latency
    semantics of the HTTP bench.
  * The N-adapter sweep is done by slicing `embedder.adapter_names` — all
    adapters stay loaded, only N heads are computed (the analogue of
    `all_lora_names[:n]` in the vLLM bench, where the server has every adapter
    loaded but a request hits N of them).

Usage:
    from multilora_embedder_fa2 import MultiLoRAEmbedder
    from transformers_benchmark import benchmark_lora_sweep

    embedder = MultiLoRAEmbedder(base_model, adapter_paths, split_layer=28)
    inputs = [(text, None), ...]          # list of (text, img_url) tuples
    df = benchmark_lora_sweep(
        embedder, list(adapter_paths), inputs,
        batch_sizes=[1, 8, 32],
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prompt_processing import get_embed_qwen_prompt

logger = logging.getLogger(__name__)

# (text, img_url) — same shape as the `inputs` used by lora_benchmark.py.
Sample = Tuple[str, Optional[str]]


@dataclass
class BenchmarkConfig:
    lora_names: List[str]
    batch_size: int
    n_warmup: int = 3      # warmup batches (not timed) to reach steady state
    n_repeats: int = 3     # timed passes over the full input set


# ---------- helpers ----------

def _batched(seq: Sequence[Sample], size: int) -> Iterator[List[Sample]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_messages(batch: Iterable[Sample]) -> List[List[Dict]]:
    """One conversation per sample; embed() accepts the batch directly."""
    return [get_embed_qwen_prompt(img_url=img_url, text=text) for text, img_url in batch]


def _compute_latency_stats(latencies: np.ndarray, n_total: int, elapsed: float) -> Dict:
    """Aggregate RPS and latency percentiles. Columns match lora_benchmark.py."""
    n_ok = len(latencies)

    row: Dict = {
        "n_requests": n_total,
        "n_success": n_ok,
        "error_rate": round((n_total - n_ok) / n_total, 4) if n_total else 0,
        "rps": round(n_ok / elapsed, 2) if elapsed > 0 else 0,
    }

    if n_ok:
        row.update({
            "mean_lat_s": round(float(latencies.mean()), 3),
            "p50_lat_s": round(float(np.percentile(latencies, 50)), 3),
            "p95_lat_s": round(float(np.percentile(latencies, 95)), 3),
            "p99_lat_s": round(float(np.percentile(latencies, 99)), 3),
        })
    else:
        row.update({k: None for k in ["mean_lat_s", "p50_lat_s", "p95_lat_s", "p99_lat_s"]})

    return row


# ---------- single config ----------

def _run_single_benchmark(
    embedder,
    config: BenchmarkConfig,
    inputs: Sequence[Sample],
) -> Dict:
    """Run one benchmark point: full input set, batched, over N adapters."""
    n_loras = len(config.lora_names)
    batches = [_build_messages(b) for b in _batched(inputs, config.batch_size)]

    saved_adapters = embedder.adapter_names
    embedder.adapter_names = config.lora_names
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        # Warmup: hit the exact (N adapters, batch_size) code path.
        for msgs in batches[: config.n_warmup]:
            embedder.embed(msgs)
        _sync()

        latencies: List[float] = []
        n_total = 0

        wall0 = time.perf_counter()
        for _ in range(config.n_repeats):
            for msgs in batches:
                pairs = len(msgs) * n_loras
                n_total += pairs
                t0 = time.perf_counter()
                try:
                    embedder.embed(msgs)
                    _sync()
                    # Every (sample, adapter) pair inherits this call's latency.
                    latencies.extend([time.perf_counter() - t0] * pairs)
                except Exception as e:  # noqa: BLE001 — record failure, keep sweeping
                    logger.warning("embed() failed (n_loras=%d, bs=%d): %s",
                                   n_loras, config.batch_size, e)
        elapsed = time.perf_counter() - wall0
    finally:
        embedder.adapter_names = saved_adapters

    stats = _compute_latency_stats(np.asarray(latencies), n_total, elapsed)
    stats.update({
        "n_loras": n_loras,
        "batch_size": config.batch_size,
        "elapsed_s": round(elapsed, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
    })
    return stats


# ---------- sweep ----------

def benchmark_lora_sweep(
    embedder,
    all_lora_names: List[str],
    inputs: Sequence[Sample],
    lora_counts: Optional[List[int]] = None,
    batch_sizes: Sequence[int] = (1, 8, 32),
    n_warmup: int = 3,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Sweep over (batch_size x n_loras) and return a summary table.

    Each point runs all `inputs` through the first N adapters of
    `all_lora_names`, batched at `batch_size`. Output columns are a superset of
    lora_benchmark.py's, so the two DataFrames can be concatenated/compared.
    """
    if lora_counts is None:
        lora_counts = list(range(1, len(all_lora_names) + 1))

    rows = []
    grid = [(bs, n) for bs in batch_sizes for n in lora_counts]
    for bs, n in tqdm(grid):
        if n > len(all_lora_names):
            logger.warning("Skipping n_loras=%d: only %d available.", n, len(all_lora_names))
            continue
        config = BenchmarkConfig(
            lora_names=all_lora_names[:n],
            batch_size=bs,
            n_warmup=n_warmup,
            n_repeats=n_repeats,
        )
        rows.append(_run_single_benchmark(embedder, config, inputs))

    return pd.DataFrame(rows)
