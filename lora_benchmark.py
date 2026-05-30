"""
LoRA adapter benchmark for vLLM/SGLang.

Requirements:
    vLLM must be started with:
        --enable-lora --max-loras <N>
        --lora-modules name1=/path/to/lora1 name2=/path/to/lora2 ...
"""

import asyncio
import logging
import time
from asyncio import Semaphore
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from utils.prompt_processing import get_embed_qwen_prompt

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    lora_names: List[str]
    max_tokens: int = 16
    temperature: float = 0.0
    concurrency: int = 512
    n_warmup: int = 8  # untimed samples fanned out to all LoRAs before measuring


@dataclass
class _InferenceResult:
    """Outcome of a single vLLM request."""
    lora_name: str
    latency: float
    success: bool


async def _request_lora(
    client: AsyncOpenAI,
    sem: Semaphore,
    lora_name: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
) -> _InferenceResult:
    """Send one chat completion request to a specific LoRA adapter."""
    async with sem:
        t0 = time.perf_counter()
        try:
            await client.chat.completions.create(
                model=lora_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return _InferenceResult(lora_name=lora_name, latency=time.perf_counter() - t0, success=True)
        except Exception as e:
            logger.warning(f"[{lora_name}] request failed: {e}")
            return _InferenceResult(lora_name=lora_name, latency=time.perf_counter() - t0, success=False)


async def _request_all_loras(
    client: AsyncOpenAI,
    sem: Semaphore,
    lora_names: List[str],
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
) -> List[_InferenceResult]:
    """Fan out one sample to all LoRA adapters simultaneously."""
    return list(await asyncio.gather(*[
        _request_lora(client, sem, lora_name, messages, max_tokens, temperature)
        for lora_name in lora_names
    ]))


def _compute_latency_stats(results: List[_InferenceResult], elapsed: float) -> Dict:
    """Aggregate RPS and latency percentiles from a list of inference results."""
    latencies = np.array([r.latency for r in results if r.success])
    n_total, n_ok = len(results), len(latencies)

    row: Dict = {
        "n_requests": n_total,
        "n_success": n_ok,
        "error_rate": round((n_total - n_ok) / n_total, 4) if n_total else 0,
        "rps": round(n_ok / elapsed, 2) if elapsed > 0 else 0,
    }

    if len(latencies):
        row.update({
            "mean_lat_s": round(latencies.mean(), 3),
            "p50_lat_s": round(np.percentile(latencies, 50), 3),
            "p95_lat_s": round(np.percentile(latencies, 95), 3),
            "p99_lat_s": round(np.percentile(latencies, 99), 3),
        })
    else:
        row.update({k: None for k in ["mean_lat_s", "p50_lat_s", "p95_lat_s", "p99_lat_s"]})

    return row


async def _run_single_benchmark(
    client: AsyncOpenAI,
    config: BenchmarkConfig,
    inputs: List[Tuple[str, Optional[str]]],
) -> Dict:
    """Run one benchmark pass: fan out all samples to all LoRAs concurrently."""
    sem = Semaphore(config.concurrency)

    # Warmup: trigger adapter loading + kernel/graph capture on every LoRA so the
    # timed pass measures steady state, not cold-start latency. Results discarded.
    if config.n_warmup and inputs:
        warmup_inputs = inputs[: config.n_warmup]
        await asyncio.gather(*[
            _request_all_loras(
                client=client,
                sem=sem,
                lora_names=config.lora_names,
                messages=get_embed_qwen_prompt(text=text, img_url=img_url),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            for text, img_url in warmup_inputs
        ])

    sample_tasks = [
        _request_all_loras(
            client=client,
            sem=sem,
            lora_names=config.lora_names,
            messages=get_embed_qwen_prompt(text=text, img_url=img_url),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        for text, img_url in inputs
    ]

    t0 = time.perf_counter()
    per_sample: List[List[_InferenceResult]] = await asyncio.gather(*sample_tasks)
    elapsed = time.perf_counter() - t0

    results = [r for sample in per_sample for r in sample]

    stats = _compute_latency_stats(results, elapsed)
    stats.update({"n_loras": len(config.lora_names), "elapsed_s": round(elapsed, 2)})
    return stats

def benchmark_lora_sweep(
    client: AsyncOpenAI,
    all_lora_names: List[str],
    inputs: List[Tuple[str, Optional[str]]],
    lora_counts: Optional[List[int]] = None,
    max_tokens: int = 1,
    temperature: float = 0.0,
    concurrency: int = 512,
    n_warmup: int = 8,
) -> pd.DataFrame:
    """Run benchmarks for increasing LoRA counts and return a summary table.

    Each run sends all samples to the first N adapters from `all_lora_names`
    simultaneously, letting vLLM batch them internally.
    """
    if lora_counts is None:
        lora_counts = list(range(1, len(all_lora_names) + 1))

    rows = []
    for n in tqdm(lora_counts):
        if n > len(all_lora_names):
            logger.warning(f"Skipping n_loras={n}: only {len(all_lora_names)} available.")
            continue

        config = BenchmarkConfig(
            lora_names=all_lora_names[:n],
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            n_warmup=n_warmup,
        )
        rows.append(asyncio.run(_run_single_benchmark(client, config, inputs)))

    return pd.DataFrame(rows)
