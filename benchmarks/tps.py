"""Tokens-per-second micro-benchmark: baseline vs tiered decode.

Measures the raw decode TPS for the hot path (no tier reassignment) to
isolate the overhead of TieredKVCache._materialize vs DynamicCache.

Usage:
    python benchmarks/tps.py
    python benchmarks/tps.py --model Qwen/Qwen2.5-3B-Instruct --tokens 256
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import DynamicCache

from fade.patch import create_tiered_cache, load_model

# --- configuration ---------------------------------------------------------- #
MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT: str = "Explain how a CPU cache hierarchy works in detail."
WARMUP_TOKENS: int = 5
MEASURE_TOKENS: int = 64


@torch.no_grad()
def _decode_loop(model, input_ids, cache, n_tokens):
    """Run n decode steps, return elapsed seconds."""
    out = model(input_ids, past_key_values=cache, use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    # Warmup.
    for _ in range(WARMUP_TOKENS):
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    # Measure.
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="TPS micro-benchmark")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--tokens", type=int, default=MEASURE_TOKENS)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    prompt_len = enc.input_ids.shape[1]
    print(f"model={args.model}  prompt={prompt_len}  measure={args.tokens}  device={DEVICE}")

    # --- Baseline ---
    cache_base = DynamicCache()
    t_base = _decode_loop(model, enc.input_ids, cache_base, args.tokens)
    tps_base = args.tokens / t_base
    print(f"baseline: {tps_base:.1f} tok/s  ({t_base:.3f}s)")

    # --- Tiered (no eviction, Phase 1-A) ---
    cache_tier = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=4,
        recent_window=64,
        int4_budget=None,
        int2_budget=0,
    )
    t_tier = _decode_loop(model, enc.input_ids, cache_tier, args.tokens)
    tps_tier = args.tokens / t_tier
    print(f"tiered:   {tps_tier:.1f} tok/s  ({t_tier:.3f}s)")

    delta = tps_tier / tps_base - 1.0
    print(f"delta:    {delta:+.1%}")


if __name__ == "__main__":
    main()
