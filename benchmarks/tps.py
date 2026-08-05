"""Tokens-per-second benchmark with FADE compression active.

Measures TPS with:
1. Long prompts (2048+ tokens) to trigger actual compression
2. Tier reassignment every N steps to measure reassignment overhead
3. Separate metrics for steady-state vs reassignment-step latency

This gives honest numbers for the real performance cost of FADE compression.

Usage:
    python benchmarks/tps.py
    python benchmarks/tps.py --model Qwen/Qwen2.5-3B-Instruct --prompt-length 2048
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import DynamicCache

from fade import FadeConfig, create_tiered_cache
from fade.patch import load_model
from fade.policy import reassign_tiers_by_position


# --- configuration ---------------------------------------------------------- #
MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

# Long prompt for realistic compression testing
PROMPT_LENGTH: int = 2048  # tokens
DECODE_TOKENS: int = 256   # number of tokens to generate
REASSIGN_EVERY: int = 32   # reassign tier every N decode steps

# Short prompt for baseline comparison (no compression)
SHORT_PROMPT: str = "Explain how a CPU cache hierarchy works in detail."


def make_long_prompt(tokenizer, target_tokens: int):
    """Create a long filler prompt of approximately target_tokens."""
    filler = "The history of computer science spans from early mechanical calculators to modern artificial intelligence. "
    repeats = max(1, target_tokens // (len(filler.split()) + 10))
    text = filler * repeats
    return tokenizer(text, return_tensors="pt").to(DEVICE)


@torch.no_grad()
def measure_baseline_tps(model, tokenizer, decode_tokens: int, use_long_prompt: bool = False):
    """Measure baseline FP16 TPS.
    
    Args:
        model: HuggingFace model
        tokenizer: model tokenizer
        decode_tokens: number of tokens to generate
        use_long_prompt: if True, use long prompt; otherwise use short prompt
    
    Returns:
        (tps, elapsed_s, prompt_len)
    """
    if use_long_prompt:
        enc = make_long_prompt(tokenizer, PROMPT_LENGTH)
    else:
        enc = tokenizer(SHORT_PROMPT, return_tensors="pt").to(DEVICE)
    
    input_ids = enc.input_ids
    prompt_len = int(input_ids.shape[1])
    
    cache = DynamicCache()
    
    # Prefill
    out = model(input_ids, past_key_values=cache, use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    
    # Warmup
    for _ in range(5):
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    
    # Measure
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(decode_tokens):
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    tps = decode_tokens / elapsed
    return tps, elapsed, prompt_len


@torch.no_grad()
def measure_fade_tps(model, tokenizer, preset: str = "balanced", decode_tokens: int = DECODE_TOKENS):
    """Measure FADE TPS with reassignment.
    
    This measures:
    1. Steady-state TPS (decode steps without reassignment)
    2. Reassignment-step latency (steps that trigger tier reassignment)
    3. Overall amortized TPS
    
    Args:
        model: HuggingFace model
        tokenizer: model tokenizer
        preset: FADE preset name ("safe", "balanced", "aggressive")
        decode_tokens: number of tokens to generate
    
    Returns:
        dict with tps_steady, tps_reassign, tps_amortized, reassign_count, etc.
    """
    # Create FADE cache with the specified preset
    config_fn = getattr(FadeConfig, preset, FadeConfig.balanced)
    config = config_fn()
    # Use position-based eviction (H2O needs attention tracking)
    config = config.with_overrides(eviction_policy="position")
    
    cache = create_tiered_cache(model, dtype=DTYPE, config=config)
    
    # Use long prompt to trigger compression
    enc = make_long_prompt(tokenizer, PROMPT_LENGTH)
    input_ids = enc.input_ids
    prompt_len = int(input_ids.shape[1])
    
    num_layers = model.config.num_hidden_layers
    
    # Prefill
    out = model(input_ids, past_key_values=cache, use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    
    # Track metrics
    steady_times: list[float] = []
    reassign_times: list[float] = []
    step_count = 0
    
    # Warmup (skip reassignment during warmup)
    for _ in range(5):
        t0 = time.perf_counter()
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        _ = time.perf_counter() - t0
        step_count += 1
    
    # Measure with reassignment
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for step in range(decode_tokens):
        step_start = time.perf_counter()
        
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        
        step_elapsed = time.perf_counter() - step_start
        step_count += 1
        
        # Trigger reassignment at intervals
        if step_count % REASSIGN_EVERY == 0:
            reassign_start = time.perf_counter()
            reassign_tiers_by_position(cache, num_layers)
            reassign_elapsed = time.perf_counter() - reassign_start
            reassign_times.append(reassign_elapsed)
        else:
            steady_times.append(step_elapsed)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - t0
    
    # Compute metrics
    tps_amortized = decode_tokens / total_elapsed
    
    avg_steady = sum(steady_times) / len(steady_times) if steady_times else 0
    avg_reassign = sum(reassign_times) / len(reassign_times) if reassign_times else 0
    
    tps_steady = 1.0 / avg_steady if avg_steady > 0 else 0
    tps_reassign = 1.0 / avg_reassign if avg_reassign > 0 else 0
    
    # Memory metrics
    kv_mib = cache.compressed_storage_bytes() / (1024 * 1024)
    
    return {
        "tps_steady": round(tps_steady, 1),
        "tps_reassign": round(tps_reassign, 1),
        "tps_amortized": round(tps_amortized, 1),
        "total_elapsed_s": round(total_elapsed, 3),
        "steady_step_count": len(steady_times),
        "reassign_count": len(reassign_times),
        "avg_steady_ms": round(avg_steady * 1000, 3),
        "avg_reassign_ms": round(avg_reassign * 1000, 3),
        "kv_cache_mib": round(kv_mib, 2),
        "prompt_tokens": prompt_len,
        "decode_tokens": decode_tokens,
        "reassign_every": REASSIGN_EVERY,
    }


def main():
    parser = argparse.ArgumentParser(description="FADE TPS benchmark with reassignment")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--prompt-length", type=int, default=PROMPT_LENGTH)
    parser.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    parser.add_argument("--reassign-every", type=int, default=REASSIGN_EVERY)
    parser.add_argument("--preset", type=str, default="balanced")
    args = parser.parse_args()
    
    # Update global config
    global PROMPT_LENGTH, DECODE_TOKENS, REASSIGN_EVERY
    PROMPT_LENGTH = args.prompt_length
    DECODE_TOKENS = args.decode_tokens
    REASSIGN_EVERY = args.reassign_every
    
    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")
    
    print(f"\n{'=' * 60}")
    print("Baseline (FP16, no compression)")
    print(f"{'=' * 60}")
    
    # Short prompt baseline
    tps_short, elapsed_short, prompt_short = measure_baseline_tps(
        model, tokenizer, args.decode_tokens, use_long_prompt=False
    )
    print(f"\nShort prompt ({prompt_short} tokens):")
    print(f"  TPS: {tps_short:.1f} tok/s  (elapsed: {elapsed_short:.3f}s)")
    
    # Long prompt baseline (for fair comparison)
    tps_long, elapsed_long, prompt_long = measure_baseline_tps(
        model, tokenizer, args.decode_tokens, use_long_prompt=True
    )
    print(f"\nLong prompt ({prompt_long} tokens):")
    print(f"  TPS: {tps_long:.1f} tok/s  (elapsed: {elapsed_long:.3f}s)")
    
    print(f"\n{'=' * 60}")
    print(f"FADE ({args.preset} preset, reassign every {REASSIGN_EVERY} steps)")
    print(f"{'=' * 60}")
    
    results = measure_fade_tps(
        model, tokenizer, preset=args.preset, decode_tokens=args.decode_tokens
    )
    
    print(f"\nPrompt: {results['prompt_tokens']} tokens")
    print(f"Decode: {results['decode_tokens']} tokens")
    print(f"\nPerformance:")
    print(f"  Steady-state TPS:      {results['tps_steady']:.1f} tok/s")
    print(f"  Reassign-step TPS:     {results['tps_reassign']:.1f} tok/s")
    print(f"  Amortized TPS:         {results['tps_amortized']:.1f} tok/s")
    print(f"\nTiming breakdown:")
    print(f"  Avg steady step:       {results['avg_steady_ms']:.3f} ms")
    print(f"  Avg reassign step:     {results['avg_reassign_ms']:.3f} ms")
    print(f"  Reassignment count:    {results['reassign_count']}")
    print(f"  Total elapsed:         {results['total_elapsed_s']:.3f}s")
    print(f"\nMemory:")
    print(f"  KV cache (compressed): {results['kv_cache_mib']:.2f} MiB")
    
    # Overhead compared to baseline
    if tps_long > 0:
        overhead_pct = (tps_long / max(results['tps_amortized'], 0.01) - 1) * 100
        print(f"\nOverhead vs baseline:    {overhead_pct:+.1f}%")
    
    print(f"\n{'=' * 60}")
    print("Notes:")
    print(f"  - Steady-state: decode steps WITHOUT reassignment")
    print(f"  - Reassign-step: decode steps WITH tier reassignment")
    print(f"  - Amortized: overall TPS including reassignment cost")
    print(f"  - Reassignment happens every {REASSIGN_EVERY} decode steps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
