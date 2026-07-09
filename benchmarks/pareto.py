"""Compression-quality Pareto frontier sweep.

Sweeps ``int4_budget`` from aggressive to unlimited, measuring KV cache
size and perplexity at each point. Outputs CSV for plotting.

Usage:
    python benchmarks/pareto.py
    python benchmarks/pareto.py --model Qwen/Qwen2.5-0.5B-Instruct --csv pareto.csv
"""

from __future__ import annotations

import argparse
import csv

import torch

from fade import FadeConfig, create_tiered_cache
from fade.eval.perplexity import perplexity
from fade.patch import forward_with_tracking, load_model
from fade.policy import reassign_tiers_by_position
from fade.tracker import AttentionTracker

# --- configuration ---------------------------------------------------------- #
MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

PPL_TEXT: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
    "Photosynthesis converts sunlight into chemical energy in chloroplasts. "
    "The Roman Empire built over 80,000 kilometers of paved roads. "
    "Quantum computers exploit superposition and entanglement to process "
    "information in fundamentally different ways than classical machines. "
) * 5

PPL_MAX_LENGTH: int = 512
PPL_STRIDE: int = 256

# Budget sweep points.
BUDGETS: list[int | None] = [50, 100, 200, 400, 800, None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto frontier sweep")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")

    # Baseline PPL.
    print("Computing baseline PPL...")
    base_ppl = perplexity(
        model, tokenizer, PPL_TEXT, max_length=PPL_MAX_LENGTH, stride=PPL_STRIDE, device=DEVICE
    )
    print(f"Baseline PPL: {base_ppl:.2f}")

    results = []

    for budget in BUDGETS:
        label = f"budget={budget}" if budget is not None else "unlimited"
        # Map budget to preset name for wikitext2_fade_ppl
        if budget is None:
            preset_name = "safe"
        elif budget <= 100:
            preset_name = "aggressive"
        else:
            preset_name = "balanced"
        print(f"\n--- {label} (preset={preset_name}) ---")

        if budget is not None:
            config = FadeConfig(
                phase="2",
                n_sink=4,
                recent_window=64,
                int4_budget=budget,
                int2_budget=0,
                eviction_policy="position",
            )
        else:
            config = FadeConfig.safe()

        # Measure KV cache size on a long prompt.
        filler = PPL_TEXT * 3
        enc = tokenizer(filler, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        S = enc.input_ids.shape[1]

        cache = create_tiered_cache(model, dtype=DTYPE, config=config)
        num_layers = model.config.num_hidden_layers
        tracker = AttentionTracker(num_layers=num_layers)
        forward_with_tracking(model, enc.input_ids, cache, tracker=tracker)
        reassign_tiers_by_position(cache, num_layers)
        kv_bytes = cache.compressed_storage_bytes()
        kv_mib = kv_bytes / (1024 * 1024)

        # Measure actual PPL using the fixed wikitext2_fade_ppl
        # This triggers real compression and measures actual quality impact
        from fade.eval.wikitext_ppl import wikitext2_fade_ppl

        ppl = wikitext2_fade_ppl(
            model,
            tokenizer,
            preset=preset_name,
            max_length=PPL_MAX_LENGTH,
            stride=PPL_STRIDE,
            device=DEVICE,
        )

        # Correct baseline: K + V cache with GQA (uses num_key_value_heads)
        # Each KV head stores: S * hidden_size / num_attention_heads * 2 bytes (fp16)
        # Total: num_kv_heads * layers * 2 (K+V) * S * head_dim * 2 bytes
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        baseline_bytes = (
            S
            * num_layers
            * kv_heads
            * 2  # K + V
            * head_dim
            * 2  # fp16 = 2 bytes
        )
        ratio = baseline_bytes / max(kv_bytes, 1)

        results.append(
            {
                "budget": budget if budget is not None else "unlimited",
                "tokens": S,
                "kv_mib": round(kv_mib, 2),
                "compression": round(ratio, 1),
                "ppl": round(ppl, 2),
                "ppl_delta_pct": round((ppl / base_ppl - 1) * 100, 1),
            }
        )
        print(
            f"  KV: {kv_mib:.2f} MiB, ~{ratio:.1f}x compression, PPL: {ppl:.2f} ({(ppl / base_ppl - 1) * 100:+.1f}%)"
        )

    # Summary.
    print(f"\n{'=' * 70}")
    print(f"  Pareto Frontier: {args.model} @ {S} tokens")
    print(f"{'=' * 70}")
    print(f"{'Budget':>10} {'KV MiB':>8} {'Compress':>10} {'PPL':>8} {'PPL Δ%':>8}")
    for r in results:
        print(
            f"{r['budget']!s:>10} {r['kv_mib']:>8.2f} {r['compression']:>9.1f}x {r['ppl']:>8.2f} {r['ppl_delta_pct']:>+7.1f}%"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
