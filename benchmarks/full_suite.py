"""Full FADE benchmark suite.

Runs needle-in-a-haystack at multiple depths, perplexity, and compression
measurements across all presets. Outputs results JSON for the README.

Usage:
    python benchmarks/full_suite.py
    python benchmarks/full_suite.py --model Qwen/Qwen2.5-0.5B-Instruct --out results.json
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import DynamicCache

from fade import FadeConfig, create_tiered_cache
from fade.backends import get_backend
from fade.eval.memory import cache_storage_bytes
from fade.eval.needle import run_needle
from fade.eval.perplexity import perplexity
from fade.patch import forward_with_tracking, load_model
from fade.policy import reassign_tiers_by_position
from fade.tracker import AttentionTracker

# --- configuration ---------------------------------------------------------- #
MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

NEEDLE_DEPTHS: list[int] = [512, 1024, 2048]
PPL_TEXT: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
) * 20

CONFIGS: list[dict] = [
    {"name": "Baseline FP16", "preset": None},
    {"name": "Safe INT4", "preset": "safe"},
    {"name": "Balanced", "preset": "balanced"},
    {"name": "Aggressive", "preset": "aggressive"},
    {"name": "Rotated 2-bit", "preset": "safe", "backend": "rotated_2bit"},
]


def _measure_kv(model, tokenizer, config_dict):
    """Measure KV cache size for a config."""
    filler = PPL_TEXT * 2
    enc = tokenizer(filler, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    S = enc.input_ids.shape[1]
    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    if config_dict["preset"] is None:
        cache = DynamicCache()
        with torch.no_grad():
            model(enc.input_ids, past_key_values=cache, use_cache=True)
        return cache_storage_bytes(cache), S

    preset_fn = getattr(FadeConfig, config_dict["preset"])
    config = preset_fn()
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    kwargs = {}
    if config_dict.get("backend") == "rotated_2bit":
        kwargs["quant_backend"] = get_backend("rotated", head_dim=head_dim, bits=2)

    cache = create_tiered_cache(model, dtype=DTYPE, config=config, **kwargs)
    tracker = AttentionTracker(num_layers=num_layers)
    forward_with_tracking(model, enc.input_ids, cache, tracker=tracker)
    reassign_tiers_by_position(cache, num_layers)
    return cache.compressed_storage_bytes(), S


def main() -> None:
    parser = argparse.ArgumentParser(description="Full FADE benchmark suite")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--out", type=str, default="benchmarks/results.json")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="eager")

    results = {"model": args.model, "device": DEVICE, "configs": []}

    # --- Needle tests ---
    print("\n=== Needle-in-a-Haystack ===")
    for depth in NEEDLE_DEPTHS:
        print(f"  Depth {depth}...", end=" ", flush=True)
        try:
            r = run_needle(model, tokenizer, target_tokens=depth, device=DEVICE)
            print(f"{'PASS' if r['passed'] else 'FAIL'} (answer: {r['answer'][:50]})")
        except Exception as e:
            r = {"passed": False, "error": str(e)}
            print(f"ERROR: {e}")
        results.setdefault("needle", {})[str(depth)] = r

    # --- Perplexity ---
    print("\n=== Perplexity ===")
    ppl = perplexity(model, tokenizer, PPL_TEXT, max_length=512, stride=256, device=DEVICE)
    print(f"  Baseline PPL: {ppl:.2f}")
    results["baseline_ppl"] = round(ppl, 2)

    # --- Compression per config ---
    print("\n=== Compression ===")
    for cfg in CONFIGS:
        kv_bytes, S = _measure_kv(model, tokenizer, cfg)
        kv_mib = kv_bytes / (1024 * 1024)
        entry = {"name": cfg["name"], "kv_mib": round(kv_mib, 2), "tokens": S}
        results["configs"].append(entry)
        print(f"  {cfg['name']:20s}: {kv_mib:.2f} MiB")

    # Compute compression ratios.
    if results["configs"]:
        base_mib = results["configs"][0]["kv_mib"]
        for cfg in results["configs"]:
            cfg["compression"] = round(base_mib / max(cfg["kv_mib"], 0.01), 1)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Results: {args.model}")
    print(f"{'=' * 60}")
    print(f"  Baseline PPL: {results['baseline_ppl']}")
    for cfg in results["configs"]:
        print(f"  {cfg['name']:20s}: {cfg['kv_mib']:.2f} MiB ({cfg['compression']}x)")
    for depth, nr in results.get("needle", {}).items():
        status = "PASS" if nr.get("passed") else "FAIL"
        print(f"  Needle @{depth}: {status}")

    # Save.
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
