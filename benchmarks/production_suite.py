"""Production benchmark suite for DGX Spark validation.

Runs FADE across multiple models and presets, measuring:
  - WikiText-2 delta-PPL (the standard KV compression metric)
  - Needle-in-a-haystack at 512, 1024, 2048, 4096 tokens
  - KV cache memory (compressed_storage_bytes) at 2048 and 4096 tokens
  - Compression ratio vs FP16 DynamicCache
  - Decode tokens-per-second (steady-state, no reassignment)

Outputs ``benchmarks/dgx_results.json`` and prints a markdown summary
table suitable for pasting into the README.

Usage:
    python benchmarks/production_suite.py
    python benchmarks/production_suite.py --models Qwen/Qwen2.5-7B-Instruct
    python benchmarks/production_suite.py --out dgx_results.json --skip-longbench
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import DynamicCache

from fade import FadeConfig, create_tiered_cache
from fade.backends import get_backend
from fade.eval.memory import cache_storage_bytes
from fade.eval.needle import run_needle
from fade.patch import forward_with_tracking, load_model
from fade.policy import reassign_tiers_by_position
from fade.tracker import AttentionTracker

# --- configuration (top of file for easy override) -------------------------- #
DEFAULT_MODELS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

NEEDLE_DEPTHS: list[int] = [512, 1024, 2048, 4096]
KV_MEASURE_LENGTHS: list[int] = [2048, 4096]
TPS_WARMUP: int = 5
TPS_MEASURE: int = 64

PRESETS: list[dict] = [
    {"name": "Baseline FP16", "preset": None},
    {"name": "Safe (INT4)", "preset": "safe"},
    {"name": "Balanced", "preset": "balanced"},
    {"name": "Aggressive", "preset": "aggressive"},
    {"name": "Rotated 2-bit", "preset": "safe", "backend": "rotated_2bit"},
]

FILLER_TEXT: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
    "Photosynthesis converts sunlight into chemical energy in chloroplasts. "
    "The Roman Empire built over 80,000 kilometers of paved roads. "
    "Quantum computers exploit superposition and entanglement to process "
    "information in fundamentally different ways than classical machines. "
) * 10


# --- helpers ---------------------------------------------------------------- #
def _make_filler(tokenizer, target_tokens: int) -> torch.Tensor:
    """Tokenize filler text, repeating until we reach target length."""
    ids = tokenizer(FILLER_TEXT, add_special_tokens=False).input_ids
    repeats = max(1, target_tokens // len(ids) + 1)
    long_ids = (ids * repeats)[:target_tokens]
    return torch.tensor([long_ids], device=DEVICE)


def _make_cache(model, preset_dict: dict, head_dim: int):
    """Create cache from a preset dict. Returns None for baseline."""
    if preset_dict["preset"] is None:
        return DynamicCache()

    preset_fn = getattr(FadeConfig, preset_dict["preset"])
    config = preset_fn()
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    kwargs = {}
    if preset_dict.get("backend") == "rotated_2bit":
        kwargs["quant_backend"] = get_backend("rotated", head_dim=head_dim, bits=2)

    return create_tiered_cache(model, dtype=DTYPE, config=config, **kwargs)


@torch.no_grad()
def measure_kv_bytes(model, tokenizer, preset_dict: dict, target_tokens: int) -> dict:
    """Prefill + reassign, return KV bytes and compression ratio."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    num_layers = text_cfg.num_hidden_layers

    input_ids = _make_filler(tokenizer, target_tokens)
    S = input_ids.shape[1]

    cache = _make_cache(model, preset_dict, head_dim)

    if preset_dict["preset"] is None:
        model(input_ids, past_key_values=cache, use_cache=True)
    else:
        tracker = AttentionTracker(num_layers=num_layers)
        forward_with_tracking(model, input_ids, cache, tracker=tracker)
        reassign_tiers_by_position(cache, num_layers)

    kv_bytes = cache_storage_bytes(cache)

    # Compute FP16 baseline bytes for ratio.
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)
    fp16_bytes = 2 * num_layers * num_kv_heads * head_dim * S * 2  # K+V, 2 bytes each
    ratio = fp16_bytes / max(kv_bytes, 1)

    return {
        "tokens": S,
        "kv_bytes": kv_bytes,
        "kv_mib": round(kv_bytes / (1024 * 1024), 2),
        "fp16_mib": round(fp16_bytes / (1024 * 1024), 2),
        "compression": round(ratio, 1),
    }


@torch.no_grad()
def measure_tps(model, tokenizer, preset_dict: dict) -> dict:
    """Decode TPS (steady-state, no reassignment)."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)

    prompt = "Explain how a CPU cache hierarchy works in detail."
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    cache = _make_cache(model, preset_dict, head_dim)

    # Prefill.
    out = model(enc.input_ids, past_key_values=cache, use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1)

    # Warmup.
    for _ in range(TPS_WARMUP):
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)

    # Measure.
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TPS_MEASURE):
        out = model(tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = TPS_MEASURE / elapsed

    return {"tps": round(tps, 1), "elapsed_s": round(elapsed, 3), "tokens": TPS_MEASURE}


def run_wikitext2_ppl(model, tokenizer) -> float | None:
    """Run WikiText-2 PPL. Returns None if datasets not installed."""
    try:
        from fade.eval.wikitext_ppl import wikitext2_perplexity

        return round(wikitext2_perplexity(model, tokenizer, device=DEVICE), 4)
    except ImportError:
        print("  [SKIP] WikiText-2 PPL requires `pip install fade-kv[eval]`")
        return None
    except Exception as e:
        print(f"  [ERROR] WikiText-2 PPL: {e}")
        return None


# --- main ------------------------------------------------------------------- #
def benchmark_model(model_id: str, skip_ppl: bool = False) -> dict:
    """Run full benchmark suite for one model."""
    print(f"\n{'=' * 70}")
    print(f"  Model: {model_id}")
    print(f"{'=' * 70}")

    model, tokenizer = load_model(model_id, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")
    text_cfg = getattr(model.config, "text_config", model.config)
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)

    result: dict = {
        "model": model_id,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "head_dim": head_dim,
        "num_layers": text_cfg.num_hidden_layers,
        "num_kv_heads": getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads),
    }

    # --- WikiText-2 PPL ---
    if not skip_ppl:
        print("\n--- WikiText-2 Perplexity ---")
        baseline_ppl = run_wikitext2_ppl(model, tokenizer)
        result["wikitext2_ppl"] = baseline_ppl
        if baseline_ppl is not None:
            print(f"  Baseline FP16 PPL: {baseline_ppl}")

        # Delta-PPL per FADE preset (P2).
        delta_ppl_results = []
        for preset_name in ["safe", "balanced", "aggressive"]:
            print(f"  {preset_name}...", end=" ", flush=True)
            try:
                from fade.eval.wikitext_ppl import wikitext2_fade_ppl

                ppl = round(
                    wikitext2_fade_ppl(model, tokenizer, preset=preset_name, device=DEVICE), 4
                )
                delta = round(ppl - baseline_ppl, 4) if baseline_ppl else 0
                delta_pct = round((delta / baseline_ppl) * 100, 2) if baseline_ppl else 0
                print(f"PPL {ppl} (delta {delta:+.4f}, {delta_pct:+.2f}%)")
                delta_ppl_results.append(
                    {"preset": preset_name, "ppl": ppl, "delta": delta, "delta_pct": delta_pct}
                )
            except Exception as e:
                print(f"ERROR: {e}")
                delta_ppl_results.append({"preset": preset_name, "error": str(e)})
        result["delta_ppl"] = delta_ppl_results
    else:
        result["wikitext2_ppl"] = None
        result["delta_ppl"] = None
        print("\n--- WikiText-2 PPL skipped ---")

    # --- Needle ---
    print("\n--- Needle-in-a-Haystack ---")
    needle_results = {}
    for depth in NEEDLE_DEPTHS:
        print(f"  Depth {depth}...", end=" ", flush=True)
        try:
            r = run_needle(model, tokenizer, target_tokens=depth, device=DEVICE)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{status} ({r['answer'][:60]})")
            needle_results[str(depth)] = r
        except Exception as e:
            print(f"ERROR: {e}")
            needle_results[str(depth)] = {"passed": False, "error": str(e)}
    result["needle"] = needle_results

    # --- KV memory + compression ---
    print("\n--- KV Compression ---")
    compression_results = []
    for preset in PRESETS:
        for target_len in KV_MEASURE_LENGTHS:
            print(f"  {preset['name']:20s} @ {target_len} tokens...", end=" ", flush=True)
            try:
                kv = measure_kv_bytes(model, tokenizer, preset, target_len)
                print(f"{kv['kv_mib']:.2f} MiB ({kv['compression']:.1f}x)")
                compression_results.append({"preset": preset["name"], **kv})
            except Exception as e:
                print(f"ERROR: {e}")
                compression_results.append(
                    {"preset": preset["name"], "tokens": target_len, "error": str(e)}
                )
    result["compression"] = compression_results

    # --- Decode TPS ---
    print("\n--- Decode TPS ---")
    tps_results = []
    for preset in PRESETS:
        print(f"  {preset['name']:20s}...", end=" ", flush=True)
        try:
            tps = measure_tps(model, tokenizer, preset)
            print(f"{tps['tps']:.1f} tok/s")
            tps_results.append({"preset": preset["name"], **tps})
        except Exception as e:
            print(f"ERROR: {e}")
            tps_results.append({"preset": preset["name"], "error": str(e)})
    result["tps"] = tps_results

    # Cleanup.
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_markdown_summary(all_results: list[dict]) -> str:
    """Generate a markdown table summarizing all results."""
    lines = [
        "",
        "## Production Benchmark Summary",
        "",
    ]

    for res in all_results:
        model = res["model"].split("/")[-1]
        lines.append(f"### {model}")
        lines.append("")

        # Compression table (2048 tokens).
        lines.append("| Config | KV Cache | Compression | Decode TPS |")
        lines.append("|--------|----------|:-----------:|:----------:|")

        comp_2k = {c["preset"]: c for c in res.get("compression", []) if c.get("tokens") == 2048}
        tps_map = {t["preset"]: t for t in res.get("tps", [])}

        for preset in PRESETS:
            name = preset["name"]
            c = comp_2k.get(name, {})
            t = tps_map.get(name, {})
            kv = f"{c.get('kv_mib', '?')} MiB" if "kv_mib" in c else "?"
            comp = f"**{c['compression']}x**" if "compression" in c else "?"
            tps = f"{t['tps']}" if "tps" in t else "?"
            lines.append(f"| {name} | {kv} | {comp} | {tps} tok/s |")

        # Needle results.
        lines.append("")
        needle = res.get("needle", {})
        if needle:
            needle_str = ", ".join(
                f"@{d}: {'✅' if n.get('passed') else '❌'}" for d, n in needle.items()
            )
            lines.append(f"Needle: {needle_str}")

        # PPL.
        ppl = res.get("wikitext2_ppl")
        if ppl is not None:
            lines.append(f"WikiText-2 PPL (baseline FP16): {ppl}")

        lines.append("")

    summary = "\n".join(lines)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FADE production benchmark suite")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model IDs to benchmark.",
    )
    parser.add_argument("--out", type=str, default="benchmarks/dgx_results.json")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip WikiText-2 PPL.")
    args = parser.parse_args()

    all_results = []
    for model_id in args.models:
        try:
            result = benchmark_model(model_id, skip_ppl=args.skip_ppl)
            all_results.append(result)
        except Exception as e:
            print(f"\n[FATAL] {model_id}: {e}")
            all_results.append({"model": model_id, "error": str(e)})

    # Save JSON.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print markdown summary.
    md = print_markdown_summary(all_results)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
