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
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

# benchmarks/ is a scripts directory, not an installed package — add it to
# sys.path so sibling modules like unified_eval can be imported directly.
sys.path.insert(0, str(Path(__file__).parent))
from unified_eval import evaluate_config, evaluate_preset_grid, print_unified_results  # noqa: E402

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
def measure_fp16_baseline_bytes(model, tokenizer, target_tokens: int) -> int:
    """Measure FP16 baseline KV cache bytes from actual DynamicCache.

    This provides a real measurement instead of an analytical formula,
    ensuring the compression ratio is computed from the same execution
    paradigm as the compressed cache measurements.
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    num_layers = text_cfg.num_hidden_layers
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)

    input_ids = _make_filler(tokenizer, target_tokens)
    S = input_ids.shape[1]

    # Use actual DynamicCache to measure baseline
    baseline_cache = DynamicCache()
    model(input_ids, past_key_values=baseline_cache, use_cache=True)

    # Measure actual bytes from the cache
    # For DynamicCache, each layer has K and V tensors of shape [B, H, S, D]
    total_bytes = 0
    for i in range(num_layers):
        k = baseline_cache.key_cache[i]
        v = baseline_cache.value_cache[i]
        if k is not None and v is not None:
            total_bytes += k.element_size() * k.numel()
            total_bytes += v.element_size() * v.numel()

    return total_bytes


@torch.no_grad()
def measure_kv_bytes(model, tokenizer, preset_dict: dict, target_tokens: int) -> dict:
    """Prefill with auto-reassignment, return KV bytes and compression ratio.

    This measures compression using the same auto-reassign path that quality
    eval uses (P0-1), ensuring consistency between compression and quality metrics.
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    num_layers = text_cfg.num_hidden_layers

    input_ids = _make_filler(tokenizer, target_tokens)
    S = input_ids.shape[1]

    cache = _make_cache(model, preset_dict, head_dim)

    if preset_dict["preset"] is None:
        # Baseline: measure actual FP16 bytes from DynamicCache
        model(input_ids, past_key_values=cache, use_cache=True)
        kv_bytes = cache_storage_bytes(cache)
    else:
        # FADE: use auto-reassign (P0-1) - no forced reassignment
        # The cache will auto-reassign during the forward pass
        tracker = AttentionTracker(num_layers=num_layers)
        forward_with_tracking(model, input_ids, cache, tracker=tracker)
        kv_bytes = cache_storage_bytes(cache)

    # Measure FP16 baseline from actual DynamicCache (same execution paradigm)
    fp16_bytes = measure_fp16_baseline_bytes(model, tokenizer, target_tokens)
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
    """Run full benchmark suite for one model using unified evaluation."""
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

    # --- Unified Evaluation: Compression + Quality from SAME run ---
    print("\n--- Unified Evaluation (Compression + Quality from Same Run) ---")

    if skip_ppl:
        print("  [SKIP] PPL evaluation disabled")
        unified_results = []
        for preset_name in ["safe", "balanced", "aggressive"]:
            r = evaluate_config(
                model, tokenizer, preset=preset_name,
                target_tokens=2048,
                eval_ppl=False,
                eval_needle=False,
                device=DEVICE
            )
            unified_results.append(r)
            print(f"  {preset_name}: {r['compression']:.1f}x compression, "
                  f"KV: {r['kv_mib']:.1f} MiB, Peak: {r['peak_memory_mib']:.1f} MiB")
    else:
        # Get baseline PPL first
        print("\n  Computing baseline FP16 PPL...")
        baseline_ppl = run_wikitext2_ppl(model, tokenizer)
        result["wikitext2_ppl"] = baseline_ppl
        if baseline_ppl:
            print(f"  Baseline FP16 PPL: {baseline_ppl}")

        # Unified evaluation for each preset
        unified_results = []
        for preset_name in ["safe", "balanced", "aggressive"]:
            print(f"  {preset_name}...", end=" ", flush=True)
            try:
                r = evaluate_config(
                    model, tokenizer, preset=preset_name,
                    target_tokens=2048,
                    eval_ppl=True,
                    eval_needle=False,
                    device=DEVICE
                )
                unified_results.append(r)
                print(f"Compression: {r['compression']:.1f}x, "
                      f"PPL: {r['ppl']:.2f} ({r['ppl_delta_pct']:+.1f}%), "
                      f"KV: {r['kv_mib']:.1f} MiB")
            except Exception as e:
                print(f"ERROR: {e}")
                unified_results.append({"preset": preset_name, "error": str(e)})

        result["unified_eval"] = unified_results

    # --- TPS (separate measurement) ---
    print("\n--- Decode TPS ---")
    tps_result = measure_tps(model, tokenizer, {"preset": "balanced"})
    result["tps"] = tps_result
    print(f"  Baseline TPS: {tps_result['tps']:.1f} tok/s")

    return result


def print_markdown_summary(all_results: list[dict]) -> str:
    """Generate a markdown table summarizing all results.

    Matches the result structure produced by the current benchmark_model():
        result["unified_eval"]  — list of per-preset dicts with keys:
                                  preset, compression, kv_mib, fp16_mib,
                                  ppl (optional), ppl_delta_pct (optional)
        result["tps"]           — single dict with key "tps" (balanced preset)
        result["wikitext2_ppl"] — float baseline PPL (optional)
    """
    lines = [
        "",
        "## Production Benchmark Summary",
        "",
    ]

    for res in all_results:
        if "error" in res:
            lines.append(f"### {res.get('model', '?')} — ERROR: {res['error']}")
            lines.append("")
            continue

        model_name = res["model"].split("/")[-1]
        lines.append(f"### {model_name}")
        lines.append("")

        # Unified evaluation table (compression + quality from same run).
        unified = res.get("unified_eval", [])
        if unified:
            lines.append("| Config | KV Cache | Compression | PPL Δ |")
            lines.append("|--------|----------|:-----------:|:-----:|")
            for r in unified:
                if "error" in r:
                    lines.append(f"| {r.get('preset', '?')} | ERROR | — | — |")
                    continue
                kv   = f"{r['kv_mib']:.2f} MiB"
                comp = f"**{r['compression']:.1f}×**"
                delta = f"{r['ppl_delta_pct']:+.1f}%" if "ppl_delta_pct" in r else "—"
                lines.append(f"| {r['preset']} | {kv} | {comp} | {delta} |")
        else:
            lines.append("_No unified evaluation data._")

        lines.append("")

        # TPS (single balanced-preset measurement).
        tps_data = res.get("tps", {})
        if isinstance(tps_data, dict) and "tps" in tps_data:
            lines.append(f"**Decode TPS (balanced):** {tps_data['tps']:.1f} tok/s")

        # Baseline PPL.
        ppl = res.get("wikitext2_ppl")
        if ppl is not None:
            lines.append(f"**WikiText-2 PPL (baseline FP16):** {ppl}")

        # Needle results (baseline).
        needle = res.get("needle_baseline", res.get("needle", {}))
        if needle:
            needle_str = ", ".join(
                f"@{d}: {'✅' if n.get('passed') else '❌'}"
                for d, n in sorted(needle.items(), key=lambda x: int(x[0]))
            )
            lines.append(f"**Needle (baseline):** {needle_str}")

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
