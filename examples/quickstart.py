"""FADE quickstart: all features in one script.

Demonstrates presets, rotated 2-bit backend, adaptive bit allocation,
and compression measurements with tier reassignment.

Usage:
    python examples/quickstart.py
"""

import torch
from transformers import DynamicCache

from fade import FadeConfig, create_tiered_cache
from fade.backends import get_backend
from fade.eval.memory import cache_storage_bytes
from fade.patch import forward_with_tracking, load_model
from fade.policy import reassign_tiers_adaptive, reassign_tiers_by_position
from fade.tracker import AttentionTracker

# --- config ----------------------------------------------------------------- #
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PROMPT = (
    "Explain the following topics in detail, one paragraph each:\n"
    "1. How CPU cache hierarchies work (L1, L2, L3)\n"
    "2. The history of the Roman road network\n"
    "3. How photosynthesis converts sunlight to energy\n"
    "4. The basics of quantum computing and qubits\n"
    "5. How the human immune system fights pathogens\n"
    "6. Continental drift and plate tectonics\n"
    "7. The origins of jazz music\n"
    "8. How language models use key-value caching\n"
)
MAX_NEW = 128
REASSIGN_EVERY = 32


def run_decode(model, tokenizer, cache, num_layers, policy="position", tracker=None):
    """Manual decode loop with tier reassignment."""
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    if tracker is None:
        tracker = AttentionTracker(num_layers=num_layers)
    out = forward_with_tracking(model, enc.input_ids, cache, tracker=tracker)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    for step in range(MAX_NEW - 1):
        out = forward_with_tracking(model, next_tok, cache, tracker=tracker)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        if (step + 1) % REASSIGN_EVERY == 0:
            if policy == "adaptive":
                reassign_tiers_adaptive(cache, tracker, num_layers, high_pct=0.5)
            else:
                reassign_tiers_by_position(cache, num_layers)
        if tokenizer.eos_token_id is not None and next_tok.item() == tokenizer.eos_token_id:
            break
    return cache.compressed_storage_bytes() / (1024 * 1024)


def main():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    model, tokenizer = load_model(MODEL_ID, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")
    num_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Baseline.
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    base_cache = DynamicCache()
    with torch.no_grad():
        model.generate(**enc, past_key_values=base_cache, max_new_tokens=MAX_NEW, do_sample=False)
    base_kv = cache_storage_bytes(base_cache) / (1024 * 1024)

    # Safe INT4.
    safe_cache = create_tiered_cache(model, dtype=DTYPE, config=FadeConfig.safe())
    safe_kv = run_decode(model, tokenizer, safe_cache, num_layers)

    # Balanced (eviction).
    bal_cache = create_tiered_cache(
        model, dtype=DTYPE, config=FadeConfig.balanced().with_overrides(eviction_policy="position")
    )
    bal_kv = run_decode(model, tokenizer, bal_cache, num_layers)

    # Rotated 2-bit.
    rot_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        config=FadeConfig.safe(),
        quant_backend=get_backend("rotated", head_dim=head_dim, bits=2),
    )
    rot_kv = run_decode(model, tokenizer, rot_cache, num_layers)

    # Adaptive bit allocation (E1).
    ada_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        config=FadeConfig(phase="2", int4_budget=200, int2_budget=200, eviction_policy="position"),
    )
    ada_kv = run_decode(model, tokenizer, ada_cache, num_layers, policy="adaptive")

    # Aggressive.
    agg_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        config=FadeConfig.aggressive().with_overrides(eviction_policy="position"),
    )
    agg_kv = run_decode(model, tokenizer, agg_cache, num_layers)

    # Summary.
    print(f"\n{'=' * 60}")
    print(f"  FADE Compression Results — {MODEL_ID}")
    print(f"{'=' * 60}")
    print(f"  {'Config':<25} {'KV MiB':>8} {'Ratio':>8}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8}")
    for label, kv in [
        ("Baseline FP16", base_kv),
        ("Safe (INT4)", safe_kv),
        ("Rotated 2-bit", rot_kv),
        ("Balanced (eviction)", bal_kv),
        ("Adaptive (E1)", ada_kv),
        ("Aggressive", agg_kv),
    ]:
        ratio = base_kv / kv if kv > 0 else 0
        print(f"  {label:<25} {kv:>7.2f} {ratio:>7.1f}x")


if __name__ == "__main__":
    main()
