"""FADE quickstart: shows actual KV cache compression in action.

The key: FADE compresses on tier reassignment, not just during generate().
This script runs a manual decode loop with periodic reassignment so you
can see the cache shrink in real time.

Usage:
    python examples/quickstart.py
"""

import torch
from transformers import DynamicCache

from fade import FadeConfig, create_tiered_cache
from fade.backends import get_backend
from fade.eval.memory import cache_storage_bytes
from fade.patch import forward_with_tracking, load_model
from fade.policy import reassign_tiers, reassign_tiers_by_position
from fade.tracker import AttentionTracker

# --- config ----------------------------------------------------------------- #
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Long prompt so there are enough tokens to compress.
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


def run_baseline(model, tokenizer):
    """Plain DynamicCache baseline."""
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    cache = DynamicCache()
    with torch.no_grad():
        out = model.generate(**enc, past_key_values=cache, max_new_tokens=MAX_NEW, do_sample=False)
    kv = cache_storage_bytes(cache) / (1024 * 1024)
    text = tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    return kv, text, enc.input_ids.shape[1]


def run_tiered(model, tokenizer, config, label):
    """Manual decode loop with tier reassignment."""
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    input_ids = enc.input_ids
    num_layers = model.config.num_hidden_layers

    cache = create_tiered_cache(model, dtype=DTYPE, config=config)
    tracker = AttentionTracker(num_layers=num_layers)

    # Prefill.
    out = forward_with_tracking(model, input_ids, cache, tracker=tracker)
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_token]

    # Decode with periodic reassignment.
    for step in range(MAX_NEW - 1):
        out = forward_with_tracking(model, next_token, cache, tracker=tracker)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)

        if (step + 1) % REASSIGN_EVERY == 0:
            if config.eviction_policy == "position":
                reassign_tiers_by_position(cache, num_layers)
            else:
                reassign_tiers(cache, tracker, num_layers)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    kv = cache.compressed_storage_bytes() / (1024 * 1024)
    all_tokens = torch.cat([input_ids, *generated], dim=-1)
    text = tokenizer.decode(all_tokens[0, input_ids.shape[1] :], skip_special_tokens=True)
    return kv, text


def main():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    model, tokenizer = load_model(MODEL_ID, device_map=DEVICE, dtype=DTYPE, attn_impl="eager")

    # Baseline.
    base_kv, base_text, prompt_tokens = run_baseline(model, tokenizer)
    print(f"\nPrompt: {prompt_tokens} tokens, generating {MAX_NEW} tokens")
    print(f"\n{'=' * 60}")
    print(f"  Baseline (FP16)  |  kv_cache = {base_kv:.2f} MiB")
    print(f"{'=' * 60}")
    print(base_text[:200])

    # Safe preset.
    safe_kv, safe_text = run_tiered(
        model, tokenizer, FadeConfig.safe().with_overrides(eviction_policy="position"), "Safe"
    )
    print(f"\n{'=' * 60}")
    print(
        f"  Safe (~3-4x)     |  kv_cache = {safe_kv:.2f} MiB  ({100 * (1 - safe_kv / base_kv):.0f}% smaller)"
    )
    print(f"{'=' * 60}")
    print(safe_text[:200])

    # Balanced preset.
    bal_kv, bal_text = run_tiered(model, tokenizer, FadeConfig.balanced(), "Balanced")
    print(f"\n{'=' * 60}")
    print(
        f"  Balanced (~5x)   |  kv_cache = {bal_kv:.2f} MiB  ({100 * (1 - bal_kv / base_kv):.0f}% smaller)"
    )
    print(f"{'=' * 60}")
    print(bal_text[:200])

    # Aggressive preset.
    agg_kv, agg_text = run_tiered(model, tokenizer, FadeConfig.aggressive(), "Aggressive")
    print(f"\n{'=' * 60}")
    print(
        f"  Aggressive (~7x) |  kv_cache = {agg_kv:.2f} MiB  ({100 * (1 - agg_kv / base_kv):.0f}% smaller)"
    )
    print(f"{'=' * 60}")
    print(agg_text[:200])

    # Rotated 2-bit (6x compression).
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    enc_r = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    rot2_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        config=FadeConfig.safe(),
        quant_backend=get_backend("rotated", head_dim=head_dim, bits=2),
    )
    tracker_r = AttentionTracker(num_layers=num_layers)
    out_r = forward_with_tracking(model, enc_r.input_ids, rot2_cache, tracker=tracker_r)
    next_tok = out_r.logits[:, -1:, :].argmax(dim=-1)
    for step in range(MAX_NEW - 1):
        out_r = forward_with_tracking(model, next_tok, rot2_cache, tracker=tracker_r)
        next_tok = out_r.logits[:, -1:, :].argmax(dim=-1)
        if (step + 1) % REASSIGN_EVERY == 0:
            reassign_tiers_by_position(rot2_cache, num_layers)
        if tokenizer.eos_token_id is not None and next_tok.item() == tokenizer.eos_token_id:
            break
    rot2_kv = rot2_cache.compressed_storage_bytes() / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(
        f"  Rotated 2-bit (~6x) |  kv_cache = {rot2_kv:.2f} MiB  ({100 * (1 - rot2_kv / base_kv):.0f}% smaller)"
    )
    print(f"{'=' * 60}")

    # Summary.
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  Baseline:      {base_kv:.2f} MiB")
    print(f"  Safe INT4:     {safe_kv:.2f} MiB  ({base_kv / safe_kv:.1f}x)")
    print(f"  Balanced:      {bal_kv:.2f} MiB  ({base_kv / bal_kv:.1f}x)")
    print(f"  Aggressive:    {agg_kv:.2f} MiB  ({base_kv / agg_kv:.1f}x)")
    print(f"  Rotated 2-bit: {rot2_kv:.2f} MiB  ({base_kv / rot2_kv:.1f}x)")


if __name__ == "__main__":
    main()
