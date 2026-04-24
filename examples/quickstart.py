"""FADE quickstart: three presets on Qwen2.5-0.5B-Instruct.

Run this script on a machine with a GPU:
    python examples/quickstart.py

Or paste the cells into a Colab/Jupyter notebook.
"""

import torch
from transformers import DynamicCache

from fade import FadeConfig, create_tiered_cache
from fade.eval.memory import cache_storage_bytes
from fade.patch import load_model

# ── Config ──────────────────────────────────────────────────────── #
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PROMPT = "Explain how a CPU cache hierarchy works, in three paragraphs."
MAX_NEW = 128


def run_preset(model, tokenizer, label, config=None):
    """Generate with a preset and print stats."""
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    if config is None:
        # Baseline: plain DynamicCache.
        cache = DynamicCache()
    else:
        cache = create_tiered_cache(model, dtype=DTYPE, config=config)

    out = model.generate(
        **enc,
        past_key_values=cache,
        max_new_tokens=MAX_NEW,
        do_sample=False,
    )
    text = tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    kv_mib = cache_storage_bytes(cache) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  {label}  |  kv_cache = {kv_mib:.2f} MiB  |  tokens = {out.shape[1]}")
    print(f"{'=' * 60}")
    print(text[:300])
    return kv_mib


def main():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    model, tokenizer = load_model(MODEL_ID, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")

    base_mib = run_preset(model, tokenizer, "Baseline (FP16)")
    safe_mib = run_preset(model, tokenizer, "FadeConfig.safe()    (~3-4x)", FadeConfig.safe())
    bal_mib = run_preset(model, tokenizer, "FadeConfig.balanced() (~5x)", FadeConfig.balanced())
    agg_mib = run_preset(model, tokenizer, "FadeConfig.aggressive(~7-8x)", FadeConfig.aggressive())

    print(f"\n── Summary ──")
    print(f"Baseline:   {base_mib:.2f} MiB")
    print(f"Safe:       {safe_mib:.2f} MiB  ({100 * (1 - safe_mib / base_mib):.0f}% smaller)")
    print(f"Balanced:   {bal_mib:.2f} MiB  ({100 * (1 - bal_mib / base_mib):.0f}% smaller)")
    print(f"Aggressive: {agg_mib:.2f} MiB  ({100 * (1 - agg_mib / base_mib):.0f}% smaller)")


if __name__ == "__main__":
    main()
