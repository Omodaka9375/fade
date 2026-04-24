"""Token-by-token divergence benchmark: baseline vs tiered.

Compares greedy decoding from a plain DynamicCache against a TieredKVCache
and reports the cumulative match rate at each position. Useful for detecting
quality degradation as context grows.

Usage:
    python benchmarks/divergence.py              # built-in prompt, Qwen-0.5B
    python benchmarks/divergence.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import torch
from transformers import DynamicCache

from fade.patch import create_tiered_cache, forward_with_tracking, load_model
from fade.policy import reassign_tiers_by_position
from fade.tracker import AttentionTracker

# --- configuration (top of file for easy modification) ---------------------- #
MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT: str = (
    "Explain the difference between a compiler and an interpreter. "
    "Then describe how just-in-time compilation combines ideas from both."
)
MAX_NEW_TOKENS: int = 128

# Tiered cache settings.
N_SINK: int = 4
RECENT_WINDOW: int = 64
REASSIGN_EVERY: int = 32


@torch.no_grad()
def _greedy_decode(model, tokenizer, input_ids, cache, max_new_tokens, tracker=None):
    """Manual greedy loop returning a list of token ids."""
    out = forward_with_tracking(model, input_ids, cache, tracker=tracker)
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    tokens = [tok.item()]
    for step in range(max_new_tokens - 1):
        out = forward_with_tracking(model, tok, cache, tracker=tracker)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        tokens.append(tok.item())
        if tracker is not None and (step + 1) % REASSIGN_EVERY == 0:
            reassign_tiers_by_position(cache, model.config.num_hidden_layers)
        if tokenizer.eos_token_id is not None and tok.item() == tokenizer.eos_token_id:
            break
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy divergence benchmark")
    parser.add_argument("--csv", type=str, default=None, help="Write per-token CSV to this path")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device_map=DEVICE, dtype=DTYPE, attn_impl="eager")
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    num_layers = model.config.num_hidden_layers
    prompt_len = enc.input_ids.shape[1]
    print(f"model={args.model}  prompt_tokens={prompt_len}  max_new={args.max_tokens}")

    # --- Baseline ---
    t0 = time.perf_counter()
    baseline_cache = DynamicCache()
    baseline_tokens = _greedy_decode(
        model, tokenizer, enc.input_ids, baseline_cache, args.max_tokens
    )
    t_base = time.perf_counter() - t0

    # --- Tiered ---
    t0 = time.perf_counter()
    tiered_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=N_SINK,
        recent_window=RECENT_WINDOW,
        int4_budget=None,
        int2_budget=0,
    )
    tracker = AttentionTracker(num_layers=num_layers)
    tiered_tokens = _greedy_decode(
        model,
        tokenizer,
        enc.input_ids,
        tiered_cache,
        args.max_tokens,
        tracker=tracker,
    )
    t_tier = time.perf_counter() - t0

    # --- Compare ---
    min_len = min(len(baseline_tokens), len(tiered_tokens))
    matches = [int(baseline_tokens[i] == tiered_tokens[i]) for i in range(min_len)]
    cum_match = []
    running = 0
    for i, m in enumerate(matches):
        running += m
        cum_match.append(running / (i + 1))

    print(f"\nbaseline: {len(baseline_tokens)} tokens in {t_base:.2f}s")
    print(f"tiered:   {len(tiered_tokens)} tokens in {t_tier:.2f}s")
    print(f"overall match: {sum(matches)}/{min_len} ({100 * sum(matches) / max(min_len, 1):.1f}%)")
    print(f"match @10: {cum_match[9]:.2%}" if min_len >= 10 else "")
    print(f"match @50: {cum_match[49]:.2%}" if min_len >= 50 else "")
    print(f"match @100: {cum_match[99]:.2%}" if min_len >= 100 else "")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["position", "baseline_tok", "tiered_tok", "match", "cumulative_match"])
            for i in range(min_len):
                w.writerow(
                    [i, baseline_tokens[i], tiered_tokens[i], matches[i], f"{cum_match[i]:.4f}"]
                )
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
