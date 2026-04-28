"""WikiText-2 perplexity evaluation.

Standard metric used by KV cache compression papers (KIVI, TurboQuant,
KVTC, kvpress). Loads WikiText-2 from HuggingFace datasets, computes
sliding-window perplexity, and optionally measures delta-PPL against a
baseline FP16 cache.

Requires ``pip install fade-kv[eval]`` (pulls ``datasets``).

Usage:
    from fade.eval.wikitext_ppl import wikitext2_perplexity
    ppl = wikitext2_perplexity(model, tokenizer, device="cuda")
"""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

# --- knobs ------------------------------------------------------------------ #
DEFAULT_MAX_LENGTH: int = 2048
DEFAULT_STRIDE: int = 1024
DEFAULT_SPLIT: str = "test"


def _load_wikitext2(split: str = DEFAULT_SPLIT) -> str:
    """Load WikiText-2 test split and return as a single string."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "WikiText-2 evaluation requires the `datasets` library. "
            "Install with: pip install fade-kv[eval]"
        ) from e

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(row["text"] for row in ds if row["text"].strip())


@torch.no_grad()
def wikitext2_perplexity(
    model,
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device = "cuda",
    split: str = DEFAULT_SPLIT,
    past_key_values=None,
) -> float:
    """Compute sliding-window perplexity on WikiText-2.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        max_length: context window per evaluation chunk.
        stride: step between chunks (overlap = max_length - stride).
        device: torch device.
        split: dataset split (default ``"test"``).
        past_key_values: optional cache object. When provided, the model
            uses this cache for each window (reset per window). Pass a
            ``TieredKVCache`` to measure compressed PPL.

    Returns:
        Perplexity (float). Lower is better.
    """
    text = _load_wikitext2(split)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: list[torch.Tensor] = []
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc="wikitext2-ppl", leave=False):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]
        target = window.clone()
        target[:, :-trg_len] = -100

        out = model(window, labels=target)
        nlls.append(out.loss.float() * trg_len)

        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    return math.exp(total_nll.item() / seq_len)


@torch.no_grad()
def wikitext2_fade_ppl(
    model,
    tokenizer,
    preset: str = "safe",
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device = "cuda",
    split: str = DEFAULT_SPLIT,
) -> float:
    """Compute WikiText-2 PPL with FADE cache compression.

    Feeds each sliding window through the model in chunks, triggers
    tier reassignment after each chunk so tokens actually get compressed
    to INT4/INT2/evicted. Measures NLL on tail tokens.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        preset: FADE preset name (``"safe"``, ``"balanced"``, ``"aggressive"``).
        max_length: context window per evaluation chunk.
        stride: step between chunks.
        device: torch device.
        split: dataset split.

    Returns:
        Perplexity (float). Compare with ``wikitext2_perplexity()`` for delta.
    """
    from fade import FadeConfig, create_tiered_cache
    from fade.policy import reassign_tiers_by_position

    text = _load_wikitext2(split)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    preset_fn = getattr(FadeConfig, preset, FadeConfig.safe)
    config = preset_fn()
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    dtype = next(model.parameters()).dtype
    num_layers = model.config.num_hidden_layers

    nlls: list[torch.Tensor] = []
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc=f"fade-ppl-{preset}", leave=False):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]
        target = window.clone()
        target[:, :-trg_len] = -100
        window_len = end - begin

        # Fresh FADE cache per window.
        cache = create_tiered_cache(model, dtype=dtype, config=config)

        # Feed the window in two halves: first half triggers tier assignment,
        # second half computes loss with compressed context.
        split_at = max(window_len // 2, 1)
        prefix = window[:, :split_at]
        suffix = window[:, split_at:]
        suffix_target = target[:, split_at:]

        # Prefill the first half — tokens go into FP16.
        model(prefix, past_key_values=cache, use_cache=True)

        # Force tier reassignment — compresses middle tokens to INT4.
        reassign_tiers_by_position(cache, num_layers)

        # Evaluate loss on the second half with compressed context.
        out = model(suffix, labels=suffix_target, past_key_values=cache, use_cache=True)
        # Scale loss to the original trg_len accounting for the suffix only.
        suffix_trg = suffix_target.ne(-100).sum().item()
        if suffix_trg > 0:
            nlls.append(out.loss.float() * suffix_trg)

        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    return math.exp(total_nll.item() / seq_len)


def wikitext2_delta_ppl(
    model,
    tokenizer,
    preset: str = "safe",
    baseline_ppl: float | None = None,
    **kwargs,
) -> dict:
    """Compute FADE WikiText-2 PPL and delta vs FP16 baseline.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        preset: FADE preset name.
        baseline_ppl: pre-computed FP16 baseline PPL. If None, computed fresh.
        **kwargs: forwarded to both perplexity functions.

    Returns:
        ``{"preset": str, "ppl": float, "baseline_ppl": float,
          "delta_ppl": float, "delta_ppl_pct": float}``
    """
    if baseline_ppl is None:
        baseline_ppl = wikitext2_perplexity(model, tokenizer, **kwargs)

    ppl = wikitext2_fade_ppl(model, tokenizer, preset=preset, **kwargs)
    delta = ppl - baseline_ppl
    delta_pct = (delta / baseline_ppl) * 100 if baseline_ppl > 0 else 0.0

    return {
        "preset": preset,
        "ppl": round(ppl, 4),
        "baseline_ppl": round(baseline_ppl, 4),
        "delta_ppl": round(delta, 4),
        "delta_ppl_pct": round(delta_pct, 2),
    }
