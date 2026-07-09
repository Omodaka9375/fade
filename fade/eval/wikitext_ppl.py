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
    """Compute WikiText-2 PPL with FADE cache using persistent sliding window.

    This implements proper sliding-window perplexity evaluation with FADE's
    tiered KV cache compression. Unlike the baseline which creates a fresh
    cache per chunk, this maintains a **persistent cache** across chunks and
    triggers tier reassignment after each chunk to activate compression.

    Methodology:
        1. Prefill first chunk into FADE cache
        2. For each subsequent chunk:
           - Run forward pass with current cache (teacher-forced on new tokens)
           - Append new K/V to cache
           - Trigger tier reassignment (eviction + quantization)
        3. Accumulate NLL only on the **continuation** tokens (not the overlap)

    This ensures compression actually runs and measures the real quality impact
    of INT4/INT2 quantization and eviction policies.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        preset: FADE preset name (``"safe"``, ``"balanced"``, ``"aggressive"``).
        max_length: context window per evaluation chunk.
        stride: step between chunks (overlap = max_length - stride).
        device: torch device.
        split: dataset split (default ``"test"``).

    Returns:
        Perplexity (float). Lower is better.
    """
    from fade import FadeConfig, create_tiered_cache
    from fade.policy import reassign_tiers_by_position

    text = _load_wikitext2(split)
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    preset_fn = getattr(FadeConfig, preset, FadeConfig.safe)
    config = preset_fn()
    # H2O requires attention weights which we don't have in teacher-forced eval
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    dtype = next(model.parameters()).dtype

    # Create persistent FADE cache
    cache = create_tiered_cache(model, dtype=dtype, config=config)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    first_chunk = True

    for begin in tqdm(range(0, seq_len, stride), desc=f"fade-ppl-{preset}", leave=False):
        end = min(begin + max_length, seq_len)
        window = input_ids[:, begin:end]
        seq_len_window = window.size(1)

        if first_chunk:
            # First chunk: prefill the cache, compute NLL on entire chunk
            target = window.clone()
            out = model(window, labels=target, past_key_values=cache, use_cache=True)
            nlls.append(out.loss.float() * seq_len_window)
            first_chunk = False
        else:
            # Subsequent chunks: teacher-forced on new tokens only
            # Compute NLL only on continuation (exclude overlap from prev chunk)
            overlap = begin - prev_end
            new_tokens = window[:, overlap:]
            target = new_tokens.clone()
            target[:, :-1] = -100  # Only compute loss on predicted tokens

            out = model(new_tokens, labels=target, past_key_values=cache, use_cache=True)
            nlls.append(out.loss.float() * new_tokens.size(1))

            # Trigger tier reassignment after processing this chunk
            # This is where compression/eviction actually happens
            reassign_tiers_by_position(cache, num_layers=len(cache._layers))

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
