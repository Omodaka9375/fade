"""WikiText-2 perplexity evaluation.

Standard metric used by KV cache compression papers (KIVI, TurboQuant,
KVTC, kvpress). Loads WikiText-2 from HuggingFace datasets, computes
sliding-window perplexity, and optionally measures delta-PPL against a
baseline FP16 cache.

This module provides TWO evaluation modes:

1. **Teacher-forced** (baseline): Computes PPL using ground-truth tokens.
   This is the standard academic metric for comparing compression quality.

2. **Auto-regressive** (generation): Computes PPL by actually generating
   tokens. This measures the REAL quality impact of compression on
   downstream generation quality.

Requires ``pip install fade-kv[eval]`` (pulls ``datasets``).

Usage:
    from fade.eval.wikitext_ppl import wikitext2_perplexity, wikitext2_fade_ppl
    
    # Teacher-forced (standard academic metric)
    ppl = wikitext2_perplexity(model, tokenizer, device="cuda")
    
    # Auto-regressive (actual generation quality)
    ppl = wikitext2_fade_ppl(model, tokenizer, preset="balanced", mode="generate")
"""

from __future__ import annotations

import math

import warnings

import torch
from tqdm import tqdm

# --- knobs ------------------------------------------------------------------ #
# max_length and stride determine context depth for compression evals.
# Context tokens = max_length - stride. With max_length=8192, stride=512:
#   context = 7680 tokens > int4_budget (400) for all presets, so eviction
#   presets are forced to actually evict and the ordering
#   safe < balanced < aggressive in Δ PPL is observed correctly.
DEFAULT_MAX_LENGTH: int = 8192
DEFAULT_STRIDE: int = 512
DEFAULT_SPLIT: str = "test"
DEFAULT_GEN_MAX_TOKENS: int = 128  # For auto-regressive mode


def _tokenize_corpus(tokenizer, text: str) -> torch.Tensor:
    """Tokenize a full corpus string, suppressing the sequence-length warning.

    WikiText-2 has ~300K tokens; models have max_position_embeddings of
    32K-131K. The tokenizer warns when the total exceeds the model limit,
    but this is intentional — we slice the corpus into windows ourselves
    and never feed the full sequence to the model.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*longer than the specified maximum sequence length.*",
        )
        return tokenizer(text, return_tensors="pt").input_ids


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
    """Compute sliding-window perplexity on WikiText-2 (teacher-forced).

    This is the standard academic metric used by KV cache compression papers.
    It computes PPL by feeding ground-truth tokens to the model (teacher-forced
    decoding), which gives a lower bound on perplexity.

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
    input_ids = _tokenize_corpus(tokenizer, text).to(device)
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
def wikitext2_fade_ppl_teacher_forced(
    model,
    tokenizer,
    preset: str = "safe",
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device = "cuda",
    split: str = DEFAULT_SPLIT,
) -> float:
    """Compute WikiText-2 PPL with FADE cache using teacher-forced evaluation.

    **NOTE**: This is the OLD implementation that was criticized in the audit.
    It uses teacher-forced decoding which doesn't measure actual generation quality.
    
    For actual generation quality, use ``wikitext2_fade_ppl(..., mode="generate")``.

    This maintains a **persistent cache** across chunks and triggers tier
    reassignment after each chunk to activate compression.

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
    input_ids = _tokenize_corpus(tokenizer, text).to(device)
    seq_len = input_ids.size(1)

    preset_fn = getattr(FadeConfig, preset, FadeConfig.safe)
    config = preset_fn()
    # H2O requires attention weights which we don't have in teacher-forced eval
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    dtype = next(model.parameters()).dtype

    nlls: list[torch.Tensor] = []
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc=f"fade-ppl-{preset}-tf", leave=False):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]

        # Step 1 — use the CONTEXT portion (all but the last trg_len tokens)
        # to prefill the cache with FP16, then compress it.
        # This simulates the scenario where a compressed cache is serving
        # as context for new tokens.
        context = window[:, :-trg_len] if trg_len < window.size(1) else None

        cache = create_tiered_cache(model, dtype=dtype, config=config)

        if context is not None and context.size(1) > 0:
            # Prefill context into cache (no loss scored here).
            with torch.no_grad():
                model(context, past_key_values=cache, use_cache=True)
            # Compress: this is the operation whose quality impact we measure.
            reassign_tiers_by_position(cache, num_layers=len(cache._layers))

        # Step 2 — score the new tokens against the compressed cache.
        # The model must predict them using a compressed KV context.
        new_tok = window[:, -trg_len:]
        target = new_tok.clone()
        out = model(new_tok, labels=target, past_key_values=cache, use_cache=True)
        nlls.append(out.loss.float() * trg_len)

        del cache

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
    mode: str = "generate",
    gen_max_tokens: int = DEFAULT_GEN_MAX_TOKENS,
) -> float:
    """Compute WikiText-2 PPL with FADE cache using auto-regressive generation.

    This is the **HONEST** evaluation mode that measures actual generation quality
    under compression. Unlike teacher-forced evaluation, this uses the model's
    own predictions, which exposes the real quality impact of KV cache compression.

    Methodology:
        1. For each chunk, use the model's previous predictions as context
        2. Generate new tokens auto-regressively (not teacher-forced)
        3. Trigger tier reassignment after each chunk
        4. Compute NLL on the ground-truth continuation tokens
        5. Accumulate NLL across all chunks

    This exposes the REAL degradation from compression because:
        - Model's predictions may diverge from ground truth
        - Compressed KV cache affects future predictions
        - Reassignment changes cache state between chunks
        - Error accumulation reveals true quality impact

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        preset: FADE preset name (``"safe"``, ``"balanced"``, ``"aggressive"``).
        max_length: context window per evaluation chunk.
        stride: step between chunks (overlap = max_length - stride).
        device: torch device.
        split: dataset split (default ``"test"``).
        mode: evaluation mode - "generate" for auto-regressive (default) or
            "teacher_forced" for the old implementation.
        gen_max_tokens: maximum tokens to generate per chunk (only used in
            "generate" mode).

    Returns:
        Perplexity (float). Lower is better.

    Example:
        # Honest generation quality measurement
        ppl = wikitext2_fade_ppl(model, tokenizer, preset="balanced", mode="generate")
        
        # Teacher-forced (academic standard, but doesn't test generation)
        ppl = wikitext2_fade_ppl(model, tokenizer, preset="balanced", mode="teacher_forced")
    """
    if mode == "teacher_forced":
        # Delegate to the teacher-forced implementation
        return wikitext2_fade_ppl_teacher_forced(
            model, tokenizer, preset, max_length, stride, device, split
        )

    # Auto-regressive generation mode
    from fade import FadeConfig, create_tiered_cache
    from fade.policy import reassign_tiers_by_position

    text = _load_wikitext2(split)
    input_ids = _tokenize_corpus(tokenizer, text).to(device)
    seq_len = input_ids.size(1)

    preset_fn = getattr(FadeConfig, preset, FadeConfig.safe)
    config = preset_fn()
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    dtype = next(model.parameters()).dtype

    # Create a fresh FADE cache — it accumulates KV state as we walk the corpus.
    cache = create_tiered_cache(model, dtype=dtype, config=config)
    num_layers = len(cache._layers)

    nlls: list[torch.Tensor] = []
    # last_token: the single token the model generated at the end of the
    # previous chunk.  None on the first chunk (no prior generation yet).
    last_token: torch.Tensor | None = None

    for begin in tqdm(range(0, seq_len, stride), desc=f"fade-ppl-{preset}-gen", leave=False):
        end = min(begin + max_length, seq_len)
        window = input_ids[:, begin:end]       # [1, W] ground-truth tokens
        window_len = window.size(1)

        if last_token is None:
            # ── First chunk: teacher-force the whole window to warm up the
            #    cache, measure NLL on the full window, then generate one
            #    token so we have a starting point for the next chunk.
            target = window.clone()
            # Standard LM loss: predict token i+1 from token i.
            out = model(window, labels=target, past_key_values=cache, use_cache=True)
            nlls.append(out.loss.float() * window_len)
        else:
            # ── Subsequent chunks: the cache already holds compressed KV from
            #    all prior chunks.  Feed only the NEW ground-truth tokens
            #    (the overlap portion from the previous stride) so the cache
            #    context stays aligned, then score the new tokens.
            #
            #    Key insight: we do NOT prepend generated_so_far — the cache
            #    IS the context.  Prepending would grow the input tensor
            #    linearly with chunk count → OOM on large corpora.
            # Only feed the NEW tokens this chunk introduces (the non-overlapping
            # portion).  stride tokens are new; max_length - stride tokens were
            # already fed in the previous chunk.
            n_new = min(stride, window_len)
            new_tokens = window[:, -n_new:]          # rightmost n_new tokens
            if new_tokens.shape[1] == 0:
                continue
            target = new_tokens.clone()
            out = model(new_tokens, labels=target, past_key_values=cache, use_cache=True)
            nlls.append(out.loss.float() * new_tokens.shape[1])

        # Generate exactly ONE token autoregressively so that compression
        # errors compound across chunks (this is what makes generation-mode
        # PPL differ from teacher-forced PPL).  We discard the token itself;
        # its only purpose is to leave the cache in the state the model would
        # reach after actual generation, not after teacher-forcing.
        last_tok_input = window[:, -1:]    # last ground-truth token as seed
        gen_out = model(last_tok_input, past_key_values=cache, use_cache=True)
        last_token = gen_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]

        # Periodic tier reassignment — keeps cache bounded.
        reassign_tiers_by_position(cache, num_layers)

    if not nlls:
        return float("inf")

    # Each NLL was already weighted by chunk length; sum then divide by total
    # tokens scored to get the mean NLL, then exponentiate to perplexity.
    total_nll = torch.stack(nlls).sum()
    total_tokens = seq_len  # same denominator as baseline wikitext2_perplexity
    return math.exp(total_nll.item() / total_tokens)


def wikitext2_delta_ppl(
    model,
    tokenizer,
    preset: str = "safe",
    baseline_ppl: float | None = None,
    mode: str = "generate",
    **kwargs,
) -> dict:
    """Compute FADE WikiText-2 PPL and delta vs FP16 baseline.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        preset: FADE preset name.
        baseline_ppl: pre-computed FP16 baseline PPL. If None, computed fresh.
        mode: evaluation mode ("generate" or "teacher_forced").
        **kwargs: forwarded to both perplexity functions.

    Returns:
        ``{"preset": str, "ppl": float, "baseline_ppl": float,
          "delta_ppl": float, "delta_ppl_pct": float}``
    """
    if baseline_ppl is None:
        baseline_ppl = wikitext2_perplexity(model, tokenizer, **kwargs)

    ppl = wikitext2_fade_ppl(
        model, tokenizer, preset=preset, mode=mode, **kwargs
    )
    delta = ppl - baseline_ppl
    delta_pct = (delta / baseline_ppl) * 100 if baseline_ppl > 0 else 0.0

    return {
        "preset": preset,
        "ppl": round(ppl, 4),
        "baseline_ppl": round(baseline_ppl, 4),
        "delta_ppl": round(delta, 4),
        "delta_ppl_pct": round(delta_pct, 2),
        "mode": mode,
    }
