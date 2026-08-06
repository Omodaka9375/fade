"""Minimal needle-in-a-haystack test.

Builds a synthetic context of ``target_tokens`` with a known "needle" inserted
at ``needle_position_frac`` of the way through, then asks the model to retrieve
the needle. Returns whether the answer contains the needle string.

Production evals should use RULER or lm-eval-harness; this is a fast local check.

Can optionally test with a custom cache (e.g., FADE's TieredKVCache) to verify
compression quality.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# --- knobs ------------------------------------------------------------------- #
DEFAULT_NEEDLE: str = "The secret passphrase is CERULEAN-KESTREL-77."
DEFAULT_QUESTION: str = "What is the secret passphrase?"
DEFAULT_FILLER: str = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
)
DEFAULT_TARGET_TOKENS: int = 2048
DEFAULT_NEEDLE_POSITION_FRAC: float = 0.5
DEFAULT_MAX_NEW_TOKENS: int = 32


@torch.no_grad()
def run_needle(
    model,
    tokenizer,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    needle: str = DEFAULT_NEEDLE,
    question: str = DEFAULT_QUESTION,
    filler: str = DEFAULT_FILLER,
    needle_position_frac: float = DEFAULT_NEEDLE_POSITION_FRAC,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str | torch.device = "cuda",
    cache_factory: Callable | None = None,
) -> dict:
    """Insert ``needle`` into a block of filler and ask the model to find it.

    Args:
        model: HuggingFace causal LM to test.
        tokenizer: matching tokenizer.
        target_tokens: approximate total prompt length in tokens.
        needle: the secret phrase to hide and retrieve.
        question: the question to ask about the needle.
        filler: text used to pad the context to ``target_tokens``.
        needle_position_frac: where to insert the needle (0.0 = beginning, 1.0 = end).
        max_new_tokens: maximum tokens to generate for the answer.
        device: torch device to use.
        cache_factory: optional callable that returns a cache object (e.g.,
            ``lambda: create_tiered_cache(model, config=FadeConfig.balanced())``).
            If provided, the model will use this cache for generation, testing
            the model's ability to retrieve the needle under compression.
            If None, uses the default HF DynamicCache (uncompressed).

    Returns:
        Dict with keys:
            - ``prompt_tokens``: total tokens in the prompt
            - ``answer``: the model's generated answer
            - ``passed``: True if the answer contains the needle
    """
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    # repeat filler until we have enough tokens
    repeats = max(1, target_tokens // max(1, len(filler_ids)))
    haystack_ids = filler_ids * repeats

    insert_at = int(len(haystack_ids) * needle_position_frac)
    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    haystack_ids = haystack_ids[:insert_at] + needle_ids + haystack_ids[insert_at:]

    context = tokenizer.decode(haystack_ids)
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    # Use provided cache if available, otherwise use default
    past_key_values = cache_factory() if cache_factory is not None else None

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        past_key_values=past_key_values,
    )
    answer = tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    passed = "CERULEAN-KESTREL-77" in answer or needle.split()[-1].rstrip(".") in answer
    return {
        "prompt_tokens": int(enc.input_ids.shape[1]),
        "answer": answer.strip(),
        "passed": bool(passed),
    }
