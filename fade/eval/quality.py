"""Structured quality evaluation suite.

``run_quality_suite`` runs needle-in-a-haystack and perplexity checks,
returning a results dict suitable for CI gating or human inspection.

Example:
    from fade.eval.quality import run_quality_suite
    results = run_quality_suite(model, tokenizer, device="cuda")
    assert results["needle"]["passed"]

Can optionally test with a custom cache (e.g., FADE's TieredKVCache) to verify
quality under compression.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from fade.eval.needle import run_needle
from fade.eval.perplexity import perplexity

# --- knobs ------------------------------------------------------------------ #
DEFAULT_NEEDLE_TARGET_TOKENS: int = 1024
DEFAULT_PPL_MAX_LENGTH: int = 1024
DEFAULT_PPL_STRIDE: int = 512
DEFAULT_PPL_TEXT: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
) * 20
DEFAULT_PPL_THRESHOLD: float = 100.0  # generous for small / random-init models


def run_quality_suite(
    model,
    tokenizer,
    device: str | torch.device = "cuda",
    needle_target_tokens: int = DEFAULT_NEEDLE_TARGET_TOKENS,
    ppl_text: str | None = None,
    ppl_max_length: int = DEFAULT_PPL_MAX_LENGTH,
    ppl_stride: int = DEFAULT_PPL_STRIDE,
    ppl_threshold: float = DEFAULT_PPL_THRESHOLD,
    run_ppl: bool = True,
    run_needle: bool = True,
    cache_factory: Callable | None = None,
) -> dict[str, Any]:
    """Run the quality suite and return a results dict.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        device: torch device.
        needle_target_tokens: target length for needle test.
        ppl_text: text for perplexity evaluation.
        ppl_max_length: max length for PPL sliding window.
        ppl_stride: stride for PPL sliding window.
        ppl_threshold: PPL threshold for pass/fail.
        run_ppl: whether to run perplexity test.
        run_needle: whether to run needle test.
        cache_factory: optional callable that returns a cache object for
            the needle test. If provided, tests model quality under
            compression. If None, uses default HF DynamicCache.

    Returns:
        ``{"needle": {..., "passed": bool}, "perplexity": {..., "passed": bool},
          "all_passed": bool}``
    """
    results: dict[str, Any] = {}

    if run_needle:
        needle_result = run_needle_test(
            model,
            tokenizer,
            device=device,
            target_tokens=needle_target_tokens,
            cache_factory=cache_factory,
        )
        results["needle"] = needle_result

    if run_ppl:
        text = ppl_text or DEFAULT_PPL_TEXT
        ppl_result = run_perplexity_test(
            model,
            tokenizer,
            text=text,
            device=device,
            max_length=ppl_max_length,
            stride=ppl_stride,
            threshold=ppl_threshold,
        )
        results["perplexity"] = ppl_result

    results["all_passed"] = all(r.get("passed", True) for r in results.values())
    return results


def run_needle_test(
    model,
    tokenizer,
    device="cuda",
    target_tokens=DEFAULT_NEEDLE_TARGET_TOKENS,
    cache_factory: Callable | None = None,
) -> dict:
    """Run needle-in-a-haystack and return result dict.

    Args:
        model: HuggingFace causal LM.
        tokenizer: matching tokenizer.
        device: torch device.
        target_tokens: target length for needle test.
        cache_factory: optional callable that returns a cache object.
            If provided, the model uses this cache for generation.

    Returns:
        Dict with keys ``prompt_tokens``, ``answer``, and ``passed``.
    """
    result = run_needle(model, tokenizer, target_tokens=target_tokens, device=device, cache_factory=cache_factory)
    return result


def run_perplexity_test(
    model,
    tokenizer,
    text: str,
    device="cuda",
    max_length=DEFAULT_PPL_MAX_LENGTH,
    stride=DEFAULT_PPL_STRIDE,
    threshold=DEFAULT_PPL_THRESHOLD,
) -> dict:
    """Run perplexity eval and return result dict with pass/fail."""
    ppl = perplexity(model, tokenizer, text, max_length=max_length, stride=stride, device=device)
    return {
        "ppl": ppl,
        "threshold": threshold,
        "passed": ppl < threshold,
    }
