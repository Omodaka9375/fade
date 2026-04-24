"""Minimal needle-in-a-haystack test.

Builds a synthetic context of ``target_tokens`` with a known "needle" inserted
at ``needle_position_frac`` of the way through, then asks the model to retrieve
the needle. Returns whether the answer contains the needle string.

Production evals should use RULER or lm-eval-harness; this is a fast local check.
"""

from __future__ import annotations

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
) -> dict:
    """Insert ``needle`` into a block of filler and ask the model to find it.

    Returns a dict with ``prompt_tokens``, ``answer``, and ``passed``.
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

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    passed = "CERULEAN-KESTREL-77" in answer or needle.split()[-1].rstrip(".") in answer
    return {
        "prompt_tokens": int(enc.input_ids.shape[1]),
        "answer": answer.strip(),
        "passed": bool(passed),
    }
