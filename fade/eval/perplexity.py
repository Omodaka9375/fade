"""Sliding-window perplexity.

Standard recipe: tokenize the corpus, slide a fixed context across it with a
given stride, compute NLL on the non-overlapping tail of each window.
"""
from __future__ import annotations

import math

import torch
from tqdm import tqdm

# --- knobs (tweak at call site or here) -------------------------------------- #
DEFAULT_MAX_LENGTH: int = 2048
DEFAULT_STRIDE: int = 1024


@torch.no_grad()
def perplexity(
    model,
    tokenizer,
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    device: str | torch.device = "cuda",
) -> float:
    """Return perplexity of ``model`` on ``text`` via sliding-window NLL."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in tqdm(range(0, seq_len, stride), desc="ppl"):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # only new tokens contribute to NLL
        window = input_ids[:, begin:end]
        target = window.clone()
        target[:, :-trg_len] = -100  # ignore overlapping prefix

        out = model(window, labels=target)
        # out.loss is mean over non-ignored targets; scale back to a sum
        nlls.append(out.loss.float() * trg_len)

        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    return math.exp(total_nll.item() / seq_len)
