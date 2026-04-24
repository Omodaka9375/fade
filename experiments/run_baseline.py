"""FP16 baseline: plain DynamicCache, no tier management.

This is the reference the tiered cache needs to beat on memory and match on
quality. Run it once, record the numbers, commit them.
"""

from __future__ import annotations

import time

import torch

from fade.eval.memory import PeakMemory
from fade.eval.needle import run_needle
from fade.eval.perplexity import perplexity
from fade.patch import load_model

# --- configuration (all knobs at the top) ------------------------------------ #
MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT: str = "Explain, in three short paragraphs, why caching matters for LLM inference."
MAX_NEW_TOKENS: int = 128

RUN_PPL: bool = True
PPL_MAX_LENGTH: int = 1024
PPL_STRIDE: int = 512
PPL_TEXT: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
) * 40  # ~a few thousand tokens

RUN_NEEDLE: bool = True
NEEDLE_TARGET_TOKENS: int = 1024


def main() -> None:
    print(f"device={DEVICE}  dtype={DTYPE}  model={MODEL_ID}")
    # Baseline doesn't need attention tracking — SDPA is fine (and faster).
    model, tokenizer = load_model(MODEL_ID, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")

    # ---------------------------- generation ---------------------------- #
    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    with PeakMemory(DEVICE) as mem:
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

    generated = out[0, enc.input_ids.shape[1] :]
    tps = generated.shape[0] / elapsed if elapsed > 0 else float("inf")
    print("\n--- generation ---")
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"\nnew_tokens={generated.shape[0]}  elapsed={elapsed:.2f}s  tps={tps:.2f}")
    print(f"peak_mem={mem.peak_mib:.1f} MiB ({mem.peak_gib:.2f} GiB)")

    # ---------------------------- perplexity ---------------------------- #
    if RUN_PPL:
        print("\n--- perplexity ---")
        ppl = perplexity(
            model,
            tokenizer,
            PPL_TEXT,
            max_length=PPL_MAX_LENGTH,
            stride=PPL_STRIDE,
            device=DEVICE,
        )
        print(f"ppl={ppl:.3f}")

    # ---------------------------- needle -------------------------------- #
    if RUN_NEEDLE:
        print("\n--- needle-in-a-haystack ---")
        result = run_needle(
            model,
            tokenizer,
            target_tokens=NEEDLE_TARGET_TOKENS,
            device=DEVICE,
        )
        print(result)


if __name__ == "__main__":
    main()
