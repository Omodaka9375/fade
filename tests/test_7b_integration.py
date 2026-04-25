"""7B model integration tests on DGX Spark.

Gated behind ``@pytest.mark.dgx`` — skipped on machines without >=80GB GPU.
Run with: ``pytest tests/test_7b_integration.py -m dgx``

Tests:
    - Qwen2.5-7B-Instruct prefill + 64 decode steps with each preset
    - Compression ratio < baseline for all presets
    - Needle-in-a-haystack PASS at 2048 tokens
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.dgx

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if _gpu_mem_gb() < 30:
        pytest.skip(f"Need >=30GB GPU memory, got {_gpu_mem_gb():.0f}GB")
    from fade.patch import load_model

    model, tokenizer = load_model(MODEL_ID, device_map=DEVICE, dtype=DTYPE, attn_impl="sdpa")
    yield model, tokenizer
    del model, tokenizer
    torch.cuda.empty_cache()


@pytest.mark.parametrize("preset", ["safe", "balanced", "aggressive"])
def test_prefill_decode_no_crash(model_and_tokenizer, preset):
    """Prefill + 64 decode steps must complete without error."""
    from fade import FadeConfig, create_tiered_cache
    from fade.patch import forward_with_tracking
    from fade.policy import reassign_tiers_by_position
    from fade.tracker import AttentionTracker

    model, tokenizer = model_and_tokenizer
    num_layers = model.config.num_hidden_layers

    config = getattr(FadeConfig, preset)()
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")

    cache = create_tiered_cache(model, dtype=DTYPE, config=config)
    tracker = AttentionTracker(num_layers=num_layers)

    prompt = "Explain how a CPU cache hierarchy works."
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    out = forward_with_tracking(model, enc.input_ids, cache, tracker=tracker)
    reassign_tiers_by_position(cache, num_layers)

    tok = out.logits[:, -1:, :].argmax(dim=-1)
    for step in range(64):
        out = forward_with_tracking(model, tok, cache, tracker=tracker)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        if (step + 1) % 32 == 0:
            reassign_tiers_by_position(cache, num_layers)

    assert cache.get_seq_length(0) > 0


@pytest.mark.parametrize("preset", ["safe", "balanced", "aggressive"])
def test_compression_below_baseline(model_and_tokenizer, preset):
    """Compressed bytes must be less than FP16 baseline."""
    from transformers import DynamicCache

    from fade import FadeConfig, create_tiered_cache
    from fade.eval.memory import cache_storage_bytes
    from fade.patch import forward_with_tracking
    from fade.policy import reassign_tiers_by_position
    from fade.tracker import AttentionTracker

    model, tokenizer = model_and_tokenizer
    num_layers = model.config.num_hidden_layers

    prompt = "The quick brown fox " * 100
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    # Baseline.
    baseline_cache = DynamicCache()
    with torch.no_grad():
        model(enc.input_ids, past_key_values=baseline_cache, use_cache=True)
    baseline_bytes = cache_storage_bytes(baseline_cache)

    # FADE.
    config = getattr(FadeConfig, preset)()
    if config.eviction_policy == "h2o":
        config = config.with_overrides(eviction_policy="position")
    cache = create_tiered_cache(model, dtype=DTYPE, config=config)
    tracker = AttentionTracker(num_layers=num_layers)
    forward_with_tracking(model, enc.input_ids, cache, tracker=tracker)
    reassign_tiers_by_position(cache, num_layers)
    fade_bytes = cache.compressed_storage_bytes()

    assert fade_bytes < baseline_bytes, f"{preset}: FADE {fade_bytes} >= baseline {baseline_bytes}"


def test_needle_pass(model_and_tokenizer):
    """Needle-in-a-haystack must PASS at 2048 tokens."""
    model, tokenizer = model_and_tokenizer
    from fade.eval.needle import run_needle

    result = run_needle(model, tokenizer, target_tokens=2048, device=DEVICE)
    assert result["passed"], f"Needle FAIL: {result['answer']}"
