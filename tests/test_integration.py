"""End-to-end smoke test with a tiny randomly initialized causal LM.

This avoids any downloads: we construct a Qwen2 with a fraction of the real
layer count/hidden size and random weights, then verify:
    1. A forward pass with ``TieredKVCache`` runs.
    2. ``output_attentions=True`` with eager attention produces real tensors.
    3. ``reassign_tiers`` succeeds and the next forward still works.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM

from fade.patch import create_tiered_cache, forward_with_tracking
from fade.policy import reassign_tiers
from fade.tracker import AttentionTracker

DEVICE = "cpu"  # keep the smoke test hermetic
DTYPE = torch.float32


def _tiny_model():
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    model.to(DTYPE).eval()
    return model, cfg


def test_tiered_cache_end_to_end():
    model, cfg = _tiny_model()
    num_layers = cfg.num_hidden_layers

    cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=None,
        int2_budget=0,
    )
    tracker = AttentionTracker(num_layers=num_layers)

    # prefill
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = forward_with_tracking(model, prompt_ids, cache, tracker=tracker)
    assert out.logits.shape == (1, 16, cfg.vocab_size)
    assert out.attentions is not None
    assert cache.get_seq_length(0) == 16

    # one decode step
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    out2 = forward_with_tracking(model, next_tok, cache, tracker=tracker)
    assert out2.logits.shape == (1, 1, cfg.vocab_size)
    assert cache.get_seq_length(0) == 17

    # reassign tiers; middle tokens should move to INT4
    reassign_tiers(cache, tracker, num_layers)
    state = cache._layers[0]
    # sinks (2) + recent (4) = 6 FP16; rest (11) INT4 with int4_budget=None
    n_sink = state.sink_k.shape[-2] if state.sink_k is not None else 0
    n_recent = state.fp16_k.shape[-2] if state.fp16_k is not None else 0
    assert n_sink + n_recent == 6
    assert state.int4_kq.shape[-2] == 17 - 6

    # next decode still works
    next_tok2 = out2.logits[:, -1:, :].argmax(dim=-1)
    out3 = forward_with_tracking(model, next_tok2, cache, tracker=tracker)
    assert out3.logits.shape == (1, 1, cfg.vocab_size)
    assert cache.get_seq_length(0) == 18


def test_tiered_cache_batched_matches_unbatched_per_row():
    """Batched forward produces the same logits as per-row forwards.

    This validates the shared-tier batching contract end-to-end with a real
    (tiny) model: identical prompts replicated across the batch must yield
    identical logits per row when the cache is tiered.
    """
    model, cfg = _tiny_model()
    num_layers = cfg.num_hidden_layers
    B = 3
    S = 12

    # Same prompt replicated across the batch.
    single = torch.randint(0, cfg.vocab_size, (1, S))
    batched_ids = single.expand(B, -1).contiguous()

    # Unbatched reference: run the same prompt through a fresh cache.
    cache_ref = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=None,
        int2_budget=0,
    )
    ref_out = forward_with_tracking(model, single, cache_ref, tracker=None)

    # Batched run.
    cache_batched = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=None,
        int2_budget=0,
    )
    batched_out = forward_with_tracking(model, batched_ids, cache_batched, tracker=None)

    assert cache_batched.batch_size == B
    assert cache_batched.get_seq_length(0) == S
    # Each row's logits must match the unbatched reference.
    for b in range(B):
        diff = (batched_out.logits[b] - ref_out.logits[0]).abs().max().item()
        assert diff < 1e-4, f"row {b} diverges from reference by {diff}"

    # Reassign tiers across the batch — should not crash.
    tracker = AttentionTracker(num_layers=num_layers)
    forward_with_tracking(
        model,
        batched_ids,
        create_tiered_cache(
            model,
            dtype=DTYPE,
            n_sink=2,
            recent_window=4,
            int4_budget=None,
            int2_budget=0,
        ),
        tracker=tracker,
    )
    reassign_tiers(cache_batched, tracker, num_layers)
    # Decode one more step.
    next_tok = batched_out.logits[:, -1:, :].argmax(dim=-1)
    out2 = forward_with_tracking(model, next_tok, cache_batched, tracker=None)
    assert out2.logits.shape == (B, 1, cfg.vocab_size)


def test_tiered_cache_with_eviction():
    """Phase 2: budget-bounded eviction with re-RoPE succeeds end-to-end."""
    model, cfg = _tiny_model()
    num_layers = cfg.num_hidden_layers

    cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=4,
        int2_budget=2,
    )
    tracker = AttentionTracker(num_layers=num_layers)

    # prefill with 20 tokens
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 20))
    out = forward_with_tracking(model, prompt_ids, cache, tracker=tracker)
    assert cache.get_seq_length(0) == 20

    # reassign: 2 sinks + 4 INT4 + 2 INT2 + 4 recent = 12 retained; 8 evicted
    reassign_tiers(cache, tracker, num_layers)
    state = cache._layers[0]
    expected_fp16 = 2 + 4  # sinks + recent
    expected_int4 = 4
    expected_int2 = 2
    n_sink = state.sink_k.shape[-2] if state.sink_k is not None else 0
    n_recent = state.fp16_k.shape[-2] if state.fp16_k is not None else 0
    assert n_sink + n_recent == expected_fp16
    assert state.int4_kq.shape[-2] == expected_int4
    assert state.int2_kq is not None
    assert state.int2_actual_count == expected_int2
    assert cache.get_seq_length(0) == expected_fp16 + expected_int4 + expected_int2

    # decode a few more tokens after eviction — should not crash
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    for _ in range(5):
        out = forward_with_tracking(model, tok, cache, tracker=tracker)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    assert cache.get_seq_length(0) == expected_fp16 + expected_int4 + expected_int2 + 5
