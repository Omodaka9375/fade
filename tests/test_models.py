"""Per-architecture integration tests (W3).

Each test creates a tiny random-init model from ``AutoConfig.for_model``,
runs a prefill + 1 decode step through ``TieredKVCache``, and verifies that:
    1. The cache produces the expected sequence length.
    2. ``reassign_tiers`` (no eviction) succeeds and the middle moves to INT4.
    3. The correct RoPE scheme was auto-detected.

No HuggingFace model downloads are required; all weights are randomly
initialized. Tests are CPU-only.

Covered architectures:
    * Qwen2      — vanilla RoPE, GQA (num_kv_heads < num_attention_heads)
    * Llama      — vanilla RoPE; also with rope_scaling for Llama-3.1
    * Mistral    — vanilla RoPE, sliding-window attention
    * Phi-3      — vanilla RoPE, long context
    * Gemma-2    — vanilla RoPE, query pre-norm
    * Falcon     — ALiBi (non-RoPE) via alibi=True
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM

from fade.patch import create_tiered_cache, forward_with_tracking
from fade.policy import reassign_tiers
from fade.rope import Llama3, NoRope, Vanilla
from fade.tracker import AttentionTracker

DEVICE = "cpu"
DTYPE = torch.float32


def _run_model(model_type: str, extra_cfg: dict | None = None, expect_scheme=Vanilla):
    """Run prefill + decode + tier reassignment on a tiny model.

    Returns ``(cache, model, cfg)`` for further assertions.
    """
    cfg_kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    # Per-architecture overrides.
    if model_type == "qwen2" or model_type == "llama":
        cfg_kwargs["num_key_value_heads"] = 2
    elif model_type == "mistral":
        cfg_kwargs["num_key_value_heads"] = 2
        cfg_kwargs["sliding_window"] = 64
    elif model_type == "phi3":
        cfg_kwargs["num_key_value_heads"] = 2
        # Phi-3 defaults padding_idx to 32000; must fit within vocab_size.
        cfg_kwargs["vocab_size"] = max(cfg_kwargs.get("vocab_size", 128), 32064)
        cfg_kwargs.setdefault("original_max_position_embeddings", 128)
    elif model_type == "gemma2":
        cfg_kwargs["num_key_value_heads"] = 2
        cfg_kwargs["head_dim"] = 16
    elif model_type == "falcon":
        cfg_kwargs["num_key_value_heads"] = 2
        cfg_kwargs.pop("intermediate_size", None)

    if extra_cfg:
        cfg_kwargs.update(extra_cfg)

    cfg = AutoConfig.for_model(model_type, **cfg_kwargs)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    model.to(DTYPE).eval()

    num_layers = cfg.num_hidden_layers
    cache = create_tiered_cache(
        model, dtype=DTYPE,
        n_sink=2, recent_window=4, int4_budget=None, int2_budget=0,
    )

    # Check auto-detected scheme.
    assert isinstance(cache.rope_scheme, expect_scheme), (
        f"expected {expect_scheme.__name__}, got {type(cache.rope_scheme).__name__}"
    )

    tracker = AttentionTracker(num_layers=num_layers)
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 16))

    # Prefill.
    out = forward_with_tracking(model, prompt_ids, cache, tracker=tracker)
    assert out.logits.shape == (1, 16, cfg.vocab_size)
    assert cache.get_seq_length(0) == 16

    # Decode.
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    out2 = forward_with_tracking(model, tok, cache, tracker=tracker)
    assert cache.get_seq_length(0) == 17

    # Reassign tiers — sinks(2) + recent(4) = 6 FP16, rest INT4.
    reassign_tiers(cache, tracker, num_layers)
    state = cache._layers[0]
    assert state.fp16_k.shape[-2] == 6
    assert state.int4_kq.shape[-2] == 17 - 6

    # Decode after reassignment.
    tok2 = out2.logits[:, -1:, :].argmax(dim=-1)
    out3 = forward_with_tracking(model, tok2, cache, tracker=tracker)
    assert out3.logits.shape == (1, 1, cfg.vocab_size)
    assert cache.get_seq_length(0) == 18

    return cache, model, cfg


# --- per-architecture tests ------------------------------------------------- #
def test_qwen2():
    _run_model("qwen2")


def test_llama_vanilla():
    _run_model("llama")


def test_llama3_scaling():
    """Llama-3.1 style rope_scaling with frequency-dependent interpolation."""
    _run_model(
        "llama",
        extra_cfg={
            "rope_theta": 500000.0,
            "rope_scaling": {
                "type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 128,
            },
        },
        expect_scheme=Llama3,
    )


def test_mistral():
    _run_model("mistral")


def test_phi3():
    _run_model("phi3")


def test_gemma2():
    _run_model("gemma2")


def test_falcon_alibi():
    """Falcon with ALiBi — non-RoPE, eviction skips re-RoPE.

    Falcon's ALiBi mask path requires torch >= 2.6 in some transformers
    builds; skip if that constraint isn't met.
    """
    try:
        _run_model("falcon", extra_cfg={"alibi": True}, expect_scheme=NoRope)
    except ValueError as e:
        if "torch>=2.6" in str(e) or "or_mask_function" in str(e):
            pytest.skip(f"Falcon ALiBi mask requires newer torch: {e}")
        raise


# --- GQA / MQA scale shape tests ------------------------------------------- #
def test_gqa_int4_scale_shapes():
    """GQA (num_kv_heads=2, num_attention_heads=4): INT4 K scales are [B, kv_heads, 1, D]."""
    cache, _model, cfg = _run_model("qwen2")
    state = cache._layers[0]
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    num_kv_heads = cfg.num_key_value_heads
    # K scales: per-channel along head_dim, shared across seq.
    assert state.int4_ks.shape[1] == num_kv_heads
    assert state.int4_ks.shape[-1] == head_dim
    # V scales: per-token, one scale per position.
    assert state.int4_vs.shape[1] == num_kv_heads
    assert state.int4_vs.shape[-1] == 1
