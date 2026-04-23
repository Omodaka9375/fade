"""HF ``model.generate()`` compatibility tests (W6).

Verify that ``TieredKVCache`` works as a drop-in ``past_key_values`` for
the standard HuggingFace generate loop — greedy, sampling, and beam search.
Uses a tiny random-init Qwen2 model so no downloads are required.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM

from fade.patch import create_tiered_cache

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


def _make_cache(model):
    return create_tiered_cache(
        model, dtype=DTYPE,
        n_sink=2, recent_window=4, int4_budget=None, int2_budget=0,
    )


def test_generate_greedy():
    model, cfg = _tiny_model()
    cache = _make_cache(model)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(ids, past_key_values=cache, max_new_tokens=6, do_sample=False)
    assert out.shape[0] == 1
    assert out.shape[1] >= 8 + 1  # at least one token generated


def test_generate_sampling():
    model, cfg = _tiny_model()
    cache = _make_cache(model)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(
        ids, past_key_values=cache,
        max_new_tokens=6, do_sample=True, temperature=0.8,
    )
    assert out.shape[0] == 1
    assert out.shape[1] >= 8 + 1


def test_generate_beam_search():
    model, cfg = _tiny_model()
    cache = _make_cache(model)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(
        ids, past_key_values=cache,
        max_new_tokens=6, num_beams=2,
    )
    assert out.shape[0] == 1
    assert out.shape[1] >= 8 + 1


def test_generate_greedy_matches_manual_loop():
    """``model.generate`` greedy should produce the same tokens as a manual loop."""
    model, cfg = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    # Via generate.
    cache_gen = _make_cache(model)
    gen_out = model.generate(
        ids, past_key_values=cache_gen,
        max_new_tokens=4, do_sample=False,
    )

    # Manual greedy loop.
    cache_man = _make_cache(model)
    with torch.no_grad():
        out = model(ids, past_key_values=cache_man, use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    manual_tokens = [tok]
    for _ in range(3):
        with torch.no_grad():
            out = model(tok, past_key_values=cache_man, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        manual_tokens.append(tok)
    manual_out = torch.cat([ids, *manual_tokens], dim=-1)

    # Compare.
    min_len = min(gen_out.shape[1], manual_out.shape[1])
    assert torch.equal(gen_out[:, :min_len], manual_out[:, :min_len])
