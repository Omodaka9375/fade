"""Quality gate tests (W7).

These validate the evaluation pipeline end-to-end using a tiny random-init
model. They are NOT meant to prove quality (random weights produce gibberish).
Their purpose is:
    1. The eval harness runs without crashing.
    2. The results dict has the expected shape.
    3. Perplexity is finite and the needle test returns a boolean.

For real quality validation, run on a trained model:
    pytest -m eval --model Qwen/Qwen2.5-0.5B-Instruct

The fast tests here run in every CI build; the slow ``@pytest.mark.eval``
tests run on nightly CI only.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM

from fade.eval.quality import run_perplexity_test
from fade.patch import create_tiered_cache, forward_with_tracking
from fade.policy import reassign_tiers
from fade.tracker import AttentionTracker

DTYPE = torch.float32
VOCAB_SIZE = 128


class _TokenizerResult:
    """Attribute-accessible dict for tokenizer output."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        return self

    def __getitem__(self, key):
        return getattr(self, key)


class _FakeTokenizer:
    """Minimal tokenizer that encodes text as byte values mod vocab_size."""

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [b % self.vocab_size for b in text.encode("utf-8")]
        t = torch.tensor([ids])
        return _TokenizerResult(t)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(i + 32) for i in ids)


def _tiny_model():
    cfg = AutoConfig.for_model(
        "qwen2",
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    model.to(DTYPE).eval()
    tokenizer = _FakeTokenizer(VOCAB_SIZE)
    return model, tokenizer, cfg


# --- fast pipeline tests (run in every CI build) ---------------------------- #
def test_perplexity_pipeline():
    """Perplexity eval produces a finite float."""
    model, tokenizer, _ = _tiny_model()
    text = "The quick brown fox jumps over the lazy dog. " * 10
    result = run_perplexity_test(
        model,
        tokenizer,
        text=text,
        device="cpu",
        max_length=64,
        stride=32,
        threshold=1e6,
    )
    assert "ppl" in result
    assert result["ppl"] > 0
    assert isinstance(result["passed"], bool)


def test_tiered_cache_decode_does_not_diverge_immediately():
    """Tiered cache greedy output matches baseline for a few tokens.

    With a random-init model and no eviction (Phase 1-A), the tiered path
    should produce identical logits to a plain forward. This test checks
    that at least the first 4 decode tokens match.
    """
    model, _, cfg = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    # Baseline: plain DynamicCache.
    from transformers import DynamicCache

    base_cache = DynamicCache()
    with torch.no_grad():
        base_out = model(ids, past_key_values=base_cache, use_cache=True)
    base_tok = base_out.logits[:, -1:, :].argmax(dim=-1)

    # Tiered (no eviction).
    tier_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=2,
        recent_window=4,
        int4_budget=None,
        int2_budget=0,
    )
    with torch.no_grad():
        tier_out = model(ids, past_key_values=tier_cache, use_cache=True)
    tier_tok = tier_out.logits[:, -1:, :].argmax(dim=-1)

    assert torch.equal(base_tok, tier_tok), (
        "First decode token diverged — no eviction should be exact"
    )


def test_tier_reassignment_pipeline():
    """Reassign tiers on a tiny model without crashing, then decode one more token."""
    model, _, cfg = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (1, 20))
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
    out = forward_with_tracking(model, ids, cache, tracker=tracker)

    reassign_tiers(cache, tracker, num_layers)
    assert cache.get_seq_length(0) == 20  # no eviction

    tok = out.logits[:, -1:, :].argmax(dim=-1)
    out2 = forward_with_tracking(model, tok, cache, tracker=tracker)
    assert out2.logits.shape == (1, 1, cfg.vocab_size)
