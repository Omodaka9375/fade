"""Unit tests for ``fade.rope`` scheme abstraction."""
from __future__ import annotations

import types
import warnings

import torch

from fade.rope import (
    LinearScaled,
    Llama3,
    NoRope,
    NtkAware,
    Vanilla,
    Yarn,
    extract_rope_scheme,
)

THETA = 10000.0
HEAD_DIM = 64
DEVICE = torch.device("cpu")


# --- inv_freq sanity ------------------------------------------------------- #
def test_vanilla_inv_freq_shape_and_monotonic():
    v = Vanilla(theta=THETA, head_dim=HEAD_DIM)
    inv = v.inv_freq(DEVICE)
    assert inv.shape == (HEAD_DIM // 2,)
    # inv_freq should be strictly decreasing (higher dims → slower freq).
    diffs = inv[1:] - inv[:-1]
    assert (diffs < 0).all()


def test_linear_scaled_stretches_positions():
    v = Vanilla(theta=THETA, head_dim=HEAD_DIM)
    ls = LinearScaled(theta=THETA, head_dim=HEAD_DIM, factor=2.0)
    positions = torch.arange(16)
    cos_v, _sin_v = v.compute_cos_sin(positions, DEVICE)
    cos_ls, _sin_ls = ls.compute_cos_sin(positions, DEVICE)
    # At position 0 both should agree (pos/factor = 0).
    assert torch.allclose(cos_v[:, :, :1], cos_ls[:, :, :1], atol=1e-6)
    # At position 2, linear-scaled should match vanilla at position 1.
    cos_v_1, _sin_v_1 = v.compute_cos_sin(torch.tensor([1]), DEVICE)
    assert torch.allclose(cos_v_1, cos_ls[:, :, 2:3], atol=1e-5)


def test_ntk_aware_changes_base_theta():
    ntk = NtkAware(theta=THETA, head_dim=HEAD_DIM, factor=2.0)
    van = Vanilla(theta=THETA * 2.0, head_dim=HEAD_DIM)
    inv_ntk = ntk.inv_freq(DEVICE)
    inv_van = van.inv_freq(DEVICE)
    assert torch.allclose(inv_ntk, inv_van)


def test_llama3_inv_freq_three_bands():
    """Llama3 should produce frequencies in three bands: keep / blend / scale."""
    l3 = Llama3(
        theta=THETA, head_dim=HEAD_DIM, factor=4.0,
        low_freq_factor=1.0, high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )
    base = Vanilla(theta=THETA, head_dim=HEAD_DIM)
    inv_l3 = l3.inv_freq(DEVICE)
    inv_base = base.inv_freq(DEVICE)
    assert inv_l3.shape == inv_base.shape
    # The highest-freq dims should be unchanged.
    assert torch.allclose(inv_l3[:4], inv_base[:4], atol=1e-6)
    # The lowest-freq dims should be scaled down by factor.
    low_dims = inv_l3[-4:]
    expected_low = inv_base[-4:] / 4.0
    assert torch.allclose(low_dims, expected_low, atol=1e-6)


def test_yarn_inv_freq_shape():
    y = Yarn(theta=THETA, head_dim=HEAD_DIM, factor=2.0)
    inv = y.inv_freq(DEVICE)
    assert inv.shape == (HEAD_DIM // 2,)
    assert torch.isfinite(inv).all()


def test_norope_identity():
    nr = NoRope(theta=THETA, head_dim=HEAD_DIM)
    positions = torch.arange(8)
    cos, sin = nr.compute_cos_sin(positions, DEVICE)
    assert cos.shape == (1, 1, 8, HEAD_DIM)
    assert sin.shape == (1, 1, 8, HEAD_DIM)
    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin))
    assert not nr.is_rope


def test_vanilla_is_rope():
    v = Vanilla(theta=THETA, head_dim=HEAD_DIM)
    assert v.is_rope


# --- extract_rope_scheme --------------------------------------------------- #
def _mock_cfg(**kwargs):
    return types.SimpleNamespace(**kwargs)


def test_extract_no_scaling():
    cfg = _mock_cfg(rope_theta=50000.0, hidden_size=128, num_attention_heads=4)
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Vanilla)
    assert scheme.theta == 50000.0
    assert scheme.head_dim == 32


def test_extract_linear_scaling():
    cfg = _mock_cfg(
        rope_theta=THETA, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={"type": "linear", "factor": 4.0},
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, LinearScaled)
    assert scheme.factor == 4.0


def test_extract_llama3():
    cfg = _mock_cfg(
        rope_theta=500000.0, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={
            "type": "llama3", "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Llama3)
    assert scheme.factor == 8.0
    assert scheme.original_max_position_embeddings == 8192


def test_extract_dynamic_ntk():
    cfg = _mock_cfg(
        rope_theta=THETA, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={"type": "dynamic", "factor": 2.0},
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, NtkAware)


def test_extract_yarn():
    cfg = _mock_cfg(
        rope_theta=THETA, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={
            "type": "yarn", "factor": 4.0,
            "beta_fast": 16.0, "beta_slow": 2.0,
            "original_max_position_embeddings": 4096,
        },
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Yarn)
    assert scheme.beta_fast == 16.0


def test_extract_alibi():
    cfg = _mock_cfg(alibi=True, hidden_size=HEAD_DIM, num_attention_heads=1)
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, NoRope)
    assert not scheme.is_rope


def test_extract_default_type_is_vanilla():
    """Transformers 5.x emits rope_scaling={'type': 'default'}."""
    cfg = _mock_cfg(
        rope_theta=THETA, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={"type": "default"},
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Vanilla)


def test_extract_unknown_type_warns():
    cfg = _mock_cfg(
        rope_theta=THETA, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={"type": "exotic_new_type", "factor": 3.0},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Vanilla)
    assert any("Unknown rope_scaling type" in str(w.message) for w in caught)


def test_extract_rope_type_key():
    """Some configs use 'rope_type' instead of 'type'."""
    cfg = _mock_cfg(
        rope_theta=THETA, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={"rope_type": "ntk", "factor": 2.0},
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, NtkAware)


# --- Proportional RoPE (Gemma 4) ------------------------------------------- #
def test_proportional_full_rotation_matches_vanilla():
    """With partial_rotary_factor=1.0, Proportional should produce the same
    inv_freq as Vanilla (but computed with head_dim denominator)."""
    from fade.rope import Proportional

    prop = Proportional(theta=THETA, head_dim=HEAD_DIM, partial_rotary_factor=1.0)
    van = Vanilla(theta=THETA, head_dim=HEAD_DIM)
    inv_p = prop.inv_freq(DEVICE)
    inv_v = van.inv_freq(DEVICE)
    assert inv_p.shape == inv_v.shape
    assert torch.allclose(inv_p, inv_v, atol=1e-6)


def test_proportional_partial_rotation_zero_pads():
    """With partial_rotary_factor=0.25, the last 75% of inv_freq should be zero."""
    from fade.rope import Proportional

    prop = Proportional(theta=THETA, head_dim=HEAD_DIM, partial_rotary_factor=0.25)
    inv = prop.inv_freq(DEVICE)
    assert inv.shape == (HEAD_DIM // 2,)
    rotary_angles = HEAD_DIM * 25 // 100 // 2  # 8
    # Rotated dims should be nonzero.
    assert (inv[:rotary_angles] > 0).all()
    # Non-rotated dims should be exactly zero.
    assert (inv[rotary_angles:] == 0).all()


def test_proportional_cos_sin_identity_on_non_rotated():
    """Non-rotated dims should have cos=1, sin=0.

    inv_freq has shape [head_dim//2]. Rotated angles occupy the first
    ``rotary_dim//2`` slots; the rest are zero-padded. After
    ``emb = cat(freqs, freqs)`` (shape [S, head_dim]), the zero-padded
    indices are ``[rotary_dim//2 .. head_dim//2-1]`` and
    ``[head_dim//2 + rotary_dim//2 .. head_dim-1]``.
    """
    from fade.rope import Proportional

    prop = Proportional(theta=THETA, head_dim=HEAD_DIM, partial_rotary_factor=0.25)
    positions = torch.arange(8)
    cos, sin = prop.compute_cos_sin(positions, DEVICE)
    assert cos.shape == (1, 1, 8, HEAD_DIM)
    half = HEAD_DIM // 2
    rope_angles = prop.rotary_dim // 2  # 8
    # First half non-rotated: indices rope_angles..half-1.
    assert torch.allclose(cos[:, :, :, rope_angles:half], torch.ones_like(cos[:, :, :, rope_angles:half]), atol=1e-5)
    assert torch.allclose(sin[:, :, :, rope_angles:half], torch.zeros_like(sin[:, :, :, rope_angles:half]), atol=1e-5)
    # Second half non-rotated: indices half+rope_angles..head_dim-1.
    assert torch.allclose(cos[:, :, :, half + rope_angles:], torch.ones_like(cos[:, :, :, half + rope_angles:]), atol=1e-5)
    assert torch.allclose(sin[:, :, :, half + rope_angles:], torch.zeros_like(sin[:, :, :, half + rope_angles:]), atol=1e-5)


def test_extract_proportional_scheme():
    from fade.rope import Proportional

    cfg = _mock_cfg(
        rope_theta=1_000_000.0, hidden_size=HEAD_DIM, num_attention_heads=1,
        rope_scaling={"type": "proportional", "partial_rotary_factor": 0.25},
    )
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Proportional)
    assert scheme.partial_rotary_factor == 0.25
    assert scheme.theta == 1_000_000.0


def test_extract_gemma4_per_layer_rope_parameters():
    """Gemma 4 config with per-layer-type rope_parameters dict."""
    from fade.rope import Proportional, Vanilla, extract_rope_schemes_per_layer

    cfg = _mock_cfg(
        hidden_size=HEAD_DIM, num_attention_heads=1, head_dim=HEAD_DIM,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
    )
    per_layer = extract_rope_schemes_per_layer(cfg)
    assert per_layer is not None
    assert isinstance(per_layer["sliding_attention"], Vanilla)
    assert isinstance(per_layer["full_attention"], Proportional)
    assert per_layer["full_attention"].theta == 1_000_000.0

    # extract_rope_scheme should return the full_attention scheme.
    scheme = extract_rope_scheme(cfg)
    assert isinstance(scheme, Proportional)
    assert scheme.partial_rotary_factor == 0.25
