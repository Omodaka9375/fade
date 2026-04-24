"""Tests for the Phase 3 PQ codebook (W9)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("sklearn", reason="PQ codebook requires scikit-learn")

from fade.codebook import PQCodebook

torch.manual_seed(42)

HEAD_DIM = 64
SUB_DIM = 32
NUM_CENTROIDS = 16  # small for fast tests
N_VECTORS = 512


def _calibration_data() -> torch.Tensor:
    """Synthetic calibration vectors with some cluster structure."""
    # 4 clusters in head_dim space.
    centers = torch.randn(4, HEAD_DIM) * 3
    labels = torch.randint(0, 4, (N_VECTORS,))
    return centers[labels] + torch.randn(N_VECTORS, HEAD_DIM) * 0.5


def test_train_produces_valid_codebook():
    data = _calibration_data()
    cb = PQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    assert cb.centroids.shape == (HEAD_DIM // SUB_DIM, NUM_CENTROIDS, SUB_DIM)
    assert cb.n_sub == HEAD_DIM // SUB_DIM
    assert cb.num_centroids == NUM_CENTROIDS
    assert cb.head_dim == HEAD_DIM
    assert torch.isfinite(cb.centroids).all()


def test_encode_produces_uint8_codes():
    data = _calibration_data()
    cb = PQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    codes = cb.encode(data)
    assert codes.shape == (N_VECTORS, HEAD_DIM // SUB_DIM)
    assert codes.dtype == torch.uint8
    assert codes.max().item() < NUM_CENTROIDS


def test_decode_produces_correct_shape():
    data = _calibration_data()
    cb = PQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    codes = cb.encode(data)
    decoded = cb.decode(codes)
    assert decoded.shape == (N_VECTORS, HEAD_DIM)
    assert decoded.dtype == torch.float32
    assert torch.isfinite(decoded).all()


def test_encode_decode_round_trip_error():
    """Reconstruction error should be significantly lower than random."""
    data = _calibration_data()
    cb = PQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    codes = cb.encode(data)
    decoded = cb.decode(codes)

    recon_err = (data - decoded).pow(2).mean().sqrt().item()
    random_err = data.pow(2).mean().sqrt().item()
    # PQ reconstruction should be well below random magnitude.
    assert recon_err < random_err * 0.5, (
        f"PQ recon error {recon_err:.3f} >= 50% of random magnitude {random_err:.3f}"
    )


def test_encode_batched_shapes():
    """Encode works with batched [..., head_dim] input."""
    data = _calibration_data()
    cb = PQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)

    batched = data.view(4, N_VECTORS // 4, HEAD_DIM)
    codes = cb.encode(batched)
    assert codes.shape == (4, N_VECTORS // 4, HEAD_DIM // SUB_DIM)

    decoded = cb.decode(codes)
    assert decoded.shape == (4, N_VECTORS // 4, HEAD_DIM)


def test_train_rejects_bad_sub_dim():
    data = torch.randn(100, 65)  # 65 not divisible by 32
    with pytest.raises(ValueError, match="divisible"):
        PQCodebook.train(data, sub_dim=32, num_centroids=16)


def test_train_rejects_too_many_centroids():
    data = torch.randn(100, 64)
    with pytest.raises(ValueError, match="uint8"):
        PQCodebook.train(data, sub_dim=32, num_centroids=512)


# --- ResidualPQCodebook ----------------------------------------------------- #
def test_residual_pq_train():
    from fade.codebook import ResidualPQCodebook

    data = _calibration_data()
    rpq = ResidualPQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    assert rpq.primary.centroids.shape == (HEAD_DIM // SUB_DIM, NUM_CENTROIDS, SUB_DIM)
    assert rpq.residual.centroids.shape == rpq.primary.centroids.shape


def test_residual_pq_encode_decode_shapes():
    from fade.codebook import ResidualPQCodebook

    data = _calibration_data()
    rpq = ResidualPQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    codes1, codes2 = rpq.encode(data)
    assert codes1.shape == (N_VECTORS, HEAD_DIM // SUB_DIM)
    assert codes2.shape == codes1.shape
    decoded = rpq.decode(codes1, codes2)
    assert decoded.shape == (N_VECTORS, HEAD_DIM)


def test_residual_pq_beats_single_stage():
    """A3 acceptance: residual PQ reconstruction error < single-stage PQ."""
    from fade.codebook import ResidualPQCodebook

    data = _calibration_data()

    # Single-stage PQ.
    single = PQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    codes_s = single.encode(data)
    recon_s = single.decode(codes_s)
    err_single = (data - recon_s).pow(2).mean().sqrt().item()

    # Residual PQ.
    rpq = ResidualPQCodebook.train(data, sub_dim=SUB_DIM, num_centroids=NUM_CENTROIDS)
    codes1, codes2 = rpq.encode(data)
    recon_r = rpq.decode(codes1, codes2)
    err_residual = (data - recon_r).pow(2).mean().sqrt().item()

    assert err_residual < err_single, (
        f"Residual PQ error {err_residual:.4f} should be < single PQ {err_single:.4f}"
    )
    improvement = 1 - err_residual / err_single
    print(f"Residual PQ improvement: {improvement:.0%} lower RMSE")
