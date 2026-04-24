"""Learned eviction policy: a tiny MLP that predicts token keep-probability.

The MLP takes per-token features (position, attention mass, layer depth,
token age) and outputs a scalar keep-probability. Tokens are ranked by
this probability and the top-k survive in INT4; the rest are evicted.

This replaces the H2O oracle for prompts above ``PREFILL_TRACK_LIMIT``
where full prefill attention is too expensive to compute.

Usage:
    from fade.learned_policy import EvictionMLP, reassign_tiers_learned

    mlp = EvictionMLP.load("fade/checkpoints/eviction_mlp.pt")
    reassign_tiers_learned(cache, mlp, num_layers, step=current_step)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from fade.cache import TIER_EVICT, TIER_FP16, TIER_INT2, TIER_INT4, TieredKVCache

# --- knobs ------------------------------------------------------------------ #
FEATURE_DIM: int = 4  # (position, attn_mass, layer_depth, token_age)
HIDDEN_DIM: int = 32
DEFAULT_CHECKPOINT: str = "fade/checkpoints/eviction_mlp.pt"


class EvictionMLP(nn.Module):
    """Tiny MLP: ``[position, attn_mass, layer_depth, token_age] -> keep_prob``.

    Two hidden layers with ReLU, sigmoid output. ~2K parameters.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Predict keep-probability for each token.

        Args:
            features: [S, feature_dim] or [B, S, feature_dim].

        Returns:
            [S] or [B, S] keep-probabilities in [0, 1].
        """
        return self.net(features).squeeze(-1)

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> EvictionMLP:
        """Load a trained checkpoint."""
        mlp = cls(**kwargs)
        mlp.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        mlp.eval()
        return mlp


def _build_features(
    S: int,
    scores: Tensor | None,
    layer_idx: int,
    num_layers: int,
    step: int,
    device: torch.device,
) -> Tensor:
    """Build [S, 4] feature matrix for the MLP.

    Features (all normalized to [0, 1]):
        0: position / S
        1: attention mass (normalized to [0, 1] via min-max)
        2: layer_depth / num_layers
        3: token_age = (step - position) / step (older = higher)
    """
    positions = torch.arange(S, dtype=torch.float32, device=device)
    pos_norm = positions / max(S - 1, 1)

    if scores is not None and scores.numel() == S:
        s = scores.float().to(device)
        s_min, s_max = s.min(), s.max()
        mass_norm = (s - s_min) / (s_max - s_min + 1e-8)
    else:
        mass_norm = torch.zeros(S, device=device)

    layer_norm = torch.full((S,), layer_idx / max(num_layers - 1, 1), device=device)

    age = (step - positions) / max(step, 1)
    age = age.clamp(0, 1)

    return torch.stack([pos_norm, mass_norm, layer_norm, age], dim=-1)  # [S, 4]


def reassign_tiers_learned(
    cache: TieredKVCache,
    mlp: EvictionMLP,
    num_layers: int,
    step: int = 1,
    scores_per_layer: list[Tensor | None] | None = None,
) -> None:
    """Assign tiers using the learned eviction MLP.

    Tokens in the middle (between sinks and recent window) are scored by the
    MLP; the top ``int4_budget`` survive as INT4, the rest are evicted.
    """
    for layer_idx in range(num_layers):
        if not cache.is_managed(layer_idx):
            continue
        S = int(cache.get_seq_length(layer_idx))
        if S == 0:
            continue

        scores = None
        if scores_per_layer is not None and layer_idx < len(scores_per_layer):
            scores = scores_per_layer[layer_idx]

        state = cache._layers[layer_idx]
        ref = state.fp16_k if state.fp16_k is not None else state.int4_kq
        device = ref.device if ref is not None else torch.device("cpu")

        features = _build_features(S, scores, layer_idx, num_layers, step, device)

        with torch.no_grad():
            keep_prob = mlp.to(device)(features)  # [S]

        # Build tier assignment using the same sink/recent/budget structure.
        tiers = torch.full((S,), TIER_EVICT, dtype=torch.long, device=device)
        sink_end = min(cache.n_sink, S)
        tiers[:sink_end] = TIER_FP16
        recent_start = max(sink_end, S - cache.recent_window)
        tiers[recent_start:] = TIER_FP16

        # Middle tokens: rank by MLP keep_prob.
        middle_mask = tiers == TIER_EVICT
        middle_idx = middle_mask.nonzero(as_tuple=False).squeeze(-1)
        if middle_idx.numel() > 0:
            if cache.int4_budget is None or cache.int4_budget >= middle_idx.numel():
                tiers[middle_idx] = TIER_INT4
            else:
                middle_scores = keep_prob[middle_idx]
                top_k = middle_scores.topk(cache.int4_budget).indices
                tiers[middle_idx[top_k]] = TIER_INT4

                # INT2 budget for remaining.
                remaining = (tiers == TIER_EVICT).nonzero(as_tuple=False).squeeze(-1)
                if remaining.numel() > 0 and cache.int2_budget > 0:
                    rem_scores = keep_prob[remaining]
                    k2 = min(cache.int2_budget, remaining.numel())
                    top2 = rem_scores.topk(k2).indices
                    tiers[remaining[top2]] = TIER_INT2

        cache.apply_tier_assignment(layer_idx, tiers)


__all__ = [
    "EvictionMLP",
    "reassign_tiers_learned",
]
