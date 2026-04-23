"""Per-token attention-mass tracking across layers.

Usage (once per forward pass):
    tracker.observe(attn_weights[layer_idx], layer_idx)

Then feed ``tracker.scores(layer_idx)`` into the tier policy.
"""
from __future__ import annotations

import torch
from torch import Tensor

# --- knobs ------------------------------------------------------------------- #
DEFAULT_EMA_ALPHA: float = 0.95  # higher -> longer memory


class AttentionTracker:
    """Exponentially-weighted per-position attention mass, one buffer per layer."""

    def __init__(
        self,
        num_layers: int,
        alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        self.num_layers = num_layers
        self.alpha = alpha
        self._scores: list[Tensor | None] = [None] * num_layers

    def observe(self, attn_weights: Tensor, layer_idx: int) -> None:
        """Accumulate attention mass from one forward pass.

        Args:
            attn_weights: [B, H, Q, K] softmax weights for this layer.
            layer_idx: transformer layer index.
        """
        # mass received per key position, summed over batch, heads, and query positions
        mass = attn_weights.detach().float().sum(dim=(0, 1, 2))  # [K]
        prev = self._scores[layer_idx]
        if prev is None:
            self._scores[layer_idx] = mass.clone()
            return
        if prev.shape[0] < mass.shape[0]:
            pad = torch.zeros(
                mass.shape[0] - prev.shape[0],
                dtype=prev.dtype,
                device=prev.device,
            )
            prev = torch.cat([prev, pad], dim=0)
        elif prev.shape[0] > mass.shape[0]:
            # positions were evicted externally; truncate to current length
            prev = prev[: mass.shape[0]]
        self._scores[layer_idx] = self.alpha * prev + (1.0 - self.alpha) * mass

    def scores(self, layer_idx: int) -> Tensor | None:
        return self._scores[layer_idx]

    def remove_positions(self, layer_idx: int, keep_mask: Tensor) -> None:
        """Drop tracker entries for positions that were evicted from the cache."""
        buf = self._scores[layer_idx]
        if buf is None:
            return
        self._scores[layer_idx] = buf[keep_mask.to(buf.device)]

    def reset(self) -> None:
        self._scores = [None] * self.num_layers
