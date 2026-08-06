"""Fused attention integration for manual decode loops.

This module provides helpers to use the INT4 fused kernel in manual decode
loops. For true drop-in HF integration (Option A), see the future work item.

Usage in manual decode:
    from fade.kernels.attention import FusedAttention

    fused_attn = FusedAttention()
    out = fused_attn(q, k_packed, k_scale, v_packed, v_scale)
"""

from __future__ import annotations

from torch import Tensor

from fade.kernels.fused_int4_attn import fused_int4_sdpa, fused_int4_sdpa_with_fallback


class FusedAttention:
    """Fused INT4 attention for manual decode loops.

    This wrapper manages the fused INT4 attention kernel and provides a clean
    interface for use in manual decode loops. It automatically handles:
        - GQA broadcasting (when K/V heads < Q heads)
        - Fallback to dequant+SDPA when Triton is unavailable
        - Causal masking for autoregressive generation

    Example::

        from fade.kernels.attention import FusedAttention
        from fade.quant import quant_k_int4, quant_v_int4

        fused_attn = FusedAttention()

        # In your decode loop:
        q = attn_proj(q)  # [B, H, S_q, D]
        k = attn_proj(k)  # [B, H, S_k, D]
        v = attn_proj(v)  # [B, H, S_k, D]

        # Compress K/V to INT4
        k_packed, k_scale = quant_k_int4(k)
        v_packed, v_scale = quant_v_int4(v)

        # Run fused attention
        out = fused_attn(q, k_packed, k_scale, v_packed, v_scale, is_causal=True)
    """

    def __init__(self, force_fused: bool = False):
        """Initialize fused attention.

        Args:
            force_fused: If True, raise an error if the fused kernel cannot
                run (instead of falling back to dequant+SDPA). Useful for
                benchmarking to ensure the fused path is always taken.
        """
        self.force_fused = force_fused

    def __call__(
        self,
        q: Tensor,
        k_packed: Tensor,
        k_scale: Tensor,
        v_packed: Tensor,
        v_scale: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        """Run fused INT4 attention.

        Args:
            q: Query tensor [B, H_q, S_q, D] in fp16.
            k_packed: Packed INT4 keys [B, H_kv, S_k, D//2] in uint8.
            k_scale: K scales [B, H_kv, 1, D] in fp16.
            v_packed: Packed INT4 values [B, H_kv, S_k, D//2] in uint8.
            v_scale: V scales [B, H_kv, S_k, 1] in fp16.
            is_causal: If True, apply causal masking (only attend to past).

        Returns:
            Output tensor [B, H_q, S_q, D] in fp16.

        Raises:
            RuntimeError: If ``force_fused=True`` and the fused kernel cannot run.
        """
        if self.force_fused:
            from fade.kernels.fused_int4_attn import _HAS_TRITON

            if not _HAS_TRITON or not q.is_cuda:
                raise RuntimeError(
                    "Fused kernel requested but Triton or CUDA not available. "
                    "Install triton and ensure you're running on CUDA."
                )
            return fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale, is_causal=is_causal)
        else:
            return fused_int4_sdpa_with_fallback(
                q, k_packed, k_scale, v_packed, v_scale, force_fused=False
            )


def apply_fused_attention(
    q: Tensor,
    k_packed: Tensor,
    k_scale: Tensor,
    v_packed: Tensor,
    v_scale: Tensor,
    is_causal: bool = False,
) -> Tensor:
    """Convenience function for fused INT4 attention.

    Equivalent to ``FusedAttention().__call__()`` but as a standalone function.

    Args:
        q: Query tensor [B, H_q, S_q, D] in fp16.
        k_packed: Packed INT4 keys [B, H_kv, S_k, D//2] in uint8.
        k_scale: K scales [B, H_kv, 1, D] in fp16.
        v_packed: Packed INT4 values [B, H_kv, S_k, D//2] in uint8.
        v_scale: V scales [B, H_kv, S_k, 1] in fp16.
        is_causal: If True, apply causal masking.

    Returns:
        Output tensor [B, H_q, S_q, D] in fp16.
    """
    fused_attn = FusedAttention()
    return fused_attn(q, k_packed, k_scale, v_packed, v_scale, is_causal=is_causal)
