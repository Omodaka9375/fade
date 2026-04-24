"""Fully-fused INT4 attention kernel (FlashAttention-2 style).

Single Triton kernel that reads packed INT4 K/V, computes QK^T with
online softmax, accumulates V, and writes fp16 output — without ever
materializing a full fp16 K/V buffer in global memory.

Key trick: split Q into even/odd columns and compute two half-width
dot products against unpacked high/low nibbles. Avoids the interleaving
problem entirely.

Requires: Triton >= 3.0, CUDA GPU with SM >= 80.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if _HAS_TRITON:

    @triton.jit
    def _fused_int4_attn_fwd(
        Q_ptr,
        K_packed_ptr,
        K_scale_ptr,
        V_packed_ptr,
        V_scale_ptr,
        Out_ptr,
        stride_qz,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kn,
        stride_kd,
        stride_ksz,
        stride_ksd,
        stride_vz,
        stride_vn,
        stride_vd,
        stride_vsz,
        stride_vsn,
        stride_oz,
        stride_om,
        stride_ok,
        N_CTX_Q,
        N_CTX_K,
        sm_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_D_HALF: tl.constexpr,
    ):
        """Fused INT4 dequant + attention forward pass.

        Grid: (Z, cdiv(N_CTX_Q, BLOCK_M)) where Z = batch * heads.
        """
        pid_z = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_d_half = tl.arange(0, BLOCK_D_HALF)

        # Load Q block [BLOCK_M, BLOCK_D] in fp16.
        q_ptrs = (
            Q_ptr + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        )
        tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

        # Split Q into even/odd columns for the half-width dot trick.
        q_even = tl.load(
            Q_ptr
            + pid_z * stride_qz
            + offs_m[:, None] * stride_qm
            + (offs_d_half * 2)[None, :] * stride_qk,
            mask=offs_m[:, None] < N_CTX_Q,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M, BLOCK_D_HALF]
        q_odd = tl.load(
            Q_ptr
            + pid_z * stride_qz
            + offs_m[:, None] * stride_qm
            + (offs_d_half * 2 + 1)[None, :] * stride_qk,
            mask=offs_m[:, None] < N_CTX_Q,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_M, BLOCK_D_HALF]

        # Load K scale [1, D] -> split into even/odd: [1, D_HALF] each.
        ks_even = tl.load(
            K_scale_ptr + pid_z * stride_ksz + (offs_d_half * 2)[None, :] * stride_ksd
        ).to(tl.float32)  # [1, BLOCK_D_HALF]
        ks_odd = tl.load(
            K_scale_ptr + pid_z * stride_ksz + (offs_d_half * 2 + 1)[None, :] * stride_ksd
        ).to(tl.float32)

        # Pre-scale Q by sm_scale and K_scale for fewer ops in the inner loop.
        q_even_scaled = q_even * sm_scale
        q_odd_scaled = q_odd * sm_scale

        # Accumulators for online softmax.
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        # Output accumulator: store as even/odd halves.
        acc_even = tl.zeros([BLOCK_M, BLOCK_D_HALF], dtype=tl.float32)
        acc_odd = tl.zeros([BLOCK_M, BLOCK_D_HALF], dtype=tl.float32)

        # Inner loop over K/V blocks.
        for start_n in range(0, N_CTX_K, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_CTX_K

            # --- Unpack K tile [BLOCK_N, BLOCK_D_HALF] uint8 ---
            kp_ptrs = (
                K_packed_ptr
                + pid_z * stride_kz
                + offs_n[:, None] * stride_kn
                + offs_d_half[None, :] * stride_kd
            )
            k_packed = tl.load(kp_ptrs, mask=mask_n[:, None], other=0).to(tl.int32)

            k_high = ((k_packed >> 4) & 0xF).to(tl.int16)
            k_low = (k_packed & 0xF).to(tl.int16)
            k_high = tl.where(k_high >= 8, k_high - 16, k_high)
            k_low = tl.where(k_low >= 8, k_low - 16, k_low)

            # Dequant: k_high * ks_even, k_low * ks_odd.
            k_high_f = k_high.to(tl.float32) * ks_even  # [BLOCK_N, BLOCK_D_HALF]
            k_low_f = k_low.to(tl.float32) * ks_odd

            # QK^T via two half-width dots: S = Q_even @ K_high^T + Q_odd @ K_low^T
            s = tl.dot(q_even_scaled, tl.trans(k_high_f)) + tl.dot(q_odd_scaled, tl.trans(k_low_f))
            # s: [BLOCK_M, BLOCK_N]

            # Mask out-of-bounds keys.
            s = tl.where(mask_n[None, :], s, float("-inf"))

            # Online softmax update.
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc_even = acc_even * alpha[:, None]
            acc_odd = acc_odd * alpha[:, None]
            m_i = m_new

            # --- Unpack V tile [BLOCK_N, BLOCK_D_HALF] uint8 ---
            vp_ptrs = (
                V_packed_ptr
                + pid_z * stride_vz
                + offs_n[:, None] * stride_vn
                + offs_d_half[None, :] * stride_vd
            )
            v_packed = tl.load(vp_ptrs, mask=mask_n[:, None], other=0).to(tl.int32)

            v_high = ((v_packed >> 4) & 0xF).to(tl.int16)
            v_low = (v_packed & 0xF).to(tl.int16)
            v_high = tl.where(v_high >= 8, v_high - 16, v_high)
            v_low = tl.where(v_low >= 8, v_low - 16, v_low)

            # V scale: per-token [BLOCK_N, 1] -> broadcast.
            vs_ptrs = V_scale_ptr + pid_z * stride_vsz + offs_n * stride_vsn
            vs = tl.load(vs_ptrs, mask=mask_n, other=0).to(tl.float32)  # [BLOCK_N]

            v_high_f = v_high.to(tl.float32) * vs[:, None]
            v_low_f = v_low.to(tl.float32) * vs[:, None]

            # Accumulate: O += P @ V (split into even/odd)
            acc_even += tl.dot(p.to(tl.float32), v_high_f)
            acc_odd += tl.dot(p.to(tl.float32), v_low_f)

        # Normalize by softmax denominator.
        acc_even = acc_even / l_i[:, None]
        acc_odd = acc_odd / l_i[:, None]

        # Write output: interleave even/odd back to [BLOCK_M, BLOCK_D].
        out_even_ptrs = (
            Out_ptr
            + pid_z * stride_oz
            + offs_m[:, None] * stride_om
            + (offs_d_half * 2)[None, :] * stride_ok
        )
        out_odd_ptrs = (
            Out_ptr
            + pid_z * stride_oz
            + offs_m[:, None] * stride_om
            + (offs_d_half * 2 + 1)[None, :] * stride_ok
        )
        mask_m = offs_m[:, None] < N_CTX_Q
        tl.store(out_even_ptrs, acc_even.to(tl.float16), mask=mask_m)
        tl.store(out_odd_ptrs, acc_odd.to(tl.float16), mask=mask_m)

    def fused_int4_sdpa(
        q: Tensor,
        k_packed: Tensor,
        k_scale: Tensor,
        v_packed: Tensor,
        v_scale: Tensor,
    ) -> Tensor:
        """Fully-fused INT4 attention (FlashAttention-2 style).

        Args:
            q: [B, H, S_q, D] fp16 queries.
            k_packed: [B, H, S_k, D//2] uint8 bit-packed INT4 K.
            k_scale: [B, H, 1, D] fp16 per-channel K scale.
            v_packed: [B, H, S_k, D//2] uint8 bit-packed INT4 V.
            v_scale: [B, H, S_k, 1] fp16 per-token V scale.

        Returns:
            [B, H, S_q, D] fp16 output. Never materializes fp16 K/V.
        """
        B, H, S_q, D = q.shape
        S_k = k_packed.shape[2]
        D_half = D // 2
        Z = B * H

        # Flatten batch*heads.
        q_flat = q.reshape(Z, S_q, D).contiguous()
        k_flat = k_packed.reshape(Z, S_k, D_half).contiguous()
        ks_flat = k_scale.reshape(Z, 1, D).contiguous()
        v_flat = v_packed.reshape(Z, S_k, D_half).contiguous()
        vs_flat = v_scale.reshape(Z, S_k, 1).contiguous()

        out = torch.empty(Z, S_q, D, dtype=torch.float16, device=q.device)

        sm_scale = 1.0 / math.sqrt(D)
        BLOCK_M = 64
        BLOCK_N = 64

        grid = (Z, triton.cdiv(S_q, BLOCK_M))

        _fused_int4_attn_fwd[grid](
            q_flat,
            k_flat,
            ks_flat,
            v_flat,
            vs_flat,
            out,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            ks_flat.stride(0),
            ks_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            vs_flat.stride(0),
            vs_flat.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            S_q,
            S_k,
            sm_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=D,
            BLOCK_D_HALF=D_half,
        )

        return out.reshape(B, H, S_q, D)


def fused_int4_sdpa_with_fallback(
    q: Tensor,
    k_packed: Tensor,
    k_scale: Tensor,
    v_packed: Tensor,
    v_scale: Tensor,
    force_fused: bool = False,
) -> Tensor:
    """Fused INT4 attention with torch fallback.

    Uses the fully-fused Triton kernel when ``force_fused=True`` and
    hardware supports it; otherwise dequantizes then calls SDPA.
    """
    if force_fused and _HAS_TRITON and q.is_cuda:
        return fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)

    from fade.quant import dequant_int4

    k = dequant_int4(k_packed, k_scale, dtype=q.dtype)
    v = dequant_int4(v_packed, v_scale, dtype=q.dtype)
    return F.scaled_dot_product_attention(q, k, v)


def check_fused_parity(
    B: int = 1,
    H: int = 4,
    S_q: int = 1,
    S_k: int = 256,
    D: int = 128,
    atol: float = 1e-2,
) -> dict:
    """Numeric parity test: fused kernel vs torch fallback."""
    if not _HAS_TRITON or not torch.cuda.is_available():
        return {"passed": False, "error": "Triton or CUDA not available"}

    from fade.quant import quant_k_int4, quant_v_int4

    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn(B, H, S_q, D, dtype=torch.float16, device=device)
    k_fp = torch.randn(B, H, S_k, D, dtype=torch.float16, device=device)
    v_fp = torch.randn(B, H, S_k, D, dtype=torch.float16, device=device)

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)

    ref = fused_int4_sdpa_with_fallback(q, k_packed, k_scale, v_packed, v_scale, force_fused=False)
    fused = fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)

    max_err = (ref - fused).abs().max().item()
    mean_err = (ref - fused).abs().mean().item()

    return {
        "passed": max_err < atol,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "atol": atol,
    }


__all__ = ["check_fused_parity", "fused_int4_sdpa", "fused_int4_sdpa_with_fallback"]
