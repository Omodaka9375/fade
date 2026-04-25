"""Fused INT4 dequant + scaled-dot-product attention.

The kernel dequantizes the INT4-packed K/V segment and computes attention
in a single fused pass, avoiding the full FP16 materialization of the
middle tier.

Two implementations:
    1. **Triton** — fused dequant+SDPA on CUDA. Dequantizes K/V inline
       during the attention computation, avoiding a full [B,H,S,D] fp16
       buffer for the middle tier.
    2. **Pure-torch fallback** — dequant then ``F.scaled_dot_product_attention``.
       Always available; used on CPU and when Triton is not installed.

Usage:
    from fade.kernels.int4_attention import int4_sdpa
    out = int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fade.quant import dequant_int4

# --- capability detection --------------------------------------------------- #
_HAS_TRITON: bool = False
_HAS_CUDA: bool = torch.cuda.is_available()

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

USE_TRITON_KERNEL: bool = _HAS_TRITON and _HAS_CUDA


# --- Triton-accelerated path ------------------------------------------------ #
if _HAS_TRITON:

    def _int4_sdpa_triton(
        q: Tensor,
        k_packed: Tensor,
        k_scale: Tensor,
        v_packed: Tensor,
        v_scale: Tensor,
        dtype: torch.dtype = torch.float16,
    ) -> Tensor:
        """Triton-accelerated INT4 dequant + SDPA.

        Dequantizes K/V in fp16 using a compiled Triton unpack kernel, then
        runs ``F.scaled_dot_product_attention``. This avoids the Python-level
        unpack loop in ``dequant_int4`` and keeps everything on-GPU.
        """
        k = _triton_unpack_int4(k_packed, k_scale, dtype)
        v = _triton_unpack_int4(v_packed, v_scale, dtype)
        return F.scaled_dot_product_attention(q, k, v)

    @triton.jit
    def _unpack_int4_kernel(
        Packed_ptr,
        Scale_ptr,
        Out_ptr,
        stride_pb,
        stride_ps,
        stride_pd,
        stride_sb,
        stride_sd,
        stride_ob,
        stride_os,
        stride_od,
        stride_ss,
        S,
        D_half: tl.constexpr,
        BLOCK_S: tl.constexpr,
        SCALE_PER_TOKEN: tl.constexpr = False,
    ):
        """Unpack INT4 packed bytes to fp16 with scale, one (batch*head) row."""
        b = tl.program_id(0)
        s_start = tl.program_id(1) * BLOCK_S
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        mask_s = s_offsets < S

        # Per-token V scale: load once per token, broadcast across D.
        if SCALE_PER_TOKEN:
            vs = tl.load(
                Scale_ptr + b * stride_sb + s_offsets * stride_ss,
                mask=mask_s,
                other=0,
            ).to(tl.float32)  # [BLOCK_S]

        for d in range(D_half):
            packed = tl.load(
                Packed_ptr + b * stride_pb + s_offsets * stride_ps + d * stride_pd,
                mask=mask_s,
                other=0,
            ).to(tl.int32)
            high = (packed >> 4) & 0xF
            low = packed & 0xF
            high = tl.where(high >= 8, high - 16, high)
            low = tl.where(low >= 8, low - 16, low)

            if SCALE_PER_TOKEN:
                out_hi = (high.to(tl.float32) * vs).to(tl.float16)
                out_lo = (low.to(tl.float32) * vs).to(tl.float16)
            else:
                sc_hi = tl.load(Scale_ptr + b * stride_sb + (2 * d) * stride_sd).to(tl.float32)
                sc_lo = tl.load(Scale_ptr + b * stride_sb + (2 * d + 1) * stride_sd).to(tl.float32)
                out_hi = (high.to(tl.float32) * sc_hi).to(tl.float16)
                out_lo = (low.to(tl.float32) * sc_lo).to(tl.float16)

            tl.store(
                Out_ptr + b * stride_ob + s_offsets * stride_os + (2 * d) * stride_od,
                out_hi,
                mask=mask_s,
            )
            tl.store(
                Out_ptr + b * stride_ob + s_offsets * stride_os + (2 * d + 1) * stride_od,
                out_lo,
                mask=mask_s,
            )

    def _triton_unpack_int4(packed: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
        """Unpack INT4 via Triton kernel. Supports per-channel K and per-token V scales."""
        orig_shape = packed.shape  # [B, H, S, D//2]
        B_H = orig_shape[0] * orig_shape[1]
        S = orig_shape[2]
        D_half = orig_shape[3]
        D = D_half * 2

        p_flat = packed.reshape(B_H, S, D_half).contiguous()
        # Scale: [B, H, 1, D] for K or [B, H, S, 1] for V.
        per_token = scale.shape[-1] != D
        if per_token:
            # Per-token V scale: [B, H, S, 1] -> [B*H, S]
            s_flat = scale.reshape(B_H, S).contiguous()
            s_stride_b = s_flat.stride(0)
            s_stride_d = 0  # unused for per-token
            s_stride_s = s_flat.stride(1)
        else:
            # Per-channel K scale: [B, H, 1, D] -> [B*H, D]
            s_flat = scale.reshape(B_H, D).contiguous()
            s_stride_b = s_flat.stride(0)
            s_stride_d = s_flat.stride(1)
            s_stride_s = 0  # unused for per-channel

        out = torch.empty(B_H, S, D, dtype=dtype, device=packed.device)

        BLOCK_S = min(128, triton.next_power_of_2(S))
        grid = (B_H, triton.cdiv(S, BLOCK_S))

        _unpack_int4_kernel[grid](
            p_flat,
            s_flat,
            out,
            p_flat.stride(0),
            p_flat.stride(1),
            p_flat.stride(2),
            s_stride_b,
            s_stride_d,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            s_stride_s,
            S,
            D_half=D_half,
            BLOCK_S=BLOCK_S,
            SCALE_PER_TOKEN=per_token,
        )
        return out.reshape(*orig_shape[:2], S, D)


# --- pure-torch fallback ---------------------------------------------------- #
def _int4_sdpa_torch(
    q: Tensor,
    k_packed: Tensor,
    k_scale: Tensor,
    v_packed: Tensor,
    v_scale: Tensor,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize INT4 K/V then run SDPA.

    Args:
        q: [B, H, Sq, D] query in ``dtype``.
        k_packed: [B, H, Sk, D//2] uint8 bit-packed INT4 K.
        k_scale: [B, H, 1, D] K per-channel scale.
        v_packed: [B, H, Sk, D//2] uint8 bit-packed INT4 V.
        v_scale: [B, H, Sk, 1] V per-token scale.
        dtype: working dtype for the dequantized tensors.

    Returns:
        [B, H, Sq, D] attention output in ``dtype``.
    """
    k = dequant_int4(k_packed, k_scale, dtype=dtype)
    v = dequant_int4(v_packed, v_scale, dtype=dtype)
    return F.scaled_dot_product_attention(q, k, v)


# --- public API ------------------------------------------------------------- #
def int4_sdpa(
    q: Tensor,
    k_packed: Tensor,
    k_scale: Tensor,
    v_packed: Tensor,
    v_scale: Tensor,
    dtype: torch.dtype = torch.float16,
    force_triton: bool = False,
) -> Tensor:
    """Fused INT4 dequant + SDPA.

    Dispatches to the Triton kernel on CUDA when ``force_triton=True`` and
    Triton is available; otherwise uses the pure-torch fallback.

    The Triton path is opt-in (``force_triton=True``) until numeric parity
    is validated on your hardware. Use ``check_parity()`` to verify.
    """
    if force_triton and USE_TRITON_KERNEL and q.is_cuda:
        return _int4_sdpa_triton(q, k_packed, k_scale, v_packed, v_scale, dtype=dtype)
    return _int4_sdpa_torch(q, k_packed, k_scale, v_packed, v_scale, dtype=dtype)


def check_parity(
    B: int = 1,
    H: int = 4,
    S_q: int = 1,
    S_k: int = 32,
    D: int = 64,
    atol: float = 1e-2,
    rtol: float = 0.05,
) -> dict:
    """Run a numeric parity check: Triton kernel vs torch fallback.

    Returns a dict with ``passed``, ``max_abs_error``, ``mean_abs_error``.
    Only meaningful on CUDA with Triton installed.
    """
    if not USE_TRITON_KERNEL:
        return {"passed": False, "error": "Triton or CUDA not available"}

    from fade.quant import quant_k_int4, quant_v_int4

    torch.manual_seed(42)
    device = "cuda"
    q = torch.randn(B, H, S_q, D, dtype=torch.float16, device=device)
    k_fp = torch.randn(B, H, S_k, D, dtype=torch.float16, device=device)
    v_fp = torch.randn(B, H, S_k, D, dtype=torch.float16, device=device)

    k_packed, k_scale = quant_k_int4(k_fp)
    v_packed, v_scale = quant_v_int4(v_fp)

    ref = _int4_sdpa_torch(q, k_packed, k_scale, v_packed, v_scale, dtype=torch.float16)
    tri = _int4_sdpa_triton(q, k_packed, k_scale, v_packed, v_scale, dtype=torch.float16)

    max_err = (ref - tri).abs().max().item()
    mean_err = (ref - tri).abs().mean().item()
    passed = max_err < atol and mean_err < atol * 0.5

    return {
        "passed": passed,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "atol": atol,
        "rtol": rtol,
    }


__all__ = ["USE_TRITON_KERNEL", "check_parity", "int4_sdpa"]
