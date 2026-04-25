"""Model integration helpers.

What lives here:
    * ``load_model`` — load a causal LM with a chosen ``attn_implementation``.
      Supports ``"eager"``, ``"sdpa"``, ``"flash_attention_2"``, or ``"auto"``
      (picks ``eager`` if ``need_attentions=True``, else ``sdpa``).
    * ``load_model_eager`` — thin backward-compatible shim around ``load_model``.
    * ``create_tiered_cache`` — factory that reads RoPE params from the model
      config and builds a ``TieredKVCache``.
    * ``forward_with_tracking`` — one forward pass; pipes attention weights
      into an ``AttentionTracker`` when one is supplied, emitting a clear
      warning if the loaded attention impl silently dropped the attentions.

Attention-impl guide:
    * ``"eager"`` — returns real attention tensors. Required for the H2O
      policy (``reassign_tiers_h2o``) because that path needs the full
      prefill attention matrix.
    * ``"sdpa"`` — default. Fast, uses ``torch.nn.functional.scaled_dot_
      product_attention``. ``output_attentions=True`` is silently ignored
      by most SDPA kernels — don't use for H2O.
    * ``"flash_attention_2"`` — fastest for long contexts; never returns
      attention weights. Safe for the ``position`` and ``ema``-decode
      policies.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fade.cache import TieredKVCache
from fade.config import FadeConfig
from fade.rope import extract_rope_scheme
from fade.tracker import AttentionTracker

AttnImpl = Literal["auto", "eager", "sdpa", "flash_attention_2"]


def load_model(
    model_id: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    attn_impl: AttnImpl = "auto",
    need_attentions: bool = False,
):
    """Load a causal LM with the requested attention implementation.

    Args:
        model_id: HuggingFace model id.
        device_map: passed through to ``from_pretrained``.
        dtype: torch dtype for model weights.
        attn_impl: ``"auto"`` | ``"eager"`` | ``"sdpa"`` | ``"flash_attention_2"``.
            ``"auto"`` picks ``"eager"`` when ``need_attentions=True`` and
            ``"sdpa"`` otherwise.
        need_attentions: hint that the H2O policy (or any code relying on
            ``output_attentions=True``) will be used. Only consulted when
            ``attn_impl="auto"``.

    Returns:
        ``(model, tokenizer)`` with ``model.eval()`` already called.
    """
    if attn_impl == "auto":
        attn_impl = "eager" if need_attentions else "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # transformers >= 5.6 renamed torch_dtype → dtype; try new name first.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
    model.eval()
    return model, tokenizer


def load_model_eager(
    model_id: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
):
    """Backward-compatible shim that loads a model with eager attention.

    Prefer ``load_model(..., attn_impl="eager")`` in new code.
    """
    return load_model(model_id, device_map=device_map, dtype=dtype, attn_impl="eager")


def _installed_attn_impl(model) -> str | None:
    """Return the attention implementation the model was loaded with, if known."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    # 5.x stores it as config._attn_implementation; 4.x may store it as
    # config.attn_implementation. Try both.
    return getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)


def _extract_rope_theta(cfg) -> float:
    """Extract RoPE theta from a HF model config.

    Handles both the legacy ``config.rope_theta`` attribute and the newer
    ``config.rope_parameters["rope_theta"]`` layout (transformers >= 5.x).
    """
    theta = getattr(cfg, "rope_theta", None)
    if theta is not None:
        return float(theta)
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return 10000.0


def create_tiered_cache(
    model,
    dtype: torch.dtype = torch.float16,
    config: FadeConfig | None = None,
    **kwargs,
) -> TieredKVCache:
    """Create a ``TieredKVCache`` pre-configured with the model's RoPE params.

    Extracts ``rope_theta`` and ``head_dim`` from the model config so the
    cache can compute cos/sin for the Phase 2 pre-RoPE / re-RoPE path.

    Args:
        model: the causal LM.
        dtype: cache dtype (default float16).
        config: optional ``FadeConfig`` providing cache knobs. If given,
            individual ``kwargs`` override fields from the config.
        **kwargs: forwarded to ``TieredKVCache`` (e.g. ``n_sink``,
            ``recent_window``, ``int4_budget``, ``int2_budget``, ``batch_size``).
    """
    cfg = model.config
    # Some multimodal models nest the text config.
    text_cfg = getattr(cfg, "text_config", cfg)
    model_type = getattr(text_cfg, "model_type", "")

    # D3: Pure-recurrent models (Mamba, RWKV, xLSTM) have no K/V cache.
    _RECURRENT_TYPES = {"mamba", "mamba2", "rwkv", "rwkv5", "rwkv6", "xlstm"}
    if model_type in _RECURRENT_TYPES:
        warnings.warn(
            f"Model type {model_type!r} is a recurrent architecture with no "
            f"standard K/V cache. Returning a plain DynamicCache. "
            f"FADE tier compression is not applicable.",
            RuntimeWarning,
            stacklevel=2,
        )
        from fade._compat import DynamicCache

        return DynamicCache()  # type: ignore[return-value]

    # D2: DeepSeek MLA stores already-compressed latent K/V.
    _DEEPSEEK_TYPES = {"deepseek", "deepseek_r1", "deepseek_r2", "deepseek_v2", "deepseek_v3"}
    _is_deepseek_mla = model_type in _DEEPSEEK_TYPES

    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    scheme = extract_rope_scheme(text_cfg, head_dim=head_dim)
    cache_kwargs: dict = {}
    if config is not None:
        cache_kwargs.update(config.to_cache_kwargs())
    cache_kwargs.update(kwargs)
    cache = TieredKVCache(
        dtype=dtype,
        rope_theta=scheme.theta,
        head_dim=head_dim,
        rope_scheme=scheme,
        **cache_kwargs,
    )

    # Detect hybrid attention models (e.g. Qwen 3.5/3.6 with DeltaNet).
    # Layers marked as "linear_attention" use a recurrent state, not K/V
    # cache — FADE skips tier management on those layers.
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        skip = {
            i
            for i, lt in enumerate(layer_types)
            if lt not in ("full_attention", "sliding_attention")
        }
        if skip:
            cache.set_skip_layers(skip)

    # D2: DeepSeek MLA — skip all layers (KV is already latent-compressed).
    if _is_deepseek_mla:
        warnings.warn(
            f"Model type {model_type!r} uses Multi-head Latent Attention. "
            f"K/V are already compressed latent vectors. FADE tier management "
            f"is disabled (all layers skipped). The cache still works as a "
            f"DynamicCache for normal inference.",
            RuntimeWarning,
            stacklevel=2,
        )
        num_layers = getattr(text_cfg, "num_hidden_layers", 0)
        cache.set_skip_layers(set(range(num_layers)))

    return cache


_WARNED_ABOUT_MISSING_ATTENTIONS: set[int] = set()


def forward_with_tracking(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    tracker: AttentionTracker | None = None,
    attention_mask: torch.Tensor | None = None,
):
    """Run one forward pass, feeding attention weights into ``tracker`` if given.

    When ``tracker`` is ``None`` no attention output is requested — works with
    any attention implementation including Flash Attention 2.

    When ``tracker`` is provided, the model must have been loaded with an
    attention implementation that returns real attention tensors (``"eager"``;
    SDPA silently drops them in most versions). If the model returns
    ``attentions=None`` we emit a ``RuntimeWarning`` (once per model) so the
    caller knows H2O / EMA scoring will degrade to position-only behavior.
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=tracker is not None,
            return_dict=True,
        )
    if tracker is not None:
        attentions = getattr(out, "attentions", None)
        if attentions is None or any(a is None for a in attentions):
            impl = _installed_attn_impl(model) or "<unknown>"
            model_key = id(model)
            # Warn once per model to keep long runs from spamming logs.
            if model_key not in _WARNED_ABOUT_MISSING_ATTENTIONS:
                _WARNED_ABOUT_MISSING_ATTENTIONS.add(model_key)
                warnings.warn(
                    f"AttentionTracker was provided but the model "
                    f"(attn_implementation={impl!r}) did not return attention "
                    f"tensors. The tracker will receive no observations and "
                    f"H2O / EMA policies will fall back to position-only "
                    f"behavior. Reload with attn_impl='eager' to enable "
                    f"attention-aware eviction.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return out
        for layer_idx, attn in enumerate(attentions):
            tracker.observe(attn, layer_idx)
    return out


__all__ = [
    "AttnImpl",
    "create_tiered_cache",
    "forward_with_tracking",
    "load_model",
    "load_model_eager",
]
