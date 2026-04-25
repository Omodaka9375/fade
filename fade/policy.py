"""Tier assignment policies.

Three eviction strategies, all sharing the same ``_assign_one_layer`` core:

  1. **EMA tracker** (``reassign_tiers``): per-position exponential-moving-average
     of attention mass accumulated during decode. The original Phase 2 approach.
  2. **H2O heavy-hitter oracle** (``reassign_tiers_h2o``): one-shot cumulative
     attention mass from the *prefill* pass. Much better scores than decode-only
     EMA because the full causal attention matrix is available.
  3. **Position / FIFO** (``reassign_tiers_by_position``): evict the oldest
     non-sink, non-recent tokens. No scores required.

All three honour ``n_sink``, ``recent_window``, ``int4_budget``, ``int2_budget``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fade.cache import TIER_EVICT, TIER_FP16, TIER_INT2, TIER_INT4, TieredKVCache
from fade.tracker import AttentionTracker

# --- knobs ------------------------------------------------------------------- #
REASSIGN_EVERY_N_TOKENS: int = 64


def reassign_tiers(
    cache: TieredKVCache,
    tracker: AttentionTracker,
    num_layers: int,
) -> None:
    """Compute new tier assignments per layer and apply them to ``cache``.

    Also compacts the tracker's score buffers so they stay aligned with the
    cache after any eviction.
    """
    for layer_idx in range(num_layers):
        if not cache.is_managed(layer_idx):
            continue
        scores = tracker.scores(layer_idx)
        if scores is None:
            continue
        tiers = _assign_one_layer(
            S=int(scores.shape[0]),
            scores=scores,
            n_sink=cache.n_sink,
            recent_window=cache.recent_window,
            int4_budget=cache.int4_budget,
            int2_budget=cache.int2_budget,
        )
        cache.apply_tier_assignment(layer_idx, tiers, scores=scores)
        # Keep tracker entries for every surviving tier.
        keep_mask = (tiers == TIER_FP16) | (tiers == TIER_INT4) | (tiers == TIER_INT2)
        tracker.remove_positions(layer_idx, keep_mask)


def reassign_tiers_h2o(
    cache: TieredKVCache,
    prefill_attentions: list[Tensor],
    num_layers: int,
    tracker: AttentionTracker | None = None,
) -> None:
    """H2O heavy-hitter oracle: use full prefill attention for scoring.

    ``prefill_attentions`` is the list of per-layer attention weight tensors
    ``[B, H, Q, K]`` returned by the model during the prefill forward pass
    (``output.attentions``). The cumulative mass over all query positions gives
    a much richer signal than the decode-only EMA.

    If a ``tracker`` is provided its decode-step scores are blended in via
    geometric mean so that tokens receiving attention during *both* prefill
    and decode are favoured.
    """
    for layer_idx in range(num_layers):
        if not cache.is_managed(layer_idx):
            continue
        S = int(cache.get_seq_length(layer_idx))
        if S == 0:
            continue

        # --- build scores from prefill attention ---
        if layer_idx < len(prefill_attentions) and prefill_attentions[layer_idx] is not None:
            attn = prefill_attentions[layer_idx]  # [B, H, Q, K]
            prefill_mass = attn.detach().float().sum(dim=(0, 1, 2))  # [K_prefill]
        else:
            state = cache._layers[layer_idx]
            ref = state.fp16_k if state.fp16_k is not None else state.int4_kq
            device = ref.device if ref is not None else torch.device("cpu")
            prefill_mass = torch.zeros(S, device=device)

        # Pad / trim to current cache length (decode tokens added after prefill).
        if prefill_mass.shape[0] < S:
            pad = torch.zeros(
                S - prefill_mass.shape[0], dtype=prefill_mass.dtype, device=prefill_mass.device
            )
            prefill_mass = torch.cat([prefill_mass, pad])
        elif prefill_mass.shape[0] > S:
            prefill_mass = prefill_mass[:S]

        # Optional: blend with decode-time tracker scores.
        if tracker is not None:
            decode_scores = tracker.scores(layer_idx)
            if decode_scores is not None:
                ds = decode_scores[:S].to(prefill_mass.device).float()
                # Geometric mean: sqrt(prefill * decode). Avoids one signal
                # dominating; tokens must be important in BOTH passes.
                prefill_mass = (prefill_mass * ds.clamp(min=1e-12)).sqrt()

        tiers = _assign_one_layer(
            S=S,
            scores=prefill_mass,
            n_sink=cache.n_sink,
            recent_window=cache.recent_window,
            int4_budget=cache.int4_budget,
            int2_budget=cache.int2_budget,
        )
        cache.apply_tier_assignment(layer_idx, tiers, scores=prefill_mass)
        keep_mask = (tiers == TIER_FP16) | (tiers == TIER_INT4) | (tiers == TIER_INT2)
        if tracker is not None:
            tracker.remove_positions(layer_idx, keep_mask)
        # Trim prefill_attentions so subsequent calls stay aligned.
        prefill_attentions[layer_idx] = None  # free memory


def reassign_tiers_by_position(
    cache: TieredKVCache,
    num_layers: int,
) -> None:
    """Position-only (FIFO) tier assignment; no attention tracking required.

    Middle tokens are ranked by their position — the *newest* middle tokens
    survive in INT4 / INT2 and the oldest are evicted. This is the simplest
    eviction strategy and avoids any dependency on ``output_attentions``.

    Also works as the Phase 1-A path (``int4_budget=None``): every middle
    token goes to INT4 and nothing is evicted.
    """
    for layer_idx in range(num_layers):
        if not cache.is_managed(layer_idx):
            continue
        S = int(cache.get_seq_length(layer_idx))
        if S == 0:
            continue
        state = cache._layers[layer_idx]
        ref = state.fp16_k if state.fp16_k is not None else state.int4_kq
        device = ref.device if ref is not None else torch.device("cpu")
        # Use position as score: higher position = more recent = higher score.
        # _assign_one_layer keeps the top-k by score, so newest survive.
        position_scores = torch.arange(S, dtype=torch.float32, device=device)
        tiers = _assign_one_layer(
            S=S,
            scores=position_scores,
            n_sink=cache.n_sink,
            recent_window=cache.recent_window,
            int4_budget=cache.int4_budget,
            int2_budget=cache.int2_budget,
        )
        cache.apply_tier_assignment(layer_idx, tiers)


def reassign_tiers_adaptive(
    cache: TieredKVCache,
    tracker: AttentionTracker,
    num_layers: int,
    high_pct: float = 0.5,
) -> None:
    """Attention-aware adaptive bit allocation.

    Splits middle tokens into tiers by attention score:
    - Top ``high_pct`` of middle tokens → INT4 (best quality)
    - Remaining middle tokens → INT2 (more compression)
    - Tokens below ``int4_budget + int2_budget`` → evicted

    This is FADE's native version of KVTC's DP bit allocation:
    tokens that matter get more bits, tokens that don't get fewer.

    Args:
        cache: the tiered KV cache.
        tracker: attention mass tracker with per-layer scores.
        num_layers: number of transformer layers.
        high_pct: fraction of middle tokens that get INT4 (rest get INT2).
    """
    for layer_idx in range(num_layers):
        if not cache.is_managed(layer_idx):
            continue
        scores = tracker.scores(layer_idx)
        if scores is None:
            continue

        S = int(scores.shape[0])
        tiers = torch.full((S,), TIER_EVICT, dtype=torch.long, device=scores.device)

        # Sinks + recent window → FP16.
        sink_end = min(cache.n_sink, S)
        tiers[:sink_end] = TIER_FP16
        recent_start = max(sink_end, S - cache.recent_window)
        tiers[recent_start:] = TIER_FP16

        # Middle tokens: split by score.
        middle_idx = (tiers == TIER_EVICT).nonzero(as_tuple=False).squeeze(-1)
        if middle_idx.numel() == 0:
            cache.apply_tier_assignment(layer_idx, tiers, scores=scores)
            keep_mask = tiers != TIER_EVICT
            tracker.remove_positions(layer_idx, keep_mask)
            continue

        middle_scores = scores[middle_idx]
        n_middle = middle_idx.numel()

        # Compute budgets.
        total_budget = n_middle
        if cache.int4_budget is not None:
            total_budget = min(total_budget, cache.int4_budget + cache.int2_budget)

        n_int4 = min(int(total_budget * high_pct), n_middle)
        n_int2 = min(total_budget - n_int4, n_middle - n_int4)

        if n_int4 + n_int2 > 0:
            # Top-k by score get INT4, next-k get INT2.
            sorted_idx = middle_scores.argsort(descending=True)
            if n_int4 > 0:
                tiers[middle_idx[sorted_idx[:n_int4]]] = TIER_INT4
            if n_int2 > 0:
                tiers[middle_idx[sorted_idx[n_int4 : n_int4 + n_int2]]] = TIER_INT2

        cache.apply_tier_assignment(layer_idx, tiers, scores=scores)
        keep_mask = (tiers == TIER_FP16) | (tiers == TIER_INT4) | (tiers == TIER_INT2)
        tracker.remove_positions(layer_idx, keep_mask)


def _assign_one_layer(
    S: int,
    scores: Tensor,
    n_sink: int,
    recent_window: int,
    int4_budget: int | None,
    int2_budget: int,
) -> Tensor:
    """Return a [S] LongTensor of TIER_* values aligned with ascending position.

    ``int4_budget=None`` means unlimited: every middle token goes to INT4 and
    nothing is evicted. This is the safe Phase 1 default.
    """
    tiers = torch.full((S,), TIER_EVICT, dtype=torch.long, device=scores.device)

    # --- mandatory FP16: sinks + recent window ---
    sink_end = min(n_sink, S)
    tiers[:sink_end] = TIER_FP16
    recent_start = max(sink_end, S - recent_window)
    tiers[recent_start:] = TIER_FP16

    # --- competitive tiers over the "middle" --- #
    middle_idx = (tiers == TIER_EVICT).nonzero(as_tuple=False).squeeze(-1)
    if middle_idx.numel() == 0:
        return tiers

    # Unlimited budget: compress every middle token. No eviction.
    if int4_budget is None or int4_budget >= int(middle_idx.numel()):
        tiers[middle_idx] = TIER_INT4
        return tiers

    middle_scores = scores[middle_idx]
    top4 = middle_scores.topk(int4_budget).indices
    tiers[middle_idx[top4]] = TIER_INT4

    remaining_idx = (tiers == TIER_EVICT).nonzero(as_tuple=False).squeeze(-1)
    if remaining_idx.numel() > 0 and int2_budget > 0:
        remaining_scores = scores[remaining_idx]
        k2 = min(int2_budget, remaining_idx.numel())
        top2 = remaining_scores.topk(k2).indices
        tiers[remaining_idx[top2]] = TIER_INT2

    return tiers
