"""Tiered-cache experiment, compared side-by-side with the FP16 baseline.

Greedy decoding, manual loop so we can schedule tier-reassignment calls between
tokens and (optionally) feed an attention tracker when budgets require scores.

Knobs (all at the top of the file):
    - PHASE: "1a" (no eviction) or "2" (budget-bounded eviction + re-RoPE)
    - PROMPT_MODE: "repetitive" or "diverse"
    - FILLER_REPEATS / MAX_NEW_TOKENS: context length controls
    - N_SINK, RECENT_WINDOW: FP16 tier sizing
    - INT4_BUDGET, INT2_BUDGET: middle tier budgets (Phase 2 only)
    - REASSIGN_EVERY: tier-policy period (higher = more tracker observations
      before first eviction, but less frequent compression)
    - PREFILL_TRACK_LIMIT: max prompt tokens for which output_attentions=True
      is safe during prefill. Above this limit tracking starts on decode only.
    - RUN_PPL, RUN_NEEDLE: enable perplexity / needle-in-a-haystack evals
"""
from __future__ import annotations

import gc
import time

import torch
from transformers import DynamicCache

from fade.eval.memory import PeakMemory, cache_storage_bytes
from fade.eval.needle import run_needle
from fade.eval.perplexity import perplexity
from fade.patch import create_tiered_cache, forward_with_tracking, load_model
from fade.policy import reassign_tiers, reassign_tiers_by_position, reassign_tiers_h2o
from fade.tracker import AttentionTracker

# --- configuration (all knobs at the top) ------------------------------------ #
MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"  # swap to 0.5B/1.5B if OOM
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

# --- prompt ------------------------------------------------------------------ #
# "diverse"      = distinct paragraphs on varied topics (summarization)
# "repetitive"   = same paragraph repeated (stress test)
# "chat"         = multi-turn conversation (recent context dominant — eviction-friendly)
# "story"        = narrative continuation (long-range generation — eviction-friendly)
PROMPT_MODE: str = "chat"

_REPETITIVE_FILLER: str = (
    "The history of caching in computer systems spans several decades. "
    "Early mainframes used small buffers to avoid slow core memory accesses. "
    "Modern CPUs organize caches hierarchically, with L1, L2, and L3 levels. "
    "Language models reuse this idea when they keep key-value tensors across "
    "generation steps, avoiding redundant attention computation. "
)

_DIVERSE_PARAGRAPHS: str = (
    "Photosynthesis converts sunlight into chemical energy. Chloroplasts in "
    "plant cells capture photons, splitting water molecules to release oxygen. "
    "The Calvin cycle then fixes carbon dioxide into glucose, which powers "
    "cellular metabolism. This process sustains nearly all life on Earth.\n\n"
    "The Roman Empire at its height stretched from Britain to Mesopotamia. "
    "Its road network, spanning over 80,000 kilometers, enabled rapid troop "
    "movement and trade. Latin, the empire's lingua franca, evolved into the "
    "Romance languages still spoken by hundreds of millions today.\n\n"
    "Quantum computers exploit superposition and entanglement to process "
    "information. Unlike classical bits, qubits can represent 0 and 1 "
    "simultaneously. Error correction remains the central challenge; current "
    "systems require thousands of physical qubits per logical qubit.\n\n"
    "The Mariana Trench descends nearly 11,000 meters below sea level. "
    "Pressure at the bottom reaches over 1,000 atmospheres. Despite this, "
    "organisms such as amphipods and xenophyophores thrive in the hadal zone, "
    "feeding on organic matter that sinks from the surface.\n\n"
    "Jazz emerged in New Orleans in the early twentieth century, blending "
    "African rhythmic traditions with European harmonic structures. Improvisation "
    "became its hallmark; musicians like Louis Armstrong and Duke Ellington "
    "shaped an art form that influenced rock, hip-hop, and electronic music.\n\n"
    "The human immune system deploys innate and adaptive defenses. Innate "
    "immunity provides rapid, non-specific responses through barriers and "
    "phagocytes. Adaptive immunity, mediated by T and B lymphocytes, learns "
    "to recognize specific pathogens and forms long-lasting memory.\n\n"
    "Continental drift, proposed by Wegener in 1912, was initially dismissed. "
    "Plate tectonics, confirmed in the 1960s by seafloor spreading evidence, "
    "explains earthquakes, volcanism, and mountain formation through the "
    "movement of lithospheric plates over the asthenosphere.\n\n"
    "Caching in computer systems spans several decades of innovation. Early "
    "mainframes used small buffers to avoid slow core memory accesses. Modern "
    "CPUs organize caches hierarchically, with L1, L2, and L3 levels. Language "
    "models reuse this idea when they keep key-value tensors across generation "
    "steps, avoiding redundant attention computation.\n\n"
)

_CHAT_PROMPT: str = (
    "<|im_start|>system\nYou are a helpful assistant. Answer clearly and concisely.<|im_end|>\n"
    "<|im_start|>user\nWhat is photosynthesis and why is it important?<|im_end|>\n"
    "<|im_start|>assistant\nPhotosynthesis is the process by which green plants, "
    "algae, and certain bacteria convert sunlight into chemical energy. Inside "
    "chloroplasts, light-dependent reactions split water molecules, releasing "
    "oxygen as a byproduct and producing ATP and NADPH. The Calvin cycle then "
    "uses these energy carriers to fix carbon dioxide from the atmosphere into "
    "glucose. This glucose serves as the primary energy source for the plant "
    "and, through food chains, for nearly all life on Earth. Photosynthesis "
    "is also responsible for producing the oxygen we breathe and for removing "
    "CO2 from the atmosphere, making it central to both life and climate "
    "regulation.<|im_end|>\n"
    "<|im_start|>user\nHow does the human immune system defend against pathogens?<|im_end|>\n"
    "<|im_start|>assistant\nThe human immune system operates in two layers. "
    "Innate immunity is the first line of defense: physical barriers like skin "
    "and mucous membranes block entry, while phagocytes such as neutrophils "
    "and macrophages engulf and destroy invaders non-specifically. If a pathogen "
    "breaches these defenses, adaptive immunity activates. Helper T cells "
    "coordinate the response, cytotoxic T cells destroy infected cells, and "
    "B cells produce antibodies that neutralize specific pathogens. After the "
    "infection clears, memory T and B cells remain, enabling a faster and "
    "stronger response upon re-exposure. This is the principle behind "
    "vaccination.<|im_end|>\n"
    "<|im_start|>user\nTell me about the Roman Empire's road network and its legacy.<|im_end|>\n"
    "<|im_start|>assistant\nAt its peak, the Roman road network comprised over "
    "80,000 kilometers of paved highways connecting provinces from Britain to "
    "Mesopotamia. Roads were built in layers: a foundation of large stones, "
    "followed by gravel and sand, topped with fitted paving stones cambered for "
    "drainage. Milestones marked distances, and way-stations provided rest for "
    "travelers. This infrastructure enabled rapid troop deployment, facilitated "
    "trade across the Mediterranean world, and spread Latin culture throughout "
    "Europe. Many modern European highways still follow Roman road alignments, "
    "and the engineering principles they pioneered remain in use today.<|im_end|>\n"
    "<|im_start|>user\nExplain how quantum computers differ from classical computers.<|im_end|>\n"
    "<|im_start|>assistant\nClassical computers use bits representing 0 or 1. "
    "Quantum computers use qubits, which exploit superposition to represent both "
    "states simultaneously. Entanglement links qubits so that measuring one "
    "instantly determines the state of its partner, regardless of distance. These "
    "properties allow quantum computers to explore many solutions in parallel, "
    "making them potentially faster for optimization, cryptography, and molecular "
    "simulation. However, qubits are fragile: environmental noise causes "
    "decoherence, and current systems require elaborate error correction with "
    "thousands of physical qubits per logical qubit. Practical, fault-tolerant "
    "quantum computing remains an active area of research.<|im_end|>\n"
    "<|im_start|>user\nDescribe what lives at the deepest point of the ocean.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_STORY_PROMPT: str = (
    "Continue this story in vivid detail:\n\n"
    "The lighthouse keeper climbed the spiral staircase for the last time. "
    "After forty years of tending the beacon on Cape Desolation, Marcus knew "
    "every crack in the stone walls, every groan of the iron railing. Tonight "
    "the sea was angrier than he had ever seen it — waves crashing against the "
    "rocks with enough force to send spray over the gallery rail sixty meters "
    "above the waterline. The radio had gone silent two hours ago, and the "
    "backup generator was running on fumes.\n\n"
    "He reached the lantern room and checked the Fresnel lens. The massive "
    "glass prism, assembled in Paris in 1887, still rotated smoothly on its "
    "mercury bath. Its beam cut through the storm like a blade, sweeping the "
    "horizon every twelve seconds. But Marcus was not watching the sea tonight. "
    "He was watching the sky.\n\n"
    "Three days ago, the astronomers at Paranal had detected something — an "
    "object that did not match any catalogued asteroid or satellite. It was "
    "decelerating. Objects in space do not decelerate unless something is "
    "acting on them. The news had spread through classified channels first, "
    "then leaked to social media, and now the world was holding its breath.\n\n"
    "Marcus trained his binoculars upward through a gap in the clouds. There "
    "it was — a pale green light, steady and unwavering, growing brighter by "
    "the minute. He set down the binoculars and picked up his pen.\n\n"
)

FILLER_REPEATS: int = 6    # repeats for diverse/repetitive modes
if PROMPT_MODE == "chat":
    PROMPT: str = _CHAT_PROMPT
elif PROMPT_MODE == "story":
    PROMPT: str = _STORY_PROMPT
elif PROMPT_MODE == "diverse":
    BASE_PROMPT: str = "Summarize the following text in one short paragraph.\n\n"
    PROMPT: str = BASE_PROMPT + (_DIVERSE_PARAGRAPHS * FILLER_REPEATS) + "\n\nSummary:"
else:
    BASE_PROMPT: str = "Summarize the following text in one short paragraph.\n\n"
    PROMPT: str = BASE_PROMPT + (_REPETITIVE_FILLER * (FILLER_REPEATS * 8)) + "\n\nSummary:"

MAX_NEW_TOKENS: int = 256

# --- batching --------------------------------------------------------------- #
# Replicate the prompt across the batch dimension. Useful for throughput
# measurements and for validating the shared-tier batching contract. When
# BATCH_SIZE > 1, greedy output is identical on every row and tiered KV-cache
# memory scales ~linearly with B.
BATCH_SIZE: int = 1

# --- cache knobs ------------------------------------------------------------- #
N_SINK: int = 4
RECENT_WINDOW: int = 64
# Higher values give the tracker more observations before the first eviction,
# improving score quality at the cost of less frequent compression.
REASSIGN_EVERY: int = 64

# --- Phase selector ---------------------------------------------------------- #
PHASE: str = "2"

# --- eviction policy --------------------------------------------------------- #
# "h2o"      = H2O heavy-hitter oracle (prefill attention mass, best quality)
# "ema"      = decode-only EMA tracker (original approach)
# "position" = FIFO, evict oldest middle tokens (no attention scores needed)
EVICTION_POLICY: str = "h2o"

if PHASE == "2":
    INT4_BUDGET: int | None = 400   # ~400 middle tokens in INT4
    INT2_BUDGET: int = 0               # INT2 disabled (too lossy for small models)
    TRACK_ATTENTION: bool = EVICTION_POLICY in ("h2o", "ema")
else:  # Phase 1-A
    INT4_BUDGET: int | None = None
    INT2_BUDGET: int = 0
    TRACK_ATTENTION: bool = False

# --- prefill tracking -------------------------------------------------------- #
# output_attentions=True during prefill allocates O(S^2 * H * L) memory.
# Below this limit we capture prefill attentions for H2O scoring; above it
# we fall back to decode-only tracking or position-based eviction.
PREFILL_TRACK_LIMIT: int = 1024

# --- evaluation -------------------------------------------------------------- #
RUN_PPL: bool = True
PPL_MAX_LENGTH: int = 1024
PPL_STRIDE: int = 512
# Set to None to use built-in diverse text; set to "wikitext" to download
# WikiText-2 from HuggingFace datasets (requires `datasets` package).
PPL_DATASET: str | None = None
RUN_NEEDLE: bool = True
NEEDLE_TARGET_TOKENS: int = 1024


@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int,
    tracker: AttentionTracker | None = None,
    reassign_every: int | None = None,
    num_layers: int = 0,
    use_tiered_policy: bool = False,
    eviction_policy: str = "h2o",
) -> torch.Tensor:
    """Greedy decode one token at a time. Returns [1, generated_len] token ids."""
    prefill_len = int(input_ids.shape[-1])
    need_prefill_attn = (
        use_tiered_policy
        and eviction_policy == "h2o"
        and prefill_len <= PREFILL_TRACK_LIMIT
    )
    # For H2O we need the raw prefill attention tensors (not just tracker EMA).
    # For EMA we just feed the tracker. For position we skip entirely.
    prefill_tracker = tracker if (tracker is not None and prefill_len <= PREFILL_TRACK_LIMIT) else None
    if need_prefill_attn:
        # Run prefill WITH output_attentions to capture the full attention matrix.
        out = forward_with_tracking(model, input_ids, past_key_values, tracker=prefill_tracker)
        prefill_attentions = list(out.attentions) if out.attentions is not None else []
    else:
        out = forward_with_tracking(model, input_ids, past_key_values, tracker=prefill_tracker)
        prefill_attentions = []

    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    generated: list[torch.Tensor] = [next_token]

    for step in range(max_new_tokens - 1):
        out = forward_with_tracking(model, next_token, past_key_values, tracker=tracker)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token)

        if reassign_every and use_tiered_policy and (step + 1) % reassign_every == 0:
            if eviction_policy == "h2o" and prefill_attentions:
                reassign_tiers_h2o(past_key_values, prefill_attentions,
                                   num_layers, tracker=tracker)
            elif eviction_policy == "position":
                reassign_tiers_by_position(past_key_values, num_layers)
            elif tracker is not None:
                reassign_tiers(past_key_values, tracker, num_layers)
            else:
                reassign_tiers_by_position(past_key_values, num_layers)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return torch.cat(generated, dim=-1)


def main() -> None:
    print(f"device={DEVICE}  dtype={DTYPE}  model={MODEL_ID}")
    # Eager attention is only needed when the H2O policy has to see the full
    # prefill attention matrix. Other policies (position / EMA-decode) run
    # fine under SDPA / Flash Attention 2.
    needs_eager = EVICTION_POLICY == "h2o" or TRACK_ATTENTION
    model, tokenizer = load_model(
        MODEL_ID, device_map=DEVICE, dtype=DTYPE,
        attn_impl="eager" if needs_eager else "sdpa",
    )
    num_layers: int = model.config.num_hidden_layers

    enc = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    if BATCH_SIZE > 1:
        enc["input_ids"] = enc.input_ids.expand(BATCH_SIZE, -1).contiguous()
        if enc.get("attention_mask") is not None:
            enc["attention_mask"] = enc.attention_mask.expand(BATCH_SIZE, -1).contiguous()
    print(f"prompt_tokens={enc.input_ids.shape[1]}  batch_size={enc.input_ids.shape[0]}")

    # --------------------------- BASELINE --------------------------- #
    print("\n=== BASELINE (plain DynamicCache, FP16 everywhere) ===")
    baseline_cache = DynamicCache()
    with PeakMemory(DEVICE) as mem_base:
        t0 = time.perf_counter()
        baseline_out = greedy_generate(
            model, tokenizer, enc.input_ids, baseline_cache,
            max_new_tokens=MAX_NEW_TOKENS,
            tracker=None, reassign_every=None, num_layers=num_layers,
            use_tiered_policy=False,
        )
        t_base = time.perf_counter() - t0
    base_cache_mib = cache_storage_bytes(baseline_cache) / (1024 * 1024)
    print(f"generated={baseline_out.shape[1]}  elapsed={t_base:.2f}s  "
          f"peak={mem_base.peak_mib:.1f} MiB  "
          f"tps={baseline_out.shape[1] / t_base:.2f}  "
          f"kv_cache={base_cache_mib:.1f} MiB")
    print(tokenizer.decode(baseline_out[0], skip_special_tokens=True))

    # Drop the baseline cache before the tiered run so the allocator can reclaim.
    del baseline_cache
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --------------------------- TIERED ----------------------------- #
    if PHASE == "2":
        label = (f"TIERED Phase 2 ({EVICTION_POLICY}, sinks+recent FP16, "
                 f"INT4_BUDGET={INT4_BUDGET}, INT2_BUDGET={INT2_BUDGET})")
    elif TRACK_ATTENTION:
        label = "TIERED (attention-aware, sinks+recent FP16, middle INT4)"
    else:
        label = "TIERED (position-only, sinks+recent FP16, middle INT4)"
    print(f"\n=== {label} ===")
    tiered_cache = create_tiered_cache(
        model,
        dtype=DTYPE,
        n_sink=N_SINK,
        recent_window=RECENT_WINDOW,
        int4_budget=INT4_BUDGET,
        int2_budget=INT2_BUDGET,
    )
    tracker = AttentionTracker(num_layers=num_layers) if TRACK_ATTENTION else None
    with PeakMemory(DEVICE) as mem_tier:
        t0 = time.perf_counter()
        tiered_out = greedy_generate(
            model, tokenizer, enc.input_ids, tiered_cache,
            max_new_tokens=MAX_NEW_TOKENS,
            tracker=tracker, reassign_every=REASSIGN_EVERY, num_layers=num_layers,
            use_tiered_policy=True, eviction_policy=EVICTION_POLICY,
        )
        t_tier = time.perf_counter() - t0
    tier_cache_mib = cache_storage_bytes(tiered_cache) / (1024 * 1024)
    print(f"generated={tiered_out.shape[1]}  elapsed={t_tier:.2f}s  "
          f"peak={mem_tier.peak_mib:.1f} MiB  "
          f"tps={tiered_out.shape[1] / t_tier:.2f}  "
          f"kv_cache={tier_cache_mib:.1f} MiB")
    print(tokenizer.decode(tiered_out[0], skip_special_tokens=True))

    # --------------------------- COMPARE ---------------------------- #
    print("\n=== DELTA ===")
    min_len = min(baseline_out.shape[1], tiered_out.shape[1])
    match_rate = (
        (baseline_out[0, :min_len] == tiered_out[0, :min_len]).float().mean().item()
    )
    print(f"greedy_token_match_rate_over_{min_len}_tokens = {match_rate:.2%}")
    if mem_base.peak_mib > 0:
        print(f"peak_memory_delta = {mem_tier.peak_mib - mem_base.peak_mib:+.1f} MiB "
              f"({100.0 * (mem_tier.peak_mib / mem_base.peak_mib - 1.0):+.1f}%)")
    if base_cache_mib > 0:
        print(f"kv_cache_delta   = {tier_cache_mib - base_cache_mib:+.2f} MiB "
              f"({100.0 * (tier_cache_mib / base_cache_mib - 1.0):+.1f}%)")
    print(f"tps_delta = {tiered_out.shape[1] / t_tier - baseline_out.shape[1] / t_base:+.2f} tok/s")

    # --------------------------- PERPLEXITY -------------------------- #
    if RUN_PPL:
        print("\n=== PERPLEXITY ===")
        ppl_text = _DIVERSE_PARAGRAPHS * 3
        if PPL_DATASET == "wikitext":
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                ppl_text = "\n\n".join(ds["text"])
                print("  (using WikiText-2-raw test set)")
            except Exception as e:
                print(f"  WikiText load failed ({e}), using built-in text")
        else:
            print("  (using built-in diverse text)")
        ppl = perplexity(
            model, tokenizer, ppl_text,
            max_length=PPL_MAX_LENGTH, stride=PPL_STRIDE, device=DEVICE,
        )
        print(f"ppl={ppl:.3f}  (baseline model, no cache tiers)")

    # --------------------------- NEEDLE ------------------------------ #
    if RUN_NEEDLE:
        print("\n=== NEEDLE-IN-A-HAYSTACK ===")
        result = run_needle(
            model, tokenizer,
            target_tokens=NEEDLE_TARGET_TOKENS, device=DEVICE,
        )
        print(result)


if __name__ == "__main__":
    main()
