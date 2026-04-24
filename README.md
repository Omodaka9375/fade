# FADE

[![CI](https://github.com/Omodaka9375/fade/actions/workflows/ci.yml/badge.svg)](https://github.com/Omodaka9375/fade/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://python.org)

**Frequency-Adaptive Decay Encoding** — drop-in KV cache compression for HuggingFace transformers. Shrinks the KV cache 3–5× with near-baseline quality.

```python
from fade import FadeConfig, create_tiered_cache

cache = create_tiered_cache(model, config=FadeConfig.safe())
out = model.generate(input_ids, past_key_values=cache, max_new_tokens=256)
```

That's it. Works with `model.generate()` — greedy, sampling, beam search.

## How it works

Tokens live in tiers based on age and attention importance:

| Tier | What's stored | When |
|------|--------------|------|
| **FP16** | Full precision | First `N_SINK` tokens + last `RECENT_WINDOW` tokens |
| **INT4** | Bit-packed 4-bit | Middle-aged tokens (the bulk of the cache) |
| **INT2** | Grouped 2-bit | Optional deeper compression (lossy) |
| **Evicted** | Nothing | Dropped when `INT4_BUDGET` is finite |

When tokens are evicted, surviving K tensors are un-RoPE'd at old positions and re-RoPE'd with contiguous StreamingLLM positions. This keeps attention distances correct.

## Install

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch  # match your CUDA version: https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
```

Optional extras: `[cuda]` (accelerate), `[eval]` (datasets), `[codebook]` (scikit-learn for PQ).

## Quick start

### Presets

```python
from fade import FadeConfig, create_tiered_cache

# Safe: ~3-4x compression, 100% greedy match. No eviction.
cache = create_tiered_cache(model, config=FadeConfig.safe())

# Balanced: ~5x compression with H2O eviction. Recommended for production.
cache = create_tiered_cache(model, config=FadeConfig.balanced())

# Aggressive: ~7-8x compression. Validate on your workload first.
cache = create_tiered_cache(model, config=FadeConfig.aggressive())
```

### Custom config

```python
cache = create_tiered_cache(model, config=FadeConfig(
    phase="2",
    n_sink=4,
    recent_window=64,
    int4_budget=400,
    eviction_policy="h2o",       # "h2o", "ema", "position", or "learned"
    middle_k_bits=4,             # K stays INT4 (outlier-sensitive)
    middle_v_bits=2,             # V at INT2 (~30% more compression)
))
```

### Manual decode loop (when you need tier reassignment)

```python
from fade.patch import forward_with_tracking, load_model
from fade.policy import reassign_tiers
from fade.tracker import AttentionTracker

model, tokenizer = load_model("Qwen/Qwen2.5-3B-Instruct", attn_impl="auto", need_attentions=True)
cache = create_tiered_cache(model, config=FadeConfig.balanced())
tracker = AttentionTracker(num_layers=model.config.num_hidden_layers)

out = forward_with_tracking(model, input_ids, cache, tracker=tracker)
for step in range(max_tokens):
    out = forward_with_tracking(model, next_token, cache, tracker=tracker)
    if (step + 1) % 64 == 0:
        reassign_tiers(cache, tracker, model.config.num_hidden_layers)
```

## Eviction policies

| Policy | Quality | Speed | Needs attention? |
|--------|---------|-------|-----------------|
| `h2o` | Best | Normal | Yes (prefill only) |
| `ema` | Good | Normal | Yes (decode only) |
| `position` | Fair | Fast | No |
| `learned` | Good* | Fast | No |

*Learned policy requires a trained checkpoint. Train with: `python scripts/train_eviction_mlp.py`

## Supported models

FADE auto-detects the RoPE scheme from the model config:

- **Qwen2 / Qwen3** — vanilla RoPE, GQA
- **Llama / Llama-3.1** — vanilla + frequency-dependent scaling
- **Mistral** — vanilla RoPE, sliding-window
- **Phi-3** — vanilla RoPE
- **Gemma-2** — vanilla RoPE
- **Gemma 4** — proportional RoPE with `partial_rotary_factor` + per-layer-type RoPE (sliding vs full attention). FADE uses the `full_attention` scheme for re-RoPE.
- **Falcon** — ALiBi (non-RoPE; re-RoPE is a no-op)

RoPE scaling types: `linear`, `llama3`, `ntk`, `dynamic`, `yarn`, `proportional` (Gemma 4). Non-RoPE models (ALiBi, Bloom, MPT) work through the `NoRope` sentinel.

### Not yet supported

- **Qwen 3.5 / 3.6** — these use a hybrid Gated DeltaNet + softmax attention architecture (3 DeltaNet layers per 1 full-attention layer). DeltaNet layers maintain a fixed-size recurrent state instead of a K/V cache, so FADE can only compress the 25% of layers that use full attention. Proper support requires the cache to understand `layer_types` and skip DeltaNet layers. This is tracked but not yet implemented.

## Batching

Supports `batch_size >= 1` with shared-tier assignment (all rows share positions and tier decisions). Batch size is pinned on the first `update()` call. Per-sequence ragged eviction is deferred to vLLM/SGLang block-manager integration.

## Checkpointing

Save and restore the compressed cache for long sessions:

```python
sd = cache.cache_state_dict()
torch.save(sd, "cache.pt")
# Later:
cache.load_cache_state_dict(torch.load("cache.pt"))
```

## Observability

```python
from fade.telemetry import JsonlExporter, attach_telemetry

exporter = JsonlExporter("events.jsonl")
attach_telemetry(cache, exporter)
# Every tier reassignment emits: layer, fp16/int4/int2/evicted counts, score stats
```

Dump cache state for debugging: `cache.dump_debug("snapshot.json")`

## PQ codebook (Phase 3)

Product quantization for ~2 bits/element — alternative to INT2:

```python
from fade.codebook import PQCodebook
cb = PQCodebook.train(calibration_vectors, sub_dim=32, num_centroids=256)
codes = cb.encode(kv_vectors)   # uint8
decoded = cb.decode(codes)       # float
```

Requires `pip install fade[codebook]`.

## Benchmarks

```pwsh
python experiments/run_baseline.py        # FP16 reference
python experiments/run_tiered.py          # side-by-side comparison
python benchmarks/tps.py                  # decode throughput
python benchmarks/divergence.py --csv o.csv  # token-by-token match rate
```

## Results

### Phase 1-A (no eviction) — Qwen2.5-0.5B, ~782 tokens
- Baseline: kv_cache 12.2 MiB
- Tiered: kv_cache **4.0 MiB (-67%)**, 100% token match

### Phase 2 (H2O eviction) — Qwen2.5-3B, 595 tokens
- Baseline: kv_cache 29.9 MiB
- Tiered: kv_cache **6.3 MiB (-79%)**, coherent output, ~29% evicted

## Project layout

```
fade/
  cache.py           # TieredKVCache (DynamicCache subclass)
  config.py          # FadeConfig with presets
  quant.py           # INT4/INT2 quantization + bit-packing
  rope.py            # RoPE scheme abstraction (7 schemes incl. Gemma 4 proportional)
  policy.py          # Tier assignment: h2o, ema, position
  learned_policy.py  # Learned eviction MLP
  tracker.py         # AttentionTracker (per-layer EMA)
  patch.py           # load_model, create_tiered_cache, forward_with_tracking
  codebook.py        # PQ codebook (Phase 3)
  telemetry.py       # Structured telemetry + exporters
  kernels/           # Fused INT4 dequant+SDPA kernel
  serving/           # vLLM / SGLang adapter stubs
  eval/              # Perplexity, needle-in-a-haystack, quality suite
experiments/         # run_baseline.py, run_tiered.py
benchmarks/          # tps.py, divergence.py
scripts/             # train_eviction_mlp.py
tests/               # 126 tests, all CPU, no downloads
```

## Gotchas

1. **Attention implementation**: `eager` is only needed for H2O prefill. Use `load_model(attn_impl="auto", need_attentions=...)` — it picks `sdpa` when attention capture isn't needed.
2. **Transformers version**: verified on 4.45 and 5.3. The `fade/_compat.py` shim handles API differences. A weekly canary CI runs against `transformers@main`.
3. **Memory measurement**: use `cache.compressed_storage_bytes()` for KV-only accounting, not `nvidia-smi` (model weights dominate peak memory).
4. **RoPE precision**: all RoPE math runs in float32 internally, casting through model dtype to match the model's rounding exactly.
5. **Batch size**: start with 1. Shared-tier batching works; per-sequence ragged eviction is not yet implemented.

## License

Apache-2.0. See [LICENSE](LICENSE).
