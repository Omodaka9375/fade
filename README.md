# FADE

[![CI](https://github.com/Omodaka9375/fade/actions/workflows/ci.yml/badge.svg)](https://github.com/Omodaka9375/fade/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/fade-kv.svg)](https://pypi.org/project/fade-kv/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omodaka9375/fade/blob/main/examples/quickstart.ipynb)

**Frequency-Adaptive Decay Encoding** — drop-in KV cache compression for HuggingFace transformers. Shrinks the KV cache 3–5× with near-baseline quality.

```python
from fade import FadeConfig, create_tiered_cache

cache = create_tiered_cache(model, config=FadeConfig.safe())
out = model.generate(input_ids, past_key_values=cache, max_new_tokens=256)
```

Works with `model.generate()` — greedy, sampling, beam search. No manual decode loop needed.

## How it works

Tokens live in tiers based on age and attention importance:

| Tier | What's stored | When |
|------|--------------|------|
| **FP16** | Full precision | First `N_SINK` tokens + last `RECENT_WINDOW` tokens |
| **INT4** | Bit-packed 4-bit | Middle-aged tokens (the bulk of the cache) |
| **INT2** | Grouped 2-bit | Optional deeper compression (lossy) |
| **PQ** | Product-quantized codes | ~2 bits/element via trained codebook (Phase 3) |
| **Evicted** | Nothing | Dropped when `INT4_BUDGET` is finite |

When tokens are evicted, surviving K tensors are un-RoPE'd at old positions and re-RoPE'd with contiguous StreamingLLM positions.

## Install

**From PyPI:**
```pwsh
pip install fade-kv
```

**From source (development):**
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch  # match your CUDA version: https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
```

Optional extras: `pip install fade-kv[cuda]` (accelerate), `fade-kv[eval]` (datasets), `fade-kv[codebook]` (scikit-learn for PQ).

## Quick start

### Presets

```python
from fade import FadeConfig, create_tiered_cache

# Safe: ~3-4x compression, 100% greedy match. No eviction.
cache = create_tiered_cache(model, config=FadeConfig.safe())

# Balanced: ~5x compression with H2O eviction.
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

### Manual decode with tier reassignment

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

*Learned policy requires a trained checkpoint: `python scripts/train_eviction_mlp.py`

## Supported models

FADE auto-detects the RoPE scheme from the model config:

- **Qwen2 / Qwen3** — vanilla RoPE, GQA
- **Llama / Llama-3.1** — vanilla + frequency-dependent scaling
- **Mistral** — vanilla RoPE, sliding-window
- **Phi-3** — vanilla RoPE
- **Gemma-2** — vanilla RoPE
- **Gemma 4** — proportional RoPE with `partial_rotary_factor` + per-layer-type dispatch
- **Falcon** — ALiBi (non-RoPE; re-RoPE is a no-op)
- **Qwen 3.5 / 3.6** — hybrid DeltaNet + softmax attention. FADE auto-detects `layer_types` and skips DeltaNet layers (only full-attention layers are tiered).

RoPE scaling types: `linear`, `llama3`, `ntk`, `dynamic`, `yarn`, `proportional`. Non-RoPE models (ALiBi, Bloom, MPT) work via the `NoRope` sentinel.

## Batching

Two modes:
- **Shared-tier** (default): all rows share positions and tier decisions. For lockstep decoding.
- **Per-sequence** (`apply_tier_assignment_per_sequence`): each row gets independent `[B, S]` tiers. For continuous-batching where sequences diverge.

## Performance

- **Pre-allocated FP16 buffer** — doubling buffer eliminates `torch.cat` on every decode step.
- **torch.compile** — `cache.enable_compile()` wraps `_materialize` between graph-break boundaries.
- **Triton INT4 kernel** — `int4_sdpa(q, k_packed, k_scale, v_packed, v_scale, force_triton=True)` runs fused INT4 unpack on CUDA. Exact parity validated on RTX 3060.
- **Dequant-cache age eviction** — `cache.max_dequant_age = N` periodically refreshes cached dequant buffers.
- **Benchmarks** — `python benchmarks/tps.py` (decode throughput), `python benchmarks/divergence.py` (quality).

## Checkpointing

```python
sd = cache.cache_state_dict()
torch.save(sd, "cache.pt")
cache.load_cache_state_dict(torch.load("cache.pt"))
```

## Observability

```python
from fade.telemetry import JsonlExporter, attach_telemetry
attach_telemetry(cache, JsonlExporter("events.jsonl"))
```

Debug dump: `cache.dump_debug("snapshot.json")`

## PQ codebook

```python
from fade.codebook import PQCodebook
cb = PQCodebook.train(calibration_vectors, sub_dim=32, num_centroids=256)
cache.set_codebooks(cb)  # enables TIER_PQ in tier assignment
```

Train codebooks from a real model: `python scripts/train_codebook.py`

## Results

| Config | Model | KV cache | Compression |
|--------|-------|----------|-------------|
| Phase 1-A | Qwen2.5-0.5B, 782 tok | 4.0 MiB | **67% smaller**, 100% token match |
| Phase 2 H2O | Qwen2.5-3B, 595 tok | 6.3 MiB | **79% smaller**, coherent output |

## Project layout

```
fade/
  cache.py           # TieredKVCache with 5 tiers (FP16/INT4/INT2/PQ/evict)
  config.py          # FadeConfig with presets
  quant.py           # INT4/INT2 quantization + bit-packing
  rope.py            # 7 RoPE schemes incl. Gemma 4 proportional
  policy.py          # Tier assignment: h2o, ema, position
  learned_policy.py  # Learned eviction MLP
  tracker.py         # AttentionTracker (per-layer EMA)
  patch.py           # load_model, create_tiered_cache, forward_with_tracking
  codebook.py        # PQ codebook train/encode/decode
  telemetry.py       # Structured telemetry + exporters
  kernels/           # Triton INT4 unpack kernel + torch fallback
  serving/           # vLLM / SGLang adapter stubs
  eval/              # Perplexity, needle, quality suite
examples/            # quickstart.py
experiments/         # run_baseline.py, run_tiered.py
benchmarks/          # tps.py, divergence.py
scripts/             # train_eviction_mlp.py, train_codebook.py
tests/               # 136 tests, all CPU, no downloads
```

## Gotchas

1. **Attention impl**: `eager` only needed for H2O prefill. Use `load_model(attn_impl="auto")`.
2. **Transformers version**: verified on 4.45 and 5.3. Weekly canary CI runs against `transformers@main`.
3. **Memory**: use `cache.compressed_storage_bytes()`, not `nvidia-smi`.
4. **RoPE precision**: all math in float32, cast through model dtype to match rounding.
5. **Hybrid models**: Qwen 3.5/3.6 DeltaNet layers are auto-skipped — only full-attention layers are tiered.
6. **Triton kernel**: opt-in via `force_triton=True`. Run `check_parity()` on your hardware first.

## License

Apache-2.0. See [LICENSE](LICENSE).
