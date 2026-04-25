# FADE

[![CI](https://github.com/Omodaka9375/fade/actions/workflows/ci.yml/badge.svg)](https://github.com/Omodaka9375/fade/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/fade-kv.svg?cacheSeconds=3600)](https://pypi.org/project/fade-kv/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omodaka9375/fade/blob/main/examples/quickstart.ipynb)

**Frequency-Adaptive Decay Encoding** — drop-in KV cache compression for HuggingFace transformers. Shrinks the KV cache **3.5–12×** with near-baseline quality (up to 23× in aggressive mode — validate on your workload).

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

```mermaid
flowchart LR
    A["New Token"] --> B["FP16 Tier\n(sinks + recent)"]
    B -->|reassign| C["INT4 Tier\n(middle)"]
    C -->|budget full| D["INT2 / PQ\n(deep compress)"]
    D -->|evict| E["Dropped\n(re-RoPE survivors)"]
    style A fill:#4CAF50,color:#fff
    style B fill:#2196F3,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#f44336,color:#fff
    style E fill:#9E9E9E,color:#fff
```

### Compression at a glance

| Config | KV cache | Compression |
|--------|----------|:-----------:|
| Baseline FP16 | 112.00 MiB | 1× |
| Safe (INT4) | 31.24 MiB | **3.6×** |
| Rotated 2-bit | 17.70 MiB | **6.3×** |
| Balanced (eviction) | 9.30 MiB | **12.0×** |
| Aggressive | 4.77 MiB | **23.5×** |

> Qwen2.5-7B-Instruct, 2048 tokens, DGX Spark. Needle: 4/4 PASS (512–4096). WikiText-2 PPL: 6.56.

### Fused kernel speed

| Path | Latency | vs FP16 |
|------|:-------:|:-------:|
| FP16 FlashAttention | 0.133 ms | 1.0× |
| **Fused INT4 (FADE)** | **0.189 ms** | **1.4×** |
| Dequant + SDPA (old) | 0.932 ms | 7.0× |

### How FADE compares (2026)

| | **FADE** | **kvpress** (NVIDIA) | **TurboQuant** (Google, ICLR 2026) | **KVTC** (NVIDIA, ICLR 2026) |
|---|---|---|---|---|
| **Approach** | Tiered quant + eviction + re-RoPE | Token eviction / scoring (30+ methods) | Rotation + Lloyd-Max codebook | PCA + DP bit allocation + entropy coding |
| **Compression** | 3.5–12× (23× aggressive) | 2–10× (eviction only) | 4–6× (3.5-bit zero-loss claimed) | 6–20× (up to 40× with entropy) |
| **Quantization** | INT4/INT2/PQ + rotated 2-bit | Via HF `QuantizedCache` | 3–4 bit | 1–6 bit adaptive |
| **Eviction** | H2O, EMA, position, adaptive, learned | 30+ methods (SnapKV, TOVA, KVzap, etc.) | None | None |
| **Re-RoPE** | ✅ StreamingLLM contiguous | Partial (KeyRerotationPress, FinchPress) | ❌ | ✅ (undo before PCA) |
| **Fused kernel** | ✅ Triton INT4 FlashAttn | ❌ | ✅ Triton fused | ✅ Triton |
| **HF generate()** | ✅ Drop-in | Pipeline + context manager | ✅ Drop-in | ❌ |
| **Serving** | ✅ fade-server (OpenAI API) | ❌ | ✅ vLLM / SGLang integration | ❌ |
| **Hybrid models** | ✅ Qwen 3.5 DeltaNet skip | ❌ | ❌ | ❌ |
| **Per-sequence batching** | ✅ Ragged tiers | ❌ | ❌ | ❌ |
| **Stars** | New | 1K+ | 1K+ (across implementations) | ~10 |
| **Install** | `pip install fade-kv` | `pip install kvpress` | `pip install turboquant-kv` | From source |

**FADE's unique advantage**: only system that combines quantization + attention-aware eviction + correct re-RoPE in one drop-in cache.

## Install

**From PyPI:**
```pwsh
pip install fade-kv
pip install fade-kv[server]  # adds fade-server CLI
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

# Safe: ~3-4x compression, no eviction.
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

### Rotated 2-bit backend (~6× compression)

```python
from fade.backends import get_backend

cache = create_tiered_cache(model, config=FadeConfig.safe(),
    quant_backend=get_backend("rotated", head_dim=64, bits=2))
```

Random orthogonal rotation spreads per-channel outliers before quantization, making 2-bit viable. Uses native PyTorch — no external dependencies.

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

| Policy     | Quality | Speed  | Needs attention?   |
|------------|---------|--------|--------------------|
| `h2o`      | Best    | Normal | Yes (prefill only) |
| `ema`      | Good    | Normal | Yes (decode only)  |
| `adaptive` | Good    | Normal | Yes (decode EMA)   |
| `position` | Fair    | Fast   | No                 |
| `learned`  | Good*   | Fast   | No                 |

`adaptive` splits middle tokens by attention score: high→INT4, low→INT2, lowest→evict.

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

- **Fused INT4 FlashAttention kernel** — single Triton kernel reads packed INT4 K/V, computes attention with online softmax, writes fp16 output. Never materializes fp16 K/V. **4.9× faster** than the old dequant+SDPA path, within 1.4× of pure fp16 FlashAttention on RTX 3060.
- **Pre-allocated FP16 buffer** — doubling buffer eliminates `torch.cat` on every decode step.
- **torch.compile** — `cache.enable_compile()` wraps `_materialize` between graph-break boundaries.
- **Dequant-cache age eviction** — `cache.max_dequant_age = N` periodically refreshes cached dequant buffers.
- **Benchmarks** — `python benchmarks/tps.py` (decode throughput), `python benchmarks/divergence.py` (quality).

```python
# Use the fused kernel directly:
from fade.kernels.fused_int4_attn import fused_int4_sdpa
out = fused_int4_sdpa(q, k_packed, k_scale, v_packed, v_scale)
```

## Inference server

OpenAI-compatible API with automatic tier management:

```pwsh
fade-server --model Qwen/Qwen2.5-0.5B-Instruct --preset balanced --port 8000
```

```pwsh
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":100}"
```

Endpoints: `/v1/chat/completions` (greedy + sampling), `/v1/models`, `/health`.

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

### DGX Spark — Qwen2.5-7B-Instruct (2048 tokens)

| Config | KV cache | Compression | Decode TPS |
|--------|----------|:-----------:|:----------:|
| Baseline FP16 | 112.00 MiB | 1.0× | 13.3 tok/s |
| Safe (INT4) | 31.24 MiB | **3.6×** | 13.3 tok/s |
| Rotated 2-bit | 17.70 MiB | **6.3×** | 13.3 tok/s |
| Balanced (eviction) | 9.30 MiB | **12.0×** | 13.3 tok/s |
| Aggressive | 4.77 MiB | **23.5×** | 13.3 tok/s |

> NVIDIA DGX Spark (Grace Blackwell, 128 GB). Needle: 4/4 PASS (512–4096 tokens). WikiText-2 baseline PPL: 6.56.

### RTX 3060 — Qwen2.5-0.5B-Instruct (2048 tokens)

| Config | KV cache | Compression | Decode TPS |
|--------|----------|:-----------:|:----------:|
| Baseline FP16 | 24.00 MiB | 1.0× | 128.5 tok/s |
| Safe (INT4) | 6.78 MiB | **3.5×** | 125.8 tok/s |
| Rotated 2-bit | 3.88 MiB | **6.2×** | 125.9 tok/s |
| Balanced (eviction) | 2.01 MiB | **11.9×** | 125.8 tok/s |
| Aggressive | 1.03 MiB | **23.3×** | 125.9 tok/s |

> Needle: 4/4 PASS (512–4096 tokens). Baseline FP16 PPL: 1.24. TPS overhead: ~2%.

### Fused Triton kernel (RTX 3060)

| Path | Time | vs FP16 |
|------|------|---------|
| FP16 SDPA | 0.133 ms | 1.0× |
| Dequant + SDPA (old) | 0.932 ms | 7.0× slower |
| **Fused INT4 (new)** | **0.189 ms** | **1.4×** |

Run benchmarks yourself: `python benchmarks/production_suite.py`, `python benchmarks/full_suite.py`

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
  kernels/           # Fused INT4 FlashAttention kernel + unpack kernel + fallback
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
6. **Triton kernels**: fused attention via `fused_int4_sdpa()`, unpack-only via `int4_sdpa(force_triton=True)`. Run `check_fused_parity()` to validate on your hardware.

## Citations

FADE builds on ideas from these papers (all independently reimplemented — see [NOTICE](NOTICE) for details):

- **H2O** — Zhang et al., 2023. *Heavy-Hitter Oracle for Efficient Generative Inference of LLMs.* [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- **StreamingLLM** — Xiao et al., 2023. *Efficient Streaming Language Models with Attention Sinks.* [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **KIVI** — Liu et al., 2024. *A Tuning-Free KV Cache Quantization Algorithm.* [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **TurboQuant** — Zandieh et al., ICLR 2026. *Online Vector Quantization with Near-optimal Distortion Rate.* [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **KVTC** — Staniszewski & Łańcucki, ICLR 2026. *KV Cache Transform Coding for Compact Storage in LLM Inference.* [arXiv:2511.01815](https://arxiv.org/abs/2511.01815)
- **KnormPress** — Devoto et al., 2024. *A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression.* [arXiv:2406.11430](https://arxiv.org/abs/2406.11430)

If you use FADE in your work:
```bibtex
@software{fade2026,
  title  = {FADE: Frequency-Adaptive Decay Encoding},
  author = {Branislav Đalić},
  url    = {https://github.com/Omodaka9375/fade},
  year   = {2026},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
