# Changelog
All notable changes to FADE will be documented in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.7.0] — Phase 2 optimization (tracker, policy, codebook, learned-policy vectorization)
### Changed
- **Tracker**: sum attention mass in source dtype instead of allocating full
  fp32 copy; cast result to fp32 after reduction.
- **Policy**: same fp16-sum optimization in `reassign_tiers_h2o`.
- **Learned policy**: batched MLP forward across all layers (single
  `mlp.to(device)` + one forward pass instead of per-layer loop).
  Also replaced `topk` calls with single `argsort` + slice.
- **Codebook**: vectorized `PQCodebook.encode` via batched `torch.cdist`
  (eliminates per-sub Python loop).
- **Codebook**: vectorized `PQCodebook.decode` via batched `torch.gather`
  (eliminates per-sub Python loop).
- **Codebook**: removed dead `flat.shape[0]` expression in `decode`.

## [0.6.0] — Phase 1 optimization (decode steady-state & reassignment quick wins)
### Changed
- **Quant**: branchless bit-arithmetic sign extension in INT4/INT2 unpack
  (`torch.where` → `x - ((x & sign_bit) << 1)`).
- **Quant**: multiply by `inv_scale` instead of dividing by `scale` in all
  six quantization functions (INT4 K/V, INT2 K/V, rotated K/V).
- **Quant**: `pad_to_group` uses single alloc + copy instead of alloc + `torch.cat`.
- **RoPE**: `Llama3.inv_freq` and `Yarn.inv_freq` fully vectorized (eliminated
  Python for-loops and per-dim `.item()` calls).
- **RoPE**: `inv_freq` cached per `(device)` on every `RopeScheme` subclass.
- **RoPE**: skip lossy fp32→model_dtype→fp32 round-trip for bf16 in
  `compute_cos_sin`.
- **Cache**: collapsed two `.item()` host/device syncs into one in both
  `apply_tier_assignment` and the per-sequence variant.
- **Cache**: FP16 pre-alloc buffer persisted across reassignments (reseeds
  existing capacity instead of nulling and reallocating).
- **Cache**: fused old/new cos/sin into a single batched `compute_cos_sin`
  call during eviction re-RoPE (halves kernel launches).
- **Policy**: single `argsort` + slice replaces two `topk` calls in
  `_assign_one_layer`.

## [Unreleased]
### Added
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- `fade/_compat.py` shim so `DynamicCache` and related symbols resolve on both
  the `transformers 4.45` and `transformers 5.3` minors.
- GitHub Actions CI workflow: lint (ruff), typecheck (pyright), test (pytest)
  on the Python × transformers matrix, plus build + release-on-tag.
- `.pre-commit-config.yaml` with ruff-format, ruff check, and typos.
- New `[cuda]`, `[serving]`, `[eval]` extras in `pyproject.toml`.
- **Batching**: `TieredKVCache` now supports `batch_size >= 1` under the
  shared-tier contract (same positions and tier assignment across rows,
  scores pooled across the batch dim). New `batch_size` kwarg is pinned on
  first `update()`; mismatched subsequent calls raise `ValueError`. New
  tests under `tests/test_cache_batched.py` lock in the contract and an
  end-to-end integration test confirms per-row logits match the unbatched
  reference. `BATCH_SIZE` knob added to `experiments/run_tiered.py`.
- **`FadeConfig` dataclass** with `safe` / `balanced` / `aggressive` presets,
  validated invariants, and `create_tiered_cache(model, config=...)` wiring.
- **Attention-impl flexibility (W4)**: new `load_model(attn_impl=...)` entry
  point supports `"eager"`, `"sdpa"`, `"flash_attention_2"`, and `"auto"`.
  `load_model_eager` kept as a backward-compat shim. `forward_with_tracking`
  degrades gracefully with a one-time `RuntimeWarning` when a tracker is
  supplied under an attention impl that drops attentions (previously it
  asserted). The `position` and EMA-decode eviction policies now work under
  SDPA / FA2 without requiring eager.
- **bf16 support** verified: `quant.py` INT4 K/V round-trip tests added for
  `torch.bfloat16` and zero-tensor edge cases confirm no NaN/Inf paths.
- **Model coverage (W3)**: new `fade/rope.py` with `RopeScheme` abstraction
  supporting `Vanilla`, `LinearScaled`, `Llama3`, `NtkAware`, `Yarn`, and
  `NoRope` (ALiBi/absolute). `extract_rope_scheme()` auto-detects from HF
  config. `TieredKVCache._compute_cos_sin` now delegates to the scheme.
  Per-architecture integration tests for Qwen2, Llama, Llama-3.1 (scaling),
  Mistral, Phi-3, Gemma-2, and Falcon (ALiBi). GQA/MQA INT4 scale shape
  assertions added.
- **HF `generate()` compatibility (W6)**: `TieredKVCache` works as a drop-in
  `past_key_values` for greedy, sampling, and beam search. No manual decode
  loop needed.
- **Cache checkpointing (W6)**: `cache_state_dict()` / `load_cache_state_dict()`
  serialize/restore the essential compressed form (no dequant buffers).
- **Serving adapter stubs (W6)**: `fade/serving/` with integration docs for
  vLLM and SGLang.
- **Quality evaluation (W7)**: `fade/eval/quality.py` with `run_quality_suite()`
  for needle + perplexity checks. `benchmarks/divergence.py` for token-by-token
  divergence analysis. `tests/test_eval_gate.py` validates the pipeline.
- **Asymmetric K/V compression (W10)**: `middle_k_bits` and `middle_v_bits`
  in `FadeConfig` and `TieredKVCache`. K stays at INT4 while V can use INT2
  for ~30% more compression. Wired through `apply_tier_assignment` and
  `_get_int4_dequant`.
- **Learned eviction policy (W10)**: `fade/learned_policy.py` with
  `EvictionMLP` (~2K params) and `reassign_tiers_learned()`. Training script
  at `scripts/train_eviction_mlp.py`. `eviction_policy="learned"` accepted
  in `FadeConfig`.
- **Observability (W8)**: `fade/telemetry.py` with `TierEvent` dataclass,
  `MetricsExporter` ABC, `StdoutExporter`, `JsonlExporter`, `ListExporter`.
  `attach_telemetry(cache, exporter)` hooks into `apply_tier_assignment`.
  `cache.dump_debug(path)` writes JSON snapshot of tier membership.
- **PQ codebook (W9)**: `PQCodebook.train()` implemented with
  `MiniBatchKMeans`. `encode()` via `torch.cdist`, `decode()` via
  `torch.gather`. Per-(layer, head) codebooks, uint8 codes,
  ~2 bits/element. Requires `scikit-learn` (`pip install fade[codebook]`).
- **Performance (W5)**: pre-allocated FP16 append buffer (doubling strategy,
  avoids `torch.cat` on every decode step). Dequant-cache age tracking with
  configurable `max_dequant_age`. `fade/kernels/int4_attention.py` with fused
  INT4 dequant+SDPA (pure-torch fallback; Triton kernel pending parity tests).
  `benchmarks/tps.py` TPS micro-benchmark.
### Changed
- `fade.__version__` is now read from package metadata via `importlib.metadata`
  so it cannot drift from `pyproject.toml`.
- `pyproject.toml` version bumped to `0.2.0` to match the package.
## [0.2.0] — Phase 2
### Added
- Re-RoPE on eviction: un-RoPE at old absolute positions, re-RoPE with
  StreamingLLM contiguous positions.
- Three eviction policies: `h2o`, `ema`, `position`.
- INT2 tier (available, but too lossy for current models).
- `create_tiered_cache(model)` convenience factory.
## [0.1.0] — Phase 1-A
### Added
- FP16 sinks + recent window + bit-packed INT4 middle tier.
- `TieredKVCache(DynamicCache)`.
- Baseline, tiered, perplexity, and needle experiments.
