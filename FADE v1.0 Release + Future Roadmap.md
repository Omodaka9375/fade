# FADE v1.0 Release + v1.1 Roadmap
## Part F — v1.0 Release (now)
### F1. Update CHANGELOG for v1.0.0
Document all changes since 0.8.0 in `CHANGELOG.md`:
* v0.9.0: benchmark infrastructure, README corrections, WikiText-2 PPL, production_suite, longbench_eval
* v1.0.0: C3 sink split, C4 running counter, D1-D5 server productionization, E1-E3 test hardening, examples/notebook update, 4-model DGX Spark validation
### F2. Bump to 1.0.0
`pyproject.toml` version bump. Tag `v1.0.0`. Only after CHANGELOG is written.
## v1.1.0 Roadmap — Performance & Evaluation
### P1. Fix LongBench evaluation harness
Current issues:
* Truncation at 8192 tokens cuts most LongBench contexts (avg 5K-18K tokens)
* Raw prompt format instead of model chat template
* Manual decode loop for FADE presets vs model.generate() for baseline (unfair)
Fix:
* Increase max_length to 32768 (or model's max_position_embeddings)
* Use `tokenizer.apply_chat_template()` for all presets including baseline
* Use `model.generate(past_key_values=cache)` for FADE presets (drop-in path)
* Validate baseline scores match published LongBench leaderboard numbers before reporting FADE delta
Est: 2 days.
### P2. Delta-PPL per preset
Run WikiText-2 through the FADE cache (not just vanilla model) to measure actual PPL degradation per compression level.
Requires modifying `wikitext2_perplexity()` to accept a `past_key_values` cache and run tier reassignment during the sliding window.
Est: 1 day.
### P3. Pre-allocated materialize output buffer (C1)
Add `state.materialize_buf_k/v` doubling buffers. Populate on `apply_tier_assignment`, append-only between reassignments. Eliminates `torch.cat` on every decode step.
Benefit: measurable at very long contexts (10K+) or high-frequency decode on small models.
Flag: `cache.use_materialize_buffer = True`
Est: 1 day.
### P4. Incremental quantization on reassignment (C2)
Only quantize delta tokens transitioning into INT4 (~64 new tokens), not entire tier.
Flag: `cache.incremental_quant = False`
Risk: high (per-channel K scale recomputation on eviction).
Est: 2 days.
### P5. CUDA Graph capture (C5)
New file: `fade/graph.py`. Capture steady-state decode between reassignments, invalidate on reassignment.
Flag: `cache.enable_cuda_graph(model, template_ids)`
Benefit: significant on Blackwell where launch overhead is the bottleneck.
Est: 2 days.
## v1.2.0 Roadmap — Architecture
### A1. Paged attention layout for ragged batching
Replace `apply_tier_assignment_per_sequence` with a paged block pool (block_size=16). Per-sequence block tables. Paged-attention Triton kernel parallel to `_fused_int4_attn_fwd`.
This is the path to production continuous-batching.
Est: 12 days. Very high risk.
### A2. Speculative pre-compression
Pre-quantize FP16 tokens likely to be evicted at the next reassignment. Amortizes quant cost across decode steps.
Flag: `cache.speculative_precompress = True`
Est: 3 days.
### A3. Mixed-precision K per-channel
Keep top-k outlier K channels at INT8/FP16, rest at INT2. Requires channel permutation per layer and fused kernel changes.
Est: 4 days.
### A4. vLLM / SGLang native integration
Move beyond the adapter stub in `fade/serving/vllm_adapter.py`. Hook into vLLM's paged block manager so FADE tiers map to vLLM pages. Requires A1 (paged layout) first.
Est: 5 days.
## v1.3.0 Roadmap — Ecosystem
### E1. lm-eval-harness integration
Wrap FADE as an lm-eval model adapter so users can run any lm-eval task with FADE compression. Provides standardized, reproducible benchmarks (MMLU, GSM8K, RULER, etc.).
Est: 2 days.
### E2. GGUF export
Export FADE's compressed KV state to GGUF-compatible format for llama.cpp interop. Useful for offline cache sharing.
Est: 3 days.
### E3. FP8 cache tier
Add FP8 (E4M3) as a tier between FP16 and INT4. 2x compression with near-zero quality loss on Hopper/Blackwell.
Est: 2 days.
### E4. Automatic config tuning
Given a model + memory budget, auto-select the best preset (n_sink, recent_window, int4_budget) via binary search on needle + PPL.
Est: 3 days.
## Priority Order
1. **F1-F2**: v1.0.0 release (today)
2. **P1-P2**: fix evals and get delta-PPL (first v1.1 items)
3. **P3-P5**: performance optimizations
4. **A1-A4**: architecture for production serving
5. **E1-E4**: ecosystem integrations
