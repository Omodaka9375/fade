Here's the full document content:

---

# FADE — Audit Findings & Fix Plan

**Status of claims below:** every finding was verified directly in source (file + approximate line refs from v1.1.0, commit `04f9ed5`).

---

## Severity legend

| Level | Meaning |
|-------|---------|
| 🔴 P0 | Headline claim is unbacked/false; must fix before any further release |
| 🟠 P1 | Misleading measurement or default; fix in same release |
| 🟡 P2 | Quality/hygiene; fix opportunistically |

---

## 🔴 P0-1. Compression never runs in the drop-in `generate()` path

**Where:** `fade/cache.py:346` (`TieredKVCache.update`), `fade/config.py:66,84` (`reassign_every`), `fade/patch.py:125` (`create_tiered_cache`)

**Problem:**
- `update()` only appends new K/V to the FP16 tier and returns `_materialize()`. Quantization/eviction happens *only* in `apply_tier_assignment()` (`cache.py:881`), which is called *only* by `reassign_tiers*()` in `fade/policy.py` (lines 52, 118, 159, 203, 227) and `fade/learned_policy.py:178`.
- Nothing in the `model.generate(past_key_values=cache)` path ever calls those. `config.reassign_every` is validated at `config.py:84` but consumed **nowhere** except `experiments/run_tiered.py:265` and `fade/server.py:195,266`.
- Consequence: the README quickstart (`create_tiered_cache` + `generate()`) runs **pure FP16 attention, always**. Zero compression.

**Fix:**
1. Add a decode-step counter to `TieredKVCache`. In `update()`, when `layer_idx == num_layers - 1` and the step counter hits `config.reassign_every`, invoke the configured policy (`position`/`ema`/`adaptive` — H2O needs prefill attentions, so fall back to `position` when unavailable).
2. Store the policy choice on the cache at `create_tiered_cache` time so `update()` knows what to call.
3. Keep a `cache.auto_reassign = True` flag (default on) so manual-loop users can disable it.
4. Add a test: `generate()` 256 tokens with `balanced`, assert `compressed_storage_bytes()` shrinks and INT4 tiers are non-empty afterward.

---

## 🔴 P0-2. "Δ PPL = 0.00" is guaranteed by construction

**Where:** `fade/eval/wikitext_ppl.py:94–150` (`wikitext2_fade_ppl`)

**Problem:**
- Per sliding window: fresh cache, **one teacher-forced forward pass**, no decode steps, no `reassign_tiers*` call. Given P0-1, every K/V tensor attention sees is exact FP16.
- The docstring — *"INT4 quantization active inside the cache's update() method … measures the real quality impact"* — is **false**.
- Δ PPL = 0.00 would result for a random 1-bit quantizer too. The README row "Balanced (eviction) 12×, Δ PPL 0.00" is impossible if eviction actually ran (77% of tokens deleted).
- The `h2o → position` swap at line ~125 is irrelevant since no policy executes.

**Fix:**
1. Rewrite `wikitext2_fade_ppl`: prefill a chunk, **force reassignment** (`reassign_tiers_by_position` or the preset's policy), then teacher-force the next chunk against the compressed cache and accumulate NLL only on that continuation. Repeat sliding forward with a persistent cache.
2. Fix the docstring regardless.
3. After P0-1 lands, the simpler alternative: run real token-by-token decode with auto-reassign and score continuation NLL.
4. Re-run and publish the *actual* Δ PPL per preset. Expect non-zero for balanced/aggressive — that's the honest result.

---

## 🔴 P0-3. Needle test never touches the FADE cache

**Where:** `fade/eval/needle.py:27–68` (`run_needle`), callers: `benchmarks/production_suite.py:243`, `benchmarks/full_suite.py`, `tests/test_7b_integration.py`

**Problem:**
- `run_needle(model, tokenizer, ...)` takes **no cache argument**; `model.generate(**enc, ...)` uses the default HF DynamicCache. "Needle 4/4 PASS" in the README validates the **uncompressed base model**, placed under compression tables.
- Scope is also weak: 1 needle, 1 depth (0.5), 4 lengths (512–4096), lenient pass check (`"77" in answer`).

**Fix:**
1. Add `cache_factory: Callable | None` parameter; when provided, pass `past_key_values=cache_factory()` to `generate()` (requires P0-1 so compression actually engages).
2. Update all callers to pass the preset's cache factory; run needle **per preset**, report per-preset pass rate.
3. Expand grid: ≥3 depths (0.1/0.5/0.9), lengths up to model context (≥16K where the model supports it), ≥3 distinct needles.
4. Keep the docstring's honesty ("use RULER for production") and mirror that caveat in the README.

---

## 🔴 P0-4. Fused INT4 Triton kernel is not integrated

**Where:** `fade/kernels/fused_int4_attn.py` (`fused_int4_sdpa`, line 277); `fade/cache.py:693–860` (`_materialize_impl`, `_assemble_parts`, `_get_int4_dequant`)

**Problem:**
- Grep confirms `fused_int4_sdpa` is referenced only inside its own module, its microbenchmark, and `tests/test_fused_blackwell.py`. The model path always dequantizes to FP16 (`_get_int4_dequant`) and hands concatenated FP16 tensors to HF's standard SDPA.
- README presents the kernel table ("1.4× of FlashAttention") next to the drop-in usage — readers will assume `generate()` uses it. It doesn't, and *cannot* with the current `_materialize()` architecture.

**Fix (choose one, be explicit):**
- **(a) Integrate:** custom attention forward (HF `attn_implementation` hook or module patch in `patch.py`) that, for INT4-tier segments, calls `fused_int4_sdpa` and combines with FP16-segment attention via online-softmax merge. Significant work (est. 1–2 weeks) — this is the real "upgrade" item.
- **(b) Relabel:** move the kernel table under an "Experimental / standalone kernel" README section stating plainly it is not yet used by `generate()`. 15 minutes. Do (b) now even if (a) is planned.

---

## 🟠 P1-1. Memory accounting hides resident FP16 copies

**Where:** `fade/cache.py:43` (`DEFAULT_MAX_DEQUANT_AGE = None`), `cache.py:171` (`cache_dequant=True` default), `cache.py:433–480` (`compressed_storage_bytes` / `_layer_compressed_bytes`), `cache.py:663–667` (fp16 doubling-buffer views), `fade/eval/memory.py:16–57`

**Problem:**
- Default config keeps a **full FP16 dequant copy of the quantized middle tier resident forever** (`cache_dequant=True`, age `None` = never evict). Actual allocator footprint ≥ FP16 baseline.
- `compressed_storage_bytes()` excludes dequant caches *and* the doubling-buffer slack (up to ~2× overcapacity is invisible).
- `PeakMemory` (allocator-based, `memory.py:16–38`) exists but no benchmark uses it. README "Gotchas" even instructs users to trust the accounting over `nvidia-smi`.

**Fix:**
1. Default `max_dequant_age` to a finite value (e.g. `reassign_every`).
2. Add `resident_bytes()` that counts everything: tiers + scales + dequant caches + full buffer capacity (not view length). Keep `compressed_storage_bytes()` as the "at-rest" metric, but report **both**.
3. Benchmarks report `torch.cuda.max_memory_allocated()` deltas alongside byte accounting.
4. Rewrite the README gotcha to explain at-rest vs resident instead of dismissing `nvidia-smi`.

---

## 🟠 P1-2. TPS benchmark measures FP16 vs FP16

**Where:** `benchmarks/production_suite.py:135–170` (`measure_tps`, docstring: "steady-state, no reassignment"), `benchmarks/tps.py:70–77` (explicitly sets `int4_budget=None, int2_budget=0`)

**Problem:**
- ~10-token prompt + 64 decode steps, no reassignment → the FADE cache is a pure FP16 append buffer for the whole measurement. "13.3 vs 13.3 tok/s" is a null result presented as "zero overhead."
- The real costs (reassignment: re-RoPE, argsort, quant; per-step dequant + `torch.cat` of tier segments) are unmeasured.

**Fix:**
1. Long prompt (≥2048 tokens, matching the README table header), ≥256 decode steps, reassignment every `reassign_every` steps (or auto after P0-1).
2. Report: steady-state TPS, reassignment-step latency separately, and amortized TPS.
3. Publish the honest overhead number.

---

## 🟠 P1-3. `pareto.py` fabricates PPL and has a GQA bug

**Where:** `benchmarks/pareto.py:96–113`

**Problem:**
- `ppl_estimate = base_ppl * (1 + evict_frac * 0.5)  # rough model` — the quality axis of the "Pareto frontier" is an **invented formula**, not a measurement.
- Baseline-bytes formula uses `num_attention_heads` instead of `num_key_value_heads` → overstates baseline ~7× for GQA models (Qwen2.5-0.5B: 14 Q / 2 KV heads).

**Fix:**
1. Replace `ppl_estimate` with a real call into the fixed `wikitext2_fade_ppl` (P0-2), or delete the script until it measures.
2. `hidden_size // num_attention_heads * num_key_value_heads` for KV bytes (or reuse `fade/eval/memory.py` helpers — single source of truth).

---

## 🟠 P1-4. Compression ratio numerator/denominator from different worlds

**Where:** `benchmarks/production_suite.py:99–133` (`measure_kv_bytes`)

**Problem:**
- Compressed bytes: measured *after an explicit forced* `reassign_tiers_by_position` — a step no quality eval performs (see P0-2/P0-3). Baseline FP16 bytes: computed analytically, never measured.
- So the table pairs "compression from a code path" with "quality from a different code path where compression is off."

**Fix:**
- After P0-1/P0-2, measure bytes **during the same run** that produces the quality numbers (post-generate, auto-reassigned state). Measure the FP16 baseline from an actual DynamicCache too.

---

## 🟠 P1-5. LongBench harness: wrong truncation, dead knob, no results

**Where:** `benchmarks/longbench_eval.py:50` (`REASSIGN_EVERY` defined, never referenced), line ~222 (`tokenizer(prompt, truncation=True, max_length=...)`)