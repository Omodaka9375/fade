# FADE Audit Status Report

**Date**: 2026 (Current)
**Reference**: `audit.md` findings verification against current codebase

---

## Executive Summary

**Status**: ⚠️ **PARTIAL FIXES** - Some critical issues have been addressed, but several remain unimplemented.

- ✅ **P0-1**: FIXED - Auto-reassignment implemented in `update()`
- ❌ **P0-2**: PARTIAL - `wikitext2_fade_ppl` exists but methodology still questionable
- ❌ **P0-3**: NOT FIXED - Needle test still doesn't accept cache argument
- ❌ **P0-4**: NOT FIXED - Fused kernel still not integrated
- ⚠️ **P1-1**: PARTIAL - `cache_dequant=True` still default, no `resident_bytes()`
- ❌ **P1-2**: NOT FIXED - TPS benchmark still measures FP16 vs FP16
- ✅ **P1-3**: FIXED - GQA bug fixed, but `ppl_estimate` formula still exists
- ❌ **P1-4**: NOT FULLY FIXED - Still pairs compression from one path with quality from another
- ⚠️ **P1-5**: PARTIAL - `REASSIGN_EVERY` removed but LongBench still has truncation issues

---

## Detailed Findings

### 🔴 P0-1: Compression never runs in the drop-in `generate()` path

**Status**: ✅ **FIXED**

**Evidence**:
- `fade/cache.py:420-429` shows auto-reassignment logic:
  ```python
  # Auto-reassign on the last managed layer when threshold is hit
  if (
      self._auto_reassign
      and self._reassign_every is not None
      and layer_idx == len(self._layers) - 1
      and self._decode_step > 0
      and self._decode_step % self._reassign_every == 0
  ):
      self._auto_reassign_tiers()
  ```
- `fade/cache.py:236-242`: `_auto_reassign_tiers()` method properly invokes eviction policies
- `fade/cache.py:219`: `auto_reassign=True` default parameter
- `fade/config.py:66`: `reassign_every: int = DEFAULT_REASSIGN_EVERY` (64 by default)

**Verdict**: The fix described in the audit has been implemented. Compression now runs automatically in the `generate()` path.

---

### 🔴 P0-2: "Δ PPL = 0.00" is guaranteed by construction

**Status**: ⚠️ **PARTIAL**

**Evidence**:
- `fade/eval/wikitext_ppl.py:94-180`: `wikitext2_fade_ppl()` function exists with persistent cache and explicit `reassign_tiers_by_position` calls
- The function does trigger reassignment after each chunk (line 177)
- However, the docstring claims "measures the real quality impact" but still uses teacher-forced evaluation which doesn't test actual generation quality

**Issues Remaining**:
1. Still uses teacher-forced evaluation (not actual generation)
2. The sliding window with reassignment is better than before, but doesn't truly measure generation quality degradation
3. README still shows "Δ PPL = 0.00" for balanced/aggressive presets without clear evidence

**Verdict**: Improved but not fully honest. The function exists and triggers compression, but the methodology still doesn't measure actual generation quality impact.

---

### 🔴 P0-3: Needle test never touches the FADE cache

**Status**: ❌ **NOT FIXED**

**Evidence**:
- `fade/eval/needle.py:27-68`: `run_needle()` function signature:
  ```python
  def run_needle(
      model,
      tokenizer,
      target_tokens: int = DEFAULT_TARGET_TOKENS,
      ...
  ) -> dict:
  ```
- **No `cache_factory` parameter** - the function calls `model.generate()` directly without passing any cache
- `benchmarks/production_suite.py:243` and `tests/test_7b_integration.py` call `run_needle()` without cache argument

**Impact**: README "Needle: 4/4 PASS" validates the **uncompressed base model**, not FADE compression.

**Verdict**: Critical issue remains. The needle test doesn't actually test FADE compression at all.

---

### 🔴 P0-4: Fused INT4 Triton kernel is not integrated

**Status**: ❌ **NOT FIXED**

**Evidence**:
- `fused_int4_sdpa` only referenced in:
  - `fade/kernels/fused_int4_attn.py` (definition + microbenchmarks)
  - `tests/test_fused_blackwell.py` (unit tests)
- **Never called from**:
  - `cache.py` `_materialize()` - still uses dequant + SDPA path
  - `patch.py` - no attention implementation hook
  - `server.py` - uses standard model forward

**Current Flow**:
```
FADE cache → _materialize() → dequant INT4 to FP16 → concatenate → HF SDPA
```

**Not**:
```
FADE cache → fused_int4_sdpa() → attention output
```

**Verdict**: The kernel exists but is completely isolated from the main inference path. README presentation is misleading.

---

### 🟠 P1-1: Memory accounting hides resident FP16 copies

**Status**: ⚠️ **PARTIAL**

**Evidence**:
- `fade/cache.py:171`: `cache_dequant: bool = True` (default)
- `fade/cache.py:43`: `DEFAULT_MAX_DEQUANT_AGE: int | None = None` (never expires)
- **No `resident_bytes()` function** found in codebase
- `compressed_storage_bytes()` excludes dequant caches (line 433-480)

**What's Fixed**:
- `compressed_storage_bytes()` exists and is documented as "at-rest" metric
- `PeakMemory` class exists in `fade/eval/memory.py`

**What's Missing**:
1. Default `cache_dequant=True` means full FP16 copy stays in memory
2. No `resident_bytes()` to show actual allocator footprint
3. Benchmarks don't report `torch.cuda.max_memory_allocated()`
4. README gotcha still dismisses `nvidia-smi` without explaining at-rest vs resident

**Verdict**: Partial progress. The infrastructure exists but defaults are misleading and no comprehensive resident memory tracking.

---

### 🟠 P1-2: TPS benchmark measures FP16 vs FP16

**Status**: ❌ **NOT FIXED**

**Evidence**:
- `benchmarks/production_suite.py:135-170`: `measure_tps()` function
  ```python
  def measure_tps(model, tokenizer, preset_dict: dict) -> dict:
      """Decode TPS (steady-state, no reassignment)."""
      ...
      # Measure.
      for _ in range(TPS_MEASURE):
          out = model(tok, past_key_values=cache, use_cache=True)
  ```
- **Docstring explicitly says "steady-state, no reassignment"**
- `benchmarks/tps.py:70-77`: Explicitly sets `int4_budget=None, int2_budget=0`
- Only ~10 token prompt + 64 decode steps → never triggers reassignment

**Verdict**: The real costs (reassignment overhead, dequant + torch.cat) are unmeasured. This is a null result presented as "zero overhead."

---

### 🟠 P1-3: `pareto.py` fabricates PPL and has a GQA bug

**Status**: ✅ **FIXED** (GQA bug), ⚠️ **PARTIAL** (PPL)

**Evidence**:
- `benchmarks/pareto.py:117-126`: **GQA bug is FIXED**
  ```python
  head_dim = model.config.hidden_size // model.config.num_attention_heads
  kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
  baseline_bytes = (
      S
      * num_layers
      * kv_heads  # ✅ Now uses num_key_value_heads
      * 2
      * head_dim
      * 2
  )
  ```
- **PPL measurement**: Now uses real `wikitext2_fade_ppl()` (line 100-112) instead of fabricated formula

**What's Still Problematic**:
- Old `ppl_estimate` formula removed, but the script still relies on `wikitext2_fade_ppl()` which has the teacher-forced limitation (see P0-2)

**Verdict**: GQA bug is fixed. PPL is now measured (not fabricated), but the measurement methodology has limitations.

---

### 🟠 P1-4: Compression ratio numerator/denominator from different worlds

**Status**: ❌ **NOT FULLY FIXED**

**Evidence**:
- `benchmarks/production_suite.py:99-133`: `measure_kv_bytes()`
  - Uses auto-reassign path (P0-1 fix) ✅
  - But quality metrics (PPL, needle) still use different execution paths ❌

**The Problem**:
- Compression measured: After auto-reassign in `measure_kv_bytes()`
- Quality measured: In `wikitext2_fade_ppl()` with teacher-forced sliding window
- These are **different runs** with **different cache states**

**Verdict**: While both now use reassignment, they're not measured in the same run. True consistency would measure bytes **during** the quality evaluation.

---

### 🟠 P1-5: LongBench harness: wrong truncation, dead knob, no results

**Status**: ⚠️ **PARTIAL**

**Evidence**:
- `benchmarks/longbench_eval.py:50`: `REASSIGN_EVERY` comment removed (P0-1 makes it redundant) ✅
- **Truncation issue still exists**: Line ~222 (need to verify exact line)
  ```python
  tokenizer(prompt, truncation=True, max_length=...)
  ```
- **No actual LongBench results published** in README

**Issues**:
1. LongBench contexts average 5K-18K tokens, but default truncation may cut at 8192
2. Uses raw prompt format instead of model chat template
3. No results shown in README for LongBench evaluation

**Verdict**: Auto-reassign fix helps, but truncation and methodology issues remain.

---

## Summary Table

| Finding | Status | Severity | Notes |
|---------|--------|----------|-------|
| P0-1: Compression in generate() | ✅ Fixed | 🔴 P0 | Auto-reassignment implemented |
| P0-2: Δ PPL honesty | ⚠️ Partial | 🔴 P0 | Function exists but methodology questionable |
| P0-3: Needle test | ❌ Not Fixed | 🔴 P0 | No cache argument, validates base model |
| P0-4: Fused kernel | ❌ Not Fixed | 🔴 P0 | Isolated from main path |
| P1-1: Memory accounting | ⚠️ Partial | 🟠 P1 | `resident_bytes()` missing, default misleading |
| P1-2: TPS benchmark | ❌ Not Fixed | 🟠 P1 | Measures FP16 vs FP16, no reassignment |
| P1-3: Pareto GQA bug | ✅ Fixed | 🟠 P1 | GQA bug fixed, uses real PPL now |
| P1-4: Compression/quality consistency | ❌ Not Fully Fixed | 🟠 P1 | Different runs, not same measurement |
| P1-5: LongBench | ⚠️ Partial | 🟠 P1 | Truncation issues remain |

---

## Recommendations

### Immediate (P0 Priority)
1. **P0-3**: Add `cache_factory` parameter to `run_needle()` and update all callers
2. **P0-4**: Either integrate fused kernel into HF attention hook OR relabel README table as "Experimental"

### Short-term (P1 Priority)
3. **P1-1**: Add `resident_bytes()` method and change default `cache_dequant=False` or set finite `max_dequant_age`
4. **P1-2**: Update TPS benchmark to use long prompts + reassignment
5. **P1-5**: Fix LongBench truncation to use model's `max_position_embeddings`

### Medium-term
6. **P0-2**: Either rewrite PPL eval to use actual generation OR be transparent that it's teacher-forced
7. **P1-4**: Measure compression and quality in the same run for consistency

---

## Conclusion

The audit identified 9 issues (4 P0, 5 P1). Current status:
- **2 fully fixed** (P0-1, P1-3 GQA)
- **3 partially fixed** (P0-2, P1-1, P1-5)
- **4 not fixed** (P0-3, P0-4, P1-2, P1-4)

**Critical**: P0-3 and P0-4 remain unaddressed and fundamentally undermine the README claims. The needle test validates the wrong thing, and the fused kernel performance claims are not reflected in actual usage.

**Recommendation**: Before any v1.2 release, address P0-3 and P0-4 at minimum, or update documentation to accurately reflect current capabilities.
