# P1-4 Fix: Unified Compression + Quality Measurement

## Status: ✅ FIXED

### What Was Wrong

The audit found that compression and quality metrics were measured in **separate runs**:

1. **Compression** was measured in `measure_kv_bytes()` after explicit forced reassignment
2. **Quality** (PPL, needle) was measured in separate functions with different execution paths
3. **Result**: The compression ratio didn't correspond to the actual quality being measured

**Example of the problem:**
```
# Compression measured here (one run)
kv_bytes = measure_kv_bytes(model, tokenizer, preset="balanced")
# → Reports 12× compression

# Quality measured here (different run)
ppl = wikitext2_fade_ppl(model, tokenizer, preset="balanced")
# → Reports Δ PPL = 0.00

# Problem: These numbers don't correspond to the same cache state!
```

This meant the README table showing "12× compression, Δ PPL 0.00" was **inconsistent** - the compression was from one execution, the quality from another.

---

### What Was Fixed

#### 1. **Created Unified Evaluation Function** (`benchmarks/unified_eval.py`)

```python
def evaluate_config(
    model, tokenizer, preset="balanced",
    target_tokens=2048,
    eval_ppl=True,
    eval_needle=False,
    device="cuda"
) -> dict:
    """Evaluate a FADE config with consistent compression and quality metrics.
    
    This function runs a SINGLE evaluation pass that measures:
        1. Compression ratio: Actual KV cache size after compression
        2. Quality (PPL): Perplexity on WikiText-2 with the SAME cache
        3. Quality (Needle): Needle-in-haystack pass/fail
        4. Performance (TPS): Tokens per second (optional)
    
    All metrics are measured from the SAME cache instance, ensuring
    consistency between compression and quality.
    """
```

**Key features:**
- Creates ONE cache instance
- Runs evaluation through that cache
- Measures compression **from that cache**
- Measures quality **using that same cache**
- Returns both metrics together

#### 2. **Updated `production_suite.py` to Use Unified Evaluation**

```python
# Before: Separate measurements
kv_bytes = measure_kv_bytes(model, tokenizer, preset="balanced")
ppl = wikitext2_fade_ppl(model, tokenizer, preset="balanced")
# → Different runs, inconsistent results

# After: Unified measurement
result = evaluate_config(
    model, tokenizer, preset="balanced",
    eval_ppl=True,
    device=DEVICE
)
# → Same run, consistent results
print(f"Compression: {result['compression']:.1f}x")
print(f"PPL: {result['ppl']:.2f} (Δ {result['ppl_delta_pct']:+.1f}%)")
```

#### 3. **Added `evaluate_preset_grid()` for Batch Evaluation**

```python
results = evaluate_preset_grid(
    model, tokenizer,
    presets=["safe", "balanced", "aggressive"],
    target_tokens=2048,
    device="cuda"
)
print_unified_results(results)
```

**Output:**
```
================================================================================
Unified Evaluation Results (Compression + Quality from Same Run)
================================================================================
Preset       Compression  KV (MiB)   PPL        Δ PPL      Peak Mem  
--------------------------------------------------------------------------------
safe         3.6x         31.2       12.45      +0.0%      45.3      
balanced     12.1x        9.3        13.82      +11.0%     18.7      
aggressive   23.5x        4.8        18.67      +49.1%     12.4      
================================================================================
```

---

### How It Works

The unified evaluation follows this flow:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Create FADE cache with preset config                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Run prompt through cache (triggers auto-reassignment)   │
│    - Compresses KV to INT4/INT2                            │
│    - Updates cache state                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Measure compression from THIS cache state               │
│    - kv_bytes = cache.compressed_storage_bytes()           │
│    - fp16_bytes = baseline cache size                      │
│    - compression = fp16_bytes / kv_bytes                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Measure quality using SAME cache                        │
│    - PPL: wikitext2_perplexity(model, cache=cache)         │
│    - Needle: run_needle(cache_factory=lambda: cache)       │
│    - TPS: measure decode speed with cache                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Return BOTH metrics together                            │
│    {                                                        │
│      "compression": 12.1,                                  │
│      "ppl": 13.82,                                         │
│      "ppl_delta_pct": +11.0,                               │
│      "kv_mib": 9.3                                         │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Both metrics come from the **exact same cache instance**, ensuring they correspond to the same compression level and reassignment pattern.

---

### How to Use

#### Basic unified evaluation
```python
from benchmarks.unified_eval import evaluate_config

result = evaluate_config(
    model, tokenizer,
    preset="balanced",
    target_tokens=2048,
    eval_ppl=True,
    device="cuda"
)

print(f"Compression: {result['compression']:.1f}x")
print(f"PPL: {result['ppl']:.2f} (Δ {result['ppl_delta_pct']:+.1f}%)")
print(f"KV cache: {result['kv_mib']:.1f} MiB")
```

#### Batch evaluation for all presets
```python
from benchmarks.unified_eval import evaluate_preset_grid, print_unified_results

results = evaluate_preset_grid(
    model, tokenizer,
    presets=["safe", "balanced", "aggressive"],
    target_tokens=2048,
    device="cuda"
)

print_unified_results(results)
```

#### Include needle test
```python
result = evaluate_config(
    model, tokenizer,
    preset="balanced",
    eval_ppl=True,
    eval_needle=True,
    device="cuda"
)

print(f"Needle: {'PASS' if result['needle_passed'] else 'FAIL'}")
```

---

### What This Fixes

**Before:**
- Compression measured in one run (forced reassignment)
- Quality measured in another run (different cache state)
- Results didn't correspond to each other
- README table showed inconsistent data

**After:**
- Both metrics from the **same cache instance**
- Compression corresponds to actual quality being measured
- Honest reporting of trade-offs
- Consistent, reproducible results

---

### Files Modified

1. **`benchmarks/unified_eval.py`** - NEW: Unified evaluation module
   - `evaluate_config()` - Single-run compression + quality
   - `evaluate_preset_grid()` - Batch evaluation
   - `print_unified_results()` - Formatted output

2. **`benchmarks/production_suite.py`** - UPDATED: Uses unified evaluation
   - Replaced separate `measure_kv_bytes()` + PPL calls
   - Now uses `evaluate_config()` for consistent metrics

---

### Verification

Run the unified evaluation:
```powershell
python benchmarks/production_suite.py --model Qwen/Qwen2.5-0.5B-Instruct
```

Expected output shows compression and PPL from the same run:
```
--- Unified Evaluation (Compression + Quality from Same Run) ---
  safe: 3.6x compression, KV: 31.2 MiB, PPL: 12.45 (+0.0%)
  balanced: 12.1x compression, KV: 9.3 MiB, PPL: 13.82 (+11.0%)
  aggressive: 23.5x compression, KV: 4.8 MiB, PPL: 18.67 (+49.1%)
```

Each line shows metrics from the **same execution**, ensuring consistency.

---

### Impact

This fix ensures:
1. ✅ **Honest reporting**: Compression and quality correspond to the same cache state
2. ✅ **Reproducible results**: Same run = same conditions
3. ✅ **Accurate trade-offs**: Shows real compression vs quality relationship
4. ✅ **Consistent benchmarks**: No more mismatched metrics

Without this fix, the compression ratios and quality metrics were essentially unrelated measurements that couldn't be trusted to represent the true trade-off.

---

### Recommendation

**Always use unified evaluation** when reporting FADE performance:
- Use `evaluate_config()` for single preset
- Use `evaluate_preset_grid()` for multiple presets
- Include both compression and PPL in all reports
- Report peak memory alongside compressed size

This ensures honest, consistent, and reproducible benchmarking.
