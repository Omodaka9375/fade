# P1-1 Fix: Memory Accounting Honesty

## Status: ✅ FIXED

### What Was Wrong

The audit found that FADE's memory accounting was misleading:

1. **Default `cache_dequant=True`** kept full FP16 copies of compressed data in memory
2. **`compressed_storage_bytes()`** only showed compressed size, hiding the actual GPU memory usage
3. **No `resident_bytes()`** method to show true memory footprint
4. **README told users** to trust `compressed_storage_bytes()` over `nvidia-smi` without explaining the difference

**Result**: Users saw "3.6× compression" in the docs but their GPU showed almost no memory savings because dequant caches were doubling the actual memory usage.

---

### What Was Fixed

#### 1. **Changed Defaults** (`fade/cache.py`)

```python
# Before:
DEFAULT_CACHE_DEQUANT: bool = True
DEFAULT_MAX_DEQUANT_AGE: int | None = None

# After:
DEFAULT_CACHE_DEQUANT: bool = False  # Prioritize honest memory savings
DEFAULT_MAX_DEQUANT_AGE: int = 64  # Auto-drop dequant after 64 updates
```

**Impact**: New users now get honest memory accounting by default. Dequant caches are dropped automatically to prevent memory bloat.

#### 2. **Added `resident_bytes()` Method**

```python
def resident_bytes(self) -> int:
    """Total bytes currently held in GPU/CPU memory, including ALL overhead.
    
    Includes:
        - Compressed K/V (INT4/INT2/PQ packed data)
        - Scales for quantization
        - Dequantized caches (if cache_dequant=True)
        - FP16 sink and recent window tensors
        - Pre-allocated buffer overhead
        - Position tensors for each tier
    """
```

**What it counts**:
- ✅ Compressed storage (INT4/INT2 packed data + scales)
- ✅ Dequant caches (when enabled)
- ✅ Position tensors (often overlooked)
- ✅ Pre-allocated buffer overhead (full capacity, not just used portion)

#### 3. **Added `memory_breakdown()` Method**

```python
def memory_breakdown(self) -> dict:
    """Return a detailed breakdown of memory usage.
    
    Returns:
        {
            "compressed_bytes": 1048576,      # Compressed K/V only
            "dequant_bytes": 4194304,         # Cached FP16 copies
            "position_bytes": 32768,          # Position tracking
            "buffer_overhead_bytes": 524288,  # Pre-alloc slack
            "resident_bytes": 5247040,        # Total
        }
    """
```

**Use case**: Debugging memory usage, understanding where bytes are spent.

#### 4. **Updated Documentation**

**README Gotchas section now says**:

> **Memory accounting**:
> - `cache.compressed_storage_bytes()` — at-rest size (compressed data only). This is the "on-disk" footprint.
> - `cache.resident_bytes()` — actual GPU memory usage (includes dequant caches, buffer overhead). This matches `nvidia-smi`.
> - Default `cache_dequant=False` prioritizes honest memory savings. Set to `True` for speed if you have memory to spare.

---

### How to Use

#### Get Compressed Size (At-Rest)
```python
cache = create_tiered_cache(model, config=FadeConfig.balanced())
# ... run inference ...
at_rest_bytes = cache.compressed_storage_bytes()
print(f"At-rest: {at_rest_bytes / 1024 / 1024:.2f} MiB")
```

#### Get Actual GPU Memory Usage
```python
resident_bytes = cache.resident_bytes()
print(f"Resident: {resident_bytes / 1024 / 1024:.2f} MiB")
# This should match torch.cuda.max_memory_allocated()
```

#### Get Detailed Breakdown
```python
breakdown = cache.memory_breakdown()
print(f"Compressed:    {breakdown['compressed_bytes'] / 1024 / 1024:.2f} MiB")
print(f"Dequant cache: {breakdown['dequant_bytes'] / 1024 / 1024:.2f} MiB")
print(f"Positions:     {breakdown['position_bytes'] / 1024 / 1024:.4f} MiB")
print(f"Buffer overhead: {breakdown['buffer_overhead_bytes'] / 1024 / 1024:.2f} MiB")
print(f"Total resident: {breakdown['resident_bytes'] / 1024 / 1024:.2f} MiB")
```

#### Control Dequant Caching
```python
# Prioritize memory savings (default)
cache = create_tiered_cache(model, config=FadeConfig.balanced(), cache_dequant=False)

# Prioritize speed (uses more memory)
cache = create_tiered_cache(model, config=FadeConfig.balanced(), cache_dequant=True)

# Auto-drop dequant after 32 updates
cache = create_tiered_cache(model, config=FadeConfig.balanced(), max_dequant_age=32)
```

---

### Verification

To verify the fix works:

```python
import torch
from fade import FadeConfig, create_tiered_cache
from fade.patch import load_model

# Load model
model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct", device_map="cuda")

# Create cache with default settings (cache_dequant=False)
cache = create_tiered_cache(model, config=FadeConfig.balanced())

# Run some inference
input_ids = tokenizer("Test prompt", return_tensors="pt").to("cuda")
out = model(input_ids, past_key_values=cache, use_cache=True)

# Compare memory metrics
print(f"Compressed:  {cache.compressed_storage_bytes() / 1024 / 1024:.2f} MiB")
print(f"Resident:    {cache.resident_bytes() / 1024 / 1024:.2f} MiB")
print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MiB")

# With dequant enabled (more memory)
cache2 = create_tiered_cache(model, config=FadeConfig.balanced(), cache_dequant=True)
out = model(input_ids, past_key_values=cache2, use_cache=True)

print(f"\nWith dequant=True:")
print(f"Compressed:  {cache2.compressed_storage_bytes() / 1024 / 1024:.2f} MiB")
print(f"Resident:    {cache2.resident_bytes() / 1024 / 1024:.2f} MiB")

# Show breakdown
breakdown = cache2.memory_breakdown()
print(f"\nBreakdown:")
for k, v in breakdown.items():
    print(f"  {k}: {v / 1024 / 1024:.2f} MiB")
```

**Expected output**:
- `compressed_storage_bytes()` shows the theoretical minimum
- `resident_bytes()` should be close to `torch.cuda.memory_allocated()`
- With `cache_dequant=True`, resident will be significantly higher than compressed

---

### Files Modified

1. **`fade/cache.py`**:
   - Changed `DEFAULT_CACHE_DEQUANT` from `True` to `False`
   - Changed `DEFAULT_MAX_DEQUANT_AGE` from `None` to `64`
   - Updated `__init__` signature to accept `max_dequant_age` parameter
   - Added `resident_bytes()` method
   - Added `memory_breakdown()` method

2. **`README.md`**:
   - Updated Gotchas section to explain memory accounting
   - Clarified difference between compressed vs resident memory
   - Documented `max_dequant_age` behavior

---

### Impact

**Before**: Users saw "3.6× compression" but their GPU showed almost no savings because dequant caches were doubling memory usage.

**After**: 
- Default behavior shows honest memory savings
- `resident_bytes()` gives accurate GPU memory footprint
- `memory_breakdown()` helps debug memory usage
- Auto-expiring dequant caches prevent memory bloat

**Trade-off**: Slightly slower inference when `cache_dequant=False` (must dequant on each access), but honest memory accounting and actual memory savings.

---

### Recommendation

**Keep the new defaults** because:
1. Honesty is critical for a compression library
2. Users can opt-in to `cache_dequant=True` if they need speed
3. `max_dequant_age=64` provides a safety net even when caching is enabled
4. Documentation now clearly explains the difference

This fix aligns with the audit's goal of making FADE's claims verifiable and honest.
