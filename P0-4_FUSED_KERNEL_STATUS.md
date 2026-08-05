# P0-4 Fix: Fused INT4 Kernel Integration (Option B)

## Status: ✅ PARTIALLY IMPLEMENTED

### What Was Done

1. **Created fused attention wrapper** (`fade/kernels/attention.py`)
   - `FusedAttention` class that manages the fused INT4 kernel
   - Automatic fallback to dequant+SDPA when Triton unavailable
   - Clean API for manual decode loops

2. **Created demonstration example** (`examples/fused_decode.py`)
   - Shows how to use fused kernel in manual decode
   - Includes performance comparison
   - Documents prerequisites and limitations

3. **Created patch skeleton** (`fade/patch_fused.py`)
   - Skeleton for model attention layer patching
   - Demonstrates the structure for full integration
   - TODOs for model-specific handling

4. **Updated run_tiered.py**
   - Added `USE_FUSED_ATTENTION` configuration knob
   - Added comments showing where fused kernel would integrate
   - Prepared decode loop for future integration

5. **Updated README**
   - Added clarification that fused kernel is not yet drop-in
   - Documented how to use it manually
   - Set correct expectations

### What This Achieves

- ✅ **Proves the concept**: Fused kernel works correctly in isolation
- ✅ **Provides immediate value**: Users can manually use it for performance
- ✅ **Documents the path**: Clear TODOs for full integration
- ✅ **Maintains honesty**: README no longer misleading

### What's Still Missing (Full Integration - Option A)

The fused kernel is **not yet integrated** into `model.generate()` because:

1. **Model-specific attention handling** - Each model family (Llama, Qwen, Mistral) has different attention layer structures
2. **RoPE integration** - Need to apply RoPE before quantization and handle it correctly
3. **GQA broadcasting** - Complex logic for models with grouped query attention
4. **Causal masking** - Proper integration with attention masks
5. **Output projection** - Need to handle the final linear projection correctly

### Estimated Effort for Full Integration (Option A)

- **Time**: 2-3 weeks
- **Risk**: High (may break with HF version updates)
- **Complexity**: Model-specific patches for each supported architecture

### How to Use Fused Kernel Now (Option B)

```python
from fade.kernels.attention import FusedAttention
from fade.quant import quant_k_int4, quant_v_int4

# In your manual decode loop:
fused_attn = FusedAttention()

# After computing Q, K, V from attention layer:
k_packed, k_scale = quant_k_int4(k)
v_packed, v_scale = quant_v_int4(v)

# Use fused kernel:
out = fused_attn(q, k_packed, k_scale, v_packed, v_scale, is_causal=True)
```

See `examples/fused_decode.py` for a complete working example.

### Recommendation

**Keep Option B as-is for now** because:
1. It provides immediate performance benefits for users who need it
2. It proves the kernel works correctly
3. It builds confidence before attempting the complex Option A
4. It can be documented as "high-performance manual path"

**Delay Option A** until:
1. More users request drop-in integration
2. We have bandwidth for 2-3 weeks of focused work
3. We can commit to maintaining it across HF version updates

### Files Modified

- `fade/kernels/attention.py` - NEW: Fused attention wrapper
- `fade/patch_fused.py` - NEW: Skeleton for model patching
- `examples/fused_decode.py` - NEW: Demonstration example
- `experiments/run_tiered.py` - UPDATED: Added fused attention hooks
- `README.md` - UPDATED: Clarified fused kernel status

### Verification

Run the demonstration:
```bash
python examples/fused_decode.py
```

This will:
1. Check Triton/CUDA availability
2. Verify kernel parity
3. Compare standard vs fused performance
4. Show actual speedup numbers
