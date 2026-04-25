
## Production Benchmark Summary

### Qwen2.5-7B-Instruct

| Config | KV Cache | Compression | Decode TPS |
|--------|----------|:-----------:|:----------:|
| Baseline FP16 | 112.0 MiB | **1.0x** | 13.3 tok/s |
| Safe (INT4) | 31.24 MiB | **3.6x** | 13.3 tok/s |
| Balanced | 9.3 MiB | **12.0x** | 13.3 tok/s |
| Aggressive | 4.77 MiB | **23.5x** | 13.3 tok/s |
| Rotated 2-bit | 17.7 MiB | **6.3x** | 13.3 tok/s |

Needle: @512: ✅, @1024: ✅, @2048: ✅, @4096: ✅
WikiText-2 PPL (baseline FP16): 6.5593
