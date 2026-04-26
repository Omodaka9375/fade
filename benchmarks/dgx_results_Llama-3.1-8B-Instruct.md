
## Production Benchmark Summary

### Llama-3.1-8B-Instruct

| Config | KV Cache | Compression | Decode TPS |
|--------|----------|:-----------:|:----------:|
| Baseline FP16 | 256.0 MiB | **1.0x** | 14.4 tok/s |
| Safe (INT4) | 71.4 MiB | **3.6x** | 14.3 tok/s |
| Balanced | 21.26 MiB | **12.0x** | 14.3 tok/s |
| Aggressive | 10.91 MiB | **23.5x** | 14.3 tok/s |
| Rotated 2-bit | 40.47 MiB | **6.3x** | 14.3 tok/s |

Needle: @512: ✅, @1024: ✅, @2048: ✅, @4096: ✅
WikiText-2 PPL (baseline FP16): 6.4508
