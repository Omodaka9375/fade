
## Production Benchmark Summary

### Qwen2.5-0.5B-Instruct

| Config | KV Cache | Compression | Decode TPS |
|--------|----------|:-----------:|:----------:|
| Baseline FP16 | 24.0 MiB | **1.0x** | 128.5 tok/s |
| Safe (INT4) | 6.78 MiB | **3.5x** | 125.8 tok/s |
| Balanced | 2.01 MiB | **11.9x** | 125.8 tok/s |
| Aggressive | 1.03 MiB | **23.3x** | 125.9 tok/s |
| Rotated 2-bit | 3.88 MiB | **6.2x** | 125.9 tok/s |

Needle: @512: ✅, @1024: ✅, @2048: ✅, @4096: ✅
