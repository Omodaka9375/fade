
## Production Benchmark Summary

### Mistral-7B-Instruct-v0.3

| Config | KV Cache | Compression | Decode TPS |
|--------|----------|:-----------:|:----------:|
| Baseline FP16 | 256.0 MiB | **1.0x** | 15.3 tok/s |
| Safe (INT4) | 71.4 MiB | **3.6x** | 15.2 tok/s |
| Balanced | 21.26 MiB | **12.0x** | 15.2 tok/s |
| Aggressive | 10.91 MiB | **23.5x** | 15.2 tok/s |
| Rotated 2-bit | 40.47 MiB | **6.3x** | 15.2 tok/s |

Needle: @512: ✅, @1024: ✅, @2048: ✅, @4096: ✅
WikiText-2 PPL (baseline FP16): 4.9827
