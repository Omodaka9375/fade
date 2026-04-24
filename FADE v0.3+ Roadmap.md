# FADE v0.3 Roadmap
What to build next, ordered by impact. Each item addresses a specific gap identified against the 2026 KV compression landscape.
## Phase A: Close the compression gap (weeks 1-3)
FADE's INT4 symmetric quant gives ~3-4×. Competitors (TurboQuant, KVTC) hit 6-9× with better math. Fix this without rewriting the tier system.
### A1. Pluggable quantization backend
Current: `quant.py` hardcodes symmetric INT4/INT2 with per-channel K / per-token V scales.
Target: `QuantBackend` protocol with `compress(tensor) -> CompressedTensor` and `decompress(ct) -> Tensor`. Ship three backends:
* `SymmetricINT4` (current, default)
* `TurboQuantBackend` (wraps `turboquant-kv` if installed, gated by `pip install fade-kv[turbo]`)
* `KVTCBackend` (wraps KVTC's PCA+DP pipeline if installed, gated by `pip install fade-kv[kvtc]`)
Wire into `FadeConfig(quant_backend="turbo")` and `apply_tier_assignment`. The tier system stays the same; only the compress/decompress step changes.
Acceptance: TurboQuant backend produces 5-6× compression on Qwen-0.5B with <2% PPL increase vs baseline.
### A2. Fused Triton attention kernel (real)
Current: Triton kernel unpacks INT4 → fp16, then calls `F.scaled_dot_product_attention`. Two memory round-trips.
Target: Single Triton kernel that reads packed INT4 K/V, computes QK^T with online softmax, accumulates V, writes fp16 output. Never materializes the full fp16 K/V buffer.
Approach: Port the FlashAttention-2 tiling pattern but replace the K/V tile loads with inline INT4 unpack. Block over S_k in tiles of 64-128.
Acceptance: Within 5% TPS of FlashAttention-2 on fp16 K/V at S_k=2048, D=128 on RTX 3060. Numeric parity <1e-3 max error vs torch fallback.
### A3. Improve PQ codebook quality
Current: MiniBatchKMeans with fixed 50 iterations, no residual correction.
Target: Add a residual quantization pass (encode, compute residual, encode residual with a second codebook). This is standard in production PQ systems and typically halves reconstruction error.
Acceptance: PQ reconstruction error < INT2 at the same bits-per-element on the Qwen-0.5B calibration set.
## Phase B: Production serving integration (weeks 3-5)
FADE can't be used in vLLM/SGLang today. Fix this.
### B1. vLLM block-manager plugin
Implement a `FadeBlockManager` that:
* Allocates blocks with tier metadata (FP16 / INT4 / PQ)
* Runs `reassign_tiers` when block utilization crosses a threshold
* Handles the re-RoPE path inside the block manager's eviction callback
* Exposes as a vLLM engine arg: `--kv-cache-impl fade --fade-preset balanced`
Acceptance: `vllm serve Qwen/Qwen2.5-0.5B-Instruct --kv-cache-impl fade` runs the OpenAI-compatible API with tiered compression.
### B2. SGLang RadixAttention adapter
Same approach for SGLang's tree-based cache. Annotate tree nodes with tier labels. Run tier reassignment on prune.
### B3. OpenAI-compatible inference server (standalone)
A lightweight FastAPI server (like TurboQuant's `turboquant-server`) that wraps FADE:
```warp-runnable-command
fade-server --model Qwen/Qwen2.5-3B-Instruct --preset balanced --port 8000
```
Handles `/v1/chat/completions` with automatic tier management. Easier adoption than requiring vLLM.
## Phase C: Quality and evaluation (weeks 4-6)
FADE has no published quality numbers on standard benchmarks. This blocks adoption.
### C1. Standard benchmark suite
Run and publish results on:
* **LongBench v2** (16 tasks)
* **RULER** (13 tasks, 4K-128K)
* **Needle-in-a-haystack** at 2K/4K/8K/16K/32K
* **HumanEval / MBPP** pass@1 (code is compression-sensitive)
* **MMLU-Pro** (general knowledge)
For each: baseline FP16 vs safe vs balanced vs aggressive. Publish as a table in the README and as JSON artifacts in the repo.
Acceptance: Results published for Qwen2.5-0.5B, Qwen2.5-3B, Llama-3.1-8B on at least LongBench + RULER.
### C2. Quality regression CI
The nightly CI currently checks perplexity on a small text. Upgrade to:
* Download Qwen2.5-0.5B
* Run RULER single-needle at 2K
* Run HumanEval 20-problem subset
* Fail if quality drops >2% absolute vs cached baseline
Runs on every PR (small subset) and nightly (full suite).
### C3. Compression-quality Pareto frontier visualization
Script that sweeps `int4_budget` from 100 to unlimited, measures PPL + needle pass rate, and plots the Pareto curve. Publish as an SVG in the repo.
## Phase D: Architecture coverage (weeks 5-7)
### D1. Full Qwen 3.5/3.6 M-RoPE support
Current: `skip_layers` works for DeltaNet passthrough, but the full-attention layers use M-RoPE (3D interleaved positions) which FADE's re-RoPE doesn't handle.
Target: Add `MRope` scheme to `rope.py` that handles the 3D position IDs (temporal, height, width) and interleaved layout. Wire into `_compute_cos_sin` with a `position_ids` override.
Acceptance: Qwen 3.5-0.8B (if available as text-only) runs end-to-end with tier reassignment.
### D2. DeepSeek-V3 MLA support
DeepSeek V3 uses Multi-head Latent Attention where K/V are low-rank projections, not full-dim. The KV cache stores compressed latent vectors. FADE needs to detect this and either skip compression (already compressed) or apply PQ on the latent space.
### D3. Mamba / RWKV / linear attention models
For pure-recurrent models (Mamba, RWKV, xLSTM), there's no K/V cache to compress. FADE should gracefully detect `model_type` and return a plain `DynamicCache` with a warning.
## Phase E: Research directions (ongoing)
### E1. Attention-aware quantization
Current: INT4 quant is uniform (same scale across all middle tokens). But tokens with high attention mass deserve better precision.
Idea: Adaptive per-token bit allocation based on the H2O score. High-attention tokens get INT4; low-attention tokens get INT2 or PQ. This is the FADE-native version of KVTC's DP bit allocation.
### E2. Speculative eviction
Predict which tokens will be evicted at the next reassignment and pre-compress them during decode. Amortizes the reassignment cost.
### E3. Cross-layer weight sharing for PQ codebooks
Current: one codebook per (layer, head). But adjacent layers often have similar K/V distributions. Share codebooks across groups of 4 layers to reduce codebook memory.
### E4. Integration with TurboQuant's random rotation
TurboQuant's key insight is that random orthogonal rotation + optimal codebook beats naive symmetric quant. Add a `RotatedINT4` backend that applies a random rotation before quantization. This is a small change (one matmul per quant/dequant) with potentially large quality improvement.
## Versioning
* **v0.2.x**: bug fixes and CI improvements on the current codebase
* **v0.3.0**: Phase A (pluggable backends + fused kernel)
* **v0.4.0**: Phase B (vLLM/SGLang integration)
* **v0.5.0**: Phase C (published benchmarks) + Phase D (architecture coverage)
* **v1.0.0**: all phases complete, battle-tested in production serving
