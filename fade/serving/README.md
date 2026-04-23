# FADE Serving Adapters

## Overview
FADE's `TieredKVCache` is a drop-in `past_key_values` for the HuggingFace
`generate()` loop. For production serving, you'll want to integrate with a
proper inference framework. This directory contains stubs and documentation
for the two major frameworks.

## vLLM Integration
vLLM manages KV cache at the **block level** via its `BlockManager`. A FADE
integration would:

1. **Replace the flat block allocator** with a tiered allocator that tracks
   which blocks hold FP16 vs INT4 data.
2. **Hook the eviction/compression logic** into vLLM's `Scheduler.schedule()`
   call — after each batch step, run `reassign_tiers` on the cache blocks
   that exceed the recent window.
3. **Override the attention kernel** to handle mixed-precision K/V blocks
   (FP16 sinks/recent + INT4 middle) in a single fused call.

Entry point: subclass `vllm.core.block_manager.BlockSpaceManager` or
implement a `vllm.worker.cache_engine.CacheEngine` plugin.

## SGLang Integration
SGLang's RadixAttention tree can be extended with a custom cache policy.
The approach is similar:

1. **Annotate tree nodes** with their tier (FP16 / INT4 / evicted).
2. **Run tier reassignment** when the tree is pruned or when decode steps
   cross the `reassign_every` threshold.
3. **Mixed-precision attention** in SGLang's paged-attention kernel.

## Status
These adapters are not yet implemented. The stubs exist so the `[serving]`
extra is installable and the integration surface is documented. If you're
interested in contributing, see `CONTRIBUTING.md`.
