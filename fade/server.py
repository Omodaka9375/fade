"""FADE inference server: OpenAI-compatible /v1/chat/completions.

Wraps a HuggingFace model with TieredKVCache and automatic tier
reassignment. Drop-in replacement for OpenAI API clients.

Usage:
    fade-server --model Qwen/Qwen2.5-0.5B-Instruct --preset balanced --port 8000
    curl http://localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
"""

from __future__ import annotations

import argparse
import time
import uuid

import torch

# --- knobs (top of file) --------------------------------------------------- #
DEFAULT_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PORT: int = 8000
DEFAULT_PRESET: str = "safe"
DEFAULT_MAX_TOKENS: int = 512
REASSIGN_EVERY: int = 64

# Global state (populated by startup).
_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32
_config = None


def _build_app():
    """Build the FastAPI app. Import here to make the module importable without fastapi."""
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    app = FastAPI(title="FADE Inference Server", version="0.3.0")

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = ""
        messages: list[Message]
        max_tokens: int = Field(default=DEFAULT_MAX_TOKENS)
        temperature: float = 0.0
        stream: bool = False

    class Choice(BaseModel):
        index: int = 0
        message: Message
        finish_reason: str = "stop"

    class Usage(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class ChatResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: list[Choice]
        usage: Usage

    @app.get("/v1/models")
    def list_models():
        return {"data": [{"id": _config["model_id"], "object": "model"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": _config["model_id"], "preset": _config["preset"]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        from fade import FadeConfig, create_tiered_cache
        from fade.patch import forward_with_tracking
        from fade.policy import reassign_tiers_by_position
        from fade.tracker import AttentionTracker

        # Build prompt from messages.
        if _tokenizer is None or _model is None:
            raise RuntimeError("Server not initialized. Call main() first.")
        if hasattr(_tokenizer, "apply_chat_template"):
            prompt = _tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in req.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)

        enc = _tokenizer(prompt, return_tensors="pt").to(_device)
        input_ids = enc.input_ids
        prompt_len = input_ids.shape[1]

        # Create tiered cache.
        preset_fn = getattr(FadeConfig, _config["preset"], FadeConfig.safe)
        fade_config = preset_fn()
        if fade_config.eviction_policy == "h2o":
            fade_config = fade_config.with_overrides(eviction_policy="position")

        cache = create_tiered_cache(_model, dtype=_dtype, config=fade_config)
        num_layers = _model.config.num_hidden_layers
        tracker = AttentionTracker(num_layers=num_layers)

        # Prefill.
        out = forward_with_tracking(_model, input_ids, cache, tracker=tracker)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated = [next_token]

        # Decode.
        max_new = min(req.max_tokens, DEFAULT_MAX_TOKENS)
        for step in range(max_new - 1):
            out = forward_with_tracking(_model, next_token, cache, tracker=tracker)

            if req.temperature > 0:
                probs = torch.softmax(out.logits[:, -1, :] / req.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = out.logits[:, -1:, :].argmax(dim=-1)

            generated.append(next_token)

            if (step + 1) % REASSIGN_EVERY == 0:
                reassign_tiers_by_position(cache, num_layers)

            if _tokenizer.eos_token_id is not None and next_token.item() == _tokenizer.eos_token_id:
                break

        gen_ids = torch.cat(generated, dim=-1)
        text = _tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        comp_tokens = gen_ids.shape[1]

        # KV stats.
        cache.compressed_storage_bytes() / (1024 * 1024)

        return ChatResponse(
            id=f"fade-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=_config["model_id"],
            choices=[Choice(message=Message(role="assistant", content=text))],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=comp_tokens,
                total_tokens=prompt_len + comp_tokens,
            ),
        )

    return app


def main():
    global _model, _tokenizer, _config

    parser = argparse.ArgumentParser(description="FADE inference server")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--preset", type=str, default=DEFAULT_PRESET, choices=["safe", "balanced", "aggressive"]
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    _config = {"model_id": args.model, "preset": args.preset}

    print(f"Loading {args.model} on {_device}...")
    from fade.patch import load_model

    _model, _tokenizer = load_model(args.model, device_map=_device, dtype=_dtype, attn_impl="eager")
    print(f"Model loaded. Preset: {args.preset}")

    app = _build_app()

    import uvicorn

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
