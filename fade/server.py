"""FADE inference server: OpenAI-compatible /v1/chat/completions.

Features:
    - SSE streaming (``stream: true``)
    - KV cache reuse across chat turns (``X-Session-Id`` header)
    - Async decode (non-blocking inference via ``asyncio.to_thread``)
    - Conditional H2O downgrade (only when prompt exceeds track limit)
    - Automatic tier reassignment every ``REASSIGN_EVERY`` steps

Usage:
    fade-server --model Qwen/Qwen2.5-0.5B-Instruct --preset balanced --port 8000
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":100}'

    # Streaming:
    curl -N http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"Hi"}],"stream":true}'

    # Session reuse (multi-turn):
    curl http://localhost:8000/v1/chat/completions \\
      -H "X-Session-Id: my-session" \\
      -d '{"messages":[{"role":"user","content":"Hi"}]}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

# --- knobs (top of file for easy override) ---------------------------------- #
DEFAULT_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PORT: int = 8000
DEFAULT_PRESET: str = "safe"
DEFAULT_MAX_TOKENS: int = 512
REASSIGN_EVERY: int = 64
SESSION_MAX_ENTRIES: int = 64
SESSION_TTL_SECONDS: float = 600.0

logger = logging.getLogger("fade.server")

# Global state (populated by startup).
_model = None
_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device == "cuda" else torch.float32
_config: dict[str, Any] = {}


# --- session store (D1) ---------------------------------------------------- #
@dataclass
class Session:
    """Holds a reusable KV cache and tracker for multi-turn chat."""

    cache: Any = None
    tracker: Any = None
    input_ids: torch.Tensor | None = None
    last_access: float = field(default_factory=time.time)


class SessionStore:
    """LRU-bounded session store with TTL eviction."""

    def __init__(self, max_entries: int = SESSION_MAX_ENTRIES, ttl: float = SESSION_TTL_SECONDS):
        self._store: OrderedDict[str, Session] = OrderedDict()
        self.max_entries = max_entries
        self.ttl = ttl

    def get(self, session_id: str) -> Session | None:
        """Retrieve session, evicting stale entries first."""
        self._evict_stale()
        session = self._store.get(session_id)
        if session is not None:
            session.last_access = time.time()
            self._store.move_to_end(session_id)
        return session

    def put(self, session_id: str, session: Session) -> None:
        """Insert or update a session, enforcing LRU cap."""
        session.last_access = time.time()
        self._store[session_id] = session
        self._store.move_to_end(session_id)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def _evict_stale(self) -> None:
        now = time.time()
        stale = [k for k, v in self._store.items() if now - v.last_access > self.ttl]
        for k in stale:
            del self._store[k]

    def __len__(self) -> int:
        return len(self._store)


_sessions = SessionStore()


# --- decode engine ---------------------------------------------------------- #
def _resolve_fade_config():
    """Build FadeConfig from the server preset."""
    from fade import FadeConfig

    preset_fn = getattr(FadeConfig, _config.get("preset", "safe"), FadeConfig.safe)
    return preset_fn()


def _maybe_downgrade_h2o(fade_config, prompt_len: int):
    """Conditional H2O downgrade (D4): only when prompt exceeds track limit."""
    if fade_config.eviction_policy == "h2o" and prompt_len > fade_config.prefill_track_limit:
        logger.info(
            "Prompt length %d exceeds prefill_track_limit %d; "
            "downgrading H2O to position eviction.",
            prompt_len,
            fade_config.prefill_track_limit,
        )
        return fade_config.with_overrides(eviction_policy="position")
    return fade_config


def _build_prompt(messages: list[dict]) -> str:
    """Build prompt string from messages using chat template if available."""
    if hasattr(_tokenizer, "apply_chat_template"):
        return _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


@torch.no_grad()
def _generate_sync(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    session_id: str | None,
):
    """Synchronous non-streaming generation.

    Returns:
        (text, prompt_len, comp_tokens, finish_reason)
    """
    from fade import create_tiered_cache
    from fade.patch import forward_with_tracking
    from fade.policy import reassign_tiers_by_position
    from fade.tracker import AttentionTracker

    if _tokenizer is None or _model is None:
        raise RuntimeError("Server not initialized.")

    prompt = _build_prompt(messages)
    enc = _tokenizer(prompt, return_tensors="pt").to(_device)
    input_ids = enc.input_ids
    prompt_len = input_ids.shape[1]
    num_layers = _model.config.num_hidden_layers

    # D1: Session reuse.
    session = _sessions.get(session_id) if session_id else None
    if session is not None and session.cache is not None:
        cache = session.cache
        tracker = session.tracker
    else:
        fade_config = _resolve_fade_config()
        fade_config = _maybe_downgrade_h2o(fade_config, prompt_len)
        cache = create_tiered_cache(_model, dtype=_dtype, config=fade_config)
        tracker = AttentionTracker(num_layers=num_layers)

    # Prefill.
    out = forward_with_tracking(_model, input_ids, cache, tracker=tracker)
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_token]
    finish_reason = "stop"

    # Decode.
    max_new = min(max_tokens, DEFAULT_MAX_TOKENS)
    for step in range(max_new - 1):
        out = forward_with_tracking(_model, next_token, cache, tracker=tracker)

        if temperature > 0:
            probs = torch.softmax(out.logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        generated.append(next_token)

        if (step + 1) % REASSIGN_EVERY == 0:
            reassign_tiers_by_position(cache, num_layers)

        if _tokenizer.eos_token_id is not None and next_token.item() == _tokenizer.eos_token_id:
            break
    else:
        finish_reason = "length"

    gen_ids = torch.cat(generated, dim=-1)
    text = _tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    comp_tokens = gen_ids.shape[1]

    # D1: Store session.
    if session_id:
        _sessions.put(session_id, Session(cache=cache, tracker=tracker, input_ids=input_ids))

    return text, prompt_len, comp_tokens, finish_reason


@torch.no_grad()
def _generate_stream_chunks(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    session_id: str | None,
):
    """Streaming generator (D2). Yields (token_text, finish_reason, prompt_len, step)."""
    from fade import create_tiered_cache
    from fade.patch import forward_with_tracking
    from fade.policy import reassign_tiers_by_position
    from fade.tracker import AttentionTracker

    if _tokenizer is None or _model is None:
        raise RuntimeError("Server not initialized.")

    prompt = _build_prompt(messages)
    enc = _tokenizer(prompt, return_tensors="pt").to(_device)
    input_ids = enc.input_ids
    prompt_len = input_ids.shape[1]
    num_layers = _model.config.num_hidden_layers

    session = _sessions.get(session_id) if session_id else None
    if session is not None and session.cache is not None:
        cache = session.cache
        tracker = session.tracker
    else:
        fade_config = _resolve_fade_config()
        fade_config = _maybe_downgrade_h2o(fade_config, prompt_len)
        cache = create_tiered_cache(_model, dtype=_dtype, config=fade_config)
        tracker = AttentionTracker(num_layers=num_layers)

    out = forward_with_tracking(_model, input_ids, cache, tracker=tracker)
    next_token = out.logits[:, -1:, :].argmax(dim=-1)

    max_new = min(max_tokens, DEFAULT_MAX_TOKENS)
    for step in range(max_new):
        token_text = _tokenizer.decode(next_token[0], skip_special_tokens=True)

        if _tokenizer.eos_token_id is not None and next_token.item() == _tokenizer.eos_token_id:
            yield token_text, "stop", prompt_len, step + 1
            break

        yield token_text, None, prompt_len, step + 1

        out = forward_with_tracking(_model, next_token, cache, tracker=tracker)
        if temperature > 0:
            probs = torch.softmax(out.logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        if (step + 1) % REASSIGN_EVERY == 0:
            reassign_tiers_by_position(cache, num_layers)
    else:
        yield "", "length", prompt_len, max_new

    if session_id:
        _sessions.put(session_id, Session(cache=cache, tracker=tracker, input_ids=input_ids))


# --- FastAPI app ------------------------------------------------------------ #
def _build_app():
    """Build the FastAPI app. Import here so module is importable without fastapi."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="FADE Inference Server", version="0.9.0")

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
        return {
            "status": "ok",
            "model": _config["model_id"],
            "preset": _config["preset"],
            "sessions": len(_sessions),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        session_id = request.headers.get("x-session-id")
        req_id = f"fade-{uuid.uuid4().hex[:8]}"

        if req.stream:
            # D2: SSE streaming response.
            async def event_stream():
                chunks = await asyncio.to_thread(
                    lambda: list(
                        _generate_stream_chunks(
                            messages, req.max_tokens, req.temperature, session_id
                        )
                    )
                )
                for token_text, finish, _prompt_len, _comp in chunks:
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _config["model_id"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_text} if finish is None else {},
                                "finish_reason": finish,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # D3: Non-streaming async decode.
        text, prompt_len, comp_tokens, finish_reason = await asyncio.to_thread(
            _generate_sync, messages, req.max_tokens, req.temperature, session_id
        )

        return ChatResponse(
            id=req_id,
            created=int(time.time()),
            model=_config["model_id"],
            choices=[
                Choice(
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_len,
                completion_tokens=comp_tokens,
                total_tokens=prompt_len + comp_tokens,
            ),
        )

    return app


def main():
    """Entry point for ``fade-server`` CLI."""
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

    _model, _tokenizer = load_model(args.model, device_map=_device, dtype=_dtype, attn_impl="sdpa")
    print(f"Model loaded. Preset: {args.preset}")

    app = _build_app()

    import uvicorn

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
