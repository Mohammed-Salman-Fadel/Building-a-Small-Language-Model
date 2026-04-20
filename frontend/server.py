from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.chat_runtime import describe_runtime, generate_chat_response, load_chat_runtime
from small_llm.config import DEFAULT_CHAT_GENERATION


class ChatTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=0, le=200)
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)


class ChatResponse(BaseModel):
    response: str
    history: list[ChatTurn]
    runtime: dict[str, Any]


app = FastAPI(title="Small LLM Frontend", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_runtime_lock = threading.Lock()
_runtime = None


def _runtime_config() -> dict[str, str | None]:
    return {
        "mode": os.getenv("SLM_FRONTEND_MODE", "auto"),
        "checkpoint_path": os.getenv("SLM_FRONTEND_CHECKPOINT") or None,
        "device_name": os.getenv("SLM_FRONTEND_DEVICE", "auto"),
    }


def _runtime_summary(runtime: Any) -> dict[str, Any]:
    checkpoint_path = str(runtime.checkpoint_path) if runtime.checkpoint_path is not None else None
    return {
        "mode": runtime.mode,
        "device": str(runtime.device),
        "checkpoint_path": checkpoint_path,
        "context_length": runtime.model_config.context_length,
        "description": describe_runtime(runtime),
    }


def _get_runtime() -> Any:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = load_chat_runtime(**_runtime_config())
        return _runtime


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    runtime = _get_runtime()
    return {
        "runtime": _runtime_summary(runtime),
        "defaults": {
            "temperature": DEFAULT_CHAT_GENERATION.temperature,
            "top_k": DEFAULT_CHAT_GENERATION.top_k,
            "max_new_tokens": DEFAULT_CHAT_GENERATION.max_new_tokens,
        },
    }


@app.post("/api/reload")
def reload_model() -> dict[str, Any]:
    global _runtime
    with _runtime_lock:
        _runtime = None
    runtime = _get_runtime()
    return {"runtime": _runtime_summary(runtime)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    runtime = _get_runtime()
    generation_config = replace(
        runtime.generation_config,
        temperature=runtime.generation_config.temperature
        if request.temperature is None
        else request.temperature,
        top_k=runtime.generation_config.top_k if request.top_k is None else request.top_k,
        max_new_tokens=runtime.generation_config.max_new_tokens
        if request.max_new_tokens is None
        else request.max_new_tokens,
    )
    request_runtime = replace(runtime, generation_config=generation_config)
    history = [{"user": turn.user, "assistant": turn.assistant} for turn in request.history]

    try:
        response, updated_history = generate_chat_response(request_runtime, history, request.message.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model generation failed: {exc}") from exc

    return ChatResponse(
        response=response,
        history=[ChatTurn(**turn) for turn in updated_history],
        runtime=_runtime_summary(runtime),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
