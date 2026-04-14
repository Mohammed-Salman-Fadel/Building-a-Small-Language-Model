from __future__ import annotations

import chainlit

from .chat_runtime import ChatRuntime, describe_runtime, generate_chat_response, load_chat_runtime


_RUNTIME: ChatRuntime | None = None


def get_runtime() -> ChatRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = load_chat_runtime(mode="auto")
    return _RUNTIME


@chainlit.on_chat_start
async def on_chat_start() -> None:
    chainlit.user_session.set("history", [])
    runtime = get_runtime()
    await chainlit.Message(content=describe_runtime(runtime)).send()


@chainlit.on_message
async def on_message(message: chainlit.Message) -> None:
    runtime = get_runtime()
    history = chainlit.user_session.get("history") or []
    response, updated_history = generate_chat_response(runtime, history, message.content)
    chainlit.user_session.set("history", updated_history)
    await chainlit.Message(content=response).send()
