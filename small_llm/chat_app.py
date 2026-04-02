from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chainlit
import torch

from .config import DEFAULT_CHAT_GENERATION, GenerationConfig, ModelConfig
from .generation import extract_response_text, generate, get_tokenizer, token_ids_to_text
from .instruction_data import format_input
from .paths import (
    DEFAULT_GUTENBERG_BEST,
    DEFAULT_GUTENBERG_LATEST,
    get_base_checkpoint_candidates,
    get_chat_checkpoint_candidates,
    resolve_first_existing,
)
from .training import load_model_and_metadata


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ChatRuntime:
    model: torch.nn.Module
    model_config: ModelConfig
    tokenizer: Any
    checkpoint_path: Path
    mode: str
    generation_config: GenerationConfig


_RUNTIME: ChatRuntime | None = None


def _resolve_runtime_checkpoint() -> tuple[Path, str]:
    chat_checkpoint = resolve_first_existing(get_chat_checkpoint_candidates())
    if chat_checkpoint is not None:
        return chat_checkpoint, "chat"

    base_candidates = get_base_checkpoint_candidates()
    base_checkpoint = resolve_first_existing(base_candidates)
    if base_checkpoint is not None:
        return base_checkpoint, "base"

    raise FileNotFoundError(
        "No compatible checkpoint was found. Expected one of: "
        f"{DEFAULT_GUTENBERG_LATEST}, {DEFAULT_GUTENBERG_BEST}, or the legacy model files."
    )


def get_runtime() -> ChatRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    checkpoint_path, mode = _resolve_runtime_checkpoint()
    model, model_config, metadata = load_model_and_metadata(checkpoint_path, device=DEVICE)
    tokenizer_name = metadata.get("tokenizer_name", "gpt2") if isinstance(metadata, dict) else "gpt2"
    runtime = ChatRuntime(
        model=model,
        model_config=model_config,
        tokenizer=get_tokenizer(tokenizer_name),
        checkpoint_path=checkpoint_path,
        mode=mode,
        generation_config=DEFAULT_CHAT_GENERATION,
    )
    _RUNTIME = runtime
    return runtime


def _conversation_to_input(history: list[dict[str, str]], user_message: str) -> str:
    transcript_lines: list[str] = []
    for turn in history:
        transcript_lines.append(f"User: {turn['user']}")
        transcript_lines.append(f"Assistant: {turn['assistant']}")
    transcript_lines.append(f"User: {user_message}")
    return "\n".join(transcript_lines)


def _build_prompt(history: list[dict[str, str]], user_message: str, mode: str) -> str:
    if mode == "chat":
        instruction = "Continue the conversation below and respond helpfully to the latest user message."
    else:
        instruction = (
            "Continue the text below as a helpful assistant reply to the latest user message. "
            "The model is running in base completion mode, so keep the response concise and conversational."
        )

    entry = {
        "instruction": instruction,
        "input": _conversation_to_input(history, user_message),
    }
    return format_input(entry) + "\n\n### Response:\n"


def _prepare_prompt(runtime: ChatRuntime, history: list[dict[str, str]], user_message: str) -> tuple[str, torch.Tensor]:
    trimmed_history = list(history)
    max_prompt_tokens = runtime.model_config.context_length - runtime.generation_config.max_new_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("The configured max_new_tokens exceeds the model context length.")

    while True:
        prompt = _build_prompt(trimmed_history, user_message, runtime.mode)
        prompt_tokens = runtime.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        if len(prompt_tokens) <= max_prompt_tokens or not trimmed_history:
            break
        trimmed_history = trimmed_history[1:]

    if len(prompt_tokens) > max_prompt_tokens:
        prompt_tokens = prompt_tokens[-max_prompt_tokens:]
        prompt = runtime.tokenizer.decode(prompt_tokens)

    tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    return prompt, tensor


@chainlit.on_chat_start
async def on_chat_start() -> None:
    chainlit.user_session.set("history", [])
    try:
        runtime = get_runtime()
    except FileNotFoundError as exc:
        await chainlit.Message(content=str(exc)).send()
        return

    if runtime.mode == "chat":
        content = (
            f"Loaded chat-tuned model from `{runtime.checkpoint_path}`.\n\n"
            "Conversation history will be rolled into the prompt automatically."
        )
    else:
        content = (
            f"Loaded base model from `{runtime.checkpoint_path}`.\n\n"
            "No chat-tuned checkpoint was found, so the interface is running in base completion mode."
        )
    await chainlit.Message(content=content).send()


@chainlit.on_message
async def on_message(message: chainlit.Message) -> None:
    try:
        runtime = get_runtime()
    except FileNotFoundError as exc:
        await chainlit.Message(content=str(exc)).send()
        return

    history = chainlit.user_session.get("history") or []
    prompt_text, prompt_tokens = _prepare_prompt(runtime, history, message.content)

    token_ids = generate(
        model=runtime.model,
        idx=prompt_tokens,
        max_new_tokens=runtime.generation_config.max_new_tokens,
        context_size=runtime.model_config.context_length,
        temperature=runtime.generation_config.temperature,
        top_k=runtime.generation_config.top_k,
        eos_id=runtime.generation_config.eos_id,
    )
    generated_text = token_ids_to_text(token_ids.cpu(), runtime.tokenizer)
    response = extract_response_text(generated_text, prompt_text)
    if not response:
        response = "I couldn't generate a response from the current checkpoint."

    updated_history = history + [{"user": message.content, "assistant": response}]
    chainlit.user_session.set("history", updated_history)
    await chainlit.Message(content=response).send()
