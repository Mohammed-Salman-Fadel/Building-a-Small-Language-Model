from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .config import DEFAULT_CHAT_GENERATION, DEFAULT_GUTENBERG_MODEL, GenerationConfig, ModelConfig
from .generation import extract_response_text, generate, get_tokenizer, token_ids_to_text
from .instruction_data import format_input
from .model import GPTModel
from .paths import (
    DEFAULT_GUTENBERG_BEST,
    DEFAULT_GUTENBERG_LATEST,
    get_base_checkpoint_candidates,
    get_chat_checkpoint_candidates,
    get_chat_training_checkpoint_candidates,
    resolve_first_existing,
)
from .training import load_model_and_metadata


@dataclass
class ChatRuntime:
    model: torch.nn.Module
    model_config: ModelConfig
    tokenizer: Any
    checkpoint_path: Path | None
    mode: str
    generation_config: GenerationConfig
    device: torch.device


def resolve_device(device_name: str = "auto") -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _resolve_checkpoint(mode: str) -> tuple[Path | None, str]:
    normalized = mode.lower()

    if normalized in {"auto", "chat"}:
        chat_candidates = [
            *get_chat_checkpoint_candidates(),
            *get_chat_training_checkpoint_candidates(),
        ]
        chat_checkpoint = resolve_first_existing(chat_candidates)
        if chat_checkpoint is not None:
            return chat_checkpoint, "chat"
        if normalized == "chat":
            raise FileNotFoundError("No chat checkpoint was found.")

    if normalized in {"auto", "base"}:
        base_checkpoint = resolve_first_existing(get_base_checkpoint_candidates())
        if base_checkpoint is not None:
            return base_checkpoint, "base"
        if normalized == "base":
            raise FileNotFoundError(
                "No base checkpoint was found. "
                f"Expected one of {DEFAULT_GUTENBERG_LATEST} or {DEFAULT_GUTENBERG_BEST}."
            )

    if normalized == "auto":
        return None, "untrained"
    if normalized == "untrained":
        return None, "untrained"

    raise ValueError(f"Unsupported chat runtime mode: {mode}")


def load_chat_runtime(
    *,
    mode: str = "auto",
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    generation_config: GenerationConfig | None = None,
    model_config: ModelConfig | None = None,
    seed: int | None = None,
) -> ChatRuntime:
    device = resolve_device(device_name)
    generation = generation_config or DEFAULT_CHAT_GENERATION

    if checkpoint_path is not None:
        resolved_path = Path(checkpoint_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
        model, resolved_model_config, metadata = load_model_and_metadata(
            resolved_path,
            device=device,
            fallback_config=model_config or DEFAULT_GUTENBERG_MODEL,
        )
        tokenizer_name = metadata.get("tokenizer_name", "gpt2") if isinstance(metadata, dict) else "gpt2"
        resolved_mode = "chat" if isinstance(metadata, dict) and metadata.get("is_chat_model") else "base"
        return ChatRuntime(
            model=model,
            model_config=resolved_model_config,
            tokenizer=get_tokenizer(tokenizer_name),
            checkpoint_path=resolved_path,
            mode=resolved_mode,
            generation_config=generation,
            device=device,
        )

    resolved_checkpoint, resolved_mode = _resolve_checkpoint(mode)
    if resolved_checkpoint is not None:
        model, resolved_model_config, metadata = load_model_and_metadata(
            resolved_checkpoint,
            device=device,
            fallback_config=model_config or DEFAULT_GUTENBERG_MODEL,
        )
        tokenizer_name = metadata.get("tokenizer_name", "gpt2") if isinstance(metadata, dict) else "gpt2"
        if isinstance(metadata, dict) and metadata.get("is_chat_model"):
            resolved_mode = "chat"
        return ChatRuntime(
            model=model,
            model_config=resolved_model_config,
            tokenizer=get_tokenizer(tokenizer_name),
            checkpoint_path=resolved_checkpoint,
            mode=resolved_mode,
            generation_config=generation,
            device=device,
        )

    if seed is not None:
        torch.manual_seed(seed)

    fresh_config = model_config or DEFAULT_GUTENBERG_MODEL
    model = GPTModel(fresh_config.to_dict())
    model.to(device)
    model.eval()
    return ChatRuntime(
        model=model,
        model_config=fresh_config,
        tokenizer=get_tokenizer(),
        checkpoint_path=None,
        mode="untrained",
        generation_config=generation,
        device=device,
    )


def describe_runtime(runtime: ChatRuntime) -> str:
    if runtime.mode == "chat" and runtime.checkpoint_path is not None:
        return f"Loaded chat model from {runtime.checkpoint_path} on {runtime.device}."
    if runtime.mode == "base" and runtime.checkpoint_path is not None:
        return (
            f"Loaded base model from {runtime.checkpoint_path} on {runtime.device}. "
            "Responses come from the pretrained model without chat fine-tuning."
        )
    return (
        f"Loaded an untrained model on {runtime.device}. "
        "Responses will usually be incoherent, which is useful for contrast."
    )


def _conversation_to_input(history: list[dict[str, str]], user_message: str) -> str:
    transcript_lines: list[str] = []
    for turn in history:
        transcript_lines.append(f"User: {turn['user']}")
        transcript_lines.append(f"Assistant: {turn['assistant']}")
    transcript_lines.append(f"User: {user_message}")
    return "\n".join(transcript_lines)


def build_prompt(history: list[dict[str, str]], user_message: str, mode: str) -> str:
    if mode == "chat":
        instruction = "Continue the conversation below and respond helpfully to the latest user message."
    elif mode == "base":
        instruction = (
            "Continue the text below as a helpful assistant reply to the latest user message. "
            "The model is running in base completion mode, so keep the response concise and conversational."
        )
    else:
        instruction = (
            "Continue the text below as if attempting to answer the latest user message. "
            "The model is untrained, so the output may be incoherent."
        )

    entry = {
        "instruction": instruction,
        "input": _conversation_to_input(history, user_message),
    }
    return format_input(entry) + "\n\n### Response:\n"


def prepare_prompt(runtime: ChatRuntime, history: list[dict[str, str]], user_message: str) -> tuple[str, torch.Tensor]:
    trimmed_history = list(history)
    max_prompt_tokens = runtime.model_config.context_length - runtime.generation_config.max_new_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("The configured max_new_tokens exceeds the model context length.")

    while True:
        prompt = build_prompt(trimmed_history, user_message, runtime.mode)
        prompt_tokens = runtime.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        if len(prompt_tokens) <= max_prompt_tokens or not trimmed_history:
            break
        trimmed_history = trimmed_history[1:]

    if len(prompt_tokens) > max_prompt_tokens:
        prompt_tokens = prompt_tokens[-max_prompt_tokens:]
        prompt = runtime.tokenizer.decode(prompt_tokens)

    tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(runtime.device)
    return prompt, tensor


def generate_chat_response(
    runtime: ChatRuntime,
    history: list[dict[str, str]],
    user_message: str,
) -> tuple[str, list[dict[str, str]]]:
    prompt_text, prompt_tokens = prepare_prompt(runtime, history, user_message)
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
        response = "I couldn't generate a response from the current runtime."

    updated_history = history + [{"user": user_message, "assistant": response}]
    return response, updated_history


def override_generation_config(
    base_config: GenerationConfig,
    *,
    temperature: float | None = None,
    top_k: int | None = None,
    max_new_tokens: int | None = None,
) -> GenerationConfig:
    return replace(
        base_config,
        temperature=base_config.temperature if temperature is None else temperature,
        top_k=base_config.top_k if top_k is None else top_k,
        max_new_tokens=base_config.max_new_tokens if max_new_tokens is None else max_new_tokens,
    )
