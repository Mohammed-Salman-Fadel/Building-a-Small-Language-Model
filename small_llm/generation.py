from __future__ import annotations

from typing import Any

import tiktoken
import torch


def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def text_to_token_ids(text: str, tokenizer: Any) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Any) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    return generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=0.0,
        top_k=None,
        eos_id=None,
    )


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            effective_top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, effective_top_k)
            cutoff = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def extract_response_text(generated_text: str, prompt_text: str) -> str:
    if generated_text.startswith(prompt_text):
        response_text = generated_text[len(prompt_text) :]
    else:
        response_text = generated_text

    for marker in ("<|endoftext|>", "### User:", "### Instruction:"):
        response_text = response_text.split(marker)[0]

    return response_text.replace("### Response:", "").strip()
