from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .paths import (
    INSTRUCTION_PROCESSED_DIR,
    get_instruction_data_candidates,
    get_instruction_response_candidates,
)


def format_input(entry: dict[str, Any]) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def _materialize_instruction_file(source_path: Path) -> Path:
    INSTRUCTION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target_path = INSTRUCTION_PROCESSED_DIR / source_path.name
    if source_path.resolve() != target_path.resolve() and not target_path.exists():
        shutil.copy2(source_path, target_path)
    return target_path if target_path.exists() else source_path


def load_instruction_data(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is not None:
        source_path = Path(path)
    else:
        candidates = get_instruction_data_candidates()
        source_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_path is None:
            raise FileNotFoundError("Could not find instruction-data.json in canonical or legacy locations.")

    materialized_path = _materialize_instruction_file(source_path)
    return json.loads(materialized_path.read_text(encoding="utf-8"))


def split_instruction_data(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict[str, Any]], tokenizer: Any) -> None:
        self.data = data
        self.encoded_texts: list[list[int]] = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(instruction_plus_input + response_text))

    def __getitem__(self, index: int) -> list[int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def custom_collate_fn(
    batch: list[list[int]],
    *,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst: list[torch.Tensor] = []
    targets_lst: list[torch.Tensor] = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        mask = targets == pad_token_id
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def create_instruction_dataloaders(
    tokenizer: Any,
    *,
    batch_size: int = 8,
    allowed_max_length: int | None = None,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    source_path: str | Path | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    data = load_instruction_data(path=source_path)
    split_map = split_instruction_data(data)

    collate = partial(
        custom_collate_fn,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    train_loader = DataLoader(
        InstructionDataset(split_map["train"], tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        InstructionDataset(split_map["validation"], tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        InstructionDataset(split_map["test"], tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, split_map


def load_instruction_responses(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is not None:
        source_path = Path(path)
    else:
        candidates = get_instruction_response_candidates()
        source_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_path is None:
            raise FileNotFoundError(
                "Could not find instruction-data-with-response.json in canonical or legacy locations."
            )

    materialized_path = _materialize_instruction_file(source_path)
    return json.loads(materialized_path.read_text(encoding="utf-8"))
