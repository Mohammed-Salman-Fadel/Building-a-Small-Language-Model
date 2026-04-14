from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .generation import get_tokenizer
from .paths import SPAM_PROCESSED_DIR, get_spam_dataset_candidates


def _materialize_spam_split(split_name: str) -> Path:
    SPAM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target_path = SPAM_PROCESSED_DIR / f"{split_name}.csv"
    if target_path.exists():
        return target_path

    source_path = next((candidate for candidate in get_spam_dataset_candidates(split_name) if candidate.exists()), None)
    if source_path is None:
        raise FileNotFoundError(f"Could not find the spam dataset split '{split_name}'.")

    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def load_spam_split(split_name: str) -> list[dict[str, Any]]:
    csv_path = _materialize_spam_split(split_name)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(
                {
                    "label": int(row["Label"]),
                    "text": row["Text"],
                }
            )
    return rows


class SpamDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        max_length: int,
        pad_token_id: int = 50256,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.rows = rows
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.tokenizer = get_tokenizer(tokenizer_name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer.encode(row["text"], allowed_special={"<|endoftext|>"})
        encoded = encoded[: self.max_length]
        if len(encoded) < self.max_length:
            encoded = encoded + [self.pad_token_id] * (self.max_length - len(encoded))

        input_tensor = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(row["label"], dtype=torch.long)
        return input_tensor, label_tensor


def create_spam_dataloaders(
    *,
    batch_size: int = 8,
    max_length: int = 120,
    pad_token_id: int = 50256,
    tokenizer_name: str = "gpt2",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_rows = load_spam_split("train")
    validation_rows = load_spam_split("validation")
    test_rows = load_spam_split("test")

    train_dataset = SpamDataset(
        train_rows,
        max_length=max_length,
        pad_token_id=pad_token_id,
        tokenizer_name=tokenizer_name,
    )
    validation_dataset = SpamDataset(
        validation_rows,
        max_length=max_length,
        pad_token_id=pad_token_id,
        tokenizer_name=tokenizer_name,
    )
    test_dataset = SpamDataset(
        test_rows,
        max_length=max_length,
        pad_token_id=pad_token_id,
        tokenizer_name=tokenizer_name,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader, {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }
