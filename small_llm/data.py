from __future__ import annotations

from typing import Sequence

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def _build_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int) -> None:
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        self.dataset = TokenSequenceDataset(token_ids, max_length=max_length, stride=stride)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]


class TokenSequenceDataset(Dataset):
    def __init__(self, token_ids: Sequence[int] | torch.Tensor, max_length: int, stride: int) -> None:
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride
        if self.token_ids.numel() <= max_length:
            raise ValueError(
                "Token sequence is too short for the configured max_length. "
                "Provide more tokens or reduce max_length."
            )
        self.start_positions = list(range(0, self.token_ids.numel() - max_length, stride))

    def __len__(self) -> int:
        return len(self.start_positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.start_positions[idx]
        input_chunk = self.token_ids[start : start + self.max_length]
        target_chunk = self.token_ids[start + 1 : start + self.max_length + 1]
        return input_chunk.clone(), target_chunk.clone()


def create_token_dataloader(
    token_ids: Sequence[int] | torch.Tensor,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    dataset = TokenSequenceDataset(token_ids, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=_build_generator(seed),
    )


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=_build_generator(seed),
    )
