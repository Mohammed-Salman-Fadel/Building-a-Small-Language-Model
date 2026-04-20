from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from .data import create_token_dataloader
from .generation import get_tokenizer
from .paths import GUTENBERG_PROCESSED_DIR, GUTENBERG_RAW_DIR, ensure_project_dirs


DEFAULT_DATASET_NAME = "common-pile/project_gutenberg_filtered"


def _import_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required for Gutenberg preparation. "
            "Install the project requirements before using small_llm.gutenberg."
        ) from exc
    return load_dataset


def _subset_cache_dir(max_books: int | None, seed: int) -> Path:
    label = "full" if max_books is None else f"books-{max_books}"
    return GUTENBERG_PROCESSED_DIR / f"{label}-seed-{seed}"


def load_gutenberg_documents(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "train",
    max_books: int | None = 1000,
    seed: int = 123,
    streaming: bool = True,
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
    progress_every: int = 25,
) -> list[dict[str, Any]]:
    ensure_project_dirs()
    load_dataset = _import_datasets()

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"Preparing Gutenberg dataset (attempt {attempt}/{max_retries}) "
                f"from '{dataset_name}' with max_books={max_books}..."
            )

            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                cache_dir=str(GUTENBERG_RAW_DIR),
            )
            shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

            documents: list[dict[str, Any]] = []
            for index, row in enumerate(shuffled_dataset):
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                documents.append(
                    {
                        "id": row.get("id", f"book-{index}"),
                        "text": text,
                        "metadata": row.get("metadata", {}),
                    }
                )

                if progress_every > 0 and len(documents) % progress_every == 0:
                    print(f"Collected {len(documents)} Gutenberg documents...")

                if max_books is not None and len(documents) >= max_books:
                    break

            print(f"Collected {len(documents)} Gutenberg documents total.")
            return documents

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break

            print(
                "Gutenberg streaming failed during download/read. "
                f"Retrying in {retry_delay_seconds:.0f}s...\n"
                f"Cause: {exc.__class__.__name__}: {exc}"
            )
            time.sleep(retry_delay_seconds)

    raise RuntimeError(
        "Failed to load the Gutenberg dataset after multiple attempts. "
        "This usually means the Hugging Face download stream was interrupted. "
        "Check your network connection and try again."
    ) from last_error


def split_documents(
    documents: list[dict[str, Any]],
    *,
    seed: int = 123,
) -> dict[str, list[dict[str, Any]]]:
    shuffled_documents = list(documents)
    random.Random(seed).shuffle(shuffled_documents)

    total = len(shuffled_documents)
    if total < 3:
        raise ValueError("Need at least 3 Gutenberg documents to create train/validation/holdout splits.")

    train_count = max(1, int(total * 0.9))
    val_count = max(1, int(total * 0.05))
    holdout_count = total - train_count - val_count

    while holdout_count < 1:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
        holdout_count = total - train_count - val_count

    train_end = train_count
    val_end = train_end + val_count

    return {
        "train": shuffled_documents[:train_end],
        "validation": shuffled_documents[train_end:val_end],
        "holdout": shuffled_documents[val_end:],
    }


def tokenize_documents(documents: list[dict[str, Any]], *, tokenizer_name: str = "gpt2") -> torch.Tensor:
    tokenizer = get_tokenizer(tokenizer_name)
    joined_text = "<|endoftext|>".join(document["text"] for document in documents)
    token_ids = tokenizer.encode(joined_text, allowed_special={"<|endoftext|>"})
    return torch.tensor(token_ids, dtype=torch.long)


def prepare_gutenberg_splits(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    max_books: int | None = 1000,
    seed: int = 123,
    force_rebuild: bool = False,
    tokenizer_name: str = "gpt2",
) -> dict[str, Path]:
    ensure_project_dirs()
    cache_dir = _subset_cache_dir(max_books=max_books, seed=seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = cache_dir / "metadata.json"
    train_path = cache_dir / "train_tokens.pt"
    validation_path = cache_dir / "validation_tokens.pt"
    holdout_path = cache_dir / "holdout_tokens.pt"

    if (
        not force_rebuild
        and metadata_path.exists()
        and train_path.exists()
        and validation_path.exists()
        and holdout_path.exists()
    ):
        return {
            "cache_dir": cache_dir,
            "metadata": metadata_path,
            "train": train_path,
            "validation": validation_path,
            "holdout": holdout_path,
        }

    documents = load_gutenberg_documents(
        dataset_name=dataset_name,
        max_books=max_books,
        seed=seed,
        streaming=True,
    )
    print("Splitting Gutenberg documents into train/validation/holdout...")
    split_map = split_documents(documents, seed=seed)

    split_paths = {
        "train": train_path,
        "validation": validation_path,
        "holdout": holdout_path,
    }

    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "max_books": max_books,
        "seed": seed,
        "tokenizer_name": tokenizer_name,
        "splits": {},
    }

    for split_name, split_documents_list in split_map.items():
        print(f"Tokenizing {split_name} split with {len(split_documents_list)} documents...")
        tokens = tokenize_documents(split_documents_list, tokenizer_name=tokenizer_name)
        torch.save(tokens, split_paths[split_name])
        metadata["splits"][split_name] = {
            "documents": len(split_documents_list),
            "tokens": int(tokens.numel()),
        }
        print(f"Saved {split_name} tokens to {split_paths[split_name]}")

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Gutenberg metadata to {metadata_path}")
    return {
        "cache_dir": cache_dir,
        "metadata": metadata_path,
        "train": train_path,
        "validation": validation_path,
        "holdout": holdout_path,
    }


def create_gutenberg_dataloaders(
    *,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle_train: bool = True,
    num_workers: int = 0,
    dataset_name: str = DEFAULT_DATASET_NAME,
    max_books: int | None = 1000,
    seed: int = 123,
    force_rebuild: bool = False,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    prepared = prepare_gutenberg_splits(
        dataset_name=dataset_name,
        max_books=max_books,
        seed=seed,
        force_rebuild=force_rebuild,
    )
    metadata = json.loads(Path(prepared["metadata"]).read_text(encoding="utf-8"))

    train_tokens = torch.load(prepared["train"])
    val_tokens = torch.load(prepared["validation"])
    holdout_tokens = torch.load(prepared["holdout"])

    train_loader = create_token_dataloader(
        train_tokens,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle_train,
        drop_last=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = create_token_dataloader(
        val_tokens,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    holdout_loader = create_token_dataloader(
        holdout_tokens,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, holdout_loader, metadata
