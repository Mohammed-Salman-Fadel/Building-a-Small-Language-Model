from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.config import DEFAULT_GUTENBERG_MODEL, DEFAULT_INSTRUCTION_FINETUNE
from small_llm.instruction_data import create_instruction_dataloaders, format_input
from small_llm.paths import (
    DEFAULT_CHAT_CHECKPOINT,
    DEFAULT_GUTENBERG_BEST,
    DEFAULT_GUTENBERG_LATEST,
    ensure_project_dirs,
    get_base_checkpoint_candidates,
    resolve_first_existing,
)
from small_llm.training import load_model_and_metadata, save_checkpoint, train_model_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction-finetune the Gutenberg model into a chat model.")
    parser.add_argument("--base-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_INSTRUCTION_FINETUNE.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_INSTRUCTION_FINETUNE.weight_decay)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.eval_freq)
    parser.add_argument("--eval-iter", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.eval_iter)
    parser.add_argument("--seed", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.seed)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_base_checkpoint(explicit_path: str) -> Path:
    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    default_candidates = [DEFAULT_GUTENBERG_LATEST, DEFAULT_GUTENBERG_BEST, *get_base_checkpoint_candidates()]
    checkpoint_path = resolve_first_existing(default_candidates)
    if checkpoint_path is None:
        raise FileNotFoundError("No base checkpoint was found for instruction finetuning.")
    return checkpoint_path


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_checkpoint = resolve_base_checkpoint(args.base_checkpoint)
    model, model_config, metadata = load_model_and_metadata(
        base_checkpoint,
        device=device,
        fallback_config=DEFAULT_GUTENBERG_MODEL,
    )
    tokenizer_name = metadata.get("tokenizer_name", "gpt2") if isinstance(metadata, dict) else "gpt2"

    import tiktoken

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    train_loader, val_loader, _, split_map = create_instruction_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        allowed_max_length=model_config.context_length,
        num_workers=args.num_workers,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_context = format_input(
        {
            "instruction": "Say hello and introduce yourself briefly.",
            "input": "",
        }
    ) + "\n\n### Response:\n"

    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        args.eval_freq,
        args.eval_iter,
        start_context,
        tokenizer,
        checkpoint_dir=None,
        model_config=model_config,
        metadata={
            "stage": "instruction-finetune",
            "base_checkpoint": str(base_checkpoint),
            "split_sizes": {key: len(value) for key, value in split_map.items()},
        },
        is_chat_model=True,
    )

    save_checkpoint(
        model,
        DEFAULT_CHAT_CHECKPOINT,
        optimizer=optimizer,
        model_config=model_config,
        metadata={
            "stage": "instruction-finetune",
            "base_checkpoint": str(base_checkpoint),
            "source_tokenizer": tokenizer_name,
        },
        tokenizer_name=tokenizer_name,
        is_chat_model=True,
    )


if __name__ == "__main__":
    main()
