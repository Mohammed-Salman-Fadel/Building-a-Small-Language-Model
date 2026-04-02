from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.config import DEFAULT_GUTENBERG_MODEL, DEFAULT_GUTENBERG_TRAINING
from small_llm.generation import get_tokenizer
from small_llm.gutenberg import create_gutenberg_dataloaders
from small_llm.model import GPTModel
from small_llm.paths import GUTENBERG_CHECKPOINT_DIR, ensure_project_dirs
from small_llm.training import train_model_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the small LLM on the Gutenberg dataset.")
    parser.add_argument("--max-books", type=int, default=DEFAULT_GUTENBERG_TRAINING.max_books)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_GUTENBERG_TRAINING.batch_size)
    parser.add_argument("--max-length", type=int, default=DEFAULT_GUTENBERG_TRAINING.max_length)
    parser.add_argument("--stride", type=int, default=DEFAULT_GUTENBERG_TRAINING.stride)
    parser.add_argument("--epochs", type=int, default=DEFAULT_GUTENBERG_TRAINING.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_GUTENBERG_TRAINING.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_GUTENBERG_TRAINING.min_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_GUTENBERG_TRAINING.weight_decay)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_GUTENBERG_TRAINING.eval_freq)
    parser.add_argument("--eval-iter", type=int, default=DEFAULT_GUTENBERG_TRAINING.eval_iter)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_GUTENBERG_TRAINING.warmup_steps)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GUTENBERG_TRAINING.grad_clip)
    parser.add_argument("--seed", type=int, default=DEFAULT_GUTENBERG_TRAINING.seed)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--emb-dim", type=int, default=DEFAULT_GUTENBERG_MODEL.emb_dim)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_GUTENBERG_MODEL.n_heads)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_GUTENBERG_MODEL.n_layers)
    parser.add_argument("--drop-rate", type=float, default=DEFAULT_GUTENBERG_MODEL.drop_rate)
    parser.add_argument(
        "--qkv-bias",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GUTENBERG_MODEL.qkv_bias,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    torch.manual_seed(args.seed)
    print("Starting Gutenberg pretraining setup...")

    train_loader, val_loader, _, metadata = create_gutenberg_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        num_workers=args.num_workers,
        max_books=args.max_books,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )
    print("Gutenberg datasets are ready.")

    model_config = replace(
        DEFAULT_GUTENBERG_MODEL,
        context_length=args.max_length,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        drop_rate=args.drop_rate,
        qkv_bias=args.qkv_bias,
    )
    model = GPTModel(model_config.to_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    print(f"Model config: {model_config}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    tokenizer = get_tokenizer()
    training_metadata = {
        "stage": "gutenberg-pretrain",
        "dataset": metadata,
        "max_books": args.max_books,
        "seed": args.seed,
    }
    print("Starting training loop...")

    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        args.eval_freq,
        args.eval_iter,
        "Once upon a time",
        tokenizer,
        checkpoint_dir=GUTENBERG_CHECKPOINT_DIR,
        model_config=model_config,
        metadata=training_metadata,
        is_chat_model=False,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_learning_rate,
        grad_clip=args.grad_clip,
    )
    print(f"Training finished. Latest checkpoint should be in {GUTENBERG_CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
