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
from small_llm.paths import DEFAULT_GUTENBERG_LATEST, GUTENBERG_CHECKPOINT_DIR, ensure_project_dirs
from small_llm.training import (
    extract_checkpoint_metadata,
    extract_training_state,
    load_model_and_metadata,
    train_model_simple,
)


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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest Gutenberg checkpoint if it exists.",
    )
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

    resume_payload: dict[str, object] | None = None
    training_state: dict[str, object] = {}
    metadata_override: dict[str, object] = {}
    resume_checkpoint = DEFAULT_GUTENBERG_LATEST if args.resume and DEFAULT_GUTENBERG_LATEST.exists() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume_checkpoint is not None:
        print(f"Resuming Gutenberg pretraining from {resume_checkpoint}")
        model, model_config, checkpoint_payload = load_model_and_metadata(
            resume_checkpoint,
            device=device,
            fallback_config=DEFAULT_GUTENBERG_MODEL,
        )
        resume_payload = checkpoint_payload
        checkpoint_metadata = extract_checkpoint_metadata(checkpoint_payload)
        training_state = extract_training_state(checkpoint_payload)
        run_config = checkpoint_metadata.get("run_config", {}) if isinstance(checkpoint_metadata, dict) else {}
        max_books = int(run_config.get("max_books", args.max_books)) if run_config.get("max_books") is not None else args.max_books
        max_length = int(run_config.get("max_length", model_config.context_length))
        stride = int(run_config.get("stride", args.stride))
        batch_size = int(run_config.get("batch_size", args.batch_size))
        seed = int(run_config.get("seed", args.seed))
        metadata_override = checkpoint_metadata
    else:
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
        model.to(device)
        max_books = args.max_books
        max_length = args.max_length
        stride = args.stride
        batch_size = args.batch_size
        seed = args.seed

    train_loader, val_loader, _, metadata = create_gutenberg_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        num_workers=args.num_workers,
        max_books=max_books,
        seed=seed,
        force_rebuild=args.force_rebuild,
    )
    print("Gutenberg datasets are ready.")
    print(f"Using device: {device}")
    print(f"Model config: {model_config}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None and isinstance(resume_payload.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    tokenizer = get_tokenizer()
    training_metadata = {
        "stage": "gutenberg-pretrain",
        "dataset": metadata,
        "max_books": max_books,
        "seed": seed,
        "run_config": {
            "batch_size": batch_size,
            "max_length": max_length,
            "stride": stride,
            "max_books": max_books,
            "seed": seed,
        },
    }
    if metadata_override:
        merged_metadata = dict(metadata_override)
        merged_metadata.update(training_metadata)
        training_metadata = merged_metadata
    resume_epoch = int(training_state.get("epoch_index", 0)) if training_state else 0
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
        start_epoch=resume_epoch,
        start_global_step=int(training_state.get("global_step", -1)),
        initial_tokens_seen=int(training_state.get("tokens_seen", 0)),
        initial_best_val=float(training_state.get("best_val", float("inf"))),
        initial_train_losses=list(training_state.get("train_losses", [])),
        initial_val_losses=list(training_state.get("val_losses", [])),
        initial_track_tokens_seen=list(training_state.get("track_tokens_seen", [])),
    )
    print(f"Training finished. Latest checkpoint should be in {GUTENBERG_CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
