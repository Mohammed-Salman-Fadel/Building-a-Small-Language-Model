from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.config import DEFAULT_SPAM_CLASSIFIER, DEFAULT_SPAM_CLASSIFIER_TRAINING
from small_llm.paths import DEFAULT_SPAM_CLASSIFIER_LATEST, SPAM_CLASSIFIER_CHECKPOINT_DIR, ensure_project_dirs
from small_llm.spam_classifier import (
    create_spam_classifier_model,
    load_spam_classifier_runtime,
    train_spam_classifier,
)
from small_llm.spam_data import create_spam_dataloaders
from small_llm.training import extract_checkpoint_metadata, extract_training_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the spam/not-spam classifier.")
    parser.add_argument("--base-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.weight_decay)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.eval_freq)
    parser.add_argument("--eval-iter", type=int, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.eval_iter)
    parser.add_argument("--max-length", type=int, default=DEFAULT_SPAM_CLASSIFIER.max_length)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPAM_CLASSIFIER_TRAINING.seed)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest spam-classifier checkpoint if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    torch.manual_seed(args.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    training_state: dict[str, object] = {}
    resume_checkpoint_payload: dict[str, object] | None = None

    if args.resume and DEFAULT_SPAM_CLASSIFIER_LATEST.exists():
        runtime = load_spam_classifier_runtime(
            checkpoint_path=DEFAULT_SPAM_CLASSIFIER_LATEST,
            device_name=device_name,
        )
        resume_checkpoint_payload = torch.load(DEFAULT_SPAM_CLASSIFIER_LATEST, map_location=runtime.device)
        checkpoint_metadata = extract_checkpoint_metadata(resume_checkpoint_payload)
        training_state = extract_training_state(resume_checkpoint_payload)
        model = runtime.model
        model_config = runtime.model_config
        tokenizer_name = resume_checkpoint_payload.get("tokenizer_name", "gpt2")
        max_length = int(checkpoint_metadata.get("max_length", args.max_length))
        print(f"Resuming spam classifier from {DEFAULT_SPAM_CLASSIFIER_LATEST}")
    else:
        model, model_config, tokenizer_name = create_spam_classifier_model(
            base_checkpoint_path=args.base_checkpoint or None,
            device_name=device_name,
        )
        max_length = min(args.max_length, model_config.context_length)
        print("Starting a new spam-classifier training run.")

    train_loader, val_loader, test_loader, split_sizes = create_spam_dataloaders(
        batch_size=args.batch_size,
        max_length=max_length,
        pad_token_id=DEFAULT_SPAM_CLASSIFIER.pad_token_id,
        tokenizer_name=tokenizer_name,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_checkpoint_payload is not None and isinstance(resume_checkpoint_payload.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(resume_checkpoint_payload["optimizer_state_dict"])

    train_spam_classifier(
        model,
        train_loader,
        val_loader,
        optimizer,
        device_name,
        args.epochs,
        args.eval_freq,
        args.eval_iter,
        checkpoint_dir=SPAM_CLASSIFIER_CHECKPOINT_DIR,
        model_config=model_config,
        metadata={
            "stage": "spam-classifier",
            "num_classes": DEFAULT_SPAM_CLASSIFIER.num_classes,
            "max_length": max_length,
            "pad_token_id": DEFAULT_SPAM_CLASSIFIER.pad_token_id,
            "split_sizes": split_sizes,
        },
        tokenizer_name=tokenizer_name,
        start_epoch=int(training_state.get("epoch_index", 0)) if training_state else 0,
        start_global_step=int(training_state.get("global_step", -1)),
        initial_best_val=float(training_state.get("best_val", float("inf"))),
    )

    print(f"Spam classifier checkpoints are in {SPAM_CLASSIFIER_CHECKPOINT_DIR}")
    print(f"Test set size: {split_sizes['test']}")


if __name__ == "__main__":
    main()
