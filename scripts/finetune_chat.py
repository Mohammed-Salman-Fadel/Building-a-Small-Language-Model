from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.config import DEFAULT_GUTENBERG_MODEL, DEFAULT_INSTRUCTION_FINETUNE
from small_llm.instruction_data import (
    DEFAULT_DOLLY_MAX_SAMPLES,
    DEFAULT_INSTRUCTION_DATASET,
    DEFAULT_OASST_LANG,
    DEFAULT_OASST_MAX_SAMPLES,
    DEFAULT_ULTRACHAT_MAX_SAMPLES,
    create_instruction_dataloaders,
    format_input,
)
from small_llm.paths import (
    DEFAULT_CHAT_CHECKPOINT,
    DEFAULT_CHAT_LATEST,
    DEFAULT_GUTENBERG_BEST,
    DEFAULT_GUTENBERG_LATEST,
    CHAT_FINETUNE_CHECKPOINT_DIR,
    ensure_project_dirs,
    get_base_checkpoint_candidates,
    resolve_first_existing,
)
from small_llm.training import (
    extract_checkpoint_metadata,
    extract_training_state,
    load_model_and_metadata,
    save_checkpoint,
    train_model_simple,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction-finetune the Gutenberg model into a chat model.")
    parser.add_argument("--base-checkpoint", type=str, default="")
    parser.add_argument(
        "--dataset",
        choices=("dolly", "ultrachat", "oasst1", "local"),
        default=DEFAULT_INSTRUCTION_DATASET,
        help="Instruction dataset to use for chat fine-tuning.",
    )
    parser.add_argument("--instruction-source-path", type=str, default="")
    parser.add_argument("--dolly-max-samples", type=int, default=DEFAULT_DOLLY_MAX_SAMPLES)
    parser.add_argument("--ultrachat-max-samples", type=int, default=DEFAULT_ULTRACHAT_MAX_SAMPLES)
    parser.add_argument("--oasst-lang", type=str, default=DEFAULT_OASST_LANG)
    parser.add_argument("--oasst-max-samples", type=int, default=DEFAULT_OASST_MAX_SAMPLES)
    parser.add_argument("--force-rebuild-dataset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_INSTRUCTION_FINETUNE.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_INSTRUCTION_FINETUNE.weight_decay)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.eval_freq)
    parser.add_argument("--eval-iter", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.eval_iter)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.save_freq)
    parser.add_argument("--seed", type=int, default=DEFAULT_INSTRUCTION_FINETUNE.seed)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest chat fine-tuning checkpoint if it exists.",
    )
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
    resume_checkpoint = DEFAULT_CHAT_LATEST if args.resume and DEFAULT_CHAT_LATEST.exists() else None
    training_state: dict[str, object] = {}

    if resume_checkpoint is not None:
        print(f"Resuming chat fine-tuning from {resume_checkpoint}")
        model, model_config, checkpoint_payload = load_model_and_metadata(
            resume_checkpoint,
            device=device,
            fallback_config=DEFAULT_GUTENBERG_MODEL,
        )
        checkpoint_metadata = extract_checkpoint_metadata(checkpoint_payload)
        training_state = extract_training_state(checkpoint_payload)
        tokenizer_name = checkpoint_payload.get("tokenizer_name", "gpt2") if isinstance(checkpoint_payload, dict) else "gpt2"
        base_checkpoint = Path(checkpoint_metadata.get("base_checkpoint", "")) if checkpoint_metadata.get("base_checkpoint") else resume_checkpoint
        run_config = checkpoint_metadata.get("run_config", {}) if isinstance(checkpoint_metadata, dict) else {}
        dataset_name = str(run_config.get("dataset", args.dataset))
        instruction_source_path = str(run_config.get("instruction_source_path", args.instruction_source_path or ""))
        oasst_lang = str(run_config.get("oasst_lang", args.oasst_lang))
        oasst_max_samples = run_config.get("oasst_max_samples", args.oasst_max_samples)
        oasst_max_samples = None if oasst_max_samples in {"", None} else int(oasst_max_samples)
        dolly_max_samples = run_config.get("dolly_max_samples", args.dolly_max_samples)
        dolly_max_samples = None if dolly_max_samples in {"", None} else int(dolly_max_samples)
        ultrachat_max_samples = run_config.get("ultrachat_max_samples", args.ultrachat_max_samples)
        ultrachat_max_samples = None if ultrachat_max_samples in {"", None} else int(ultrachat_max_samples)
        batch_size = int(run_config.get("batch_size", args.batch_size))
        seed = int(run_config.get("seed", args.seed))
        save_freq = int(run_config.get("save_freq", args.save_freq))
    else:
        base_checkpoint = resolve_base_checkpoint(args.base_checkpoint)
        model, model_config, checkpoint_payload = load_model_and_metadata(
            base_checkpoint,
            device=device,
            fallback_config=DEFAULT_GUTENBERG_MODEL,
        )
        checkpoint_metadata = extract_checkpoint_metadata(checkpoint_payload)
        tokenizer_name = checkpoint_payload.get("tokenizer_name", "gpt2") if isinstance(checkpoint_payload, dict) else "gpt2"
        dataset_name = args.dataset
        instruction_source_path = args.instruction_source_path
        oasst_lang = args.oasst_lang
        oasst_max_samples = args.oasst_max_samples
        dolly_max_samples = args.dolly_max_samples
        ultrachat_max_samples = args.ultrachat_max_samples
        batch_size = args.batch_size
        seed = args.seed
        save_freq = args.save_freq

    torch.manual_seed(seed)

    import tiktoken

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    train_loader, val_loader, _, split_map = create_instruction_dataloaders(
        tokenizer,
        batch_size=batch_size,
        allowed_max_length=model_config.context_length,
        num_workers=args.num_workers,
        device=device,
        source_path=instruction_source_path or None,
        dataset_name=dataset_name,
        oasst_lang=oasst_lang,
        oasst_max_samples=oasst_max_samples,
        dolly_max_samples=dolly_max_samples,
        ultrachat_max_samples=ultrachat_max_samples,
        force_rebuild=args.force_rebuild_dataset,
        seed=seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("optimizer_state_dict"), dict):
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
    start_context = format_input(
        {
            "instruction": "Say hello and introduce yourself briefly.",
            "input": "",
        }
    ) + "\n\n### Response:\n"
    resume_epoch = int(training_state.get("epoch_index", 0)) if training_state else 0
    resume_step = int(training_state.get("step_in_epoch", -1)) if training_state else -1
    if resume_checkpoint is not None and resume_step >= 0:
        print(f"Resuming inside epoch {resume_epoch + 1} from batch {resume_step + 2}.")

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
        checkpoint_dir=CHAT_FINETUNE_CHECKPOINT_DIR,
        model_config=model_config,
        metadata={
            "stage": "instruction-finetune",
            "base_checkpoint": str(base_checkpoint),
            "dataset": dataset_name,
            "split_sizes": {key: len(value) for key, value in split_map.items()},
            "run_config": {
                "batch_size": batch_size,
                "epochs": args.epochs,
                "seed": seed,
                "dataset": dataset_name,
                "instruction_source_path": instruction_source_path,
                "oasst_lang": oasst_lang,
                "oasst_max_samples": oasst_max_samples,
                "dolly_max_samples": dolly_max_samples,
                "ultrachat_max_samples": ultrachat_max_samples,
                "save_freq": save_freq,
            },
        },
        is_chat_model=True,
        warmup_steps=DEFAULT_INSTRUCTION_FINETUNE.warmup_steps,
        min_learning_rate=DEFAULT_INSTRUCTION_FINETUNE.min_learning_rate,
        grad_clip=DEFAULT_INSTRUCTION_FINETUNE.grad_clip,
        save_freq=save_freq,
        start_epoch=resume_epoch,
        start_global_step=int(training_state.get("global_step", -1)),
        start_step_in_epoch=resume_step,
        initial_tokens_seen=int(training_state.get("tokens_seen", 0)),
        initial_best_val=float(training_state.get("best_val", float("inf"))),
        initial_train_losses=list(training_state.get("train_losses", [])),
        initial_val_losses=list(training_state.get("val_losses", [])),
        initial_track_tokens_seen=list(training_state.get("track_tokens_seen", [])),
    )

    save_checkpoint(
        model,
        DEFAULT_CHAT_CHECKPOINT,
        optimizer=optimizer,
        model_config=model_config,
        metadata={
            "stage": "instruction-finetune",
            "base_checkpoint": str(base_checkpoint),
            "dataset": dataset_name,
            "source_tokenizer": tokenizer_name,
        },
        tokenizer_name=tokenizer_name,
        is_chat_model=True,
    )


if __name__ == "__main__":
    main()
