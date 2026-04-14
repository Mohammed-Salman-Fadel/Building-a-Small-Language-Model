from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import DEFAULT_GUTENBERG_MODEL, DEFAULT_SPAM_CLASSIFIER, GPT2_SMALL_1024, ModelConfig, SpamClassifierConfig, config_from_dict
from .generation import get_tokenizer
from .model import GPTModel
from .paths import get_base_checkpoint_candidates, get_spam_classifier_checkpoint_candidates, resolve_first_existing
from .training import (
    calc_loss_loader,
    extract_checkpoint_metadata,
    extract_model_state_dict,
    extract_training_state,
    load_model_and_metadata,
    save_checkpoint,
)


SPAM_HINT_PATTERNS = (
    "free",
    "winner",
    "claim",
    "cash",
    "prize",
    "call now",
    "txt",
    "urgent",
    "limited time",
    "guaranteed",
    "congratulations",
    "ringtone",
    "award",
    "reply",
    "stop texts",
    "18+",
    "www.",
    "http://",
    "https://",
)


@dataclass
class SpamClassifierRuntime:
    model: torch.nn.Module | None
    model_config: ModelConfig | None
    classifier_config: SpamClassifierConfig
    tokenizer: Any | None
    checkpoint_path: Path | None
    device: torch.device
    mode: str


class GPTSpamClassifier(GPTModel):
    def __init__(self, cfg: dict[str, Any] | ModelConfig, num_classes: int = 2) -> None:
        config = config_from_dict(cfg)
        super().__init__(config.to_dict())
        self.out_head = nn.Linear(config.emb_dim, num_classes)


def _resolve_device(device_name: str = "auto") -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _resolve_classifier_config(metadata: dict[str, Any], model_config: ModelConfig) -> SpamClassifierConfig:
    num_classes = metadata.get("num_classes")
    max_length = metadata.get("max_length")
    pad_token_id = metadata.get("pad_token_id")
    dropout = metadata.get("dropout")
    return SpamClassifierConfig(
        num_classes=int(num_classes) if num_classes is not None else DEFAULT_SPAM_CLASSIFIER.num_classes,
        max_length=min(
            model_config.context_length,
            int(max_length) if max_length is not None else DEFAULT_SPAM_CLASSIFIER.max_length,
        ),
        pad_token_id=int(pad_token_id) if pad_token_id is not None else DEFAULT_SPAM_CLASSIFIER.pad_token_id,
        dropout=float(dropout) if dropout is not None else DEFAULT_SPAM_CLASSIFIER.dropout,
    )


def create_spam_classifier_model(
    *,
    base_checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
) -> tuple[GPTSpamClassifier, ModelConfig, str]:
    device = _resolve_device(device_name)
    resolved_base = Path(base_checkpoint_path) if base_checkpoint_path is not None else resolve_first_existing(
        get_base_checkpoint_candidates()
    )

    if resolved_base is not None and resolved_base.exists():
        base_model, model_config, metadata = load_model_and_metadata(
            resolved_base,
            device=device,
            fallback_config=DEFAULT_GUTENBERG_MODEL,
        )
        tokenizer_name = metadata.get("tokenizer_name", "gpt2") if isinstance(metadata, dict) else "gpt2"
        classifier_model = GPTSpamClassifier(model_config, num_classes=DEFAULT_SPAM_CLASSIFIER.num_classes)
        classifier_model.load_state_dict(base_model.state_dict(), strict=False)
        classifier_model.to(device)
        return classifier_model, model_config, tokenizer_name

    model_config = DEFAULT_GUTENBERG_MODEL
    classifier_model = GPTSpamClassifier(model_config, num_classes=DEFAULT_SPAM_CLASSIFIER.num_classes)
    classifier_model.to(device)
    return classifier_model, model_config, "gpt2"


def load_spam_classifier_runtime(
    *,
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    allow_heuristic_fallback: bool = False,
) -> SpamClassifierRuntime:
    device = _resolve_device(device_name)
    resolved_path = Path(checkpoint_path) if checkpoint_path is not None else resolve_first_existing(
        get_spam_classifier_checkpoint_candidates()
    )
    if resolved_path is None or not resolved_path.exists():
        if allow_heuristic_fallback:
            return SpamClassifierRuntime(
                model=None,
                model_config=None,
                classifier_config=DEFAULT_SPAM_CLASSIFIER,
                tokenizer=None,
                checkpoint_path=None,
                device=device,
                mode="heuristic",
            )
        raise FileNotFoundError("No spam-classifier checkpoint was found.")

    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    metadata = extract_checkpoint_metadata(checkpoint)
    model_config = config_from_dict(checkpoint.get("model_config", GPT2_SMALL_1024.to_dict()))
    classifier_config = _resolve_classifier_config(metadata, model_config)
    tokenizer_name = checkpoint.get("tokenizer_name", "gpt2") if isinstance(checkpoint, dict) else "gpt2"

    model = GPTSpamClassifier(model_config, num_classes=classifier_config.num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return SpamClassifierRuntime(
        model=model,
        model_config=model_config,
        classifier_config=classifier_config,
        tokenizer=get_tokenizer(tokenizer_name),
        checkpoint_path=resolved_path,
        device=device,
        mode="trained",
    )


def describe_spam_classifier(runtime: SpamClassifierRuntime) -> str:
    if runtime.mode == "trained":
        return (
            f"Loaded spam classifier from {runtime.checkpoint_path} on {runtime.device}. "
            "In classifier mode, 'true' means spam and 'false' means not spam."
        )
    return (
        f"No trained spam-classifier checkpoint was found, so a heuristic classifier is active on {runtime.device}. "
        "In classifier mode, 'true' means spam and 'false' means not spam."
    )


def classify_text(runtime: SpamClassifierRuntime, text: str) -> bool:
    if runtime.mode == "heuristic":
        return heuristic_classify_text(text)
    if runtime.model is None or runtime.tokenizer is None or runtime.model_config is None:
        raise ValueError("Trained classifier runtime is not fully initialized.")

    encoded = runtime.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    max_length = min(runtime.classifier_config.max_length, runtime.model_config.context_length)
    if max_length <= 0:
        raise ValueError("Classifier max_length must be positive.")

    encoded = encoded[:max_length]
    if len(encoded) < max_length:
        encoded = encoded + [runtime.classifier_config.pad_token_id] * (max_length - len(encoded))

    input_tensor = torch.tensor(encoded, dtype=torch.long, device=runtime.device).unsqueeze(0)
    with torch.no_grad():
        logits = runtime.model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return bool(predicted_label == 1)


def heuristic_classify_text(text: str) -> bool:
    lowered = text.lower()
    score = 0
    for pattern in SPAM_HINT_PATTERNS:
        if pattern in lowered:
            score += 1
    if any(char.isdigit() for char in lowered):
        score += 1
    if lowered.count("!") >= 2:
        score += 1
    return score >= 2


def calc_classifier_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device | str,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_classifier_loss_loader(
    data_loader: Any,
    model: torch.nn.Module,
    device: torch.device | str,
    num_batches: int | None = None,
) -> float:
    if len(data_loader) == 0:
        return float("nan")

    batches_to_use = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    for batch_index, (input_batch, target_batch) in enumerate(data_loader):
        if batch_index >= batches_to_use:
            break
        loss = calc_classifier_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / batches_to_use


def calc_classifier_accuracy(
    data_loader: Any,
    model: torch.nn.Module,
    device: torch.device | str,
    num_batches: int | None = None,
) -> float:
    if len(data_loader) == 0:
        return float("nan")

    model.eval()
    batches_to_use = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    correct_predictions = 0
    total_examples = 0
    with torch.no_grad():
        for batch_index, (input_batch, target_batch) in enumerate(data_loader):
            if batch_index >= batches_to_use:
                break
            logits = model(input_batch.to(device))[:, -1, :]
            predictions = torch.argmax(logits, dim=-1)
            targets = target_batch.to(device)
            correct_predictions += (predictions == targets).sum().item()
            total_examples += targets.numel()
    model.train()
    return correct_predictions / max(1, total_examples)


def train_spam_classifier(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    *,
    checkpoint_dir: str | Path | None = None,
    model_config: ModelConfig | None = None,
    metadata: dict[str, Any] | None = None,
    tokenizer_name: str = "gpt2",
    start_epoch: int = 0,
    start_global_step: int = -1,
    initial_best_val: float = float("inf"),
) -> tuple[list[float], list[float], list[float], list[float]]:
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    best_val = initial_best_val
    global_step = start_global_step

    def build_metadata(epoch_index: int) -> dict[str, Any]:
        payload = dict(metadata or {})
        payload["training_state"] = {
            "epoch_index": epoch_index,
            "global_step": global_step,
            "best_val": best_val,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
        }
        return payload

    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                loss = calc_classifier_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()

                if eval_freq > 0 and global_step % eval_freq == 0:
                    train_loss = calc_classifier_loss_loader(train_loader, model, device, eval_iter)
                    val_loss = calc_classifier_loss_loader(val_loader, model, device, eval_iter)
                    train_acc = calc_classifier_accuracy(train_loader, model, device, eval_iter)
                    val_acc = calc_classifier_accuracy(val_loader, model, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                        f"Train acc {train_acc:.3f}, Val acc {val_acc:.3f}"
                    )

                    improved = val_loss < best_val
                    if improved:
                        best_val = val_loss
                    if checkpoint_root is not None:
                        save_checkpoint(
                            model,
                            checkpoint_root / "latest.pth",
                            optimizer=optimizer,
                            model_config=model_config,
                            metadata=build_metadata(epoch),
                            tokenizer_name=tokenizer_name,
                        )
                        if improved:
                            save_checkpoint(
                                model,
                                checkpoint_root / "best.pth",
                                optimizer=optimizer,
                                model_config=model_config,
                                metadata=build_metadata(epoch),
                                tokenizer_name=tokenizer_name,
                            )
    except KeyboardInterrupt:
        print("Classifier training interrupted. Saving the latest checkpoint before exiting...")
        if checkpoint_root is not None:
            save_checkpoint(
                model,
                checkpoint_root / "latest.pth",
                optimizer=optimizer,
                model_config=model_config,
                metadata=build_metadata(epoch if 'epoch' in locals() else start_epoch),
                tokenizer_name=tokenizer_name,
            )
        raise

    if checkpoint_root is not None:
        final_epoch = start_epoch + num_epochs - 1
        save_checkpoint(
            model,
            checkpoint_root / "latest.pth",
            optimizer=optimizer,
            model_config=model_config,
            metadata=build_metadata(final_epoch),
            tokenizer_name=tokenizer_name,
        )
        if not (checkpoint_root / "best.pth").exists():
            save_checkpoint(
                model,
                checkpoint_root / "best.pth",
                optimizer=optimizer,
                model_config=model_config,
                metadata=build_metadata(final_epoch),
                tokenizer_name=tokenizer_name,
            )

    return train_losses, val_losses, train_accs, val_accs
