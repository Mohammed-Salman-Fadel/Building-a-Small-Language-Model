from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ModelConfig, as_config_dict, config_from_dict, infer_legacy_model_config
from .generation import generate_text_simple, text_to_token_ids, token_ids_to_text
from .model import GPTModel
from .paths import FIGURES_DIR


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device | str,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(
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
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / batches_to_use


def evaluate_model(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    device: torch.device | str,
    eval_iter: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    start_context: str,
) -> None:
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        decoded_text = token_ids_to_text(token_ids.cpu(), tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tokenizer_name: str = "gpt2",
    is_chat_model: bool = False,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "tokenizer_name": tokenizer_name,
        "is_chat_model": is_chat_model,
        "metadata": dict(metadata or {}),
    }
    if model_config is not None:
        payload["model_config"] = as_config_dict(model_config)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def extract_model_state_dict(checkpoint_payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("Checkpoint payload must be a state_dict or a dict containing a state_dict.")

    for key in ("model_state_dict", "state_dict", "model"):
        state_dict = checkpoint_payload.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    if checkpoint_payload and all(torch.is_tensor(value) for value in checkpoint_payload.values()):
        return checkpoint_payload

    raise ValueError("Could not find a model state dict in the checkpoint payload.")


def _resolve_model_config(
    checkpoint_payload: Any,
    checkpoint_path: str | Path,
    fallback_config: ModelConfig | Mapping[str, Any] | None = None,
) -> ModelConfig:
    if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("model_config"), dict):
        return config_from_dict(checkpoint_payload["model_config"])

    inferred = infer_legacy_model_config(checkpoint_path)
    if inferred is not None:
        return inferred

    if fallback_config is not None:
        return config_from_dict(fallback_config)

    raise ValueError(
        f"Could not infer a model configuration for checkpoint: {checkpoint_path}. "
        "Use checkpoints saved through small_llm.save_checkpoint or pass a fallback config."
    )


def load_model_and_metadata(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    fallback_config: ModelConfig | Mapping[str, Any] | None = None,
) -> tuple[GPTModel, ModelConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    model_config = _resolve_model_config(checkpoint, checkpoint_path, fallback_config=fallback_config)
    model = GPTModel(model_config.to_dict())
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, model_config, metadata


def train_model_simple(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: Any,
    *,
    checkpoint_dir: str | Path | None = None,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    is_chat_model: bool = False,
    warmup_steps: int = 0,
    min_learning_rate: float | None = None,
    grad_clip: float | None = None,
) -> tuple[list[float], list[float], list[int]]:
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    train_losses: list[float] = []
    val_losses: list[float] = []
    track_tokens_seen: list[int] = []
    tokens_seen = 0
    global_step = -1
    best_val = float("inf")
    total_steps = max(1, num_epochs * len(train_loader))
    base_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

    def set_learning_rate(step_index: int) -> float | None:
        if min_learning_rate is None:
            return None

        if warmup_steps > 0 and step_index < warmup_steps:
            warmup_scale = float(step_index + 1) / float(warmup_steps)
            target_lrs = [base_lr * warmup_scale for base_lr in base_lrs]
        else:
            if total_steps <= warmup_steps:
                cosine_progress = 1.0
            else:
                cosine_progress = min(
                    1.0,
                    float(step_index - warmup_steps) / float(total_steps - warmup_steps),
                )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            target_lrs = [
                min_learning_rate + (base_lr - min_learning_rate) * cosine_decay
                for base_lr in base_lrs
            ]

        for param_group, target_lr in zip(optimizer.param_groups, target_lrs):
            param_group["lr"] = target_lr
        return target_lrs[0]

    try:
        for epoch in range(num_epochs):
            model.train()

            for input_batch, target_batch in train_loader:
                global_step += 1
                current_lr = set_learning_rate(global_step)

                optimizer.zero_grad(set_to_none=True)
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                tokens_seen += input_batch.numel()

                if eval_freq > 0 and global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    train_ppl = math.exp(min(train_loss, 20.0))
                    val_ppl = math.exp(min(val_loss, 20.0))
                    lr_display = optimizer.param_groups[0]["lr"] if current_lr is None else current_lr
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                        f"Train ppl {train_ppl:.1f}, Val ppl {val_ppl:.1f}, "
                        f"LR {lr_display:.2e}"
                    )

                    if checkpoint_root is not None:
                        save_checkpoint(
                            model,
                            checkpoint_root / "latest.pth",
                            optimizer=optimizer,
                            model_config=model_config,
                            metadata=metadata,
                            is_chat_model=is_chat_model,
                        )
                        if val_loss < best_val:
                            best_val = val_loss
                            save_checkpoint(
                                model,
                                checkpoint_root / "best.pth",
                                optimizer=optimizer,
                                model_config=model_config,
                                metadata=metadata,
                                is_chat_model=is_chat_model,
                            )

            generate_and_print_sample(model, tokenizer, device, start_context)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the latest checkpoint before exiting...")
        if checkpoint_root is not None:
            save_checkpoint(
                model,
                checkpoint_root / "latest.pth",
                optimizer=optimizer,
                model_config=model_config,
                metadata=metadata,
                is_chat_model=is_chat_model,
            )
            if not (checkpoint_root / "best.pth").exists():
                save_checkpoint(
                    model,
                    checkpoint_root / "best.pth",
                    optimizer=optimizer,
                    model_config=model_config,
                    metadata=metadata,
                    is_chat_model=is_chat_model,
                )
        raise

    if checkpoint_root is not None:
        save_checkpoint(
            model,
            checkpoint_root / "latest.pth",
            optimizer=optimizer,
            model_config=model_config,
            metadata=metadata,
            is_chat_model=is_chat_model,
        )
        if not (checkpoint_root / "best.pth").exists():
            save_checkpoint(
                model,
                checkpoint_root / "best.pth",
                optimizer=optimizer,
                model_config=model_config,
                metadata=metadata,
                is_chat_model=is_chat_model,
            )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(
    epochs_seen: list[float] | list[int],
    tokens_seen: list[float] | list[int],
    train_losses: list[float],
    val_losses: list[float],
    output_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = Path(output_path) if output_path is not None else FIGURES_DIR / "loss-plot.pdf"

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(figure_path)
    plt.close(fig)
    return figure_path
