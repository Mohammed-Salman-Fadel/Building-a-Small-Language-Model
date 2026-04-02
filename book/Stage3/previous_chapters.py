from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm import (
    GPTDatasetV1,
    GPTModel,
    MultiHeadAttention,
    calc_loss_batch,
    calc_loss_loader,
    create_dataloader_v1,
    download_and_load_gpt2,
    evaluate_model,
    generate,
    generate_and_print_sample,
    generate_text_simple,
    load_weights_into_gpt,
    plot_losses,
    text_to_token_ids,
    token_ids_to_text,
    train_model_simple,
)

__all__ = [
    "GPTDatasetV1",
    "GPTModel",
    "MultiHeadAttention",
    "calc_loss_batch",
    "calc_loss_loader",
    "create_dataloader_v1",
    "download_and_load_gpt2",
    "evaluate_model",
    "generate",
    "generate_and_print_sample",
    "generate_text_simple",
    "load_weights_into_gpt",
    "plot_losses",
    "text_to_token_ids",
    "token_ids_to_text",
    "train_model_simple",
]
