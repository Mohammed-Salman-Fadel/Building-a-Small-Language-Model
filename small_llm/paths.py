from __future__ import annotations

from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GUTENBERG_RAW_DIR = RAW_DATA_DIR / "gutenberg"
GUTENBERG_PROCESSED_DIR = PROCESSED_DATA_DIR / "gutenberg"
INSTRUCTION_PROCESSED_DIR = PROCESSED_DATA_DIR / "instruction"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
GPT2_ARTIFACT_DIR = ARTIFACTS_DIR / "gpt2"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
GUTENBERG_CHECKPOINT_DIR = CHECKPOINTS_DIR / "gutenberg"
FINETUNED_DIR = ARTIFACTS_DIR / "finetuned"

OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DEFAULT_GUTENBERG_LATEST = GUTENBERG_CHECKPOINT_DIR / "latest.pth"
DEFAULT_GUTENBERG_BEST = GUTENBERG_CHECKPOINT_DIR / "best.pth"
DEFAULT_CHAT_CHECKPOINT = FINETUNED_DIR / "gutenberg-chat.pth"

LEGACY_CHAT_CHECKPOINTS = [
    REPO_ROOT / "models" / "gpt2-medium355M-sft.pth",
    REPO_ROOT / "gpt2-medium355M-sft.pth",
]

LEGACY_BASE_CHECKPOINTS = [
    REPO_ROOT / "models" / "model.pth",
    REPO_ROOT / "Stage2" / "model.pth",
    REPO_ROOT / "model.pth",
    REPO_ROOT / "models" / "model_and_optimizer.pth",
    REPO_ROOT / "Stage2" / "model_and_optimizer.pth",
    REPO_ROOT / "model_and_optimizer.pth",
]

LEGACY_INSTRUCTION_DATA_PATHS = [
    REPO_ROOT / "Stage3" / "instruction_dataset" / "instruction-data.json",
    REPO_ROOT / "Stage3" / "instruction-data.json",
    REPO_ROOT / "book" / "Stage3" / "instruction_dataset" / "instruction-data.json",
    REPO_ROOT / "book" / "Stage3" / "instruction-data.json",
]

LEGACY_INSTRUCTION_RESPONSE_PATHS = [
    REPO_ROOT / "Stage3" / "instruction_dataset" / "instruction-data-with-response.json",
    REPO_ROOT / "book" / "Stage3" / "instruction_dataset" / "instruction-data-with-response.json",
]


def ensure_project_dirs() -> None:
    for path in (
        GUTENBERG_RAW_DIR,
        GUTENBERG_PROCESSED_DIR,
        INSTRUCTION_PROCESSED_DIR,
        GPT2_ARTIFACT_DIR,
        GUTENBERG_CHECKPOINT_DIR,
        FINETUNED_DIR,
        FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def get_chat_checkpoint_candidates() -> list[Path]:
    return [DEFAULT_CHAT_CHECKPOINT, *LEGACY_CHAT_CHECKPOINTS]


def get_base_checkpoint_candidates() -> list[Path]:
    return [DEFAULT_GUTENBERG_LATEST, DEFAULT_GUTENBERG_BEST, *LEGACY_BASE_CHECKPOINTS]


def get_instruction_data_candidates() -> list[Path]:
    return [INSTRUCTION_PROCESSED_DIR / "instruction-data.json", *LEGACY_INSTRUCTION_DATA_PATHS]


def get_instruction_response_candidates() -> list[Path]:
    return [
        INSTRUCTION_PROCESSED_DIR / "instruction-data-with-response.json",
        *LEGACY_INSTRUCTION_RESPONSE_PATHS,
    ]
