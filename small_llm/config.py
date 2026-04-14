from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 256
    n_heads: int = 8
    n_layers: int = 8
    drop_rate: float = 0.0
    qkv_bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    max_length: int = 256
    stride: int = 128
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    eval_freq: int = 50
    eval_iter: int = 20
    max_books: int | None = 1000
    seed: int = 123
    warmup_steps: int = 200
    grad_clip: float = 1.0


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 128
    eos_id: int = 50256


@dataclass(frozen=True)
class SpamClassifierConfig:
    num_classes: int = 2
    dropout: float = 0.1
    pad_token_id: int = 50256
    max_length: int = 120


GPT2_SMALL_124M = ModelConfig(
    vocab_size=50257,
    context_length=256,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

GPT2_SMALL_1024 = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.0,
    qkv_bias=True,
)

GPT2_MEDIUM_355M = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    drop_rate=0.0,
    qkv_bias=True,
)

GPT2_LARGE_774M = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1280,
    n_heads=20,
    n_layers=36,
    drop_rate=0.0,
    qkv_bias=True,
)

GPT2_XL_1558M = ModelConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1600,
    n_heads=25,
    n_layers=48,
    drop_rate=0.0,
    qkv_bias=True,
)

DEFAULT_GUTENBERG_MODEL = ModelConfig(
    vocab_size=50257,
    context_length=256,
    emb_dim=384,
    n_heads=6,
    n_layers=10,
    drop_rate=0.0,
    qkv_bias=True,
)
DEFAULT_GUTENBERG_TRAINING = TrainingConfig(
    batch_size=2,
    max_length=256,
    stride=128,
    learning_rate=3e-4,
    min_learning_rate=3e-5,
    weight_decay=0.01,
    num_epochs=3,
    eval_freq=50,
    eval_iter=20,
    max_books=1000,
    seed=123,
    warmup_steps=400,
    grad_clip=1.0,
)
DEFAULT_INSTRUCTION_FINETUNE = TrainingConfig(
    batch_size=4,
    max_length=256,
    stride=256,
    learning_rate=5e-5,
    min_learning_rate=1e-5,
    weight_decay=0.01,
    num_epochs=1,
    eval_freq=25,
    eval_iter=5,
    max_books=None,
    seed=123,
    warmup_steps=50,
    grad_clip=1.0,
)
DEFAULT_SPAM_CLASSIFIER = SpamClassifierConfig()
DEFAULT_SPAM_CLASSIFIER_TRAINING = TrainingConfig(
    batch_size=8,
    max_length=120,
    stride=120,
    learning_rate=5e-5,
    min_learning_rate=1e-5,
    weight_decay=0.01,
    num_epochs=5,
    eval_freq=10,
    eval_iter=20,
    max_books=None,
    seed=123,
    warmup_steps=25,
    grad_clip=1.0,
)
DEFAULT_CHAT_GENERATION = GenerationConfig()


def as_config_dict(config: ModelConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, ModelConfig):
        return config.to_dict()
    return dict(config)


def config_from_dict(config: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config
    return ModelConfig(**dict(config))


def infer_legacy_model_config(path: str | Path) -> ModelConfig | None:
    name = Path(path).name.lower()
    full_path = str(path).lower()

    if "gpt2-medium355m-sft" in name:
        return GPT2_MEDIUM_355M
    if name in {"model.pth", "model_and_optimizer.pth"}:
        return GPT2_SMALL_124M
    if "355m" in full_path:
        return GPT2_MEDIUM_355M
    if "774m" in full_path:
        return GPT2_LARGE_774M
    if "1558m" in full_path:
        return GPT2_XL_1558M
    if "124m" in full_path:
        return GPT2_SMALL_1024
    return None
