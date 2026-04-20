from __future__ import annotations

import json
import random
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .paths import (
    INSTRUCTION_PROCESSED_DIR,
    get_instruction_data_candidates,
    get_instruction_response_candidates,
)


DEFAULT_INSTRUCTION_DATASET = "dolly"
DEFAULT_DOLLY_DATASET_NAME = "databricks/databricks-dolly-15k"
DEFAULT_DOLLY_MAX_SAMPLES: int | None = None
DEFAULT_OASST_DATASET_NAME = "OpenAssistant/oasst1"
DEFAULT_OASST_LANG = "en"
DEFAULT_OASST_MAX_SAMPLES = 12000
DEFAULT_ULTRACHAT_DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DEFAULT_ULTRACHAT_MAX_SAMPLES = 20000
DEFAULT_SPLIT_SEED = 123


def _build_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _import_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required for OASST1 preparation. "
            "Install the project requirements before using OASST1 fine-tuning."
        ) from exc
    return load_dataset


def format_input(entry: dict[str, Any]) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def _materialize_instruction_file(source_path: Path) -> Path:
    INSTRUCTION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target_path = INSTRUCTION_PROCESSED_DIR / source_path.name
    if source_path.resolve() != target_path.resolve() and not target_path.exists():
        shutil.copy2(source_path, target_path)
    return target_path if target_path.exists() else source_path


def _oasst_cache_path(lang: str, max_samples: int | None) -> Path:
    label = "full" if max_samples is None else f"max-{max_samples}"
    return INSTRUCTION_PROCESSED_DIR / f"oasst1-{lang}-{label}-splits.json"


def _dolly_cache_path(max_samples: int | None) -> Path:
    label = "full" if max_samples is None else f"max-{max_samples}"
    return INSTRUCTION_PROCESSED_DIR / f"dolly-{label}-splits.json"


def _ultrachat_cache_path(max_samples: int | None) -> Path:
    label = "full" if max_samples is None else f"max-{max_samples}"
    return INSTRUCTION_PROCESSED_DIR / f"ultrachat-{label}-splits.json"


def load_instruction_data(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is not None:
        source_path = Path(path)
    else:
        candidates = get_instruction_data_candidates()
        source_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_path is None:
            raise FileNotFoundError("Could not find instruction-data.json in canonical or legacy locations.")

    materialized_path = _materialize_instruction_file(source_path)
    return json.loads(materialized_path.read_text(encoding="utf-8"))


def split_instruction_data(
    data: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    shuffled_data = list(data)
    random.Random(seed).shuffle(shuffled_data)

    train_portion = int(len(shuffled_data) * 0.85)
    test_portion = int(len(shuffled_data) * 0.1)

    train_data = shuffled_data[:train_portion]
    test_data = shuffled_data[train_portion : train_portion + test_portion]
    val_data = shuffled_data[train_portion + test_portion :]

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }


def _normalize_role(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _message_is_usable(message: dict[str, Any], *, lang: str) -> bool:
    text = str(message.get("text", "") or "").strip()
    if not text:
        return False
    if message.get("deleted") is True:
        return False
    message_lang = str(message.get("lang", "") or "").strip().lower()
    if lang and message_lang and message_lang != lang.lower():
        return False
    review_result = message.get("review_result")
    if review_result is False:
        return False
    return True


def _build_conversation_context(message_id: str, messages_by_id: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current_id = message_id

    while current_id:
        message = messages_by_id.get(current_id)
        if message is None:
            break
        chain.append(
            {
                "role": _normalize_role(message.get("role")),
                "text": str(message.get("text", "") or "").strip(),
            }
        )
        parent_id = message.get("parent_id")
        current_id = str(parent_id) if parent_id else ""

    chain.reverse()
    return [entry for entry in chain if entry["text"]]


def _conversation_to_text(conversation: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for turn in conversation:
        speaker = "User" if turn["role"] == "prompter" else "Assistant"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def _to_conversation_entry(user_message: str, assistant_message: str) -> dict[str, Any]:
    return {
        "instruction": "Continue the conversation below and respond helpfully to the latest user message.",
        "input": f"User: {user_message.strip()}",
        "output": assistant_message.strip(),
    }


def _is_low_quality_response(text: str) -> bool:
    lowered = text.lower()
    blocked_phrases = (
        "as an ai language model",
        "i am an ai language model",
        "i'm an ai language model",
        "my name is open assistant",
        "i am open assistant",
        "i'm open assistant",
    )
    return any(phrase in lowered for phrase in blocked_phrases)


def _row_rank(row: dict[str, Any]) -> int:
    value = row.get("rank")
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rows_to_oasst_entries(rows: list[dict[str, Any]], *, lang: str) -> list[dict[str, Any]]:
    messages_by_id = {
        str(row["message_id"]): row
        for row in rows
        if row.get("message_id") and _message_is_usable(row, lang=lang)
    }

    entries: list[dict[str, Any]] = []
    for row in rows:
        if _normalize_role(row.get("role")) != "assistant":
            continue
        if not _message_is_usable(row, lang=lang):
            continue
        if _row_rank(row) != 0:
            continue

        parent_id = row.get("parent_id")
        if not parent_id:
            continue
        parent = messages_by_id.get(str(parent_id))
        if parent is None or _normalize_role(parent.get("role")) != "prompter":
            continue

        conversation = _build_conversation_context(str(parent_id), messages_by_id)
        if not conversation:
            continue

        output = str(row.get("text", "") or "").strip()
        if _is_low_quality_response(output):
            continue

        entries.append(
            {
                "instruction": "Continue the conversation below and respond helpfully to the latest user message.",
                "input": _conversation_to_text(conversation),
                "output": output,
            }
        )

    return entries


def _dolly_row_to_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    instruction = str(row.get("instruction", "") or "").strip()
    context = str(row.get("context", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    if not instruction or not response or _is_low_quality_response(response):
        return None

    user_message = instruction
    if context:
        user_message = f"{instruction}\n\nContext:\n{context}"
    return _to_conversation_entry(user_message, response)


def _ultrachat_row_to_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []

    entries: list[dict[str, Any]] = []
    for index in range(len(messages) - 1):
        current = messages[index]
        following = messages[index + 1]
        if not isinstance(current, dict) or not isinstance(following, dict):
            continue
        if current.get("role") != "user" or following.get("role") != "assistant":
            continue

        user_message = str(current.get("content", "") or "").strip()
        assistant_message = str(following.get("content", "") or "").strip()
        if not user_message or not assistant_message or _is_low_quality_response(assistant_message):
            continue
        entries.append(_to_conversation_entry(user_message, assistant_message))

    return entries


def _maybe_cap_entries(entries: list[dict[str, Any]], *, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    shuffled_entries = list(entries)
    random.Random(seed).shuffle(shuffled_entries)
    if max_samples is not None:
        return shuffled_entries[:max_samples]
    return shuffled_entries


def _filter_examples_by_token_length(
    split_map: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    *,
    max_length: int | None,
) -> dict[str, list[dict[str, Any]]]:
    if max_length is None:
        return split_map

    filtered: dict[str, list[dict[str, Any]]] = {}
    for split_name, entries in split_map.items():
        kept_entries: list[dict[str, Any]] = []
        for entry in entries:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            token_count = len(tokenizer.encode(instruction_plus_input + response_text))
            if token_count <= max_length:
                kept_entries.append(entry)
        if not kept_entries:
            raise ValueError(
                f"No {split_name} instruction examples fit the allowed_max_length={max_length}. "
                "Increase the model context length or use a shorter fine-tuning dataset."
            )
        filtered[split_name] = kept_entries
    return filtered


def load_oasst1_instruction_splits(
    *,
    lang: str = DEFAULT_OASST_LANG,
    max_samples: int | None = DEFAULT_OASST_MAX_SAMPLES,
    force_rebuild: bool = False,
    dataset_name: str = DEFAULT_OASST_DATASET_NAME,
) -> dict[str, list[dict[str, Any]]]:
    INSTRUCTION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _oasst_cache_path(lang, max_samples)
    if cache_path.exists() and not force_rebuild:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    load_dataset = _import_datasets()
    print(f"Preparing OASST1 instruction data from '{dataset_name}' (lang={lang}, max_samples={max_samples})...")

    train_rows = [dict(row) for row in load_dataset(dataset_name, split="train")]
    validation_rows = [dict(row) for row in load_dataset(dataset_name, split="validation")]

    train_entries = _rows_to_oasst_entries(train_rows, lang=lang)
    validation_entries = _rows_to_oasst_entries(validation_rows, lang=lang)

    combined_entries = train_entries + validation_entries
    random.Random(DEFAULT_SPLIT_SEED).shuffle(combined_entries)
    if max_samples is not None:
        combined_entries = combined_entries[:max_samples]

    split_map = split_instruction_data(combined_entries, seed=DEFAULT_SPLIT_SEED)
    cache_path.write_text(json.dumps(split_map, ensure_ascii=True), encoding="utf-8")
    print(
        "Prepared OASST1 splits: "
        f"train={len(split_map['train'])}, validation={len(split_map['validation'])}, test={len(split_map['test'])}"
    )
    return split_map


def load_dolly_instruction_splits(
    *,
    max_samples: int | None = DEFAULT_DOLLY_MAX_SAMPLES,
    force_rebuild: bool = False,
    dataset_name: str = DEFAULT_DOLLY_DATASET_NAME,
) -> dict[str, list[dict[str, Any]]]:
    INSTRUCTION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _dolly_cache_path(max_samples)
    if cache_path.exists() and not force_rebuild:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    load_dataset = _import_datasets()
    print(f"Preparing Dolly instruction data from '{dataset_name}' (max_samples={max_samples})...")

    rows = [dict(row) for row in load_dataset(dataset_name, split="train")]
    entries = [entry for row in rows if (entry := _dolly_row_to_entry(row)) is not None]
    entries = _maybe_cap_entries(entries, max_samples=max_samples, seed=DEFAULT_SPLIT_SEED)
    split_map = split_instruction_data(entries, seed=DEFAULT_SPLIT_SEED)
    cache_path.write_text(json.dumps(split_map, ensure_ascii=True), encoding="utf-8")
    print(
        "Prepared Dolly splits: "
        f"train={len(split_map['train'])}, validation={len(split_map['validation'])}, test={len(split_map['test'])}"
    )
    return split_map


def load_ultrachat_instruction_splits(
    *,
    max_samples: int | None = DEFAULT_ULTRACHAT_MAX_SAMPLES,
    force_rebuild: bool = False,
    dataset_name: str = DEFAULT_ULTRACHAT_DATASET_NAME,
) -> dict[str, list[dict[str, Any]]]:
    INSTRUCTION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _ultrachat_cache_path(max_samples)
    if cache_path.exists() and not force_rebuild:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    load_dataset = _import_datasets()
    print(f"Preparing UltraChat instruction data from '{dataset_name}' (max_samples={max_samples})...")

    rows = [dict(row) for row in load_dataset(dataset_name, split="train_sft")]
    entries: list[dict[str, Any]] = []
    for row in rows:
        entries.extend(_ultrachat_row_to_entries(row))

    entries = _maybe_cap_entries(entries, max_samples=max_samples, seed=DEFAULT_SPLIT_SEED)
    split_map = split_instruction_data(entries, seed=DEFAULT_SPLIT_SEED)
    cache_path.write_text(json.dumps(split_map, ensure_ascii=True), encoding="utf-8")
    print(
        "Prepared UltraChat splits: "
        f"train={len(split_map['train'])}, validation={len(split_map['validation'])}, test={len(split_map['test'])}"
    )
    return split_map


def load_instruction_splits(
    *,
    dataset_name: str = DEFAULT_INSTRUCTION_DATASET,
    source_path: str | Path | None = None,
    oasst_lang: str = DEFAULT_OASST_LANG,
    oasst_max_samples: int | None = DEFAULT_OASST_MAX_SAMPLES,
    dolly_max_samples: int | None = DEFAULT_DOLLY_MAX_SAMPLES,
    ultrachat_max_samples: int | None = DEFAULT_ULTRACHAT_MAX_SAMPLES,
    force_rebuild: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    normalized = dataset_name.lower()
    if normalized == "local":
        data = load_instruction_data(path=source_path)
        return split_instruction_data(data, seed=DEFAULT_SPLIT_SEED)
    if normalized == "dolly":
        return load_dolly_instruction_splits(
            max_samples=dolly_max_samples,
            force_rebuild=force_rebuild,
        )
    if normalized == "ultrachat":
        return load_ultrachat_instruction_splits(
            max_samples=ultrachat_max_samples,
            force_rebuild=force_rebuild,
        )
    if normalized == "oasst1":
        return load_oasst1_instruction_splits(
            lang=oasst_lang,
            max_samples=oasst_max_samples,
            force_rebuild=force_rebuild,
        )
    raise ValueError(f"Unsupported instruction dataset: {dataset_name}")


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict[str, Any]], tokenizer: Any) -> None:
        self.data = data
        self.encoded_texts: list[tuple[list[int], int]] = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            prompt_tokens = tokenizer.encode(instruction_plus_input)
            full_tokens = tokenizer.encode(instruction_plus_input + response_text)
            self.encoded_texts.append((full_tokens, len(prompt_tokens)))

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def custom_collate_fn(
    batch: list[list[int] | tuple[list[int], int]],
    *,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_batch: list[tuple[list[int], int]] = []
    for item in batch:
        if isinstance(item, tuple):
            token_ids, prompt_length = item
        else:
            token_ids, prompt_length = item, 0
        normalized_batch.append((token_ids, prompt_length))

    batch_max_length = max(len(token_ids) + 1 for token_ids, _ in normalized_batch)
    inputs_lst: list[torch.Tensor] = []
    targets_lst: list[torch.Tensor] = []

    for token_ids, prompt_length in normalized_batch:
        new_item = token_ids.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        # In instruction tuning we care about learning the assistant reply, not
        # re-predicting the prompt template and user transcript.
        response_start = min(max(0, prompt_length - 1), targets.numel())
        if response_start > 0:
            targets[:response_start] = ignore_index

        mask = targets == pad_token_id
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def create_instruction_dataloaders(
    tokenizer: Any,
    *,
    batch_size: int = 8,
    allowed_max_length: int | None = None,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    source_path: str | Path | None = None,
    dataset_name: str = DEFAULT_INSTRUCTION_DATASET,
    oasst_lang: str = DEFAULT_OASST_LANG,
    oasst_max_samples: int | None = DEFAULT_OASST_MAX_SAMPLES,
    dolly_max_samples: int | None = DEFAULT_DOLLY_MAX_SAMPLES,
    ultrachat_max_samples: int | None = DEFAULT_ULTRACHAT_MAX_SAMPLES,
    force_rebuild: bool = False,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    split_map = load_instruction_splits(
        dataset_name=dataset_name,
        source_path=source_path,
        oasst_lang=oasst_lang,
        oasst_max_samples=oasst_max_samples,
        dolly_max_samples=dolly_max_samples,
        ultrachat_max_samples=ultrachat_max_samples,
        force_rebuild=force_rebuild,
    )
    split_map = _filter_examples_by_token_length(
        split_map,
        tokenizer,
        max_length=allowed_max_length,
    )

    collate = partial(
        custom_collate_fn,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    train_loader = DataLoader(
        InstructionDataset(split_map["train"], tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        generator=_build_generator(seed),
    )
    val_loader = DataLoader(
        InstructionDataset(split_map["validation"], tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        InstructionDataset(split_map["test"], tokenizer),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, split_map


def load_instruction_responses(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is not None:
        source_path = Path(path)
    else:
        candidates = get_instruction_response_candidates()
        source_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_path is None:
            raise FileNotFoundError(
                "Could not find instruction-data-with-response.json in canonical or legacy locations."
            )

    materialized_path = _materialize_instruction_file(source_path)
    return json.loads(materialized_path.read_text(encoding="utf-8"))
