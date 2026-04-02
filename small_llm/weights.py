from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .paths import GPT2_ARTIFACT_DIR, ensure_project_dirs


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load the original GPT-2 checkpoints. "
            "Install the project requirements before calling download_and_load_gpt2."
        ) from exc
    return tf


def download_and_load_gpt2(model_size: str, models_dir: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    ensure_project_dirs()
    tf = _import_tensorflow()

    models_root = Path(models_dir) if models_dir is not None else GPT2_ARTIFACT_DIR
    model_dir = models_root / model_size
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        backup_url = f"{backup_base_url}/{model_size}/{filename}"
        destination = model_dir / filename
        download_file(file_url, destination, backup_url=backup_url)

    tf_ckpt_path = tf.train.latest_checkpoint(str(model_dir))
    settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def download_file(url: str, destination: str | Path, backup_url: str | None = None) -> Path:
    destination_path = Path(destination)

    def _attempt_download(download_url: str) -> bool:
        with urllib.request.urlopen(download_url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            if destination_path.exists() and file_size == destination_path.stat().st_size:
                print(f"File already exists and is up-to-date: {destination_path}")
                return True

            progress_name = os.path.basename(download_url)
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_name) as progress_bar:
                with destination_path.open("wb") as file:
                    while True:
                        chunk = response.read(1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return destination_path
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None and _attempt_download(backup_url):
            return destination_path
        raise RuntimeError(
            f"Failed to download the GPT-2 file from {url}"
            + (f" or {backup_url}" if backup_url else "")
        )

    return destination_path


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    tf = _import_tensorflow()
    params: dict[str, Any] = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[variable_name_parts[-1]] = variable_array

    return params


def assign(left: torch.Tensor, right: np.ndarray) -> torch.nn.Parameter:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: torch.nn.Module, params: dict[str, Any]) -> None:
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block_index in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][block_index]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[block_index].att.W_query.weight = assign(
            gpt.trf_blocks[block_index].att.W_query.weight,
            q_w.T,
        )
        gpt.trf_blocks[block_index].att.W_key.weight = assign(
            gpt.trf_blocks[block_index].att.W_key.weight,
            k_w.T,
        )
        gpt.trf_blocks[block_index].att.W_value.weight = assign(
            gpt.trf_blocks[block_index].att.W_value.weight,
            v_w.T,
        )

        q_b, k_b, v_b = np.split(params["blocks"][block_index]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[block_index].att.W_query.bias = assign(gpt.trf_blocks[block_index].att.W_query.bias, q_b)
        gpt.trf_blocks[block_index].att.W_key.bias = assign(gpt.trf_blocks[block_index].att.W_key.bias, k_b)
        gpt.trf_blocks[block_index].att.W_value.bias = assign(gpt.trf_blocks[block_index].att.W_value.bias, v_b)

        gpt.trf_blocks[block_index].att.out_proj.weight = assign(
            gpt.trf_blocks[block_index].att.out_proj.weight,
            params["blocks"][block_index]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[block_index].att.out_proj.bias = assign(
            gpt.trf_blocks[block_index].att.out_proj.bias,
            params["blocks"][block_index]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[block_index].ff.layers[0].weight = assign(
            gpt.trf_blocks[block_index].ff.layers[0].weight,
            params["blocks"][block_index]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[block_index].ff.layers[0].bias = assign(
            gpt.trf_blocks[block_index].ff.layers[0].bias,
            params["blocks"][block_index]["mlp"]["c_fc"]["b"],
        )
        gpt.trf_blocks[block_index].ff.layers[2].weight = assign(
            gpt.trf_blocks[block_index].ff.layers[2].weight,
            params["blocks"][block_index]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[block_index].ff.layers[2].bias = assign(
            gpt.trf_blocks[block_index].ff.layers[2].bias,
            params["blocks"][block_index]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[block_index].norm1.scale = assign(
            gpt.trf_blocks[block_index].norm1.scale,
            params["blocks"][block_index]["ln_1"]["g"],
        )
        gpt.trf_blocks[block_index].norm1.shift = assign(
            gpt.trf_blocks[block_index].norm1.shift,
            params["blocks"][block_index]["ln_1"]["b"],
        )
        gpt.trf_blocks[block_index].norm2.scale = assign(
            gpt.trf_blocks[block_index].norm2.scale,
            params["blocks"][block_index]["ln_2"]["g"],
        )
        gpt.trf_blocks[block_index].norm2.shift = assign(
            gpt.trf_blocks[block_index].norm2.shift,
            params["blocks"][block_index]["ln_2"]["b"],
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
