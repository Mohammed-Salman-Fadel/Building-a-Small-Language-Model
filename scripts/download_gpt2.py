from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.paths import GPT2_ARTIFACT_DIR, ensure_project_dirs
from small_llm.weights import download_and_load_gpt2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the original GPT-2 checkpoints into the project artifacts.")
    parser.add_argument(
        "--model-size",
        choices=("124M", "355M", "774M", "1558M"),
        default="124M",
        help="Which GPT-2 checkpoint size to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(GPT2_ARTIFACT_DIR),
        help="Directory where the GPT-2 files should be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    settings, _ = download_and_load_gpt2(args.model_size, models_dir=args.output_dir)
    model_dir = Path(args.output_dir) / args.model_size
    print(f"Downloaded GPT-2 {args.model_size} to {model_dir}")
    print(f"Config summary: n_layer={settings['n_layer']}, n_head={settings['n_head']}, n_embd={settings['n_embd']}")


if __name__ == "__main__":
    main()
