from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.weights import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt

__all__ = ["download_and_load_gpt2", "load_gpt2_params_from_tf_ckpt"]
