from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_llm.chat_runtime import (
    describe_runtime,
    generate_chat_response,
    load_chat_runtime,
    override_generation_config,
)
from small_llm.config import DEFAULT_CHAT_GENERATION
from small_llm.spam_classifier import classify_text, describe_spam_classifier, load_spam_classifier_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the model from the terminal using a trained, base, or untrained runtime."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "chat", "base", "untrained"),
        default="auto",
        help="Which runtime to use. 'auto' prefers chat, then base, then falls back to untrained.",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Optional explicit checkpoint path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--classifier-checkpoint", type=str, default="", help="Optional explicit classifier checkpoint.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CHAT_GENERATION.temperature)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CHAT_GENERATION.top_k)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_CHAT_GENERATION.max_new_tokens)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def print_banner() -> None:
    print("CLI chat is ready.")
    print("Commands: /reset clears history, /info shows runtime info, /quit exits.")
    print("Mode switches: --classifier enables spam classification, --slm switches back to the language model.")
    print()


def main() -> None:
    args = parse_args()
    generation_config = override_generation_config(
        DEFAULT_CHAT_GENERATION,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    runtime = load_chat_runtime(
        mode=args.mode,
        checkpoint_path=args.checkpoint or None,
        device_name=args.device,
        generation_config=generation_config,
        seed=args.seed,
    )
    classifier_runtime = None
    active_mode = "slm"

    print(describe_runtime(runtime))
    print_banner()

    history: list[dict[str, str]] = []
    while True:
        try:
            user_message = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting CLI chat.")
            break

        if not user_message:
            continue
        if user_message.lower() in {"/quit", "/exit"}:
            print("Exiting CLI chat.")
            break
        if user_message == "--classifier":
            classifier_runtime = load_spam_classifier_runtime(
                checkpoint_path=args.classifier_checkpoint or None,
                device_name=args.device,
                allow_heuristic_fallback=True,
            )
            active_mode = "classifier"
            print(describe_spam_classifier(classifier_runtime))
            print()
            continue
        if user_message == "--slm":
            runtime = load_chat_runtime(
                mode=args.mode,
                checkpoint_path=args.checkpoint or None,
                device_name=args.device,
                generation_config=generation_config,
                seed=args.seed,
            )
            active_mode = "slm"
            print(describe_runtime(runtime))
            print()
            continue
        if user_message.lower() == "/reset":
            history = []
            print("History cleared.")
            continue
        if user_message.lower() == "/info":
            if active_mode == "classifier" and classifier_runtime is not None:
                print(describe_spam_classifier(classifier_runtime))
            else:
                print(describe_runtime(runtime))
            continue

        started_at = time.perf_counter()
        if active_mode == "classifier":
            if classifier_runtime is None:
                print("No classifier runtime is loaded. Use --classifier to switch into classifier mode.")
                print()
                continue
            is_spam = classify_text(classifier_runtime, user_message)
            response = "true" if is_spam else "false"
        else:
            response, history = generate_chat_response(runtime, history, user_message)
        elapsed = time.perf_counter() - started_at
        print(f"Model> {response}")
        print(f"[generated in {elapsed:.2f}s]")
        print()


if __name__ == "__main__":
    main()
