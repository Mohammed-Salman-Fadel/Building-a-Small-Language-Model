from __future__ import annotations

import sys

import finetune_chat


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--resume", *sys.argv[1:]]
    finetune_chat.main()
