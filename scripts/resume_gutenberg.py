from __future__ import annotations

import sys

import train_gutenberg


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--resume", *sys.argv[1:]]
    train_gutenberg.main()
