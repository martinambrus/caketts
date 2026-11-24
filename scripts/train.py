#!/usr/bin/env python3
"""Run the bounded stage2 acoustic trainer (fp32 CPU or CUDA only).

Checkpoints loaded with --resume must be trusted local files. --max-steps is
an absolute optimizer-step target, allowing bounded continuation of a run.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import run_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    try:
        run_training(args.config, args.resume, args.max_steps)
    except ValueError as error:
        # No transcript text is included in training diagnostics.
        p.exit(2, f"training error: {error}\n")
    except OSError:
        p.exit(2, "training error: input/output operation failed\n")


if __name__ == "__main__":
    main()
