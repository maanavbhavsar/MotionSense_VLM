"""Run CLIP-only vs motion-aware comparison on the validation set.

Prints a single table with both accuracies (or N/A if a checkpoint is missing).
Use config/val.yaml so both models are evaluated on the same val split.

Usage
-----
From the project root:

    python scripts/run_comparison.py
    python scripts/run_comparison.py --val-config config/val.yaml
    python scripts/run_comparison.py --motion-checkpoint outputs/checkpoints/motion_aware.pt
"""

import argparse
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import load_config
from eval import run_benchmark, run_benchmark_motion_aware


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CLIP-only vs motion-aware accuracy on validation set.",
    )
    parser.add_argument(
        "--val-config",
        type=str,
        default="config/val.yaml",
        help="Config for val set (default: config/val.yaml).",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=str,
        default=None,
        help="CLIP-only checkpoint (default: cfg paths.checkpoints / clip_only_head.pt).",
    )
    parser.add_argument(
        "--motion-checkpoint",
        type=str,
        default=None,
        help="Motion-aware checkpoint (default: cfg paths.checkpoints / motion_aware.pt).",
    )
    args = parser.parse_args()

    cfg = load_config(args.val_config)

    # ---- CLIP-only ----
    try:
        clip_results = run_benchmark(cfg, checkpoint_path=args.clip_checkpoint)
        clip_acc = clip_results["clip_only_acc"]
    except FileNotFoundError as e:
        clip_acc = None
        print(f"CLIP-only: {e}", file=sys.stderr)

    # ---- Motion-aware ----
    try:
        motion_results = run_benchmark_motion_aware(cfg, checkpoint_path=args.motion_checkpoint)
        motion_acc = motion_results["motion_aware_acc"]
    except FileNotFoundError as e:
        motion_acc = None
        print(f"Motion-aware: {e}", file=sys.stderr)

    # ---- Table ----
    print()
    print("Validation set (config: {})".format(args.val_config))
    print("-" * 50)
    print("Model              | Val accuracy")
    print("-" * 50)
    clip_str = f"{clip_acc * 100:.2f}%" if clip_acc is not None else "N/A (no checkpoint)"
    motion_str = f"{motion_acc * 100:.2f}%" if motion_acc is not None else "N/A (no checkpoint)"
    print("CLIP-only          | {}".format(clip_str))
    print("Motion-aware       | {}".format(motion_str))
    print("-" * 50)


if __name__ == "__main__":
    main()
