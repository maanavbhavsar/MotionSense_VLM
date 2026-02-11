"""Run evaluation on trained CLIP-only checkpoint.

Usage
-----
From the project root (MotionSense_VLM):

    python scripts/run_eval.py
    python scripts/run_eval.py --config config/default.yaml
    python scripts/run_eval.py --checkpoint outputs/checkpoints/clip_only_head.pt
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when running as python scripts/run_eval.py
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import load_config
from eval import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP-only head on UCF101 subset.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: cfg paths.checkpoints / clip_only_head.pt).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_benchmark(cfg, checkpoint_path=args.checkpoint)

    for name, value in results.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
