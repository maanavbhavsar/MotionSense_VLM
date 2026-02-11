"""Run a single-video demo for MotionSense-VLM.

Examples
--------
From the inner MotionSense_VLM project:

    # CLIP-only
    python scripts/run_demo.py --mode clip_only --video PATH/TO/video.avi

    # Motion-aware (requires motion_aware.pt)
    python scripts/run_demo.py --mode motion_aware --video PATH/TO/video.avi

By default this uses config/default.yaml and writes figures to outputs/demo/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# Ensure project root is on path when running as python scripts/run_demo.py
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import load_config
from data import VideoActionDataset
from demos.visualize import (
    visualize_clip_only,
    visualize_motion_aware,
)
from models import CLIPEncoder, MotionSenseFusion


def _load_class_mapping(cfg: Dict, checkpoint_path: Path) -> Dict[int, str]:
    """Load class_to_idx from checkpoint and invert to idx_to_class."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    # Ensure deterministic ordering
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return idx_to_class


def _build_clip_only_models(
    cfg: Dict,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[CLIPEncoder, nn.Module, Dict[int, str]]:
    model_cfg = cfg["model"]
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    clip_head_state = ckpt["clip_head_state"]

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    head = nn.Linear(clip_encoder.embed_dim, len(idx_to_class)).to(device)
    head.load_state_dict(clip_head_state)
    return clip_encoder, head, idx_to_class


def _build_motion_aware_models(
    cfg: Dict,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[CLIPEncoder, MotionSenseFusion, Dict[int, str], int]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    fusion_state = ckpt["fusion_state"]
    cfg_ckpt = ckpt.get("config", cfg)

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    model_cfg = cfg_ckpt["model"]
    flow_bins = model_cfg.get("flow_bins", 8)
    flow_dim = flow_bins + 2

    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    fusion = MotionSenseFusion(
        clip_dim=clip_encoder.embed_dim,
        flow_dim=flow_dim,
        num_classes=len(idx_to_class),
        hidden_dim=model_cfg.get("fusion_hidden", 128),
    ).to(device)
    fusion.load_state_dict(fusion_state)

    return clip_encoder, fusion, idx_to_class, flow_bins


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single-video demo for MotionSense-VLM.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to a video file to demo.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["clip_only", "motion_aware"],
        default="clip_only",
        help="Which model to use for the demo.",
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=str,
        default=None,
        help="Path to clip_only_head.pt (default: cfg paths.checkpoints / clip_only_head.pt).",
    )
    parser.add_argument(
        "--motion-checkpoint",
        type=str,
        default=None,
        help="Path to motion_aware.pt (default: cfg paths.checkpoints / motion_aware.pt).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/demo",
        help="Directory to write demo figures.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths_cfg = cfg["paths"]
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "clip_only":
        ckpt_path = (
            Path(args.clip_checkpoint)
            if args.clip_checkpoint
            else Path(paths_cfg["checkpoints"]) / "clip_only_head.pt"
        )
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"CLIP-only checkpoint not found: {ckpt_path}. "
                "Run scripts/train.py first."
            )

        clip_encoder, head, idx_to_class = _build_clip_only_models(
            cfg, device, ckpt_path
        )
        fig_path = visualize_clip_only(
            video_path=video_path,
            cfg=cfg,
            clip_encoder=clip_encoder,
            head=head,
            idx_to_class=idx_to_class,
            device=device,
            outdir=outdir,
        )
        print(f"Saved CLIP-only demo figure to {fig_path}")

    else:  # motion_aware
        ckpt_path = (
            Path(args.motion_checkpoint)
            if args.motion_checkpoint
            else Path(paths_cfg["checkpoints"]) / "motion_aware.pt"
        )
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Motion-aware checkpoint not found: {ckpt_path}. "
                "Run scripts/train_motion_aware.py first."
            )

        clip_encoder, fusion, idx_to_class, flow_bins = _build_motion_aware_models(
            cfg, device, ckpt_path
        )
        frames_path, flow_path = visualize_motion_aware(
            video_path=video_path,
            cfg=cfg,
            clip_encoder=clip_encoder,
            fusion=fusion,
            idx_to_class=idx_to_class,
            device=device,
            flow_bins=flow_bins,
            outdir=outdir,
        )
        print(f"Saved motion-aware frames figure to {frames_path}")
        print(f"Saved motion-aware flow figure to {flow_path}")


if __name__ == "__main__":
    main()

