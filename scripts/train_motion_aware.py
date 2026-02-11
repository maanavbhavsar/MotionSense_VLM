"""Train the motion-aware model (CLIP + optical flow fusion) on UCF101 subset.

Uses the data pipeline with flow: VideoActionDataset(return_flow_stats=True) and
collate_motion_aware so each batch is (frames, flow_stats, labels). Saves
motion_aware.pt for run_benchmark_motion_aware and run_comparison.py.

Usage
-----
From the project root:

    python scripts/train_motion_aware.py
    python scripts/train_motion_aware.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import load_config
from data import VideoActionDataset, collate_motion_aware
from models import CLIPEncoder, MotionSenseFusion


def train(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    flow_bins = model_cfg.get("flow_bins", 8)
    flow_dim = flow_bins + 2

    dataset = VideoActionDataset(
        root=data_cfg["root"],
        classes=data_cfg.get("classes"),
        sample_rate=data_cfg.get("frame_sample_rate", 1),
        max_frames=data_cfg.get("max_frames"),
        img_size=(data_cfg.get("img_size", 224), data_cfg.get("img_size", 224)),
        max_clips_per_class=data_cfg.get("max_clips_per_class"),
        return_flow_stats=True,
        flow_bins=flow_bins,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_motion_aware,
    )

    num_classes = dataset.num_classes
    print(f"Found {len(dataset)} clips across {num_classes} classes (with flow).")

    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    for p in clip_encoder.parameters():
        p.requires_grad = False

    fusion = MotionSenseFusion(
        clip_dim=clip_encoder.embed_dim,
        flow_dim=flow_dim,
        num_classes=num_classes,
        hidden_dim=model_cfg.get("fusion_hidden", 128),
    ).to(device)

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=cfg["train"]["lr"])
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        fusion.train()
        clip_encoder.eval()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for frames, flow_stats, labels in loader:
            frames = frames.to(device)
            flow_stats = flow_stats.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                clip_emb = clip_encoder(frames)
            logits = fusion(clip_emb, flow_stats)
            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * frames.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "motion_aware.pt"
    torch.save(
        {
            "fusion_state": fusion.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train motion-aware (CLIP + flow) fusion on UCF101 subset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
