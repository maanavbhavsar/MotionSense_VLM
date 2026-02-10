"""Train a CLIP-only classifier on a UCF101 subset using config/default.yaml.

Usage
-----
From the project root (MotionSense_VLM):

    python scripts/train.py
    python scripts/train.py --config config/default.yaml
    python -m scripts.train --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on path when running as python scripts/train.py
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import load_config
from data import VideoActionDataset, collate_basic
from models import CLIPEncoder


def build_dataloader(cfg: Dict[str, Any]) -> DataLoader:
    data_cfg = cfg["data"]

    dataset = VideoActionDataset(
        root=data_cfg["root"],
        classes=data_cfg.get("classes"),
        sample_rate=data_cfg.get("frame_sample_rate", 1),
        max_frames=data_cfg.get("max_frames"),
        img_size=(data_cfg.get("img_size", 224), data_cfg.get("img_size", 224)),
        max_clips_per_class=data_cfg.get("max_clips_per_class"),
    )

    train_cfg = cfg["train"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_basic,
    )
    return loader


def build_models(cfg: Dict[str, Any], device: torch.device) -> tuple[CLIPEncoder, nn.Module]:
    model_cfg = cfg["model"]

    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    # Freeze CLIP weights – we only train a small linear head
    for p in clip_encoder.parameters():
        p.requires_grad = False

    # Simple linear classifier on top of CLIP embeddings
    # We infer num_classes from a small helper dataset if needed, but in practice
    # the caller should pass it in.
    raise RuntimeError(
        "build_models() must be called with num_classes; use build_models_with_dataset instead."
    )


def build_models_with_dataset(
    cfg: Dict[str, Any],
    device: torch.device,
    num_classes: int,
) -> tuple[CLIPEncoder, nn.Module]:
    model_cfg = cfg["model"]

    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    for p in clip_encoder.parameters():
        p.requires_grad = False

    head = nn.Linear(clip_encoder.embed_dim, num_classes).to(device)
    return clip_encoder, head


def train(cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build dataset & loader once so we know num_classes
    data_cfg = cfg["data"]
    dataset = VideoActionDataset(
        root=data_cfg["root"],
        classes=data_cfg.get("classes"),
        sample_rate=data_cfg.get("frame_sample_rate", 1),
        max_frames=data_cfg.get("max_frames"),
        img_size=(data_cfg.get("img_size", 224), data_cfg.get("img_size", 224)),
        max_clips_per_class=data_cfg.get("max_clips_per_class"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_basic,
    )

    num_classes = dataset.num_classes
    print(f"Found {len(dataset)} clips across {num_classes} classes.")

    clip_encoder, head = build_models_with_dataset(cfg, device, num_classes)
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg["train"]["lr"])

    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        head.train()
        clip_encoder.eval()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in loader:
            # frames: (B, T, C, H, W)
            frames = frames.to(device)
            labels = labels.to(device)

            # CLIPEncoder internally uses no_grad for the CLIP backbone,
            # and we have requires_grad=False on its params; gradients flow
            # only through the head.
            clip_emb = clip_encoder(frames)  # (B, embed_dim)
            logits = head(clip_emb)          # (B, num_classes)

            loss = ce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * frames.size(0)
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")

    # Save checkpoint
    paths_cfg = cfg["paths"]
    ckpt_dir = Path(paths_cfg["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "clip_only_head.pt"

    torch.save(
        {
            "clip_head_state": head.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP-only head on UCF101 subset.")
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

