"""Evaluate trained CLIP-only and motion-aware models on UCF101 subset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import load_config
from data import VideoActionDataset, collate_basic, collate_motion_aware
from models import CLIPEncoder, MotionSenseFusion

from .metrics import accuracy


def run_benchmark(
    cfg: Dict[str, Any],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """
    Load checkpoint, run evaluation on the dataset from config, return accuracy.

    Args:
        cfg: Config dict (from load_config).
        checkpoint_path: Path to clip_only_head.pt. If None, uses
            cfg["paths"]["checkpoints"] / "clip_only_head.pt".
        device: Device string or torch.device. If None, uses cuda if available.

    Returns:
        Dict with "clip_only_acc": float.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    paths_cfg = cfg["paths"]
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(paths_cfg["checkpoints"]) / "clip_only_head.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run training first: python scripts/train.py"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    clip_head_state = ckpt["clip_head_state"]

    data_cfg = cfg["data"]
    dataset = VideoActionDataset(
        root=data_cfg["root"],
        classes=data_cfg.get("classes"),
        sample_rate=data_cfg.get("frame_sample_rate", 1),
        max_frames=data_cfg.get("max_frames"),
        img_size=(data_cfg.get("img_size", 224), data_cfg.get("img_size", 224)),
        max_clips_per_class=data_cfg.get("max_clips_per_class"),
    )

    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Dataset class_to_idx does not match checkpoint. "
            "Ensure eval uses same config/classes as training."
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_basic,
    )

    model_cfg = cfg["model"]
    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    head = nn.Linear(clip_encoder.embed_dim, dataset.num_classes).to(device)
    head.load_state_dict(clip_head_state)
    head.eval()
    clip_encoder.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            clip_emb = clip_encoder(frames)
            logits = head(clip_emb)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total else 0.0
    return {"clip_only_acc": acc}


def run_benchmark_motion_aware(
    cfg: Dict[str, Any],
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """
    Evaluate motion-aware (CLIP + flow fusion) checkpoint on the dataset from config.

    Checkpoint must contain: fusion_state, class_to_idx, config.
    Uses same data root as cfg (e.g. val set for fair comparison).

    Returns:
        Dict with "motion_aware_acc": float.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    paths_cfg = cfg["paths"]
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(paths_cfg["checkpoints"]) / "motion_aware.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Motion-aware checkpoint not found: {ckpt_path}. "
            "Train motion-aware model first and save as motion_aware.pt"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    fusion_state = ckpt["fusion_state"]
    cfg_ckpt = ckpt.get("config", cfg)

    data_cfg = cfg["data"]
    model_cfg = cfg_ckpt["model"]
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

    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Dataset class_to_idx does not match checkpoint. "
            "Use same config/classes as training."
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motion_aware,
    )

    clip_encoder = CLIPEncoder(model_name=model_cfg["clip_name"], device=str(device))
    fusion = MotionSenseFusion(
        clip_dim=clip_encoder.embed_dim,
        flow_dim=flow_dim,
        num_classes=dataset.num_classes,
        hidden_dim=model_cfg.get("fusion_hidden", 128),
    ).to(device)
    fusion.load_state_dict(fusion_state)
    fusion.eval()
    clip_encoder.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for frames, flow_stats, labels in loader:
            frames = frames.to(device)
            flow_stats = flow_stats.to(device)
            labels = labels.to(device)

            clip_emb = clip_encoder(frames)
            logits = fusion(clip_emb, flow_stats)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total else 0.0
    return {"motion_aware_acc": acc}
