"""Visualization helpers for MotionSense-VLM demos.

This module focuses on *offline* visualization: given a video path and
trained models, it runs inference and saves a few concise figures you
can drop into slides or send to reviewers.

Key entrypoints (used by scripts/run_demo.py):
    - visualize_clip_only(...)
    - visualize_motion_aware(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.loader import load_video_frames, frames_to_tensor
from data.flow import compute_flow_stats


@dataclass
class PredictionResult:
    top_classes: List[Tuple[str, float]]
    logits: torch.Tensor
    probs: torch.Tensor


def _ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _frames_grid_figure(
    frames: np.ndarray,
    max_frames: int = 16,
    cols: int = 4,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Create a matplotlib figure showing a grid of RGB frames.

    Args:
        frames: (T, H, W, 3) uint8 RGB.
        max_frames: cap number of frames to display.
        cols: number of columns in the grid.
    """
    T = min(len(frames), max_frames)
    if T == 0:
        raise ValueError("No frames provided for visualization.")

    idxs = np.linspace(0, len(frames) - 1, T).astype(int)
    sel = frames[idxs]

    rows = int(np.ceil(T / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i < T:
            ax.imshow(sel[i])
            ax.set_title(f"t={idxs[i]}")

    fig.tight_layout()
    return fig


def _prediction_text(top_classes: List[Tuple[str, float]], mode: str) -> str:
    lines = [f"Mode: {mode}", "Top predictions:"]
    for cls, p in top_classes:
        lines.append(f"  - {cls}: {p*100:.1f}%")
    return "\n".join(lines)


def _attach_caption(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        fontsize=9,
        ha="left",
        va="bottom",
        family="monospace",
    )


def _flow_hist_figure(
    flow_stats: np.ndarray,
    num_bins: int,
    figsize: Tuple[int, int] = (6, 4),
) -> plt.Figure:
    """Plot flow magnitude stats + direction histogram."""
    if flow_stats.shape[0] != num_bins + 2:
        raise ValueError(
            f"Expected flow_stats shape ({num_bins + 2},), got {flow_stats.shape}"
        )
    mean_mag, std_mag = flow_stats[0], flow_stats[1]
    hist = flow_stats[2:]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bins = np.arange(num_bins)
    ax.bar(bins, hist)
    ax.set_xticks(bins)
    ax.set_xlabel("Direction bin")
    ax.set_ylabel("Normalized count")
    ax.set_title(f"Flow histogram (mean={mean_mag:.3f}, std={std_mag:.3f})")
    fig.tight_layout()
    return fig


def compute_topk(
    logits: torch.Tensor,
    idx_to_class: Dict[int, str],
    k: int = 5,
) -> PredictionResult:
    """Convert logits to top-k (class, prob) pairs."""
    probs = F.softmax(logits, dim=-1)  # (1, C)
    topk = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    top_classes: List[Tuple[str, float]] = []
    for cls_idx, p in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
        name = idx_to_class.get(cls_idx, f"class_{cls_idx}")
        top_classes.append((name, p))
    return PredictionResult(top_classes=top_classes, logits=logits, probs=probs)


def infer_clip_only(
    video_path: str | Path,
    cfg: Dict,
    clip_encoder: torch.nn.Module,
    head: torch.nn.Module,
    idx_to_class: Dict[int, str],
    device: torch.device,
) -> Tuple[np.ndarray, PredictionResult]:
    """Run CLIP-only inference on a single video and return frames + prediction."""
    data_cfg = cfg["data"]
    frames = load_video_frames(
        video_path,
        sample_rate=data_cfg.get("frame_sample_rate", 1),
        max_frames=data_cfg.get("max_frames"),
        img_size=(data_cfg.get("img_size", 224), data_cfg.get("img_size", 224)),
    )  # (T, H, W, 3) uint8
    tensor = frames_to_tensor(frames)  # (T, 3, H, W)
    tensor = tensor.unsqueeze(0).to(device)  # (1, T, 3, H, W)

    clip_encoder.eval()
    head.eval()
    with torch.no_grad():
        clip_emb = clip_encoder(tensor)  # (1, D)
        logits = head(clip_emb)  # (1, C)

    pred = compute_topk(logits, idx_to_class, k=5)
    return frames, pred


def infer_motion_aware(
    video_path: str | Path,
    cfg: Dict,
    clip_encoder: torch.nn.Module,
    fusion: torch.nn.Module,
    idx_to_class: Dict[int, str],
    device: torch.device,
    flow_bins: int,
) -> Tuple[np.ndarray, np.ndarray, PredictionResult]:
    """Run motion-aware inference (CLIP + flow fusion) on a single video."""
    data_cfg = cfg["data"]
    frames = load_video_frames(
        video_path,
        sample_rate=data_cfg.get("frame_sample_rate", 1),
        max_frames=data_cfg.get("max_frames"),
        img_size=(data_cfg.get("img_size", 224), data_cfg.get("img_size", 224)),
    )  # (T, H, W, 3) uint8
    tensor = frames_to_tensor(frames)  # (T, 3, H, W), CLIP-normalized
    tensor = tensor.to(device)

    # Flow must be computed from raw [0,255] frames; tensor is CLIP-normalized (invalid for flow).
    flow_stats = compute_flow_stats(frames, num_bins=flow_bins)  # (bins+2,)
    flow_tensor = torch.from_numpy(flow_stats).float().unsqueeze(0).to(device)  # (1, F)

    clip_encoder.eval()
    fusion.eval()
    with torch.no_grad():
        clip_emb = clip_encoder(tensor.unsqueeze(0))  # (1, D)
        logits = fusion(clip_emb, flow_tensor)  # (1, C)

    pred = compute_topk(logits, idx_to_class, k=5)
    return frames, flow_stats, pred


def visualize_clip_only(
    video_path: str | Path,
    cfg: Dict,
    clip_encoder: torch.nn.Module,
    head: torch.nn.Module,
    idx_to_class: Dict[int, str],
    device: torch.device,
    outdir: str | Path,
) -> Path:
    """Run CLIP-only inference and save a frames grid with predictions."""
    outdir_p = _ensure_outdir(outdir)
    frames, pred = infer_clip_only(
        video_path, cfg, clip_encoder, head, idx_to_class, device
    )
    fig = _frames_grid_figure(frames)
    caption = _prediction_text(pred.top_classes, mode="clip_only")
    _attach_caption(fig, caption)

    stem = Path(video_path).stem
    out_path = outdir_p / f"{stem}_clip_only.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def visualize_motion_aware(
    video_path: str | Path,
    cfg: Dict,
    clip_encoder: torch.nn.Module,
    fusion: torch.nn.Module,
    idx_to_class: Dict[int, str],
    device: torch.device,
    flow_bins: int,
    outdir: str | Path,
) -> Tuple[Path, Path]:
    """Run motion-aware inference and save frames + flow visualizations."""
    outdir_p = _ensure_outdir(outdir)
    frames, flow_stats, pred = infer_motion_aware(
        video_path, cfg, clip_encoder, fusion, idx_to_class, device, flow_bins
    )

    # Frames + predictions figure
    fig_frames = _frames_grid_figure(frames)
    caption = _prediction_text(pred.top_classes, mode="motion_aware")
    _attach_caption(fig_frames, caption)

    stem = Path(video_path).stem
    frames_path = outdir_p / f"{stem}_motion_frames.png"
    fig_frames.savefig(frames_path, dpi=150)
    plt.close(fig_frames)

    # Flow histogram figure
    fig_flow = _flow_hist_figure(flow_stats, num_bins=flow_bins)
    flow_path = outdir_p / f"{stem}_motion_flow.png"
    fig_flow.savefig(flow_path, dpi=150)
    plt.close(fig_flow)

    return frames_path, flow_path

