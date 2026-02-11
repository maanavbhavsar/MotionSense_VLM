"""Optical flow computation and motion statistics for MotionSense-VLM."""

from typing import Union

import cv2
import numpy as np
import torch


def compute_flow_stats(
    frames: Union[np.ndarray, torch.Tensor],
    num_bins: int = 8,
) -> np.ndarray:
    """
    Compute motion statistics from a sequence of frames (magnitude + direction bins).

    Uses Farneback optical flow between consecutive frames, then aggregates
    magnitude (mean, std) and direction histogram into a fixed-size vector.
    Output shape is (num_bins + 2,) matching FlowExtractor.feature_dim.

    Args:
        frames: (T, H, W, 3) uint8 numpy, or (T, C, H, W) tensor in [0, 1].
        num_bins: Number of direction bins (default 8).

    Returns:
        (num_bins + 2,) float32: [mean_mag, std_mag, bin_0, ..., bin_{num_bins-1}].
    """
    if isinstance(frames, torch.Tensor):
        # (T, C, H, W) -> (T, H, W, 3), scale to 0-255
        frames = frames.permute(0, 2, 3, 1).numpy()
        if frames.max() <= 1.0:
            frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    T, H, W, _ = frames.shape
    if T < 2:
        return np.zeros(num_bins + 2, dtype=np.float32)

    # Grayscale for flow
    grays = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(T)]

    magnitudes: list[np.ndarray] = []
    angles_deg: list[np.ndarray] = []

    for i in range(T - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        magnitudes.append(mag.ravel())
        angles_deg.append(ang.ravel())

    mag_all = np.concatenate(magnitudes)
    ang_all = np.concatenate(angles_deg)

    mean_mag = float(np.mean(mag_all))
    std_mag = float(np.std(mag_all))
    if std_mag == 0:
        std_mag = 1.0

    # Direction bins: 0–360° into num_bins
    bin_edges = np.linspace(0, 360, num_bins + 1)
    hist, _ = np.histogram(ang_all, bins=bin_edges)
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

    out = np.concatenate([[mean_mag, std_mag], hist])
    return out
