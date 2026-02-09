"""Video loading and frame sampling utilities."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def load_video_frames(
    video_path: str | Path,
    sample_rate: int = 1,
    max_frames: Optional[int] = None,
    img_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load a video and sample frames sparsely.

    Args:
        video_path: Path to video file.
        sample_rate: Sample 1 frame every N seconds
            (for a ~30fps video, 1 ≈ 1 fps, 2 ≈ 1 frame every 2 seconds, etc.).
        max_frames: Cap number of frames (None = no cap).
        img_size: (H, W) for resizing.

    Returns:
        Frames array of shape (N, H, W, 3) in RGB, uint8.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Every `sample_rate` seconds -> roughly fps * sample_rate frames interval
    frame_interval = max(1, int(fps * sample_rate))

    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # OpenCV resize expects (W, H)
            frame_resized = cv2.resize(frame_rgb, img_size[::-1])
            frames.append(frame_resized)

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {path}")

    return np.stack(frames, axis=0)


def frames_to_tensor(
    frames: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert frames (N, H, W, 3) uint8 to tensor (N, 3, H, W) for CLIP-style models.

    Uses CLIP/ImageNet-style normalization if normalize=True.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected frames of shape (N, H, W, 3), got {tuple(frames.shape)}"
        )

    # (N, H, W, 3) -> (N, 3, H, W), scale to [0, 1]
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    if not normalize:
        return x

    # CLIP normalization statistics
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (x - mean) / std

