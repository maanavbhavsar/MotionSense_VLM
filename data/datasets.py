"""PyTorch datasets for video action recognition (Phase 2)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .loader import load_video_frames, frames_to_tensor


class VideoActionDataset(Dataset):
    """
    Dataset of video clips with action labels.

    Expects directory structure:

        root/
            class_a/
                clip1.mp4
                clip2.mp4
            class_b/
                ...

    This Phase 2 version only handles RGB frame tensors (no motion/flow yet).
    """

    def __init__(
        self,
        root: str | Path,
        class_to_idx: Optional[Dict[str, int]] = None,
        classes: Optional[List[str]] = None,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
        img_size: Tuple[int, int] = (224, 224),
        max_classes: Optional[int] = None,
        max_clips_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.img_size = img_size
        self.max_classes = max_classes
        self.max_clips_per_class = max_clips_per_class

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Discover classes and (video_path, label_idx) samples
        self.samples: List[Tuple[Path, int]] = []

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        elif classes is not None:
            class_list = classes[: max_classes] if max_classes is not None else classes
            self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        else:
            discovered = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            if not discovered:
                raise ValueError(
                    f"No class subdirectories found under {self.root}. "
                    "Expected structure: root/class_name/video_file.mp4"
                )
            if max_classes is not None:
                discovered = discovered[: max_classes]
            self.class_to_idx = {c: i for i, c in enumerate(discovered)}

        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            clips: List[Path] = []
            for ext in ("*.mp4", "*.avi", "*.mkv"):
                clips.extend(class_dir.glob(ext))
            clips = sorted(clips)
            if max_clips_per_class is not None:
                clips = clips[: max_clips_per_class]
            for p in clips:
                self.samples.append((p, idx))

        if not self.samples:
            raise ValueError(
                f"No video files found under {self.root} "
                "(looked for *.mp4, *.avi, *.mkv in class subdirectories)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        frames = load_video_frames(
            path,
            sample_rate=self.sample_rate,
            max_frames=self.max_frames,
            img_size=self.img_size,
        )
        tensor = frames_to_tensor(frames)
        return tensor, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def collate_basic(batch):
    """
    Simple collate function for (frames, label) samples.

    Pads variable-length frame sequences to the max length in the batch by
    repeating the last frame.
    """

    frames_list, labels = zip(*batch)
    max_n = max(f.shape[0] for f in frames_list)
    # frames: (N, 3, H, W)
    _, C, H, W = frames_list[0].shape

    padded = []
    for f in frames_list:
        n = f.shape[0]
        if n < max_n:
            pad = f[-1:].expand(max_n - n, -1, -1, -1).clone()
            f = torch.cat([f, pad], dim=0)
        padded.append(f)

    return (
        torch.stack(padded),  # (B, T, C, H, W)
        torch.tensor(labels, dtype=torch.long),
    )

