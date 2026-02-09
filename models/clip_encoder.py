"""CLIP-based semantic encoder for video frames."""

from typing import Optional

import torch
import torch.nn as nn

try:
    import open_clip
except ImportError:
    open_clip = None

# Map original CLIP model names (ViT-B/32) to open_clip names (ViT-B-32)
_CLIP_NAME_MAP = {
    "ViT-B/32": "ViT-B-32",
    "ViT-B/16": "ViT-B-16",
    "ViT-L/14": "ViT-L-14",
    "ViT-L/14@336px": "ViT-L-14-336",
    "RN50": "RN50",
}


def _to_open_clip_name(name: str) -> str:
    """Convert original CLIP naming to open_clip naming."""
    return _CLIP_NAME_MAP.get(name, name.replace("/", "-"))


class CLIPEncoder(nn.Module):
    """
    Wraps pretrained CLIP (via open_clip) to embed sparse video frames.
    Uses temporal mean pooling over frames.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "Install open-clip-torch: pip install open-clip-torch"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        open_clip_name = _to_open_clip_name(model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            open_clip_name,
            pretrained="openai",
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        # Get embed_dim from text projection (compatible across model types)
        proj = self.model.text_projection
        if isinstance(proj, nn.Linear):
            self.embed_dim = proj.out_features
        else:
            self.embed_dim = proj.shape[1]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, N, 3, H, W) - batch of N frames per video

        Returns:
            (B, embed_dim) - one embedding per video
        """
        B, N, C, H, W = frames.shape
        # Flatten to (B*N, 3, H, W)
        x = frames.view(B * N, C, H, W).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(x)
        # (B*N, D) -> (B, N, D) -> mean over N -> (B, D)
        emb = emb.view(B, N, -1).mean(dim=1)
        return emb
