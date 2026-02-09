"""Late fusion of CLIP semantic + optical flow motion features."""

from typing import Optional

import torch
import torch.nn as nn


class MotionSenseFusion(nn.Module):
    """
    Concatenates CLIP embedding + flow stats and classifies with a small MLP.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        flow_dim: int = 10,
        num_classes: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.flow_dim = flow_dim
        fusion_dim = clip_dim + flow_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        clip_emb: torch.Tensor,
        flow_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clip_emb: (B, clip_dim)
            flow_stats: (B, flow_dim)

        Returns:
            (B, num_classes) logits
        """
        x = torch.cat([clip_emb, flow_stats], dim=1)
        return self.classifier(x)
