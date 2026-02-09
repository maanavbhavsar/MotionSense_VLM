"""Optical flow feature dimension and passthrough (actual flow in data/flow.py)."""

import torch
import torch.nn as nn


class FlowExtractor(nn.Module):
    """
    Extracts motion statistics from frames (used at dataset/collate level).
    This module mainly provides the feature dim for fusion; actual extraction
    happens in compute_flow_stats (CPU/NumPy) for simplicity.
    """

    def __init__(self, num_bins: int = 8):
        super().__init__()
        self.num_bins = num_bins
        self.feature_dim = num_bins + 2

    def forward(self, flow_stats: torch.Tensor) -> torch.Tensor:
        """Pass-through; expects precomputed flow stats (B, feature_dim)."""
        return flow_stats
