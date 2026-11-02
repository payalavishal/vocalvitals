"""1D CNN for binary swallow-event detection."""

from __future__ import annotations

import torch
from torch import nn


class SwallowBinary1DCNN(nn.Module):
    """Three-block Conv1D network; returns one unnormalized logit per window."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=31, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, channels, samples), got {tuple(inputs.shape)}"
            )
        features = self.features(inputs).squeeze(-1)
        return self.classifier(features).squeeze(-1)
