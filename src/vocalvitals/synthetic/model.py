"""2D CNN for artificial Normal / Delayed / Aspirating demo labels."""

from __future__ import annotations

import torch
from torch import nn


class SyntheticThreeClassCNN(nn.Module):
    """Small 2D CNN over 64-bin log-mel features."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected (batch, 1, mel_bins, frames), got {tuple(inputs.shape)}")
        return self.classifier(self.features(inputs).flatten(1))
