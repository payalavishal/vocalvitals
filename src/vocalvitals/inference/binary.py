"""Experimental inference for the real-data binary swallow-event model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vocalvitals.models.binary_cnn import SwallowBinary1DCNN


class BinarySwallowPredictor:
    """Predict swallow / non-swallow for already-preprocessed windows only."""

    def __init__(self, checkpoint_path: Path, threshold: float = 0.5) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint.get("model_name") != "SwallowBinary1DCNN":
            raise ValueError("Checkpoint is not a SwallowBinary1DCNN model.")
        self.model = SwallowBinary1DCNN(in_channels=4).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> dict[str, float | str | bool]:
        values = np.asarray(window, dtype=np.float32)
        if values.shape != (4, 3200):
            raise ValueError(f"Expected window shape (4, 3200), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Window contains NaN or infinite values")
        batch = torch.from_numpy(values[None]).to(self.device)
        probability = float(torch.sigmoid(self.model(batch))[0].cpu().item())
        label = "swallow" if probability >= self.threshold else "non_swallow"
        confidence = probability if label == "swallow" else 1.0 - probability
        return {
            "label": label,
            "swallow_probability": probability,
            "confidence": confidence,
            "threshold": self.threshold,
            "experimental_only": True,
            "clinical_validity": False,
        }
