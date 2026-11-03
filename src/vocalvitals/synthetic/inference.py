"""Inference helper for synthetic-only demo waveforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vocalvitals.synthetic.features import waveform_to_logmel
from vocalvitals.synthetic.model import SyntheticThreeClassCNN
from vocalvitals.synthetic.waveforms import CLASS_NAMES, SAMPLE_RATE


class SyntheticDemoPredictor:
    def __init__(self, checkpoint_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if not checkpoint.get("synthetic_only", False):
            raise ValueError("Refusing to load a non-synthetic checkpoint here.")
        self.model = SyntheticThreeClassCNN().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> dict[str, object]:
        feature = waveform_to_logmel(waveform, sample_rate=SAMPLE_RATE)
        batch = torch.from_numpy(feature[None, None]).to(self.device)
        probabilities = torch.softmax(self.model(batch), dim=1)[0].cpu().numpy()
        index = int(np.argmax(probabilities))
        return {
            "label": CLASS_NAMES[index],
            "confidence": float(probabilities[index]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
            "synthetic_only": True,
        }
