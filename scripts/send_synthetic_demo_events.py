"""Generate artificial waveforms, classify them, and post demo events to the API."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from vocalvitals.api.models import SyntheticLabel
from vocalvitals.synthetic.inference import SyntheticDemoPredictor
from vocalvitals.synthetic.waveforms import DemoClass, generate_waveform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--patient-id", default="DEMO-001")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/synthetic_demo_best.pt",
    )
    args = parser.parse_args()

    predictor = SyntheticDemoPredictor(Path(args.checkpoint))
    for index in range(args.count):
        source_class = DemoClass(index % len(DemoClass))
        waveform = generate_waveform(source_class, seed=1000 + index)
        prediction = predictor.predict(waveform)
        payload = {
            "patient_id": args.patient_id,
            "label": SyntheticLabel(prediction["label"]).value,
            "confidence": prediction["confidence"],
            "source": "synthetic_demo",
        }
        response = requests.post(f"{args.api_url.rstrip('/')}/event", json=payload, timeout=10)
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()
