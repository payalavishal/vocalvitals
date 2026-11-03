"""Evaluate the selected checkpoint once on the untouched participant-held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from vocalvitals.data.dataset import sha256_file
from vocalvitals.models.binary_cnn import SwallowBinary1DCNN


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/swallow_binary_windows.npz"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("artifacts/reports/participant_split.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/swallow_binary_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/test_metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Confirm that model selection is frozen and the held-out test may be evaluated once.",
    )
    args = parser.parse_args()
    if not args.confirm_final:
        parser.error("Refusing to access the held-out test without --confirm-final.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SwallowBinary1DCNN(in_channels=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with np.load(args.dataset, allow_pickle=False) as data:
        features = data["X"].astype(np.float32, copy=False)
        labels = data["y"].astype(np.int64, copy=False)

    manifest = json.loads(args.split.read_text(encoding="utf-8"))
    test_split = manifest["splits"]["test"]
    test_indices = np.asarray(test_split["indices"], dtype=np.int64)
    test_data = TensorDataset(
        torch.from_numpy(features[test_indices]),
        torch.from_numpy(labels[test_indices].astype(np.float32)),
    )
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    y_true: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device, non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(batch_y.numpy())

    truth = np.concatenate(y_true).astype(np.int64)
    scores = np.concatenate(probabilities)
    predictions = (scores >= 0.5).astype(np.int64)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "auc": float(roc_auc_score(truth, scores)),
        "confusion_matrix_labels": ["non_swallow", "swallow"],
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    report = {
        "evaluation": "one-time held-out test evaluation",
        "threshold": 0.5,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_metrics": checkpoint["validation_metrics"],
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "test_window_count": len(test_indices),
        "test_participant_count": len(test_split["participants"]),
        "test_participants": test_split["participants"],
        "class_counts": {
            "non_swallow": int(np.sum(truth == 0)),
            "swallow": int(np.sum(truth == 1)),
        },
        "metrics": metrics,
        "interpretation": (
            "Experimental swallow-event detection only; these metrics do not "
            "measure aspiration detection or clinical performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_window_count": len(test_indices), "metrics": metrics}, indent=2))
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
