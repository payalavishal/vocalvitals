"""Train the real-data binary swallow-event detector.

The validation split selects the checkpoint. The held-out test split is never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vocalvitals.models.binary_cnn import SwallowBinary1DCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics_for(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int | list[list[int]]]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        logits = model(batch_x.to(device, non_blocking=True))
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch_y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probabilities)


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
        "--history",
        type=Path,
        default=Path("artifacts/reports/training_history.csv"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with np.load(args.dataset, allow_pickle=False) as data:
        features = data["X"].astype(np.float32, copy=False)
        labels = data["y"].astype(np.int64, copy=False)

    split_manifest = json.loads(args.split.read_text(encoding="utf-8"))
    split_indices = {
        name: np.asarray(split_manifest["splits"][name]["indices"], dtype=np.int64)
        for name in ("train", "validation")
    }
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]

    train_labels = labels[train_indices]
    positive_count = int(np.sum(train_labels == 1))
    negative_count = int(np.sum(train_labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training split must contain both classes.")
    positive_weight = torch.tensor(
        [negative_count / positive_count], dtype=torch.float32, device=device
    )

    train_dataset = TensorDataset(
        torch.from_numpy(features[train_indices]),
        torch.from_numpy(labels[train_indices].astype(np.float32)),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(features[validation_indices]),
        torch.from_numpy(labels[validation_indices].astype(np.float32)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SwallowBinary1DCNN(in_channels=features.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            sample_count += len(batch_y)

        train_loss = total_loss / max(sample_count, 1)
        val_y, val_probabilities = predict_probabilities(model, validation_loader, device)
        val_metrics = metrics_for(val_y, val_probabilities)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": None,
            **{
                f"validation_{key}": value
                for key, value in val_metrics.items()
                if key != "confusion_matrix"
            },
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.5f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        current_f1 = float(val_metrics["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "SwallowBinary1DCNN",
                    "input_shape": [4, 3200],
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "training_config": {
                        "seed": args.seed,
                        "epochs_requested": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "positive_class_weight": float(positive_weight.item()),
                        "threshold": 0.5,
                        "selection_metric": "validation_f1",
                    },
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    with args.history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print(f"Best epoch: {best_epoch}; validation F1: {best_f1:.4f}")
    print(f"Checkpoint saved: {args.checkpoint}")
    print("Held-out test split was not used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
