"""Train a synthetic-only three-class demo model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vocalvitals.synthetic.features import batch_logmel
from vocalvitals.synthetic.model import SyntheticThreeClassCNN
from vocalvitals.synthetic.waveforms import CLASS_NAMES, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)

    waveforms, labels = make_dataset(args.samples_per_class, args.seed)
    features = batch_logmel(waveforms)[:, None, :, :]
    train_x, val_x, train_y, val_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyntheticThreeClassCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_tensor = torch.from_numpy(val_x.astype(np.float32)).to(device)

    best_accuracy = -1.0
    best_epoch = 0
    checkpoint_path = Path("artifacts/checkpoints/synthetic_demo_best.pt")
    report_path = Path("artifacts/reports/synthetic_demo_metrics.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            count += len(batch_y)

        model.eval()
        with torch.no_grad():
            predictions = model(val_tensor).argmax(dim=1).cpu().numpy()
        accuracy = float(accuracy_score(val_y, predictions))
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={total_loss / max(count, 1):.4f} val_accuracy={accuracy:.4f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "SyntheticThreeClassCNN",
                    "class_names": list(CLASS_NAMES),
                    "sample_rate": 16_000,
                    "n_mels": 64,
                    "synthetic_only": True,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
    )
    model.eval()
    with torch.no_grad():
        predictions = model(val_tensor).argmax(dim=1).cpu().numpy()
    report = {
        "synthetic_only": True,
        "samples_per_class": args.samples_per_class,
        "validation_samples": len(val_y),
        "class_names": list(CLASS_NAMES),
        "best_epoch": best_epoch,
        "validation_accuracy": float(accuracy_score(val_y, predictions)),
        "confusion_matrix": confusion_matrix(val_y, predictions, labels=[0, 1, 2])
        .astype(int)
        .tolist(),
        "interpretation": (
            "Artificially generated waveforms only. These labels do not "
            "represent patient observations or clinical aspiration detection."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
