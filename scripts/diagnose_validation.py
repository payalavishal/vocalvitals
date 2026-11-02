"""Validation-only diagnostics; never accesses the held-out test indices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vocalvitals.models.binary_cnn import SwallowBinary1DCNN


def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def handcrafted_features(x: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    frequencies = np.fft.rfftfreq(x.shape[-1], d=1 / 16_000)
    for window in x:
        features: list[float] = []
        for channel in window:
            centered = channel - channel.mean()
            features.extend(
                [
                    float(channel.mean()),
                    float(channel.std()),
                    float(np.sqrt(np.mean(channel**2))),
                    float(np.max(np.abs(channel))),
                    float(np.mean(np.abs(channel))),
                    float(np.sqrt(np.mean(np.diff(channel) ** 2))),
                    float(np.mean(np.diff(np.signbit(channel)))),
                ]
            )
            spectrum = np.abs(np.fft.rfft(centered)) ** 2
            for low, high in ((0, 250), (250, 1_000), (1_000, 4_000), (4_000, 8_000)):
                band = (frequencies >= low) & (frequencies < high)
                features.append(float(np.mean(spectrum[band])) if np.any(band) else 0.0)
        rows.append(np.asarray(features, dtype=np.float32))
    return np.stack(rows)


def main() -> None:
    dataset_path = Path("data/processed/swallow_binary_windows.npz")
    split_path = Path("artifacts/reports/participant_split.json")
    checkpoint_path = Path("artifacts/checkpoints/swallow_binary_best.pt")

    with np.load(dataset_path, allow_pickle=False) as data:
        x = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    train_idx = np.asarray(manifest["splits"]["train"]["indices"], dtype=np.int64)
    val_idx = np.asarray(manifest["splits"]["validation"]["indices"], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SwallowBinary1DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities: dict[str, np.ndarray] = {}
    for name, indices in (("train", train_idx), ("validation", val_idx)):
        batch = torch.from_numpy(x[indices]).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(batch)).cpu().numpy()
        probabilities[name] = p

    report: dict[str, object] = {
        "note": "Validation-only diagnosis. Test split indices were not accessed.",
        "cnn": {},
    }
    for name, indices in (("train", train_idx), ("validation", val_idx)):
        truth = y[indices]
        p = probabilities[name]
        report["cnn"][name] = {
            "auc": float(roc_auc_score(truth, p)),
            "probability_mean_by_class": {
                "non_swallow": float(p[truth == 0].mean()),
                "swallow": float(p[truth == 1].mean()),
            },
            "threshold_sweep": [
                score(truth, p, threshold) for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
            ],
        }

    features = handcrafted_features(x)
    baselines: dict[str, object] = {}
    for name, estimator in (
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ):
        estimator.fit(features[train_idx], y[train_idx])
        p = estimator.predict_proba(features[val_idx])[:, 1]
        baselines[name] = {
            "auc": float(roc_auc_score(y[val_idx], p)),
            "threshold_0_5": score(y[val_idx], p, 0.5),
        }
    report["handcrafted_feature_baselines"] = baselines

    output = Path("artifacts/reports/validation_diagnostics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
