"""Position-augmented swallow classifier (RESEARCH ONLY).

This experiment augments the 200 ms four-channel windows with window-position
features derived from the recording metadata (start_ms, recording length and
per-recording aggregates). It reaches ~95.6% validation / ~93.0% test accuracy
on the guard100_balanced dataset, but this is NOT a clinical detector: the
position features exploit a sampling artifact (swallow events cluster early in
these short clips while negatives are sampled uniformly across the recording).
Do not deploy, do not report as swallow-detection accuracy, and do not connect
to the synthetic API/dashboard. See docs/ACCURACY_CEILING_NOTE.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATASET = Path("data/interim/guard100_balanced")
REPORT = Path("artifacts/reports/position_augmented_experiment.json")
CHECKPOINT = Path("artifacts/checkpoints/position_augmented_ensemble.joblib")


def build_features(records: list[dict]) -> np.ndarray:
    starts = np.asarray([r["start_ms"] for r in records], dtype=np.float64)
    rec_dur = {}
    for r in records:
        rec_dur[r["recording_id"]] = max(rec_dur.get(r["recording_id"], 0.0), r["end_ms"])
    recd = np.asarray([rec_dur[r["recording_id"]] for r in records], dtype=np.float64)

    rec_starts: dict[str, list[float]] = defaultdict(list)
    for r, s in zip(records, starts, strict=True):
        rec_starts[r["recording_id"]].append(float(s))
    rec_mean = {k: float(np.mean(v)) for k, v in rec_starts.items()}
    rec_std = {k: (float(np.std(v)) if len(v) > 1 else 1.0) for k, v in rec_starts.items()}
    rec_min = {k: float(np.min(v)) for k, v in rec_starts.items()}
    rec_max = {k: float(np.max(v)) for k, v in rec_starts.items()}
    rec_n = {k: len(v) for k, v in rec_starts.items()}

    rm = np.asarray([rec_mean[r["recording_id"]] for r in records])
    rs = np.asarray([rec_std[r["recording_id"]] for r in records])
    rmin = np.asarray([rec_min[r["recording_id"]] for r in records])
    rmax = np.asarray([rec_max[r["recording_id"]] for r in records])
    rn = np.asarray([rec_n[r["recording_id"]] for r in records], dtype=np.float64)

    columns = [
        starts,
        recd,
        starts / np.where(recd > 0, recd, 1.0),
        starts / (recd + 1.0),
        starts - recd,
        starts + recd,
        np.log1p(np.clip(starts, 0.0, None)),
        np.log1p(np.clip(recd, 0.0, None)),
        rm,
        rs,
        rmin,
        rmax,
        rn,
        starts - rm,
        (starts - rm) / (rs + 1.0),
        starts / np.where(rn > 0, rn, 1.0),
    ]
    features = np.column_stack(columns)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": (
            float((tp / (tp + fn) + tn / (tn + fp)) / 2.0) if (tp + fn) and (tn + fp) else 0.0
        ),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix_labels": ["non_swallow", "swallow"],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    metadata = json.loads(
        (args.dataset / "swallow_binary_metadata.json").read_text(encoding="utf-8")
    )
    records = metadata["records"]
    split_path = args.dataset / "participant_split.json"
    if not split_path.exists():
        split_path = Path("data/interim/guard100_balanced/participant_split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))

    with np.load(args.dataset / "swallow_binary_windows.npz", allow_pickle=False) as data:
        labels = data["y"].astype(np.int64)

    features = build_features(records)
    train_idx = np.asarray(split["splits"]["train"]["indices"], dtype=np.int64)
    val_idx = np.asarray(split["splits"]["validation"]["indices"], dtype=np.int64)
    test_idx = np.asarray(split["splits"]["test"]["indices"], dtype=np.int64)

    # Train diverse base models.
    base_models = {
        "rf1": RandomForestClassifier(
            n_estimators=1500, min_samples_leaf=1, max_features="sqrt", random_state=1, n_jobs=-1
        ),
        "rf2": RandomForestClassifier(
            n_estimators=1500, min_samples_leaf=2, max_features="log2", random_state=2, n_jobs=-1
        ),
        "et": ExtraTreesClassifier(
            n_estimators=1500, min_samples_leaf=1, max_features="sqrt", random_state=3, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=31, random_state=4
        ),
    }

    train_predictions = {}
    val_predictions = {}
    test_predictions = {}
    for name, model in base_models.items():
        model.fit(features[train_idx], labels[train_idx])
        train_predictions[name] = model.predict_proba(features[train_idx])[:, 1]
        val_predictions[name] = model.predict_proba(features[val_idx])[:, 1]
        test_predictions[name] = model.predict_proba(features[test_idx])[:, 1]

    # Logistic meta-learner (stacking).
    stack_train = np.column_stack(list(train_predictions.values()))
    stack_val = np.column_stack(list(val_predictions.values()))
    stack_test = np.column_stack(list(test_predictions.values()))
    meta = LogisticRegression(C=0.5, max_iter=2000)
    meta.fit(stack_train, labels[train_idx])

    val_scores = meta.predict_proba(stack_val)[:, 1]
    test_scores = meta.predict_proba(stack_test)[:, 1]

    # Threshold selected on validation only.
    best_accuracy = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.01, 0.99, 0.001):
        accuracy = accuracy_score(labels[val_idx], (val_scores >= threshold).astype(int))
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = float(threshold)

    val_metrics = metrics(labels[val_idx], val_scores, best_threshold)
    test_metrics = metrics(labels[test_idx], test_scores, best_threshold)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "research_only": True,
            "clinical_validity": False,
            "base_models": base_models,
            "meta_learner": meta,
            "threshold": best_threshold,
            "feature_builder": "build_features",
            "note": ("Position-augmented ensemble exploiting a sampling artifact. "
                    "Not a clinical detector. See docs/ACCURACY_CEILING_NOTE.md."),
        },
        args.checkpoint,
    )

    report = {
        "research_only": True,
        "clinical_validity": False,
        "note": (
            "Position-augmented ensemble. Exploits window-position sampling artifact; "
            "NOT a clinical detector. See docs/ACCURACY_CEILING_NOTE.md."
        ),
        "dataset": str(args.dataset),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "threshold": best_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Position-augmented (research only)")
    print("Validation:", json.dumps(val_metrics))
    print("Test:", json.dumps(test_metrics))
    print("Checkpoint:", args.checkpoint)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
