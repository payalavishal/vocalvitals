"""Signal-only baseline for the real-data binary swallow task (RESEARCH ONLY).

This experiment uses *only* information contained in the 200 ms four-channel
window. It deliberately excludes window position within the recording
(``start_ms``), recording length, and any per-recording aggregate, so it cannot
benefit from the sampling artifact documented in
``docs/ACCURACY_CEILING_NOTE.md``.

The expected result is roughly 0.70-0.73 validation AUC. That is the honest
signal-only ceiling for this dataset and it is far below the 98-99% figure that
was requested. This script exists to make that ceiling reproducible and
auditable rather than to chase it.

Models are fitted once on the participant-disjoint training split and scored on
validation. No threshold or model selection uses the test split; the test
metrics printed here are descriptive only, because that split was already spent
by ``scripts/final_evaluate_guard100_balanced.py``.

Usage:
    python scripts/experiment_signal_only_baseline.py
    python scripts/experiment_signal_only_baseline.py --reuse-features
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASET = Path("data/interim/guard100_balanced")
REPORT = Path("artifacts/reports/signal_only_baseline_experiment.json")
FEATURE_CACHE = Path("artifacts/reports/signal_only_features.npz")

SAMPLE_RATE = 16_000
N_FFT = 256
HOP = 64
N_BANDS = 40
N_SEGMENTS = 10
BAND_EDGES = np.geomspace(40.0, 7600.0, N_BANDS + 1)
SEED = 42


def _channel_scale(windows: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Per-channel scale from training windows only; uses no label information."""
    scale = np.quantile(np.abs(windows[train_idx]), 0.995, axis=(0, 2))
    return np.maximum(scale, 1e-6).astype(np.float32)


def _log_mel(windows: np.ndarray) -> np.ndarray:
    """Coarse log-mel power per channel, shape (n, channels, frames, bands)."""
    n_samples = windows.shape[-1]
    taper = np.hanning(N_FFT).astype(np.float32)
    n_frames = (n_samples - N_FFT) // HOP + 1
    offsets = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)

    masks: list[np.ndarray] = []
    for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:], strict=True):
        mask = (frequencies >= low) & (frequencies < high)
        if not mask.any():
            mask = np.zeros_like(frequencies, dtype=bool)
            mask[np.argmin(np.abs(frequencies - low))] = True
        masks.append(mask)

    mel = np.empty((windows.shape[0], windows.shape[1], n_frames, N_BANDS), dtype=np.float32)
    for channel in range(windows.shape[1]):
        framed = windows[:, channel][:, offsets] * taper
        power = np.abs(np.fft.rfft(framed, axis=-1)) ** 2
        for band, mask in enumerate(masks):
            mel[:, channel, :, band] = power[:, :, mask].mean(axis=-1)
    return np.log1p(mel * 1000.0)


def _handcrafted(windows: np.ndarray) -> np.ndarray:
    """Per-channel envelope, spectral-shape and cross-channel ratio features."""
    frequencies = np.fft.rfftfreq(windows.shape[-1], 1.0 / SAMPLE_RATE)
    power = np.abs(np.fft.rfft(windows, axis=-1)) ** 2
    normalized = power / (power.sum(axis=-1, keepdims=True) + 1e-12)

    centroid = (normalized * frequencies).sum(axis=-1)
    bandwidth = np.sqrt((normalized * (frequencies - centroid[..., None]) ** 2).sum(axis=-1))
    cumulative = np.cumsum(normalized, axis=-1)
    roll_50 = frequencies[np.argmax(cumulative > 0.50, axis=-1)]
    roll_85 = frequencies[np.argmax(cumulative > 0.85, axis=-1)]
    flatness = np.exp(np.log(power + 1e-12).mean(axis=-1)) / (power.mean(axis=-1) + 1e-12)
    rms = np.sqrt((windows**2).mean(axis=-1) + 1e-12)
    crest = np.abs(windows).max(axis=-1) / rms
    zero_crossing = (np.diff(np.signbit(windows), axis=-1) != 0).mean(axis=-1)

    frames = windows.reshape(windows.shape[0], windows.shape[1], 50, -1)
    frame_rms = np.sqrt((frames**2).mean(axis=-1) + 1e-12)
    envelope_cv = frame_rms.std(axis=-1) / (frame_rms.mean(axis=-1) + 1e-12)
    envelope_peak = frame_rms.max(axis=-1) / (frame_rms.mean(axis=-1) + 1e-12)
    flux = np.abs(np.diff(frame_rms, axis=-1)).mean(axis=-1) / (frame_rms.mean(axis=-1) + 1e-12)

    blocks = np.stack(
        [
            np.log(rms),
            crest,
            flatness,
            centroid,
            bandwidth,
            roll_50,
            roll_85,
            zero_crossing,
            envelope_cv,
            envelope_peak,
            flux,
        ],
        axis=-1,
    ).reshape(windows.shape[0], -1)
    cross_channel = np.log(rms[:, 1:].mean(axis=1)) - np.log(rms[:, 0])
    return np.column_stack([blocks, cross_channel]).astype(np.float32)


def build_features(windows: np.ndarray) -> np.ndarray:
    """Signal-only feature matrix. No metadata or recording context is used."""
    mel = _log_mel(windows)
    frames = mel.shape[2]
    segment = frames // N_SEGMENTS
    pooled = mel[:, :, : segment * N_SEGMENTS].reshape(
        mel.shape[0], mel.shape[1], N_SEGMENTS, -1, N_BANDS
    ).mean(axis=3)
    mel_features = np.concatenate(
        [
            pooled.reshape(mel.shape[0], -1),
            np.diff(pooled, axis=2).reshape(mel.shape[0], -1),
            mel.mean(axis=2).reshape(mel.shape[0], -1),
            mel.std(axis=2).reshape(mel.shape[0], -1),
        ],
        axis=1,
    )
    return np.hstack([mel_features, _handcrafted(windows)]).astype(np.float32)


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix_labels": ["non_swallow", "swallow"],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Accuracy-maximising threshold, chosen on validation data only."""
    best_accuracy = -1.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.96, 0.005):
        accuracy = accuracy_score(y_true, (scores >= threshold).astype(int))
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = float(threshold)
    return best_threshold


def _base_models() -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=2,
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=SEED,
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=3.0, kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
        ),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000),
        ),
    }


def _load_features(dataset: Path, train_idx: np.ndarray, reuse: bool) -> np.ndarray:
    if reuse and FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE, allow_pickle=False) as cached:
            print(f"reused cached features from {FEATURE_CACHE}", flush=True)
            return cached["features"].astype(np.float32)

    with np.load(dataset / "raw_windows.npz", allow_pickle=False) as data:
        windows = data["X"].astype(np.float32)
    scale = _channel_scale(windows, train_idx)
    normalized = np.clip(windows / scale[None, :, None], -8.0, 8.0).astype(np.float32)

    started = time.time()
    features = build_features(normalized)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"features={features.shape} extracted_in={time.time() - started:.1f}s", flush=True)

    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, features=features, channel_scale=scale)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="reuse the cached feature matrix if it exists",
    )
    args = parser.parse_args()

    with np.load(args.dataset / "swallow_binary_windows.npz", allow_pickle=False) as data:
        labels = data["y"].astype(np.int64)
    split = json.loads((args.dataset / "participant_split.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["splits"]["train"]["indices"], dtype=np.int64)
    val_idx = np.asarray(split["splits"]["validation"]["indices"], dtype=np.int64)
    test_idx = np.asarray(split["splits"]["test"]["indices"], dtype=np.int64)

    features = _load_features(args.dataset, train_idx, args.reuse_features)
    print(
        f"train={len(train_idx)} validation={len(val_idx)} test={len(test_idx)} "
        "(test split is spent; metrics descriptive only)",
        flush=True,
    )

    per_model: dict[str, dict[str, object]] = {}
    val_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    for name, model in _base_models().items():
        started = time.time()
        model.fit(features[train_idx], labels[train_idx])
        val_scores = model.predict_proba(features[val_idx])[:, 1]
        test_scores = model.predict_proba(features[test_idx])[:, 1]
        val_columns.append(val_scores)
        test_columns.append(test_scores)
        threshold = _select_threshold(labels[val_idx], val_scores)
        per_model[name] = {
            "validation_auc": float(roc_auc_score(labels[val_idx], val_scores)),
            "test_auc_descriptive_only": float(roc_auc_score(labels[test_idx], test_scores)),
            "validation_metrics": metrics(labels[val_idx], val_scores, threshold),
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"{name:14s} validation_auc={per_model[name]['validation_auc']:.4f} "
            f"({per_model[name]['seconds']}s)",
            flush=True,
        )

    # A small, pre-specified set of blends; the winner is chosen on validation
    # AUC only. Nothing here looks at the test split.
    candidates = {
        "equal_weight": (0.25, 0.25, 0.25, 0.25),
        "tree_heavy": (0.40, 0.40, 0.10, 0.10),
        "extra_trees_only": (1.0, 0.0, 0.0, 0.0),
        "trees_plus_svm": (0.45, 0.35, 0.20, 0.0),
    }
    val_matrix = np.column_stack(val_columns)
    test_matrix = np.column_stack(test_columns)
    blend_table: dict[str, dict[str, float]] = {}
    for name, weights in candidates.items():
        weight_vector = np.asarray(weights, dtype=np.float64)
        blend_table[name] = {
            "weights": dict(zip(("extra_trees", "random_forest", "svm_rbf", "logistic"),
                                weights, strict=True)),
            "validation_auc": float(
                roc_auc_score(labels[val_idx], val_matrix @ weight_vector)
            ),
        }
    winner = max(blend_table, key=lambda key: blend_table[key]["validation_auc"])
    weights = np.asarray(candidates[winner], dtype=np.float64)

    val_blend = val_matrix @ weights
    test_blend = test_matrix @ weights
    threshold = _select_threshold(labels[val_idx], val_blend)

    val_metrics = metrics(labels[val_idx], val_blend, threshold)
    test_metrics = metrics(labels[test_idx], test_blend, threshold)

    report = {
        "research_only": True,
        "clinical_validity": False,
        "uses_window_position": False,
        "note": (
            "Signal-only baseline. No start_ms, recording length, or per-recording "
            "aggregate features are used, so the documented position sampling "
            "artifact cannot contribute. The held-out test split is spent; its "
            "metrics are descriptive only and must not drive model selection."
        ),
        "dataset": str(args.dataset),
        "features": {
            "count": int(features.shape[1]),
            "log_mel_bands": N_BANDS,
            "n_fft": N_FFT,
            "hop_length": HOP,
            "pooled_segments": N_SEGMENTS,
            "includes_handcrafted": True,
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "per_model": per_model,
        "ensemble": {
            "selected_on_validation_auc": winner,
            "candidates": blend_table,
            "threshold": threshold,
        },
        "validation_metrics": val_metrics,
        "test_metrics_descriptive_only": test_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("SIGNAL-ONLY BASELINE (research only, no position features)")
    print(
        "validation  accuracy={:.4f} balanced={:.4f} auc={:.4f}".format(
            val_metrics["accuracy"], val_metrics["balanced_accuracy"], val_metrics["auc"]
        )
    )
    print(
        "test(spent) accuracy={:.4f} balanced={:.4f} auc={:.4f}".format(
            test_metrics["accuracy"], test_metrics["balanced_accuracy"], test_metrics["auc"]
        )
    )
    print("Report:", args.output)


if __name__ == "__main__":
    main()
