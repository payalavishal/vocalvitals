"""Create participant-disjoint train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from vocalvitals.data.dataset import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/swallow_binary_windows.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/participant_split.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        labels = data["y"]
        groups = data["participant_id"].astype(str)

    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    test_indices = folds[0][1]
    validation_indices = folds[1][1]
    train_indices = np.concatenate([folds[index][1] for index in range(2, 7)])

    splits = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    participant_sets = {
        name: sorted(set(groups[indices].tolist())) for name, indices in splits.items()
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = set(participant_sets[first]) & set(participant_sets[second])
        if overlap:
            raise RuntimeError(
                f"Participant leakage between {first} and {second}: {sorted(overlap)}"
            )

    manifest = {
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "seed": args.seed,
        "method": "StratifiedGroupKFold(n_splits=7); folds 0/1 held out for test/validation",
        "group_key": "participant_id",
        "splits": {},
    }
    for name, indices in splits.items():
        selected_labels = labels[indices]
        manifest["splits"][name] = {
            "indices": indices.astype(int).tolist(),
            "participants": participant_sets[name],
            "window_count": int(len(indices)),
            "class_counts": {
                "non_swallow": int(np.sum(selected_labels == 0)),
                "swallow": int(np.sum(selected_labels == 1)),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split manifest: {args.output}")
    for name, split in manifest["splits"].items():
        print(
            f"{name}: {split['window_count']} windows, "
            f"{len(split['participants'])} participants, "
            f"classes={split['class_counts']}"
        )
    print("Participant overlap: none")


if __name__ == "__main__":
    main()
