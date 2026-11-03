"""Audit raw VocalVitals recordings and annotations without changing source files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from vocalvitals.data.dataset import discover_recordings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/raw_data_audit.json")
    )
    args = parser.parse_args()

    pairs = discover_recordings(args.raw_dir)
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    participant_ids: set[str] = set()
    annotation_rows = 0
    duration_values: list[float] = []
    invalid_bounds: list[dict[str, object]] = []
    duration_mismatches: list[dict[str, object]] = []
    nonfinite_files: list[str] = []
    non_four_channel_files: list[dict[str, object]] = []

    for pair in pairs:
        signal = np.load(pair.signal_path, mmap_mode="r", allow_pickle=False)
        shape_counts[str(tuple(signal.shape))] += 1
        dtype_counts[str(signal.dtype)] += 1
        if signal.ndim != 2 or signal.shape[1] != 4:
            non_four_channel_files.append(
                {"file": pair.signal_path.name, "shape": list(signal.shape)}
            )
        elif not np.isfinite(signal).all():
            nonfinite_files.append(pair.signal_path.name)

        with pair.annotation_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                annotation_rows += 1
                participant = row.get("participant", "").strip()
                if participant:
                    participant_ids.add(participant)
                try:
                    start = float(row["start"])
                    end = float(row["end"])
                    duration = float(row["duration"])
                except (KeyError, TypeError, ValueError):
                    invalid_bounds.append({"file": pair.annotation_path.name, "row": row})
                    continue
                duration_values.append(duration)
                if end < start:
                    invalid_bounds.append({"file": pair.annotation_path.name, "row": row})
                if abs((end - start) - duration) > 1.5:
                    duration_mismatches.append(
                        {
                            "file": pair.annotation_path.name,
                            "start": start,
                            "end": end,
                            "duration": duration,
                        }
                    )

    report = {
        "recording_pairs": len(pairs),
        "annotation_rows": annotation_rows,
        "unique_participants": len(participant_ids),
        "participant_ids": sorted(participant_ids),
        "signal_shapes": dict(shape_counts),
        "signal_dtypes": dict(dtype_counts),
        "annotation_duration_ms": {
            "min": min(duration_values) if duration_values else None,
            "max": max(duration_values) if duration_values else None,
            "below_20_ms": sum(value < 20 for value in duration_values),
            "above_200_ms": sum(value > 200 for value in duration_values),
        },
        "invalid_bounds": invalid_bounds,
        "duration_mismatches": duration_mismatches,
        "non_four_channel_files": non_four_channel_files,
        "nonfinite_files": nonfinite_files,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "participant_ids"}, indent=2
        )
    )
    print(f"\nFull report: {args.report}")


if __name__ == "__main__":
    main()
