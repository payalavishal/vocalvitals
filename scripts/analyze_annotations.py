"""Summarize annotation geometry against recording durations."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from vocalvitals.data.dataset import discover_recordings

RAW = Path("data/raw")
SAMPLE_RATE = 44_100


def main() -> None:
    pairs = discover_recordings(RAW)
    rows_total = 0
    empty_files = 0
    events_per_file: Counter[int] = Counter()
    outside_start = 0
    outside_end = 0
    negative_start = 0
    short_recordings = 0
    too_long = 0
    duration_mismatch = 0
    examples: list[str] = []

    for pair in pairs:
        signal = np.load(pair.signal_path, mmap_mode="r", allow_pickle=False)
        recording_ms = signal.shape[0] * 1000.0 / SAMPLE_RATE
        with pair.annotation_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        rows_total += len(rows)
        events_per_file[len(rows)] += 1
        if not rows:
            empty_files += 1
        for row in rows:
            start, end = float(row["start"]), float(row["end"])
            duration = float(row["duration"])
            if start < 0:
                negative_start += 1
            if start < 0 or start >= recording_ms:
                outside_start += 1
            if end < 0 or end > recording_ms:
                outside_end += 1
            if duration > 200:
                too_long += 1
            if duration < 20:
                short_recordings += 1
            if abs((end - start) - duration) > 1.5:
                duration_mismatch += 1
            if (start < 0 or end > recording_ms or duration > 200) and len(examples) < 25:
                examples.append(
                    f"{pair.stem}: {start=}, {end=}, {duration=}, recording_ms={recording_ms:.1f}"
                )

    print(f"recordings={len(pairs)} annotation_rows={rows_total}")
    print(f"empty_annotation_files={empty_files}")
    print(f"files_by_annotation_row_count={dict(sorted(events_per_file.items()))}")
    print(
        f"negative_start={negative_start} start_outside={outside_start} "
        f"end_outside={outside_end} duration_gt_200ms={too_long} "
        f"duration_lt_20ms={short_recordings} duration_mismatch={duration_mismatch}"
    )
    print("flagged_examples:")
    print("\n".join(examples))


if __name__ == "__main__":
    main()
