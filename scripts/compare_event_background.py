"""Compare raw signal energy in annotated event regions and safe background regions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from vocalvitals.data.dataset import discover_recordings
from vocalvitals.data.preprocessing import EVENT_GUARD_MS, MIN_EVENT_MS, _negative_candidates

RAW = Path("data/raw")
SAMPLE_RATE = 44_100


def main() -> None:
    event_values: list[list[float]] = [[], [], [], []]
    background_values: list[list[float]] = [[], [], [], []]
    for pair in discover_recordings(RAW):
        signal = np.load(pair.signal_path, mmap_mode="r", allow_pickle=False)
        duration_ms = len(signal) * 1000.0 / SAMPLE_RATE
        with pair.annotation_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        events = [
            (float(row["start"]), float(row["end"]), float(row["duration"]), row["participant"])
            for row in rows
        ]
        for start_ms, end_ms, duration_ms_event, _ in events:
            if duration_ms_event < MIN_EVENT_MS:
                continue
            start = max(0, int(start_ms * SAMPLE_RATE / 1000))
            end = min(len(signal), int(end_ms * SAMPLE_RATE / 1000))
            if end <= start:
                continue
            segment = np.asarray(signal[start:end], dtype=np.float64)
            for channel in range(4):
                event_values[channel].append(float(np.sqrt(np.mean(segment[:, channel] ** 2))))

        for start_ms in _negative_candidates(duration_ms, events, EVENT_GUARD_MS):
            start = int(start_ms * SAMPLE_RATE / 1000)
            end = min(len(signal), start + int(0.2 * SAMPLE_RATE))
            if end - start < int(0.2 * SAMPLE_RATE):
                continue
            segment = np.asarray(signal[start:end], dtype=np.float64)
            for channel in range(4):
                background_values[channel].append(float(np.sqrt(np.mean(segment[:, channel] ** 2))))

    for channel in range(4):
        event = np.asarray(event_values[channel])
        background = np.asarray(background_values[channel])
        test = mannwhitneyu(event, background, alternative="two-sided")
        print(
            f"channel {channel}: event n={len(event)} median={np.median(event):.6g}; "
            f"background n={len(background)} median={np.median(background):.6g}; "
            f"Mann-Whitney p={test.pvalue:.4g}"
        )


if __name__ == "__main__":
    main()
