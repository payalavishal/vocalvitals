"""Convert raw cervical-auscultation recordings into fixed-size binary windows.

The source annotations describe swallow events. Non-swallow windows are sampled only
from regions outside annotated events plus a configurable safety margin. This is a
research preprocessing pipeline, not an aspiration classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from vocalvitals.data.dataset import discover_recordings

ORIGINAL_SAMPLE_RATE = 44_100
TARGET_SAMPLE_RATE = 16_000
WINDOW_MS = 200
WINDOW_SAMPLES = TARGET_SAMPLE_RATE * WINDOW_MS // 1000
EVENT_GUARD_MS = 50
MIN_EVENT_MS = 20
DEFAULT_NEGATIVE_COUNT = 1_281
DEFAULT_SEED = 42


@dataclass(frozen=True)
class WindowRecord:
    label: int
    participant_id: str
    recording_id: str
    start_ms: float
    end_ms: float
    annotation_index: int | None


def _read_events(path: Path) -> list[tuple[float, float, float, str]]:
    events: list[tuple[float, float, float, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            start = float(row["start"])
            end = float(row["end"])
            duration = float(row["duration"])
            if end < start:
                raise ValueError(f"Invalid annotation bounds in {path.name}: {row}")
            events.append((start, end, duration, row["participant"].strip()))
    return events


def _resample(signal: np.ndarray) -> np.ndarray:
    if signal.ndim != 2 or signal.shape[1] != 4:
        raise ValueError(f"Expected (samples, 4) signal, got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinite values")
    # 44,100 Hz -> 16,000 Hz = rational ratio 160/441; polyphase filtering
    # performs anti-alias filtering before downsampling.
    result = resample_poly(signal.astype(np.float32, copy=False), 160, 441, axis=0)
    return np.asarray(result, dtype=np.float32)


def _peak_normalize(window: np.ndarray) -> np.ndarray:
    peaks = np.max(np.abs(window), axis=0, keepdims=True)
    safe_peaks = np.where(peaks > 1e-8, peaks, 1.0)
    return (window / safe_peaks).astype(np.float32, copy=False)


def _fixed_window(signal: np.ndarray, center_ms: float) -> np.ndarray:
    center_sample = int(round(center_ms * TARGET_SAMPLE_RATE / 1000.0))
    start = center_sample - WINDOW_SAMPLES // 2
    end = start + WINDOW_SAMPLES
    src_start = max(0, start)
    src_end = min(signal.shape[0], end)
    window = np.zeros((WINDOW_SAMPLES, 4), dtype=np.float32)
    if src_end > src_start:
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        window[dst_start:dst_end] = signal[src_start:src_end]
    return _peak_normalize(window).T


def _merge_intervals(
    intervals: list[tuple[float, float]], duration_ms: float
) -> list[tuple[float, float]]:
    clipped = sorted(
        (max(0.0, start), min(duration_ms, end))
        for start, end in intervals
        if end > 0 and start < duration_ms
    )
    merged: list[tuple[float, float]] = []
    for start, end in clipped:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _negative_candidates(
    duration_ms: float,
    events: list[tuple[float, float, float, str]],
    guard_ms: float,
) -> list[float]:
    guarded = _merge_intervals(
        [(start - guard_ms, end + guard_ms) for start, end, _, _ in events],
        duration_ms,
    )
    safe_intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in guarded:
        if start - cursor >= WINDOW_MS:
            safe_intervals.append((cursor, start))
        cursor = max(cursor, end)
    if duration_ms - cursor >= WINDOW_MS:
        safe_intervals.append((cursor, duration_ms))

    starts: list[float] = []
    for start, end in safe_intervals:
        count = int((end - start) // WINDOW_MS)
        starts.extend(start + index * WINDOW_MS for index in range(count))
    return starts


def build_dataset(
    raw_dir: Path,
    output_dir: Path,
    negative_count: int = DEFAULT_NEGATIVE_COUNT,
    seed: int = DEFAULT_SEED,
    guard_ms: float = EVENT_GUARD_MS,
) -> tuple[Path, Path]:
    pairs = discover_recordings(raw_dir)
    positive_windows: list[np.ndarray] = []
    positive_records: list[WindowRecord] = []
    negative_pool: list[tuple[np.ndarray, WindowRecord]] = []

    for pair in pairs:
        original = np.load(pair.signal_path, mmap_mode="r", allow_pickle=False)
        signal = _resample(original)
        duration_ms = signal.shape[0] * 1000.0 / TARGET_SAMPLE_RATE
        events = _read_events(pair.annotation_path)

        for event_index, (start_ms, end_ms, event_duration_ms, participant_id) in enumerate(events):
            if event_duration_ms < MIN_EVENT_MS:
                continue
            clipped_start = min(max(start_ms, 0.0), duration_ms)
            clipped_end = min(max(end_ms, 0.0), duration_ms)
            if clipped_end <= clipped_start:
                continue
            center_ms = (clipped_start + clipped_end) / 2.0
            positive_windows.append(_fixed_window(signal, center_ms))
            positive_records.append(
                WindowRecord(
                    label=1,
                    participant_id=participant_id,
                    recording_id=pair.stem,
                    start_ms=center_ms - WINDOW_MS / 2,
                    end_ms=center_ms + WINDOW_MS / 2,
                    annotation_index=event_index,
                )
            )

        participant_id = (
            events[0][3]
            if events
            else re.match(r"^([a-z]+\d+)", pair.stem, re.IGNORECASE).group(1).upper()
        )
        for start_ms in _negative_candidates(duration_ms, events, guard_ms):
            center_ms = start_ms + WINDOW_MS / 2
            negative_pool.append(
                (
                    _fixed_window(signal, center_ms),
                    WindowRecord(
                        label=0,
                        participant_id=participant_id,
                        recording_id=pair.stem,
                        start_ms=start_ms,
                        end_ms=start_ms + WINDOW_MS,
                        annotation_index=None,
                    ),
                )
            )

    if len(negative_pool) < negative_count:
        raise ValueError(
            f"Requested {negative_count} negative windows but only "
            f"{len(negative_pool)} non-overlapping candidates are available."
        )

    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(negative_pool), size=negative_count, replace=False))
    negative_windows = [negative_pool[index][0] for index in selected_indices]
    negative_records = [negative_pool[index][1] for index in selected_indices]

    windows = np.stack(positive_windows + negative_windows).astype(np.float32)
    records = positive_records + negative_records
    permutation = rng.permutation(len(records))
    windows = windows[permutation]
    records = [records[index] for index in permutation]

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "swallow_binary_windows.npz"
    metadata_path = output_dir / "swallow_binary_metadata.json"

    np.savez_compressed(
        dataset_path,
        X=windows,
        y=np.asarray([record.label for record in records], dtype=np.int64),
        participant_id=np.asarray([record.participant_id for record in records]),
        recording_id=np.asarray([record.recording_id for record in records]),
    )
    metadata = {
        "source": "local signals.zip + labels.zip",
        "original_sample_rate_hz": ORIGINAL_SAMPLE_RATE,
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "window_ms": WINDOW_MS,
        "window_shape": [4, WINDOW_SAMPLES],
        "channel_layout": "microphone + 3 accelerometer axes",
        "event_guard_ms": guard_ms,
        "minimum_event_duration_ms": MIN_EVENT_MS,
        "negative_windows_requested": negative_count,
        "negative_candidates_available": len(negative_pool),
        "random_seed": seed,
        "class_counts": {
            "non_swallow": len(negative_records),
            "swallow": len(positive_records),
        },
        "records": [asdict(record) for record in records],
        "interpretation": (
            "Binary swallow-event detection only. The source labels do not "
            "establish normal, delayed, or aspirating clinical outcomes."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--negative-count", type=int, default=DEFAULT_NEGATIVE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--guard-ms", type=float, default=EVENT_GUARD_MS)
    args = parser.parse_args()
    dataset_path, metadata_path = build_dataset(
        args.raw_dir, args.output_dir, args.negative_count, args.seed, args.guard_ms
    )
    print(f"Dataset saved: {dataset_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
