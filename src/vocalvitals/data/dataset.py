"""Dataset discovery and integrity checks for VocalVitals real recordings."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordingPair:
    stem: str
    signal_path: Path
    annotation_path: Path


def discover_recordings(raw_dir: Path) -> list[RecordingPair]:
    """Pair .npy signals and .csv annotations by filename stem."""
    signals_dir = raw_dir / "signals"
    labels_dir = raw_dir / "labels"
    if not signals_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {signals_dir} and {labels_dir}. "
            "Extract signals.zip and labels.zip into data/raw/."
        )

    signals = {p.stem.lower(): p for p in signals_dir.glob("*.npy")}
    labels = {p.stem.lower(): p for p in labels_dir.glob("*.csv")}
    missing_labels = sorted(set(signals) - set(labels))
    missing_signals = sorted(set(labels) - set(signals))
    if missing_labels or missing_signals:
        raise ValueError(
            f"Unpaired files: {len(missing_labels)} signals without labels; "
            f"{len(missing_signals)} labels without signals. "
            f"Examples: {missing_labels[:5]} / {missing_signals[:5]}"
        )
    return [RecordingPair(stem, signals[stem], labels[stem]) for stem in sorted(signals)]


def read_annotations(path: Path) -> list[dict[str, str]]:
    """Read annotation rows without assuming units or changing source values."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"participant", "start", "end"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path.name}: expected columns {sorted(required)}, found {reader.fieldnames}"
            )
        return list(reader)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
