"""Print a reproducible inventory of the extracted source dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vocalvitals.data.dataset import discover_recordings, read_annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    pairs = discover_recordings(args.raw_dir)
    participants: set[str] = set()
    rows = 0
    extensions = Counter()
    for pair in pairs:
        extensions[pair.signal_path.suffix] += 1
        annotations = read_annotations(pair.annotation_path)
        rows += len(annotations)
        participants.update(
            row["participant"].strip() for row in annotations if row.get("participant", "").strip()
        )

    print(f"Recording pairs: {len(pairs)}")
    print(f"Annotation rows: {rows}")
    print(f"Unique participant IDs: {len(participants)}")
    print(f"Signal extensions: {dict(extensions)}")
    print("First 10 pairs:")
    for pair in pairs[:10]:
        print(f"  {pair.stem}: {pair.signal_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
