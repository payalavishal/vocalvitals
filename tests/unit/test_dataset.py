from pathlib import Path

import pytest

from vocalvitals.data.dataset import discover_recordings, read_annotations


def test_discover_recordings_pairs_by_stem(tmp_path: Path) -> None:
    (tmp_path / "signals").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "signals" / "recording.npy").write_bytes(b"")
    (tmp_path / "labels" / "recording.csv").write_text(
        "participant,start,end\nP01,10,20\n", encoding="utf-8"
    )
    pairs = discover_recordings(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].stem == "recording"


def test_discover_recordings_rejects_unpaired_files(tmp_path: Path) -> None:
    (tmp_path / "signals").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "signals" / "orphan.npy").write_bytes(b"")
    with pytest.raises(ValueError, match="Unpaired files"):
        discover_recordings(tmp_path)


def test_read_annotations_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        read_annotations(path)
