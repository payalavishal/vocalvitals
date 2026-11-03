import numpy as np

from vocalvitals.data.preprocessing import (
    WINDOW_SAMPLES,
    _fixed_window,
    _negative_candidates,
    _resample,
)


def test_resample_44100_to_16000() -> None:
    source = np.zeros((44_100, 4), dtype=np.float64)
    result = _resample(source)
    assert result.shape == (16_000, 4)
    assert result.dtype == np.float32


def test_fixed_window_normalizes_each_channel_independently() -> None:
    source = np.ones((16_000, 4), dtype=np.float32)
    source *= np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = _fixed_window(source, center_ms=500)
    assert result.shape == (4, WINDOW_SAMPLES)
    assert np.allclose(np.max(np.abs(result), axis=1), 1.0)


def test_fixed_window_pads_at_recording_start() -> None:
    source = np.ones((16_000, 4), dtype=np.float32)
    result = _fixed_window(source, center_ms=0)
    assert result.shape == (4, WINDOW_SAMPLES)
    assert np.all(result[:, : WINDOW_SAMPLES // 2] == 0)


def test_negative_candidates_avoid_guarded_events() -> None:
    events = [(400.0, 500.0, 100.0, "P01")]
    starts = _negative_candidates(1_000.0, events, guard_ms=50.0)
    assert starts == [0.0, 550.0, 750.0]
    for start in starts:
        assert start + 200 <= 350 or start >= 550
