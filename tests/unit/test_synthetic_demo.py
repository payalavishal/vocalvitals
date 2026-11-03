import numpy as np
import pytest


def test_synthetic_waveforms_have_expected_shape_and_range() -> None:
    from vocalvitals.synthetic.waveforms import (
        CLASS_NAMES,
        DURATION_SECONDS,
        SAMPLE_RATE,
        DemoClass,
        generate_waveform,
    )

    assert len(CLASS_NAMES) == 3
    for label in DemoClass:
        waveform = generate_waveform(label, seed=123)
        assert waveform.shape == (int(SAMPLE_RATE * DURATION_SECONDS),)
        assert waveform.dtype == np.float32
        assert np.isfinite(waveform).all()
        assert np.max(np.abs(waveform)) <= 1.0


def test_logmel_has_64_mel_bins() -> None:
    pytest.importorskip("librosa")
    from vocalvitals.synthetic.features import waveform_to_logmel
    from vocalvitals.synthetic.waveforms import DemoClass, generate_waveform

    features = waveform_to_logmel(generate_waveform(DemoClass.NORMAL, seed=4))
    assert features.shape[0] == 64
    assert features.dtype == np.float32
