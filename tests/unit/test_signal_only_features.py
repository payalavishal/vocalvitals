"""Guards for the research-only signal-only baseline feature builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "experiment_signal_only_baseline.py"
    spec = importlib.util.spec_from_file_location("experiment_signal_only_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_feature_builder_is_metadata_free_and_finite() -> None:
    module = _load_script()
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((3, 4, 3200), dtype=np.float32)
    features = module.build_features(windows)
    assert features.shape[0] == 3
    assert np.isfinite(features).all()


def test_feature_builder_is_deterministic() -> None:
    module = _load_script()
    rng = np.random.default_rng(1)
    windows = rng.standard_normal((2, 4, 3200), dtype=np.float32)
    assert np.array_equal(module.build_features(windows), module.build_features(windows))


def test_feature_builder_rejects_position_metadata_argument() -> None:
    module = _load_script()
    with pytest.raises(TypeError):
        module.build_features(np.zeros((1, 4, 3200), dtype=np.float32), start_ms=0.0)
