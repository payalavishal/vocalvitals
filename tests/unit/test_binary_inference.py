import numpy as np
import pytest


def test_binary_predictor_checks_window_shape(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from vocalvitals.inference.binary import BinarySwallowPredictor
    from vocalvitals.models.binary_cnn import SwallowBinary1DCNN

    checkpoint = tmp_path / "model.pt"
    model = SwallowBinary1DCNN()
    torch.save(
        {
            "model_name": "SwallowBinary1DCNN",
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    predictor = BinarySwallowPredictor(checkpoint)
    prediction = predictor.predict(np.zeros((4, 3200), dtype=np.float32))
    assert prediction["label"] in {"swallow", "non_swallow"}
    assert prediction["clinical_validity"] is False

    with pytest.raises(ValueError, match="Expected window shape"):
        predictor.predict(np.zeros((4, 100), dtype=np.float32))
