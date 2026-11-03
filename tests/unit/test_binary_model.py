import pytest


def test_binary_cnn_returns_one_logit_per_window() -> None:
    torch = pytest.importorskip("torch")
    from vocalvitals.models.binary_cnn import SwallowBinary1DCNN

    model = SwallowBinary1DCNN(in_channels=4)
    output = model(torch.zeros(2, 4, 3200))
    assert output.shape == (2,)
