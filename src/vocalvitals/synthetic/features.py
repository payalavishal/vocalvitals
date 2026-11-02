"""Log-mel features for the synthetic audio demonstration."""

from __future__ import annotations

import librosa
import numpy as np

from vocalvitals.synthetic.waveforms import SAMPLE_RATE


def waveform_to_logmel(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = 64,
) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {waveform.shape}")
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32, copy=False),
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32, copy=False)


def batch_logmel(waveforms: np.ndarray) -> np.ndarray:
    return np.stack([waveform_to_logmel(waveform) for waveform in waveforms])
