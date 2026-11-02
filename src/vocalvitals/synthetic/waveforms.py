"""Generate artificial waveforms for a clearly separated three-class demo."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class DemoClass(IntEnum):
    NORMAL = 0
    DELAYED = 1
    ASPIRATING = 2


CLASS_NAMES = ("Normal", "Delayed", "Aspirating")
SAMPLE_RATE = 16_000
DURATION_SECONDS = 2.0


def _damped_burst(time: np.ndarray, onset: float, frequency: float, amplitude: float) -> np.ndarray:
    relative = time - onset
    active = relative >= 0
    envelope = np.zeros_like(time)
    envelope[active] = np.exp(-relative[active] * 5.5)
    phase = 2 * np.pi * frequency * np.maximum(relative, 0)
    tone = np.sin(phase) + 0.28 * np.sin(2.1 * phase)
    return amplitude * envelope * tone


def generate_waveform(
    label: DemoClass | int,
    seed: int | None = None,
    sample_rate: int = SAMPLE_RATE,
    duration_seconds: float = DURATION_SECONDS,
) -> np.ndarray:
    """Return one artificial 2-second mono waveform in float32.

    These patterns are invented for software demonstration and do not model
    physiology, aspiration, or any clinical condition.
    """
    demo_class = DemoClass(label)
    rng = np.random.default_rng(seed)
    sample_count = int(sample_rate * duration_seconds)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    signal = 0.012 * rng.standard_normal(sample_count).astype(np.float32)

    if demo_class == DemoClass.NORMAL:
        signal += _damped_burst(time, 0.48, 115.0, 0.55)
        signal += _damped_burst(time, 0.92, 165.0, 0.25)
    elif demo_class == DemoClass.DELAYED:
        signal += _damped_burst(time, 1.18, 105.0, 0.55)
        signal += _damped_burst(time, 1.55, 150.0, 0.22)
    else:
        signal += _damped_burst(time, 0.62, 120.0, 0.45)
        turbulence = rng.standard_normal(sample_count).astype(np.float32)
        mask = (time >= 0.72) & (time <= 1.35)
        signal[mask] += 0.10 * turbulence[mask]
        signal += _damped_burst(time, 1.42, 190.0, 0.18)

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32, copy=False)


def make_dataset(samples_per_class: int = 120, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    waveforms: list[np.ndarray] = []
    labels: list[int] = []
    for label in DemoClass:
        for _ in range(samples_per_class):
            waveforms.append(generate_waveform(label, seed=int(rng.integers(0, 2**31 - 1))))
            labels.append(int(label))
    order = rng.permutation(len(labels))
    return (
        np.stack(waveforms)[order],
        np.asarray(labels, dtype=np.int64)[order],
    )
