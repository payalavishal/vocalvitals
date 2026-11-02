# VocalVitals reconstruction log

## 2026-11-02 — Workspace initialized

- Confirmed `C:\VOCALVITALS` exists and was empty at inspection.
- Created source, data, artifact, configuration, documentation, and test directories.
- Added initial package metadata and safety-focused project documentation.
- No Python launcher, Python executable, or Git executable was found on PATH during the initial environment check. Verify/install these before running project code.
- Dataset availability has not yet been confirmed.
- No model metrics have been reproduced in this reconstruction.

## Historical implementation facts to preserve

- Real-data task: binary swallow / non-swallow classification.
- Input described in prior report: (4, 3200), 16 kHz, 200 ms.
- Historical dataset report: 3,108 windows, 1,827 positive and 1,281 negative.
- Historical validation report: 2,600 training windows and 508 validation windows; no independent test set was documented.
- Historical reported metrics: 99.41% accuracy, 100% swallow recall, 98.55% specificity, 99.50% F1, AUC 0.9999.
- A separate older README reportedly showed 56% accuracy and AUC 0.48. This discrepancy must be resolved through a reproducible experiment.
- Synthetic three-class demo is separate from real-data binary inference.
- Earlier extraction, resampling, and deployment limitations are listed in the source technical documentation and must be addressed rather than silently carried forward.


## 2026-11-03 — Dataset recovery and first model baseline

- Recovered both original ZIP archives from Downloads and preserved them unchanged.
- Verified SHA-256 checksums in docs/DATA_PROVENANCE.md.
- Extracted 1,285 signals and 1,285 matching label CSV files under data/raw/.
- Audited 1,876 annotation rows, 174 participant IDs in annotations, 193 empty annotation CSVs, and four-channel finite float64 arrays.
- Installed Python 3.11.9, created .venv, installed core dependencies, and installed PyTorch 2.11.0+cu128 with torchaudio 2.11.0+cu128.
- Verified CUDA is available on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Implemented polyphase 44.1 kHz -> 16 kHz resampling, 200 ms window extraction, per-channel peak normalization, and guarded non-swallow sampling.
- Current processed dataset: 3,154 windows (1,873 swallow, 1,281 non-swallow), shape (N, 4, 3200), float32.
- Current participant-disjoint split: train 2,248 / validation 449 / test 457; participant overlap is zero.
- Added a three-block 1D CNN and validation-only training/diagnostic scripts.
- First CNN baseline: validation AUC 0.5263, F1 0.7189 at threshold 0.5; not a successful detector.
- Handcrafted random-forest baseline validation AUC: 0.5817. No test evaluation has been run.
- Eight unit tests pass; Ruff checks pass.
- Initialized a local Git repository and configured the existing GitHub repository as origin. No commit or push has been made.
- The historical 3,108-window count and 99% validation results remain unreproduced. See docs/EXPERIMENT_LOG.md.


## 2026-11-03 — Synthetic demo and service reconstruction

- Implemented artificial waveform generation for three distinct synthetic classes, 64-bin log-mel extraction, and a separate 2D CNN.
- Trained the synthetic demo model; validation accuracy was 100% on generated data only. This is not clinical evidence.
- Implemented synthetic-only FastAPI event ingestion, patient listing, and trend endpoints.
- Added optional MQTT publisher and API subscriber bridge restricted to synthetic_demo events.
- Implemented a Streamlit synthetic demo dashboard.
- Smoke-tested six synthetic events through the predictor and HTTP API; dashboard returned HTTP 200.
- Added API integration tests, synthetic feature tests, and updated the dependency lock file.
- Full suite: 14 passed; Ruff clean; pip check clean.
- No MQTT broker, piezo/ADC/Raspberry Pi hardware, database, authentication, or caregiver alert has been tested or integrated.
