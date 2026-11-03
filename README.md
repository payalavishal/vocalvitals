# VocalVitals

Research prototype for swallow-event detection from four-channel cervical-auscultation recordings, with a separate synthetic demonstration pipeline.

> **Research-only notice:** VocalVitals is not a medical device. It is not clinically validated and must not be used for diagnosis, aspiration detection, treatment decisions, or patient-care alerts.

## Two separate pipelines

1. **Real-data binary pipeline:** classify annotated 200 ms, four-channel windows as swallow or non-swallow.
2. **Synthetic demonstration pipeline:** demonstrate Normal / Delayed / Aspirating labels using generated signals. These labels are synthetic and are not clinical findings.

The two pipelines must remain clearly separated in code, APIs, reports, and UI. The API and dashboard accept synthetic_demo events only; the real-data binary model is not connected to them.

## Reconstruction status

The previous codebase was lost. The prior technical report is treated as a reconstruction specification, not proof that historical results can be reproduced.

### Historical results (not yet reproduced)

The previous report described 3,108 windows (1,827 swallow and 1,281 non-swallow), with validation accuracy 99.41%, F1 99.50%, and AUC 0.9999. An earlier README also described a much weaker baseline. Neither result is currently verified. We will not claim equivalence until the original split, preprocessing, checkpoint, and evaluation procedure are reconciled.

### Recovered source data

The original archives remain in the user's Downloads folder. Extracted copies are stored under data/raw/:

- 1,285 .npy four-channel signal files.
- 1,285 matching .csv annotation files.
- 1,876 annotation rows across 174 participant IDs. Processed windows cover 175 IDs because one participant appears only in recordings with empty annotation files.
- 193 annotation files contain no swallow rows.
- All signal arrays have four channels and finite values; source arrays are float64.
- The source annotations include three events shorter than 20 ms, six longer than 200 ms, and three events with negative start times. These are logged and handled explicitly by the preprocessing implementation.

The processed dataset currently contains 3,154 windows: 1,873 swallow and 1,281 non-swallow. This differs from the historical report's 3,108 windows. The difference is intentional and visible; we will investigate it rather than silently dropping records to force a match.

## Current real-data pipeline

The initial 50 ms-guard dataset was later audited and found to contain negative windows overlapping positive windows. A separately versioned corrected 100 ms-guard dataset is used for newer experiments; the original preprocessing defaults and saved baseline are retained for reproducibility. See docs/EXPERIMENT_LOG.md for the audit and results. The 100 ms-guard balanced ensemble scored 71.16% validation accuracy / 0.7507 AUC, but only 59.67% accuracy / 0.6506 AUC on its one-time held-out test. It is not a useful or validated clinical detector, and the test set is now spent. Do not use validation accuracy as expected real-world accuracy.

The current preprocessing implementation:

- Reads 44.1 kHz recordings and uses polyphase anti-aliased resampling to 16 kHz.
- Creates fixed 200 ms windows with four channels in channel-first order.
- Centers swallow windows on the clipped annotated event interval.
- Uses per-channel peak normalization.
- Samples 1,281 non-swallow windows from non-overlapping regions outside swallow annotations plus a 50 ms guard.
- Uses random seed 42 for negative-window selection.
- Writes the processed arrays and full window metadata under data/processed/.

The original 50 ms-guard baseline split used 2,248 train, 449 validation, and 457 test windows. Subsequent corrected 100 ms-guard balanced experiments use 2,676 train, 527 validation, and 543 test windows, split by participant with no participant overlap (127 / 25 / 23 participants). The latest experiment's test set has already been evaluated once and is spent; do not tune against it.

## Repository layout

- data/raw/ - extracted original recordings and annotations (never commit raw/private data).
- data/interim/ - intermediate validated files.
- data/processed/ - generated windows and metadata (excluded from Git).
- src/vocalvitals/ - application and ML source code.
- configs/ - reproducible configuration.
- scripts/ - dataset audit, preprocessing, and split utilities.
- artifacts/ - reports and model checkpoints (excluded from Git by default).
- tests/ - unit and integration tests.
- docs/ - design decisions, data provenance, and experiment records.

## Setup

Use Python 3.11. A local virtual environment is recommended:

1. Open PowerShell.
2. Change directory to C:\VOCALVITALS.
3. Activate .venv with .\.venv\Scripts\Activate.ps1.
4. Install the project with python -m pip install -e ".[dev,api,synthetic]".
5. Restore pinned packages with python -m pip install -r requirements-lock.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple.

The lockfile pins the CUDA-enabled PyTorch 2.11.0+cu128 build used on the RTX 3050 laptop. For another machine, install the PyTorch build appropriate to its GPU or CPU.

## Rebuild commands

- Inspect raw files: python scripts\inspect_dataset.py
- Audit raw data: python scripts\audit_raw_data.py
- Analyze annotation geometry: python scripts\analyze_annotations.py
- Diagnose validation only: python scripts\diagnose_validation.py
- Compare event/background signal energy: python scripts\compare_event_background.py
- Rebuild binary windows: python -m vocalvitals.data.preprocessing
- Create participant-disjoint splits: python scripts\create_splits.py
- Run tests: python -m pytest -q
- Run static checks: python -m ruff check .
- Train binary model: python scripts\train_binary.py
- Signal-only ceiling experiment (research): python scripts\experiment_signal_only_baseline.py
- Position-augmented research ensemble (not a detector): python scripts\experiment_position_augmented.py
- Evaluate held-out test once, only after protocol freeze: python scripts\evaluate_binary.py --confirm-final
- Train synthetic demo: python scripts\train_synthetic_demo.py
- Start demo API: python -m uvicorn vocalvitals.api.app:app --reload
- Send synthetic demo events: python scripts\send_synthetic_demo_events.py
- Start demo dashboard: python -m streamlit run src\vocalvitals\synthetic\dashboard.py

## Reconstruction principles

- Preserve participant-wise separation; never split windows randomly across participants.
- Keep an untouched test set separate from validation/model selection.
- Record data checksums, preprocessing parameters, package versions, random seeds, and split manifests.
- Do not describe the binary model as detecting aspiration.
- Do not claim the synthetic pipeline is clinically meaningful.
- Do not invent or tune results to match historical numbers.

## Milestones

- **M0 - Workspace scaffold:** complete.
- **M1 - Dataset recovery and inventory:** complete; checksums and source details are recorded in docs/DATA_PROVENANCE.md.
- **M2 - Reproducible binary preprocessing and participant splits:** implemented and unit-tested; historical count discrepancy remains under investigation.
- **M3 - Binary model reconstruction:** exploratory model comparison completed. The corrected balanced ensemble reached 71.16% validation accuracy / 0.7507 AUC, but only 59.67% accuracy / 0.6506 AUC on its one-time held-out test. Five-fold participant-group out-of-fold AUC was 0.6892 for the spectrogram CNN. A signal-only ceiling study (EXP-006, `scripts/experiment_signal_only_baseline.py`) with a much richer feature set reached 67.74% validation accuracy / 0.7273 AUC, confirming that waveform information alone caps out near 0.73 AUC on this dataset. A position-augmented research ensemble reaches 92.45% test accuracy but only by exploiting a window-sampling artefact; it is research-only and must not be reported as detector accuracy. Generalisation remains weak; the model is not clinically useful or validated. See docs/ACCURACY_CEILING_NOTE.md and docs/EXPERIMENT_LOG.md.
- **M4 - Synthetic demo reconstruction:** implemented and trained on generated waveforms only; validation accuracy is not clinical evidence.
- **M5 - API/dashboard/MQTT demo:** implemented and locally smoke-tested for HTTP. MQTT broker and real piezo/Raspberry Pi hardware are not validated.










