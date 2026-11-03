# Dataset provenance and integrity

## Source archives

The original ZIP files were found in the user's Downloads folder on 2026-11-02. The archives were preserved; extracted copies were placed under C:\VOCALVITALS\data\raw\.

| Archive | Compressed size (bytes) | SHA-256 |
|---|---:|---|
| signals.zip | 1,406,291,807 | 16704601076E34BEBEE136552A2B4AD6D930CC74B914E06BA62D413792B1215D |
| labels.zip | 301,884 | 017C7FB60906ED8858F56C2EFC43564E25845098D7497BF485AEF0E30FA5F70B |

## Extracted inventory

- 1,285 signal files in data/raw/signals/.
- 1,285 annotation files in data/raw/labels/.
- All 1,285 signal/annotation basenames match.
- Signal arrays are two-dimensional, with four channels, and float64 dtype.
- No non-finite signal values were found.
- 1,876 annotation rows were read from the CSV files.
- 174 participant IDs appear in annotation rows. The processed windows cover 175 IDs because one participant appears only in recordings with empty annotation files.
- 193 annotation files have no event rows.
- Three annotated events are shorter than the configured 20 ms minimum.
- Six annotated events exceed 200 ms.
- Three annotations have negative start times; no annotation end time exceeds its recording duration.
- No rows had end-before-start bounds or duration values inconsistent with end minus start (tolerance 1.5 ms).

The source data does not include a verified independent aspiration outcome label. It supports a swallow-event versus non-swallow-window research task only.

## Processed dataset

Current preprocessing output:

- data/processed/swallow_binary_windows.npz
- data/processed/swallow_binary_metadata.json
- Shape: (3154, 4, 3200)
- dtype: float32
- Class counts: 1,873 swallow and 1,281 non-swallow
- Target sample rate: 16,000 Hz
- Window duration: 200 ms
- Channels: 4
- Resampling: polyphase 160/441 conversion from 44,100 Hz
- Negative-window guard around annotated events: 50 ms
- Minimum accepted event duration: 20 ms
- Negative-window selection seed: 42

The processed positive count (1,873) differs from the historical report's 1,827. The historical report's full filtering/extraction code was not recovered. Do not silently force the new data to match the historical count.

## Participant split

The split manifest is at artifacts/reports/participant_split.json.

| Split | Windows | Participants | Non-swallow | Swallow |
|---|---:|---:|---:|---:|
| Train | 2,248 | 121 | 913 | 1,335 |
| Validation | 449 | 28 | 183 | 266 |
| Test | 457 | 26 | 185 | 272 |

The split uses StratifiedGroupKFold with participant IDs as groups. There is no participant overlap between train, validation, and test. The test set must not be used for model selection.

## Reproduction

Run the following commands from C:\VOCALVITALS after activating the project environment:

- python scripts\audit_raw_data.py
- python -m vocalvitals.data.preprocessing
- python scripts\create_splits.py

Do not upload the raw archives, extracted signals, processed arrays, or participant-level metadata to a public repository.


