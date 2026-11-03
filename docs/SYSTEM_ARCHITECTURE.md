# System architecture

## 1. Real-data binary research pipeline

Raw four-channel cervical-auscultation recordings and event annotations flow through:

1. Dataset pairing and raw-data audit.
2. Polyphase resampling from 44.1 kHz to 16 kHz.
3. Fixed 200 ms event and non-event window extraction.
4. Per-channel peak normalization.
5. Participant-disjoint train / validation / test split.
6. Three-block 1D CNN training and validation-only model selection.
7. Offline binary inference helper returning swallow / non-swallow only.

The binary model is experimental and has not demonstrated useful participant-held-out validation performance. It is not connected to the synthetic API or dashboard.

## 2. Synthetic demonstration pipeline

Artificial waveforms are generated for three software-demo classes:

- Normal
- Delayed
- Aspirating

The waveform generator is deterministic for a given seed. The audio is converted to 64-bin log-mel features and passed to a separate 2D CNN. The predictor is explicitly marked synthetic-only.

The synthetic classifier output can be sent to the demo API using the event sender script.

## 3. Demo service path

- FastAPI POST /event accepts only source=synthetic_demo.
- GET /patients returns in-memory patient summaries.
- GET /patients/{patient_id}/trend returns recent demo events.
- EventStore retains up to 500 events per patient in process memory.
- Optional MQTT publisher sends synthetic_demo event JSON to vocalvitals/demo/events.
- Optional API MQTT bridge subscribes to the same topic and inserts validated demo events into EventStore.
- Streamlit dashboard reads the HTTP API and displays demo events.

The API has no database, authentication, access control, or durable retention. Do not use it with real patient data.

## 4. Hardware boundary

No verified piezoelectric sensor, analog front-end, ADC, Raspberry Pi, sensor calibration, or end-to-end hardware capture is included in this reconstruction. MQTT transport is implemented but has not been tested against a broker. The laptop microphone and synthetic model must not be presented as a validated swallow or aspiration monitor.
