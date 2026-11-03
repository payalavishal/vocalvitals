# Safety and limitations

## Intended use

VocalVitals is a student research and software demonstration prototype only. It is not a medical device and is not clinically validated.

## Real-data model

- The supplied annotations identify swallow events; they do not label aspiration outcomes, delayed swallowing, or clinical dysphagia states.
- The real-data model predicts only swallow / non-swallow windows.
- The historical 99% validation claim has not been reproduced.
- An initial participant-disjoint baseline validation AUC was 0.5263.
- A corrected 100 ms-guard balanced experiment reached 71.16% validation accuracy and 0.7507 validation AUC, but its one-time held-out test accuracy was only 59.67% and AUC 0.6506.
- The held-out test accuracy was below the 68.67% majority-class baseline for that test split.
- Five-fold participant-group out-of-fold AUC was 0.6892 for the spectrogram CNN. The validation/test gap and lower out-of-fold score indicate weak participant-level generalization.
- The held-out test set has been evaluated once and is spent. It must not be reused for model selection or threshold tuning.
- Further unbiased evaluation requires a genuinely new cohort of participants or a pre-specified nested group cross-validation protocol.
- Do not deploy any checkpoint or use these predictions to guide patient care.

## Synthetic three-class model

- Normal / Delayed / Aspirating labels are generated from artificial waveforms.
- Synthetic validation accuracy measures only separation of generated patterns; it is not real swallow, delay, aspiration, or clinical evidence.
- The synthetic predictor is not connected to real sensor recordings or the real-data binary model.
- Never describe synthetic outputs as a diagnosis or real aspiration detection.

## Service and data

- The FastAPI store is in-memory and loses events when the process stops.
- No authentication, authorization, encryption, audit trail, durable storage, or retention controls are implemented.
- MQTT has no broker-level authentication/TLS configuration in this prototype and is not integration-tested.
- Do not submit real patient identifiers, recordings, or protected health information to the demo API.
- Do not publish raw recordings, annotation CSVs, processed windows, checkpoints, serialized estimators, or participant-level metadata to a public Git repository.

## Hardware

No piezo sensor, analog conditioning, ADC, Raspberry Pi, latency, power, or caregiver-notification path has been verified. A live hardware claim must wait for documented bench testing and appropriate clinical governance.
