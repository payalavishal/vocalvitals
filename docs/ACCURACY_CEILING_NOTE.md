# Accuracy ceiling note (real-data binary pipeline)

The user requested 98–99% accuracy on the current dataset. A thorough
audit of the recovered data, preprocessing, splits, and model families
shows that this is **not attainable** on this dataset, and no code change
can make it so without fabricating results (which the project's
reconstruction principles explicitly forbid).

This note now reports measured numbers from two reproducible experiments
rather than estimates. See EXP-006 and EXP-007 in `docs/EXPERIMENT_LOG.md`.

## Measured signal-only ceiling

`scripts/experiment_signal_only_baseline.py` builds a deliberately rich
signal-only feature set (3,405 log-mel, envelope, spectral-shape and
cross-channel features) and excludes **all** metadata, including window
position.

| Model | Validation AUC |
|---|---:|
| Extra Trees | 0.7242 |
| Random Forest | 0.7219 |
| RBF SVM | 0.6696 |
| Logistic regression | 0.6567 |
| Best pre-specified blend | 0.7273 |

Blend at validation-selected threshold: 67.74% validation accuracy,
61.88% on the spent test split (AUC 0.6643).

Earlier experiments with smaller feature sets sat between 0.65 and 0.70
AUC. Better features moved the ceiling by about seven AUC points and no
further. Every model family converges on the same region, which is the
signature of an information limit rather than a modelling limit.

## Measured position-augmented result (not a detector)

`scripts/experiment_position_augmented.py` adds 16 window-position
features derived from recording metadata and stacks four learners.

| Split | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Validation | 95.45% | 95.35% | 0.9891 |
| Test (spent) | 92.45% | 92.48% | 0.9846 |

## Why position helps (and why it is not clinical signal)

`start_ms` (window position within its recording) is a strong predictor
because of how negatives were sampled: swallow events cluster early in these
short clips, while non-swallow windows are sampled uniformly across the
whole recording. This is a sampling artifact, not a physiological
discriminant. A real detector must not rely on it; it is documented here
only to explain the observed ceiling.

Two checks support that reading. First, adding position alone lifts AUC
from about 0.73 to 0.99 while the waveforms are unchanged. Second, roughly
92–95% is reachable from position features even under a deliberately leaky
random window split, which is what a label leak looks like.

## Why 98–99% is still not reachable

The remaining gap to 99% is not a tuning gap. Within this dataset the
labelled quantity is "window centred on an annotated swallow event" versus
"window from guarded background", and the signals themselves overlap
heavily across unseen participants. Two thirds to three quarters of that
distinction is recoverable from the waveform; the rest is not present in
200 ms of four-channel audio from this cohort.

## Recommendation

1. Do not report 98–99% accuracy for this dataset. The defensible figures
   are ~0.73 validation AUC signal-only, and ~92% test accuracy *only* when
   the position artefact is explicitly used and disclosed as non-clinical.
2. Treat the signal-only result as the honest baseline for any report,
   thesis or demo of the model.
3. Improving accuracy requires new data: more participants, denser
   annotation inside each recording (which removes the position shortcut),
   or an independent aspiration outcome label, which this dataset does not
   contain.
4. The held-out test split has been evaluated and is spent. A new unbiased
   estimate needs a fresh cohort or a pre-registered nested group-CV
   protocol.
