# Audio track

**Owner:** Hugo Santos

Keep audio preprocessing, frozen speech-encoder embeddings, spoof classifiers, and audio-only benchmarks here. Put track notebooks under `notebooks/audio/`.

Export one row per source clip with a stable clip ID, shared split name (`train`, `validation`, or `test`), spoof label, and model score. Keep official ASVspoof split membership intact and document label encoding and preprocessing settings with results.

## Modules

| File | Purpose |
| --- | --- |
| `features.py` | `SETTINGS` (single source for preprocessing, MFCC, and encoder revision), ASVspoof protocol loading, trim/normalize/fixed window, noise and mu-law degradations, MFCC features. |
| `encoder.py` | Frozen wav2vec 2.0 at the pinned revision, time-mean embeddings, and the embedding cache. |
| `evaluate.py` | Logistic-regression classifier, validation-F1 threshold, ROC AUC/EER/F1. |
| `tests/` | Synthetic-signal tests; no dataset needed. Run from the repo root: `python -m unittest discover -s src/audio/tests`. |

`notebooks/audio/audio_spoof_detection.ipynb` imports these modules. Change settings in `SETTINGS`, then delete old embedding caches under `outputs/audio/`.
