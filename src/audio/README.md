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

Run `notebooks/audio/01_mfcc_baseline.ipynb` first, then `02_baseline_diagnostics.ipynb` and `03_wav2vec2_spoof_detection.ipynb`, each in a fresh kernel.

Smoke mode is default. Notebook 01 saves a shared split manifest, deterministic out-of-fold assignments, classifier, threshold, metrics and scores. Notebook 02 uses development data for diagnostics; notebook 03 compares frozen encoder results with saved MFCC metrics. Scores include `sample_id, split, label, score, fold`, with higher scores meaning spoof.

Generated files live in ignored `outputs/audio/{smoke,full}/development/`. Set `FULL_RUN = True` for full train/development data. Set `FINAL_EVALUATION = True` in notebooks 01 and 03 only after freezing choices to evaluate the official test split, writing separately under `final/`. External evaluation is deferred. Existing tracked output files remain tracked; new outputs are ignored.

Change settings in `SETTINGS`, then delete old embedding caches and the split manifest under the chosen output directory before rerunning.
