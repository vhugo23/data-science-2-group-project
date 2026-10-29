# Fusion track

**Owner:** Joshua Eworo

Keep source pairing, leakage checks, score-level fusion, robustness experiments, and shared evaluation here. Put track notebooks under `notebooks/fusion/`.

Pair records must retain both the audio clip ID and dialogue ID. Partition source pools before pairing so no source crosses splits. Train learned fusion on out-of-fold component scores; freeze models, fusion weights, and thresholds before external tests and robustness comparisons.
