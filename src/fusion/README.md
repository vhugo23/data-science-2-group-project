# Fusion track

**Owner:** Joshua Eworo

Keep source pairing, leakage checks, score-level fusion, robustness experiments, and shared evaluation here. Put track notebooks under `notebooks/fusion/`.

Pair records must retain both the audio clip ID and dialogue ID. Partition source pools before pairing so no source crosses splits. Train learned fusion on out-of-fold component scores; freeze models, fusion weights, and thresholds before external tests and robustness comparisons.

## Modules

| File | Purpose |
| --- | --- |
| `scores.py` | Adapter: renames each track's score table to `sample_id, split, label, score` (text: `dialogue_id`→`sample_id`, `scam_score`→`score`) and validates splits, labels and scores. Extra columns (`fold`, `turn_fraction`, ...) are kept. `external` rows are rejected unless `allow_external=True`. |
| `pairing.py` | `build_pairs` makes the pair manifest; `source_counts` reports unique clips/dialogues and reuse per split. |
| `checks.py` | `check_pairs` re-checks a manifest against the pools and raises `LeakageError` listing every violated rule. |
| `dummy.py` | Fake score tables (ids start with `dummy_`) for development. |
| `tests/` | Run from the repo root with `python -m pytest src/fusion/tests`. |

## Pairing rules

- Labels: audio 1 = spoof, text 1 = scam. Fusion target = spoof AND scam.
- Pairs are built inside each split from that split's clips and dialogues only. A split missing from either pool is skipped with a warning (for example, text test scores are not available yet).
- Each clip is used at most once. Each dialogue is paired equally often with genuine and spoof clips, so it is used at most `2 * (max_reuse // 2)` times; the default `max_reuse=3` therefore means at most twice (once genuine, once spoof). `max_reuse` below 2 is rejected.
- The four cells (genuine/spoof x benign/scam) have equal size in every split. By default the pair count is the most the scarcest ingredient allows; `n_per_cell` can request fewer, and a request that is too large raises an error.
- Manifest columns: `pair_id, split, audio_id, text_id, audio_label, text_label, target, cell, dummy`. `dummy` is true when built from fake or smoke scores (inferred from `dummy_` ids; pass `dummy=True` for smoke-run scores).
- Resample dialogues (and clips if ever reused), not pairs, when computing confidence intervals.
