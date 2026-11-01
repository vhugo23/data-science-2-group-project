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
| `dummy.py` | Fake score tables (ids start with `dummy_`) for development, with a `fold` column (0-4 on train, -1 elsewhere) so dummy runs pass the out-of-fold check. `degrade_scores` weakens a dummy table for exercising the condition code paths; test scaffolding only, real degradation means re-scoring degraded inputs. |
| `../metrics.py` | Shared metrics (not fusion-only): `classification_metrics` is unchanged; added `exact_eer` (interpolated ROC crossing, ties admitted together), `average_precision` (PR-AUC), `recall_at_fpr`, `metrics_at_threshold`, `safe_logit` / `safe_log_score` (explicit `eps`, default 1e-6), `cluster_bootstrap_ci` and `paired_cluster_bootstrap_ci` (resample clusters, percentile interval, seeded). Higher score = more likely positive; thresholds use `>=`. |
| `models.py` | `fit_fusion` fits audio-only, text-only, the fixed product rule (sum of log-scores, equal weights, **default eps**) and a logistic regression on log-scores (**tuned eps**). Fits on train only and tunes C and eps on validation only (ties go to the smaller C, then the smaller eps); any other split raises. Requires out-of-fold train scores: both tables need a `fold` column that is 0 or more on train rows, otherwise it raises naming `fold`; `require_oof=False` is an explicit escape hatch that records `oof_verified=False` and marks the model `plumbing_only`. Stores a validation threshold per system at 5% FPR, the learned weight ratio `w_audio / w_text`, and SHA-256 hashes of the train+validation inputs. `save_model` / `load_model` freeze and reload it as JSON (never overwrites). Warns (`NegativeWeightWarning`) if a learned weight is negative. |
| `evaluate.py` | `evaluate` scores a frozen model on one named condition (`clean`, `noise` + severity, `in_the_wild`, `text_prefix` + fraction) and returns a tidy table: ROC-AUC, PR-AUC, and TPR/FPR at the frozen threshold per system, dialogue-clustered bootstrap CIs, and the paired learned-minus-product interval. Every row carries `condition, severity, split, seed, dummy, plumbing_only, scores_hash`. Refuses the train split; the test split needs `allow_test=True`. Raises if any pair id is missing from a score table. `weight_ratio_sweep` gives AUC against `w_audio / w_text` for a condition and states the `eps` it used (default eps). Neither function fits or tunes. |
| `tests/` | Run from the repo root with `python -m pytest src/fusion/tests`. |

## Pairing rules

- Labels: audio 1 = spoof, text 1 = scam. Fusion target = spoof AND scam.
- Pairs are built inside each split from that split's clips and dialogues only. A split missing from either pool is skipped with a warning (for example, text test scores are not available yet).
- Each clip is used at most once. Each dialogue is paired equally often with genuine and spoof clips, so `uses_per_dialogue` (the most times one dialogue appears) must be even. The default 2 means once with a genuine clip and once with a spoof clip; 4 is allowed for sensitivity checks. Odd values and values below 2 are rejected.
- The four cells (genuine/spoof x benign/scam) have equal size in every split. By default the pair count is the most the scarcest ingredient allows; `n_per_cell` can request fewer, and a request that is too large raises an error.
- Manifest columns: `pair_id, split, audio_id, text_id, audio_label, text_label, target, cell, dummy`. `dummy` is a required argument to `build_pairs`: pass `True` for fake or smoke-run scores and `False` otherwise. It is copied into every row.
- Resample dialogues (and clips if ever reused), not pairs, when computing confidence intervals.
