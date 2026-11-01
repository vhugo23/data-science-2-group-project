"""Evaluate frozen fusion models on a named condition, and sweep the audio/text weight ratio.

Both functions read pair tables and score tables passed in as arguments. Each score
table must hold one condition (clean, noise at a severity, in_the_wild, text_prefix at a
fraction), so run one call per condition and concatenate the results. Nothing here fits or
tunes: thresholds and weights come from the frozen model. The validation split is the
default; the test split needs allow_test=True. A higher score means more likely positive.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .models import SYSTEMS, declared_dummy, evaluation_hash, join_scores
from src.metrics import (
    DEFAULT_EPS,
    average_precision,
    cluster_bootstrap_ci,
    metrics_at_threshold,
    paired_cluster_bootstrap_ci,
    safe_log_score,
)

CONDITIONS = ("clean", "noise", "in_the_wild", "text_prefix")
EVALUATION_SPLITS = ("validation", "test")
DEFAULT_RATIOS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RESULT_COLUMNS = ["condition", "severity", "split", "seed", "dummy", "plumbing_only", "scores_hash",
                  "system", "metric", "threshold", "estimate", "ci_low", "ci_high", "confidence",
                  "n_pairs", "n_clusters"]
SWEEP_COLUMNS = ["condition", "severity", "split", "dummy", "eps", "ratio", "roc_auc", "pr_auc"]


def _check_condition(condition, severity):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if condition in ("clean", "in_the_wild"):
        if severity is not None:
            raise ValueError(f"severity must be None for condition {condition!r}, got {severity!r}")
    elif severity is None:
        raise ValueError(f"condition {condition!r} needs a severity")
    elif not np.isfinite(severity):
        raise ValueError(f"severity must be a finite number, got {severity!r}")
    elif condition == "text_prefix" and not 0 < severity <= 1:
        raise ValueError(f"text_prefix severity is a fraction in (0, 1], got {severity!r}")


def _check_split(split, allow_test):
    if split == "train":
        raise ValueError("cannot evaluate on the train split: the models were fit on it")
    if split not in EVALUATION_SPLITS:
        raise ValueError(f"split must be one of {EVALUATION_SPLITS}, got {split!r}")
    if split == "test" and not allow_test:
        raise ValueError("the test split is only used for final, frozen runs: pass allow_test=True")


def evaluate(model, pairs, audio_scores, text_scores, *, condition, severity=None, split="validation",
             allow_test=False, seed=0, n_boot=1000, alpha=0.05):
    """Tidy results for one condition: one row per (system, metric), columns in RESULT_COLUMNS.

    Systems are audio_only, text_only, product (default eps) and learned (tuned eps), plus
    learned_minus_product, whose roc_auc and pr_auc rows are paired-bootstrap intervals for
    the difference. Metrics are roc_auc, pr_auc, and tpr_at_threshold / fpr_at_threshold at the
    system's frozen validation threshold (a sample is positive when score >= threshold; the
    threshold is never re-chosen here). Intervals resample whole dialogues (grouped by text_id)
    with the given seed. Every row carries condition, severity (NaN when none), split, seed, dummy,
    plumbing_only, and scores_hash, a hash of the score rows evaluated. dummy is True if the pair
    table flags it or the model was fit on dummy data. A higher score means more likely positive.

    Raises ValueError for an unknown condition or a missing/misplaced severity, the train split,
    the test split without allow_test, pair ids missing from a score table, and single-class targets.
    """
    _check_condition(condition, severity)
    _check_split(split, allow_test)
    dummy = declared_dummy(pairs) or model.dummy
    joined = join_scores(pairs, audio_scores, text_scores, split)
    y = joined["target"].to_numpy()
    groups = joined["text_id"].to_numpy()
    scores = model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
    common = {"condition": condition, "severity": np.nan if severity is None else float(severity),
              "split": split, "seed": seed, "dummy": bool(dummy), "plumbing_only": bool(model.plumbing_only),
              "scores_hash": evaluation_hash(pairs, audio_scores, text_scores, split)}
    boot = dict(n_boot=n_boot, alpha=alpha, seed=seed)

    def row(system, metric, threshold, result):
        return {**common, "system": system, "metric": metric, "threshold": threshold,
                "estimate": result["estimate"], "ci_low": result["low"], "ci_high": result["high"],
                "confidence": result["confidence"], "n_pairs": len(joined), "n_clusters": result["n_clusters"]}

    rows = []
    for system in SYSTEMS:
        threshold = model.thresholds[system]
        metrics = {
            "roc_auc": roc_auc_score,
            "pr_auc": average_precision,
            "tpr_at_threshold": lambda yy, ss, t=threshold: metrics_at_threshold(yy, ss, t)["tpr"],
            "fpr_at_threshold": lambda yy, ss, t=threshold: metrics_at_threshold(yy, ss, t)["fpr"],
        }
        for name, metric in metrics.items():
            rows.append(row(system, name, threshold, cluster_bootstrap_ci(y, scores[system], groups, metric, **boot)))
    for name, metric in (("roc_auc", roc_auc_score), ("pr_auc", average_precision)):
        result = paired_cluster_bootstrap_ci(y, scores["learned"], scores["product"], groups, metric, **boot)
        rows.append(row("learned_minus_product", name, np.nan, result))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def weight_ratio_sweep(pairs, audio_scores, text_scores, *, condition, severity=None, split="validation",
                       allow_test=False, ratios=DEFAULT_RATIOS, eps=DEFAULT_EPS):
    """ROC-AUC and PR-AUC of w_audio * log(audio) + w_text * log(text) as w_audio / w_text varies.

    One row per ratio (w_text is 1, w_audio is the ratio), columns in SWEEP_COLUMNS. No model
    is used and nothing is tuned or selected: this only shows how the metric moves with the
    ratio. The log-scores use the eps argument, which defaults to the default eps
    (metrics.DEFAULT_EPS, the same as the product rule); the eps column states the value used.
    Ratios must be positive and finite. Condition, severity and split rules match evaluate().
    A higher score means more likely positive.
    """
    _check_condition(condition, severity)
    _check_split(split, allow_test)
    ratios = [float(r) for r in ratios]
    if not ratios or not all(np.isfinite(r) and r > 0 for r in ratios):
        raise ValueError(f"ratios must be a non-empty list of positive finite numbers, got {ratios}")
    dummy = declared_dummy(pairs)
    joined = join_scores(pairs, audio_scores, text_scores, split)
    y = joined["target"].to_numpy()
    log_audio = safe_log_score(joined["audio_score"].to_numpy(), eps)
    log_text = safe_log_score(joined["text_score"].to_numpy(), eps)
    rows = []
    for ratio in ratios:
        score = ratio * log_audio + log_text
        rows.append({"condition": condition, "severity": np.nan if severity is None else float(severity),
                     "split": split, "dummy": bool(dummy), "eps": float(eps), "ratio": ratio,
                     "roc_auc": float(roc_auc_score(y, score)), "pr_auc": average_precision(y, score)})
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)
