import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_score)
    return results


# ---------------------------------------------------------------------------
# Score-based metrics, safe transforms and cluster bootstrap.
#
# Convention for everything below: label 1 is the positive class and a higher
# score means more likely positive. A threshold t predicts positive when
# score >= t.
# ---------------------------------------------------------------------------

DEFAULT_EPS = 1e-6


def _validate(y_true, y_score, need_both_classes=True):
    """Return labels and scores as arrays, or raise ValueError with a clear message."""
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("labels and scores must be 1-dimensional")
    if len(y) != len(s):
        raise ValueError(f"labels and scores differ in length: {len(y)} vs {len(s)}")
    if len(y) == 0:
        raise ValueError("labels and scores are empty")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite (no NaN or infinity)")
    y = y.astype(int)
    if need_both_classes and y.min() == y.max():
        raise ValueError(
            f"need both classes present, but every label is {int(y[0])} (n={len(y)})"
        )
    return y, s


def _roc_points(y, s):
    """ROC operating points (fpr, tpr), one per distinct score, strictest first.

    Samples with equal scores are admitted together, so ties never depend on row
    order. The first point is (0, 0) (nothing accepted) and the last is (1, 1).
    """
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_of_group]
    fp = (last_of_group + 1) - tp
    tpr = np.r_[0.0, tp / y.sum()]
    fpr = np.r_[0.0, fp / (len(y) - y.sum())]
    return fpr, tpr


def exact_eer(y_true, y_score):
    """Equal error rate: the error rate where false positive rate = false negative rate.

    A higher score means more likely positive (label 1); a threshold t predicts
    positive when score >= t. Raises ValueError if only one class is present.

    Interpolation: the ROC curve is a polyline through one operating point per
    distinct score. The EER is where that polyline meets FPR = FNR (= 1 - TPR).
    If a point lies exactly on that line its FPR is returned; otherwise the
    crossing is found by linear interpolation between the two neighbouring
    points (equivalent to a randomised choice between the two thresholds). This
    is exact, unlike picking the closest operating point.

    Ties: samples with equal scores are admitted at the same threshold, so they
    form one diagonal ROC segment and the result does not depend on row order.
    A scorer with all scores equal has EER 0.5.
    """
    y, s = _validate(y_true, y_score)
    fpr, tpr = _roc_points(y, s)
    gap = fpr - (1.0 - tpr)  # nondecreasing along the curve, from -1 to +1
    i = int(np.argmax(gap >= 0))
    if gap[i] == 0:
        return float(fpr[i])
    weight = -gap[i - 1] / (gap[i] - gap[i - 1])
    return float(fpr[i - 1] + weight * (fpr[i] - fpr[i - 1]))


def average_precision(y_true, y_score):
    """PR-AUC as average precision: sum over thresholds of (recall step) * precision.

    A higher score means more likely positive (label 1). Raises ValueError if
    only one class is present. Uses the step-wise definition, not trapezoids.
    """
    y, s = _validate(y_true, y_score)
    return float(average_precision_score(y, s))


def recall_at_fpr(y_true, y_score, max_fpr):
    """Highest recall (TPR) among operating points whose FPR is <= max_fpr.

    A higher score means more likely positive (label 1); each operating point is
    a threshold t predicting positive when score >= t. Only real operating
    points are used (no interpolation), so the answer is always achievable.
    Raises ValueError if only one class is present or max_fpr is outside [0, 1].
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")
    y, s = _validate(y_true, y_score)
    fpr, tpr = _roc_points(y, s)
    return float(tpr[fpr <= max_fpr + 1e-12].max())


def metrics_at_threshold(y_true, y_score, threshold):
    """Confusion counts and rates at a threshold chosen in advance.

    A higher score means more likely positive (label 1); a sample is predicted
    positive when score >= threshold (inclusive). The threshold is never tuned
    here, so pass one chosen on validation data. Precision and F1 are 0.0 when
    undefined (nothing predicted positive), matching zero_division=0 in
    classification_metrics. Raises ValueError if only one class is present.
    """
    y, s = _validate(y_true, y_score)
    predicted = s >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    tn = int(np.sum(~predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    positives, negatives = tp + fn, fp + tn
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tp / positives,
        "fpr": fp / negatives,
        "fnr": fn / positives,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    }


def _check_eps(eps):
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")


def _check_probabilities(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or p.min(initial=0.0) < 0.0 or p.max(initial=1.0) > 1.0:
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return p


def safe_logit(p, eps=DEFAULT_EPS):
    """log(p / (1 - p)) after clipping p to [eps, 1 - eps], so 0 and 1 stay finite.

    eps is the clipping margin, default 1e-6 (largest magnitude about 13.8). It
    must be in (0, 0.5). Order is preserved except that values closer than eps to
    0 or 1 become equal. Raises ValueError for NaN or values outside [0, 1].
    """
    _check_eps(eps)
    p = np.clip(_check_probabilities(p), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def safe_log_score(p, eps=DEFAULT_EPS):
    """log(p) after clipping p to [eps, 1], so a score of 0 stays finite.

    eps is the floor, default 1e-6 (so the smallest output is about -13.8). It
    must be in (0, 0.5). Raises ValueError for NaN or values outside [0, 1].
    """
    _check_eps(eps)
    return np.log(np.clip(_check_probabilities(p), eps, 1.0))


def _cluster_indices(groups, n):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != n:
        raise ValueError(f"groups must be 1-dimensional with {n} entries, got shape {groups.shape}")
    # Clusters are ordered by first appearance, not by label, so the draws for a
    # given seed do not depend on whether IDs are numbers or strings.
    _, first_row, inverse = np.unique(groups, return_index=True, return_inverse=True)
    rank = np.empty(len(first_row), dtype=int)
    rank[np.argsort(first_row)] = np.arange(len(first_row))
    inverse = rank[inverse]
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def _bootstrap(y, columns, groups, statistic, n_boot, alpha, seed):
    """Resample whole clusters with replacement and summarise with percentiles."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    clusters = _cluster_indices(groups, len(y))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(clusters), len(clusters))
        rows = np.concatenate([clusters[k] for k in draw])
        if y[rows].min() == y[rows].max():
            continue  # a resample with one class cannot score ranking metrics
        values.append(statistic(y[rows], [column[rows] for column in columns]))
    if not values:
        raise ValueError("every bootstrap resample had a single class; use more clusters")
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": float(statistic(y, columns)),
        "low": float(low),
        "high": float(high),
        "confidence": 1.0 - alpha,
        "n_boot": n_boot,
        "n_valid": len(values),
        "n_skipped": n_boot - len(values),
        "n_clusters": len(clusters),
        "seed": seed,
    }


def cluster_bootstrap_ci(y_true, y_score, groups, metric, n_boot=1000, alpha=0.05, seed=0):
    """Percentile confidence interval for metric(y_true, y_score), resampling clusters.

    groups gives each row's cluster ID (dialogue for pairs, clip for standalone
    audio). Whole clusters are drawn with replacement, keeping their rows
    together, because rows in a cluster are not independent. Passing a unique ID
    per row gives the ordinary (naive) bootstrap, which is too narrow when rows
    cluster. metric is any function metric(y, scores) -> float, e.g. roc_auc_score,
    average_precision or exact_eer; a higher score means more likely positive
    (label 1), as those metrics assume.

    Returns a dict: estimate (on the full data), low, high, confidence
    (1 - alpha), n_boot, n_valid, n_skipped, n_clusters, seed. Resamples that
    contain a single class are skipped and counted in n_skipped. seed makes the
    interval reproducible. Raises ValueError for single-class input.
    """
    y, s = _validate(y_true, y_score)
    return _bootstrap(y, [s], groups, lambda yy, cols: metric(yy, cols[0]),
                      n_boot, alpha, seed)


def paired_cluster_bootstrap_ci(y_true, score_a, score_b, groups, metric,
                                n_boot=1000, alpha=0.05, seed=0):
    """Percentile interval for metric(y, score_a) - metric(y, score_b), resampling clusters.

    Both score columns are scored on the same resampled clusters in every
    replicate, so the interval reflects the difference itself rather than two
    independent uncertainties. A positive difference favours score_a. In both
    columns a higher score means more likely positive (label 1). Other arguments
    and the returned dict match cluster_bootstrap_ci; estimate is the difference
    on the full data. If the interval excludes 0, the difference is unlikely to
    be resampling noise. Raises ValueError for single-class input.
    """
    y, a = _validate(y_true, score_a)
    _, b = _validate(y_true, score_b)
    return _bootstrap(y, [a, b], groups,
                      lambda yy, cols: metric(yy, cols[0]) - metric(yy, cols[1]),
                      n_boot, alpha, seed)
