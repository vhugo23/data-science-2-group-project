"""Spoof classifier, validation threshold selection, and score metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def new_classifier(seed, C=1.0):
    """Scaled, class-balanced logistic regression; scaling is refit inside each CV fold."""
    return make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed))


def select_threshold(labels, scores):
    """Threshold maximizing F1; call it on validation labels only, never test."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def equal_error_rate(labels, scores):
    """Discrete ROC approximation of the EER; a summary metric, not an operating threshold."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2)


def evaluate_scores(labels, scores, threshold):
    """ROC AUC, EER, and F1 at a fixed threshold (label 1 = spoof)."""
    scores = np.asarray(scores)
    return {"roc_auc": roc_auc_score(labels, scores),
            "eer": equal_error_rate(labels, scores),
            "f1": f1_score(labels, scores >= threshold, zero_division=0),
            "threshold": threshold, "n": len(labels)}
