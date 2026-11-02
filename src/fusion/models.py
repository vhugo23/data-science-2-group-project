"""Fusion models: fit on train, choose C and eps on validation, freeze, reload.

Systems, all scored on the pair target (spoof AND scam):
  audio_only  the raw audio score
  text_only   the raw text score
  product     log(audio) + log(text), equal weights, using the default eps
  learned     logistic regression on [log(audio), log(text)], using the tuned eps

A higher score means more likely positive (label 1) everywhere. Nothing here reads or
tunes on the test split.
"""
import dataclasses
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .checks import check_pairs
from .scores import CONTRACT_COLUMNS
from src.metrics import DEFAULT_EPS, safe_log_score

SYSTEMS = ("audio_only", "text_only", "product", "learned")
SCHEMA = 1
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_EPS_GRID = (1e-6, 1e-4, 1e-2)
DEFAULT_THRESHOLD_FPR = 0.05


class NegativeWeightWarning(UserWarning):
    """A learned fusion weight is negative: that branch is being used backwards."""


def _few(values, limit=5):
    values = list(values)
    return f"{values[:limit]}" + (f" (+{len(values) - limit} more)" if len(values) > limit else "")


# ---------------------------------------------------------------------------
# Tables: dummy flag, joining, hashing
# ---------------------------------------------------------------------------

def declared_dummy(pairs):
    """The pair table's dummy flag (one value for the whole table).

    Raises ValueError for mixed flags, or for ids that start with ``dummy_`` when the
    flag says the run is real.
    """
    flags = set(pairs["dummy"].astype(bool))
    if len(flags) != 1:
        raise ValueError("the pair table has mixed dummy flags; it must be all True or all False")
    flag = flags.pop()
    if not flag:
        ids = pd.concat([pairs["audio_id"], pairs["text_id"]]).astype(str)
        if ids.str.startswith("dummy_").any():
            raise ValueError("pair ids start with 'dummy_' but the dummy flag is False")
    return flag


def join_scores(pairs, audio_scores, text_scores, split):
    """Pairs of one split with ``audio_score`` and ``text_score`` columns added.

    Raises ValueError if the split has no pairs, if any pair id is missing from a score
    table (the ids are named), and LeakageError if the pairs and score tables disagree
    about splits, labels or reuse. Each score table must hold one condition: repeated
    sample ids are rejected. Pairs of other splits need no scores.
    """
    part = pairs[pairs["split"] == split]
    if part.empty:
        raise ValueError(f"no pairs in split {split!r}")
    for column, table, name in (("audio_id", audio_scores, "audio"), ("text_id", text_scores, "text")):
        missing = sorted(set(part[column]) - set(table["sample_id"]))
        if missing:
            raise ValueError(f"pair ids missing from the {name} score table: {_few(missing)}")
    uses = max(2, int(part.groupby("text_id").size().max()))
    check_pairs(part, audio_scores, text_scores, uses_per_dialogue=uses)
    audio = audio_scores.set_index("sample_id")["score"]
    text = text_scores.set_index("sample_id")["score"]
    joined = part.copy()
    joined["audio_score"] = joined["audio_id"].map(audio).astype(float)
    joined["text_score"] = joined["text_id"].map(text).astype(float)
    return joined.reset_index(drop=True)


def _digest(table, columns):
    table = table[columns].sort_values(columns, kind="stable")
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def input_hashes(pairs, audio_scores, text_scores, splits=("train", "validation")):
    """SHA-256 of the pair rows in ``splits`` and of the score rows those pairs use.

    Only the contract columns of the score tables are hashed, rows are sorted first,
    and rows of other splits or rows no pair uses do not matter.
    """
    part = pairs[pairs["split"].isin(splits)]
    audio = audio_scores[audio_scores["sample_id"].isin(part["audio_id"])]
    text = text_scores[text_scores["sample_id"].isin(part["text_id"])]
    return {"pairs": _digest(part, list(part.columns)),
            "audio": _digest(audio, CONTRACT_COLUMNS),
            "text": _digest(text, CONTRACT_COLUMNS)}


def evaluation_hash(pairs, audio_scores, text_scores, split):
    """One SHA-256 over the audio and text score rows used by one split's pairs."""
    hashes = input_hashes(pairs, audio_scores, text_scores, splits=(split,))
    return hashlib.sha256((hashes["audio"] + hashes["text"]).encode("ascii")).hexdigest()


def check_inputs_match(model, pairs, audio_scores, text_scores):
    """Raise ValueError naming every table whose train+validation rows differ from the fit."""
    now = input_hashes(pairs, audio_scores, text_scores)
    changed = [name for name in ("pairs", "audio", "text") if now[name] != model.hashes[name]]
    if changed:
        raise ValueError(f"inputs differ from the ones the model was fit on: {changed}")


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def choose_threshold(y_true, y_score, max_fpr):
    """Smallest threshold whose false positive rate is <= max_fpr (so recall is highest).

    Intended for validation data only. A higher score means more likely positive; a
    sample is accepted when score >= threshold, and tied scores are accepted together.
    The candidates are the observed scores. If even the highest score accepts too many
    negatives, the threshold is the next float above it, which accepts nothing.
    Raises ValueError if only one class is present or max_fpr is outside [0, 1].
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")
    y, s = np.asarray(y_true), np.asarray(y_score, dtype=float)
    if len(y) == 0 or y.min() == y.max():
        raise ValueError("need both classes present to choose a threshold")
    negatives = np.sort(s[y == 0])
    candidates = np.unique(s)
    fpr = 1.0 - np.searchsorted(negatives, candidates, side="left") / len(negatives)
    allowed = np.flatnonzero(fpr <= max_fpr + 1e-12)
    if len(allowed):
        return float(candidates[allowed[0]])
    return float(np.nextafter(candidates[-1], np.inf))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionModel:
    """A frozen fusion model: every number needed to score pairs, plus how it was made.

    product uses the default eps (metrics.DEFAULT_EPS) and equal weights; it is never
    tuned. learned uses the tuned eps stored in ``eps`` and the fitted ``weights``.
    ``thresholds`` holds one operating threshold per system, chosen on validation at
    ``threshold_fpr``. ``weight_ratio`` is w_audio / w_text of the learned rule.
    A higher score means more likely positive (label 1) for every system.
    oof_verified is True only if the train scores were checked to be out-of-fold;
    plumbing_only is its opposite and marks results that must not be reported as findings.
    """

    weights: dict
    intercept: float
    eps: float
    C: float
    product: dict
    thresholds: dict
    threshold_fpr: float
    hashes: dict
    tuning: list
    fit_split: str
    tune_split: str
    n_fit_pairs: int
    n_tune_pairs: int
    oof_verified: bool
    plumbing_only: bool
    dummy: bool
    schema: int = field(default=SCHEMA)

    @property
    def weight_ratio(self):
        """w_audio / w_text of the learned rule (the product rule is 1.0)."""
        return self.weights["audio"] / self.weights["text"]

    def system_scores(self, audio_p, text_p):
        """Score arrays for the four systems, from audio and text probabilities in [0, 1]."""
        a, t = np.asarray(audio_p, dtype=float), np.asarray(text_p, dtype=float)
        if a.ndim != 1 or a.shape != t.shape:
            raise ValueError(f"audio and text scores must be 1-dimensional and equal length, got {a.shape} and {t.shape}")
        product = self.product
        learned = (self.intercept
                   + self.weights["audio"] * safe_log_score(a, self.eps)
                   + self.weights["text"] * safe_log_score(t, self.eps))
        return {
            "audio_only": a.copy(),
            "text_only": t.copy(),
            "product": product["w_audio"] * safe_log_score(a, product["eps"])
                       + product["w_text"] * safe_log_score(t, product["eps"]),
            "learned": learned,
        }

    def to_dict(self):
        data = dataclasses.asdict(self)
        data["weight_ratio"] = self.weight_ratio
        return data

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported model schema {data.get('schema')!r}; expected {SCHEMA}")
        names = [f.name for f in dataclasses.fields(cls)]
        missing = [name for name in names if name not in data]
        if missing:
            raise ValueError(f"model file is missing fields: {missing}")
        return cls(**{name: data[name] for name in names})


def save_model(model, path):
    """Write the model as JSON. Refuses to overwrite: a frozen model file is never edited."""
    with open(Path(path), "x", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2)


def load_model(path):
    with open(Path(path), encoding="utf-8") as handle:
        return FusionModel.from_dict(json.load(handle))


def _check_out_of_fold(table, name, used_ids):
    if "fold" not in table.columns:
        raise ValueError(
            f"the {name} score table has no 'fold' column, so its train scores cannot be shown to be "
            "out-of-fold; supply it, or pass require_oof=False for a plumbing-only run")
    rows = table[(table["split"] == "train") & table["sample_id"].isin(used_ids)]
    fold = pd.to_numeric(rows["fold"], errors="coerce")
    if (fold.isna() | (fold < 0)).any():
        raise ValueError(f"{name} train rows used for fitting are not out-of-fold: 'fold' must be 0 or more "
                         "on every train row (a missing value or -1 means in-sample)")


def _grid(values, name, valid):
    values = sorted({float(v) for v in values})
    if not values:
        raise ValueError(f"{name} must not be empty")
    bad = [v for v in values if not valid(v)]
    if bad:
        raise ValueError(f"invalid values in {name}: {bad}")
    return values


def _features(joined, eps):
    return np.c_[safe_log_score(joined["audio_score"], eps), safe_log_score(joined["text_score"], eps)]


def _require_both_classes(y, name):
    if y.min() == y.max():
        raise ValueError(f"the {name} pairs need both classes of the target (spoof AND scam)")


def fit_fusion(pairs, audio_scores, text_scores, *, c_grid=DEFAULT_C_GRID, eps_grid=DEFAULT_EPS_GRID,
               threshold_fpr=DEFAULT_THRESHOLD_FPR, fit_split="train", tune_split="validation",
               require_oof=True):
    """Fit the learned rule on train pairs, choose C and eps on validation, return a FusionModel.

    Fitting uses the train split only and tuning the validation split only; any other
    fit_split or tune_split raises ValueError, and test or external rows are never read.
    For every (C, eps) in the grids a logistic regression on [log(audio), log(text)] is fit
    on train and scored by ROC-AUC on validation. The best wins; exact ties go to the smaller
    C, then the smaller eps, so grid order does not matter. The chosen model is not refit on
    validation. The product rule is fixed: equal weights and the default eps, never tuned;
    the learned rule uses the tuned eps. One threshold per system is then chosen on
    validation at threshold_fpr (default 5%) and stored. A higher score means more likely
    positive.

    Train scores must be out-of-fold: with require_oof=True (default) both score tables need a
    'fold' column that is 0 or more on every train row used, else ValueError. require_oof=False
    skips the check, records oof_verified=False and marks the model plumbing_only. A warning
    (NegativeWeightWarning) is issued if a learned weight is negative.
    """
    if fit_split != "train":
        raise ValueError(f"models are fit on the train split only, got fit_split={fit_split!r}")
    if tune_split != "validation":
        raise ValueError(f"C and eps are tuned on the validation split only, got tune_split={tune_split!r}")
    dummy = declared_dummy(pairs)
    train = join_scores(pairs, audio_scores, text_scores, "train")
    validation = join_scores(pairs, audio_scores, text_scores, "validation")
    y_train, y_validation = train["target"].to_numpy(), validation["target"].to_numpy()
    _require_both_classes(y_train, "train")
    _require_both_classes(y_validation, "validation")

    if require_oof:
        _check_out_of_fold(audio_scores, "audio", train["audio_id"])
        _check_out_of_fold(text_scores, "text", train["text_id"])

    cs = _grid(c_grid, "c_grid", lambda v: np.isfinite(v) and v > 0)
    epsilons = _grid(eps_grid, "eps_grid", lambda v: 0 < v < 0.5)
    tuning, best, best_fit = [], None, None
    for c in cs:
        for eps in epsilons:
            fitted = LogisticRegression(C=c, max_iter=1000).fit(_features(train, eps), y_train)
            auc = float(roc_auc_score(y_validation, fitted.decision_function(_features(validation, eps))))
            tuning.append({"C": c, "eps": eps, "roc_auc": auc})
            if best is None or auc > best["roc_auc"]:  # strict: earlier (smaller C, eps) wins ties
                best, best_fit = tuning[-1], fitted

    weights = {"audio": float(best_fit.coef_[0][0]), "text": float(best_fit.coef_[0][1])}
    for branch, weight in weights.items():
        if weight < 0:
            warnings.warn(f"learned weight is negative for {branch} ({weight:.3f}): that branch is "
                          "being used backwards, so check its score direction", NegativeWeightWarning, stacklevel=2)

    model = FusionModel(
        weights=weights, intercept=float(best_fit.intercept_[0]), eps=best["eps"], C=best["C"],
        product={"w_audio": 1.0, "w_text": 1.0, "eps": DEFAULT_EPS},
        thresholds={}, threshold_fpr=float(threshold_fpr),
        hashes=input_hashes(pairs, audio_scores, text_scores),
        tuning=tuning, fit_split=fit_split, tune_split=tune_split,
        n_fit_pairs=len(train), n_tune_pairs=len(validation),
        oof_verified=bool(require_oof), plumbing_only=not require_oof, dummy=dummy)
    scores = model.system_scores(validation["audio_score"].to_numpy(), validation["text_score"].to_numpy())
    thresholds = {system: choose_threshold(y_validation, scores[system], threshold_fpr) for system in SYSTEMS}
    return dataclasses.replace(model, thresholds=thresholds)
