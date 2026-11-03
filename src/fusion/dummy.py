"""Fake score tables for developing fusion code before real scores exist.

Every id starts with ``dummy_`` so nothing built from these can be mistaken for
a result. Nothing here writes files.
"""
import numpy as np
import pandas as pd

from .scores import SPLITS

DEFAULT_AUDIO = {"train": 400, "validation": 120, "test": 120}
DEFAULT_TEXT = {"train": 300, "validation": 100, "test": 100}


def _table(prefix, counts, positive_fraction, strength, rng):
    frames = []
    for split in SPLITS:
        n = counts[split]
        positives = int(round(n * positive_fraction))
        label = rng.permutation(np.r_[np.ones(positives, int), np.zeros(n - positives, int)])
        # Scores are sigmoids of a noisy logit; strength is the class gap in logit units.
        logit = rng.normal((label - 0.5) * strength, 1.0)
        frames.append(pd.DataFrame({
            "sample_id": [f"dummy_{prefix}_{split}_{i:04d}" for i in range(n)],
            "split": split, "label": label, "score": 1 / (1 + np.exp(-logit)),
            # Out-of-fold fold index: 0-4 on train, -1 elsewhere (no random draws, so
            # adding it does not change the scores the other tests rely on).
            "fold": np.arange(n) % 5 if split == "train" else -1,
        }))
    return pd.concat(frames, ignore_index=True)


def make_dummy_pools(n_audio=None, n_text=None, spoof_fraction=0.5, scam_fraction=0.5,
                     audio_strength=2.0, text_strength=4.0, seed=0):
    """Return (audio_table, text_table) in contract form (label 1 = spoof / scam).

    Both tables carry a ``fold`` column: 0-4 on train rows (as if out-of-fold) and -1 elsewhere.
    """
    rng = np.random.default_rng(seed)
    audio = _table("audio", n_audio or DEFAULT_AUDIO, spoof_fraction, audio_strength, rng)
    text = _table("text", n_text or DEFAULT_TEXT, scam_fraction, text_strength, rng)
    return audio, text


def degrade_scores(table, shrink, noise=0.0, seed=0):
    """Return a copy of a score table whose scores are weaker: logit * shrink + N(0, noise).

    Test scaffolding only. It lets dummy runs exercise the noise, in-the-wild and
    text-prefix code paths. Real degradation means re-scoring degraded inputs (noisy
    audio, truncated dialogues) through the actual models, never rescaling scores.
    Ids, splits, labels and folds are kept.
    """
    if shrink < 0 or noise < 0:
        raise ValueError(f"shrink and noise must be non-negative, got shrink={shrink}, noise={noise}")
    rng = np.random.default_rng(seed)
    p = np.clip(table["score"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p)) * shrink + rng.normal(0.0, noise, len(p))
    out = table.copy()
    out["score"] = 1 / (1 + np.exp(-logit))
    return out
