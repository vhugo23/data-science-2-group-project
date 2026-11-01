"""Build balanced audio-text pairs inside each split.

Rules (see src/fusion/README.md): pools are split before pairing, so a pair only
joins a clip and a dialogue from the same split; each clip is used once; each
dialogue is used equally often with genuine and spoof clips, at most
``uses_per_dialogue`` times in total. Labels: audio 1 = spoof, text 1 = scam, and
target = spoof AND scam.
"""
import warnings

import numpy as np
import pandas as pd

from .checks import CELLS, LeakageError, pool_problems
from .scores import SPLITS

PAIR_COLUMNS = ["pair_id", "split", "audio_id", "text_id", "audio_label", "text_label", "target", "cell", "dummy"]


def _ids(pool, split, label):
    rows = pool[(pool["split"] == split) & (pool["label"] == label)]
    return np.sort(rows["sample_id"].to_numpy(dtype=str))


def _dialogue_slots(ids, n, per_type, rng):
    """List n dialogue uses, spread as evenly as possible and never above per_type each."""
    ids = rng.permutation(ids)
    base, extra = divmod(n, len(ids))
    counts = np.full(len(ids), base)
    counts[:extra] += 1
    assert counts.max() <= per_type
    return np.repeat(ids, counts)


def _split_pairs(split, audio, text, per_type, n_per_cell, rng):
    genuine, spoof = _ids(audio, split, 0), _ids(audio, split, 1)
    benign, scam = _ids(text, split, 0), _ids(text, split, 1)
    capacity = min(len(genuine) // 2, len(spoof) // 2, len(benign) * per_type, len(scam) * per_type)
    n = capacity if n_per_cell is None else n_per_cell
    if n > capacity:
        raise ValueError(f"n_per_cell={n} is too large for split {split!r}; at most {capacity} "
                         f"(clips: {len(genuine)} genuine, {len(spoof)} spoof; "
                         f"dialogues: {len(benign)} benign, {len(scam)} scam)")
    if n < 1:
        raise ValueError(f"split {split!r} cannot form any balanced pairs")

    clips = {0: rng.permutation(genuine)[:2 * n], 1: rng.permutation(spoof)[:2 * n]}
    # The same dialogues take the extra uses for genuine and spoof, which keeps each balanced.
    dialogues = {0: _dialogue_slots(benign, n, per_type, rng), 1: _dialogue_slots(scam, n, per_type, rng)}
    rows = []
    for (audio_label, text_label), cell in CELLS.items():
        part = clips[audio_label][text_label * n:(text_label + 1) * n]
        for i, (clip, dialogue) in enumerate(zip(part, dialogues[text_label])):
            rows.append((f"{split}-{cell}-{i:04d}", split, clip, dialogue,
                         audio_label, text_label, audio_label & text_label, cell))
    return rows


def build_pairs(audio_pool, text_pool, *, dummy, uses_per_dialogue=2, n_per_cell=None, seed=0):
    """Return the pair manifest.

    uses_per_dialogue is the most times one dialogue appears. It must be even, because
    half the uses pair the dialogue with genuine clips and half with spoof clips:
    2 (default) is one of each; 4 is for sensitivity checks.
    n_per_cell defaults to the most pairs the scarcest ingredient allows.
    dummy is required: True when the scores are fake or from a smoke run, else False.
    """
    if uses_per_dialogue < 2 or uses_per_dialogue % 2:
        raise ValueError(f"uses_per_dialogue must be an even number of at least 2, got {uses_per_dialogue}")
    per_type = uses_per_dialogue // 2
    problems = pool_problems(audio_pool, "audio") + pool_problems(text_pool, "text")
    if problems:
        raise LeakageError(problems)

    in_audio, in_text = set(audio_pool["split"]), set(text_pool["split"])
    splits = [split for split in SPLITS if split in in_audio and split in in_text]
    for split in SPLITS:
        if (split in in_audio) != (split in in_text):
            absent = "text" if split in in_audio else "audio"
            warnings.warn(f"split {split!r} skipped: no rows in the {absent} pool")
    if not splits:
        raise ValueError("audio and text pools share no split")

    rng = np.random.default_rng(seed)
    rows = []
    for split in splits:
        rows += _split_pairs(split, audio_pool, text_pool, per_type, n_per_cell, rng)
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS[:-1])
    pairs["dummy"] = bool(dummy)
    return pairs


def source_counts(pairs):
    """Per split: pairs, unique clips and dialogues, and the largest reuse of each."""
    rows = []
    for split, part in pairs.groupby("split", sort=False):
        rows.append({"split": split, "pairs": len(part),
                     "unique_audio": part["audio_id"].nunique(), "unique_text": part["text_id"].nunique(),
                     "max_audio_reuse": part.groupby("audio_id").size().max(),
                     "max_text_reuse": part.groupby("text_id").size().max()})
    return pd.DataFrame(rows)
