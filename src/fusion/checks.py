"""Independent leakage and consistency checks for a pair table.

The checks read only the pair table and the two source pools, so they also
validate a manifest that was loaded from a file or edited by hand.
"""
import pandas as pd

from .scores import SPLITS

PAIR_COLUMNS = ["pair_id", "split", "audio_id", "text_id", "audio_label", "text_label", "target", "cell"]
POOL_COLUMNS = ["sample_id", "split", "label"]
CELLS = {(0, 0): "genuine_benign", (0, 1): "genuine_scam", (1, 0): "spoof_benign", (1, 1): "spoof_scam"}


class LeakageError(ValueError):
    """Raised with every problem found, not just the first."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _few(values, limit=5):
    values = list(values)
    return f"{values[:limit]}" + (f" (+{len(values) - limit} more)" if len(values) > limit else "")


def pool_problems(pool, name):
    """Problems with a source pool: duplicates, ids in several splits, bad names or labels."""
    missing = [column for column in POOL_COLUMNS if column not in pool.columns]
    if missing:
        return [f"{name} pool is missing columns: {missing}"]
    problems = []
    if pool["sample_id"].duplicated().any():
        problems.append(f"{name} pool has duplicate sample_id values: {_few(pool.loc[pool['sample_id'].duplicated(), 'sample_id'])}")
    crossing = pool.groupby("sample_id")["split"].nunique()
    if (crossing > 1).any():
        problems.append(f"{name} pool: sample_id appears in more than one split: {_few(crossing[crossing > 1].index)}")
    unknown = sorted(set(pool["split"]) - set(SPLITS) - {"external"})
    if unknown:
        problems.append(f"{name} pool has unknown split names: {unknown}")
    if not pool["label"].isin([0, 1]).all():
        problems.append(f"{name} pool has labels other than 0 and 1")
    return problems


def check_pairs(pairs, audio_pool, text_pool, uses_per_dialogue=2):
    """Raise LeakageError listing every violated rule; return None when all hold."""
    missing = [column for column in PAIR_COLUMNS if column not in pairs.columns]
    if missing:
        raise LeakageError([f"pair table is missing columns: {missing}"])

    problems = pool_problems(audio_pool, "audio") + pool_problems(text_pool, "text")
    if any("missing columns" in problem for problem in problems):
        raise LeakageError(problems)

    if pairs["pair_id"].duplicated().any():
        problems.append("pair_id values are not unique")
    bad_split = sorted(set(pairs["split"]) - set(SPLITS))
    if bad_split:
        problems.append(f"split values not allowed in pairs: {bad_split}")

    audio_known = audio_pool.drop_duplicates("sample_id").set_index("sample_id")
    text_known = text_pool.drop_duplicates("sample_id").set_index("sample_id")
    for column, known, name in [("audio_id", audio_known, "audio"), ("text_id", text_known, "text")]:
        unknown = pairs.loc[~pairs[column].isin(known.index), column]
        if len(unknown):
            problems.append(f"{column} values not in the {name} pool: {_few(unknown)}")
        pool_split = pairs[column].map(known["split"])
        wrong = (pool_split != pairs["split"]) & pool_split.notna()
        if wrong.any():
            problems.append(f"pair split differs from the {name} source's split for {wrong.sum()} pairs")
        pool_label = pairs[column].map(known["label"])
        label_column = column.replace("_id", "_label")
        wrong = (pool_label != pairs[label_column]) & pool_label.notna()
        if wrong.any():
            problems.append(f"{label_column} disagrees with the {name} pool for {wrong.sum()} pairs")
        crossing = pairs.groupby(column)["split"].nunique()
        if (crossing > 1).any():
            problems.append(f"{column} appears in more than one split: {_few(crossing[crossing > 1].index)}")

    reused = pairs.loc[pairs["audio_id"].duplicated(), "audio_id"]
    if len(reused):
        problems.append(f"audio clips used more than once: {_few(reused.unique())}")

    uses = pairs.groupby("text_id").size()
    if (uses > uses_per_dialogue).any():
        problems.append(f"dialogues used more than {uses_per_dialogue} times: {_few(uses[uses > uses_per_dialogue].index)}")
    by_type = pairs.groupby(["text_id", "audio_label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    unequal = by_type[by_type[0] != by_type[1]]
    if len(unequal):
        problems.append(f"dialogues not paired equally with genuine and spoof clips: {_few(unequal.index)}")

    expected = (pairs["audio_label"] == 1) & (pairs["text_label"] == 1)
    if (pairs["target"] != expected.astype(int)).any():
        problems.append("target is not (audio_label AND text_label) for some pairs")
    expected_cell = [CELLS.get((a, t)) for a, t in zip(pairs["audio_label"], pairs["text_label"])]
    if (pairs["cell"] != pd.Series(expected_cell, index=pairs.index)).any():
        problems.append("cell does not match the pair's labels for some pairs")

    for split, part in pairs.groupby("split"):
        sizes = part.groupby(["audio_label", "text_label"]).size().reindex(list(CELLS), fill_value=0)
        if sizes.nunique() != 1 or sizes.iloc[0] == 0:
            problems.append(f"cells are not balanced in split {split!r}: {dict(zip(CELLS.values(), sizes))}")

    if problems:
        raise LeakageError(problems)
