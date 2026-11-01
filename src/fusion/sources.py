"""Find and load the audio and text score tables for the demo notebook.

Everything is read from files under a root folder given by the caller; nothing is written.
Test rows are dropped while the file is being read, so they never reach a table unless the
caller asks for them (``include_test=True``). ``external`` rows are always dropped. If either
score file is missing, both tables are replaced by dummy tables and the bundle says why.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .dummy import make_dummy_pools
from .scores import normalize_scores

# text_model -> (folder, model name stored in the score file's ``model`` column)
TEXT_MODELS = {"minilm": ("minilm", "minilm_logreg"), "baseline": ("baseline", "tfidf_logreg")}
PREFIX_FILES = {"minilm": "validation_prefix_scores.csv"}
TEXT_SCORES_FILE = "fusion_scores_train_validation.csv"
PREFIX_TOLERANCE = 1e-6
_NUMERIC = ("label", "score", "scam_score", "fold", "turn_fraction")


@dataclass(frozen=True, eq=False)
class SourceBundle:
    """The tables and provenance the notebook needs.

    audio / text: contract-form score tables (train and validation, plus test if requested).
    prefix_tables: {turn_fraction: single-condition text table}, validation only.
    audio_origin: "real", "smoke" or "dummy"; text_origin: "real" or "dummy".
    dummy: True if either table is smoke or dummy, so the pair table must be flagged dummy.
    has_folds: both tables carry a ``fold`` column (needed for the out-of-fold check).
    test_rows_loaded: the caller asked for test rows. test_available: test rows exist in both tables.
    test_unavailable_reason: why not, when they were requested but are missing.
    prefix_mismatches: validation dialogues whose turn_fraction=1.0 score differs from the clean
    text score (None when there is nothing to compare).
    notes: human-readable lines for the banner. paths: files used, relative to the root.
    """

    audio: pd.DataFrame
    text: pd.DataFrame
    prefix_tables: dict
    audio_origin: str
    text_origin: str
    dummy: bool
    has_folds: bool
    test_rows_loaded: bool
    test_available: bool
    test_unavailable_reason: object
    prefix_mismatches: object
    notes: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)


def _read_scores(path, allowed_splits):
    """Stream a score CSV and keep only rows whose split is allowed.

    Other rows (test, external) are skipped without being parsed or stored.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            raise ValueError(f"{Path(path).name} has no 'split' column")
        rows = [row for row in reader if row["split"] in allowed_splits]
        columns = list(reader.fieldnames)
    frame = pd.DataFrame(rows, columns=columns)
    for column in _NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _audio_origin(settings_path, relative):
    if not settings_path.exists():
        return "smoke", f"{relative.rsplit('/', 1)[0]}/settings.json is missing, so the audio run is treated as a smoke run"
    try:
        mode = json.loads(settings_path.read_text(encoding="utf-8")).get("mode")
    except (OSError, ValueError):
        return "smoke", "settings.json could not be read, so the audio run is treated as a smoke run"
    return ("smoke", None) if mode == "smoke" else ("real", None)


def _prefix_tables(root, text_model, folder, model_name):
    """({fraction: table}, notes) from the validation prefix file, if there is one."""
    notes = []
    if text_model not in PREFIX_FILES:
        return {}, [f"No prefix scores exist for the {text_model} model, so there are no text_prefix conditions."]
    relative = f"outputs/text/{folder}/{PREFIX_FILES[text_model]}"
    path = Path(root) / relative
    if not path.exists():
        return {}, [f"Missing: {relative}; there are no text_prefix conditions."]
    frame = _read_scores(path, {"validation"})
    table = normalize_scores(frame, "text", model=model_name)
    tables = {}
    for fraction, part in table.groupby("turn_fraction"):
        tables[float(fraction)] = part.drop(columns="turn_fraction").reset_index(drop=True)
    return tables, notes


def _prefix_mismatches(prefix_tables, clean_text):
    """Count validation dialogues whose turn_fraction=1.0 score differs from the clean score."""
    if 1.0 not in prefix_tables:
        return None
    clean = clean_text[clean_text["split"] == "validation"].set_index("sample_id")["score"]
    full = prefix_tables[1.0].set_index("sample_id")["score"]
    matched = full.index.isin(clean.index)
    difference = np.abs(full[matched].to_numpy() - clean.loc[full.index[matched]].to_numpy())
    return int((~matched).sum() + (difference > PREFIX_TOLERANCE).sum())


def load_sources(root, *, audio_run="smoke", audio_model="wav2vec2", text_model="minilm",
                 include_test=False, seed=0):
    """Load the score tables under ``root`` (a folder that holds outputs/audio and outputs/text).

    Reads outputs/audio/<audio_run>/<audio_model>_scores.csv (smoke if its settings.json says so
    or is missing) and outputs/text/<text_model folder>/fusion_scores_train_validation.csv, plus
    the validation prefix scores when they exist. If either main file is missing, both tables are
    dummy tables (made from ``seed``) and the notes name the missing file. Test rows are loaded
    only when include_test is True. Raises ValueError for an unknown text_model or a file that
    cannot be parsed.
    """
    if text_model not in TEXT_MODELS:
        raise ValueError(f"text_model must be one of {sorted(TEXT_MODELS)}, got {text_model!r}")
    folder, model_name = TEXT_MODELS[text_model]
    allowed = {"train", "validation"} | ({"test"} if include_test else set())
    audio_relative = f"outputs/audio/{audio_run}/{audio_model}_scores.csv"
    text_relative = f"outputs/text/{folder}/{TEXT_SCORES_FILE}"
    audio_path, text_path = Path(root) / audio_relative, Path(root) / text_relative

    missing = [relative for relative, path in ((audio_relative, audio_path), (text_relative, text_path))
               if not path.exists()]
    if missing:
        audio, text = make_dummy_pools(seed=seed)
        audio, text = (table[table["split"].isin(allowed)].reset_index(drop=True) for table in (audio, text))
        notes = [f"Missing: {relative}." for relative in missing]
        notes.append("Both score tables are DUMMY tables, so every result is dummy. Degraded conditions are "
                     "made by degrade_scores (test scaffolding), not by re-scoring degraded inputs.")
        return SourceBundle(audio=audio, text=text, prefix_tables={}, audio_origin="dummy", text_origin="dummy",
                            dummy=True, has_folds=True, test_rows_loaded=include_test,
                            test_available=include_test, test_unavailable_reason=None,
                            prefix_mismatches=None, notes=notes, paths={})

    notes = []
    audio = normalize_scores(_read_scores(audio_path, allowed), "audio")
    text = normalize_scores(_read_scores(text_path, allowed), "text", model=model_name)
    audio_origin, why = _audio_origin(audio_path.parent / "settings.json", audio_relative)
    if why:
        notes.append(why.capitalize() + ".")
    if audio_origin == "smoke":
        notes.append("The audio scores come from a smoke run, so results are plumbing-only and not findings.")

    prefix_tables, prefix_notes = _prefix_tables(root, text_model, folder, model_name)
    notes += prefix_notes
    mismatches = _prefix_mismatches(prefix_tables, text)
    if prefix_tables and mismatches is None:
        notes.append("Prefix check skipped: no turn_fraction=1.0 rows to compare with the clean text scores.")
    if mismatches:
        notes.append(f"Prefix mismatch: {mismatches} validation dialogues have turn_fraction=1.0 prefix scores "
                     "that differ from the clean text scores.")

    missing_test = []
    if include_test:
        for name, table in (("audio", audio), ("text", text)):
            if not (table["split"] == "test").any():
                missing_test.append(f"the {name} score table has no test rows")
    reason = "; ".join(missing_test) if missing_test else None
    if reason:
        notes.append(f"Test pairs cannot be built: {reason}.")

    return SourceBundle(
        audio=audio, text=text, prefix_tables=prefix_tables, audio_origin=audio_origin, text_origin="real",
        dummy=audio_origin != "real", has_folds="fold" in audio.columns and "fold" in text.columns,
        test_rows_loaded=include_test, test_available=include_test and reason is None,
        test_unavailable_reason=reason, prefix_mismatches=mismatches, notes=notes,
        paths={"audio": audio_relative, "text": text_relative})
