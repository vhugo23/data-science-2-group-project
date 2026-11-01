"""Tests for the adapter that normalizes teammates' score tables to the contract."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.scores import CONTRACT_COLUMNS, load_scores, normalize_scores


def audio_table():
    return pd.DataFrame({"sample_id": ["a1", "a2", "a3"], "split": ["train", "validation", "test"],
                         "label": [0, 1, 1], "score": [0.2, 0.7, 0.9]})


def text_table():
    return pd.DataFrame({"dialogue_id": ["d1", "d2", "d3"], "split": ["train", "train", "validation"],
                         "label": [1, 0, 1], "model": "minilm_logreg",
                         "scam_score": [0.9, 0.1, 0.8], "fold": [0, 1, -1]})


class NormalizeScoresTests(unittest.TestCase):
    def test_audio_table_already_matches_contract(self):
        table = normalize_scores(audio_table(), "audio")
        self.assertEqual(list(table.columns), CONTRACT_COLUMNS)
        self.assertEqual(table["score"].tolist(), [0.2, 0.7, 0.9])

    def test_text_columns_are_renamed_and_extras_kept(self):
        table = normalize_scores(text_table(), "text")
        self.assertEqual(list(table.columns[:4]), CONTRACT_COLUMNS)
        self.assertEqual(table["sample_id"].tolist(), ["d1", "d2", "d3"])
        self.assertEqual(table["score"].tolist(), [0.9, 0.1, 0.8])
        self.assertIn("fold", table.columns)
        self.assertNotIn("model", table.columns)

    def test_input_is_not_modified(self):
        original = text_table()
        before = original.copy()
        normalize_scores(original, "text")
        pd.testing.assert_frame_equal(original, before)

    def test_multiple_models_need_an_explicit_choice(self):
        both = pd.concat([text_table(), text_table().assign(model="tfidf_logreg", scam_score=0.5)])
        with self.assertRaisesRegex(ValueError, "model"):
            normalize_scores(both, "text")
        chosen = normalize_scores(both, "text", model="tfidf_logreg")
        self.assertEqual(len(chosen), 3)
        self.assertTrue((chosen["score"] == 0.5).all())

    def test_unknown_model_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model"):
            normalize_scores(text_table(), "text", model="nope")

    def test_external_split_rejected_unless_allowed(self):
        table = audio_table().assign(split=["external", "external", "external"])
        with self.assertRaisesRegex(ValueError, "split"):
            normalize_scores(table, "audio")
        allowed = normalize_scores(table, "audio", allow_external=True)
        self.assertEqual(set(allowed["split"]), {"external"})

    def test_unknown_split_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "split"):
            normalize_scores(audio_table().assign(split="dev"), "audio")

    def test_bad_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "label"):
            normalize_scores(audio_table().assign(label=[0, 1, 2]), "audio")

    def test_bad_scores_rejected(self):
        for bad in [np.nan, np.inf]:
            with self.assertRaisesRegex(ValueError, "score"):
                normalize_scores(audio_table().assign(score=[0.2, bad, 0.9]), "audio")

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_scores(pd.concat([audio_table(), audio_table().iloc[[0]]]), "audio")

    def test_same_id_in_two_splits_rejected(self):
        table = pd.concat([audio_table(), audio_table().iloc[[0]].assign(split="test")])
        with self.assertRaisesRegex(ValueError, "more than one split"):
            normalize_scores(table, "audio")

    def test_prefix_table_allows_repeats_across_turn_fractions(self):
        base = text_table().iloc[:2]
        prefix = pd.concat([base.assign(turn_fraction=0.25), base.assign(turn_fraction=1.0)])
        table = normalize_scores(prefix, "text")
        self.assertEqual(len(table), 4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_scores(pd.concat([prefix, prefix.iloc[[0]]]), "text")

    def test_missing_columns_named_in_error(self):
        with self.assertRaisesRegex(ValueError, "score"):
            normalize_scores(audio_table().drop(columns=["score"]), "audio")

    def test_unknown_kind_rejected(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            normalize_scores(audio_table(), "video")

    def test_load_scores_reads_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text.csv"
            text_table().to_csv(path, index=False)
            table = load_scores(path, "text")
        self.assertEqual(table["sample_id"].tolist(), ["d1", "d2", "d3"])

    def test_ids_are_strings_and_labels_ints(self):
        table = normalize_scores(audio_table().assign(sample_id=[1, 2, 3]), "audio")
        self.assertTrue(all(isinstance(value, str) for value in table["sample_id"]))
        self.assertEqual(table["label"].dtype.kind, "i")


if __name__ == "__main__":
    unittest.main()
