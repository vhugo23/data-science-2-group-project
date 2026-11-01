"""Tests for building audio-text pairs. Dummy score tables only; no real data."""
import unittest
import warnings

import pandas as pd

from src.fusion.dummy import make_dummy_pools
from src.fusion.pairing import build_pairs, source_counts

CELLS = ["genuine_benign", "genuine_scam", "spoof_benign", "spoof_scam"]


def pools(**kwargs):
    return make_dummy_pools(seed=1, **kwargs)


class BuildPairsTests(unittest.TestCase):
    def setUp(self):
        self.audio, self.text = pools()
        self.pairs = build_pairs(self.audio, self.text, seed=7, dummy=True)

    def test_no_source_crosses_splits(self):
        for column in ["audio_id", "text_id"]:
            self.assertTrue((self.pairs.groupby(column)["split"].nunique() == 1).all())

    def test_pair_split_matches_both_source_splits(self):
        audio_split = self.pairs["audio_id"].map(self.audio.set_index("sample_id")["split"])
        text_split = self.pairs["text_id"].map(self.text.set_index("sample_id")["split"])
        self.assertTrue((audio_split == self.pairs["split"]).all())
        self.assertTrue((text_split == self.pairs["split"]).all())

    def test_cells_are_equal_within_each_split(self):
        counts = self.pairs.groupby(["split", "cell"]).size().unstack()
        self.assertEqual(sorted(counts.columns), CELLS)
        for _, row in counts.iterrows():
            self.assertEqual(row.nunique(), 1)

    def test_target_is_spoof_and_scam(self):
        expected = (self.pairs["audio_label"] == 1) & (self.pairs["text_label"] == 1)
        self.assertTrue((self.pairs["target"] == expected.astype(int)).all())
        self.assertEqual(set(self.pairs["target"]), {0, 1})

    def test_each_clip_used_at_most_once(self):
        self.assertTrue(self.pairs["audio_id"].is_unique)

    def test_dialogue_uses_capped_and_equal_genuine_spoof(self):
        uses = self.pairs.groupby("text_id").size()
        self.assertLessEqual(uses.max(), 2)
        by_type = self.pairs.groupby(["text_id", "audio_label"]).size().unstack(fill_value=0)
        self.assertTrue((by_type[0] == by_type[1]).all())

    def test_four_uses_for_sensitivity_checks_stays_balanced(self):
        audio, text = pools(n_audio={"train": 800, "validation": 200, "test": 200},
                            n_text={"train": 40, "validation": 20, "test": 20})
        pairs = build_pairs(audio, text, uses_per_dialogue=6, seed=3, dummy=True)
        uses = pairs.groupby("text_id").size()
        self.assertGreater(uses.max(), 2)
        self.assertLessEqual(uses.max(), 6)
        by_type = pairs.groupby(["text_id", "audio_label"]).size().unstack(fill_value=0)
        self.assertTrue((by_type[0] == by_type[1]).all())

    def test_remainder_uses_keep_dialogues_balanced(self):
        # 10 pairs per cell over 7 benign / 7 scam dialogues: 3 dialogues get an extra use.
        audio, text = pools(n_audio={"train": 40, "validation": 40, "test": 40},
                            n_text={"train": 14, "validation": 14, "test": 14})
        pairs = build_pairs(audio, text, uses_per_dialogue=4, seed=2, dummy=True)
        self.assertEqual(pairs.groupby("text_id").size().nunique(), 2)
        by_type = pairs.groupby(["text_id", "audio_label"]).size().unstack(fill_value=0)
        self.assertTrue((by_type[0] == by_type[1]).all())

    def test_uses_per_dialogue_must_be_even_and_positive(self):
        for bad in [0, 1, 3, 5, -2]:
            with self.assertRaisesRegex(ValueError, "uses_per_dialogue"):
                build_pairs(self.audio, self.text, uses_per_dialogue=bad, dummy=True)

    def test_default_is_two_uses_one_genuine_one_spoof(self):
        uses = self.pairs.groupby("text_id").size()
        self.assertEqual(uses.max(), 2)
        by_type = self.pairs.groupby(["text_id", "audio_label"]).size().unstack(fill_value=0)
        self.assertTrue((by_type[0] == 1).all() and (by_type[1] == 1).all())

    def test_dummy_must_be_stated(self):
        with self.assertRaises(TypeError):
            build_pairs(self.audio, self.text)

    def test_same_seed_same_pairs_different_seed_differs(self):
        again = build_pairs(self.audio, self.text, seed=7, dummy=True)
        pd.testing.assert_frame_equal(self.pairs, again)
        other = build_pairs(self.audio, self.text, seed=8, dummy=True)
        self.assertFalse(self.pairs[["audio_id", "text_id"]].equals(other[["audio_id", "text_id"]]))

    def test_default_pair_count_is_scarcest_ingredient(self):
        # Dummy default: validation has 60 genuine + 60 spoof clips -> at most 30 per cell.
        audio, text = pools(n_audio={"train": 200, "validation": 120, "test": 120},
                            n_text={"train": 400, "validation": 400, "test": 400})
        pairs = build_pairs(audio, text, seed=0, dummy=True)
        self.assertEqual(pairs[pairs["split"] == "validation"].groupby("cell").size().iloc[0], 60 // 2)

    def test_requesting_too_many_pairs_raises(self):
        with self.assertRaisesRegex(ValueError, "n_per_cell"):
            build_pairs(self.audio, self.text, n_per_cell=10_000, dummy=True)

    def test_fewer_pairs_can_be_requested(self):
        pairs = build_pairs(self.audio, self.text, n_per_cell=5, seed=0, dummy=True)
        self.assertTrue((pairs.groupby(["split", "cell"]).size() == 5).all())

    def test_source_in_two_splits_is_rejected(self):
        audio = self.audio.copy()
        clone = audio.iloc[[0]].copy()
        clone["split"] = "test" if clone["split"].iloc[0] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "more than one split"):
            build_pairs(pd.concat([audio, clone]), self.text, dummy=True)

    def test_duplicate_id_in_a_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_pairs(pd.concat([self.audio, self.audio.iloc[[0]]]), self.text, dummy=True)

    def test_split_missing_from_text_is_skipped_with_warning(self):
        text = self.text[self.text["split"] != "test"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pairs = build_pairs(self.audio, text, seed=0, dummy=True)
        self.assertEqual(set(pairs["split"]), {"train", "validation"})
        self.assertTrue(any("test" in str(w.message) for w in caught))

    def test_external_split_is_never_paired(self):
        audio = self.audio.copy()
        audio.loc[audio.index[:10], "split"] = "external"
        pairs = build_pairs(audio, self.text, seed=0, dummy=True)
        self.assertNotIn("external", set(pairs["split"]))

    def test_dummy_flag_is_carried_exactly_as_given(self):
        self.assertTrue(self.pairs["dummy"].all())
        real = build_pairs(self.audio, self.text, seed=0, dummy=False)
        self.assertFalse(real["dummy"].any())

    def test_source_counts_report_unique_sources_and_max_reuse(self):
        summary = source_counts(self.pairs).set_index("split")
        for split, row in summary.iterrows():
            part = self.pairs[self.pairs["split"] == split]
            self.assertEqual(row["pairs"], len(part))
            self.assertEqual(row["unique_audio"], part["audio_id"].nunique())
            self.assertEqual(row["unique_text"], part["text_id"].nunique())
            self.assertEqual(row["max_text_reuse"], part.groupby("text_id").size().max())
            self.assertEqual(row["max_audio_reuse"], 1)


if __name__ == "__main__":
    unittest.main()
