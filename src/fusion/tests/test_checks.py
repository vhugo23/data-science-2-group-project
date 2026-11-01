"""Each leakage rule is tested by deliberately breaking a good pair table."""
import unittest

import pandas as pd

from src.fusion.checks import LeakageError, check_pairs
from src.fusion.dummy import make_dummy_pools
from src.fusion.pairing import build_pairs


class CheckPairsTests(unittest.TestCase):
    def setUp(self):
        self.audio, self.text = make_dummy_pools(seed=1)
        self.pairs = build_pairs(self.audio, self.text, seed=7)

    def assert_fails(self, pairs, pattern, audio=None, text=None, **kwargs):
        with self.assertRaisesRegex(LeakageError, pattern):
            check_pairs(pairs, self.audio if audio is None else audio,
                        self.text if text is None else text, **kwargs)

    def test_good_pairs_pass(self):
        check_pairs(self.pairs, self.audio, self.text)

    def test_missing_column(self):
        self.assert_fails(self.pairs.drop(columns=["target"]), "missing columns")

    def test_duplicate_pair_id(self):
        bad = pd.concat([self.pairs, self.pairs.iloc[[0]]], ignore_index=True)
        self.assert_fails(bad, "pair_id")

    def test_unknown_audio_id(self):
        bad = self.pairs.copy()
        bad.loc[0, "audio_id"] = "not_a_clip"
        self.assert_fails(bad, "not in the audio pool")

    def test_unknown_text_id(self):
        bad = self.pairs.copy()
        bad.loc[0, "text_id"] = "not_a_dialogue"
        self.assert_fails(bad, "not in the text pool")

    def test_pair_split_disagrees_with_source_split(self):
        bad = self.pairs.copy()
        bad.loc[0, "split"] = "test" if bad.loc[0, "split"] != "test" else "train"
        self.assert_fails(bad, "split")

    def test_clip_appears_in_two_splits(self):
        bad = self.pairs.copy()
        other = bad[bad["split"] != bad.loc[0, "split"]].index[0]
        bad.loc[other, "audio_id"] = bad.loc[0, "audio_id"]
        self.assert_fails(bad, "audio_id.*more than one split")

    def test_dialogue_appears_in_two_splits(self):
        bad = self.pairs.copy()
        other = bad[bad["split"] != bad.loc[0, "split"]].index[0]
        bad.loc[other, "text_id"] = bad.loc[0, "text_id"]
        self.assert_fails(bad, "text_id.*more than one split")

    def test_clip_used_twice(self):
        bad = self.pairs.copy()
        same_split = bad[(bad["split"] == bad.loc[0, "split"]) & (bad.index != 0)].index[0]
        bad.loc[same_split, "audio_id"] = bad.loc[0, "audio_id"]
        self.assert_fails(bad, "clip.*more than once")

    def test_dialogue_over_reuse_cap(self):
        bad = self.pairs.copy()
        part = bad[bad["split"] == "train"]
        victim = part.iloc[0]["text_id"]
        for index in part.index[1:6]:
            bad.loc[index, "text_id"] = victim
        self.assert_fails(bad, "reuse cap")

    def test_dialogue_unequal_genuine_and_spoof(self):
        bad = self.pairs.copy()
        part = bad[(bad["split"] == "train") & (bad["audio_label"] == 1)]
        victim = part.iloc[0]["text_id"]
        # Move another spoof pair onto the same dialogue so it has 2 spoof, 1 genuine.
        bad.loc[part.index[5], "text_id"] = victim
        self.assert_fails(bad, "equally", max_reuse=4)

    def test_wrong_target(self):
        bad = self.pairs.copy()
        bad.loc[0, "target"] = 1 - bad.loc[0, "target"]
        self.assert_fails(bad, "target")

    def test_label_disagrees_with_pool(self):
        bad = self.pairs.copy()
        bad.loc[0, "audio_label"] = 1 - bad.loc[0, "audio_label"]
        self.assert_fails(bad, "label")

    def test_unbalanced_cells(self):
        bad = self.pairs.iloc[1:].reset_index(drop=True)
        self.assert_fails(bad, "balanced")

    def test_external_split_in_pairs(self):
        bad = self.pairs.copy()
        bad.loc[0, "split"] = "external"
        self.assert_fails(bad, "split")

    def test_all_problems_are_reported_together(self):
        bad = self.pairs.copy()
        bad.loc[0, "target"] = 1 - bad.loc[0, "target"]
        bad.loc[1, "audio_id"] = "not_a_clip"
        with self.assertRaises(LeakageError) as caught:
            check_pairs(bad, self.audio, self.text)
        self.assertGreaterEqual(len(caught.exception.problems), 2)


if __name__ == "__main__":
    unittest.main()
