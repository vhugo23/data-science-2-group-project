"""Threshold selection, metrics, and encoder cache naming."""
import unittest

import numpy as np

from src.audio.encoder import cache_path
from src.audio.evaluate import evaluate_scores, new_classifier, select_threshold


class EvaluateTests(unittest.TestCase):
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])

    def test_threshold_separates_perfect_scores(self):
        threshold = select_threshold(self.labels, self.scores)
        self.assertTrue(0.3 < threshold <= 0.6)

    def test_perfect_scores(self):
        result = evaluate_scores(self.labels, self.scores, 0.5)
        self.assertEqual((result["roc_auc"], result["eer"], result["f1"], result["n"]), (1.0, 0.0, 1.0, 6))

    def test_reversed_scores(self):
        result = evaluate_scores(self.labels, 1 - self.scores, 0.5)
        self.assertEqual(result["roc_auc"], 0.0)
        self.assertEqual(result["eer"], 1.0)
        self.assertEqual(result["f1"], 0.0)

    def test_classifier_is_seeded_and_balanced(self):
        model = new_classifier(seed=3)
        self.assertEqual(model[-1].random_state, 3)
        self.assertEqual(model[-1].class_weight, "balanced")


class CacheNameTests(unittest.TestCase):
    def test_name_matches_existing_smoke_caches(self):
        self.assertEqual(cache_path("out", "train").name, "wav2vec2-base_16k_4s_trim30_train.npz")


if __name__ == "__main__":
    unittest.main()
