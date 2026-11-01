"""Tests for the extended metrics in src/metrics.py.

Every expected number below was worked out by hand; the comments show how.
Convention used throughout: label 1 is the positive class and a higher score
means "more positive".
"""
import inspect
import math
import unittest

import numpy as np

from src.metrics import (
    average_precision,
    cluster_bootstrap_ci,
    exact_eer,
    metrics_at_threshold,
    paired_cluster_bootstrap_ci,
    recall_at_fpr,
    safe_log_score,
    safe_logit,
)

# Four samples that alternate: 0.9 pos, 0.6 neg, 0.4 pos, 0.1 neg.
# ROC points (fpr, tpr) from the strictest threshold down:
# (0, 0) -> (0, .5) -> (.5, .5) -> (.5, 1) -> (1, 1)
ALT_Y = [1, 0, 1, 0]
ALT_S = [0.9, 0.6, 0.4, 0.1]


class ExactEerTests(unittest.TestCase):
    def test_perfect_separation_is_zero(self):
        self.assertEqual(exact_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0.0)

    def test_perfect_inversion_is_one(self):
        self.assertEqual(exact_eer([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_hits_a_roc_point_exactly(self):
        # The ROC passes through (.5, .5), where FPR = FNR = .5.
        self.assertAlmostEqual(exact_eer(ALT_Y, ALT_S), 0.5)

    def test_interpolates_between_roc_points(self):
        # neg: .5   pos: .2 .6 .9. FNR is 1/3 at every useful threshold while FPR
        # jumps from 0 to 1, so no threshold has FPR = FNR. Interpolating the ROC
        # line between (0, 2/3) and (1, 2/3): fpr = 1 - tpr = 1/3 -> EER = 1/3.
        # (The notebook's nearest-point rule would report 1/6 here.)
        eer = exact_eer([0, 1, 1, 1], [0.5, 0.2, 0.6, 0.9])
        self.assertAlmostEqual(eer, 1 / 3)

    def test_all_scores_tied_is_one_half(self):
        # Constant scorer: the ROC is the diagonal, which crosses FPR = FNR at .5.
        self.assertAlmostEqual(exact_eer([0, 0, 1, 1], [0.5] * 4), 0.5)

    def test_partial_tie_is_split_not_broken_arbitrarily(self):
        # neg: .3 .5   pos: .5 .9. The tied 0.5 pair moves together, giving the
        # diagonal segment (0, .5) -> (.5, 1); it meets fpr = 1 - tpr at fpr = .25.
        # Breaking the tie "positive first" would say 0, "negative first" .5.
        y = [0, 0, 1, 1]
        s = [0.3, 0.5, 0.5, 0.9]
        self.assertAlmostEqual(exact_eer(y, s), 0.25)

    def test_row_order_does_not_matter(self):
        y = [0, 0, 1, 1]
        s = [0.3, 0.5, 0.5, 0.9]
        order = [3, 1, 0, 2]
        self.assertAlmostEqual(
            exact_eer([y[i] for i in order], [s[i] for i in order]), exact_eer(y, s)
        )

    def test_random_scores_are_near_one_half(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 20000)
        self.assertAlmostEqual(exact_eer(y, rng.random(20000)), 0.5, delta=0.02)

    def test_accepts_arrays_and_lists(self):
        self.assertEqual(exact_eer(np.array([0, 1]), np.array([0.1, 0.9])), 0.0)


class AveragePrecisionTests(unittest.TestCase):
    def test_perfect_separation_is_one(self):
        self.assertEqual(average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_alternating_example(self):
        # Positives are found at ranks 1 and 3: precision 1/1 and 2/3, each worth
        # half the recall. AP = .5 * 1 + .5 * 2/3 = 5/6.
        self.assertAlmostEqual(average_precision(ALT_Y, ALT_S), 5 / 6)


class RecallAtFprTests(unittest.TestCase):
    def test_zero_fpr_budget(self):
        self.assertAlmostEqual(recall_at_fpr(ALT_Y, ALT_S, 0.0), 0.5)

    def test_budget_between_points_does_not_interpolate(self):
        # Only the real operating points are used, so .25 gets the best TPR with
        # FPR <= .25, which is the (0, .5) point.
        self.assertAlmostEqual(recall_at_fpr(ALT_Y, ALT_S, 0.25), 0.5)

    def test_budget_that_allows_everything_found(self):
        self.assertAlmostEqual(recall_at_fpr(ALT_Y, ALT_S, 0.5), 1.0)

    def test_rejects_budget_outside_unit_interval(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                recall_at_fpr(ALT_Y, ALT_S, bad)


class MetricsAtThresholdTests(unittest.TestCase):
    def test_counts_and_rates_at_half(self):
        # score >= 0.5 predicts positive: 0.9 (pos) and 0.6 (neg) -> tp 1, fp 1,
        # tn 1 (0.1), fn 1 (0.4).
        out = metrics_at_threshold(ALT_Y, ALT_S, 0.5)
        self.assertEqual((out["tp"], out["fp"], out["tn"], out["fn"]), (1, 1, 1, 1))
        for key in ("tpr", "fpr", "fnr", "precision", "accuracy", "f1"):
            self.assertAlmostEqual(out[key], 0.5, msg=key)
        self.assertEqual(out["threshold"], 0.5)

    def test_threshold_is_inclusive(self):
        out = metrics_at_threshold([1, 0], [0.5, 0.2], 0.5)
        self.assertEqual(out["tp"], 1)

    def test_nothing_predicted_positive_gives_zero_precision(self):
        out = metrics_at_threshold(ALT_Y, ALT_S, 0.95)
        self.assertEqual(out["tp"] + out["fp"], 0)
        self.assertEqual(out["precision"], 0.0)
        self.assertEqual(out["tpr"], 0.0)

    def test_does_not_pick_its_own_threshold(self):
        # The same data at two thresholds must give different answers.
        low = metrics_at_threshold(ALT_Y, ALT_S, 0.05)
        high = metrics_at_threshold(ALT_Y, ALT_S, 0.95)
        self.assertEqual(low["fpr"], 1.0)
        self.assertEqual(high["fpr"], 0.0)


class SafeTransformTests(unittest.TestCase):
    def test_logit_of_half_is_zero(self):
        self.assertAlmostEqual(float(safe_logit(0.5)), 0.0)

    def test_logit_known_value(self):
        self.assertAlmostEqual(float(safe_logit(0.9)), math.log(9))

    def test_logit_is_finite_at_zero_and_one_and_symmetric(self):
        low, high = safe_logit(np.array([0.0, 1.0]), eps=1e-6)
        self.assertTrue(math.isfinite(low) and math.isfinite(high))
        self.assertAlmostEqual(float(low), -float(high))
        self.assertAlmostEqual(float(high), math.log((1 - 1e-6) / 1e-6))

    def test_logit_keeps_order(self):
        values = safe_logit(np.array([0.0, 0.2, 0.7, 1.0]))
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_log_score_clips_zero(self):
        self.assertAlmostEqual(float(safe_log_score(0.0, eps=1e-6)), math.log(1e-6))
        self.assertAlmostEqual(float(safe_log_score(1.0)), 0.0)

    def test_rejects_values_outside_unit_interval_and_nan(self):
        for bad in ([-0.1], [1.1], [float("nan")]):
            with self.assertRaises(ValueError):
                safe_logit(np.array(bad))
            with self.assertRaises(ValueError):
                safe_log_score(np.array(bad))

    def test_rejects_bad_epsilon(self):
        for eps in (0.0, -1e-3, 0.5, 0.7):
            with self.assertRaises(ValueError):
                safe_logit([0.3], eps=eps)


class GuardTests(unittest.TestCase):
    def test_single_class_raises_a_clear_error(self):
        functions = [
            lambda y, s: exact_eer(y, s),
            lambda y, s: average_precision(y, s),
            lambda y, s: recall_at_fpr(y, s, 0.1),
            lambda y, s: metrics_at_threshold(y, s, 0.5),
        ]
        for function in functions:
            with self.assertRaisesRegex(ValueError, "both classes"):
                function([1, 1, 1], [0.1, 0.5, 0.9])
            with self.assertRaisesRegex(ValueError, "both classes"):
                function([0, 0], [0.1, 0.9])

    def test_other_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            exact_eer([0, 1], [0.1, 0.2, 0.3])  # length mismatch
        with self.assertRaises(ValueError):
            exact_eer([0, 2], [0.1, 0.2])  # label not 0/1
        with self.assertRaises(ValueError):
            exact_eer([0, 1], [0.1, float("nan")])  # non-finite score
        with self.assertRaises(ValueError):
            exact_eer([], [])  # empty


def recall_at_half(y, s):
    """Simple metric used by the bootstrap tests, independent of src/metrics."""
    y, s = np.asarray(y), np.asarray(s)
    return float(np.mean(s[y == 1] > 0.5))


def clustered_example():
    """200 rows in 20 clusters of 10; every row in a cluster has the same score.

    Clusters 0-9 are positive: seven score .9 (caught) and three score .1 (missed).
    Clusters 10-19 are negative and score .2. True recall at .5 is 0.7.
    """
    y = np.r_[np.ones(100), np.zeros(100)].astype(int)
    s = np.r_[np.repeat([0.9] * 7 + [0.1] * 3, 10), np.full(100, 0.2)]
    groups = np.r_[np.repeat(np.arange(10), 10), np.repeat(np.arange(10, 20), 10)]
    return y, s, groups


class ClusterBootstrapTests(unittest.TestCase):
    def test_same_seed_gives_same_interval(self):
        y, s, groups = clustered_example()
        first = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=200, seed=3)
        second = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=200, seed=3)
        self.assertEqual(first, second)

    def test_different_seed_gives_different_interval(self):
        y, s, groups = clustered_example()
        first = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=200, seed=1)
        second = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=200, seed=2)
        self.assertNotEqual((first["low"], first["high"]), (second["low"], second["high"]))

    def test_result_has_estimate_and_ordered_bounds(self):
        y, s, groups = clustered_example()
        out = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=200, seed=0)
        self.assertAlmostEqual(out["estimate"], 0.7)  # point estimate on the full data
        self.assertLessEqual(out["low"], out["estimate"])
        self.assertLessEqual(out["estimate"], out["high"])
        self.assertEqual(out["n_clusters"], 20)
        self.assertAlmostEqual(out["confidence"], 0.95)

    def test_confidence_follows_alpha(self):
        y, s, groups = clustered_example()
        out = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=50, alpha=0.1, seed=0)
        self.assertAlmostEqual(out["confidence"], 0.9)

    def test_clustered_interval_is_wider_than_naive(self):
        # Treating each row as its own cluster is the naive bootstrap. Because the
        # 10 rows of a cluster move together, resampling clusters must be wider.
        y, s, groups = clustered_example()
        clustered = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=1000, seed=0)
        naive = cluster_bootstrap_ci(
            y, s, np.arange(len(y)), recall_at_half, n_boot=1000, seed=0
        )
        clustered_width = clustered["high"] - clustered["low"]
        naive_width = naive["high"] - naive["low"]
        self.assertGreater(clustered_width, 2 * naive_width)

    def test_identical_clusters_give_a_zero_width_interval(self):
        # Two clusters, one per class, nothing to vary except the count of copies.
        y = [1, 1, 0, 0]
        s = [0.9, 0.9, 0.1, 0.1]
        groups = ["a", "a", "b", "b"]
        out = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=100, seed=0)
        self.assertEqual((out["low"], out["high"]), (1.0, 1.0))

    def test_replicates_missing_a_class_are_skipped_and_counted(self):
        # With only two clusters (one per class) half of the draws lack a class.
        y = [1, 1, 0, 0]
        s = [0.9, 0.9, 0.1, 0.1]
        groups = ["a", "a", "b", "b"]
        out = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=400, seed=0)
        self.assertGreater(out["n_skipped"], 0)
        self.assertEqual(out["n_valid"] + out["n_skipped"], 400)

    def test_group_labels_may_be_strings_or_numbers(self):
        y, s, groups = clustered_example()
        as_text = np.array([f"clip_{g}" for g in groups])
        a = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=100, seed=5)
        b = cluster_bootstrap_ci(y, s, as_text, recall_at_half, n_boot=100, seed=5)
        self.assertEqual((a["low"], a["high"]), (b["low"], b["high"]))

    def test_works_with_exact_eer_as_the_metric(self):
        y, s, groups = clustered_example()
        out = cluster_bootstrap_ci(y, s, groups, exact_eer, n_boot=100, seed=0)
        self.assertTrue(0.0 <= out["low"] <= out["high"] <= 1.0)

    def test_alpha_controls_the_interval_level(self):
        y, s, groups = clustered_example()
        wide = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=500, alpha=0.05, seed=0)
        narrow = cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=500, alpha=0.5, seed=0)
        self.assertGreaterEqual(wide["high"] - wide["low"], narrow["high"] - narrow["low"])

    def test_bad_arguments_raise(self):
        y, s, groups = clustered_example()
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(y, s, groups[:-1], recall_at_half)  # group length
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(y, s, groups, recall_at_half, alpha=1.5)
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(y, s, groups, recall_at_half, n_boot=0)
        with self.assertRaisesRegex(ValueError, "both classes"):
            cluster_bootstrap_ci([1, 1], [0.2, 0.3], ["a", "b"], recall_at_half)


def dense_interpolation_eer(y, s, per_segment=2000):
    """Independent EER: brute-force ROC points, then a dense grid on each segment.

    Shares no code with src/metrics.py. Thresholds are the distinct scores (so
    ties move together), plus one above the maximum. Between two neighbouring
    points the ROC is a straight line, sampled at `per_segment` places; the EER
    is the (fpr + fnr) / 2 of the sample where |fpr - fnr| is smallest.
    """
    y, s = np.asarray(y), np.asarray(s)
    thresholds = np.append(np.sort(np.unique(s))[::-1], s.max() + 1.0)
    points = []
    for t in thresholds:
        accepted = s >= t
        points.append((np.mean(accepted[y == 0]), np.mean(accepted[y == 1])))
    best_gap, best = np.inf, None
    for (f0, r0), (f1, r1) in zip(points[1:], points[:-1]):
        w = np.linspace(0.0, 1.0, per_segment)
        fpr = f0 + w * (f1 - f0)
        fnr = 1.0 - (r0 + w * (r1 - r0))
        i = np.argmin(np.abs(fpr - fnr))
        if abs(fpr[i] - fnr[i]) < best_gap:
            best_gap, best = abs(fpr[i] - fnr[i]), (fpr[i] + fnr[i]) / 2
    return best


class EerCrossCheckTests(unittest.TestCase):
    def test_matches_dense_interpolation_on_random_data(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(20, 120))
            y = rng.integers(0, 2, n)
            if y.min() == y.max():
                y[0] = 1 - y[0]
            # Mix of separation levels; rounding creates many ties.
            s = np.round(rng.normal(0, 1, n) + y * rng.uniform(0, 2), 1)
            with self.subTest(seed=seed):
                self.assertAlmostEqual(exact_eer(y, s), dense_interpolation_eer(y, s), delta=2e-3)

    def test_matches_dense_interpolation_with_heavy_ties(self):
        rng = np.random.default_rng(99)
        y = rng.integers(0, 2, 60)
        s = rng.integers(0, 4, 60).astype(float)
        self.assertAlmostEqual(exact_eer(y, s), dense_interpolation_eer(y, s), delta=2e-3)


class PairedClusterBootstrapTests(unittest.TestCase):
    def setUp(self):
        # Two scorers on the same 20 clusters of 10 rows. A misses the last 3
        # positive clusters. B also misses 3 positive clusters, but a different set.
        self.y, self.a, self.groups = clustered_example()
        self.b = np.r_[np.repeat([0.1] * 3 + [0.9] * 7, 10), np.full(100, 0.2)]

    def test_estimate_is_difference_on_full_data(self):
        out = paired_cluster_bootstrap_ci(
            self.y, self.a, self.b, self.groups, recall_at_half, n_boot=100, seed=0
        )
        self.assertAlmostEqual(out["estimate"], 0.0)  # both catch 70 of 100
        self.assertEqual(out["n_clusters"], 20)
        self.assertAlmostEqual(out["confidence"], 0.95)

    def test_identical_scorers_give_exactly_zero_difference(self):
        out = paired_cluster_bootstrap_ci(
            self.y, self.a, self.a, self.groups, recall_at_half, n_boot=200, seed=0
        )
        self.assertEqual((out["estimate"], out["low"], out["high"]), (0.0, 0.0, 0.0))

    def test_sign_flips_when_arguments_swap(self):
        ab = paired_cluster_bootstrap_ci(self.y, self.a, self.b, self.groups, recall_at_half, n_boot=200, seed=4)
        ba = paired_cluster_bootstrap_ci(self.y, self.b, self.a, self.groups, recall_at_half, n_boot=200, seed=4)
        self.assertAlmostEqual(ab["low"], -ba["high"])
        self.assertAlmostEqual(ab["high"], -ba["low"])

    def test_clear_improvement_gives_interval_above_zero(self):
        better = np.r_[np.repeat([0.9] * 10, 10), np.full(100, 0.2)]  # catches all
        # 90% interval: with only 10 positive clusters, about 2.8% of resamples
        # contain no missed cluster (0.7 ** 10), which would touch 0 at 95%.
        out = paired_cluster_bootstrap_ci(
            self.y, better, self.a, self.groups, recall_at_half, n_boot=300, alpha=0.1, seed=0
        )
        self.assertAlmostEqual(out["estimate"], 0.3)
        self.assertGreater(out["low"], 0.0)

    def test_uses_the_same_clusters_for_both_columns(self):
        # Same clusters for both columns means a scorer compared with itself can
        # never differ; independent resampling would give a wide, nonzero interval.
        out = paired_cluster_bootstrap_ci(
            self.y, self.a, self.a.copy(), self.groups, recall_at_half, n_boot=200, seed=1
        )
        self.assertEqual(out["high"] - out["low"], 0.0)

    def test_same_seed_is_reproducible(self):
        args = (self.y, self.a, self.b, self.groups, recall_at_half)
        one = paired_cluster_bootstrap_ci(*args, n_boot=100, seed=7)
        two = paired_cluster_bootstrap_ci(*args, n_boot=100, seed=7)
        self.assertEqual(one, two)

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap_ci(self.y, self.a, self.b[:-1], self.groups, recall_at_half)
        with self.assertRaisesRegex(ValueError, "both classes"):
            paired_cluster_bootstrap_ci([1, 1], [0.2, 0.3], [0.4, 0.5], ["a", "b"], recall_at_half)


class DocumentationContractTests(unittest.TestCase):
    def test_score_direction_and_threshold_rule_are_documented(self):
        for function in (exact_eer, average_precision, recall_at_fpr, cluster_bootstrap_ci,
                         paired_cluster_bootstrap_ci):
            doc = (function.__doc__ or "").lower()
            with self.subTest(function=function.__name__):
                self.assertIn("higher score", doc)
        for function in (metrics_at_threshold, recall_at_fpr, exact_eer):
            with self.subTest(function=function.__name__):
                self.assertIn(">=", function.__doc__)

    def test_eer_docstring_states_interpolation_and_ties(self):
        doc = exact_eer.__doc__.lower()
        self.assertIn("interpolat", doc)
        self.assertIn("tie", doc)

    def test_epsilon_is_an_explicit_documented_parameter(self):
        for function in (safe_logit, safe_log_score):
            parameter = inspect.signature(function).parameters["eps"]
            with self.subTest(function=function.__name__):
                self.assertEqual(parameter.default, 1e-6)
                self.assertIn("eps", function.__doc__)
                self.assertIn("1e-6", function.__doc__)

    def test_log_score_epsilon_changes_the_floor(self):
        self.assertAlmostEqual(float(safe_log_score(0.0, eps=1e-3)), math.log(1e-3))
        self.assertAlmostEqual(float(safe_log_score(0.0)), math.log(1e-6))


class ExistingFunctionUnchangedTests(unittest.TestCase):
    def test_classification_metrics_keeps_keys_and_values(self):
        from src.metrics import classification_metrics

        out = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
        self.assertEqual(list(out), ["accuracy", "precision", "recall", "f1", "roc_auc"])
        self.assertAlmostEqual(out["accuracy"], 0.75)
        self.assertAlmostEqual(out["roc_auc"], 1.0)
        self.assertEqual(list(classification_metrics([0, 1], [0, 1])), ["accuracy", "precision", "recall", "f1"])


if __name__ == "__main__":
    unittest.main()
