"""Tests for evaluate(), the weight-ratio sweep and the dummy degradation helper.

Dummy tables only; the text test scores are withheld and nothing here needs them.
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.fusion.dummy import degrade_scores, make_dummy_pools
from src.fusion.evaluate import DEFAULT_RATIOS, RESULT_COLUMNS, SWEEP_COLUMNS, evaluate, weight_ratio_sweep
from src.fusion.models import SYSTEMS, evaluation_hash, fit_fusion, join_scores
from src.fusion.pairing import build_pairs
from src.metrics import (
    DEFAULT_EPS,
    average_precision,
    cluster_bootstrap_ci,
    metrics_at_threshold,
    paired_cluster_bootstrap_ci,
)


def make_inputs(audio_strength=2.0, text_strength=4.0, seed=1):
    audio, text = make_dummy_pools(audio_strength=audio_strength, text_strength=text_strength, seed=seed)
    pairs = build_pairs(audio, text, dummy=True, seed=7)
    return pairs, audio, text


class DummyFoldTests(unittest.TestCase):
    def test_every_dummy_table_has_folds_zero_to_four_on_train_and_minus_one_elsewhere(self):
        for table in make_dummy_pools(seed=3):
            train = table[table["split"] == "train"]
            self.assertEqual(sorted(train["fold"].unique()), [0, 1, 2, 3, 4])
            self.assertTrue((table.loc[table["split"] != "train", "fold"] == -1).all())

    def test_folds_are_roughly_equal_in_size(self):
        audio, _ = make_dummy_pools(seed=3)
        sizes = audio[audio["split"] == "train"]["fold"].value_counts()
        self.assertLessEqual(sizes.max() - sizes.min(), 1)

    def test_folds_do_not_change_scores_labels_or_ids(self):
        # Adding the fold column must not disturb the random stream the other tests rely on.
        audio, _ = make_dummy_pools(seed=1)
        self.assertEqual(audio.loc[0, "sample_id"], "dummy_audio_train_0000")
        self.assertEqual(list(audio.columns[:4]), ["sample_id", "split", "label", "score"])


class DegradeScoresTests(unittest.TestCase):
    def setUp(self):
        self.audio, _ = make_dummy_pools(seed=1)

    def test_docstring_says_this_is_scaffolding_not_real_degradation(self):
        doc = degrade_scores.__doc__.lower()
        self.assertIn("test scaffolding only", doc)
        self.assertIn("re-scoring", doc)

    def test_keeps_ids_splits_labels_and_folds(self):
        out = degrade_scores(self.audio, shrink=0.3, noise=0.5, seed=0)
        for column in ("sample_id", "split", "label", "fold"):
            self.assertTrue(out[column].equals(self.audio[column]))
        self.assertTrue(out["score"].between(0, 1).all())

    def test_no_shrink_and_no_noise_changes_nothing(self):
        out = degrade_scores(self.audio, shrink=1.0, noise=0.0)
        np.testing.assert_allclose(out["score"], self.audio["score"])

    def test_shrinking_lowers_auc_and_the_seed_is_reproducible(self):
        weak = degrade_scores(self.audio, shrink=0.2, noise=1.0, seed=3)
        again = degrade_scores(self.audio, shrink=0.2, noise=1.0, seed=3)
        self.assertTrue(weak.equals(again))
        self.assertLess(roc_auc_score(weak["label"], weak["score"]),
                        roc_auc_score(self.audio["label"], self.audio["score"]))

    def test_input_is_not_modified(self):
        before = self.audio.copy()
        degrade_scores(self.audio, shrink=0.5, noise=1.0)
        self.assertTrue(self.audio.equals(before))

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            degrade_scores(self.audio, shrink=-1.0)
        with self.assertRaises(ValueError):
            degrade_scores(self.audio, shrink=1.0, noise=-0.1)


class EvaluateTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs, cls.audio, cls.text = make_inputs()
        cls.model = fit_fusion(cls.pairs, cls.audio, cls.text)
        cls.results = evaluate(cls.model, cls.pairs, cls.audio, cls.text,
                               condition="clean", seed=11, n_boot=200)

    def test_columns_and_row_count(self):
        self.assertEqual(list(self.results.columns), RESULT_COLUMNS)
        # 4 systems x 4 metrics, plus the paired learned-minus-product difference for ROC-AUC and PR-AUC.
        self.assertEqual(len(self.results), 18)
        systems = set(self.results["system"])
        self.assertEqual(systems, set(SYSTEMS) | {"learned_minus_product"})
        self.assertEqual(set(self.results["metric"]), {"roc_auc", "pr_auc", "tpr_at_threshold", "fpr_at_threshold"})
        diff = self.results[self.results["system"] == "learned_minus_product"]
        self.assertEqual(set(diff["metric"]), {"roc_auc", "pr_auc"})

    def test_every_row_carries_condition_severity_seed_split_and_dummy(self):
        self.assertEqual(set(self.results["condition"]), {"clean"})
        self.assertTrue(self.results["severity"].isna().all())
        self.assertEqual(set(self.results["seed"]), {11})
        self.assertEqual(set(self.results["split"]), {"validation"})
        self.assertEqual(set(self.results["dummy"]), {True})
        self.assertEqual(set(self.results["plumbing_only"]), {False})  # verified out-of-fold dummy run

    def test_n_pairs_equals_the_pair_table_count_even_with_extra_score_rows(self):
        expected = (self.pairs["split"] == "validation").sum()
        self.assertEqual(set(self.results["n_pairs"]), {expected})
        extra = self.audio.iloc[:1].assign(sample_id="dummy_audio_unused")
        again = evaluate(self.model, self.pairs, pd.concat([self.audio, extra]), self.text, condition="clean", n_boot=20)
        self.assertEqual(set(again["n_pairs"]), {expected})

    def test_every_row_records_the_hash_of_the_evaluated_score_tables(self):
        expected = evaluation_hash(self.pairs, self.audio, self.text, "validation")
        self.assertEqual(set(self.results["scores_hash"]), {expected})
        degraded = degrade_scores(self.audio, 0.5, 0.5, seed=1)
        other = evaluate(self.model, self.pairs, degraded, self.text, condition="noise", severity=10, n_boot=20)
        self.assertEqual(len(set(other["scores_hash"])), 1)
        self.assertNotEqual(set(other["scores_hash"]), {expected})

    def test_threshold_rows_report_tpr_and_fpr_at_the_frozen_validation_threshold(self):
        joined = join_scores(self.pairs, self.audio, self.text, "validation")
        y = joined["target"].to_numpy()
        scores = self.model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
        for system in SYSTEMS:
            direct = metrics_at_threshold(y, scores[system], self.model.thresholds[system])
            tpr = self.results.query("system == @system and metric == 'tpr_at_threshold'").iloc[0]
            fpr = self.results.query("system == @system and metric == 'fpr_at_threshold'").iloc[0]
            with self.subTest(system=system):
                self.assertAlmostEqual(tpr["estimate"], direct["tpr"])
                self.assertAlmostEqual(fpr["estimate"], direct["fpr"])
                self.assertEqual(tpr["threshold"], self.model.thresholds[system])
                self.assertLessEqual(fpr["estimate"], 0.05 + 1e-12)  # chosen on this very split
                self.assertLessEqual(tpr["ci_low"], tpr["ci_high"])
        auc_row = self.results.query("system == 'product' and metric == 'roc_auc'").iloc[0]
        self.assertEqual(auc_row["threshold"], self.model.thresholds["product"])

    def test_threshold_is_not_retuned_on_the_evaluated_data(self):
        degraded = degrade_scores(self.audio, 0.3, 1.0, seed=1)
        out = evaluate(self.model, self.pairs, degraded, self.text, condition="noise", severity=5, n_boot=20)
        row = out.query("system == 'audio_only' and metric == 'fpr_at_threshold'").iloc[0]
        self.assertEqual(row["threshold"], self.model.thresholds["audio_only"])

    def test_estimates_match_a_direct_computation(self):
        joined = join_scores(self.pairs, self.audio, self.text, "validation")
        y = joined["target"].to_numpy()
        scores = self.model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
        for system in SYSTEMS:
            auc = self.results.query("system == @system and metric == 'roc_auc'")["estimate"].item()
            ap = self.results.query("system == @system and metric == 'pr_auc'")["estimate"].item()
            self.assertAlmostEqual(auc, roc_auc_score(y, scores[system]), msg=system)
            self.assertAlmostEqual(ap, average_precision_score(y, scores[system]), msg=system)

    def test_intervals_are_clustered_by_dialogue_with_the_given_seed(self):
        joined = join_scores(self.pairs, self.audio, self.text, "validation")
        y = joined["target"].to_numpy()
        scores = self.model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
        expected = cluster_bootstrap_ci(y, scores["product"], joined["text_id"].to_numpy(),
                                        roc_auc_score, n_boot=200, seed=11)
        row = self.results.query("system == 'product' and metric == 'roc_auc'").iloc[0]
        self.assertEqual((row["ci_low"], row["ci_high"]), (expected["low"], expected["high"]))
        self.assertAlmostEqual(row["confidence"], 0.95)
        self.assertEqual(row["n_clusters"], joined["text_id"].nunique())
        self.assertEqual(row["n_pairs"], len(joined))
        self.assertLess(row["n_clusters"], row["n_pairs"])  # dialogues are reused, so clustering matters

    def test_pr_auc_uses_the_project_average_precision(self):
        joined = join_scores(self.pairs, self.audio, self.text, "validation")
        y = joined["target"].to_numpy()
        scores = self.model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
        expected = cluster_bootstrap_ci(y, scores["text_only"], joined["text_id"].to_numpy(),
                                        average_precision, n_boot=200, seed=11)
        row = self.results.query("system == 'text_only' and metric == 'pr_auc'").iloc[0]
        self.assertEqual((row["ci_low"], row["ci_high"]), (expected["low"], expected["high"]))

    def test_paired_row_is_learned_minus_product_on_the_same_clusters(self):
        joined = join_scores(self.pairs, self.audio, self.text, "validation")
        y = joined["target"].to_numpy()
        scores = self.model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
        expected = paired_cluster_bootstrap_ci(y, scores["learned"], scores["product"],
                                               joined["text_id"].to_numpy(), roc_auc_score,
                                               n_boot=200, seed=11)
        row = self.results.query("system == 'learned_minus_product' and metric == 'roc_auc'").iloc[0]
        self.assertEqual((row["estimate"], row["ci_low"], row["ci_high"]),
                         (expected["estimate"], expected["low"], expected["high"]))
        learned = self.results.query("system == 'learned' and metric == 'roc_auc'")["estimate"].item()
        product = self.results.query("system == 'product' and metric == 'roc_auc'")["estimate"].item()
        self.assertAlmostEqual(row["estimate"], learned - product)

    def test_same_seed_gives_the_same_table_and_a_new_seed_changes_intervals(self):
        again = evaluate(self.model, self.pairs, self.audio, self.text, condition="clean", seed=11, n_boot=200)
        pd.testing.assert_frame_equal(self.results, again)
        other = evaluate(self.model, self.pairs, self.audio, self.text, condition="clean", seed=12, n_boot=200)
        self.assertFalse(np.array_equal(self.results["ci_low"], other["ci_low"]))
        self.assertTrue(np.array_equal(self.results["estimate"], other["estimate"]))

    def test_inputs_are_not_modified(self):
        pairs, audio, text = self.pairs.copy(), self.audio.copy(), self.text.copy()
        evaluate(self.model, pairs, audio, text, condition="clean", n_boot=20)
        self.assertTrue(pairs.equals(self.pairs) and audio.equals(self.audio) and text.equals(self.text))


class EvaluateRulesTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()
        self.model = fit_fusion(self.pairs, self.audio, self.text)

    def run_eval(self, **kwargs):
        args = dict(condition="clean", n_boot=20)
        args.update(kwargs)
        return evaluate(self.model, self.pairs, self.audio, self.text, **args)

    def test_refuses_the_train_split(self):
        with self.assertRaisesRegex(ValueError, "train"):
            self.run_eval(split="train")

    def test_test_split_needs_an_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow_test"):
            self.run_eval(split="test")
        out = self.run_eval(split="test", allow_test=True)
        self.assertEqual(set(out["split"]), {"test"})

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_eval(split="external", allow_test=True)

    def test_condition_names_and_severity_rules(self):
        with self.assertRaisesRegex(ValueError, "condition"):
            self.run_eval(condition="sunny")
        with self.assertRaisesRegex(ValueError, "severity"):
            self.run_eval(condition="noise")  # noise needs a severity
        with self.assertRaisesRegex(ValueError, "severity"):
            self.run_eval(condition="clean", severity=1)  # clean has none
        with self.assertRaisesRegex(ValueError, "severity"):
            self.run_eval(condition="in_the_wild", severity=1)
        for fraction in (0.0, -0.2, 1.5):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "fraction"):
                    self.run_eval(condition="text_prefix", severity=fraction)

    def test_severity_is_recorded_for_noise_and_text_prefix(self):
        noise = self.run_eval(condition="noise", severity=10)
        self.assertEqual(set(noise["severity"]), {10})
        prefix = self.run_eval(condition="text_prefix", severity=0.5)
        self.assertEqual(set(prefix["severity"]), {0.5})
        self.assertEqual(set(prefix["condition"]), {"text_prefix"})

    def test_dummy_ids_with_a_false_dummy_flag_are_refused(self):
        with self.assertRaisesRegex(ValueError, "dummy"):
            evaluate(self.model, self.pairs.assign(dummy=False), self.audio, self.text,
                     condition="clean", n_boot=20)

    def test_mixed_dummy_flags_are_refused(self):
        pairs = self.pairs.copy()
        pairs.loc[pairs.index[0], "dummy"] = False
        with self.assertRaisesRegex(ValueError, "dummy"):
            evaluate(self.model, pairs, self.audio, self.text, condition="clean", n_boot=20)

    def test_real_looking_ids_with_a_true_dummy_flag_are_marked_dummy(self):
        # A smoke run: ids look real but the caller declares the run is a smoke test.
        # Fitting uses the dummy tables; only the evaluation inputs are renamed here.
        audio = self.audio.assign(sample_id=self.audio["sample_id"].str.replace("dummy_", "LA_", regex=False))
        text = self.text.assign(sample_id=self.text["sample_id"].str.replace("dummy_", "d_", regex=False))
        pairs = self.pairs.assign(audio_id=self.pairs["audio_id"].str.replace("dummy_", "LA_", regex=False),
                                  text_id=self.pairs["text_id"].str.replace("dummy_", "d_", regex=False))
        out = evaluate(self.model, pairs, audio, text, condition="clean", n_boot=20)
        self.assertEqual(set(out["dummy"]), {True})

    def test_dummy_model_marks_results_dummy_even_if_pairs_are_flagged_real(self):
        audio = self.audio.assign(sample_id=self.audio["sample_id"].str.replace("dummy_", "LA_", regex=False))
        text = self.text.assign(sample_id=self.text["sample_id"].str.replace("dummy_", "d_", regex=False))
        pairs = self.pairs.assign(audio_id=self.pairs["audio_id"].str.replace("dummy_", "LA_", regex=False),
                                  text_id=self.pairs["text_id"].str.replace("dummy_", "d_", regex=False),
                                  dummy=False)
        out = evaluate(self.model, pairs, audio, text, condition="clean", n_boot=20)
        self.assertEqual(set(out["dummy"]), {True})  # the model itself was fit on dummy data

    def test_pair_id_missing_from_a_condition_score_table_raises_and_names_it(self):
        used = self.pairs.loc[self.pairs["split"] == "validation", "audio_id"].iloc[0]
        for table_name in ("audio", "text"):
            with self.subTest(table=table_name):
                column = {"audio": "audio_id", "text": "text_id"}[table_name]
                used = self.pairs.loc[self.pairs["split"] == "validation", column].iloc[0]
                audio, text = self.audio, self.text
                if table_name == "audio":
                    audio = audio[audio["sample_id"] != used]
                else:
                    text = text[text["sample_id"] != used]
                with self.assertRaisesRegex(ValueError, f"missing.*{table_name}.*{used}"):
                    evaluate(self.model, self.pairs, audio, text, condition="noise", severity=5, n_boot=20)

    def test_pairs_of_other_splits_do_not_need_scores(self):
        text = self.text[self.text["split"] != "test"]
        out = evaluate(self.model, self.pairs, self.audio, text, condition="clean", n_boot=20)
        self.assertEqual(set(out["n_pairs"]), {(self.pairs["split"] == "validation").sum()})

    def test_plumbing_only_models_mark_every_result_row(self):
        audio, text = self.audio.drop(columns="fold"), self.text.drop(columns="fold")
        model = fit_fusion(self.pairs, audio, text, require_oof=False)
        out = evaluate(model, self.pairs, audio, text, condition="clean", n_boot=20)
        self.assertEqual(set(out["plumbing_only"]), {True})

    def test_stacked_conditions_in_one_score_table_are_rejected(self):
        stacked = pd.concat([self.audio, self.audio.assign(score=0.5)], ignore_index=True)
        with self.assertRaises(ValueError):
            evaluate(self.model, self.pairs, stacked, self.text, condition="clean", n_boot=20)

    def test_split_with_one_target_class_gives_a_clear_error(self):
        pairs = self.pairs.copy()
        mask = pairs["split"] == "validation"
        pairs.loc[mask, "target"] = 0
        pairs.loc[mask, "audio_label"] = 0  # keep the table internally consistent
        with self.assertRaises(ValueError):
            evaluate(self.model, pairs, self.audio, self.text, condition="clean", n_boot=20)


class WeightRatioSweepTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()

    def sweep(self, pairs=None, audio=None, text=None, **kwargs):
        return weight_ratio_sweep(self.pairs if pairs is None else pairs,
                                  self.audio if audio is None else audio,
                                  self.text if text is None else text,
                                  condition=kwargs.pop("condition", "clean"), **kwargs)

    def test_columns_and_one_row_per_ratio(self):
        out = self.sweep()
        self.assertEqual(list(out.columns), SWEEP_COLUMNS)
        self.assertEqual(out["ratio"].tolist(), list(DEFAULT_RATIOS))
        self.assertEqual(set(out["eps"]), {DEFAULT_EPS})  # the sweep states which eps it used
        self.assertEqual(set(out["condition"]), {"clean"})
        self.assertEqual(set(out["split"]), {"validation"})
        self.assertEqual(set(out["dummy"]), {True})
        self.assertTrue(out[["roc_auc", "pr_auc"]].apply(lambda c: c.between(0, 1)).all().all())

    def test_custom_eps_is_applied_and_recorded(self):
        default = self.sweep(ratios=(1.0,))
        coarse = self.sweep(ratios=(1.0,), eps=0.2)
        self.assertEqual(set(coarse["eps"]), {0.2})
        self.assertNotEqual(coarse["roc_auc"].item(), default["roc_auc"].item())

    def test_sweep_docstring_states_the_eps_it_uses(self):
        doc = weight_ratio_sweep.__doc__.lower()
        self.assertIn("default eps", doc)
        self.assertIn("eps column", doc)

    def test_sweep_reports_missing_ids_like_evaluate(self):
        used = self.pairs.loc[self.pairs["split"] == "validation", "audio_id"].iloc[0]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.sweep(audio=self.audio[self.audio["sample_id"] != used])

    def test_default_grid_is_positive_increasing_and_contains_one(self):
        ratios = list(DEFAULT_RATIOS)
        self.assertIn(1.0, ratios)
        self.assertEqual(ratios, sorted(ratios))
        self.assertTrue(all(r > 0 for r in ratios))

    def test_ratio_one_equals_the_product_rule(self):
        model = fit_fusion(self.pairs, self.audio, self.text)
        results = evaluate(model, self.pairs, self.audio, self.text, condition="clean", n_boot=20)
        product = results.query("system == 'product' and metric == 'roc_auc'")["estimate"].item()
        out = self.sweep(ratios=(1.0,), eps=DEFAULT_EPS)
        self.assertAlmostEqual(out["roc_auc"].item(), product)

    def test_extreme_ratios_approach_the_single_branch_systems(self):
        joined = join_scores(self.pairs, self.audio, self.text, "validation")
        y = joined["target"]
        audio_auc = roc_auc_score(y, joined["audio_score"])
        text_auc = roc_auc_score(y, joined["text_score"])
        out = self.sweep(ratios=(1e-6, 1e6)).set_index("ratio")
        self.assertAlmostEqual(out.loc[1e-6, "roc_auc"], text_auc, delta=0.01)
        self.assertAlmostEqual(out.loc[1e6, "roc_auc"], audio_auc, delta=0.01)

    def test_the_stronger_branch_wins_at_the_matching_end_of_the_grid(self):
        text_led = self.sweep(*make_inputs(audio_strength=0.5, text_strength=6.0)).set_index("ratio")
        audio_led = self.sweep(*make_inputs(audio_strength=6.0, text_strength=0.5)).set_index("ratio")
        low, high = min(DEFAULT_RATIOS), max(DEFAULT_RATIOS)
        self.assertGreater(text_led.loc[low, "roc_auc"], text_led.loc[high, "roc_auc"])
        self.assertGreater(audio_led.loc[high, "roc_auc"], audio_led.loc[low, "roc_auc"])

    def test_conditions_can_be_stacked_into_one_tidy_table(self):
        weaker = degrade_scores(self.audio, shrink=0.3, noise=1.0, seed=0)
        both = pd.concat([self.sweep(),
                          self.sweep(audio=weaker, condition="noise", severity=10)], ignore_index=True)
        self.assertEqual(set(both["condition"]), {"clean", "noise"})
        clean_end = both.query("condition == 'clean' and ratio == @DEFAULT_RATIOS[-1]")["roc_auc"].item()
        noise_end = both.query("condition == 'noise' and ratio == @DEFAULT_RATIOS[-1]")["roc_auc"].item()
        self.assertLess(noise_end, clean_end)  # audio-heavy weighting suffers when audio degrades

    def test_refuses_train_and_gates_test(self):
        with self.assertRaisesRegex(ValueError, "train"):
            self.sweep(split="train")
        with self.assertRaisesRegex(ValueError, "allow_test"):
            self.sweep(split="test")
        self.assertEqual(set(self.sweep(split="test", allow_test=True)["split"]), {"test"})

    def test_bad_ratios_are_rejected(self):
        for ratios in ((), (0.0, 1.0), (-1.0,), (float("inf"),), (float("nan"),)):
            with self.subTest(ratios=ratios):
                with self.assertRaises(ValueError):
                    self.sweep(ratios=ratios)

    def test_validates_condition_like_evaluate(self):
        with self.assertRaisesRegex(ValueError, "condition"):
            self.sweep(condition="sunny")
        with self.assertRaisesRegex(ValueError, "severity"):
            self.sweep(condition="noise")


class EndToEndTests(unittest.TestCase):
    def test_fit_freeze_reload_evaluate_every_condition_and_sweep(self):
        import tempfile
        from pathlib import Path

        from src.fusion.models import load_model, save_model

        pairs, audio, text = make_inputs()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(fit_fusion(pairs, audio, text), path)
            model = load_model(path)

        conditions = [
            ("clean", None, audio, text),
            ("noise", 20, degrade_scores(audio, 0.7, 0.5, seed=1), text),
            ("noise", 5, degrade_scores(audio, 0.3, 1.0, seed=2), text),
            ("in_the_wild", None, degrade_scores(audio, 0.5, 1.5, seed=3), text),
            ("text_prefix", 0.5, audio, degrade_scores(text, 0.4, 0.5, seed=4)),
        ]
        results, sweeps = [], []
        for name, severity, a, t in conditions:
            results.append(evaluate(model, pairs, a, t, condition=name, severity=severity, seed=5, n_boot=50))
            sweeps.append(weight_ratio_sweep(pairs, a, t, condition=name, severity=severity))
        results, sweeps = pd.concat(results, ignore_index=True), pd.concat(sweeps, ignore_index=True)

        self.assertEqual(len(results), 5 * 18)
        self.assertEqual(len(sweeps), 5 * len(DEFAULT_RATIOS))
        self.assertTrue(results["dummy"].all() and sweeps["dummy"].all())
        self.assertFalse(results[["estimate", "ci_low", "ci_high"]].isna().any().any())

        def auc(condition, severity, system):
            rows = results[(results["condition"] == condition) & (results["system"] == system)
                           & (results["metric"] == "roc_auc")]
            rows = rows[rows["severity"].isna()] if severity is None else rows[rows["severity"] == severity]
            return rows["estimate"].item()

        # Degrading audio hurts the audio branch and leaves the text branch untouched.
        self.assertLess(auc("noise", 5, "audio_only"), auc("clean", None, "audio_only"))
        self.assertAlmostEqual(auc("noise", 5, "text_only"), auc("clean", None, "text_only"))

if __name__ == "__main__":
    unittest.main()
