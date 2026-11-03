"""Tests for fitting, freezing and reloading the fusion models. Dummy tables only."""
import dataclasses
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.fusion.checks import LeakageError
from src.fusion.dummy import make_dummy_pools
from src.fusion.models import (
    SYSTEMS,
    FusionModel,
    NegativeWeightWarning,
    check_inputs_match,
    choose_threshold,
    evaluation_hash,
    fit_fusion,
    input_hashes,
    join_scores,
    load_model,
    save_model,
)
from src.fusion.pairing import build_pairs
from src.metrics import DEFAULT_EPS, safe_log_score


def make_inputs(audio_strength=2.0, text_strength=4.0, seed=1):
    audio, text = make_dummy_pools(audio_strength=audio_strength, text_strength=text_strength, seed=seed)
    pairs = build_pairs(audio, text, dummy=True, seed=7)
    return pairs, audio, text


class JoinScoresTests(unittest.TestCase):
    def test_adds_both_scores_for_the_requested_split_only(self):
        pairs, audio, text = make_inputs()
        joined = join_scores(pairs, audio, text, "validation")
        self.assertEqual(set(joined["split"]), {"validation"})
        self.assertEqual(len(joined), (pairs["split"] == "validation").sum())
        row = joined.iloc[0]
        self.assertEqual(row["audio_score"], audio.set_index("sample_id").loc[row["audio_id"], "score"])
        self.assertEqual(row["text_score"], text.set_index("sample_id").loc[row["text_id"], "score"])

    def test_inputs_are_not_modified(self):
        pairs, audio, text = make_inputs()
        before = (pairs.copy(), audio.copy(), text.copy())
        join_scores(pairs, audio, text, "train")
        for table, original in zip((pairs, audio, text), before):
            self.assertTrue(table.equals(original))

    def test_score_table_in_another_split_is_a_leakage_error(self):
        pairs, audio, text = make_inputs()
        audio = audio.copy()
        audio.loc[audio["split"] == "train", "split"] = "test"
        with self.assertRaises(LeakageError):
            join_scores(pairs, audio, text, "train")

    def test_repeated_ids_in_a_score_table_are_rejected(self):
        # One table must hold one condition; stacked conditions would double count.
        pairs, audio, text = make_inputs()
        with self.assertRaises(ValueError):
            join_scores(pairs, pd.concat([audio, audio.iloc[:1]]), text, "train")


class MissingIdTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()

    def test_pair_id_missing_from_the_audio_table_is_named(self):
        used = self.pairs.loc[self.pairs["split"] == "validation", "audio_id"].iloc[0]
        audio = self.audio[self.audio["sample_id"] != used]
        with self.assertRaisesRegex(ValueError, f"missing.*audio.*{used}"):
            join_scores(self.pairs, audio, self.text, "validation")

    def test_pair_id_missing_from_the_text_table_is_named(self):
        used = self.pairs.loc[self.pairs["split"] == "train", "text_id"].iloc[0]
        text = self.text[self.text["sample_id"] != used]
        with self.assertRaisesRegex(ValueError, f"missing.*text.*{used}"):
            join_scores(self.pairs, self.audio, text, "train")

    def test_fit_refuses_when_a_train_pair_id_is_missing(self):
        used = self.pairs.loc[self.pairs["split"] == "train", "audio_id"].iloc[0]
        with self.assertRaisesRegex(ValueError, "missing"):
            fit_fusion(self.pairs, self.audio[self.audio["sample_id"] != used], self.text)

    def test_other_splits_missing_scores_do_not_block_a_validation_join(self):
        # Withheld text test scores: the pair table may name test dialogues that have no scores yet.
        text = self.text[self.text["split"] != "test"]
        joined = join_scores(self.pairs, self.audio, text, "validation")
        self.assertEqual(len(joined), (self.pairs["split"] == "validation").sum())


class FitRulesTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()

    def test_refuses_to_fit_on_any_split_but_train(self):
        for split in ("validation", "test", "external", "everything"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "train"):
                    fit_fusion(self.pairs, self.audio, self.text, fit_split=split)

    def test_refuses_to_tune_on_any_split_but_validation(self):
        for split in ("train", "test", "external"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "validation"):
                    fit_fusion(self.pairs, self.audio, self.text, tune_split=split)

    def test_needs_validation_pairs_to_choose_c_and_eps(self):
        no_validation = self.pairs[self.pairs["split"] != "validation"]
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_fusion(no_validation, self.audio, self.text)

    def test_needs_train_pairs(self):
        no_train = self.pairs[self.pairs["split"] != "train"]
        with self.assertRaisesRegex(ValueError, "train"):
            fit_fusion(no_train, self.audio, self.text)

    def test_empty_grids_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_fusion(self.pairs, self.audio, self.text, c_grid=())
        with self.assertRaises(ValueError):
            fit_fusion(self.pairs, self.audio, self.text, eps_grid=())

    def test_nothing_in_the_test_split_influences_the_fit(self):
        full = fit_fusion(self.pairs, self.audio, self.text)

        # Corrupt every test score and label.
        audio = self.audio.copy()
        text = self.text.copy()
        for table in (audio, text):
            mask = table["split"] == "test"
            table.loc[mask, "score"] = 0.5
        corrupted = fit_fusion(self.pairs, audio, text)

        # Remove the test split altogether (the text test scores are withheld today).
        without_pairs = self.pairs[self.pairs["split"] != "test"]
        without_test = fit_fusion(without_pairs, self.audio[self.audio["split"] != "test"],
                                  self.text[self.text["split"] != "test"])
        for other in (corrupted, without_test):
            self.assertEqual(other.to_dict(), full.to_dict())

    def test_a_different_train_set_changes_the_weights(self):
        other_pairs, other_audio, other_text = make_inputs(seed=2)
        a = fit_fusion(self.pairs, self.audio, self.text)
        b = fit_fusion(other_pairs, other_audio, other_text)
        self.assertNotEqual(a.weights, b.weights)

    def test_c_and_eps_come_from_the_grids_and_are_recorded_with_their_validation_scores(self):
        model = fit_fusion(self.pairs, self.audio, self.text, c_grid=(0.1, 1.0), eps_grid=(1e-6, 1e-3))
        self.assertIn(model.C, (0.1, 1.0))
        self.assertIn(model.eps, (1e-6, 1e-3))
        self.assertEqual(model.fit_split, "train")
        self.assertEqual(model.tune_split, "validation")
        self.assertEqual(len(model.tuning), 4)  # one row per (C, eps) pair
        best = max(row["roc_auc"] for row in model.tuning)
        chosen = [row for row in model.tuning if row["C"] == model.C and row["eps"] == model.eps]
        self.assertEqual(chosen[0]["roc_auc"], best)
        self.assertEqual(model.n_fit_pairs, (self.pairs["split"] == "train").sum())
        self.assertEqual(model.n_tune_pairs, (self.pairs["split"] == "validation").sum())

    def test_grid_order_and_duplicates_do_not_change_the_model(self):
        a = fit_fusion(self.pairs, self.audio, self.text, c_grid=(0.1, 1.0, 10.0), eps_grid=(1e-6, 1e-3))
        b = fit_fusion(self.pairs, self.audio, self.text, c_grid=(10.0, 1.0, 0.1, 1.0), eps_grid=(1e-3, 1e-6, 1e-3))
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual([(r["C"], r["eps"]) for r in a.tuning],
                         sorted((r["C"], r["eps"]) for r in a.tuning))  # canonical order

    def test_exact_ties_prefer_the_smaller_c_then_the_smaller_eps(self):
        # No dummy score is below 1e-9, so these two eps values clip nothing and tie exactly.
        for eps_grid in ((1e-9, 1e-12), (1e-12, 1e-9)):
            model = fit_fusion(self.pairs, self.audio, self.text, c_grid=(1.0,), eps_grid=eps_grid)
            self.assertEqual(model.eps, 1e-12)
            self.assertEqual(model.tuning[0]["roc_auc"], model.tuning[1]["roc_auc"])

    def test_bad_grid_values_are_rejected(self):
        for c in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                fit_fusion(self.pairs, self.audio, self.text, c_grid=(c,))
        with self.assertRaises(ValueError):
            fit_fusion(self.pairs, self.audio, self.text, eps_grid=(0.0,))


class OutOfFoldTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()

    def test_dummy_tables_have_folds_so_the_default_fit_passes_and_is_verified(self):
        model = fit_fusion(self.pairs, self.audio, self.text)
        self.assertTrue(model.oof_verified)
        self.assertFalse(model.plumbing_only)

    def test_tables_without_a_fold_column_fail_naming_the_column(self):
        for name in ("audio", "text"):
            tables = {"audio": self.audio, "text": self.text}
            tables[name] = tables[name].drop(columns="fold")
            with self.subTest(table=name):
                with self.assertRaisesRegex(ValueError, f"{name}.*'fold'"):
                    fit_fusion(self.pairs, tables["audio"], tables["text"])

    def test_train_rows_marked_as_not_out_of_fold_are_refused(self):
        text = self.text.copy()
        text.loc[text["split"] == "train", "fold"] = -1
        with self.assertRaisesRegex(ValueError, "out-of-fold"):
            fit_fusion(self.pairs, self.audio, text)

    def test_missing_fold_values_on_train_rows_are_refused(self):
        audio = self.audio.copy()
        audio["fold"] = audio["fold"].astype(float)
        audio.loc[audio.index[audio["split"] == "train"][0], "fold"] = np.nan
        with self.assertRaisesRegex(ValueError, "fold"):
            fit_fusion(self.pairs, audio, self.text)

    def test_escape_hatch_records_unverified_and_plumbing_only(self):
        audio, text = self.audio.drop(columns="fold"), self.text.drop(columns="fold")
        model = fit_fusion(self.pairs, audio, text, require_oof=False)
        self.assertFalse(model.oof_verified)
        self.assertTrue(model.plumbing_only)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.json"
            save_model(model, path)
            self.assertTrue(load_model(path).plumbing_only)

    def test_escape_hatch_never_claims_verification_even_when_folds_exist(self):
        model = fit_fusion(self.pairs, self.audio, self.text, require_oof=False)
        self.assertFalse(model.oof_verified)
        self.assertTrue(model.plumbing_only)


class SystemScoreTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()
        self.model = fit_fusion(self.pairs, self.audio, self.text)
        self.joined = join_scores(self.pairs, self.audio, self.text, "validation")
        self.a = self.joined["audio_score"].to_numpy()
        self.t = self.joined["text_score"].to_numpy()

    def test_all_four_systems_are_scored(self):
        self.assertEqual(SYSTEMS, ("audio_only", "text_only", "product", "learned"))
        scores = self.model.system_scores(self.a, self.t)
        self.assertEqual(tuple(scores), SYSTEMS)
        for values in scores.values():
            self.assertEqual(values.shape, self.a.shape)

    def test_single_branch_systems_rank_like_the_raw_score(self):
        scores = self.model.system_scores(self.a, self.t)
        y = self.joined["target"].to_numpy()
        self.assertAlmostEqual(roc_auc_score(y, scores["audio_only"]), roc_auc_score(y, self.a))
        self.assertAlmostEqual(roc_auc_score(y, scores["text_only"]), roc_auc_score(y, self.t))

    def test_product_is_the_sum_of_log_scores_with_equal_weights(self):
        expected = safe_log_score(self.a, DEFAULT_EPS) + safe_log_score(self.t, DEFAULT_EPS)
        np.testing.assert_allclose(self.model.system_scores(self.a, self.t)["product"], expected)

    def test_product_rule_does_not_depend_on_the_training_data(self):
        other = fit_fusion(*make_inputs(seed=2))
        np.testing.assert_array_equal(self.model.system_scores(self.a, self.t)["product"],
                                      other.system_scores(self.a, self.t)["product"])
        self.assertEqual(self.model.product, other.product)
        self.assertEqual(self.model.product, {"w_audio": 1.0, "w_text": 1.0, "eps": DEFAULT_EPS})

    def test_learned_is_a_linear_rule_on_log_scores_with_the_stored_eps(self):
        m = self.model
        expected = (m.intercept + m.weights["audio"] * safe_log_score(self.a, m.eps)
                    + m.weights["text"] * safe_log_score(self.t, m.eps))
        np.testing.assert_allclose(self.model.system_scores(self.a, self.t)["learned"], expected)

    def test_scores_outside_the_unit_interval_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.system_scores(np.array([1.5]), np.array([0.5]))

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.system_scores(self.a, self.t[:-1])


class WeightRatioTests(unittest.TestCase):
    def test_ratio_is_audio_weight_over_text_weight(self):
        model = fit_fusion(*make_inputs())
        self.assertAlmostEqual(model.weight_ratio, model.weights["audio"] / model.weights["text"])

    def test_ratio_follows_which_branch_is_stronger(self):
        audio_led = fit_fusion(*make_inputs(audio_strength=6.0, text_strength=0.5))
        text_led = fit_fusion(*make_inputs(audio_strength=0.5, text_strength=6.0))
        self.assertGreater(audio_led.weight_ratio, 1.5)
        self.assertLess(text_led.weight_ratio, 0.8)
        self.assertGreater(audio_led.weight_ratio, text_led.weight_ratio)


class FreezeAndReloadTests(unittest.TestCase):
    def setUp(self):
        self.inputs = make_inputs()
        self.model = fit_fusion(*self.inputs)

    def test_model_is_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.model.C = 5.0

    def test_round_trip_through_a_file_reproduces_every_score(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(self.model, path)
            reloaded = load_model(path)
        self.assertIsInstance(reloaded, FusionModel)
        self.assertEqual(reloaded.to_dict(), self.model.to_dict())
        joined = join_scores(*self.inputs, "validation")
        a, t = joined["audio_score"].to_numpy(), joined["text_score"].to_numpy()
        for system in SYSTEMS:
            np.testing.assert_array_equal(reloaded.system_scores(a, t)[system],
                                          self.model.system_scores(a, t)[system])

    def test_file_is_plain_json_with_weights_eps_c_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(self.model, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("weights", "intercept", "eps", "C", "weight_ratio", "hashes", "product",
                    "fit_split", "tune_split", "dummy", "oof_verified", "plumbing_only", "tuning", "schema",
                    "thresholds", "threshold_fpr"):
            self.assertIn(key, data)
        self.assertEqual(sorted(data["hashes"]), ["audio", "pairs", "text"])

    def test_loading_a_file_with_missing_fields_or_wrong_schema_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(self.model, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["weights"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(path)
            data = self.model.to_dict()
            data["schema"] = 999
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_model(path)

    def test_save_refuses_to_overwrite_a_frozen_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_model(self.model, path)
            with self.assertRaises(FileExistsError):
                save_model(self.model, path)

    def test_model_carries_the_dummy_flag_from_the_pair_table(self):
        self.assertTrue(self.model.dummy)
        pairs, audio, text = make_inputs()
        pairs = pairs.assign(dummy=False)
        # Dummy ids with dummy=False is a contradiction the fit must not hide.
        with self.assertRaisesRegex(ValueError, "dummy"):
            fit_fusion(pairs, audio, text)


class ThresholdTests(unittest.TestCase):
    def test_hand_example_allows_exactly_five_percent_false_positives(self):
        negatives = np.arange(1, 21) / 20  # 0.05 ... 1.00
        y = np.r_[np.zeros(20), np.ones(2)].astype(int)
        s = np.r_[negatives, [1.5, 2.0]]
        # t = 1.0 accepts one of twenty negatives (5%); t = 0.95 would accept two (10%).
        self.assertEqual(choose_threshold(y, s, 0.05), 1.0)

    def test_threshold_is_inclusive_so_a_score_equal_to_it_is_accepted(self):
        y = np.r_[np.zeros(20), [1]].astype(int)
        s = np.r_[np.arange(1, 21) / 20, [1.0]]
        t = choose_threshold(y, s, 0.05)
        self.assertEqual(np.mean(s[y == 0] >= t), 0.05)

    def test_ties_at_the_top_that_exceed_the_budget_are_not_split(self):
        y = np.r_[np.zeros(20), [1]].astype(int)
        s = np.r_[np.full(20, 0.5), [0.9]]
        t = choose_threshold(y, s, 0.05)
        self.assertGreater(t, 0.5)  # 0.5 would accept all 20 tied negatives
        self.assertEqual(t, 0.9)  # smallest score that keeps FPR at 0 while still catching the positive

    def test_zero_budget_and_a_positive_below_every_negative(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(choose_threshold(y, np.array([0.2, 0.8, 0.5, 0.9]), 0.0), 0.9)
        t = choose_threshold(y, np.array([0.7, 0.8, 0.1, 0.2]), 0.0)
        self.assertGreater(t, 0.8)  # nothing can be caught without a false positive

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [0.1, 0.9], 1.5)
        with self.assertRaisesRegex(ValueError, "both classes"):
            choose_threshold([1, 1], [0.1, 0.9], 0.05)

    def test_model_stores_one_validation_threshold_per_system_at_five_percent(self):
        pairs, audio, text = make_inputs()
        model = fit_fusion(pairs, audio, text)
        self.assertEqual(model.threshold_fpr, 0.05)
        self.assertEqual(tuple(model.thresholds), SYSTEMS)
        joined = join_scores(pairs, audio, text, "validation")
        y = joined["target"].to_numpy()
        scores = model.system_scores(joined["audio_score"].to_numpy(), joined["text_score"].to_numpy())
        for system in SYSTEMS:
            accepted = scores[system][y == 0] >= model.thresholds[system]
            with self.subTest(system=system):
                self.assertLessEqual(accepted.mean(), 0.05 + 1e-12)

    def test_thresholds_come_from_validation_only(self):
        pairs, audio, text = make_inputs()
        base = fit_fusion(pairs, audio, text)
        test_changed = audio.copy()
        test_changed.loc[test_changed["split"] == "test", "score"] = 0.5
        self.assertEqual(fit_fusion(pairs, test_changed, text).thresholds, base.thresholds)
        validation_changed = audio.copy()
        validation_changed.loc[validation_changed["split"] == "validation", "score"] = 0.5
        self.assertNotEqual(fit_fusion(pairs, validation_changed, text).thresholds["audio_only"],
                            base.thresholds["audio_only"])

    def test_thresholds_survive_a_round_trip(self):
        model = fit_fusion(*make_inputs())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.json"
            save_model(model, path)
            self.assertEqual(load_model(path).thresholds, model.thresholds)


class NegativeWeightTests(unittest.TestCase):
    def test_warns_when_a_learned_weight_is_negative_and_still_records_it(self):
        pairs, audio, text = make_inputs()
        flipped = audio.assign(score=1.0 - audio["score"])  # now a high score means genuine
        with self.assertWarnsRegex(NegativeWeightWarning, "negative.*audio"):
            model = fit_fusion(pairs, flipped, text)
        self.assertLess(model.weights["audio"], 0)

    def test_no_warning_when_both_weights_are_positive(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", NegativeWeightWarning)
            model = fit_fusion(*make_inputs())
        self.assertGreater(min(model.weights.values()), 0)

    def test_warning_is_a_user_warning(self):
        self.assertTrue(issubclass(NegativeWeightWarning, UserWarning))


class HashTests(unittest.TestCase):
    def setUp(self):
        self.pairs, self.audio, self.text = make_inputs()

    def test_hashes_are_stable_and_ignore_row_order(self):
        a = input_hashes(self.pairs, self.audio, self.text)
        b = input_hashes(self.pairs.sample(frac=1, random_state=0), self.audio.sample(frac=1, random_state=0),
                         self.text.sample(frac=1, random_state=0))
        self.assertEqual(a, b)

    def test_changing_a_train_score_changes_the_audio_hash_only(self):
        base = input_hashes(self.pairs, self.audio, self.text)
        audio = self.audio.copy()
        audio.loc[audio["split"] == "train", "score"] += 1e-9
        changed = input_hashes(self.pairs, audio, self.text)
        self.assertNotEqual(changed["audio"], base["audio"])
        self.assertEqual(changed["text"], base["text"])
        self.assertEqual(changed["pairs"], base["pairs"])

    def test_test_split_rows_are_not_hashed(self):
        base = input_hashes(self.pairs, self.audio, self.text)
        text = self.text.copy()
        text.loc[text["split"] == "test", "score"] = 0.5
        self.assertEqual(input_hashes(self.pairs, self.audio, text), base)

    def test_evaluation_hash_covers_only_the_evaluated_split_rows(self):
        base = evaluation_hash(self.pairs, self.audio, self.text, "validation")
        self.assertEqual(len(base), 64)
        other_split = self.audio.copy()
        other_split.loc[other_split["split"] == "train", "score"] = 0.5
        self.assertEqual(evaluation_hash(self.pairs, other_split, self.text, "validation"), base)
        same_split = self.audio.copy()
        same_split.loc[same_split["split"] == "validation", "score"] += 1e-9
        self.assertNotEqual(evaluation_hash(self.pairs, same_split, self.text, "validation"), base)
        unused = self.audio.iloc[:1].assign(sample_id="dummy_audio_extra")
        self.assertEqual(evaluation_hash(self.pairs, pd.concat([self.audio, unused]), self.text, "validation"), base)

    def test_fit_stores_the_hashes_and_check_accepts_the_same_inputs(self):
        model = fit_fusion(self.pairs, self.audio, self.text)
        self.assertEqual(model.hashes, input_hashes(self.pairs, self.audio, self.text))
        check_inputs_match(model, self.pairs, self.audio, self.text)  # does not raise

    def test_check_rejects_edited_inputs_and_names_the_table(self):
        model = fit_fusion(self.pairs, self.audio, self.text)
        text = self.text.copy()
        text.loc[text["split"] == "train", "score"] *= 0.5
        with self.assertRaisesRegex(ValueError, "text"):
            check_inputs_match(model, self.pairs, self.audio, text)


class DocumentationTests(unittest.TestCase):
    def test_eps_policy_is_documented(self):
        doc = (fit_fusion.__doc__ + FusionModel.__doc__).lower()
        self.assertIn("default eps", doc)  # product rule
        self.assertIn("tuned eps", doc)  # learned rule
        self.assertIn("higher score", doc)
        self.assertIn("require_oof", doc)

    def test_choose_threshold_docstring_states_the_rule(self):
        self.assertIn(">=", choose_threshold.__doc__)
        self.assertIn("validation", choose_threshold.__doc__.lower())


class IsolationTests(unittest.TestCase):
    def test_harness_modules_do_not_import_the_text_track_or_its_test_gate(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("models.py", "evaluate.py"):
            source = (root / name).read_text(encoding="utf-8")
            with self.subTest(module=name):
                self.assertNotIn("src.text", source)
                self.assertNotIn("challenge", source)

    def test_fit_makes_no_warnings_on_dummy_inputs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit_fusion(*make_inputs())


if __name__ == "__main__":
    unittest.main()
