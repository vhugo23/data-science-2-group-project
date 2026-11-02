"""Tests for locating and loading score tables for the demo notebook.

All files are written into temporary folders; nothing reads the real outputs/ tree.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.fusion.scores import CONTRACT_COLUMNS
from src.fusion.sources import load_sources
from src.fusion.tests.score_fixtures import AUDIO_PATH, PREFIX_PATH, TEXT_PATH, write, write_audio, write_text

class RealSourceTests(unittest.TestCase):
    def test_full_audio_and_real_text_are_real_and_not_dummy(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            bundle = load_sources(root, audio_run="smoke")
        self.assertEqual((bundle.audio_origin, bundle.text_origin), ("real", "real"))
        self.assertFalse(bundle.dummy)
        self.assertTrue(bundle.has_folds)
        self.assertEqual(list(bundle.audio.columns[:4]), CONTRACT_COLUMNS)
        self.assertEqual(list(bundle.text.columns[:4]), CONTRACT_COLUMNS)
        self.assertNotIn("model", bundle.text.columns)
        self.assertTrue(bundle.text["sample_id"].str.startswith("d_").all())

    def test_smoke_audio_with_real_text_stays_smoke_and_is_flagged_dummy(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="smoke")
            write_text(root)
            bundle = load_sources(root)
        self.assertEqual((bundle.audio_origin, bundle.text_origin), ("smoke", "real"))
        self.assertTrue(bundle.dummy)
        self.assertFalse(bundle.has_folds)  # the smoke audio table has no fold column
        self.assertFalse(bundle.audio["sample_id"].str.startswith("dummy_").any())  # not replaced by fake tables
        notes = " ".join(bundle.notes).lower()
        self.assertIn("smoke", notes)
        self.assertIn("plumbing-only", notes)

    def test_missing_settings_file_is_treated_as_smoke_and_says_so(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, settings=False)
            write_text(root)
            bundle = load_sources(root)
        self.assertEqual(bundle.audio_origin, "smoke")
        self.assertTrue(any("settings.json" in note for note in bundle.notes))

    def test_text_model_choice_selects_the_matching_folder_and_model_name(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root, model_dir="baseline", model_name="tfidf_logreg")
            bundle = load_sources(root, text_model="baseline")
            self.assertEqual(bundle.text_origin, "real")
            self.assertEqual(bundle.prefix_tables, {})  # no prefix scores for the baseline
            self.assertTrue(any("prefix" in note.lower() for note in bundle.notes))

    def test_unknown_text_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "text_model"):
                load_sources(root, text_model="gpt")

    def test_recorded_paths_are_relative_and_notes_have_no_absolute_paths(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            bundle = load_sources(root)
        self.assertEqual(bundle.paths["audio"], AUDIO_PATH)
        self.assertEqual(bundle.paths["text"], TEXT_PATH)
        for text in list(bundle.paths.values()) + bundle.notes:
            self.assertNotIn(root, text)

    def test_loading_creates_no_files(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root)
            write_text(root)
            before = sorted(str(p) for p in Path(root).rglob("*"))
            load_sources(root)
            self.assertEqual(sorted(str(p) for p in Path(root).rglob("*")), before)


class DummyFallbackTests(unittest.TestCase):
    def test_missing_audio_file_makes_both_tables_dummy_and_names_the_file(self):
        with tempfile.TemporaryDirectory() as root:
            write_text(root)
            bundle = load_sources(root)
        self.assertEqual((bundle.audio_origin, bundle.text_origin), ("dummy", "dummy"))
        self.assertTrue(bundle.dummy)
        self.assertTrue(bundle.audio["sample_id"].str.startswith("dummy_").all())
        self.assertTrue(bundle.text["sample_id"].str.startswith("dummy_").all())
        self.assertTrue(any("missing" in note.lower() and AUDIO_PATH in note for note in bundle.notes))
        self.assertTrue(bundle.has_folds)  # dummy tables carry folds

    def test_missing_text_file_makes_both_tables_dummy_and_names_the_file(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root)
            bundle = load_sources(root)
        self.assertEqual((bundle.audio_origin, bundle.text_origin), ("dummy", "dummy"))
        self.assertTrue(any("missing" in note.lower() and TEXT_PATH in note for note in bundle.notes))

    def test_empty_root_falls_back_without_error(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root)
        self.assertTrue(bundle.dummy)
        self.assertEqual(bundle.prefix_tables, {})

    def test_dummy_tables_are_reproducible_from_the_seed(self):
        with tempfile.TemporaryDirectory() as root:
            a = load_sources(root, seed=3)
            b = load_sources(root, seed=3)
            c = load_sources(root, seed=4)
        pd.testing.assert_frame_equal(a.audio, b.audio)
        self.assertFalse(a.audio["score"].equals(c.audio["score"]))


class TestRowsAreFilteredAtReadTimeTests(unittest.TestCase):
    def test_test_rows_never_enter_the_tables_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, fold=True)
            write_text(root)
            bundle = load_sources(root)
        for table in (bundle.audio, bundle.text):
            self.assertEqual(set(table["split"]), {"train", "validation"})
        self.assertFalse(bundle.test_rows_loaded)

    def test_an_unparseable_test_score_is_never_even_parsed_when_the_flag_is_off(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, fold=True, test_score="NOT_A_NUMBER")
            write_text(root, test_score="NOT_A_NUMBER")
            bundle = load_sources(root)  # would raise if test rows were read into the tables
        self.assertEqual(len(bundle.audio), 8)
        self.assertEqual(len(bundle.text), 8)

    def test_with_the_flag_on_test_rows_are_loaded_and_parsed(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, fold=True)
            write_text(root)
            bundle = load_sources(root, include_test=True)
            self.assertEqual(set(bundle.audio["split"]), {"train", "validation", "test"})
            self.assertEqual(set(bundle.text["split"]), {"train", "validation", "test"})
            self.assertTrue(bundle.test_rows_loaded)
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, fold=True, test_score="NOT_A_NUMBER")
            write_text(root)
            with self.assertRaises(ValueError):
                load_sources(root, include_test=True)

    def test_external_rows_are_dropped_even_with_the_test_flag_on(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, fold=True)
            write_text(root)
            bundle = load_sources(root, include_test=True)
        self.assertNotIn("external", set(bundle.audio["split"]))

    def test_dummy_tables_also_respect_the_test_flag(self):
        with tempfile.TemporaryDirectory() as root:
            off = load_sources(root)
            on = load_sources(root, include_test=True)
        self.assertEqual(set(off.audio["split"]), {"train", "validation"})
        self.assertEqual(set(off.text["split"]), {"train", "validation"})
        self.assertIn("test", set(on.audio["split"]))

    def test_prefix_tables_are_validation_only(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, fold=True)
            write_text(root)
            bundle = load_sources(root, include_test=True)
        for table in bundle.prefix_tables.values():
            self.assertEqual(set(table["split"]), {"validation"})


class TestSplitAvailabilityTests(unittest.TestCase):
    def test_requesting_test_when_the_text_file_has_no_test_rows_gives_a_reason(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root, splits=("train", "validation"))
            bundle = load_sources(root, include_test=True)
        self.assertFalse(bundle.test_available)
        self.assertIn("text", bundle.test_unavailable_reason.lower())
        self.assertIn("test", bundle.test_unavailable_reason.lower())
        self.assertTrue(any("cannot be built" in note for note in bundle.notes))

    def test_audio_without_test_rows_is_named_too(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True, splits=("train", "validation"))
            write_text(root)
            bundle = load_sources(root, include_test=True)
        self.assertFalse(bundle.test_available)
        self.assertIn("audio", bundle.test_unavailable_reason.lower())

    def test_both_missing_names_both(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True, splits=("train", "validation"))
            write_text(root, splits=("train", "validation"))
            reason = load_sources(root, include_test=True).test_unavailable_reason.lower()
        self.assertIn("audio", reason)
        self.assertIn("text", reason)

    def test_no_reason_when_the_test_split_was_not_requested_or_is_available(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            off = load_sources(root, include_test=False)
            on = load_sources(root, include_test=True)
        self.assertIsNone(off.test_unavailable_reason)
        self.assertFalse(off.test_available)  # not loaded, so not available
        self.assertIsNone(on.test_unavailable_reason)
        self.assertTrue(on.test_available)

    def test_dummy_tables_have_test_rows_when_requested(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root, include_test=True)
        self.assertTrue(bundle.test_available)


class PrefixConsistencyTests(unittest.TestCase):
    def load(self, **kwargs):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root, **kwargs)
            return load_sources(root)

    def test_consistent_prefix_scores_report_zero_mismatches(self):
        bundle = self.load(prefix_consistent=True)
        self.assertEqual(bundle.prefix_mismatches, 0)
        self.assertFalse(any("mismatch" in note.lower() for note in bundle.notes))

    def test_inconsistent_prefix_scores_are_counted_and_noted_but_do_not_raise(self):
        bundle = self.load(prefix_consistent=False)
        self.assertEqual(bundle.prefix_mismatches, 2)
        self.assertTrue(any("2" in note and "mismatch" in note.lower() for note in bundle.notes))
        self.assertIn(1.0, bundle.prefix_tables)  # the tables are still returned

    def test_a_prefix_dialogue_absent_from_the_clean_table_counts_as_a_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            path = Path(root) / PREFIX_PATH
            path.write_text(path.read_text(encoding="utf-8")
                            + "d_validation_9,validation,1,minilm_logreg,0.50,-1,1.0\n", encoding="utf-8")
            bundle = load_sources(root)
        self.assertEqual(bundle.prefix_mismatches, 1)

    def test_tiny_floating_point_differences_are_not_mismatches(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            path = Path(root) / PREFIX_PATH
            text = path.read_text(encoding="utf-8").replace(",0.10,-1,1.0", ",0.1000001,-1,1.0")
            path.write_text(text, encoding="utf-8")
            self.assertEqual(load_sources(root).prefix_mismatches, 0)

    def test_no_check_when_there_are_no_prefix_scores_or_no_full_dialogue_rows(self):
        self.assertIsNone(self.load(prefix=False).prefix_mismatches)
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            path = Path(root) / PREFIX_PATH
            kept = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.endswith(",1.0")]
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            bundle = load_sources(root)
        self.assertIsNone(bundle.prefix_mismatches)
        self.assertTrue(any("turn_fraction=1.0" in note for note in bundle.notes))

    def test_dummy_bundles_have_nothing_to_check(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(load_sources(root).prefix_mismatches)


class PrefixTableTests(unittest.TestCase):
    def test_one_contract_table_per_turn_fraction(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            bundle = load_sources(root)
        self.assertEqual(sorted(bundle.prefix_tables), [0.25, 0.5, 1.0])
        for table in bundle.prefix_tables.values():
            self.assertEqual(list(table.columns[:4]), CONTRACT_COLUMNS)
            self.assertNotIn("turn_fraction", table.columns)
            self.assertTrue(table["sample_id"].is_unique)  # one condition per table

    def test_missing_prefix_file_gives_no_prefix_tables_and_a_note(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root, prefix=False)
            bundle = load_sources(root)
        self.assertEqual(bundle.prefix_tables, {})
        self.assertTrue(any(PREFIX_PATH in note for note in bundle.notes))


if __name__ == "__main__":
    unittest.main()
