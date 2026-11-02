"""Hygiene checks on the committed demo notebook (it is parsed, never executed here)."""
import json
import re
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "fusion" / "fusion_pipeline.ipynb"


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code(notebook):
    """(source, cell) for code cells only: markdown and raw cells are never scanned."""
    return [("".join(cell["source"]), cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def code_text(notebook):
    return "\n".join(source for source, _ in code(notebook))


class ScanOnlyCodeCellsTests(unittest.TestCase):
    """The hygiene checks below read code cells only, so documentation may mention anything."""

    def fake(self):
        forbidden = "C:\\Users\\someone\\data, to_csv, savefig, allow_test=True, outputs/fusion, condition=\"in_the_wild\""
        return {"cells": [
            {"cell_type": "markdown", "source": [forbidden]},
            {"cell_type": "raw", "source": [forbidden]},
            {"cell_type": "code", "source": ["x = 1\n", "y = 2"], "outputs": [], "execution_count": None},
        ]}

    def test_markdown_and_raw_cells_are_skipped(self):
        notebook = self.fake()
        self.assertEqual(code_text(notebook), "x = 1\ny = 2")
        self.assertEqual(len(code(notebook)), 1)

    def test_the_real_notebook_may_discuss_outputs_and_in_the_wild_in_markdown(self):
        notebook = load()
        markdown = " ".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        self.assertIn("In-the-Wild", markdown)
        self.assertIn("outputs/", markdown)  # the outputs/ policy is documented where readers see it


class NotebookHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load()
        cls.cells = code(cls.notebook)
        cls.source = code_text(cls.notebook)

    def test_no_outputs_and_no_execution_counts_are_saved(self):
        for source, cell in self.cells:
            self.assertEqual(cell.get("outputs", []), [], source[:60])
            self.assertIsNone(cell.get("execution_count"), source[:60])

    def test_no_absolute_paths_anywhere_in_the_notebook(self):
        for pattern in (r"[A-Za-z]:[\\/]", r"[A-Za-z]:\\\\", r"/Users/", r"/home/", r"~/"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, self.source))

    def test_flags_are_defined_at_the_top_and_default_to_safe_values(self):
        first = self.cells[0][0] + (self.cells[1][0] if len(self.cells) > 1 else "")
        for line in ("RUN_TEST_SPLIT = False", "RUN_EXTERNAL = False", "SAVE_FIGURES = False", "SEED = 0",
                     'TEXT_MODEL = "minilm"', "LOWER_IS_WORSE = False"):
            with self.subTest(line=line):
                self.assertRegex(first, rf"(?m)^{re.escape(line)}(?=\s|$)")

    def test_test_split_access_is_only_ever_controlled_by_the_flag(self):
        values = re.findall(r"allow_test\s*=\s*([A-Za-z_]+)", self.source)
        self.assertTrue(values)
        self.assertEqual(set(values), {"RUN_TEST_SPLIT"})
        self.assertIn("include_test=RUN_TEST_SPLIT", self.source)
        self.assertNotRegex(self.source, r"include_test\s*=\s*True")

    def test_notebook_never_writes_files_itself(self):
        for forbidden in ("to_csv", "to_json", "to_excel", "savefig", ".write_text", ".write_bytes", "shutil"):
            with self.subTest(call=forbidden):
                self.assertNotIn(forbidden, self.source)
        self.assertNotRegex(self.source, r"open\([^)]*['\"][wax]b?['\"]")
        # Reading the score files is done inside sources.py; the notebook names no outputs path.
        self.assertNotRegex(self.source, r"[\"'][^\"'\n]*outputs[^\"'\n]*[\"']")
        self.assertIn("save_figure", self.source)  # the only way figures may be written
        self.assertRegex(self.source, r"if SAVE_FIGURES")

    def test_frozen_model_round_trip_uses_a_temporary_folder(self):
        self.assertIn("tempfile", self.source)
        self.assertIn("save_model", self.source)
        self.assertIn("load_model", self.source)

    def test_notebook_uses_the_harness_and_the_helpers(self):
        for name in ("load_sources", "build_pairs", "check_pairs", "source_counts", "fit_fusion", "evaluate(",
                     "weight_ratio_sweep", "banner_lines", "flag_label", "assert_flags_match",
                     "plot_metric_vs_severity", "plot_difference", "plot_weight_sweep", "degrade_scores"):
            with self.subTest(name=name):
                self.assertIn(name, self.source)

    def test_the_seed_is_used_for_loading_pairing_evaluation_and_the_banner(self):
        for call in ("load_sources(", "build_pairs(", "evaluate(", "banner_lines("):
            start = self.source.index(call)
            block = self.source[start:start + 700]  # the call's arguments, however many lines they span
            with self.subTest(call=call):
                self.assertRegex(block, r"seed\s*=\s*SEED")

    def test_an_unavailable_test_split_is_reported_not_silently_skipped(self):
        self.assertIn("test_unavailable_reason", self.source)
        self.assertIn("banner_lines(", self.source)

    def test_in_the_wild_is_never_evaluated(self):
        self.assertNotRegex(self.source, r"condition\s*=\s*[\"']in_the_wild[\"']")

    def test_dummy_flag_is_passed_from_the_sources_to_the_pairs(self):
        self.assertRegex(self.source, r"dummy\s*=\s*bundle\.dummy")

    def test_oof_check_is_on_unless_the_sources_cannot_support_it(self):
        self.assertRegex(self.source, r"require_oof\s*=\s*[A-Za-z_.]*has_folds")

    def test_there_is_a_visible_banner_cell_before_any_results(self):
        index = next(i for i, (source, _) in enumerate(self.cells) if "banner_lines" in source)
        evaluate_index = next(i for i, (source, _) in enumerate(self.cells) if "evaluate(" in source)
        self.assertLess(index, evaluate_index)


if __name__ == "__main__":
    unittest.main()
