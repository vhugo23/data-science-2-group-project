"""Tests for the data-prep and figure helpers behind the demo notebook (headless matplotlib)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.evaluate import RESULT_COLUMNS, SWEEP_COLUMNS
from src.fusion.plots import (
    OKABE_ITO,
    SYSTEM_ORDER,
    SYSTEM_STYLES,
    assert_flags_match,
    banner_html,
    banner_lines,
    condition_label,
    difference_curves,
    figure_png,
    flag_label,
    plot_difference,
    plot_metric_vs_severity,
    plot_weight_sweep,
    results_flags,
    save_figure,
    severity_curves,
    sweep_curves,
)
from src.fusion.sources import load_sources
from src.fusion.tests.score_fixtures import write_audio, write_text

CONDITIONS = [("clean", None), ("noise", 5), ("noise", 10), ("noise", 20),
              ("text_prefix", 0.25), ("text_prefix", 0.5), ("text_prefix", 1.0)]


def make_results(conditions=CONDITIONS, dummy=True, plumbing_only=False, split="validation"):
    rows = []
    for c, (condition, severity) in enumerate(conditions):
        for s, system in enumerate(SYSTEM_ORDER + ("learned_minus_product",)):
            for metric in ("roc_auc", "pr_auc"):
                estimate = 0.9 - 0.01 * s - 0.02 * c if system != "learned_minus_product" else 0.01 * (c + 1)
                rows.append({
                    "condition": condition, "severity": np.nan if severity is None else float(severity),
                    "split": split, "seed": 0, "dummy": dummy, "plumbing_only": plumbing_only,
                    "scores_hash": "h", "system": system, "metric": metric, "threshold": np.nan,
                    "estimate": estimate, "ci_low": estimate - 0.05, "ci_high": estimate + 0.05,
                    "confidence": 0.95, "n_pairs": 40, "n_clusters": 20})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def make_sweeps(conditions=CONDITIONS[:4], dummy=True):
    rows = []
    for condition, severity in conditions:
        for ratio in (0.25, 1.0, 4.0):
            rows.append({"condition": condition, "severity": np.nan if severity is None else float(severity),
                         "split": "validation", "dummy": dummy, "eps": 1e-6, "ratio": ratio,
                         "roc_auc": 0.8, "pr_auc": 0.7})
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


class FlagTests(unittest.TestCase):
    def test_real_results_show_both_flags_as_false(self):
        label = flag_label(make_results(dummy=False, plumbing_only=False))
        self.assertIn("dummy=False", label)
        self.assertIn("plumbing_only=False", label)
        self.assertNotIn("DUMMY", label)
        self.assertNotIn("PLUMBING", label)

    def test_dummy_results_are_loudly_marked(self):
        label = flag_label(make_results(dummy=True, plumbing_only=False))
        self.assertIn("DUMMY", label)
        self.assertIn("dummy=True", label)
        self.assertIn("plumbing_only=False", label)

    def test_plumbing_only_results_are_loudly_marked(self):
        label = flag_label(make_results(dummy=False, plumbing_only=True))
        self.assertIn("PLUMBING-ONLY", label)
        self.assertIn("plumbing_only=True", label)

    def test_one_flagged_row_is_enough(self):
        results = make_results(dummy=False)
        results.loc[results.index[3], "dummy"] = True
        self.assertEqual(results_flags(results), (True, False))

    def test_missing_columns_or_empty_results_raise(self):
        with self.assertRaises(ValueError):
            flag_label(make_results().drop(columns="dummy"))
        with self.assertRaises(ValueError):
            flag_label(make_results().iloc[0:0])

    def test_flags_must_match_what_the_sources_declared(self):
        results = make_results(dummy=True, plumbing_only=True)
        assert_flags_match(True, True, results)  # does not raise
        with self.assertRaisesRegex(ValueError, "dummy"):
            assert_flags_match(False, True, results)
        with self.assertRaisesRegex(ValueError, "plumbing"):
            assert_flags_match(True, False, results)


class BannerTests(unittest.TestCase):
    def lines(self, bundle, **kwargs):
        args = dict(run_test_split=False, run_external=False, require_oof=True, seed=0)
        args.update(kwargs)
        return banner_lines(bundle, **args)

    def test_in_the_wild_is_always_reported_as_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root)
        for run_external in (False, True):
            text = " ".join(self.lines(bundle, run_external=run_external))
            with self.subTest(run_external=run_external):
                self.assertIn("In-the-Wild", text)
                self.assertIn("skipped", text)
        self.assertIn("RUN_EXTERNAL", " ".join(self.lines(bundle, run_external=True)))

    def test_test_split_line_follows_the_flag(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root)
        off = " ".join(self.lines(bundle, run_test_split=False)).lower()
        on = " ".join(self.lines(bundle, run_test_split=True)).lower()
        self.assertIn("test split", off)
        self.assertIn("not loaded", off)
        self.assertIn("test split", on)
        self.assertNotIn("not loaded", on)

    def test_dummy_fallback_is_announced_with_the_reason(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root)
        text = " ".join(self.lines(bundle))
        self.assertIn("DUMMY", text)
        self.assertIn("missing", text.lower())

    def test_smoke_audio_is_announced_as_plumbing_only(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="smoke")
            write_text(root)
            bundle = load_sources(root)
        text = " ".join(self.lines(bundle, require_oof=False)).lower()
        self.assertIn("smoke", text)
        self.assertIn("plumbing-only", text)
        self.assertIn("require_oof=false", text)

    def test_the_seed_is_shown(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root)
        self.assertIn("seed 7", " ".join(self.lines(bundle, seed=7)).lower())

    def test_requested_test_split_that_cannot_be_built_is_explained(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root, splits=("train", "validation"))
            bundle = load_sources(root, include_test=True)
        text = " ".join(self.lines(bundle, run_test_split=True))
        self.assertIn("test pairs cannot be built", text.lower())
        self.assertIn("text", text.lower())
        self.assertIn(bundle.test_unavailable_reason, text)

    def test_no_such_line_when_the_test_split_is_available_or_not_requested(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            available = load_sources(root, include_test=True)
            off = load_sources(root)
        self.assertNotIn("cannot be built", " ".join(self.lines(available, run_test_split=True)).lower())
        self.assertNotIn("cannot be built", " ".join(self.lines(off, run_test_split=False)).lower())

    def test_prefix_mismatches_give_a_visible_warning_naming_the_count(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root, prefix_consistent=False)
            bundle = load_sources(root)
        warning = [line for line in self.lines(bundle) if "WARNING" in line]
        self.assertEqual(len(warning), 1)
        self.assertIn("2", warning[0])
        self.assertIn("turn_fraction=1.0", warning[0])

    def test_no_warning_when_the_prefix_scores_agree(self):
        with tempfile.TemporaryDirectory() as root:
            write_audio(root, mode="full", fold=True)
            write_text(root)
            bundle = load_sources(root)
        self.assertFalse([line for line in self.lines(bundle) if "WARNING" in line])

    def test_lines_are_plain_strings_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = load_sources(root)
            lines = self.lines(bundle)
        self.assertTrue(lines and all(isinstance(line, str) for line in lines))
        for line in lines:
            self.assertNotIn(root, line)


class BannerHtmlTests(unittest.TestCase):
    def test_text_is_escaped_and_each_line_is_an_item(self):
        out = banner_html(["a <b> & c", "second"])
        self.assertIn("a &lt;b&gt; &amp; c", out)
        self.assertEqual(out.count("<li>"), 2)

    def test_warning_lines_get_the_alert_colour_and_clean_banners_do_not(self):
        self.assertIn("#D55E00", banner_html(["DUMMY DATA: fake"]))
        self.assertIn("#D55E00", banner_html(["WARNING: 3 mismatches"]))
        self.assertIn("#009E73", banner_html(["Real score tables"]))


class ConditionLabelTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(condition_label("clean", np.nan), "clean")
        self.assertEqual(condition_label("noise", 5.0), "noise 5")
        self.assertEqual(condition_label("text_prefix", 0.5), "text_prefix 0.5")
        self.assertEqual(condition_label("in_the_wild", None), "in_the_wild")


class SeverityCurveTests(unittest.TestCase):
    def ticks(self, curves):
        return list(curves.drop_duplicates("position").sort_values("position")["tick"])

    def test_noise_default_runs_clean_then_severity_as_supplied(self):
        curves = severity_curves(make_results(), "roc_auc", family="noise")
        self.assertEqual(self.ticks(curves), ["clean", "5", "10", "20"])

    def test_lower_is_worse_runs_clean_to_the_lowest_value(self):
        curves = severity_curves(make_results(), "roc_auc", family="noise", lower_is_worse=True)
        self.assertEqual(self.ticks(curves), ["clean", "20", "10", "5"])

    def test_text_prefix_always_runs_from_the_full_dialogue_to_the_shortest_prefix(self):
        for flag in (False, True):
            curves = severity_curves(make_results(), "roc_auc", family="text_prefix", lower_is_worse=flag)
            self.assertEqual(self.ticks(curves), ["clean", "1", "0.5", "0.25"])

    def test_one_row_per_system_and_position_with_values_taken_from_the_results(self):
        results = make_results()
        curves = severity_curves(results, "pr_auc", family="noise")
        self.assertEqual(set(curves["system"]), set(SYSTEM_ORDER))
        self.assertEqual(len(curves), 4 * 4)
        row = curves.query("system == 'text_only' and tick == '10'").iloc[0]
        source = results.query("system == 'text_only' and metric == 'pr_auc' and condition == 'noise' and severity == 10").iloc[0]
        self.assertEqual((row["estimate"], row["ci_low"], row["ci_high"]),
                         (source["estimate"], source["ci_low"], source["ci_high"]))
        self.assertEqual(list(curves.columns), ["system", "position", "tick", "estimate", "ci_low", "ci_high"])

    def test_the_paired_difference_rows_are_not_part_of_the_system_curves(self):
        curves = severity_curves(make_results(), "roc_auc", family="noise")
        self.assertNotIn("learned_minus_product", set(curves["system"]))

    def test_family_without_results_gives_an_empty_table(self):
        curves = severity_curves(make_results([("clean", None), ("noise", 5)]), "roc_auc", family="text_prefix")
        self.assertTrue(curves.empty)

    def test_bad_arguments_raise(self):
        results = make_results()
        with self.assertRaisesRegex(ValueError, "metric"):
            severity_curves(results, "eer", family="noise")
        with self.assertRaisesRegex(ValueError, "family"):
            severity_curves(results, "roc_auc", family="in_the_wild")
        with self.assertRaisesRegex(ValueError, "clean"):
            severity_curves(make_results([("noise", 5)]), "roc_auc", family="noise")
        with self.assertRaisesRegex(ValueError, "split"):
            severity_curves(pd.concat([make_results(), make_results(split="test")]), "roc_auc", family="noise")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            severity_curves(pd.concat([make_results(), make_results()]), "roc_auc", family="noise")


class DifferenceCurveTests(unittest.TestCase):
    def test_only_the_learned_minus_product_rows_are_returned_in_curve_order(self):
        results = make_results()
        curves = difference_curves(results, "roc_auc", family="noise", lower_is_worse=True)
        self.assertEqual(list(curves.sort_values("position")["tick"]), ["clean", "20", "10", "5"])
        self.assertEqual(set(curves["system"]), {"learned_minus_product"})
        row = curves.query("tick == '5'").iloc[0]
        source = results.query("system == 'learned_minus_product' and metric == 'roc_auc' "
                               "and condition == 'noise' and severity == 5").iloc[0]
        self.assertEqual(row["estimate"], source["estimate"])
        self.assertEqual((row["ci_low"], row["ci_high"]), (source["ci_low"], source["ci_high"]))


class SweepCurveTests(unittest.TestCase):
    def test_labels_and_condition_order(self):
        sweeps = pd.concat([make_sweeps([("text_prefix", 0.5), ("noise", 10), ("clean", None), ("noise", 5)])])
        out = sweep_curves(sweeps)
        self.assertEqual(list(out.drop_duplicates("label")["label"]),
                         ["clean", "noise 5", "noise 10", "text_prefix 0.5"])

    def test_lower_is_worse_reverses_the_noise_order_only(self):
        out = sweep_curves(make_sweeps([("clean", None), ("noise", 5), ("noise", 20)]), lower_is_worse=True)
        self.assertEqual(list(out.drop_duplicates("label")["label"]), ["clean", "noise 20", "noise 5"])

    def test_values_are_unchanged_and_nothing_is_selected(self):
        sweeps = make_sweeps()
        out = sweep_curves(sweeps)
        self.assertEqual(len(out), len(sweeps))
        self.assertNotIn("best", " ".join(out.columns).lower())


class StyleTests(unittest.TestCase):
    def test_palette_is_the_okabe_ito_colour_blind_safe_set(self):
        self.assertEqual(len(OKABE_ITO), 8)
        for system in SYSTEM_ORDER:
            self.assertIn(SYSTEM_STYLES[system]["color"], OKABE_ITO.values())

    def test_systems_differ_in_colour_line_style_and_marker(self):
        for key in ("color", "linestyle", "marker"):
            values = [SYSTEM_STYLES[system][key] for system in SYSTEM_ORDER]
            with self.subTest(key=key):
                self.assertEqual(len(set(values)), len(SYSTEM_ORDER))

    def test_each_system_has_a_readable_label(self):
        labels = [SYSTEM_STYLES[system]["label"] for system in SYSTEM_ORDER]
        self.assertEqual(len(set(labels)), 4)


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.results = make_results()
        self.flag = flag_label(self.results)

    def test_metric_figure_has_one_panel_per_family_and_the_flag_in_the_title(self):
        fig = plot_metric_vs_severity(self.results, "roc_auc", flag=self.flag)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn(self.flag, fig.get_suptitle())
        self.assertIn("ROC-AUC", fig.get_suptitle())

    def test_four_distinct_lines_per_panel_and_a_legend_naming_the_systems(self):
        fig = plot_metric_vs_severity(self.results, "pr_auc", flag=self.flag)
        ax = fig.axes[0]
        lines = [line for line in ax.get_lines() if line.get_label() in
                 {SYSTEM_STYLES[s]["label"] for s in SYSTEM_ORDER}]
        self.assertEqual(len(lines), 4)
        self.assertEqual(len({line.get_linestyle() for line in lines}), 4)
        self.assertEqual(len({line.get_marker() for line in lines}), 4)
        self.assertEqual(len({line.get_color() for line in lines}), 4)
        labels = ax.get_legend_handles_labels()[1]
        for system in SYSTEM_ORDER:
            self.assertIn(SYSTEM_STYLES[system]["label"], labels)

    def test_tick_labels_follow_lower_is_worse(self):
        fig = plot_metric_vs_severity(self.results, "roc_auc", flag=self.flag, lower_is_worse=True)
        ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        self.assertEqual(ticks, ["clean", "20", "10", "5"])

    def test_only_families_with_results_get_panels(self):
        results = make_results([("clean", None), ("noise", 5), ("noise", 10)])
        fig = plot_metric_vs_severity(results, "roc_auc", flag=flag_label(results))
        self.assertEqual(len(fig.axes), 1)

    def test_clean_only_results_raise_a_clear_error(self):
        results = make_results([("clean", None)])
        with self.assertRaisesRegex(ValueError, "degraded"):
            plot_metric_vs_severity(results, "roc_auc", flag=flag_label(results))

    def test_difference_figure_has_a_zero_line_and_the_flag(self):
        fig = plot_difference(self.results, flag=self.flag)
        self.assertIn(self.flag, fig.get_suptitle())
        self.assertIn("learned minus product", fig.get_suptitle().lower())
        zero_lines = [line for line in fig.axes[0].get_lines() if list(line.get_ydata()) == [0, 0]]
        self.assertTrue(zero_lines)

    def test_sweep_figure_has_one_line_per_condition_a_log_axis_and_the_flag(self):
        sweeps = make_sweeps()
        fig = plot_weight_sweep(sweeps, learned_ratio=1.7, flag=self.flag)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xscale(), "log")
        self.assertIn(self.flag, fig.get_suptitle())
        labels = ax.get_legend_handles_labels()[1]
        for label in ("clean", "noise 5", "noise 10", "noise 20"):
            self.assertIn(label, labels)
        self.assertIn("learned ratio", " ".join(labels).lower())
        self.assertIn("ratio 1", " ".join(labels).lower())

    def test_sweep_figure_survives_a_negative_learned_ratio(self):
        fig = plot_weight_sweep(make_sweeps(), learned_ratio=-0.4, flag=self.flag)
        self.assertNotIn("learned ratio", " ".join(fig.axes[0].get_legend_handles_labels()[1]).lower())

    def test_figure_png_returns_png_bytes_without_touching_disk(self):
        data = figure_png(plot_difference(self.results, flag=self.flag))
        self.assertEqual(data[:4], bytes([0x89]) + b"PNG")

    def test_every_figure_refuses_a_missing_flag(self):
        for bad in ("", "no flag here"):
            with self.subTest(flag=bad):
                with self.assertRaisesRegex(ValueError, "flag"):
                    plot_metric_vs_severity(self.results, "roc_auc", flag=bad)
                with self.assertRaisesRegex(ValueError, "flag"):
                    plot_difference(self.results, flag=bad)
                with self.assertRaisesRegex(ValueError, "flag"):
                    plot_weight_sweep(make_sweeps(), learned_ratio=1.0, flag=bad)


@unittest.skipIf(shutil.which("git") is None, "git is not available")
class SaveFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / ".gitignore").write_text("data/\n", encoding="utf-8")
        results = make_results()
        self.fig = plot_difference(results, flag=flag_label(results))

    def tearDown(self):
        self.tmp.cleanup()

    def test_saves_into_a_git_ignored_folder(self):
        path = save_figure(self.fig, "difference", self.root / "data" / "figs", repo_root=self.root)
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix, ".png")

    def test_refuses_a_folder_that_is_not_git_ignored(self):
        with self.assertRaisesRegex(ValueError, "ignored"):
            save_figure(self.fig, "difference", self.root / "notebooks" / "figs", repo_root=self.root)
        self.assertFalse((self.root / "notebooks").exists())

    def test_refuses_anything_under_outputs_even_if_it_were_ignored(self):
        (self.root / ".gitignore").write_text("data/\noutputs/\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outputs"):
            save_figure(self.fig, "difference", self.root / "outputs" / "fusion", repo_root=self.root)
        self.assertFalse((self.root / "outputs").exists())

    def test_refuses_a_folder_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            with self.assertRaisesRegex(ValueError, "repository"):
                save_figure(self.fig, "difference", Path(elsewhere), repo_root=self.root)


if __name__ == "__main__":
    unittest.main()
