"""Data prep and figures for the fusion demo notebook.

The data-prep functions (flags, banner lines, severity curves, sweep ordering) are plain
pandas and are unit tested. The figure functions return matplotlib ``Figure`` objects built
without pyplot, so they work headless and never open windows. Every figure title carries the
dummy/plumbing flag taken from the results. A higher score means more likely positive.
"""
import html
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

SYSTEM_ORDER = ("audio_only", "text_only", "product", "learned")
METRICS = ("roc_auc", "pr_auc")
METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
FAMILIES = ("noise", "text_prefix")
FAMILY_LABELS = {"noise": "Audio noise", "text_prefix": "Text prefix (fraction of turns)"}
DIFFERENCE = "learned_minus_product"
_CONDITION_RANK = {"clean": 0, "noise": 1, "in_the_wild": 2, "text_prefix": 3}

# Okabe-Ito colour-blind-safe palette.
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7"}
# Colour, line style and marker all differ per system, so no figure relies on colour alone.
SYSTEM_STYLES = {
    "audio_only": {"color": OKABE_ITO["blue"], "linestyle": "--", "marker": "s", "label": "Audio only"},
    "text_only": {"color": OKABE_ITO["orange"], "linestyle": ":", "marker": "^", "label": "Text only"},
    "product": {"color": OKABE_ITO["bluish_green"], "linestyle": "-.", "marker": "D", "label": "Product rule"},
    "learned": {"color": OKABE_ITO["vermillion"], "linestyle": "-", "marker": "o", "label": "Learned (logistic)"},
}
_SWEEP_COLORS = [OKABE_ITO[k] for k in ("black", "blue", "orange", "bluish_green", "vermillion",
                                        "sky_blue", "reddish_purple", "yellow")]
_SWEEP_STYLES = ["-", "--", "-.", ":"]
_SWEEP_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_INK = "#333333"
_GRID = "#dddddd"


# ---------------------------------------------------------------------------
# Flags and banner
# ---------------------------------------------------------------------------

def results_flags(results):
    """(dummy, plumbing_only): True if any results row carries the flag."""
    missing = [c for c in ("dummy", "plumbing_only") if c not in results.columns]
    if missing:
        raise ValueError(f"results are missing flag columns: {missing}")
    if results.empty:
        raise ValueError("results are empty, so there are no flags to show")
    return bool(results["dummy"].astype(bool).any()), bool(results["plumbing_only"].astype(bool).any())


def flag_label(results):
    """Text for figure titles: always dummy=..., plumbing_only=..., with loud words when set."""
    dummy, plumbing = results_flags(results)
    words = [word for word, on in (("DUMMY", dummy), ("PLUMBING-ONLY", plumbing)) if on]
    prefix = f"[{' | '.join(words)}] " if words else ""
    return f"{prefix}dummy={dummy}, plumbing_only={plumbing}"


def assert_flags_match(expected_dummy, expected_plumbing, results):
    """Raise ValueError if the results' flags differ from what the sources declared."""
    dummy, plumbing = results_flags(results)
    if dummy != bool(expected_dummy):
        raise ValueError(f"results have dummy={dummy} but the sources declared dummy={bool(expected_dummy)}")
    if plumbing != bool(expected_plumbing):
        raise ValueError(f"results have plumbing_only={plumbing} but the sources declared "
                         f"plumbing_only={bool(expected_plumbing)}")


def banner_lines(bundle, *, run_test_split, run_external, require_oof, seed):
    """Plain-text lines for the notebook banner, built from a sources.SourceBundle."""
    lines = []
    if bundle.audio_origin == "dummy":
        lines.append("DUMMY DATA: the real score files were not both found, so every table is a fake dummy table "
                     "and every result is dummy.")
    elif bundle.dummy:
        lines.append(f"PLUMBING-ONLY: audio source is {bundle.audio_origin}, text source is {bundle.text_origin}. "
                     "The pair table is flagged dummy, so nothing here is a finding.")
    else:
        lines.append("Real score tables: audio and text sources are both real.")
    lines.append(f"Audio source: {bundle.audio_origin}. Text source: {bundle.text_origin}.")
    lines += [note for note in bundle.notes if not note.startswith("Prefix mismatch:")]
    lines.append(f"Seed {seed} is used for loading, pairing and the bootstrap.")
    lines.append("Test split: loaded (RUN_TEST_SPLIT is True)." if run_test_split
                 else "Test split: not loaded (RUN_TEST_SPLIT is False); test rows never enter memory.")
    if require_oof:
        lines.append("require_oof=True: train scores are verified out-of-fold through the fold column.")
    else:
        lines.append("require_oof=False: the train scores could not be verified out-of-fold (no fold column), "
                     "so all results are plumbing-only.")
    design = ("its pairing is not built yet. Planned design: each clean test pair has its audio clip swapped for an "
              "In-the-Wild clip with the same audio label; the dialogue, text score and target stay unchanged")
    lines.append(("RUN_EXTERNAL is True but In-the-Wild is skipped: " if run_external else "In-the-Wild is skipped: ")
                 + design + ".")
    if bundle.prefix_mismatches:
        lines.append(f"WARNING: {bundle.prefix_mismatches} validation dialogues have turn_fraction=1.0 prefix "
                     "scores that differ from the clean text scores; check the prefix file before trusting "
                     "the text_prefix conditions.")
    return lines


def banner_html(lines):
    """A visible banner as HTML (text is escaped); for IPython.display.HTML."""
    alert = any(word in line for line in lines for word in ("DUMMY", "PLUMBING", "WARNING", "cannot be built"))
    border = "#D55E00" if alert else "#009E73"
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    return (f'<div style="border-left:6px solid {border};padding:8px 14px;margin:8px 0;'
            f'background:rgba(128,128,128,0.10)"><ul style="margin:0;padding-left:18px">{items}</ul></div>')


# ---------------------------------------------------------------------------
# Curve data
# ---------------------------------------------------------------------------

def condition_label(condition, severity):
    """'clean', 'noise 5', 'text_prefix 0.5'; the severity is omitted when there is none."""
    if severity is None or pd.isna(severity):
        return condition
    return f"{condition} {float(severity):g}"


def _single_split(table):
    splits = table["split"].unique()
    if len(splits) != 1:
        raise ValueError(f"results must come from one split, got {sorted(splits)}")
    return splits[0]


def _runs_descending(family, lower_is_worse):
    """Text-prefix fractions always run full -> shortest; noise follows lower_is_worse."""
    return True if family == "text_prefix" else bool(lower_is_worse)


def _curves(results, metric, family, lower_is_worse, systems):
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    _single_split(results)
    columns = ["system", "position", "tick", "estimate", "ci_low", "ci_high"]
    chosen = results[(results["metric"] == metric) & results["system"].isin(systems)]
    degraded = chosen[chosen["condition"] == family]
    if degraded.empty:
        return pd.DataFrame(columns=columns)
    clean = chosen[chosen["condition"] == "clean"]
    if clean.empty:
        raise ValueError("clean results are needed as the reference point for the curves")
    rows = pd.concat([clean, degraded])
    if rows.duplicated(["system", "condition", "severity"]).any():
        raise ValueError("duplicate result rows for the same system, condition and severity")
    severities = sorted(degraded["severity"].unique(), reverse=_runs_descending(family, lower_is_worse))
    order = [("clean", np.nan)] + [(family, s) for s in severities]
    out = []
    for position, (condition, severity) in enumerate(order):
        tick = "clean" if condition == "clean" else f"{severity:g}"
        part = rows[rows["condition"] == condition]
        if condition != "clean":
            part = part[part["severity"] == severity]
        for system in systems:
            row = part[part["system"] == system]
            if row.empty:
                continue
            row = row.iloc[0]
            out.append({"system": system, "position": position, "tick": tick, "estimate": row["estimate"],
                        "ci_low": row["ci_low"], "ci_high": row["ci_high"]})
    frame = pd.DataFrame(out, columns=columns)
    frame["system"] = pd.Categorical(frame["system"], categories=list(systems), ordered=True)
    frame = frame.sort_values(["system", "position"], kind="stable").reset_index(drop=True)
    frame["system"] = frame["system"].astype(str)
    return frame


def severity_curves(results, metric, *, family, lower_is_worse=False):
    """Per-system curves for one condition family, clean first.

    Columns: system, position, tick, estimate, ci_low, ci_high. Ticks show the severity as
    supplied. Noise runs clean -> severities in ascending order, or descending (clean -> lowest
    value, for SNR-style severities) with lower_is_worse=True. Text-prefix always runs from the
    full dialogue (clean) to the shortest prefix. Returns an empty table if the family has no results.
    """
    return _curves(results, metric, family, lower_is_worse, SYSTEM_ORDER)


def difference_curves(results, metric, *, family, lower_is_worse=False):
    """The paired learned-minus-product rows on the same positions as severity_curves."""
    return _curves(results, metric, family, lower_is_worse, (DIFFERENCE,))


def sweep_curves(sweeps, lower_is_worse=False):
    """Weight-ratio sweep rows with a ``label`` and an ``order`` column, in plotting order.

    Conditions run clean, noise, in_the_wild, text_prefix; noise follows lower_is_worse and
    text_prefix runs from the longest prefix. Nothing is selected: values are unchanged.
    """
    out = sweeps.copy()
    out["label"] = [condition_label(c, s) for c, s in zip(out["condition"], out["severity"])]
    sign = np.where(out["condition"].isin(["text_prefix"]) | ((out["condition"] == "noise") & lower_is_worse), -1.0, 1.0)
    severity = out["severity"].fillna(0.0).to_numpy(dtype=float)
    out["_rank"] = out["condition"].map(_CONDITION_RANK).fillna(len(_CONDITION_RANK))
    out["_severity"] = sign * severity
    out = out.sort_values(["_rank", "_severity", "ratio"], kind="stable")
    labels = list(dict.fromkeys(out["label"]))
    out["order"] = out["label"].map({label: i for i, label in enumerate(labels)})
    return out.drop(columns=["_rank", "_severity"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _require_flag(flag):
    if not flag or "dummy=" not in flag:
        raise ValueError("every figure needs the dummy/plumbing flag (use flag_label(results)) in its title")


def _style_axes(ax):
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors=_INK, labelsize=9)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)


def _panels(results, metric, lower_is_worse, systems_curves):
    panels = []
    for family in FAMILIES:
        curves = systems_curves(results, metric, family=family, lower_is_worse=lower_is_worse)
        if not curves.empty:
            panels.append((family, curves))
    return panels


def _set_ticks(ax, curves, family):
    ticks = curves.drop_duplicates("position").sort_values("position")
    ax.set_xticks(ticks["position"])
    ax.set_xticklabels(ticks["tick"])
    ax.set_xlabel(FAMILY_LABELS[family] + " (severity as supplied)" if family == "noise" else FAMILY_LABELS[family])


def plot_metric_vs_severity(results, metric, *, flag, lower_is_worse=False):
    """ROC-AUC or PR-AUC against severity for the four systems, one panel per condition family."""
    _require_flag(flag)
    split = _single_split(results)
    panels = _panels(results, metric, lower_is_worse, severity_curves)
    if not panels:
        raise ValueError("there are no degraded conditions in these results to plot against severity")
    fig = Figure(figsize=(5.4 * len(panels) + 0.6, 4.4), layout="constrained")
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for ax, (family, curves) in zip(axes, panels):
        for system in SYSTEM_ORDER:
            part = curves[curves["system"] == system]
            style = SYSTEM_STYLES[system]
            ax.fill_between(part["position"], part["ci_low"], part["ci_high"], color=style["color"],
                            alpha=0.12, linewidth=0)
            ax.plot(part["position"], part["estimate"], color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=6, linewidth=1.8, label=style["label"])
        _set_ticks(ax, curves, family)
        _style_axes(ax)
        ax.set_title(FAMILY_LABELS[family], fontsize=10, color=_INK)
    axes[0].set_ylabel(METRIC_LABELS[metric])
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"{METRIC_LABELS[metric]} vs severity ({split}) - {flag}\n"
                 "shaded bands: 95% dialogue-cluster bootstrap intervals", fontsize=11, color=_INK)
    return fig


def plot_difference(results, *, flag, lower_is_worse=False):
    """Learned minus product for ROC-AUC and PR-AUC against severity, with the paired interval."""
    _require_flag(flag)
    split = _single_split(results)
    columns = []
    for family in FAMILIES:
        if not difference_curves(results, "roc_auc", family=family).empty:
            columns.append(family)
    if not columns:
        raise ValueError("there are no degraded conditions in these results to plot against severity")
    fig = Figure(figsize=(5.4 * len(columns) + 0.6, 7.2), layout="constrained")
    axes = fig.subplots(2, len(columns), squeeze=False)
    color = OKABE_ITO["vermillion"]
    for r, metric in enumerate(METRICS):
        for c, family in enumerate(columns):
            ax = axes[r][c]
            curves = difference_curves(results, metric, family=family, lower_is_worse=lower_is_worse)
            ax.axhline(0, color="#666666", linewidth=1.0, linestyle="-")
            ax.errorbar(curves["position"], curves["estimate"],
                        yerr=[curves["estimate"] - curves["ci_low"], curves["ci_high"] - curves["estimate"]],
                        color=color, marker="o", markersize=6, linewidth=1.8, capsize=3, label="Learned minus product")
            _set_ticks(ax, curves, family)
            _style_axes(ax)
            ax.set_title(f"{METRIC_LABELS[metric]} - {FAMILY_LABELS[family]}", fontsize=10, color=_INK)
            if c == 0:
                ax.set_ylabel(f"{METRIC_LABELS[metric]} difference")
    axes[0][0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"Learned minus product ({split}) - {flag}\n"
                 "bars: paired 95% dialogue-cluster bootstrap intervals; above 0 favours the learned rule",
                 fontsize=11, color=_INK)
    return fig


def plot_weight_sweep(sweeps, learned_ratio, *, flag, lower_is_worse=False):
    """ROC-AUC and PR-AUC against w_audio / w_text (log axis), one line per condition.

    Ratio 1 (the product rule) is marked, and so is the learned ratio when it is positive.
    Nothing is selected from the sweep.
    """
    _require_flag(flag)
    split = _single_split(sweeps)
    curves = sweep_curves(sweeps, lower_is_worse)
    fig = Figure(figsize=(11.0, 4.6), layout="constrained")
    axes = fig.subplots(1, 2, sharex=True)
    for ax, metric in zip(axes, METRICS):
        for i, (label, part) in enumerate(curves.groupby("label", sort=False)):
            ax.plot(part["ratio"], part[metric], color=_SWEEP_COLORS[i % len(_SWEEP_COLORS)],
                    linestyle=_SWEEP_STYLES[i % len(_SWEEP_STYLES)], marker=_SWEEP_MARKERS[i % len(_SWEEP_MARKERS)],
                    markersize=6, linewidth=1.6, label=label)
        ax.set_xscale("log")
        ratios = sorted(curves["ratio"].unique())
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"{r:g}" for r in ratios])
        ax.minorticks_off()
        ax.axvline(1.0, color="#666666", linewidth=1.0, linestyle=":", label="ratio 1 (product rule)")
        if np.isfinite(learned_ratio) and learned_ratio > 0:
            ax.axvline(learned_ratio, color=OKABE_ITO["vermillion"], linewidth=1.4, linestyle="-",
                       alpha=0.7, label=f"learned ratio {learned_ratio:.2f}")
        ax.set_xlabel("w_audio / w_text (log scale)")
        ax.set_ylabel(METRIC_LABELS[metric])
        _style_axes(ax)
    if not (np.isfinite(learned_ratio) and learned_ratio > 0):
        axes[1].text(0.02, 0.03, f"learned ratio {learned_ratio:.2f} is not positive: not shown", fontsize=8,
                     color=_INK, transform=axes[1].transAxes)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"AUC against the audio/text weight ratio ({split}) - {flag}", fontsize=11, color=_INK)
    return fig


def figure_png(fig, dpi=110):
    """PNG bytes of a figure, held in memory only (nothing is written to disk)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


def save_figure(fig, name, folder, *, repo_root):
    """Save a figure as PNG in a git-ignored folder inside the repository; return the path.

    Raises ValueError, before writing anything, if the folder is outside the repository, is
    under outputs/, or is not ignored by git.
    """
    if not name or Path(name).name != name:
        raise ValueError(f"figure name must be a plain file name, got {name!r}")
    repo_root, folder = Path(repo_root).resolve(), Path(folder).resolve()
    try:
        relative = folder.relative_to(repo_root)
    except ValueError:
        raise ValueError(f"{folder} is outside the repository {repo_root}") from None
    if relative.parts and relative.parts[0] == "outputs":
        raise ValueError("figures are never written under outputs/")
    target = folder / f"{name}.png"
    check = subprocess.run(["git", "check-ignore", "-q", str(target)], cwd=repo_root, capture_output=True)
    if check.returncode == 1:
        raise ValueError(f"{relative.as_posix() or '.'} is not git-ignored; refusing to save figures there")
    if check.returncode != 0:
        raise ValueError("could not ask git whether the folder is ignored: " + check.stderr.decode(errors="replace"))
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    return target
