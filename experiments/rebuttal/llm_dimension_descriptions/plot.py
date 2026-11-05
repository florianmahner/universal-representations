#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

from src.colors import GRAY, GRAY_DARK, GRAY_PALE, INDIGO, ROSE, TEAL
from src.plotting import PANEL_LABEL_FONT, PAPER_WIDTH, paper_style

# The four-way rating taxonomy shared by the human study and the LLM raters,
# in its fixed reporting order, with display name and color per label.
LABELS = {
    "semantic": ("Semantic", INDIGO),
    "visual": ("Visual", ROSE),
    "both": ("Both", TEAL),
    "neither": ("Neither", GRAY_DARK),
}

# The fine-grained tag vocabulary the classification prompt uses.
TAG_DISPLAY = {
    "object_category": "object/category",
    "superordinate_category": "broad category",
    "function_affordance": "function/use",
    "action_interaction": "action/interaction",
    "scene_context": "scene/context",
    "part_whole": "object part",
    "relation_role": "relation/role",
    "material_identity": "material type",
    "other_semantic": "other semantic",
    "color_tone": "color/tone",
    "texture": "texture",
    "material_appearance": "material appearance",
    "shape_form": "shape/form",
    "spatial_layout": "spatial layout",
    "viewpoint_orientation": "viewpoint/orientation",
    "lighting": "lighting",
    "scale_detail": "scale/detail",
    "background_context": "background/context",
    "photographic_style": "photographic style",
    "other_visual": "other visual",
    "food_drink": "food/drink",
    "body_people": "body/people",
    "tools_instruments": "tools/instruments",
    "furniture_household": "furniture/household",
    "natural_objects": "natural objects",
    "built_environment": "built environment",
    "sports_leisure": "sports/leisure",
    "single_coherent_theme": "one clear theme",
    "multiple_distinct_themes": "several distinct themes",
    "weak_partial_pattern": "weak/partial pattern",
    "polysemantic": "multiple meanings",
    "no_clear_pattern": "no clear pattern",
}


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.15,
        1.10,
        label,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_FONT,
        fontweight="bold",
        va="top",
    )


def _quiet_axis(axis: plt.Axes, *, x_grid: bool = False) -> None:
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    if x_grid:
        axis.grid(axis="x", color=GRAY_PALE, linewidth=0.35)
        axis.set_axisbelow(True)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
        )
        paths.append(str(path))
    plt.close(fig)
    return paths


def _require_completed(
    frame: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    required = {
        "dim_id",
        "universality",
        "majority_label",
        "n_classifications",
        "semantic_strength",
        "visual_strength",
        "derived_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "dimension_summary.csv does not yet contain classification fields: "
            + ", ".join(missing)
        )

    classified = frame[
        frame["semantic_strength"].notna()
        & frame["visual_strength"].notna()
        & frame["derived_label"].notna()
    ].copy()
    n_expected = int(summary.get("n_dimensions", len(frame)))
    n_complete = int(summary.get("dimensions_with_all_classifications", 0))
    if n_complete != n_expected:
        raise RuntimeError(
            "Refusing to plot an incomplete run: "
            f"{n_complete}/{n_expected} dimensions have all classifications."
        )
    if classified.empty:
        raise ValueError("No classified dimensions are available to plot")

    classified["human_label"] = classified["majority_label"].replace(
        {"mixed": "both"}
    )
    classified["llm_label"] = classified["derived_label"].replace(
        {"mixed": "both"}
    )
    for column in (
        "universality",
        "semantic_strength",
        "visual_strength",
        "semantic_strength_sd",
        "visual_strength_sd",
    ):
        if column in classified:
            classified[column] = pd.to_numeric(classified[column], errors="coerce")
    return classified


def _comparison_counts(frame: pd.DataFrame) -> np.ndarray:
    order = list(LABELS)
    counts = np.zeros((len(order), len(order)), dtype=int)
    for human, llm in zip(frame["human_label"], frame["llm_label"]):
        if human in LABELS and llm in LABELS:
            counts[order.index(human), order.index(llm)] += 1
    return counts


def plot_interpretation_overview(
    frame: pd.DataFrame, thresholds: dict[str, Any], figure_dir: Path
) -> list[str]:
    """Continuous semantic/visual scores and comparison with human labels."""
    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 2.50))
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.84,
        wspace=0.42,
    )

    axis = axes[0]
    for label in order[::-1]:
        subset = frame[frame["llm_label"] == label]
        axis.scatter(
            subset["visual_strength"],
            subset["semantic_strength"],
            s=4.5,
            alpha=0.42 if label != "neither" else 0.24,
            color=LABELS[label][1],
            edgecolors="none",
            rasterized=True,
        )
    semantic_threshold = float(thresholds.get("semantic", 0.5))
    visual_threshold = float(thresholds.get("visual", 0.5))
    axis.axhline(
        semantic_threshold, color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=0
    )
    axis.axvline(
        visual_threshold, color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=0
    )
    axis.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        yticks=[0, 0.25, 0.5, 0.75, 1],
        xlabel="Visual score",
        ylabel="Semantic score",
    )
    axis.set_title(
        f"Semantic and visual scores (n={len(frame):,})",
        pad=16,
    )
    quadrant_positions = {
        "semantic": (0.04, 0.95, "left", "top"),
        "both": (0.96, 0.95, "right", "top"),
        "visual": (0.96, 0.05, "right", "bottom"),
        "neither": (0.04, 0.05, "left", "bottom"),
    }
    for label, (x, y, horizontal, vertical) in quadrant_positions.items():
        display, color = LABELS[label]
        axis.text(
            x,
            y,
            display,
            transform=axis.transAxes,
            ha=horizontal,
            va=vertical,
            color=color,
            fontweight="medium",
        )
    _quiet_axis(axis)
    _panel_label(axis, "a")

    axis = axes[1]
    counts = _comparison_counts(frame)
    row_totals = counts.sum(axis=1)
    fractions = np.divide(
        counts,
        row_totals[:, None],
        out=np.zeros_like(counts, dtype=float),
        where=row_totals[:, None] > 0,
    )
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for column, label in enumerate(order):
        display, color = LABELS[label]
        values = fractions[:, column]
        bars = axis.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            label=display,
        )
        for row, (bar, value) in enumerate(zip(bars, values)):
            if value >= 0.09:
                axis.text(
                    left[row] + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="medium",
                )
        left += values
    axis.set(
        xlim=(0, 1),
        yticks=y,
        yticklabels=[
            f"{LABELS[label][0]} (n={row_totals[index]:,})"
            for index, label in enumerate(order)
        ],
        xlabel="Fraction of dimensions",
    )
    axis.set_title("LLM labels within each human rating", pad=16)
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.legend(
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    _quiet_axis(axis, x_grid=True)
    _panel_label(axis, "b")

    return _save(fig, figure_dir, "interpretation_overview")


def _clean_tag(tag: str) -> str:
    return TAG_DISPLAY.get(tag, tag.replace("_", " "))


def _tag_prevalence(frame: pd.DataFrame, prefix: str) -> pd.Series:
    columns = [column for column in frame if column.startswith(f"{prefix}__")]
    if not columns:
        raise ValueError(f"No {prefix} columns found in dimension_summary.csv")
    prevalence = frame[columns].apply(pd.to_numeric, errors="coerce").mean()
    prevalence.index = [
        _clean_tag(column.split("__", 1)[1]) for column in prevalence.index
    ]
    return prevalence.sort_values()


def _prevalence_panel(
    axis: plt.Axes,
    prevalence: pd.Series,
    *,
    title: str,
    color: str,
) -> None:
    y = np.arange(len(prevalence))
    bars = axis.barh(
        y,
        prevalence.to_numpy(),
        height=0.68,
        color=color,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.3,
    )
    axis.set(
        xlim=(0, 1),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        yticks=y,
        yticklabels=prevalence.index.tolist(),
        xlabel="Fraction of LLM judgments",
        title=title,
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for bar, value in zip(bars, prevalence.to_numpy()):
        if value <= 0:
            continue
        if value > 0.94:
            x, horizontal, text_color = value - 0.02, "right", "white"
        else:
            x, horizontal, text_color = value + 0.015, "left", GRAY_DARK
        axis.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            ha=horizontal,
            va="center",
            color=text_color,
        )
    _quiet_axis(axis, x_grid=True)


def plot_fine_grained_taxonomy(
    frame: pd.DataFrame, figure_dir: Path
) -> list[str]:
    """Semantic and visual criteria in a spacious two-panel row."""
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 2.75))
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.16,
        top=0.90,
        wspace=0.72,
    )
    _prevalence_panel(
        axes[0],
        _tag_prevalence(frame, "semantic_tag"),
        title="Semantic criteria",
        color=INDIGO,
    )
    _prevalence_panel(
        axes[1],
        _tag_prevalence(frame, "visual_tag"),
        title="Visual criteria",
        color=ROSE,
    )
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    return _save(fig, figure_dir, "fine_grained_taxonomy")


def plot_domains_and_structure(
    frame: pd.DataFrame, figure_dir: Path
) -> list[str]:
    """Semantic domains and the structural quality of descriptions."""
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 3.15))
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.14,
        top=0.91,
        wspace=0.72,
    )
    _prevalence_panel(
        axes[0],
        _tag_prevalence(frame, "semantic_domain"),
        title="Semantic domains",
        color=INDIGO,
    )
    _prevalence_panel(
        axes[1],
        _tag_prevalence(frame, "structural_tag"),
        title="Description structure",
        color=GRAY_DARK,
    )
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    return _save(fig, figure_dir, "fine_grained_domains_structure")


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return math.nan, math.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def _binned_series(
    x: pd.Series, y: pd.Series, n_bins: int = 10
) -> tuple[pd.Series, pd.Series, pd.Series]:
    valid = x.notna() & y.notna()
    data = pd.DataFrame(
        {"x": x[valid].astype(float), "y": y[valid].astype(float)}
    )
    bins = pd.qcut(data["x"], n_bins, duplicates="drop")
    grouped = data.groupby(bins, observed=True)
    return (
        grouped["x"].mean(),
        grouped["y"].mean(),
        grouped["y"].sem().fillna(0),
    )


def plot_universality(
    frame: pd.DataFrame, figure_dir: Path
) -> tuple[list[str], dict[str, float]]:
    """Universality by four-way label and by continuous semantic/visual scores."""
    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 2.35))
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.20,
        top=0.88,
        wspace=0.38,
    )

    axis = axes[0]
    rng = np.random.default_rng(0)
    positions = np.arange(len(order))
    values = [
        frame.loc[frame["llm_label"] == label, "universality"].dropna().to_numpy()
        for label in order
    ]
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.48,
        showfliers=False,
        patch_artist=True,
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
        medianprops={"linewidth": 1.0, "color": GRAY_DARK},
    )
    for patch, label in zip(box["boxes"], order):
        patch.set_facecolor(LABELS[label][1])
        patch.set_alpha(0.30)
    for position, label, label_values in zip(positions, order, values):
        axis.scatter(
            position + rng.uniform(-0.16, 0.16, len(label_values)),
            label_values,
            s=2.2,
            alpha=0.18,
            color=LABELS[label][1],
            linewidths=0,
            rasterized=True,
        )
    axis.set(
        xticks=positions,
        xticklabels=[LABELS[label][0] for label in order],
        xlabel="LLM label",
        ylabel="Universality",
        title="Universality by LLM label",
    )
    _quiet_axis(axis)
    _panel_label(axis, "a")

    axis = axes[1]
    correlations: dict[str, float] = {}
    for field, label, color, marker in (
        ("semantic_strength", "Semantic", INDIGO, "o"),
        ("visual_strength", "Visual", ROSE, "s"),
    ):
        rho, p_value = _safe_spearman(frame["universality"], frame[field])
        correlations[f"spearman_universality_{field}"] = rho
        correlations[f"pvalue_universality_{field}"] = p_value
        centers, means, sem = _binned_series(
            frame["universality"], frame[field]
        )
        axis.errorbar(
            centers,
            means,
            yerr=1.96 * sem,
            color=color,
            marker=marker,
            markerfacecolor="white" if marker == "s" else color,
            markeredgecolor=color,
            markeredgewidth=0.6,
            markersize=3.2,
            linewidth=1.0,
            capsize=1.4,
            label=f"{label} ($\\rho$={rho:.2f})",
        )
    axis.set(
        xlabel="Universality",
        ylabel="Mean score",
        ylim=(0, 1),
        title="Semantic and visual scores",
    )
    axis.legend(frameon=False, loc="best")
    _quiet_axis(axis)
    _panel_label(axis, "b")

    return _save(fig, figure_dir, "universality_interpretation"), correlations


def make_figures(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    dimension_path = output_dir / "dimension_summary.csv"
    if not summary_path.exists() or not dimension_path.exists():
        raise FileNotFoundError(
            "Expected summary.json and dimension_summary.csv under "
            f"{output_dir}"
        )

    paper_style()
    summary = json.loads(summary_path.read_text())
    frame = _require_completed(pd.read_csv(dimension_path), summary)
    figure_dir = output_dir / "figures"
    files: list[str] = []
    files.extend(
        plot_interpretation_overview(
            frame, summary.get("thresholds", {}), figure_dir
        )
    )
    files.extend(plot_fine_grained_taxonomy(frame, figure_dir))
    files.extend(plot_domains_and_structure(frame, figure_dir))
    universality_files, correlations = plot_universality(frame, figure_dir)
    files.extend(universality_files)

    counts = (
        frame["llm_label"]
        .value_counts()
        .reindex(list(LABELS), fill_value=0)
        .astype(int)
        .to_dict()
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(dimension_path),
        "n_classified_dimensions": int(len(frame)),
        "n_expected_dimensions": int(summary.get("n_dimensions", len(frame))),
        "llm_label_counts": counts,
        **correlations,
        "files": files,
    }
    (figure_dir / "plot_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    """Redraw the figures from the tables ``run.py`` already wrote."""
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    manifest = make_figures(out)
    print(
        f"Wrote {len(manifest['files'])} figure files for "
        f"{manifest['n_classified_dimensions']} dimensions to "
        f"{out / 'figures'}"
    )


if __name__ == "__main__":
    main()
