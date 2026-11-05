#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import GRAY_DARK, GRAY_PALE, ROSE, TEAL
from src.plotting import paper_style


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.17,
        1.16,
        label,
        transform=axis.transAxes,
        fontsize=7,
        fontweight="bold",
        va="top",
    )


def _shared_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate(
        [values[np.isfinite(values)] for values in arrays]
    )
    low, high = finite.min(), finite.max()
    padding = max(0.015, 0.06 * (high - low))
    return max(0.0, float(low - padding)), min(1.0, float(high + padding))


def _grouped_selection_distribution(
    selection_distribution: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    exact = selection_distribution.set_index("selection_count")["fraction"]
    labels = ["0", "1", "2", "3", "4", "≥5"]
    values = np.asarray(
        [
            exact.get(0, 0.0),
            exact.get(1, 0.0),
            exact.get(2, 0.0),
            exact.get(3, 0.0),
            exact.get(4, 0.0),
            exact[exact.index >= 5].sum(),
        ],
        dtype=float,
    )
    return labels, values


def make_figure(
    *,
    per_dimension: pd.DataFrame,
    per_model: pd.DataFrame,
    selection_distribution: pd.DataFrame,
    statistics: dict[str, Any],
    params: dict[str, Any],
    output_dir: Path,
) -> None:
    paper_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 5.4,
            "axes.titleweight": "normal",
            "font.size": 5,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15))
    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.22,
        top=0.76,
        wspace=0.50,
    )

    comparisons = statistics["comparisons"]

    # Panel a: independent best-match versus Hungarian.
    ax = axes[0]
    x = per_dimension["hungarian_adjusted"].to_numpy()
    y = per_dimension["independent_adjusted"].to_numpy()
    limit = _shared_limits(x, y)
    ax.scatter(x, y, s=2.2, alpha=0.22, color=ROSE, linewidths=0, rasterized=True)
    ax.plot(limit, limit, color=GRAY_DARK, lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.set(
        xlim=limit,
        ylim=limit,
        xlabel="Hungarian universality",
        ylabel="Independent best-match",
        title="Many-to-one sensitivity",
    )
    ax.set_aspect("equal", adjustable="box")
    dim_stats = comparisons["independent_vs_hungarian"]["per_dimension"]
    ax.text(
        0.04,
        0.95,
        f"Spearman ρ = {dim_stats['spearman_adjusted']:.3f}\n"
        f"mean Δ = {dim_stats['mean_delta_adjusted']:+.3f}",
        transform=ax.transAxes,
        va="top",
        color=GRAY_DARK,
    )
    _panel_label(ax, "a")

    # Panel b: sequential greedy one-to-one versus Hungarian.
    ax = axes[1]
    x = per_dimension["hungarian_adjusted"].to_numpy()
    y = per_dimension["sequential_greedy_adjusted"].to_numpy()
    limit = _shared_limits(x, y)
    ax.scatter(x, y, s=2.2, alpha=0.22, color=TEAL, linewidths=0, rasterized=True)
    ax.plot(limit, limit, color=GRAY_DARK, lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.set(
        xlim=limit,
        ylim=limit,
        xlabel="Hungarian universality",
        ylabel="Sequential greedy",
        title="One-to-one algorithm",
    )
    ax.set_aspect("equal", adjustable="box")
    dim_stats = comparisons["sequential_greedy_vs_hungarian"]["per_dimension"]
    ax.text(
        0.04,
        0.95,
        f"Spearman ρ = {dim_stats['spearman_adjusted']:.3f}\n"
        f"mean Δ = {dim_stats['mean_delta_adjusted']:+.3f}",
        transform=ax.transAxes,
        va="top",
        color=GRAY_DARK,
    )
    _panel_label(ax, "b")

    # Panel c: model means under both alternatives.
    ax = axes[2]
    x = per_model["hungarian_adjusted"].to_numpy()
    independent = per_model["independent_adjusted"].to_numpy()
    sequential = per_model["sequential_greedy_adjusted"].to_numpy()
    limit = _shared_limits(x, independent, sequential)
    ax.scatter(
        x,
        independent,
        s=8,
        alpha=0.6,
        facecolor=ROSE,
        edgecolor="white",
        linewidth=0.25,
        label="Independent",
    )
    ax.scatter(
        x,
        sequential,
        s=8,
        alpha=0.6,
        facecolor=TEAL,
        edgecolor="white",
        linewidth=0.25,
        label="Sequential",
    )
    ax.plot(limit, limit, color=GRAY_DARK, lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.set(
        xlim=limit,
        ylim=limit,
        xlabel="Hungarian model mean",
        ylabel="Alternative model mean",
        title="Model-level estimates",
    )
    ax.set_aspect("equal", adjustable="box")
    independent_model = comparisons["independent_vs_hungarian"]["per_model"]
    sequential_model = comparisons["sequential_greedy_vs_hungarian"]["per_model"]
    ax.text(
        0.04,
        0.95,
        f"ρ independent = {independent_model['spearman_adjusted']:.3f}\n"
        f"ρ sequential = {sequential_model['spearman_adjusted']:.3f}",
        transform=ax.transAxes,
        va="top",
        color=GRAY_DARK,
    )
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=4.5,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    _panel_label(ax, "c")

    # Panel d: how many target dimensions claim each source dimension.
    ax = axes[3]
    labels, fractions = _grouped_selection_distribution(selection_distribution)
    bars = ax.bar(
        np.arange(len(labels)),
        fractions,
        color=TEAL,
        edgecolor="white",
        linewidth=0.3,
        width=0.82,
    )
    bars[0].set_color(ROSE)
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        xlabel="Independent selections/source dim.",
        ylabel="Fraction of source dimensions",
        title="Many-to-one collisions",
    )
    ax.set_ylim(0, max(fractions) * 1.2)
    ax.grid(axis="x", visible=False)
    unused = statistics["independent_best_match_collisions"][
        "fraction_of_source_dimensions_never_selected"
    ]
    ax.text(
        0.96,
        0.95,
        f"{unused:.1%} unused",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GRAY_DARK,
    )
    _panel_label(ax, "d")

    fig.suptitle(
        "Universality is robust to three matching rules",
        x=0.5,
        y=0.965,
        fontsize=7,
        fontweight="normal",
    )
    context = (
        f"THINGS · rank {params['rank']} · {params['n_models']} models · "
        f"{params['n_perms']} method-specific null permutations"
    )
    fig.text(0.5, 0.865, context, ha="center", va="center", color=GRAY_DARK)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(color=GRAY_PALE, linewidth=0.35)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "matching_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        output_dir / "matching_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    """Redraw the figure from the tables ``run.py`` already wrote."""
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    make_figure(
        per_dimension=pd.read_csv(out / "per_dimension.csv"),
        per_model=pd.read_csv(out / "per_model.csv"),
        selection_distribution=pd.read_csv(out / "selection_count_distribution.csv"),
        statistics=json.loads((out / "stats.json").read_text()),
        params=json.loads((out / "params.json").read_text()),
        output_dir=out,
    )


if __name__ == "__main__":
    main()
