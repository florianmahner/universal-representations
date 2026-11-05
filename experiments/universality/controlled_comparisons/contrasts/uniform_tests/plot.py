from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.colors import GRAY, GRAY_DARK, GRAY_LIGHT, INDIGO, SAND
from src.plotting import create_figure, save_figure, setup_plotting

setup_plotting()


def plot_contrast_pvalues(tbl: pd.DataFrame, out_path: Path) -> None:
    """Horizontal plot of -log10(p) per contrast.

    Shows raw p (faded dot), BH-FDR q (mid dot), Bonferroni-corrected p (solid dot).
    Reference line at -log10(.05). Contrasts sorted by raw p.
    """
    testable = tbl[tbl["p_value"].notna()].copy().sort_values("p_value", ascending=False).reset_index(drop=True)
    n = len(testable)

    fig, ax = create_figure("single", pad_left=1.9)
    ax.axvline(-np.log10(0.05), color=GRAY_LIGHT, linewidth=0.8, linestyle="--", zorder=0)

    y = np.arange(n)
    offsets = np.array([+0.22, 0.0, -0.22])
    colors = [GRAY, SAND, INDIGO]
    labels_series = ["Raw", "BH-FDR", "Bonferroni (pre-specified family)"]
    cols = ["p_value", "p_bh_fdr", "p_bonferroni"]

    for col, offset, color, label in zip(cols, offsets, colors, labels_series):
        ps = testable[col].values
        xs = -np.log10(ps)
        for i, (p, x) in enumerate(zip(ps, xs)):
            sig = p < 0.05
            ax.scatter(x, y[i] + offset, color=color, s=28,
                       edgecolors="white", linewidth=0.3,
                       alpha=1.0 if sig else 0.45, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['label']} (n={r['n_total']}, {r['test']})" for _, r in testable.iterrows()],
        fontsize=6,
    )
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.text(-np.log10(0.05) + 0.05, n - 0.3, "α=.05", fontsize=6, color=GRAY_DARK, va="top")

    handles = [
        Line2D([0], [0], marker="o", color=GRAY, linestyle="", markersize=5, label="Raw p"),
        Line2D([0], [0], marker="o", color=SAND, linestyle="", markersize=5, label="BH-FDR q"),
        Line2D(
            [0], [0], marker="o", color=INDIGO, linestyle="", markersize=5,
            label="Bonferroni p (all pre-specified rows)",
        ),
        Line2D([0], [0], marker="o", color=GRAY, linestyle="", markersize=5, alpha=0.45, label="p ≥ .05"),
    ]
    ax.legend(handles=handles, fontsize=5, loc="lower right", frameon=False)

    save_figure(fig, out_path, formats=("pdf", "png"))
