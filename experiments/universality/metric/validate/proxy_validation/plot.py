#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

from src.colors import CYAN, GRAY_LIGHT, ROSE
from src.plotting import create_figure, save_figure, setup_plotting


def scatter_vs_ridge(df: pd.DataFrame, out: Path) -> None:
    """Each proxy metric against the held-out ridge R2 universality it stands in for."""
    setup_plotting()
    fig, axes = create_figure("single", ncols=2)

    panels = [
        ("cos2_perm_adj", "Cos² perm (adjusted)", ROSE),
        ("max_corr", "Max |corr|", CYAN),
    ]

    for ax, (metric, label, color) in zip(axes, panels):
        ax.scatter(
            df[metric], df["ridge_r2"],
            alpha=0.35, s=12, color=color, edgecolors="none",
        )
        lo = min(df[metric].min(), df["ridge_r2"].min()) - 0.02
        hi = max(df[metric].max(), df["ridge_r2"].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color=GRAY_LIGHT, ls="--", lw=0.8, zorder=0)

        r, _ = pearsonr(df[metric], df["ridge_r2"])
        rho, _ = spearmanr(df[metric], df["ridge_r2"])
        ax.text(0.05, 0.92, f"$r$ = {r:.2f}\n$\\rho$ = {rho:.2f}",
                transform=ax.transAxes, va="top")

        ax.set_xlabel(f"{label} universality")
        ax.set_ylabel("Ridge $R^2$ universality")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, out / "proxy_validation", formats=("pdf",))


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    del cfg  # the figure is fully specified by the table run.py wrote
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    scatter_vs_ridge(pd.read_csv(out / "metric_validation.csv"), out)


if __name__ == "__main__":
    main()
