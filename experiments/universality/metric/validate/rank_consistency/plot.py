#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.colors import GRAY_DARK, GRAY_LIGHT, ROSE
from src.plotting import create_figure, save_figure, setup_plotting

RANKS = [10, 30, 50, 100, 200]


def _model_mean_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide format: rows=models, cols=ranks, values=mean universality."""
    means = df.groupby(["rank", "model"])["universality"].mean().reset_index()
    return means.pivot(index="model", columns="rank", values="universality").dropna()


def plot_rank_consistency(df: pd.DataFrame, output_dir: Path):
    """Three-panel: distributions per rank | mean ± std | stability heatmap."""
    ranks = sorted(df["rank"].unique())

    fig, axes = create_figure("single", ncols=3)

    ax = axes[0]
    sns.violinplot(
        data=df, x="rank", y="universality", order=ranks,
        color=ROSE, inner="box", ax=ax, cut=0,
    )
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Universality")

    ax = axes[1]
    summary = (
        df.groupby("rank")["universality"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("rank")
    )
    ax.errorbar(
        summary["rank"], summary["mean"], yerr=summary["std"],
        fmt="o-", color=GRAY_DARK, markersize=5, capsize=3,
    )
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Mean universality")

    ax = axes[2]
    wide = _model_mean_wide(df)
    n = len(ranks)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(wide[ranks[i]], wide[ranks[j]])
            corr[i, j] = rho
            corr[j, i] = rho
    corr_df = pd.DataFrame(corr, index=ranks, columns=ranks)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f", cmap="YlGnBu",
        vmin=corr[~np.eye(n, dtype=bool)].min() - 0.02, vmax=1.0,
        ax=ax, square=True, annot_kws={"fontsize": 7},
        xticklabels=[str(r) for r in ranks],
        yticklabels=[str(r) for r in ranks],
    )
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Dimensionality")

    fig.tight_layout()
    save_figure(fig, output_dir / "rank_consistency", formats=("pdf",))
    plt.close(fig)


def plot_rank_stability_scatter(df: pd.DataFrame, output_dir: Path, ref: int = 50):
    """Multi-panel scatter grid: ordinal model ranking at ref rank vs each other rank."""
    wide = _model_mean_wide(df)
    all_ranks = sorted(wide.columns)
    others = [r for r in all_ranks if r != ref]
    if not others or ref not in wide.columns:
        return

    ranks_ordinal = wide.rank(ascending=False)
    n_models = len(wide)

    fig, axes = create_figure("single", ncols=len(others))
    if len(others) == 1:
        axes = [axes]

    for ax, other in zip(axes, others):
        x = ranks_ordinal[ref]
        y = ranks_ordinal[other]
        ax.scatter(x, y, s=12, alpha=0.5, color=GRAY_DARK, edgecolors="none")
        ax.plot([1, n_models], [1, n_models],
                ls="--", color=GRAY_LIGHT, lw=0.8, zorder=0)
        ax.set_xlim(0, n_models + 1)
        ax.set_ylim(0, n_models + 1)
        ax.invert_xaxis()
        ax.invert_yaxis()

        rho, _ = spearmanr(x, y)
        ax.text(0.05, 0.95, f"ρ = {rho:.3f}",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel(f"Rank at k = {ref}")
        ax.set_ylabel(f"Rank at k = {other}")
        ax.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, output_dir / f"rank_stability_ref{ref}", formats=("pdf",))
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    setup_plotting()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    frames: dict[int, pd.DataFrame] = {}
    for rank in RANKS:
        path = (
            Path("experiments/universality/shared/data")
            / f"universality_{cfg.dataset}_rank_{rank}.csv"
        )
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table = table[table["metric"] == "cos2_perm_adj"].reset_index(drop=True)
        if not table.empty:
            frames[rank] = table

    if not frames:
        print("No ranks with cos2_perm_adj data found")
        return

    available_ranks = list(frames)
    print(f"Using ranks: {available_ranks}")
    df = pd.concat(frames.values(), ignore_index=True)

    plot_rank_consistency(df, out)
    ref = cfg.factorization.rank
    plot_rank_stability_scatter(
        df, out, ref=ref if ref in frames else available_ranks[-1]
    )

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
