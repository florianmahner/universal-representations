#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.colors import GRAY_DARK, GRAY_LIGHT
from src.plotting import save_figure, setup_plotting

log = logging.getLogger(__name__)


def _model_means(rank: int, dataset: str) -> pd.Series:
    """Model-mean universality as a named Series (index=model)."""
    table = (
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    df = pd.read_csv(table)
    df = df[df["metric"] == "cos2_perm_adj"]
    return df.groupby("model")["universality"].mean().rename(dataset)


def _available_ranks(ds_a: str, ds_b: str, ranks: list[int]) -> list[int]:
    """Ranks for which both datasets ship universality tables (ObjectNet: rank 50 only)."""
    data_dir = Path("experiments/universality/shared/data")
    return [
        rank
        for rank in ranks
        if (data_dir / f"universality_{ds_a}_rank_{rank}.csv").exists()
        and (data_dir / f"universality_{ds_b}_rank_{rank}.csv").exists()
    ]


def plot_model_scatter(ds_a: str, ds_b: str, ranks: list[int], output_dir: Path) -> None:
    """Scatter of model-mean universality across two datasets, one panel per rank."""
    n = len(ranks)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.4), squeeze=False)
    axes = axes.ravel()

    for i, (ax, rank) in enumerate(zip(axes, ranks)):
        ma, mb = _model_means(rank, ds_a), _model_means(rank, ds_b)
        merged = pd.concat([ma, mb], axis=1).dropna()
        rho, _ = spearmanr(merged[ds_a], merged[ds_b])
        log.info(f"Model scatter rank {rank}: n={len(merged)}, ρ={rho:.3f}")

        ax.scatter(
            merged[ds_a], merged[ds_b],
            s=10, alpha=0.5, color=GRAY_DARK, edgecolors="none",
        )
        lo = min(merged[ds_a].min(), merged[ds_b].min()) - 0.02
        hi = max(merged[ds_a].max(), merged[ds_b].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], ls="--", color=GRAY_LIGHT, lw=0.8, zorder=0)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.set_title(f"Rank {rank}", fontsize=9)
        ax.text(
            0.05, 0.95, f"ρ = {rho:.2f}\nn = {len(merged)}",
            transform=ax.transAxes, va="top", fontsize=7,
        )
        if i > 0:
            ax.set_yticklabels([])

    axes[0].set_ylabel(f"Mean universality ({ds_b})")
    axes[n // 2].set_xlabel(f"Mean universality ({ds_a})")
    fig.tight_layout()
    save_figure(fig, output_dir / "universality_across_datasets", formats=("pdf",))


def plot_rank_rank_scatter(ds_a: str, ds_b: str, ranks: list[int], output_dir: Path) -> None:
    """Ordinal rank scatter across datasets, one panel per NMF rank."""
    n = len(ranks)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.4), squeeze=False)
    axes = axes.ravel()

    for i, (ax, rank) in enumerate(zip(axes, ranks)):
        ma, mb = _model_means(rank, ds_a), _model_means(rank, ds_b)
        merged = pd.concat([ma, mb], axis=1).dropna()
        ra = merged[ds_a].rank(ascending=False)
        rb = merged[ds_b].rank(ascending=False)
        nm = len(merged)
        rho, _ = spearmanr(ra, rb)

        ax.scatter(ra, rb, s=10, alpha=0.5, color=GRAY_DARK, edgecolors="none")
        ax.plot([1, nm], [1, nm], ls="--", color=GRAY_LIGHT, lw=0.8, zorder=0)
        ax.set(xlim=(0, nm + 1), ylim=(0, nm + 1), aspect="equal")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title(f"Rank {rank}", fontsize=9)
        ax.text(
            0.05, 0.05, f"ρ = {rho:.2f}",
            transform=ax.transAxes, va="bottom", fontsize=7,
        )
        if i > 0:
            ax.set_yticklabels([])

    axes[0].set_ylabel(f"Model rank ({ds_b})")
    axes[n // 2].set_xlabel(f"Model rank ({ds_a})")
    fig.tight_layout()
    save_figure(fig, output_dir / "model_ordering", formats=("pdf",))


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup_plotting()

    ds_a, ds_b = cfg.dataset, "objectnet"
    ranks = _available_ranks(ds_a, ds_b, [10, 30, 50, 100, 200])
    if not ranks:
        raise SystemExit(f"No rank has universality tables for both {ds_a} and {ds_b}")
    log.info(f"Ranks with data for both datasets: {ranks}")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    plot_model_scatter(ds_a, ds_b, ranks, out)
    plot_rank_rank_scatter(ds_a, ds_b, ranks, out)

    log.info(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
