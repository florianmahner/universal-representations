#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.colors import CYAN, GRAY_LIGHT, ROSE, TEAL
from src.plotting import create_figure, save_figure, setup_plotting


def metric_style() -> dict[str, tuple[str, str]]:
    """The compared universality metrics, in plotting order, with label and colour."""
    return {
        "max_corr": ("Max |corr|", CYAN),
        "cos2_perm": ("Cos² perm (raw)", TEAL),
        "cos2_perm_adj": ("Cos² perm (adjusted)", ROSE),
    }


def load_all_metrics(rank: int, dataset: str, metrics: list[str]) -> pd.DataFrame:
    """Load all metrics for one rank into wide format (one column per metric)."""
    path = (
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    df = pd.read_csv(path)
    available = [m for m in metrics if m in df["metric"].unique()]
    wide = df[df["metric"].isin(available)].pivot_table(
        index=["model", "dimension"], columns="metric", values="universality",
    ).reset_index()
    wide.columns.name = None
    return wide


def _scatter(ax, x, y, color):
    ax.scatter(x, y, s=2, alpha=0.25, color=color, rasterized=True)
    rho, _ = spearmanr(x, y)
    lo = min(x.min(), y.min()) - 0.02
    hi = max(x.max(), y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color=GRAY_LIGHT, linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(f"ρ = {rho:.3f}", fontsize=9)


def plot_metric_comparison(
    wide: pd.DataFrame,
    style: dict[str, tuple[str, str]],
    output_dir: Path,
):
    """Three-panel: distributions | raw vs adjusted | max_corr vs adjusted."""
    setup_plotting()
    available = [m for m in style if m in wide.columns]
    labels = {metric: style[metric][0] for metric in style}

    fig, axes = create_figure("single", ncols=3)

    ax = axes[0]
    df_long = wide[available].melt(var_name="metric", value_name="universality")
    df_long["metric"] = df_long["metric"].map(labels)
    palette = [style[m][1] for m in available]
    sns.violinplot(
        data=df_long, x="metric", y="universality", hue="metric",
        palette=palette, inner="box", ax=ax, cut=0, legend=False,
    )
    ax.set(xlabel="", ylabel="Universality")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    ax = axes[1]
    _scatter(ax, wide["cos2_perm"], wide["cos2_perm_adj"], ROSE)
    ax.set_xlabel(labels["cos2_perm"])
    ax.set_ylabel(labels["cos2_perm_adj"])

    ax = axes[2]
    _scatter(ax, wide["max_corr"], wide["cos2_perm_adj"], ROSE)
    ax.set_xlabel(labels["max_corr"])
    ax.set_ylabel(labels["cos2_perm_adj"])

    fig.tight_layout()
    save_figure(fig, output_dir / "metric_comparison", formats=("pdf",))
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    style = metric_style()
    wide = load_all_metrics(cfg.factorization.rank, cfg.dataset, list(style))
    plot_metric_comparison(wide, style, out)

    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
