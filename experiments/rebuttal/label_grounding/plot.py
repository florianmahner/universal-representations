from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.colors import GRAY, GRAY_DARK, PURPLE, ROSE, TEAL
from src.plotting import save_figure, setup_plotting

# The four objective reference measures, in figure order, each with the kind of
# structure it carries (semantic vs low-level) and its axis label.
REFERENCE_SETS = {
    "bert": ("semantic", "BERT (word)"),
    "wordnet": ("semantic", "WordNet"),
    "pixel": ("low-level", "pixel stats"),
    "conv1": ("low-level", "untrained conv1"),
}


def _violin_by_label(ax, df: pd.DataFrame, col: str, ylabel: str) -> None:
    label_colors = {"semantic": ROSE, "both": PURPLE, "visual": TEAL, "neither": GRAY}
    order = list(label_colors)
    sns.violinplot(
        data=df, x="majority_label", y=col, order=order,
        hue="majority_label", palette=label_colors, legend=False,
        inner=None, cut=0, linewidth=0.5, saturation=1, ax=ax,
    )
    for coll in ax.collections:
        coll.set_alpha(0.3)
    sns.boxplot(
        data=df, x="majority_label", y=col, order=order,
        width=0.18, showfliers=False, ax=ax,
        boxprops={"facecolor": "white", "linewidth": 0.6},
        whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
        medianprops={"linewidth": 1.0, "color": GRAY_DARK},
    )
    sns.stripplot(
        data=df, x="majority_label", y=col, order=order,
        hue="majority_label", palette=label_colors, legend=False,
        size=1.1, alpha=0.35, jitter=0.25, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    lo = np.nanpercentile(df[col], 0.5)
    ax.set_ylim(bottom=min(-0.05, lo - 0.02))
    ax.tick_params(axis="x", rotation=20)


def plot_taxonomy_violins(
    rated: pd.DataFrame, out: Path, prefix: str = "r2_", suffix: str = ""
) -> None:
    """Q1a: per-measure R^2 split by human label — one row, four measures."""
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), constrained_layout=True)
    level = "image-level " if suffix else ""
    for ax, (name, (_, label)) in zip(axes, REFERENCE_SETS.items()):
        _violin_by_label(ax, rated, f"{prefix}{name}", f"{level}{label} $R^2$")
    save_figure(fig, out / f"taxonomy_grounding{suffix}", tight=False,
                formats=("pdf", "png"))


def _auc_panel(ax, per_set: dict, title: str) -> None:
    names = list(REFERENCE_SETS)
    xs = np.arange(len(names))
    w = 0.4
    for offset, key, color, lab in (
        (-w / 2, "auc_semantic_label", ROSE, "recover 'semantic' label"),
        (+w / 2, "auc_visual_label", TEAL, "recover 'visual' label"),
    ):
        aucs = [per_set[n][key]["auc"] for n in names]
        lo = [per_set[n][key]["auc"] - per_set[n][key]["ci_low"] for n in names]
        hi = [per_set[n][key]["ci_high"] - per_set[n][key]["auc"] for n in names]
        ax.bar(xs + offset, aucs, width=w, color=color, label=lab)
        ax.errorbar(xs + offset, aucs, yerr=[lo, hi], fmt="none",
                    ecolor=GRAY_DARK, linewidth=0.8, capsize=2)
    ax.axhline(0.5, color=GRAY_DARK, linewidth=0.7, linestyle="--")
    ax.text(len(names) - 0.5, 0.505, "chance", fontsize=6.5, va="bottom", ha="right",
            color=GRAY_DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels([REFERENCE_SETS[n][1] for n in names], fontsize=7.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title, fontsize=8)


def plot_label_recovery(stats: dict, out: Path) -> None:
    """Q1b: AUC for recovering each human label from each measure (both labels,
    so the specificity — off-target pairings sit at the chance line — is
    visible), at the concept level and, if available, the image level."""
    img = stats.get("image_level")
    n_panels = 2 if img else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 3.0),
                             constrained_layout=True, squeeze=False)
    _auc_panel(axes[0, 0], stats["per_set"], "concept level (12-image means)")
    if img:
        _auc_panel(axes[0, 1], img["per_set"],
                   "image level (held-out concepts)")
        axes[0, 1].set_ylabel("")
    axes[0, 0].set_ylabel("AUC (recover human label)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper center", ncol=2,
                      bbox_to_anchor=(0.5, 1.16), handletextpad=0.4,
                      columnspacing=1.2)
    save_figure(fig, out / "label_recovery_auc", tight=False, formats=("pdf", "png"))


def _binned_trend(ax, x: pd.Series, y: pd.Series, color: str, n_bins: int = 12) -> None:
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    rho, p = spearmanr(x, y)  # rank-based, on all points before clipping
    lo, hi = np.nanpercentile(y, [1, 99])
    ax.scatter(x, y, s=2, alpha=0.12, color=GRAY, linewidths=0, rasterized=True)
    bins = pd.qcut(x, n_bins, duplicates="drop")
    centers = x.groupby(bins, observed=True).mean()
    mean = y.groupby(bins, observed=True).mean()
    sem = y.groupby(bins, observed=True).sem()
    ax.errorbar(centers, mean, yerr=1.96 * sem, color=color,
                marker="o", markersize=3, linewidth=1.2, capsize=1.5)
    pad = 0.05 * (hi - lo)
    ax.set_ylim(min(lo, -pad) - pad, hi + pad)
    p_text = f"p = {p:.2g}" if p >= 1e-300 else "p < 1e-300"
    ax.text(0.05, 0.95, f"$\\rho$ = {rho:.2f}\n{p_text}",
            transform=ax.transAxes, va="top", fontsize=7.5)


def plot_universality(
    scores: pd.DataFrame, out: Path, prefix: str = "r2_", suffix: str = ""
) -> None:
    """Q2: universality vs objective predictability, per measure, no labels."""
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), constrained_layout=True)
    level = "image-level " if suffix else ""
    for ax, (name, (kind, label)) in zip(axes, REFERENCE_SETS.items()):
        color = ROSE if kind == "semantic" else TEAL
        _binned_trend(ax, scores["universality"], scores[f"{prefix}{name}"], color)
        ax.set_xlabel("universality")
        ax.set_ylabel(f"{level}{label} $R^2$")
        ax.set_title(f"{kind} · {label}", fontsize=8)
    save_figure(fig, out / f"universality_grounding{suffix}", tight=False,
                formats=("pdf", "png"))


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    del cfg  # the figures are fully specified by the tables run.py wrote
    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    scores = pd.read_csv(out / "grounding_scores.csv")
    rated = pd.read_csv(out / "rated_dims_scores.csv")
    stats = json.loads((out / "stats.json").read_text())
    plot_taxonomy_violins(rated, out)
    plot_label_recovery(stats, out)
    plot_universality(scores, out)
    if all(f"r2_img_{n}" in rated for n in REFERENCE_SETS):
        plot_taxonomy_violins(rated, out, prefix="r2_img_", suffix="_image")
        plot_universality(scores, out, prefix="r2_img_", suffix="_image")
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
