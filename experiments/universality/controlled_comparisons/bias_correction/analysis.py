#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats

from src.colors import CYCLE, GRAY_DARK
from src.plotting import setup_plotting, save_figure


def load_decomposed(path: Path) -> pd.DataFrame:
    """Load the precomputed within/between decomposition."""
    return pd.read_csv(path)


def plot_decomposition_bars(df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing within vs between universality by architecture."""
    setup_plotting()
    families = ["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"]
    families = [f for f in families if f in df["model_class"].values]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("universality_within", "Within-Family"),
        ("universality_between", "Between-Family"),
        ("universality_overall", "Original (Mixed)"),
    ]
    colors = [CYCLE[0], CYCLE[1], GRAY_DARK]

    for ax, (col, label), color in zip(axes, metrics, colors):
        means = [df[df["model_class"] == f][col].mean() for f in families]
        stds = [df[df["model_class"] == f][col].std() for f in families]
        ns = [len(df[df["model_class"] == f]) for f in families]
        sems = [s / np.sqrt(n) for s, n in zip(stds, ns)]

        x = range(len(families))
        ax.bar(x, means, yerr=sems, capsize=4, color=color, edgecolor="white",
               alpha=0.8, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(families, ns)], fontsize=8)
        ax.set_ylabel("Universality")
        ax.set_title(label, fontsize=11)

        groups = [df[df["model_class"] == f][col].dropna().values for f in families]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            fstat, p = stats.f_oneway(*groups)
            p_str = f"p={p:.4f}" if p >= 0.0001 else "p<.0001"
            ax.set_xlabel(f"F={fstat:.1f}, {p_str}", fontsize=9)

    fig.suptitle("Universality Decomposition: Within-Family vs. Between-Family",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_gap_analysis(df: pd.DataFrame) -> plt.Figure:
    """Show the within-between gap and pool composition for each family."""
    setup_plotting()
    families = ["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"]
    families = [f for f in families if f in df["model_class"].values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: within vs between per family (paired)
    ax = axes[0]
    x = np.arange(len(families))
    w = 0.35
    within_means = [df[df["model_class"] == f]["universality_within"].mean() for f in families]
    between_means = [df[df["model_class"] == f]["universality_between"].mean() for f in families]
    within_sems = [df[df["model_class"] == f]["universality_within"].sem() for f in families]
    between_sems = [df[df["model_class"] == f]["universality_between"].sem() for f in families]

    ax.bar(x - w/2, within_means, w, yerr=within_sems, capsize=3,
           label="Within-family", color=CYCLE[0], edgecolor="white")
    ax.bar(x + w/2, between_means, w, yerr=between_sems, capsize=3,
           label="Between-family", color=CYCLE[1], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel("Universality")
    ax.set_title("Within- vs. Between-Family Universality")
    ax.legend(fontsize=8)

    # Right: pool composition
    ax = axes[1]
    within_fracs = []
    for f in families:
        sub = df[df["model_class"] == f]
        n_w = sub["n_within"].iloc[0] if len(sub) > 0 else 0
        n_b = sub["n_between"].iloc[0] if len(sub) > 0 else 0
        within_fracs.append(n_w / (n_w + n_b) if (n_w + n_b) > 0 else 0)

    ax.bar(x, within_fracs, color=CYCLE[0], edgecolor="white",
           label="Within-family", alpha=0.8)
    ax.bar(x, [1 - f for f in within_fracs], bottom=within_fracs,
           color=CYCLE[1], edgecolor="white",
           label="Between-family", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel("Fraction of Comparison Pool")
    ax.set_title("Composition of Comparison Pool")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    for i, frac in enumerate(within_fracs):
        ax.text(i, frac / 2, f"{frac:.0%}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(i, frac + (1 - frac) / 2, f"{1-frac:.0%}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    fig.tight_layout()
    return fig


def plot_architecture_corrected(df: pd.DataFrame, data_root: Path) -> plt.Figure:
    """Side-by-side comparison of architecture effect with different metrics."""
    setup_plotting()

    contrasts = pd.read_csv(data_root / "meta" / "things22k" / "model_metadata.csv")
    ms_map = dict(zip(contrasts["feature_uid"].astype(str), contrasts["model_string"]))
    df = df.copy()
    df["model_string"] = df["model"].map(ms_map)

    arch_models = contrasts[contrasts["compare_architecture"].notna()][
        ["model_string", "compare_architecture"]
    ]
    merged = df.merge(arch_models, on="model_string", how="inner")

    metrics = [
        ("universality_overall", "Original"),
        ("universality_between", "Between-Family\n(corrected)"),
        ("universality_within", "Within-Family"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    order = ["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"]
    order = [o for o in order if o in merged["compare_architecture"].values]
    palette = {o: CYCLE[i] for i, o in enumerate(order)}

    for ax, (col, label) in zip(axes, metrics):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.violinplot(
                data=merged, x="compare_architecture", y=col, order=order,
                hue="compare_architecture", palette=palette, legend=False,
                inner=None, alpha=0.3, cut=0, ax=ax,
            )
            sns.stripplot(
                data=merged, x="compare_architecture", y=col, order=order,
                color="black", alpha=0.5, size=4, jitter=0.15, ax=ax,
            )
            sns.boxplot(
                data=merged, x="compare_architecture", y=col, order=order,
                hue="compare_architecture", palette=palette, legend=False,
                width=0.3, boxprops=dict(alpha=0.7), fliersize=0,
                whiskerprops=dict(alpha=0.7), capprops=dict(alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5), ax=ax,
            )

        groups = [merged[merged["compare_architecture"] == g][col].dropna().values
                  for g in order if len(merged[merged["compare_architecture"] == g]) > 1]
        if len(groups) >= 2:
            fstat, p = stats.f_oneway(*groups)
            p_str = f"p={p:.4f}" if p >= 0.0001 else "p<.0001"
            ax.set_xlabel(f"F={fstat:.1f}, {p_str}", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Universality" if ax == axes[0] else "")
        labels = [f"{g}\n(n={len(merged[merged['compare_architecture']==g])})" for g in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8)

    fig.suptitle(
        "Architecture Effect: Original vs. Bias-Corrected Universality",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    setup_plotting()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    df = load_decomposed(out / "universality_decomposed.csv")
    print(f"Loaded decomposed universality for {len(df)} models")

    fig = plot_decomposition_bars(df)
    save_figure(fig, out / "bias_decomposition_bars", formats=("pdf",))

    fig = plot_gap_analysis(df)
    save_figure(fig, out / "bias_gap_analysis", formats=("pdf",))

    fig = plot_architecture_corrected(df, Path(cfg.data_root))
    save_figure(fig, out / "bias_architecture_corrected", formats=("pdf",))

    print("Bias correction plots saved to", out)


if __name__ == "__main__":
    main()
