from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns

from src.colors import (
    CYCLE,
    GRAY,
    GRAY_DARK,
    GRAY_LIGHT,
    GREEN,
    INDIGO,
    PURPLE,
    WINE,
)
from src.plotting import create_figure, save_figure

PALETTE_CATEGORICAL = CYCLE + [GREEN, WINE]


def _group_order(df: pd.DataFrame) -> list[str]:
    """Sort groups by mean specificity descending."""
    model_means = df.groupby(["model", "group"])["specificity"].mean().reset_index()
    order = (
        model_means.groupby("group")["specificity"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return order


def _group_palette(groups: list[str]) -> dict[str, str]:
    """Assign colors from PALETTE_CATEGORICAL to groups."""
    return {g: PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)] for i, g in enumerate(groups)}


def _make_group_labels(df: pd.DataFrame, groups: list[str]) -> dict[str, str]:
    """Create x-axis labels with group size: 'GroupName\\n(n=N)'."""
    counts = df.groupby("group")["model"].nunique()
    return {g: f"{g}\n(n={counts.get(g, 0)})" for g in groups}


def plot_specificity_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    grouping_name: str = "",
) -> None:
    """Violin + strip + box of model-level specificity by group."""
    model_means = df.groupby(["model", "group"])["specificity"].mean().reset_index()
    order = _group_order(df)
    palette = _group_palette(order)
    labels = _make_group_labels(df, order)
    model_means["label"] = model_means["group"].map(labels)
    label_order = [labels[g] for g in order]

    size = "full_width" if len(order) > 5 else "wide"
    fig, ax = create_figure(size)

    sns.violinplot(
        data=model_means, x="label", y="specificity", hue="label",
        order=label_order, hue_order=label_order,
        palette=[palette[g] for g in order],
        inner=None, alpha=0.3, cut=0, legend=False, ax=ax,
    )
    sns.stripplot(
        data=model_means, x="label", y="specificity", hue="label",
        order=label_order, hue_order=label_order,
        palette=[palette[g] for g in order],
        alpha=0.5, size=3, jitter=0.15, legend=False, ax=ax,
    )
    sns.boxplot(
        data=model_means, x="label", y="specificity", order=label_order,
        color=GRAY_LIGHT, width=0.3, fliersize=0,
        boxprops=dict(alpha=0.7), medianprops=dict(color=GRAY_DARK),
        whiskerprops=dict(color=GRAY),
        capprops=dict(color=GRAY), ax=ax,
    )

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Specificity (within \u2212 between)")
    if grouping_name:
        ax.set_title(grouping_name, fontweight="normal")

    if len(order) > 6:
        ax.tick_params(axis="x", rotation=45)

    save_figure(fig, output_dir / "specificity_distribution", formats=("pdf", "png"))


def plot_within_vs_between_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    grouping_name: str = "",
) -> None:
    """Scatter of model-level within vs between, colored by group."""
    model_means = df.groupby(["model", "group"]).agg(
        within=("within", "mean"),
        between=("between", "mean"),
    ).reset_index()

    order = _group_order(df)
    palette = _group_palette(order)

    fig, ax = create_figure("square")

    for group in order:
        gdf = model_means[model_means["group"] == group]
        ax.scatter(
            gdf["between"].values, gdf["within"].values,
            c=palette[group], label=group, s=20, alpha=0.7, edgecolors="none",
        )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Between-group mean max-corr")
    ax.set_ylabel("Within-group mean max-corr")
    ax.legend(fontsize=7, loc="lower right")

    if grouping_name:
        ax.set_title(grouping_name, fontweight="normal")

    save_figure(fig, output_dir / "within_vs_between", formats=("pdf", "png"))


def plot_dimension_profiles(
    df: pd.DataFrame,
    output_dir: Path,
    top_n_groups: int = 6,
) -> None:
    """Per-group bar chart of dimension-level specificity (sorted)."""
    order = _group_order(df)[:top_n_groups]

    for group in order:
        gdf = df[df["group"] == group]
        dim_means = gdf.groupby("dimension")["specificity"].mean().sort_values(ascending=False)

        fig, ax = create_figure("wide")
        colors = [
            INDIGO if v >= 0 else PURPLE
            for v in dim_means.values
        ]
        ax.bar(range(len(dim_means)), dim_means.values, color=colors, alpha=0.7, width=0.8)
        ax.axhline(0, color=GRAY, linewidth=0.6)
        ax.set_xlabel("Dimension (sorted)")
        ax.set_ylabel("Mean specificity")
        n_models = gdf["model"].nunique()
        ax.set_title(f"{group} (n={n_models})", fontweight="normal")

        save_figure(fig, output_dir / f"dimension_profile_{group}", formats=("pdf", "png"))


def plot_rank_curves(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Group-mean specificity vs NMF rank with SEM error bands."""
    model_means = df.groupby(["model", "group", "rank"])["specificity"].mean().reset_index()
    group_stats = model_means.groupby(["group", "rank"]).agg(
        mean=("specificity", "mean"),
        sem=("specificity", "sem"),
    ).reset_index()

    order = _group_order(df[df["rank"] == df["rank"].min()])
    palette = _group_palette(order)

    fig, ax = create_figure("wide")

    for group in order:
        gdf = group_stats[group_stats["group"] == group].sort_values("rank")
        ranks = gdf["rank"].values
        means = gdf["mean"].values
        sems = gdf["sem"].values

        ax.plot(ranks, means, color=palette[group], label=group, marker="o", markersize=4)
        ax.fill_between(
            ranks, means - sems, means + sems,
            color=palette[group], alpha=0.15,
        )

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("NMF rank")
    ax.set_ylabel("Mean specificity")
    ax.legend(fontsize=7)

    save_figure(fig, output_dir / "specificity_vs_rank", formats=("pdf", "png"))


def plot_permutation_results(
    perm_df: pd.DataFrame,
    output_dir: Path,
    grouping_name: str = "",
) -> None:
    """Horizontal bar chart of observed specificity with significance markers."""
    perm_df = perm_df.sort_values("observed", ascending=True)

    fig, ax = create_figure("wide")

    colors = [
        INDIGO if row["p_value_fdr"] < 0.05 else GRAY
        for _, row in perm_df.iterrows()
    ]
    ax.barh(range(len(perm_df)), perm_df["observed"].values, color=colors, alpha=0.7)

    for i, (_, row) in enumerate(perm_df.iterrows()):
        if row["p_value_fdr"] < 0.001:
            marker = "***"
        elif row["p_value_fdr"] < 0.01:
            marker = "**"
        elif row["p_value_fdr"] < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            x_pos = row["observed"] + 0.002
            ax.text(x_pos, i, marker, va="center", fontsize=8)

    ax.set_yticks(range(len(perm_df)))
    ax.set_yticklabels(perm_df["group"].values)
    ax.axvline(0, color=GRAY, linewidth=0.6)
    ax.set_xlabel("Mean specificity")
    if grouping_name:
        ax.set_title(grouping_name, fontweight="normal")

    save_figure(fig, output_dir / "permutation_test", formats=("pdf", "png"))
