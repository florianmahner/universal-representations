from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.data import ThingsDataset
from src.colors import CYCLE, GRAY_DARK, GRAY_PALE, INDIGO, PURPLE, ROSE, SAND
from src.plotting import setup_plotting, save_figure

log = logging.getLogger(__name__)



# Analysis 1: Noise vs Signal


def plot_noise_vs_signal(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    figures_dir: Path,
) -> None:
    """Scatter of universality vs reconstruction importance (clipped) for all dims."""
    setup_plotting()

    y_col = "delta_r2_clipped" if "delta_r2_clipped" in df.columns else "delta_r2"

    fig, ax = plt.subplots(figsize=(8, 6))

    classes = sorted(df["model_class"].dropna().unique())
    for i, mc in enumerate(classes):
        mask = df["model_class"] == mc
        ax.scatter(
            df.loc[mask, "universality"].values,
            df.loc[mask, y_col].values,
            alpha=0.25, s=10, label=mc,
            color=CYCLE[i % len(CYCLE)],
            rasterized=True,
        )

    ax.axvline(thresholds["universality"], ls="--", color=GRAY_DARK, alpha=0.4, lw=0.8)
    ax.axhline(thresholds["importance"], ls="--", color=GRAY_DARK, alpha=0.4, lw=0.8)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    label_kw = dict(fontsize=8, alpha=0.5, ha="center", va="center")
    ut = thresholds["universality"]
    it = thresholds["importance"]
    ax.text(ut / 2, ylim[1] * 0.9, "idiosyncratic\nimportant", **label_kw)
    ax.text((ut + xlim[1]) / 2, ylim[1] * 0.9, "universal\nimportant", **label_kw)
    ax.text(ut / 2, it / 3, "idiosyncratic\nnoise", **label_kw)
    ax.text((ut + xlim[1]) / 2, it / 3, "universal\nunimportant", **label_kw)

    # Correlation annotation (on clipped values)
    r, p = sp_stats.spearmanr(df["universality"], df[y_col])
    ax.annotate(
        f"Spearman r = {r:.3f}, p = {p:.1e}",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color=GRAY_DARK,
    )

    ax.set_xlabel("Universality (mean max-correlation)")
    ax.set_ylabel("Reconstruction importance (delta R², clipped at 1.0)")
    ax.legend(title="Architecture", markerscale=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir / "noise_vs_signal", formats=("pdf",))


def plot_noise_vs_signal_log(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    figures_dir: Path,
) -> None:
    """Scatter of universality vs log-scale reconstruction importance (raw, unclipped)."""
    setup_plotting()

    fig, ax = plt.subplots(figsize=(8, 6))

    classes = sorted(df["model_class"].dropna().unique())
    for i, mc in enumerate(classes):
        mask = df["model_class"] == mc
        ax.scatter(
            df.loc[mask, "universality"].values,
            df.loc[mask, "delta_r2"].values,
            alpha=0.25, s=10, label=mc,
            color=CYCLE[i % len(CYCLE)],
            rasterized=True,
        )

    ax.axvline(thresholds["universality"], ls="--", color=GRAY_DARK, alpha=0.4, lw=0.8)
    ax.axhline(thresholds["importance"], ls="--", color=GRAY_DARK, alpha=0.4, lw=0.8)
    ax.set_yscale("symlog", linthresh=0.01)

    r, p = sp_stats.spearmanr(df["universality"], df["delta_r2"])
    ax.annotate(
        f"Spearman r = {r:.3f}, p = {p:.1e}",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9, color=GRAY_DARK,
    )

    ax.set_xlabel("Universality (mean max-correlation)")
    ax.set_ylabel("Reconstruction importance (delta R², symlog scale)")
    ax.legend(title="Architecture", markerscale=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir / "noise_vs_signal_log", formats=("pdf",))


def plot_noise_vs_signal_by_task(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    figures_dir: Path,
    n_tasks: int = 8,
) -> None:
    """Faceted scatter of universality vs importance by training task."""
    setup_plotting()

    y_col = "delta_r2_clipped" if "delta_r2_clipped" in df.columns else "delta_r2"

    task_counts = df["train_task"].value_counts()
    top_tasks = task_counts.head(n_tasks).index.tolist()
    df_sub = df[df["train_task"].isin(top_tasks)]

    ncols = 4
    nrows = (len(top_tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, task in enumerate(top_tasks):
        ax = axes[idx]
        mask = df_sub["train_task"] == task
        n = int(mask.sum())
        ax.scatter(
            df_sub.loc[mask, "universality"].values,
            df_sub.loc[mask, y_col].values,
            alpha=0.3, s=8, color=CYCLE[idx % len(CYCLE)],
            rasterized=True,
        )
        ax.axvline(thresholds["universality"], ls="--", color=GRAY_DARK, alpha=0.3, lw=0.6)
        ax.axhline(thresholds["importance"], ls="--", color=GRAY_DARK, alpha=0.3, lw=0.6)
        ax.set_title(f"{task} (n={n})", fontsize=9)

    for idx in range(len(top_tasks), len(axes)):
        axes[idx].set_visible(False)

    # Shared axis labels (compat with matplotlib < 3.4)
    fig.text(0.5, 0.01, "Universality", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "Delta R² (clipped at 1.0)", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    save_figure(fig, figures_dir / "noise_vs_signal_by_task", formats=("pdf",))


def plot_quadrant_composition(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Stacked bar of quadrant membership counts by model_class."""
    setup_plotting()

    quadrant_order = [
        "universal-important", "universal-unimportant",
        "idiosyncratic-important", "idiosyncratic-noise",
    ]
    quadrant_colors = [INDIGO, ROSE, SAND, GRAY_PALE]

    counts = df.groupby(["model_class", "quadrant"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=quadrant_order, fill_value=0)
    counts = counts.sort_values(quadrant_order[0], ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(counts))

    for q, color in zip(quadrant_order, quadrant_colors):
        vals = counts[q].values
        ax.barh(range(len(counts)), vals, left=bottom, label=q, color=color, alpha=0.85, height=0.6)
        bottom += vals

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Number of dimensions")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, figures_dir / "quadrant_composition", formats=("pdf",))


# Analysis 2: Task-Specificity


def plot_idiosyncratic_by_group(
    df: pd.DataFrame,
    group_col: str,
    figures_dir: Path,
    min_count: int = 20,
) -> None:
    """Box plot of delta_r2 for idiosyncratic dims, grouped by a metadata column."""
    setup_plotting()

    df_idio = df[df["quadrant"].str.startswith("idiosyncratic")]
    group_counts = df_idio[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_count].index.tolist()
    df_plot = df_idio[df_idio[group_col].isin(valid_groups)]

    if len(valid_groups) == 0:
        log.warning(f"No groups with >= {min_count} idiosyncratic dims for {group_col}")
        return

    # Sort by median delta_r2
    medians = df_plot.groupby(group_col)["delta_r2"].median().sort_values(ascending=False)
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(order) + 1)))

    box_data = [df_plot.loc[df_plot[group_col] == g, "delta_r2"].values for g in order]
    bp = ax.boxplot(
        box_data, vert=False, patch_artist=True, widths=0.6,
        labels=[f"{g} (n={len(d)})" for g, d in zip(order, box_data)],
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(SAND)
        patch.set_alpha(0.6)

    ax.set_xlabel("Reconstruction importance (LOO delta R-squared)")
    ax.set_title(f"Idiosyncratic dimensions by {group_col}")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"idiosyncratic_by_{group_col}", formats=("pdf",))


def plot_top_idiosyncratic_images(
    df: pd.DataFrame,
    figures_dir: Path,
    n_show: int = 20,
    n_images: int = 8,
) -> None:
    """Image grids for the top most-important idiosyncratic dimensions."""
    setup_plotting()

    ds = ThingsDataset()
    categories = np.array(ds.categories)

    df_idio = df[df["quadrant"] == "idiosyncratic-important"]
    df_top = df_idio.nlargest(n_show, "delta_r2")

    if len(df_top) == 0:
        log.warning("No idiosyncratic-important dimensions found")
        return

    from experiments.universality.shared import load_components_for_rank
    components = load_components_for_rank(rank=50)

    fig, axes = plt.subplots(len(df_top), n_images, figsize=(1.8 * n_images, 1.8 * len(df_top)))
    if len(df_top) == 1:
        axes = axes[np.newaxis, :]

    for row, (_, dim_row) in enumerate(df_top.iterrows()):
        model = dim_row["model"]
        d = int(dim_row["dimension"])

        if model not in components:
            continue

        W = components[model]
        top_idx = np.argsort(-W[:, d])[:n_images]

        for col in range(n_images):
            ax = axes[row, col]
            try:
                ax.imshow(ds.get_image(top_idx[col]))
            except Exception:
                pass
            ax.axis("off")

            if col == 0:
                short_model = model.split("_")[0] if "_" in model else model[:12]
                ax.set_title(
                    f"{short_model} d{d}\nuniv={dim_row['universality']:.2f} "
                    f"dR2={dim_row['delta_r2']:.3f}",
                    fontsize=6, loc="left",
                )
            else:
                cat = categories[top_idx[col]]
                ax.set_title(cat, fontsize=5)

    fig.suptitle("Top idiosyncratic-important dimensions", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save_figure(fig, figures_dir / "top_idiosyncratic_images", formats=("pdf",))


# Analysis 3: Correlation Profiles


def plot_correlation_profiles(
    profiles: np.ndarray,
    models: list[str],
    df: pd.DataFrame,
    profile_stats: pd.DataFrame,
    figures_dir: Path,
    n_examples: int = 12,
) -> None:
    """Histogram panel showing example correlation profiles by type."""
    setup_plotting()

    # Merge profile stats with dimension classification
    merged = profile_stats.merge(
        df[["model", "dimension", "quadrant", "universality", "delta_r2"]],
        on=["model", "dimension"],
        how="left",
    )
    model_to_idx = {m: i for i, m in enumerate(models)}

    # Pick examples: 4 universal, 4 flat-idiosyncratic, 4 bimodal/subgroup
    examples = []
    for profile_type, quadrant_prefix, n in [
        ("widespread", "universal", 4),
        ("flat", "idiosyncratic", 4),
        ("bimodal", "idiosyncratic", 4),
    ]:
        candidates = merged[
            (merged["profile_type"] == profile_type)
            & merged["quadrant"].str.startswith(quadrant_prefix)
        ]
        if len(candidates) == 0:
            # Fallback: any dim with this profile type
            candidates = merged[merged["profile_type"] == profile_type]
        if len(candidates) == 0:
            candidates = merged[merged["quadrant"].str.startswith(quadrant_prefix)]
        # Pick highest-importance examples for visual interest
        picked = candidates.nlargest(n, "delta_r2")
        for _, row in picked.iterrows():
            examples.append(row)

    n_examples = len(examples)
    if n_examples == 0:
        log.warning("No examples found for correlation profiles")
        return

    ncols = 4
    nrows = (n_examples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    type_colors = {
        "widespread": INDIGO,
        "flat": PURPLE,
        "bimodal": SAND,
        "mixed": ROSE,
    }

    for idx, row in enumerate(examples):
        if idx >= len(axes):
            break
        ax = axes[idx]
        mi = model_to_idx.get(row["model"])
        if mi is None:
            continue
        d = int(row["dimension"])
        corrs = profiles[mi, d, :]
        valid = corrs[~np.isnan(corrs)]

        color = type_colors.get(row.get("profile_type", "mixed"), ROSE)
        ax.hist(valid, bins=30, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(valid.mean(), color=GRAY_DARK, ls="--", lw=1)
        ax.set_xlim(0, 1)

        short_model = row["model"].split("_")[0] if "_" in row["model"] else row["model"][:10]
        ax.set_title(
            f"{short_model} d{d} ({row.get('profile_type', '?')})\n"
            f"univ={row['universality']:.2f}, n_high={int(row.get('n_high', 0))}",
            fontsize=7,
        )
        ax.tick_params(labelsize=7)

    for idx in range(n_examples, len(axes)):
        axes[idx].set_visible(False)

    fig.text(0.5, 0.01, "Max correlation with other models", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "Count", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout()
    save_figure(fig, figures_dir / "correlation_profiles", formats=("pdf",))


def plot_profile_classification(
    profile_stats: pd.DataFrame,
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Scatter of mean correlation vs n_high_matches, colored by quadrant."""
    setup_plotting()

    merged = profile_stats.merge(
        df[["model", "dimension", "quadrant"]],
        on=["model", "dimension"],
        how="left",
    )

    quadrant_colors = {
        "universal-important": INDIGO,
        "universal-unimportant": ROSE,
        "idiosyncratic-important": SAND,
        "idiosyncratic-noise": GRAY_PALE,
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for quadrant, color in quadrant_colors.items():
        mask = merged["quadrant"] == quadrant
        ax.scatter(
            merged.loc[mask, "profile_mean"].values,
            merged.loc[mask, "n_high"].values,
            alpha=0.25, s=10, color=color, label=quadrant,
            rasterized=True,
        )

    ax.set_xlabel("Mean max-correlation (universality)")
    ax.set_ylabel("Number of models with correlation > 0.5")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures_dir / "profile_classification", formats=("pdf",))


def plot_subgroup_sharing(
    sharing_df: pd.DataFrame,
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Summarize what taxonomy groups share idiosyncratic dimensions."""
    setup_plotting()

    if len(sharing_df) == 0:
        log.warning("No subgroup-shared dimensions to plot")
        return

    # Filter to idiosyncratic dims only
    idio_sharing = sharing_df.merge(
        df[["model", "dimension", "quadrant"]],
        on=["model", "dimension"],
        how="left",
    )
    idio_sharing = idio_sharing.dropna(subset=["quadrant"])
    idio_sharing = idio_sharing[idio_sharing["quadrant"].str.startswith("idiosyncratic")]

    if len(idio_sharing) == 0:
        log.warning("No idiosyncratic subgroup-shared dimensions")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of n_high_matches for idiosyncratic dims
    ax = axes[0]
    ax.hist(
        idio_sharing["n_high_matches"].values,
        bins=30, color=SAND, alpha=0.7, edgecolor="white",
    )
    ax.set_xlabel("Number of models sharing dimension (corr > 0.5)")
    ax.set_ylabel("Count of idiosyncratic dimensions")
    ax.set_title("Sharing breadth")

    # Right: source_task distribution of subgroup-shared dims
    ax = axes[1]
    task_counts = idio_sharing["source_task"].value_counts().head(10)
    ax.barh(
        range(len(task_counts)),
        task_counts.values,
        color=SAND, alpha=0.7, height=0.6,
    )
    ax.set_yticks(range(len(task_counts)))
    ax.set_yticklabels(task_counts.index, fontsize=9)
    ax.set_xlabel("Number of subgroup-shared idiosyncratic dimensions")
    ax.set_title("Source training task")
    ax.invert_yaxis()

    fig.suptitle("Subgroup sharing among idiosyncratic dimensions", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / "subgroup_sharing", formats=("pdf",))


# Analysis 4: Within-Model Structure


def plot_within_model_reconstruction(
    within_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Within-model structure: norm fractions and inter-correlation."""
    setup_plotting()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    classes = sorted(within_df["model_class"].dropna().unique())

    # Panel 1: fraction of dims vs norm fraction in idiosyncratic subspace
    ax = axes[0]
    for i, mc in enumerate(classes):
        mask = within_df["model_class"] == mc
        ax.scatter(
            within_df.loc[mask, "frac_idiosyncratic"].values,
            within_df.loc[mask, "idio_norm_fraction"].values,
            alpha=0.6, s=30, label=mc,
            color=CYCLE[i % len(CYCLE)],
        )
    # Diagonal reference (norm proportional to dim count)
    ax.plot([0, 1], [0, 1], ls=":", color=GRAY_DARK, alpha=0.4, lw=1)
    ax.set_xlabel("Fraction of idiosyncratic dimensions")
    ax.set_ylabel("Fraction of total norm in idiosyncratic subspace")
    ax.set_title("Norm allocation")
    ax.legend(title="Architecture", fontsize=7, markerscale=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Panel 2: mean inter-correlation among idiosyncratic dims
    ax = axes[1]
    valid = within_df.dropna(subset=["mean_idio_inter_correlation"])
    for i, mc in enumerate(classes):
        mask = valid["model_class"] == mc
        ax.scatter(
            valid.loc[mask, "frac_idiosyncratic"].values,
            valid.loc[mask, "mean_idio_inter_correlation"].values,
            alpha=0.6, s=30,
            color=CYCLE[i % len(CYCLE)],
        )
    ax.set_xlabel("Fraction of idiosyncratic dimensions")
    ax.set_ylabel("Mean |correlation| among idiosyncratic dims")
    ax.set_title("Idiosyncratic subspace coherence")

    # Panel 3: norm fraction vs mean universality
    ax = axes[2]
    for i, mc in enumerate(classes):
        mask = within_df["model_class"] == mc
        ax.scatter(
            within_df.loc[mask, "mean_universality"].values,
            within_df.loc[mask, "idio_norm_fraction"].values,
            alpha=0.6, s=30,
            color=CYCLE[i % len(CYCLE)],
        )
    ax.set_xlabel("Mean universality across all dims")
    ax.set_ylabel("Fraction of total norm in idiosyncratic subspace")
    ax.set_title("Model universality vs idiosyncratic norm")

    fig.tight_layout()
    save_figure(fig, figures_dir / "within_model_reconstruction", formats=("pdf",))


def plot_idiosyncratic_fraction(
    within_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Bar chart of idiosyncratic fraction by model_class and train_task."""
    setup_plotting()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, col in enumerate(["model_class", "train_task"]):
        ax = axes[idx]
        group_means = within_df.groupby(col)["frac_idiosyncratic"].agg(["mean", "std", "count"])
        group_means = group_means[group_means["count"] >= 3].sort_values("mean", ascending=False)

        y = range(len(group_means))
        ax.barh(
            y, group_means["mean"].values,
            xerr=group_means["std"].values,
            color=SAND, alpha=0.7, height=0.6,
            capsize=2,
        )
        labels = [f"{name} (n={int(row['count'])})" for name, row in group_means.iterrows()]
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Fraction of idiosyncratic dimensions")
        ax.set_title(f"By {col}")
        ax.invert_yaxis()

    fig.suptitle("Idiosyncratic dimension prevalence", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figures_dir / "idiosyncratic_fraction", formats=("pdf",))
