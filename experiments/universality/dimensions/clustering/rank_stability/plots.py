from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import ThingsDataset
from src.colors import INDIGO, PURPLE, ROSE
from src.plotting import create_figure, save_figure, setup_plotting

log = logging.getLogger(__name__)


def plot_rank_summary(
    per_rank_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Bar chart of cluster count and mean coherence per rank."""
    setup_plotting()
    ranks = per_rank_summary["rank"].values
    n_sub = per_rank_summary["n_substantial"].values
    coherence = per_rank_summary["mean_coherence"].values

    fig, axes = create_figure("wide", ncols=2)

    ax = axes[0]
    ax.bar(range(len(ranks)), n_sub, color=INDIGO, edgecolor="white")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("NMF rank")
    ax.set_ylabel("Substantial clusters")
    ax.set_title("Number of clusters per rank")

    ax = axes[1]
    ax.bar(range(len(ranks)), coherence, color=ROSE, edgecolor="white")
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("NMF rank")
    ax.set_ylabel("Mean coherence (Pearson r)")
    ax.set_title("Within-cluster coherence per rank")

    save_figure(fig, figures_dir / "rank_summary", formats=("pdf",))


def plot_pairwise_match_quality(
    pairwise_stats: pd.DataFrame,
    ranks: list[int],
    figures_dir: Path,
) -> None:
    """Heatmap of mean match correlation for each pair of ranks."""
    setup_plotting()
    n = len(ranks)
    mat = np.ones((n, n))

    for _, row in pairwise_stats.iterrows():
        i = ranks.index(int(row["rank_1"]))
        j = ranks.index(int(row["rank_2"]))
        mat[i, j] = row["mean_corr"]
        mat[j, i] = row["mean_corr"]

    fig, ax = create_figure("square")
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(r) for r in ranks])
    ax.set_xlabel("NMF rank")
    ax.set_ylabel("NMF rank")
    ax.set_title("Mean match correlation between ranks")

    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    save_figure(fig, figures_dir / "pairwise_match_quality", formats=("pdf",))


def plot_match_distributions(
    all_matches: dict,
    reference_rank: int,
    figures_dir: Path,
) -> None:
    """Histograms of per-match correlation for reference rank vs each other."""
    setup_plotting()
    pairs = [(r1, r2) for (r1, r2) in sorted(all_matches.keys())
             if r1 == reference_rank or r2 == reference_rank]

    if not pairs:
        return

    ncols = min(len(pairs), 3)
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if len(pairs) == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, (r1, r2) in enumerate(pairs):
        ax = axes_flat[i]
        mr = all_matches[(r1, r2)]
        corrs = np.array([c for _, _, c in mr.matches])

        ax.hist(corrs, bins=25, range=(0, 1), color=ROSE,
                edgecolor="white", alpha=0.8)
        ax.axvline(0.5, color=PURPLE, linestyle="--", linewidth=1,
                   label="threshold=0.5")
        other = r2 if r1 == reference_rank else r1
        ax.set_title(f"Rank {reference_rank} vs {other}")
        ax.set_xlabel("Match correlation")
        ax.set_ylabel("Count")

        frac = float(np.mean(corrs >= 0.5))
        ax.text(0.95, 0.95, f"{frac:.0%} above 0.5\nmedian={np.median(corrs):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    save_figure(fig, figures_dir / "match_distributions", formats=("pdf",))


def plot_concept_tracking(
    concept_tracking: pd.DataFrame,
    reference_rank: int,
    ranks: list[int],
    figures_dir: Path,
    n_show: int = 40,
) -> None:
    """Heatmap: rows=reference-rank concepts, columns=ranks, color=match quality."""
    setup_plotting()
    other_ranks = sorted(r for r in ranks if r != reference_rank)
    match_cols = [f"match_r{r}" for r in other_ranks]

    df = concept_tracking.head(n_show).copy()

    mat = df[match_cols].values
    labels = [f"C{cid}" for cid in df["concept_id"].values]

    fig_h = max(4, n_show * 0.2)
    fig, ax = plt.subplots(figsize=(max(4, len(other_ranks) * 1.2), fig_h))

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(other_ranks)))
    ax.set_xticklabels([str(r) for r in other_ranks])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("NMF rank")
    ax.set_ylabel(f"Cluster (rank {reference_rank})")
    ax.set_title(f"Concept stability: rank-{reference_rank} clusters at other ranks")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.4 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color=color)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Match correlation")
    fig.tight_layout()
    save_figure(fig, figures_dir / "concept_tracking", formats=("pdf",))


def plot_stable_vs_specific(
    concept_tracking: pd.DataFrame,
    rank_results: dict,
    reference_rank: int,
    ranks: list[int],
    ds: ThingsDataset,
    figures_dir: Path,
    n_show: int = 8,
) -> None:
    """Image grids for most stable and most rank-specific concepts."""
    setup_plotting()
    ref = rank_results[reference_rank]
    other_ranks = sorted(r for r in ranks if r != reference_rank)

    stable = concept_tracking.nlargest(n_show, "mean_match_corr")
    specific = concept_tracking.nsmallest(n_show, "mean_match_corr")

    n_imgs = 6
    fig, axes = plt.subplots(
        2 * n_show, n_imgs,
        figsize=(n_imgs * 1.2, 2 * n_show * 1.2),
    )

    for section_idx, (section_df, section_label) in enumerate([
        (stable, "Stable"),
        (specific, "Rank-specific"),
    ]):
        for row_idx, (_, row) in enumerate(section_df.iterrows()):
            cid = int(row["concept_id"])
            ax_row = section_idx * n_show + row_idx

            mask = ref.dim_df["cluster"] == cid
            idx_dims = np.where(mask.values)[0]
            if len(idx_dims) == 0:
                continue

            # We need W_all to get centroid — reconstruct from dim_df
            # Since we don't store W_all, use ds to load components
            # Actually, the centroid is in ref.centroids
            centroid = ref.centroids.get(cid)
            if centroid is None:
                continue

            top_imgs = np.argsort(-centroid)[:n_imgs]

            for col_idx, img_idx in enumerate(top_imgs):
                ax = axes[ax_row, col_idx]
                try:
                    img = ds.get_image(img_idx)
                    ax.imshow(img)
                except Exception:
                    pass
                ax.axis("off")
                if col_idx == 0:
                    corr_str = f"{row['mean_match_corr']:.2f}"
                    label = f"C{cid} (r={corr_str})"
                    if row_idx == 0:
                        label = f"{section_label}\n{label}"
                    ax.set_title(label, fontsize=6, loc="left")

    fig.tight_layout(pad=0.5)
    save_figure(fig, figures_dir / "stable_vs_specific", formats=("pdf",))


def plot_all(results: dict, figures_dir: Path) -> None:
    """Generate all figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = results["params"]
    ranks = params["ranks"]
    reference_rank = params["reference_rank"]

    plot_rank_summary(results["per_rank_summary"], figures_dir)
    plot_pairwise_match_quality(results["pairwise_stats"], ranks, figures_dir)
    plot_match_distributions(results["all_matches"], reference_rank, figures_dir)

    if results["concept_tracking"] is not None:
        plot_concept_tracking(
            results["concept_tracking"], reference_rank, ranks, figures_dir,
        )

        ds = results["rank_results"][reference_rank].categories
        # Get ThingsDataset for image loading
        from src.data import ThingsDataset
        ds_obj = ThingsDataset()

        plot_stable_vs_specific(
            results["concept_tracking"],
            results["rank_results"],
            reference_rank,
            ranks,
            ds_obj,
            figures_dir,
        )

    log.info(f"Figures saved to {figures_dir}")
