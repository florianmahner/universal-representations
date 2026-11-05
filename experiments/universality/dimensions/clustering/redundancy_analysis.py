#!/usr/bin/env python3
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from experiments.universality.shared import load_components_for_rank
from src.colors import PURPLE, ROSE
from src.data import ThingsDataset
from src.plotting import CLASS_COLORS, save_figure, setup_plotting

log = logging.getLogger(__name__)

# Cosine-distance cut used to call two dimensions near-duplicates; this is the
# redundancy protocol of the analysis, not a tunable of the factorization.
CLUSTER_THRESHOLD = 0.30


def compute_within_model_stats(
    components: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    """For each model, compute within-model pairwise cosine similarities."""
    records = []
    all_within_sims = []

    for model, W in sorted(components.items()):
        n_dims = W.shape[1]
        sim = 1.0 - squareform(pdist(W.T, metric="cosine"))
        np.fill_diagonal(sim, np.nan)
        triu = sim[np.triu_indices_from(sim, k=1)]
        all_within_sims.append(triu)

        n_high = int((triu >= (1 - CLUSTER_THRESHOLD)).sum())
        n_pairs = len(triu)

        # Effective distinct dims: cluster at same threshold, count clusters
        condensed = pdist(W.T, metric="cosine")
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=CLUSTER_THRESHOLD, criterion="distance")
        n_effective = len(set(labels))

        records.append({
            "model": model,
            "n_dims": n_dims,
            "mean_within_sim": float(np.nanmean(triu)),
            "max_within_sim": float(np.nanmax(triu)),
            "median_within_sim": float(np.nanmedian(triu)),
            "n_high_pairs": n_high,
            "frac_high_pairs": n_high / n_pairs,
            "n_effective_dims": n_effective,
            "redundancy_ratio": 1.0 - n_effective / n_dims,
        })

    df = pd.DataFrame(records)
    all_sims = np.concatenate(all_within_sims)
    return df, all_sims


def plot_overall_distribution(all_within_sims: np.ndarray, out: Path) -> None:
    """Histogram of all within-model pairwise cosine similarities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_within_sims, bins=100, color=ROSE, alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.axvline(1 - CLUSTER_THRESHOLD, color=PURPLE, ls="--", lw=1.5,
               label=f"Cluster threshold (sim={1-CLUSTER_THRESHOLD})")
    ax.set_xlabel("Cosine similarity (within-model dimension pairs)")
    ax.set_ylabel("Count")
    ax.set_title("Within-model pairwise dimension similarity")
    ax.legend()

    n_above = (all_within_sims >= (1 - CLUSTER_THRESHOLD)).sum()
    pct = 100 * n_above / len(all_within_sims)
    ax.annotate(f"{pct:.1f}% above threshold\n({n_above:,} pairs)",
                xy=(1 - CLUSTER_THRESHOLD, ax.get_ylim()[1] * 0.8),
                xytext=(0.85, ax.get_ylim()[1] * 0.7),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#666"))

    fig.tight_layout()
    save_figure(fig, out / "within_model_similarity_distribution", formats=("pdf",))


def plot_redundancy_per_model(
    stats_df: pd.DataFrame, ds: ThingsDataset, rank: int, out: Path
) -> None:
    """Effective dims and redundancy ratio per model, colored by architecture."""
    model_meta = ds.model_info.set_index("name")
    model_meta = model_meta[~model_meta.index.duplicated()]

    stats = stats_df.copy()
    stats["arch"] = stats["model"].map(
        lambda m: model_meta.loc[m, "model_class"]
        if m in model_meta.index else "Unknown"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: effective dims per model, sorted
    stats_sorted = stats.sort_values("n_effective_dims")
    colors = [CLASS_COLORS.get(a, "#999") for a in stats_sorted["arch"].values]
    axes[0].barh(range(len(stats_sorted)), stats_sorted["n_effective_dims"].values,
                 color=colors, alpha=0.7, height=1.0, edgecolor="none")
    axes[0].set_xlabel(f"Effective distinct dimensions (out of {rank})")
    axes[0].set_ylabel(f"Models (n={len(stats)})")
    axes[0].set_title("Effective dimensions per model")
    axes[0].set_yticks([])
    axes[0].axvline(rank, color="#ccc", ls=":", lw=1)
    axes[0].axvline(stats["n_effective_dims"].median(), color="#666", ls="--", lw=1,
                    label=f"Median = {stats['n_effective_dims'].median():.0f}")
    axes[0].legend(fontsize=8)

    # Right: boxplot by architecture
    arch_order = ["Convolutional", "Transformer", "MLP-Mixer"]
    box_data = [
        stats.loc[stats["arch"] == a, "n_effective_dims"].values
        for a in arch_order if a in stats["arch"].values
    ]
    existing_archs = [a for a in arch_order if a in stats["arch"].values]
    bp = axes[1].boxplot(box_data, tick_labels=existing_archs, patch_artist=True, widths=0.6)
    for patch, arch in zip(bp["boxes"], existing_archs):
        patch.set_facecolor(CLASS_COLORS.get(arch, "#999"))
        patch.set_alpha(0.7)

    axes[1].set_ylabel("Effective distinct dimensions")
    axes[1].set_title("Effective dimensions by architecture")

    for i, arch in enumerate(existing_archs):
        n = len(stats[stats["arch"] == arch])
        axes[1].annotate(f"n={n}", (i + 1, axes[1].get_ylim()[0]),
                         ha="center", va="bottom", fontsize=8, color="#666")

    fig.tight_layout()
    save_figure(fig, out / "effective_dimensions", formats=("pdf",))


def plot_most_redundant_heatmaps(
    components: dict[str, np.ndarray],
    stats_df: pd.DataFrame,
    rank: int,
    out: Path,
) -> None:
    """Show within-model similarity matrices for most and least redundant models."""
    most_redundant = stats_df.nsmallest(4, "n_effective_dims")
    least_redundant = stats_df.nlargest(4, "n_effective_dims")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, (_, row) in enumerate(most_redundant.iterrows()):
        model = row["model"]
        W = components[model]
        sim = 1.0 - squareform(pdist(W.T, metric="cosine"))
        np.fill_diagonal(sim, 1.0)
        im = axes[0, col].imshow(sim, vmin=0, vmax=1, cmap="RdYlBu_r")
        short_name = model.split("_", 1)[-1][:25] if "_" in model else model[:25]
        axes[0, col].set_title(
            f"{short_name}\neff={int(row['n_effective_dims'])}/{rank}", fontsize=8
        )
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

    for col, (_, row) in enumerate(least_redundant.iterrows()):
        model = row["model"]
        W = components[model]
        sim = 1.0 - squareform(pdist(W.T, metric="cosine"))
        np.fill_diagonal(sim, 1.0)
        axes[1, col].imshow(sim, vmin=0, vmax=1, cmap="RdYlBu_r")
        short_name = model.split("_", 1)[-1][:25] if "_" in model else model[:25]
        axes[1, col].set_title(
            f"{short_name}\neff={int(row['n_effective_dims'])}/{rank}", fontsize=8
        )
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Most redundant", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Least redundant", fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine similarity")
    fig.suptitle("Within-model dimension similarity matrices", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "similarity_heatmaps", formats=("pdf",))


def plot_redundant_dims_detail(
    components: dict[str, np.ndarray],
    stats_df: pd.DataFrame,
    ds: ThingsDataset,
    rank: int,
    out: Path,
) -> None:
    """For the most redundant model, show top images for each of its redundant dims."""
    worst = stats_df.nsmallest(1, "n_effective_dims").iloc[0]
    model = worst["model"]
    W = components[model]
    categories = np.array(ds.categories)

    # Cluster this model's dims
    condensed = pdist(W.T, metric="cosine")
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=CLUSTER_THRESHOLD, criterion="distance")

    # Find the largest within-model cluster
    label_counts = Counter(labels)
    biggest_label, biggest_size = label_counts.most_common(1)[0]
    dim_indices = [d for d, lab in enumerate(labels) if lab == biggest_label]

    n_show = min(12, biggest_size)
    n_imgs = 6

    fig, axes = plt.subplots(n_show, n_imgs, figsize=(10, 1.5 * n_show))

    for row, dim_idx in enumerate(dim_indices[:n_show]):
        w = W[:, dim_idx]
        top_imgs = np.argsort(-w)[:n_imgs]
        top_cats = [categories[i] for i in np.argsort(-w)[:20]]
        cat_counts = Counter(top_cats)
        label = cat_counts.most_common(1)[0][0]

        for col in range(n_imgs):
            ax = axes[row, col]
            try:
                ax.imshow(ds.get_image(top_imgs[col]))
            except Exception:
                pass
            ax.axis("off")
            if col == 0:
                ax.set_title(f"dim {dim_idx}: {label}", fontsize=7, loc="left")

    short_name = model.split("_", 1)[-1] if "_" in model else model
    fig.suptitle(
        f"Most redundant model: {short_name}\n"
        f"Largest within-model cluster: {biggest_size} dims (out of {rank})",
        fontweight="bold", fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, out / "redundant_dims_detail", formats=("pdf",))


def plot_dims_per_model_per_cluster(cluster_df: pd.DataFrame, out: Path) -> None:
    """Distribution of how many dims each model contributes to its clusters."""
    dpm = cluster_df.groupby(["cluster", "model"]).size().reset_index(name="n_dims")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of dims-per-model-per-cluster
    axes[0].hist(dpm["n_dims"].values, bins=range(1, dpm["n_dims"].max() + 2),
                 color=ROSE, alpha=0.8, edgecolor="white", linewidth=0.3,
                 align="left")
    axes[0].set_xlabel("Dimensions per model per cluster")
    axes[0].set_ylabel("Count")
    axes[0].set_title("How many dimensions does each model\ncontribute to each cluster?")

    pct_one = 100 * (dpm["n_dims"] == 1).mean()
    axes[0].annotate(f"{pct_one:.0f}% are singletons\n(1 dim per model per cluster)",
                     xy=(1, axes[0].get_ylim()[1] * 0.8), fontsize=9, color="#666")

    # Right: for clusters with >= 10 members, scatter of cluster size vs mean dims-per-model
    cluster_sizes = cluster_df["cluster"].value_counts()
    substantial = cluster_sizes[cluster_sizes >= 10].index
    dpm_sub = dpm[dpm["cluster"].isin(substantial)]
    cluster_mean_dpm = dpm_sub.groupby("cluster")["n_dims"].mean()
    cluster_n_models = cluster_df[cluster_df["cluster"].isin(substantial)].groupby("cluster")["model"].nunique()

    axes[1].scatter(cluster_n_models.values,
                    cluster_mean_dpm[cluster_n_models.index].values,
                    s=cluster_sizes[cluster_n_models.index].values / 5,
                    alpha=0.5, color=ROSE)
    axes[1].axhline(1.0, color="#ccc", ls=":", lw=1)
    axes[1].set_xlabel("Number of unique models in cluster")
    axes[1].set_ylabel("Mean dims per model in cluster")
    axes[1].set_title("Cluster model count vs. redundancy\n(point size = total cluster dims)")

    fig.tight_layout()
    save_figure(fig, out / "dims_per_model_per_cluster", formats=("pdf",))


def write_summary(
    stats_df: pd.DataFrame, all_within_sims: np.ndarray, rank: int, out: Path
) -> None:
    """Write markdown summary."""
    n_above = int((all_within_sims >= (1 - CLUSTER_THRESHOLD)).sum())
    total_pairs = len(all_within_sims)
    median_eff = stats_df["n_effective_dims"].median()
    mean_eff = stats_df["n_effective_dims"].mean()
    min_eff = stats_df["n_effective_dims"].min()
    max_eff = stats_df["n_effective_dims"].max()
    worst = stats_df.nsmallest(1, "n_effective_dims").iloc[0]
    best = stats_df.nlargest(1, "n_effective_dims").iloc[0]

    md = f"""# Dimension Redundancy Analysis

## Question

How much within-model redundancy exists in the rank-{rank} NMF dimensions?
Some agglomerative clusters are inflated because a single model contributes
many near-duplicate dimensions to the same cluster. This analysis quantifies
the problem.

## Method

For each of the {len(stats_df)} models, we compute pairwise cosine similarities between
its {rank} NMF dimensions. We define "redundant pairs" as those with cosine
similarity >= {1-CLUSTER_THRESHOLD} (matching the clustering threshold of
{CLUSTER_THRESHOLD} cosine distance). We also compute an "effective number
of distinct dimensions" by clustering each model's {rank} dimensions internally
at the same threshold and counting clusters.

## Key findings

### Overall within-model similarity

- **{n_above:,}** of {total_pairs:,} within-model pairs ({100*n_above/total_pairs:.1f}%) exceed the clustering threshold
- Most within-model pairs have low similarity (median ~{np.median(all_within_sims):.2f}), but there is a substantial tail above 0.7

![Within-model similarity distribution](within_model_similarity_distribution.png)

### Effective dimensions per model

- **Median**: {median_eff:.0f} effective dims (out of {rank})
- **Mean**: {mean_eff:.1f}
- **Range**: {min_eff:.0f} to {max_eff:.0f}
- **Most redundant**: {worst['model']} with {int(worst['n_effective_dims'])}/{rank} effective dims
- **Least redundant**: {best['model']} with {int(best['n_effective_dims'])}/{rank} effective dims

![Effective dimensions](effective_dimensions.png)

### Similarity matrices: most vs least redundant

![Heatmaps](similarity_heatmaps.png)

*Top row: most redundant models show large blocks of high similarity (red), indicating groups of near-duplicate dimensions. Bottom row: least redundant models have clean diagonal structure with low off-diagonal similarity.*

### What do redundant dimensions look like?

![Detail](redundant_dims_detail.png)

*Top images for each dimension in the largest within-model cluster of the most redundant model ({worst['model']}). If images across rows look nearly identical, the dimensions are true duplicates. If they show subtle differences, the dimensions may capture related-but-distinct sub-concepts.*

### Redundancy across clusters

![Dims per model](dims_per_model_per_cluster.png)

## Summary statistics

| Metric | Value |
|--------|-------|
| Models analyzed | {len(stats_df)} |
| Total within-model pairs | {total_pairs:,} |
| Pairs above sim >= {1-CLUSTER_THRESHOLD} | {n_above:,} ({100*n_above/total_pairs:.1f}%) |
| Median effective dims | {median_eff:.0f} / {rank} |
| Models with <= 30 effective dims | {(stats_df['n_effective_dims'] <= 30).sum()} |
| Models with <= 20 effective dims | {(stats_df['n_effective_dims'] <= 20).sum()} |

## Implications for clustering

If we proceed with the agglomerative clustering as-is, the large clusters
are partly artifacts of within-model redundancy. Options:

1. **Pre-deduplicate**: merge near-duplicate dims within each model before
   clustering across models
2. **Post-deduplicate**: weight each model equally when profiling clusters
3. **Accept and document**: treat redundancy as an NMF factorization property
   worth reporting
"""

    (out / "README.md").write_text(md)
    log.info("  Saved README.md")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_plotting()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank = cfg.factorization.rank

    log.info("Loading components...")
    components = load_components_for_rank(
        rank=rank,
        models=list(cfg.models) or None,
        dataset=cfg.dataset,
    )
    ds = ThingsDataset()

    log.info("Computing within-model statistics...")
    stats_df, all_within_sims = compute_within_model_stats(components)
    stats_df.to_csv(out / "model_redundancy_stats.csv", index=False)

    # Cluster assignments from the sibling clustering run, when available
    cluster_path = out / "dimension_clusters.csv"
    cluster_df = pd.read_csv(cluster_path) if cluster_path.exists() else None

    log.info("Generating figures...")
    plot_overall_distribution(all_within_sims, out)
    plot_redundancy_per_model(stats_df, ds, rank, out)
    plot_most_redundant_heatmaps(components, stats_df, rank, out)
    plot_redundant_dims_detail(components, stats_df, ds, rank, out)
    if cluster_df is not None:
        plot_dims_per_model_per_cluster(cluster_df, out)

    write_summary(stats_df, all_within_sims, rank, out)
    log.info(f"All outputs saved to {out}")


if __name__ == "__main__":
    main()
