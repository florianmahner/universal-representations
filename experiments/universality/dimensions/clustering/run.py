#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from experiments.universality.shared import load_components_for_rank
from src.colors import GRAY_PALE, PURPLE, ROSE, SAND
from src.data import ThingsDataset
from src.plotting import save_figure, setup_plotting

# Correlation-distance cut of the dimension dendrogram. The manuscript's
# rating-set construction re-cuts the same linkage at this value, so the two
# analyses must share it.
THRESHOLD = 0.75
# A cluster counts as substantial once it holds this many dimensions.
MIN_CLUSTER_SIZE = 10

log = logging.getLogger(__name__)


def load_dimensions(
    rank: int,
    models: list[str] | None,
    dataset: str,
) -> tuple[np.ndarray, pd.DataFrame, ThingsDataset]:
    """Load all W vectors and build dimension metadata."""
    ds = ThingsDataset()
    components = load_components_for_rank(rank=rank, models=models, dataset=dataset)

    w_vectors = []
    dim_records = []
    for model, W in sorted(components.items()):
        for d in range(W.shape[1]):
            w_vectors.append(W[:, d])
            dim_records.append({"model": model, "dimension": d})

    W_all = np.array(w_vectors, dtype=np.float32)
    dim_df = pd.DataFrame(dim_records)

    log.info(
        f"Loaded {W_all.shape[0]} dimensions from {len(components)} models "
        f"({W_all.shape[1]} images)"
    )
    return W_all, dim_df, ds


def compute_linkage(W_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute condensed correlation distances and complete-linkage hierarchy."""
    log.info("Computing pairwise correlation distances...")
    t0 = perf_counter()
    condensed = pdist(W_all, metric="correlation")
    log.info(
        f"  {len(condensed):,} pairwise distances in {perf_counter() - t0:.1f}s"
    )

    log.info("Running complete-linkage clustering...")
    t0 = perf_counter()
    Z = linkage(condensed, method="complete")
    log.info(f"  Linkage computed in {perf_counter() - t0:.1f}s")

    return Z, condensed


def flat_partition(Z: np.ndarray, threshold: float) -> np.ndarray:
    """Cut dendrogram at distance threshold to get flat cluster labels."""
    labels = fcluster(Z, t=threshold, criterion="distance")
    n_clusters = len(set(labels))
    sizes = pd.Series(labels).value_counts()
    n_substantial = int((sizes >= MIN_CLUSTER_SIZE).sum())
    log.info(
        f"Flat partition at threshold {threshold}: "
        f"{n_clusters} clusters, {n_substantial} substantial (>={MIN_CLUSTER_SIZE})"
    )
    return labels


def run_null_model(
    labels: np.ndarray,
    models_arr: np.ndarray,
    n_models: int,
    seed: int,
    n_perms: int = 100,
) -> dict[str, np.ndarray]:
    """Permutation null: shuffle model labels, recompute coverage stats."""
    log.info(f"Running null model ({n_perms} permutations)...")
    rng = np.random.RandomState(seed)

    real_coverage = _coverage_stats(labels, models_arr, n_models)
    null_results: dict[str, list[int]] = {key: [] for key in real_coverage}

    for _ in range(n_perms):
        perm_models = rng.permutation(models_arr)
        perm_cov = _coverage_stats(labels, perm_models, n_models)
        for key, values in null_results.items():
            values.append(perm_cov[key])

    result = {}
    for key, vals in null_results.items():
        arr = np.array(vals)
        result[key] = arr
        real_val = real_coverage[key]
        p_val = (arr >= real_val).mean()
        log.info(
            f"  {key}: real={real_val}, "
            f"null mean={arr.mean():.1f} ± {arr.std():.1f}, p={p_val:.4f}"
        )

    result["real"] = real_coverage
    return result


def _coverage_stats(
    labels: np.ndarray,
    models_arr: np.ndarray,
    n_models: int,
) -> dict[str, int]:
    """Count clusters spanning >= X% of models."""
    df = pd.DataFrame({"cluster": labels, "model": models_arr})
    models_per = df.groupby("cluster")["model"].nunique()
    return {
        f"spanning_{int(t * 100)}pct": int((models_per >= n_models * t).sum())
        for t in (0.25, 0.50, 0.75)
    }


def profile_clusters(
    cluster_ids: list[int],
    dim_df: pd.DataFrame,
    W_all: np.ndarray,
    sim_matrix: np.ndarray,
    categories: np.ndarray,
    n_models: int,
    n_top_images: int = 16,
    n_top_cats: int = 8,
) -> list[dict]:
    """Build semantic profiles and coherence for clusters."""
    log.info(f"Profiling {len(cluster_ids)} clusters...")
    profiles = []
    for cid in cluster_ids:
        mask = dim_df["cluster"] == cid
        idx = np.where(mask.values)[0]
        member_models = dim_df.loc[mask, "model"].unique()

        mean_w = W_all[idx].mean(axis=0)
        # Pick top images: at most one per category
        sorted_imgs = np.argsort(-mean_w)
        top_img_idx = []
        seen_cats = set()
        for img_i in sorted_imgs:
            cat = categories[img_i]
            if cat not in seen_cats:
                top_img_idx.append(img_i)
                seen_cats.add(cat)
                if len(top_img_idx) >= n_top_images:
                    break
        top_img_idx = np.array(top_img_idx)
        cat_counts = Counter(categories[sorted_imgs[:200]])

        coherence = np.nan
        if len(idx) >= 2:
            sub = sim_matrix[np.ix_(idx, idx)]
            triu = sub[np.triu_indices_from(sub, k=1)]
            coherence = float(triu.mean())

        profiles.append({
            "cid": cid,
            "size": int(mask.sum()),
            "n_models": len(member_models),
            "model_frac": len(member_models) / n_models,
            "coherence": coherence,
            "top_img_idx": top_img_idx,
            "top_cats": [
                f"{c[0]} ({c[1]})" for c in cat_counts.most_common(n_top_cats)
            ],
        })
    return profiles


def compute_universality_connection(
    dim_df: pd.DataFrame,
    rank: int,
    dataset: str,
    n_models: int,
) -> pd.DataFrame | None:
    """Merge cluster assignments with per-dimension universality scores."""
    univ_path = Path(
        "experiments/universality/shared/data"
    ) / f"universality_{dataset}_rank_{rank}.csv"
    if not univ_path.exists():
        log.warning("No universality data found, skipping connection analysis")
        return None

    univ = pd.read_csv(univ_path)
    univ = univ[univ["metric"] == "cos2_perm_adj"].reset_index(drop=True)

    merged = univ.merge(dim_df[["model", "dimension", "cluster"]], on=["model", "dimension"])

    cluster_models = merged.groupby("cluster")["model"].nunique()
    merged["cluster_model_coverage"] = merged["cluster"].map(
        lambda c: cluster_models.get(c, 0) / n_models
    )

    bins = [0, 0.10, 0.25, 0.50, 1.01]
    bin_labels = ["<10%", "10-25%", "25-50%", ">50%"]
    merged["coverage_bin"] = pd.cut(
        merged["cluster_model_coverage"], bins=bins, labels=bin_labels, right=False,
    )
    log.info(
        f"  Universality connection: {len(merged)} dimensions matched, "
        f"correlation r={merged['universality'].corr(merged['cluster_model_coverage']):.3f}"
    )
    return merged


def plot_dendrogram(
    Z: np.ndarray,
    threshold: float,
    n_dimensions: int,
    out: Path,
) -> None:
    """Truncated dendrogram with threshold line and branch annotations."""
    fig, ax = plt.subplots(figsize=(16, 7))

    with plt.rc_context({"lines.linewidth": 0.7}):
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=250,
            ax=ax,
            color_threshold=threshold,
            above_threshold_color="#aaa",
            no_labels=True,
        )

    ax.axhline(threshold, color=PURPLE, ls="--", lw=1.2, alpha=0.8,
               label=f"Threshold = {threshold}")
    ax.set_ylabel("Correlation distance (1 - Pearson r)")
    ax.set_xlabel(f"Dimensions (truncated to top 250 merges, {n_dimensions} total)")
    ax.legend(loc="upper right")

    fig.suptitle("Concept hierarchy — agglomerative clustering dendrogram", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "dendrogram", formats=("pdf",))


def plot_dual_cut(
    Z: np.ndarray,
    dim_df: pd.DataFrame,
    W_all: np.ndarray,
    categories: np.ndarray,
    fine_threshold: float,
    out: Path,
) -> None:
    """Show coarse and fine clustering to illustrate multi-scale hierarchy."""
    coarse_threshold = min(1.35, fine_threshold * 1.6)
    coarse_labels = fcluster(Z, t=coarse_threshold, criterion="distance")
    fine_labels = fcluster(Z, t=fine_threshold, criterion="distance")

    coarse_df = dim_df.copy()
    coarse_df["coarse"] = coarse_labels
    coarse_df["fine"] = fine_labels

    coarse_sizes = coarse_df["coarse"].value_counts()
    coarse_models = coarse_df.groupby("coarse")["model"].nunique()
    top_coarse = coarse_sizes[coarse_sizes >= MIN_CLUSTER_SIZE].index

    coarse_info = []
    for cid in coarse_sizes.index:
        if cid not in top_coarse:
            continue
        mask = coarse_df["coarse"] == cid
        mean_w = W_all[mask.values].mean(axis=0)
        cat_counts = Counter(categories[np.argsort(-mean_w)[:100]])
        top_cat = cat_counts.most_common(1)[0][0] if cat_counts else "?"

        sub_fine = coarse_df.loc[mask, "fine"].value_counts()
        n_sub = int((sub_fine >= MIN_CLUSTER_SIZE).sum())

        coarse_info.append({
            "cid": cid,
            "size": int(coarse_sizes[cid]),
            "n_models": int(coarse_models[cid]),
            "top_cat": top_cat,
            "n_subclusters": n_sub,
        })

    coarse_info.sort(key=lambda x: -x["n_models"])

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(coarse_info) + 1)))

    y = np.arange(len(coarse_info))
    sizes = [c["size"] for c in coarse_info]
    labels = [f"{c['top_cat']} ({c['n_models']}m)" for c in coarse_info]

    axes[0].barh(y, sizes, color=ROSE, alpha=0.8, height=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Number of dimensions")
    axes[0].set_title(f"Coarse clusters (threshold={coarse_threshold}, {len(coarse_info)} clusters)")
    axes[0].invert_yaxis()

    n_subs = [c["n_subclusters"] for c in coarse_info]
    axes[1].barh(y, n_subs, color=SAND, alpha=0.8, height=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Number of fine sub-clusters")
    axes[1].set_title(f"Fine sub-clusters (threshold={fine_threshold})")
    axes[1].invert_yaxis()

    fig.suptitle("Multi-scale concept hierarchy", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "dual_cut", formats=("pdf",))


def plot_null_model(dim_df: pd.DataFrame, n_models: int, out: Path) -> None:
    """Cluster size vs model coverage with theoretical expectation curve."""
    cluster_sizes = dim_df.groupby("cluster").size()
    cluster_coverage = dim_df.groupby("cluster")["model"].nunique()
    substantial = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index

    sizes = cluster_sizes[substantial].values
    coverages = cluster_coverage[substantial].values
    coverage_frac = coverages / n_models

    # Theoretical: expected unique models if n draws from 162 with replacement
    s_range = np.arange(1, sizes.max() + 1)
    expected_models = n_models * (1 - ((n_models - 1) / n_models) ** s_range)
    expected_frac = expected_models / n_models

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sizes, coverage_frac, s=20, alpha=0.5, color=ROSE,
               label="Real clusters", zorder=3)
    ax.plot(s_range, expected_frac, color=PURPLE, lw=2, ls="--",
            label="Expected (random model assignment)", zorder=2)

    ax.set_xlabel("Cluster size (number of dimensions)")
    ax.set_ylabel("Model coverage (fraction of 162 models)")
    ax.set_title("Model diversity: real clusters vs. random expectation")
    ax.legend(fontsize=9)
    ax.set_xlim(0, sizes.max() * 1.05)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_figure(fig, out / "null_model", formats=("pdf",))


def plot_top_images(
    profiles: list[dict],
    ds: ThingsDataset,
    out: Path,
    n_show: int = 20,
) -> None:
    """Image grid for top clusters by model coverage."""
    n_show = min(n_show, len(profiles))
    fig, axes = plt.subplots(n_show, 8, figsize=(14, 1.7 * n_show))

    for row, prof in enumerate(profiles[:n_show]):
        for col in range(8):
            ax = axes[row, col]
            try:
                ax.imshow(ds.get_image(prof["top_img_idx"][col]))
            except Exception:
                pass
            ax.axis("off")
            if col == 0:
                cat_label = prof["top_cats"][0].split(" (")[0] if prof["top_cats"] else "?"
                ax.set_title(
                    f"C{prof['cid']} | {prof['n_models']}m | {cat_label}...",
                    fontsize=6, loc="left",
                )

    fig.suptitle(
        "Top clusters by model coverage — highest-scoring images",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, out / "top_images", formats=("pdf",))


def plot_universality_connection(univ_merged: pd.DataFrame, out: Path) -> None:
    """Box plot of per-dimension universality by cluster model-coverage bin."""
    fig, ax = plt.subplots(figsize=(7, 5))

    bin_order = ["<10%", "10-25%", "25-50%", ">50%"]
    existing_bins = [b for b in bin_order if b in univ_merged["coverage_bin"].cat.categories]

    box_data = [
        univ_merged.loc[univ_merged["coverage_bin"] == b, "universality"].values
        for b in existing_bins
    ]

    bp = ax.boxplot(
        box_data, tick_labels=existing_bins, patch_artist=True, widths=0.6,
    )
    colors_list = [GRAY_PALE, ROSE, SAND, PURPLE]
    for patch, color in zip(bp["boxes"], colors_list[:len(existing_bins)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(box_data):
        ax.annotate(
            f"n={len(data)}", (i + 1, ax.get_ylim()[0]),
            ha="center", va="bottom", fontsize=8, color="#666",
        )

    ax.set_xlabel("Cluster model coverage")
    ax.set_ylabel("Per-dimension universality (max-correlation)")
    ax.set_title("Per-dimension universality by cluster model coverage")

    fig.tight_layout()
    save_figure(fig, out / "universality_connection", formats=("pdf",))


def run_analysis(
    rank: int,
    dataset: str,
    models: list[str] | None,
    seed: int,
    out: Path,
) -> dict:
    """Run full agglomerative clustering analysis."""
    W_all, dim_df, ds = load_dimensions(rank, models, dataset)
    models_arr = dim_df["model"].values
    n_models = int(dim_df["model"].nunique())
    categories = np.array(ds.categories)

    Z, condensed = compute_linkage(W_all)
    np.save(out / "linkage.npy", Z)
    np.save(out / "condensed_dist.npy", condensed)

    labels = flat_partition(Z, THRESHOLD)
    dim_df["cluster"] = labels

    # Correlation similarity = 1 - correlation distance = Pearson r
    sim_matrix = 1.0 - squareform(condensed)
    np.fill_diagonal(sim_matrix, 1.0)

    dim_df.to_csv(out / "dimension_clusters.csv", index=False)

    cluster_sizes = dim_df["cluster"].value_counts().sort_values(ascending=False)
    cluster_models = dim_df.groupby("cluster")["model"].nunique()
    big_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index.tolist()

    top_cluster_ids = (
        cluster_models[cluster_models.index.isin(big_clusters)]
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )

    profiles = profile_clusters(
        top_cluster_ids, dim_df, W_all, sim_matrix, categories, n_models,
    )

    profile_df = pd.DataFrame([{
        "cluster": f"C{p['cid']}",
        "size": p["size"],
        "n_models": p["n_models"],
        "model_pct": f"{p['model_frac']*100:.0f}%",
        "coherence": f"{p['coherence']:.3f}" if not np.isnan(p["coherence"]) else "",
        "top_categories": ", ".join(p["top_cats"][:5]),
    } for p in profiles])
    profile_df.to_csv(out / "cluster_profiles.csv", index=False)

    null_results = run_null_model(labels, models_arr, n_models, seed)
    np.savez(
        out / "null_model.npz",
        **{k: v for k, v in null_results.items() if k != "real"},
    )
    (out / "null_model_real.json").write_text(json.dumps(null_results["real"]))

    univ_merged = compute_universality_connection(dim_df, rank, dataset, n_models)
    if univ_merged is not None:
        univ_merged.to_csv(out / "universality_connection.csv", index=False)

    coherence_vals = [p["coherence"] for p in profiles if not np.isnan(p["coherence"])]
    params = {
        "rank": rank,
        "threshold": THRESHOLD,
        "n_models": n_models,
        "n_dimensions": len(dim_df),
        "n_clusters": len(cluster_sizes),
        "n_substantial": len(big_clusters),
        "mean_coherence": float(np.mean(coherence_vals)) if coherence_vals else None,
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))
    log.info(f"Analysis complete. Outputs in {out}")

    return {
        "Z": Z,
        "dim_df": dim_df,
        "profiles": profiles,
        "univ_merged": univ_merged,
        "ds": ds,
        "W_all": W_all,
        "categories": categories,
        "n_models": n_models,
    }


def run_plots(results: dict, out: Path) -> None:
    """Generate all figures from precomputed results."""
    setup_plotting()

    log.info("Generating figures...")
    plot_dendrogram(results["Z"], THRESHOLD, len(results["dim_df"]), out)
    plot_dual_cut(
        results["Z"], results["dim_df"], results["W_all"],
        results["categories"], THRESHOLD, out,
    )
    plot_null_model(results["dim_df"], results["n_models"], out)
    plot_top_images(results["profiles"], results["ds"], out)
    if results["univ_merged"] is not None:
        plot_universality_connection(results["univ_merged"], out)
    log.info(f"All figures saved to {out}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    results = run_analysis(
        cfg.factorization.rank,
        cfg.dataset,
        list(cfg.models) or None,
        cfg.seed,
        out,
    )
    run_plots(results, out)


if __name__ == "__main__":
    main()
