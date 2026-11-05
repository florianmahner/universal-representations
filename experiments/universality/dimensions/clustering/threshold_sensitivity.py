#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import fcluster

from experiments.universality.shared import load_universality
from src.colors import INDIGO, PURPLE, ROSE
from src.plotting import setup_plotting, create_figure, save_figure

# The manuscript cuts the dimension dendrogram at this correlation distance;
# the sweep exists to show the result does not hinge on that choice.
MANUSCRIPT_THRESHOLD = 0.75

log = logging.getLogger(__name__)


def sweep_thresholds(
    Z: np.ndarray,
    dim_df: pd.DataFrame,
    univ: pd.DataFrame,
    thresholds: np.ndarray,
    min_cluster_size: int = 3,
) -> pd.DataFrame:
    """Recluster at each threshold and compute coverage-universality metrics."""
    n_models = dim_df["model"].nunique()
    merged_base = dim_df.merge(
        univ[["model", "dimension", "universality"]],
        on=["model", "dimension"],
        how="inner",
    )

    records = []
    for thresh in thresholds:
        labels = fcluster(Z, t=thresh, criterion="distance")
        n_clusters = len(set(labels))

        tmp = merged_base.copy()
        tmp["cluster"] = labels[tmp.index.values]

        sizes = tmp.groupby("cluster").size()
        big = sizes[sizes >= min_cluster_size].index
        n_big = len(big)

        if n_big < 2:
            records.append({
                "threshold": thresh,
                "n_clusters": n_clusters,
                "n_clusters_min3": n_big,
                "r_dim": np.nan,
                "r_cluster": np.nan,
            })
            continue

        cluster_models = tmp.groupby("cluster")["model"].nunique()
        tmp["coverage"] = tmp["cluster"].map(lambda c: cluster_models.get(c, 0) / n_models)

        # Per-dimension correlation
        valid = tmp["coverage"].notna() & tmp["universality"].notna()
        r_dim = float(tmp.loc[valid, "coverage"].corr(tmp.loc[valid, "universality"]))

        # Per-cluster correlation (mean universality vs coverage)
        cluster_agg = (
            tmp[tmp["cluster"].isin(big)]
            .groupby("cluster")
            .agg(mean_univ=("universality", "mean"), coverage=("coverage", "first"))
        )
        r_cluster = float(cluster_agg["mean_univ"].corr(cluster_agg["coverage"]))

        records.append({
            "threshold": thresh,
            "n_clusters": n_clusters,
            "n_clusters_min3": n_big,
            "r_dim": r_dim,
            "r_cluster": r_cluster,
        })
        log.info(
            "  threshold=%.2f  clusters=%d (>=%d: %d)  r_dim=%.3f  r_cluster=%.3f",
            thresh, n_clusters, min_cluster_size, n_big, r_dim, r_cluster,
        )

    return pd.DataFrame(records)


def plot_sensitivity(results: pd.DataFrame, figures_dir: Path) -> None:
    """Two-panel figure: correlation vs threshold, and number of clusters."""
    setup_plotting()
    fig, axes = create_figure("wide", ncols=2)
    ax_corr, ax_n = axes

    t = results["threshold"].values

    # Left: correlation with universality
    ax_corr.plot(t, results["r_dim"].values, color=INDIGO,
                 linewidth=1.5, label="Per-dimension")
    ax_corr.plot(t, results["r_cluster"].values, color=ROSE,
                 linewidth=1.5, label="Per-cluster (mean)")

    ax_corr.axvline(MANUSCRIPT_THRESHOLD, color=PURPLE, linestyle="--", linewidth=0.8,
                    alpha=0.6, label=f"Manuscript threshold ({MANUSCRIPT_THRESHOLD})")
    ax_corr.set_xlabel("Distance threshold")
    ax_corr.set_ylabel("Pearson r (coverage vs universality)")
    ax_corr.legend(fontsize=6)

    # Right: number of clusters
    ax_n.plot(t, results["n_clusters"].values, color=INDIGO,
              linewidth=1.5, label="Total")
    ax_n.plot(t, results["n_clusters_min3"].values, color=ROSE,
              linewidth=1.5, label="Size >= 3")
    ax_n.axvline(MANUSCRIPT_THRESHOLD, color=PURPLE, linestyle="--", linewidth=0.8,
                 alpha=0.6)
    ax_n.set_xlabel("Distance threshold")
    ax_n.set_ylabel("Number of clusters")
    ax_n.set_yscale("log")
    ax_n.legend(fontsize=6)

    fig.tight_layout()
    save_figure(fig, figures_dir / "threshold_sensitivity", formats=("pdf",))


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    log.info("Loading saved linkage and dimension data")
    Z = np.load(out / "linkage.npy")
    dim_df = pd.read_csv(out / "dimension_clusters.csv")
    univ = load_universality(rank=cfg.factorization.rank, dataset=cfg.dataset)

    thresholds = np.linspace(0.4, 1.4, 30)

    log.info(
        "Sweeping %d thresholds from %.2f to %.2f",
        len(thresholds), thresholds[0], thresholds[-1],
    )
    results = sweep_thresholds(Z, dim_df, univ, thresholds)

    results.to_csv(out / "threshold_sensitivity.csv", index=False)
    log.info("Saved results to %s", out / "threshold_sensitivity.csv")

    plot_sensitivity(results, out)
    log.info("Done")


if __name__ == "__main__":
    main()
