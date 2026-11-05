from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.universality.dimensions.clustering.run import (
    MIN_CLUSTER_SIZE,
    compute_linkage,
    flat_partition,
    load_dimensions,
)

log = logging.getLogger(__name__)


@dataclass
class RankClusterResult:
    rank: int
    dim_df: pd.DataFrame
    centroids: dict[int, np.ndarray]
    cluster_stats: pd.DataFrame
    n_models: int
    categories: np.ndarray


@dataclass
class MatchResult:
    matches: list[tuple[int, int, float]]
    unmatched_r1: list[int]
    unmatched_r2: list[int]
    corr_matrix: np.ndarray
    cids_r1: list[int]
    cids_r2: list[int]


def compute_cluster_centroids(
    W_all: np.ndarray,
    dim_df: pd.DataFrame,
    min_size: int = MIN_CLUSTER_SIZE,
) -> dict[int, np.ndarray]:
    """Mean activation profile for each substantial cluster."""
    sizes = dim_df["cluster"].value_counts()
    big = sizes[sizes >= min_size].index
    centroids = {}
    for cid in big:
        idx = np.where((dim_df["cluster"] == cid).values)[0]
        centroids[int(cid)] = W_all[idx].mean(axis=0)
    return centroids


def compute_cluster_stats(
    dim_df: pd.DataFrame,
    W_all: np.ndarray,
    centroids: dict[int, np.ndarray],
    n_models: int,
) -> pd.DataFrame:
    """Per-cluster summary stats for substantial clusters."""
    rows = []
    for cid in sorted(centroids.keys()):
        mask = dim_df["cluster"] == cid
        idx = np.where(mask.values)[0]
        member_models = dim_df.loc[mask, "model"].nunique()

        coherence = np.nan
        if len(idx) >= 2:
            sub_W = W_all[idx]
            corr = np.corrcoef(sub_W)
            triu = corr[np.triu_indices_from(corr, k=1)]
            coherence = float(np.nanmean(triu))

        rows.append({
            "cid": cid,
            "size": int(mask.sum()),
            "n_models": member_models,
            "model_frac": member_models / n_models,
            "coherence": coherence,
        })
    return pd.DataFrame(rows)


def cluster_at_rank(
    rank: int,
    threshold: float,
    max_models: int | None = None,
    cache_dir: Path | None = None,
) -> RankClusterResult:
    """Run clustering at one rank, with optional caching of expensive steps."""
    cache = cache_dir / f"rank_{rank}" if cache_dir else None

    W_all, dim_df, ds = load_dimensions(rank, max_models)
    n_models = dim_df["model"].nunique()
    n_dims = len(dim_df)

    cache_valid = False
    if cache and (cache / "dim_clusters.csv").exists():
        cached_df = pd.read_csv(cache / "dim_clusters.csv")
        if len(cached_df) == n_dims and set(cached_df["model"]) == set(dim_df["model"]):
            log.info(f"Loading cached clustering for rank {rank}")
            dim_df = cached_df
            cache_valid = True

    if not cache_valid:
        t0 = perf_counter()
        Z, condensed = compute_linkage(W_all)
        log.info(f"Rank {rank}: linkage in {perf_counter() - t0:.1f}s")

        labels = flat_partition(Z, threshold)
        dim_df["cluster"] = labels

        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "linkage.npy", Z)
            dim_df.to_csv(cache / "dim_clusters.csv", index=False)

    centroids = compute_cluster_centroids(W_all, dim_df)
    stats = compute_cluster_stats(dim_df, W_all, centroids, n_models)

    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        stats.to_csv(cache / "cluster_stats.csv", index=False)

    categories = np.array(ds.categories)

    log.info(
        f"Rank {rank}: {len(centroids)} substantial clusters "
        f"from {len(dim_df)} dimensions ({n_models} models)"
    )
    return RankClusterResult(
        rank=rank,
        dim_df=dim_df,
        centroids=centroids,
        cluster_stats=stats,
        n_models=n_models,
        categories=categories,
    )


def match_clusters(
    centroids_r1: dict[int, np.ndarray],
    centroids_r2: dict[int, np.ndarray],
) -> MatchResult:
    """Optimal 1-to-1 cluster matching via Hungarian algorithm on correlations."""
    cids_r1 = sorted(centroids_r1.keys())
    cids_r2 = sorted(centroids_r2.keys())

    C1 = np.array([centroids_r1[c] for c in cids_r1])
    C2 = np.array([centroids_r2[c] for c in cids_r2])

    stacked = np.vstack([C1, C2])
    full_corr = np.corrcoef(stacked)
    n1, n2 = len(cids_r1), len(cids_r2)
    corr_matrix = full_corr[:n1, n1:]

    cost = 1.0 - np.abs(corr_matrix)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_r1 = set()
    matched_r2 = set()
    for i, j in zip(row_ind, col_ind):
        matches.append((cids_r1[i], cids_r2[j], float(corr_matrix[i, j])))
        matched_r1.add(cids_r1[i])
        matched_r2.add(cids_r2[j])

    matches.sort(key=lambda x: x[2], reverse=True)

    return MatchResult(
        matches=matches,
        unmatched_r1=[c for c in cids_r1 if c not in matched_r1],
        unmatched_r2=[c for c in cids_r2 if c not in matched_r2],
        corr_matrix=corr_matrix,
        cids_r1=cids_r1,
        cids_r2=cids_r2,
    )


def compute_pairwise_stats(
    all_matches: dict[tuple[int, int], MatchResult],
    match_threshold: float = 0.5,
) -> pd.DataFrame:
    """Summary statistics for each pairwise rank comparison."""
    rows = []
    for (r1, r2), mr in sorted(all_matches.items()):
        corrs = [c for _, _, c in mr.matches]
        rows.append({
            "rank_1": r1,
            "rank_2": r2,
            "n_clusters_r1": len(mr.cids_r1),
            "n_clusters_r2": len(mr.cids_r2),
            "n_matched": len(mr.matches),
            "mean_corr": float(np.mean(corrs)),
            "median_corr": float(np.median(corrs)),
            "frac_above_threshold": float(np.mean(np.array(corrs) >= match_threshold)),
            "n_unmatched_r1": len(mr.unmatched_r1),
            "n_unmatched_r2": len(mr.unmatched_r2),
        })
    return pd.DataFrame(rows)


def track_concepts(
    rank_results: dict[int, RankClusterResult],
    all_matches: dict[tuple[int, int], MatchResult],
    reference_rank: int = 50,
    match_threshold: float = 0.5,
) -> pd.DataFrame:
    """Track each reference-rank cluster across all other ranks.

    For each substantial cluster at the reference rank, find its best match
    at every other rank and record the match quality.
    """
    ref = rank_results[reference_rank]
    other_ranks = sorted(r for r in rank_results if r != reference_rank)

    ref_stats = ref.cluster_stats.set_index("cid")
    rows = []

    for cid in sorted(ref.centroids.keys()):
        ref_centroid = ref.centroids[cid]
        stat = ref_stats.loc[cid]

        top_cats = _top_categories(ref.dim_df, cid, ref.categories)

        row = {
            "concept_id": cid,
            "ref_size": int(stat["size"]),
            "ref_n_models": int(stat["n_models"]),
            "ref_model_frac": float(stat["model_frac"]),
            "ref_coherence": float(stat["coherence"]),
            "top_categories": top_cats,
        }

        for other_rank in other_ranks:
            best_corr = _find_best_match_corr(
                cid, reference_rank, other_rank, all_matches, rank_results,
            )
            row[f"match_r{other_rank}"] = best_corr

        n_good = sum(
            1 for r in other_ranks
            if row.get(f"match_r{r}", 0) >= match_threshold
        )
        row["n_ranks_matched"] = n_good
        row["is_core_stable"] = n_good == len(other_ranks)
        row["mean_match_corr"] = float(np.mean([
            row[f"match_r{r}"] for r in other_ranks
            if not np.isnan(row.get(f"match_r{r}", np.nan))
        ])) if other_ranks else np.nan

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean_match_corr", ascending=False)
    return df.reset_index(drop=True)


def _find_best_match_corr(
    cid: int,
    ref_rank: int,
    other_rank: int,
    all_matches: dict[tuple[int, int], MatchResult],
    rank_results: dict[int, RankClusterResult],
) -> float:
    """Find best correlation for a reference cluster at another rank.

    Uses the full correlation matrix (not just the 1-to-1 assignment)
    to find the highest correlation with any cluster at the other rank.
    """
    key = (min(ref_rank, other_rank), max(ref_rank, other_rank))
    mr = all_matches[key]

    if ref_rank < other_rank:
        if cid not in mr.cids_r1:
            return np.nan
        idx = mr.cids_r1.index(cid)
        return float(np.max(np.abs(mr.corr_matrix[idx, :])))
    else:
        if cid not in mr.cids_r2:
            return np.nan
        idx = mr.cids_r2.index(cid)
        return float(np.max(np.abs(mr.corr_matrix[:, idx])))


def _top_categories(
    dim_df: pd.DataFrame,
    cid: int,
    categories: np.ndarray,
    n: int = 3,
) -> str:
    """Get top category labels for a cluster (for display)."""
    # We don't have W_all here, so use category frequency from dim_df
    # This is a rough label; full profiling uses W_all centroids
    return f"C{cid}"


def identify_splits(
    rank_results: dict[int, RankClusterResult],
    all_matches: dict[tuple[int, int], MatchResult],
    reference_rank: int = 50,
    split_threshold: float = 0.3,
) -> list[dict]:
    """Detect clusters that split from reference rank to higher ranks.

    A cluster "splits" if multiple clusters at a higher rank correlate
    with it above split_threshold.
    """
    ref = rank_results[reference_rank]
    higher_ranks = sorted(r for r in rank_results if r > reference_rank)
    splits = []

    for higher_rank in higher_ranks:
        key = (reference_rank, higher_rank)
        if key not in all_matches:
            continue
        mr = all_matches[key]

        for cid in sorted(ref.centroids.keys()):
            if cid not in mr.cids_r1:
                continue
            idx = mr.cids_r1.index(cid)
            row_corrs = np.abs(mr.corr_matrix[idx, :])
            above = np.where(row_corrs >= split_threshold)[0]
            if len(above) > 1:
                children = [
                    (mr.cids_r2[j], float(row_corrs[j]))
                    for j in above
                ]
                children.sort(key=lambda x: x[1], reverse=True)
                splits.append({
                    "parent_cid": cid,
                    "parent_rank": reference_rank,
                    "child_rank": higher_rank,
                    "children": children,
                    "n_children": len(children),
                })

    return splits


def run_full_analysis(
    ranks: list[int],
    threshold: float,
    match_threshold: float,
    reference_rank: int,
    max_models: int | None,
    cache_dir: Path,
) -> dict:
    """Run complete cross-rank stability analysis."""
    rank_results: dict[int, RankClusterResult] = {}
    for rank in ranks:
        log.info(f"\n{'='*60}")
        log.info(f"Clustering at rank {rank}")
        log.info(f"{'='*60}")
        rank_results[rank] = cluster_at_rank(rank, threshold, max_models, cache_dir)

    log.info(f"\n{'='*60}")
    log.info("Cross-rank matching")
    log.info(f"{'='*60}")
    all_matches: dict[tuple[int, int], MatchResult] = {}
    for r1, r2 in combinations(sorted(ranks), 2):
        log.info(f"Matching rank {r1} vs {r2}...")
        all_matches[(r1, r2)] = match_clusters(
            rank_results[r1].centroids,
            rank_results[r2].centroids,
        )
        mr = all_matches[(r1, r2)]
        corrs = [c for _, _, c in mr.matches]
        log.info(
            f"  {len(mr.matches)} matched, "
            f"mean corr={np.mean(corrs):.3f}, "
            f"median={np.median(corrs):.3f}"
        )

    pairwise_stats = compute_pairwise_stats(all_matches, match_threshold)

    concept_tracking = None
    if reference_rank in ranks:
        concept_tracking = track_concepts(
            rank_results, all_matches, reference_rank, match_threshold,
        )
        n_core = concept_tracking["is_core_stable"].sum()
        n_total = len(concept_tracking)
        log.info(f"Core stable concepts: {n_core}/{n_total}")

    splits = []
    if reference_rank in ranks:
        splits = identify_splits(
            rank_results, all_matches, reference_rank,
        )

    per_rank_summary = pd.DataFrame([{
        "rank": r,
        "n_dimensions": len(res.dim_df),
        "n_clusters_total": res.dim_df["cluster"].nunique(),
        "n_substantial": len(res.centroids),
        "mean_coherence": float(res.cluster_stats["coherence"].mean()),
        "mean_model_frac": float(res.cluster_stats["model_frac"].mean()),
        "n_models": res.n_models,
    } for r, res in sorted(rank_results.items())])

    return {
        "rank_results": rank_results,
        "all_matches": all_matches,
        "pairwise_stats": pairwise_stats,
        "per_rank_summary": per_rank_summary,
        "concept_tracking": concept_tracking,
        "splits": splits,
        "params": {
            "ranks": ranks,
            "threshold": threshold,
            "match_threshold": match_threshold,
            "reference_rank": reference_rank,
        },
    }
