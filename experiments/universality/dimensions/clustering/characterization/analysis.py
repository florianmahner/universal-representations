import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from experiments.universality.shared import (
    load_components_for_rank,
    load_model_info,
    load_universality,
)
from src.data import ThingsDataset

log = logging.getLogger(__name__)

CLUSTERING_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

DEFAULT_TIER_EDGES = [0.0, 0.05, 0.15, 1.01]
DEFAULT_TIER_LABELS = ["<5%", "5-15%", ">=15%"]


def gini(x: np.ndarray) -> float:
    """Gini coefficient (0 = uniform, 1 = maximally concentrated)."""
    x = np.sort(np.abs(x).ravel())
    n = len(x)
    total = x.sum()
    if total < 1e-12:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * (idx * x).sum()) / (n * total) - (n + 1) / n


def top_unique_categories(
    weights: np.ndarray,
    categories: np.ndarray,
    n: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """Get indices of top-n unique categories by weight (deduplicating repeats)."""
    sorted_idx = np.argsort(-weights)
    seen: set[str] = set()
    top_idx: list[int] = []
    top_cats: list[str] = []
    for i in sorted_idx:
        cat = categories[i]
        if cat not in seen:
            seen.add(cat)
            top_idx.append(int(i))
            top_cats.append(cat)
            if len(top_idx) >= n:
                break
    return np.array(top_idx), top_cats


def load_data(rank: int = 50) -> tuple[np.ndarray, pd.DataFrame, ThingsDataset]:
    """Load W matrices, cluster assignments, universality scores, and metadata.

    Returns:
        W_all: (n_dims, n_items) stacked dimension vectors
        dim_df: per-dimension DataFrame with cluster, universality, model metadata
        ds: ThingsDataset instance
    """
    ds = ThingsDataset()

    clusters_path = CLUSTERING_OUTPUT_DIR / "dimension_clusters.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found at {clusters_path}. "
            "Run the clustering analysis first."
        )
    clusters = pd.read_csv(clusters_path)
    cluster_models = sorted(clusters["model"].unique())
    log.info("Loaded cluster assignments for %d models", len(cluster_models))

    components = load_components_for_rank(rank=rank, models=cluster_models)
    loaded_models = sorted(components.keys())
    log.info("Loaded W matrices for %d models", len(loaded_models))

    w_vectors: list[np.ndarray] = []
    records: list[dict] = []
    for model in loaded_models:
        W = components[model]
        for d in range(W.shape[1]):
            w_vectors.append(W[:, d])
            records.append({"model": model, "dimension": d})

    W_all = np.array(w_vectors, dtype=np.float32)
    dim_df = pd.DataFrame(records)

    dim_df = dim_df.merge(clusters, on=["model", "dimension"], how="left")

    univ = load_universality(rank=rank)
    dim_df = dim_df.merge(
        univ[["model", "dimension", "universality"]],
        on=["model", "dimension"],
        how="left",
    )

    info = load_model_info()
    meta_cols = ["feature_uid", "model_class", "train_task", "objective_category"]
    available = [c for c in meta_cols if c in info.columns]
    dim_df = dim_df.merge(
        info[available], left_on="model", right_on="feature_uid", how="left"
    )
    if "feature_uid" in dim_df.columns:
        dim_df = dim_df.drop(columns=["feature_uid"])

    log.info("Built dim_df: %d rows, W_all: %s", len(dim_df), W_all.shape)
    return W_all, dim_df, ds


def build_cluster_profiles(
    W_all: np.ndarray,
    dim_df: pd.DataFrame,
    categories: np.ndarray,
    min_size: int = 3,
    n_top: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-cluster profiles with concentration and provenance metrics.

    Returns:
        profiles_df: sorted by model_coverage descending
        prototypes: dict mapping cluster_id (str) to mean W vector (n_items,)
    """
    n_total_models = dim_df["model"].nunique()
    profiles: list[dict] = []
    prototypes: dict[str, np.ndarray] = {}

    for cluster_id, group in dim_df.groupby("cluster"):
        if len(group) < min_size:
            continue

        idx = group.index.values
        W_cluster = W_all[idx]

        prototype = W_cluster.mean(axis=0)
        cid = str(cluster_id)
        prototypes[cid] = prototype

        n_models = group["model"].nunique()
        coverage = n_models / n_total_models

        top_idx, top_cats = top_unique_categories(prototype, categories, n=n_top)

        p = prototype / (prototype.sum() + 1e-12)
        ent = scipy_entropy(p[p > 0])
        effective_n = np.exp(ent)
        g = gini(prototype)
        top10_frac = float(np.sort(prototype)[-10:].sum() / (prototype.sum() + 1e-12))

        mean_univ = float(group["universality"].mean())

        if len(idx) > 1:
            centered = W_cluster - W_cluster.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            normed = centered / np.maximum(norms, 1e-8)
            corr = normed @ normed.T
            mask = np.triu(np.ones(len(idx), dtype=bool), k=1)
            coherence = float(corr[mask].mean())
        else:
            coherence = 1.0

        if "train_task" in group.columns:
            task_counts = group["train_task"].dropna().value_counts()
            if len(task_counts) > 0:
                dominant_task = task_counts.index[0]
                dominant_task_frac = float(task_counts.iloc[0] / len(group))
                task_ent = float(scipy_entropy(task_counts.values)) if len(task_counts) > 1 else 0.0
            else:
                dominant_task, dominant_task_frac, task_ent = "unknown", 1.0, 0.0
        else:
            dominant_task, dominant_task_frac, task_ent = "unknown", 1.0, 0.0

        member_models = group["model"].tolist()
        specific_tasks = sorted({
            m.split("_", 1)[1]
            for m in member_models
            if m.startswith("taskonomy_")
        })
        detail_tasks = ", ".join(specific_tasks) if specific_tasks else ""

        if "model_class" in group.columns:
            arch_counts = group["model_class"].dropna().value_counts()
            if len(arch_counts) > 0:
                dominant_arch = arch_counts.index[0]
                dominant_arch_frac = float(arch_counts.iloc[0] / len(group))
                arch_ent = float(scipy_entropy(arch_counts.values)) if len(arch_counts) > 1 else 0.0
            else:
                dominant_arch, dominant_arch_frac, arch_ent = "unknown", 1.0, 0.0
        else:
            dominant_arch, dominant_arch_frac, arch_ent = "unknown", 1.0, 0.0

        profiles.append({
            "cluster": cid,
            "size": len(group),
            "n_models": n_models,
            "model_coverage": coverage,
            "mean_universality": mean_univ,
            "coherence": coherence,
            "gini": g,
            "effective_n": effective_n,
            "top10_weight_frac": top10_frac,
            "top_categories": ", ".join(top_cats),
            "dominant_task": dominant_task,
            "dominant_task_frac": dominant_task_frac,
            "task_entropy": task_ent,
            "dominant_arch": dominant_arch,
            "dominant_arch_frac": dominant_arch_frac,
            "arch_entropy": arch_ent,
            "detail_tasks": detail_tasks,
        })

    profiles_df = (
        pd.DataFrame(profiles)
        .sort_values("model_coverage", ascending=False)
        .reset_index(drop=True)
    )
    log.info("Built profiles for %d clusters (min_size=%d)", len(profiles_df), min_size)
    return profiles_df, prototypes


def compute_provenance_by_tier(
    dim_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    tier_edges: list[float] | None = None,
    tier_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate model family composition by coverage tier.

    Returns long-form DataFrame with columns: tier, grouping, category, fraction.
    Includes a 'baseline' tier for overall composition.
    """
    if tier_edges is None:
        tier_edges = DEFAULT_TIER_EDGES
    if tier_labels is None:
        tier_labels = DEFAULT_TIER_LABELS

    coverage_map = profiles_df.set_index("cluster")["model_coverage"].to_dict()

    def assign_tier(cluster_id: str) -> str | None:
        cid = str(cluster_id)
        cov = coverage_map.get(cid)
        if cov is None:
            return None
        for i in range(len(tier_labels)):
            if tier_edges[i] <= cov < tier_edges[i + 1]:
                return tier_labels[i]
        return None

    df = dim_df.copy()
    df["tier"] = df["cluster"].astype(str).map(assign_tier)
    df = df.dropna(subset=["tier"])

    records: list[dict] = []

    for tier in tier_labels:
        tier_df = df[df["tier"] == tier]
        if len(tier_df) == 0:
            continue
        for col, grouping in [("train_task", "task"), ("model_class", "architecture")]:
            if col not in tier_df.columns:
                continue
            counts = tier_df[col].value_counts(normalize=True)
            for cat, frac in counts.items():
                records.append({"tier": tier, "grouping": grouping, "category": cat, "fraction": frac})

    for col, grouping in [("train_task", "task"), ("model_class", "architecture")]:
        if col not in dim_df.columns:
            continue
        counts = dim_df[col].value_counts(normalize=True)
        for cat, frac in counts.items():
            records.append({"tier": "baseline", "grouping": grouping, "category": cat, "fraction": frac})

    return pd.DataFrame(records)
