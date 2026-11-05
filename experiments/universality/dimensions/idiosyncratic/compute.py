from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.universality.shared import (
    load_components_for_rank,
    load_model_info,
    load_universality,
)

log = logging.getLogger(__name__)

LOO_IMPORTANCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "reconstruction_importance"
    / "outputs"
)


def load_and_merge(rank: int = 50) -> pd.DataFrame:
    """Merge universality scores with LOO reconstruction importance.

    Returns a DataFrame with one row per model/dimension (8,100 rows for
    the manuscript's 162 models at rank 50), containing
    both universality and importance metrics plus model metadata.
    """
    univ = load_universality(rank=rank)
    log.info(f"Loaded universality: {len(univ)} rows, {univ['model'].nunique()} models")

    importance_path = LOO_IMPORTANCE_DIR / f"loo_importance_full_rank_{rank}.csv"
    if not importance_path.exists():
        raise FileNotFoundError(
            f"Reconstruction importance not found at {importance_path}. "
            "Run experiments.universality.dimensions.reconstruction_importance.run first."
        )
    loo = pd.read_csv(importance_path)
    if "rank" in loo.columns:
        loo = loo[loo["rank"] == rank].copy()
    log.info(f"Loaded LOO importance: {len(loo)} rows, {loo['model'].nunique()} models")

    loo_cols = ["model", "dimension", "delta_r2", "r2_full"]
    missing = set(loo_cols).difference(loo.columns)
    if missing:
        raise ValueError(
            f"{importance_path} is missing columns: {sorted(missing)}"
        )
    if loo.duplicated(["model", "dimension"]).any():
        raise ValueError("Duplicate model/dimension rows in reconstruction importance")
    merged = univ.merge(
        loo[loo_cols],
        on=["model", "dimension"],
        how="inner",
        validate="one_to_one",
    )

    meta = load_model_info()
    id_col = "feature_uid" if "feature_uid" in meta.columns else "name"
    meta_cols = ["model_class", "train_task", "train_data", "source"]
    meta_sub = meta[[id_col] + meta_cols].rename(columns={id_col: "model"})
    meta_sub = meta_sub.drop_duplicates(subset="model")
    merged = merged.merge(meta_sub, on="model", how="left")

    log.info(
        f"Merged: {len(merged)} dimensions from {merged['model'].nunique()} models"
    )
    return merged


def compute_thresholds(
    df: pd.DataFrame,
    univ_threshold: float | None = None,
    importance_threshold: float | None = None,
    clip_importance: float = 1.0,
) -> dict[str, float]:
    """Compute data-driven thresholds (Q25 universality, Q75 delta_r2).

    Thresholds are computed on the *clipped* delta_r2 so that extreme
    outliers (delta_r2 >> 1 in some transformer models) don't bias Q75.
    """
    clipped = df["delta_r2"].clip(upper=clip_importance)
    return {
        "universality": univ_threshold or float(df["universality"].quantile(0.25)),
        "importance": importance_threshold or float(clipped.quantile(0.75)),
    }


def classify_dimensions(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    clip_importance: float = 1.0,
) -> pd.DataFrame:
    """Add quadrant classification and clipped importance.

    Adds ``delta_r2_clipped`` (capped at *clip_importance*) used for
    quadrant thresholds and scatter plots.  Raw ``delta_r2`` is kept.
    """
    df = df.copy()
    df["delta_r2_clipped"] = df["delta_r2"].clip(upper=clip_importance)

    low_univ = df["universality"] < thresholds["universality"]
    high_imp = df["delta_r2_clipped"] >= thresholds["importance"]

    conditions = [
        (~low_univ & high_imp),
        (~low_univ & ~high_imp),
        (low_univ & high_imp),
        (low_univ & ~high_imp),
    ]
    labels = [
        "universal-important",
        "universal-unimportant",
        "idiosyncratic-important",
        "idiosyncratic-noise",
    ]

    df["quadrant"] = np.select(conditions, labels, default="unknown")
    return df


def compute_correlation_profiles(
    rank: int = 50,
    max_models: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute full per-model correlation profiles for every dimension.

    For each dimension d of model i, computes max|correlation| with every
    other model's best-matching dimension.

    Returns
    -------
    profiles : ndarray of shape (n_models, rank, n_models)
        profiles[i, d, j] = max|corr| of dim d of model i with model j.
        Diagonal (i==j) is NaN.
    models : list[str]
        Sorted model names corresponding to axis 0/2.
    """
    components = load_components_for_rank(rank=rank, max_models=max_models)
    models = sorted(components.keys())
    n_models = len(models)
    n_items = next(iter(components.values())).shape[0]

    log.info(f"Pre-standardizing {n_models} W matrices...")
    standardized: dict[str, np.ndarray] = {}
    for m, W in components.items():
        standardized[m] = (W - W.mean(0)) / (W.std(0) + 1e-8)

    profiles = np.full((n_models, rank, n_models), np.nan, dtype=np.float32)
    n_pairs = n_models * (n_models - 1) // 2

    log.info(f"Computing {n_pairs:,} pairwise correlation matrices...")
    for i in range(n_models):
        if i % 20 == 0:
            log.info(f"  Model {i}/{n_models}...")
        zi = standardized[models[i]]
        for j in range(i + 1, n_models):
            zj = standardized[models[j]]
            corr_mat = (zj.T @ zi) / n_items  # (rank, rank)
            abs_corr = np.abs(corr_mat)
            profiles[j, :, i] = abs_corr.max(axis=1)  # best match for each dim of j
            profiles[i, :, j] = abs_corr.max(axis=0)  # best match for each dim of i

    log.info("Correlation profiles computed.")
    return profiles, models


def compute_profile_stats(
    profiles: np.ndarray,
    models: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize correlation profile shapes for each dimension.

    Computes distributional statistics over the ~159 max-correlations
    per dimension to classify profile shapes.
    """
    n_models, rank, _ = profiles.shape
    records = []

    for i, model in enumerate(models):
        for d in range(rank):
            corrs = profiles[i, d, :]
            valid = corrs[~np.isnan(corrs)]

            records.append({
                "model": model,
                "dimension": d,
                "profile_mean": float(valid.mean()),
                "profile_std": float(valid.std()),
                "profile_median": float(np.median(valid)),
                "profile_max": float(valid.max()),
                "profile_min": float(valid.min()),
                "n_high": int((valid > 0.5).sum()),
                "n_moderate": int(((valid > 0.3) & (valid <= 0.5)).sum()),
                "n_low": int((valid <= 0.2).sum()),
                "profile_skew": float(_skewness(valid)),
            })

    stats = pd.DataFrame(records)

    # Classify profile shape
    stats["profile_type"] = "mixed"
    stats.loc[stats["n_high"] == 0, "profile_type"] = "flat"
    stats.loc[
        (stats["n_high"] >= 5) & (stats["n_low"] >= 50), "profile_type"
    ] = "bimodal"
    stats.loc[stats["n_high"] >= 100, "profile_type"] = "widespread"

    return stats


def _skewness(x: np.ndarray) -> float:
    """Compute skewness without scipy dependency."""
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s < 1e-10:
        return 0.0
    return float(((x - m) ** 3).mean() / s**3)


def identify_sharing_taxonomy(
    profiles: np.ndarray,
    models: list[str],
    df: pd.DataFrame,
    corr_threshold: float = 0.5,
) -> pd.DataFrame:
    """For subgroup-shared dims, identify which model groups share them.

    Returns a DataFrame with one row per idiosyncratic dim that has
    at least 3 high-correlation matches, listing the matching models'
    taxonomy breakdown.
    """
    meta = load_model_info()
    id_col = "feature_uid" if "feature_uid" in meta.columns else "name"
    model_to_class = dict(zip(meta[id_col], meta["model_class"]))
    model_to_task = dict(zip(meta[id_col], meta["train_task"]))
    model_to_data = dict(zip(meta[id_col], meta["train_data"]))

    n_models, rank, _ = profiles.shape
    records = []

    for i, model in enumerate(models):
        for d in range(rank):
            corrs = profiles[i, d, :]
            high_mask = corrs > corr_threshold
            high_mask[i] = False  # exclude self
            n_high = int(high_mask.sum())

            if n_high < 3:
                continue

            high_models = [models[j] for j in range(n_models) if high_mask[j]]
            classes = [model_to_class.get(m, "unknown") for m in high_models]
            tasks = [model_to_task.get(m, "unknown") for m in high_models]
            datasets = [model_to_data.get(m, "unknown") for m in high_models]

            records.append({
                "model": model,
                "dimension": d,
                "n_high_matches": n_high,
                "matching_classes": _top_counts(classes),
                "matching_tasks": _top_counts(tasks),
                "matching_datasets": _top_counts(datasets),
                "source_class": model_to_class.get(model, "unknown"),
                "source_task": model_to_task.get(model, "unknown"),
            })

    return pd.DataFrame(records)


def _top_counts(values: list[str], n: int = 3) -> str:
    """Summarize top-N value counts as a string."""
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}({v})" for k, v in sorted_counts[:n])


def compute_within_model_structure(
    df: pd.DataFrame,
    rank: int = 50,
    max_models: int | None = None,
) -> pd.DataFrame:
    """Analyze universal vs idiosyncratic subspace structure per model.

    For each model, splits dimensions into universal and idiosyncratic,
    then measures reconstruction contribution and inter-correlation.

    R² is computed via Gram matrices (rank x rank) to avoid allocating
    any (n_images x n_images) matrices.
    """
    components = load_components_for_rank(rank=rank, max_models=max_models)

    dim_info = df[["model", "dimension", "universality", "quadrant"]].copy()

    records = []
    for model, W in sorted(components.items()):
        model_dims = dim_info[dim_info["model"] == model].sort_values("dimension")
        if len(model_dims) == 0:
            continue

        is_idio = model_dims["quadrant"].str.startswith("idiosyncratic").values
        n_idio = int(is_idio.sum())
        n_univ = int((~is_idio).sum())

        r2_univ = _r2_subspace(W, ~is_idio) if n_univ > 0 else np.nan
        r2_idio = _r2_subspace(W, is_idio) if n_idio > 0 else np.nan

        # Inter-correlation among idiosyncratic dims
        if n_idio >= 2:
            W_idio = W[:, is_idio]
            W_idio_std = (W_idio - W_idio.mean(0)) / (W_idio.std(0) + 1e-8)
            corr_idio = (W_idio_std.T @ W_idio_std) / W.shape[0]
            triu_idx = np.triu_indices(n_idio, k=1)
            mean_inter_corr = float(np.abs(corr_idio[triu_idx]).mean())
        else:
            mean_inter_corr = np.nan

        # Norm fraction in each subspace
        total_norm = float(np.linalg.norm(W, "fro") ** 2)
        idio_norm_frac = float(np.linalg.norm(W[:, is_idio], "fro") ** 2 / total_norm) if n_idio > 0 else 0.0
        univ_norm_frac = float(np.linalg.norm(W[:, ~is_idio], "fro") ** 2 / total_norm) if n_univ > 0 else 0.0

        records.append({
            "model": model,
            "n_dims": W.shape[1],
            "n_idiosyncratic": n_idio,
            "n_universal": n_univ,
            "frac_idiosyncratic": n_idio / W.shape[1],
            "r2_universal_subspace": r2_univ,
            "r2_idiosyncratic_subspace": r2_idio,
            "idio_norm_fraction": idio_norm_frac,
            "univ_norm_fraction": univ_norm_frac,
            "mean_idio_inter_correlation": mean_inter_corr,
            "mean_universality": float(model_dims["universality"].mean()),
        })

    result = pd.DataFrame(records)

    meta = load_model_info()
    id_col = "feature_uid" if "feature_uid" in meta.columns else "name"
    meta_cols = ["model_class", "train_task", "train_data"]
    meta_sub = meta[[id_col] + meta_cols].rename(columns={id_col: "model"})
    meta_sub = meta_sub.drop_duplicates(subset="model")
    result = result.merge(meta_sub, on="model", how="left")

    log.info(f"Within-model structure computed for {len(result)} models")
    return result


def _r2_subspace(W: np.ndarray, col_mask: np.ndarray) -> float:
    """R² of column-subset reconstruction vs full, via Gram matrices only.

    Measures how well W[:, col_mask] @ W[:, col_mask].T approximates
    W @ W.T on the upper triangle. All computation uses (rank x rank)
    matrices — no (n_images x n_images) allocation.

    Key identity: for S = A @ A.T, ||S||²_F = ||A.T @ A||²_F.
    Column partition W = [W_sub | W_rest] gives S_full = S_sub + S_rest,
    so residual is S_rest = W_rest @ W_rest.T.
    """
    n = W.shape[0]
    n_pairs = n * (n - 1) / 2
    W_sub = W[:, col_mask]

    # Gram matrices (at most rank x rank)
    G_full = W.T @ W
    G_sub = W_sub.T @ W_sub
    C = W.T @ W_sub  # cross-Gram: trace(S_full @ S_sub) = ||C||²_F

    # ||S_full - S_sub||²_F = ||S_full||²_F - 2*trace(S_full @ S_sub) + ||S_sub||²_F
    frob_full = float((G_full ** 2).sum())
    frob_sub = float((G_sub ** 2).sum())
    frob_cross = float((C ** 2).sum())
    frob_res = frob_full - 2 * frob_cross + frob_sub

    # Diagonal elements (n-vectors, cheap)
    diag_full = (W ** 2).sum(axis=1)
    diag_sub = (W_sub ** 2).sum(axis=1)
    diag_res = diag_full - diag_sub

    # Upper-triangle sum: sum_upper(X²) = (||X||²_F - sum(diag(X)²)) / 2
    ss_res = (frob_res - float((diag_res ** 2).sum())) / 2

    # SS_tot for S_full on upper triangle
    sum_upper_full_sq = (frob_full - float((diag_full ** 2).sum())) / 2
    col_sums = W.sum(axis=0)
    sum_all = float((col_sums ** 2).sum())  # 1.T @ S_full @ 1
    sum_upper_full = (sum_all - float(diag_full.sum())) / 2
    mean_upper = sum_upper_full / n_pairs
    ss_tot = sum_upper_full_sq - n_pairs * mean_upper ** 2

    if ss_tot < 1e-10:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
