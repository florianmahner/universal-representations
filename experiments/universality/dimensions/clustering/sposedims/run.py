from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..characterization.analysis import load_data

log = logging.getLogger(__name__)


def load_spose(
    categories_22k: list[str],
    data_root: Path = Path("data"),
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load SPoSE embedding expanded to 22k image level.

    Returns:
        X: (22248, 68) array — SPoSE scores repeated per category
        dim_names: list of SPoSE dimension column names
        metadata: DataFrame with label, quality, concept, avg_interpretability per dim
    """
    behavior_dir = Path(data_root) / "embeddings" / "things" / "human_behavior"
    df = pd.read_csv(behavior_dir / "human_embedding.csv")
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    cat_to_row = {name: i for i, name in enumerate(df["object_name"])}
    H = df[dim_cols].values  # (1854, 68)

    # Expand to 22k: each image gets its category's SPoSE scores
    X = np.empty((len(categories_22k), len(dim_cols)), dtype=np.float32)
    for i, cat in enumerate(categories_22k):
        X[i] = H[cat_to_row[cat]]

    metadata = pd.read_csv(behavior_dir / "dimension_metadata.csv")
    return X, dim_cols, metadata


def _r2_only(y: np.ndarray, X: np.ndarray, alphas: np.ndarray) -> float:
    """Cross-validated ridge R² for a feature subset."""
    return float(np.clip(
        cross_val_score(RidgeCV(alphas=alphas), X, y, cv=5, scoring="r2").mean(),
        0, 1,
    ))


def _score_one_dim(
    dim_idx: int,
    y: np.ndarray,
    X: np.ndarray,
    alphas: np.ndarray,
) -> dict:
    """Score one NMF dim: ridge R² from all 68 SPoSE dims + per-dim correlations."""
    ridge = RidgeCV(alphas=alphas)
    r2 = float(np.clip(
        cross_val_score(ridge, X, y, cv=5, scoring="r2").mean(), 0, 1
    ))

    # Per-SPoSE-dim Pearson correlation
    y_centered = y - y.mean()
    y_std = y_centered.std()
    corrs = np.zeros(X.shape[1], dtype=np.float32)
    if y_std > 1e-12:
        for j in range(X.shape[1]):
            x_centered = X[:, j] - X[:, j].mean()
            x_std = x_centered.std()
            if x_std > 1e-12:
                corrs[j] = (x_centered * y_centered).mean() / (x_std * y_std)

    return {"dim_idx": dim_idx, "r2": r2, "corrs": corrs}


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    # 1. Load NMF data
    log.info("Loading NMF data (rank=%d)", cfg.factorization.rank)
    W_all, dim_df, ds = load_data(rank=cfg.factorization.rank)
    log.info("W_all: %s, dims: %d", W_all.shape, len(dim_df))

    # 2. Load SPoSE expanded to 22k
    log.info("Loading SPoSE embedding")
    X_raw, spose_names, spose_meta = load_spose(ds.categories, Path(cfg.data_root))
    log.info("SPoSE: %s (%d dims)", X_raw.shape, len(spose_names))

    # 3. Standardize SPoSE features
    X = StandardScaler().fit_transform(X_raw)

    # 4. Score all NMF dims
    alphas = np.logspace(-2, 4, 7)
    n_dims = W_all.shape[0]
    log.info("Scoring %d NMF dimensions with %d parallel jobs", n_dims, cfg.n_jobs)
    t0 = time.perf_counter()

    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(_score_one_dim)(i, W_all[i], X, alphas)
        for i in range(n_dims)
    )
    log.info("Scoring time: %.0fs", time.perf_counter() - t0)

    # 5. Build outputs
    r2_values = np.array([r["r2"] for r in results])
    corr_matrix = np.array([r["corrs"] for r in results])  # (n_dims, 68)

    # Per-dimension R² + metadata
    dim_out = dim_df.copy()
    dim_out["spose_r2"] = r2_values
    # Top-3 most correlated SPoSE dims per NMF dim (with metadata)
    meta_lookup = spose_meta.set_index("dim")
    for i in range(len(results)):
        top3 = np.argsort(-np.abs(corr_matrix[i]))[:3]
        for k, prefix in enumerate(["top1", "top2", "top3"]):
            idx = top3[k]
            dim_out.loc[i, f"{prefix}_spose"] = spose_names[idx]
            dim_out.loc[i, f"{prefix}_corr"] = corr_matrix[i, idx]
            dim_out.loc[i, f"{prefix}_label"] = meta_lookup.loc[idx, "label"]
            dim_out.loc[i, f"{prefix}_quality"] = meta_lookup.loc[idx, "quality"]
            dim_out.loc[i, f"{prefix}_interpretability"] = float(
                meta_lookup.loc[idx, "avg_interpretability"]
            )
    # Quality-specific R² (ridge using only SPoSE dims of each type)
    quality_col = {
        "semantic": "spose_r2_semantic",
        "visual": "spose_r2_visual",
        "mix visual-semantic": "spose_r2_mix",
        "unclear": "spose_r2_unclear",
    }
    log.info("Computing quality-specific R²")
    t1 = time.perf_counter()
    for quality, col_name in quality_col.items():
        q_idx = spose_meta.index[spose_meta["quality"] == quality].values
        if len(q_idx) == 0:
            dim_out[col_name] = 0.0
            continue
        X_q = X[:, q_idx]
        log.info("  %s: %d SPoSE dims", quality, len(q_idx))
        r2_q = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
            delayed(_r2_only)(W_all[i], X_q, alphas) for i in range(n_dims)
        )
        dim_out[col_name] = r2_q
    log.info("Quality-specific R² time: %.0fs", time.perf_counter() - t1)

    dim_out.to_csv(out / "dim_r2.csv", index=False)
    log.info("Saved dim_r2.csv: %d rows", len(dim_out))

    # Full correlation matrix
    corr_df = pd.DataFrame(corr_matrix, columns=spose_names)
    corr_df.to_csv(out / "spose_correlations.csv", index=False)
    log.info("Saved spose_correlations.csv: %s", corr_matrix.shape)

    # 6. Aggregate per cluster
    records = []
    for cluster_id, group in dim_out.groupby("cluster"):
        idx = group.index.values
        mean_corr = corr_matrix[idx].mean(axis=0)
        top_spose = np.argsort(-np.abs(mean_corr))[:3]

        row = {
            "cluster": cluster_id,
            "size": len(group),
            "mean_universality": group["universality"].mean(),
            "mean_spose_r2": group["spose_r2"].mean(),
            "median_spose_r2": group["spose_r2"].median(),
        }
        for k, prefix in enumerate(["top1", "top2", "top3"]):
            idx = top_spose[k]
            row[f"{prefix}_spose"] = spose_names[idx]
            row[f"{prefix}_label"] = meta_lookup.loc[idx, "label"]
            row[f"{prefix}_quality"] = meta_lookup.loc[idx, "quality"]
            row[f"{prefix}_interpretability"] = float(
                meta_lookup.loc[idx, "avg_interpretability"]
            )
            row[f"{prefix}_mean_corr"] = mean_corr[idx]
        records.append(row)

    cluster_df = pd.DataFrame(records).sort_values("mean_universality", ascending=False)
    cluster_df.to_csv(out / "cluster_spose.csv", index=False)
    log.info("Saved cluster_spose.csv: %d clusters", len(cluster_df))

    # Summary
    log.info("\n=== SPoSE R² summary ===")
    log.info("  mean=%.4f  median=%.4f  std=%.4f  max=%.4f",
             r2_values.mean(), np.median(r2_values),
             r2_values.std(), r2_values.max())
    log.info("  dims with R²>0.10: %d (%.1f%%)",
             (r2_values > 0.10).sum(),
             100 * (r2_values > 0.10).mean())
    log.info("  dims with R²>0.30: %d (%.1f%%)",
             (r2_values > 0.30).sum(),
             100 * (r2_values > 0.30).mean())

    log.info("Done. Outputs in %s", out)


if __name__ == "__main__":
    main()
