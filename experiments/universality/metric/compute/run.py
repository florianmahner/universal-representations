#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.shared import (
    load_components_for_rank,
    load_model_info,
)
from experiments.universality.shared.core import (
    METRICS,
    cos2_perm_null_thresholds,
    null_adjust,
)

RANKS = [10, 30, 50, 100, 200]
N_NULL_PERMS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_universality_for_model(
    target_name: str,
    components: dict[str, np.ndarray],
    n_null_perms: int = 1000,
    null_seed: int = 42,
) -> list[dict]:
    """Compute per-dimension universality for one target model.

    Computes all raw metrics from METRICS, plus null-adjusted cos2_perm.
    """
    w_target = components[target_name]
    n_dims = w_target.shape[1]

    sources = [(name, w) for name, w in components.items() if name != target_name]
    n_others = len(sources)

    all_scores = {name: np.empty((n_others, n_dims)) for name in METRICS}
    for i, (_, w_source) in enumerate(sources):
        for metric_name, metric_fn in METRICS.items():
            all_scores[metric_name][i] = metric_fn(w_source, w_target)

    thresholds = cos2_perm_null_thresholds(
        w_target,
        [w for _, w in sources],
        n_perms=n_null_perms,
        seed=null_seed,
    )

    rows = []
    for metric_name in METRICS:
        scores = all_scores[metric_name]
        for d in range(n_dims):
            rows.append({
                "model": target_name,
                "dimension": d,
                "metric": metric_name,
                "universality": float(scores[:, d].mean()),
                "std": float(scores[:, d].std()),
                "min": float(scores[:, d].min()),
                "max": float(scores[:, d].max()),
            })

    cos2_scores = all_scores["cos2_perm"]
    for d in range(n_dims):
        adj = null_adjust(cos2_scores[:, d], thresholds[:, d])
        rows.append({
            "model": target_name,
            "dimension": d,
            "metric": "cos2_perm_adj",
            "universality": float(adj.mean()),
            "std": float(adj.std()),
            "min": float(adj.min()),
            "max": float(adj.max()),
        })

    return rows


def get_dnn_models() -> list[str]:
    """Return model names from metadata (excludes non-DNN entries like monkey IT)."""
    meta = load_model_info()
    return meta["feature_uid"].astype(str).tolist()


def compute_universality_for_rank(
    rank: int,
    max_models: int | None = None,
    n_jobs: int = 1,
    n_null_perms: int = 1000,
    seed: int = 42,
    dataset: str = "things22k",
) -> pd.DataFrame:
    """Compute per-dimension universality for all DNN models at one rank."""
    dnn_models = get_dnn_models()
    components = load_components_for_rank(
        rank=rank, models=dnn_models, max_models=max_models, dataset=dataset,
    )
    model_names = list(components.keys())
    logger.info(f"Rank {rank}: {len(model_names)} models loaded")

    if n_jobs == 1:
        rows = []
        for i, name in enumerate(model_names):
            if (i + 1) % 20 == 0:
                logger.info(f"  Rank {rank}: {i + 1}/{len(model_names)}")
            rows.extend(compute_universality_for_model(
                name, components, n_null_perms=n_null_perms, null_seed=seed + i,
            ))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(compute_universality_for_model)(
                name, components, n_null_perms=n_null_perms, null_seed=seed + i,
            )
            for i, name in enumerate(model_names)
        )
        rows = [r for batch in results for r in batch]

    df = pd.DataFrame(rows)
    df["rank"] = rank
    return df


def load_all_ranks(dataset: str = "things22k", metric: str = "cos2_perm") -> pd.DataFrame:
    """Load every per-rank universality table written for a dataset."""
    tables = Path("experiments/universality/shared/data")
    prefix = f"universality_{dataset}_rank_"
    csvs = sorted(p for p in tables.glob(f"{prefix}*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No {prefix}*.csv found in {tables}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    return df[df["metric"] == metric]


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    tables = Path("experiments/universality/shared/data")
    tables.mkdir(parents=True, exist_ok=True)

    for rank in sorted(RANKS):
        logger.info(f"Computing universality for rank {rank} (dataset={cfg.dataset})...")
        df = compute_universality_for_rank(
            rank, n_jobs=cfg.n_jobs, n_null_perms=N_NULL_PERMS,
            seed=cfg.seed, dataset=cfg.dataset,
        )

        if df.empty:
            logger.warning(f"  Rank {rank}: no models loaded, skipping")
            continue

        out_path = tables / f"universality_{cfg.dataset}_rank_{rank}.csv"
        df.to_csv(out_path, index=False)

        for metric_name in list(METRICS) + ["cos2_perm_adj"]:
            sub = df[df["metric"] == metric_name]
            if not sub.empty:
                mean_u = sub["universality"].mean()
                logger.info(f"  Rank {rank} [{metric_name}]: mean={mean_u:.3f}")
        logger.info(f"  Saved to {out_path.name}")

    logger.info("\nSummary across all available ranks (cos2_perm_adj):")
    df_all = load_all_ranks(dataset=cfg.dataset, metric="cos2_perm_adj")
    summary = df_all.groupby("rank")["universality"].agg(["mean", "std", "median"]).round(3)
    print(summary)


if __name__ == "__main__":
    main()
