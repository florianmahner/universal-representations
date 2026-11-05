#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.universality.shared import load_components_for_rank
from experiments.universality.shared.core import (
    cos2_perm_null_thresholds,
    cos2_permutation,
    max_correlation,
    null_adjust,
)

N_MODELS = 15
N_NULL_PERMS = 1000

logger = logging.getLogger(__name__)


def ridge_r2(
    W_source: np.ndarray,
    W_target: np.ndarray,
    alphas: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Held-out R² for each target dimension using nested ridge selection."""
    if alphas is None:
        alphas = np.logspace(-3, 3, 10)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    r2_scores = np.zeros(W_target.shape[1])
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i in range(W_target.shape[1]):
        # Scaling is refit inside every outer training fold. RidgeCV selects
        # alpha from the training fold; the returned predictions are strictly
        # out of sample.
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
        prediction = cross_val_predict(
            model,
            W_source,
            W_target[:, i],
            cv=folds,
            n_jobs=1,
        )
        r2_scores[i] = r2_score(W_target[:, i], prediction)
    return r2_scores


def compute_universality_for_target(
    target_name: str,
    components: dict[str, np.ndarray],
    n_null_perms: int = 1000,
    ridge_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Compute per-dimension universality using all three metrics for one target."""
    w_target = components[target_name]
    n_dims = w_target.shape[1]
    sources = {name: w for name, w in components.items() if name != target_name}

    max_corr_scores = []
    cos2_scores = []
    ridge_scores = []
    for source_index, w_source in enumerate(sources.values()):
        max_corr_scores.append(max_correlation(w_source, w_target))
        cos2_scores.append(cos2_permutation(w_source, w_target))
        ridge_scores.append(
            ridge_r2(
                w_source,
                w_target,
                n_splits=ridge_folds,
                seed=seed + source_index,
            )
        )

    thresholds = cos2_perm_null_thresholds(
        w_target,
        list(sources.values()),
        n_perms=n_null_perms,
        seed=seed,
    )

    cos2_raw = np.stack(cos2_scores)
    cos2_adj = np.empty_like(cos2_raw)
    for d in range(n_dims):
        cos2_adj[:, d] = null_adjust(cos2_raw[:, d], thresholds[:, d])

    mc_univ = np.mean(max_corr_scores, axis=0)
    adj_univ = cos2_adj.mean(axis=0)
    ridge_univ = np.mean(ridge_scores, axis=0)

    rows = []
    for d in range(n_dims):
        rows.append({
            "model": target_name,
            "dimension": d,
            "max_corr": float(mc_univ[d]),
            "cos2_perm_adj": float(adj_univ[d]),
            "ridge_r2": float(ridge_univ[d]),
        })
    return rows


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    ridge_folds = 5
    components = load_components_for_rank(
        rank=cfg.factorization.rank,
        models=list(cfg.models) or None,
        max_models=N_MODELS,
        dataset=cfg.dataset,
    )
    model_names = list(components.keys())
    logger.info(f"Validating on {len(model_names)} models, rank {cfg.factorization.rank}")

    all_rows = []
    for i, name in enumerate(model_names):
        logger.info(f"  [{i+1}/{len(model_names)}] {name}")
        all_rows.extend(compute_universality_for_target(
            name,
            components,
            n_null_perms=N_NULL_PERMS,
            ridge_folds=ridge_folds,
            seed=cfg.seed + i,
        ))

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "metric_validation.csv", index=False)

    for metric in ("max_corr", "cos2_perm_adj"):
        r, _ = stats.pearsonr(df[metric], df["ridge_r2"])
        rho, _ = stats.spearmanr(df[metric], df["ridge_r2"])
        logger.info(f"  {metric} vs ridge_r2: r={r:.3f}, ρ={rho:.3f}")

    params = {
        "n_models": len(model_names),
        "rank": cfg.factorization.rank,
        "n_null_perms": N_NULL_PERMS,
        "ridge_folds": ridge_folds,
        "seed": cfg.seed,
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))
    logger.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
