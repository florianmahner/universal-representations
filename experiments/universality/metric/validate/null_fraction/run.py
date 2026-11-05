#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import stats as sp_stats
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests

from experiments.universality.shared import load_components_for_rank, load_model_info
from experiments.universality.shared.core import l2_normalize_columns

N_PERM = 100

logger = logging.getLogger(__name__)


def cos2_perm_from_norm(t_norm: np.ndarray, s_norm: np.ndarray) -> np.ndarray:
    """Hungarian-matched cos² given pre-normalized matrices. Returns (n_dims,)."""
    c2 = (t_norm.T @ s_norm) ** 2
    row_ind, col_ind = linear_sum_assignment(-c2)
    scores = np.zeros(c2.shape[0])
    scores[row_ind] = c2[row_ind, col_ind]
    return scores


def process_one_target(
    target_name: str,
    components: dict[str, np.ndarray],
    n_perm: int,
    seed: int,
) -> dict:
    """Compute raw mean and null mean/std per dimension for one target model."""
    rng = np.random.default_rng(seed)
    W_target = components[target_name]
    n_items, n_dims = W_target.shape

    source_norms = [
        l2_normalize_columns(W) for name, W in components.items() if name != target_name
    ]

    t_norm = l2_normalize_columns(W_target)
    raw = np.mean([cos2_perm_from_norm(t_norm, s) for s in source_norms], axis=0)

    null_means = np.empty((n_perm, n_dims))
    for b in range(n_perm):
        t_shuf = l2_normalize_columns(W_target[rng.permutation(n_items)])
        null_means[b] = np.mean(
            [cos2_perm_from_norm(t_shuf, s) for s in source_norms], axis=0
        )

    return {
        "model": target_name,
        "raw": raw,
        "null_mean": null_means.mean(axis=0),
        "null_std": null_means.std(axis=0),
    }


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    dataset = cfg.dataset

    meta = load_model_info()
    models = meta["feature_uid"].astype(str).tolist()
    components = load_components_for_rank(rank=rank, models=models, dataset=dataset)
    model_names = list(components.keys())
    logger.info(f"Rank {rank}, {len(model_names)} models, n_perm={N_PERM}")

    seeds = [cfg.seed + i for i in range(len(model_names))]
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(process_one_target)(name, components, N_PERM, s)
        for name, s in zip(model_names, seeds)
    )

    rows = []
    for res in results:
        raw, nm, ns = res["raw"], res["null_mean"], res["null_std"]
        z = np.where(ns > 1e-10, (raw - nm) / ns, 0.0)
        p_one_sided = sp_stats.norm.sf(z)  # P(Z > z) under null
        for d in range(len(raw)):
            rows.append({
                "model": res["model"],
                "dimension": d,
                "raw": float(raw[d]),
                "null_mean": float(nm[d]),
                "null_std": float(ns[d]),
                "zscore": float(z[d]),
                "p_value": float(p_one_sided[d]),
            })

    df = pd.DataFrame(rows)
    n_total = len(df)

    _, p_fdr, _, _ = multipletests(df["p_value"].values, alpha=0.05, method="fdr_bh")
    _, p_bonf, _, _ = multipletests(df["p_value"].values, alpha=0.05, method="bonferroni")
    df["p_fdr"] = p_fdr
    df["p_bonferroni"] = p_bonf
    df["sig_nominal"] = df["p_value"] < 0.05
    df["sig_fdr"] = p_fdr < 0.05
    df["sig_bonferroni"] = p_bonf < 0.05
    df["rank"] = rank
    df["dataset"] = dataset
    per_dim_path = out / f"per_dimension_rank_{rank}_{dataset}.csv"
    df.to_csv(per_dim_path, index=False)
    logger.info(f"Saved per-dim data to {per_dim_path}")

    per_model = df.groupby("model").agg(
        raw_mean=("raw", "mean"),
        frac_sig_nominal=("sig_nominal", "mean"),
        frac_sig_fdr=("sig_fdr", "mean"),
        frac_sig_bonf=("sig_bonferroni", "mean"),
    ).reset_index()
    per_model.to_csv(out / f"per_model_rank_{rank}_{dataset}.csv", index=False)

    summary = {
        "rank": rank,
        "dataset": dataset,
        "n_models": len(model_names),
        "n_dimensions_total": n_total,
        "n_perm": N_PERM,
        "nominal": {
            "alpha": 0.05,
            "n_sig": int(df["sig_nominal"].sum()),
            "frac_sig": float(df["sig_nominal"].mean()),
            "pct_sig": round(100 * float(df["sig_nominal"].mean()), 1),
        },
        "fdr_bh": {
            "alpha": 0.05,
            "n_sig": int(df["sig_fdr"].sum()),
            "frac_sig": float(df["sig_fdr"].mean()),
            "pct_sig": round(100 * float(df["sig_fdr"].mean()), 1),
        },
        "bonferroni": {
            "alpha": 0.05,
            "n_sig": int(df["sig_bonferroni"].sum()),
            "frac_sig": float(df["sig_bonferroni"].mean()),
            "pct_sig": round(100 * float(df["sig_bonferroni"].mean()), 1),
        },
        "raw_cos2_perm": {
            "median": float(df["raw"].median()),
            "iqr_25": float(df["raw"].quantile(0.25)),
            "iqr_75": float(df["raw"].quantile(0.75)),
            "min": float(df["raw"].min()),
            "max": float(df["raw"].max()),
        },
        "null_mean": {
            "median": float(df["null_mean"].median()),
        },
        "per_model_frac_sig_fdr": {
            "median": float(per_model["frac_sig_fdr"].median()),
            "min": float(per_model["frac_sig_fdr"].min()),
            "max": float(per_model["frac_sig_fdr"].max()),
            "n_models_100pct": int((per_model["frac_sig_fdr"] == 1.0).sum()),
            "n_models_0pct": int((per_model["frac_sig_fdr"] == 0.0).sum()),
        },
    }

    summary_path = out / f"summary_rank_{rank}_{dataset}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print(f"NULL EXCEEDANCE (rank={rank}, dataset={dataset}, n_perm={N_PERM})")
    print("=" * 60)
    print(f"Total dimensions: {n_total}  ({len(model_names)} models x {rank} dims)")
    print(f"  Nominal   (p<.05):        {summary['nominal']['n_sig']:>5d}  ({summary['nominal']['pct_sig']}%)")
    print(f"  FDR-BH    (q<.05):        {summary['fdr_bh']['n_sig']:>5d}  ({summary['fdr_bh']['pct_sig']}%)")
    print(f"  Bonferroni (p<.05/N):     {summary['bonferroni']['n_sig']:>5d}  ({summary['bonferroni']['pct_sig']}%)")
    print(f"\nRaw cos²-perm: median={summary['raw_cos2_perm']['median']:.3f}  "
          f"IQR=[{summary['raw_cos2_perm']['iqr_25']:.3f}, {summary['raw_cos2_perm']['iqr_75']:.3f}]  "
          f"max={summary['raw_cos2_perm']['max']:.3f}")
    print(f"Null mean (median): {summary['null_mean']['median']:.3f}")
    print(f"Per-model frac sig (FDR): median={summary['per_model_frac_sig_fdr']['median']:.2f}  "
          f"min={summary['per_model_frac_sig_fdr']['min']:.2f}  "
          f"max={summary['per_model_frac_sig_fdr']['max']:.2f}")


if __name__ == "__main__":
    main()
