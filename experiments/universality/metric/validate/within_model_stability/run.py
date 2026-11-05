#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.shared.core import (
    cos2_permutation,
    cos2_perm_null_thresholds,
    null_adjust,
)

N_NULL_PERMS = 200

logger = logging.getLogger(__name__)


def load_seeds(
    embeddings_dir: Path, model: str, rank: int, sigma: float, seeds: list[int]
) -> list[np.ndarray]:
    """Load one W matrix per seed for a model at its selected sigma."""
    base = embeddings_dir / model / f"rank_{rank}" / f"sigma_{sigma}"
    ws = []
    for s in seeds:
        npz = base / f"seed_{s}" / "w.npz"
        npy = base / f"seed_{s}" / "w.npy"
        if npz.exists():
            ws.append(np.load(npz)["w"])
        elif npy.exists():
            ws.append(np.load(npy))
        else:
            raise FileNotFoundError(f"No W for {model} seed {s} at {base}")
    return ws


def compute_for_model(
    embeddings_dir: Path,
    model: str,
    rank: int,
    sigma: float,
    best_seed: int,
    seeds: list[int],
    n_null_perms: int = N_NULL_PERMS,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute the manuscript's all-pair, pair-null-calibrated stability ceiling."""
    ws = load_seeds(embeddings_dir, model, rank, sigma, seeds)
    n_dims = ws[0].shape[1]
    if not 0 <= best_seed < len(ws):
        raise ValueError(f"{model}: best_seed={best_seed} is outside 0..{len(ws) - 1}")

    pairs = list(itertools.combinations(range(len(ws)), 2))
    pair_scores = np.empty((len(pairs), n_dims))
    for i, (a, b) in enumerate(pairs):
        pair_scores[i] = cos2_permutation(ws[a], ws[b])
    raw_stability = pair_scores.mean(axis=0)

    # One null for the model: seed 0 as target, the remaining seeds as sources.
    thresholds = cos2_perm_null_thresholds(
        ws[0],
        ws[1:],
        n_perms=n_null_perms,
        seed=seed,
        n_source_samples=len(ws) - 1,
    )
    threshold = thresholds[0]

    adjusted_per_pair = np.empty_like(pair_scores)
    for i in range(len(pairs)):
        adjusted_per_pair[i] = null_adjust(pair_scores[i], threshold)
    adjusted_stability = adjusted_per_pair.mean(axis=0)


    return pd.DataFrame({
        "model": model,
        "dimension": np.arange(n_dims),
        "raw_stability": raw_stability,
        "null_threshold": threshold,
        "adjusted_stability": adjusted_stability,
    })


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    rank = cfg.factorization.rank
    dataset = cfg.dataset
    seeds = list(cfg.factorization.seeds)
    embeddings_dir = Path(cfg.data_root) / "embeddings" / dataset
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    sel = pd.read_csv(
        Path(cfg.data_root) / "meta" / dataset / f"bandwidth_selection_rank_{rank}.csv"
    )
    if cfg.models:
        sel = sel[sel["model"].isin(list(cfg.models))]
    logger.info(
        "Computing within-model stability for %d models at rank %d", len(sel), rank
    )

    dfs = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(compute_for_model)(
            embeddings_dir,
            row["model"],
            rank,
            row["sigma_mult"],
            int(row["best_seed"]),
            seeds,
            n_null_perms=N_NULL_PERMS,
            seed=cfg.seed + i,
        )
        for i, (_, row) in enumerate(sel.iterrows())
    )
    df = pd.concat(dfs, ignore_index=True)
    df["rank"] = rank
    df["dataset"] = dataset
    per_dim_path = out / f"per_dimension_rank_{rank}_{dataset}.csv"
    df.to_csv(per_dim_path, index=False)
    logger.info("Saved per-dim data to %s", per_dim_path)

    n_seeds = len(seeds)
    summary = {
        "rank": rank,
        "dataset": dataset,
        "n_models": int(df["model"].nunique()),
        "n_dimensions_total": len(df),
        "n_seeds_per_model": n_seeds,
        "n_seed_pairs_per_model": n_seeds * (n_seeds - 1) // 2,
        "n_null_perms": N_NULL_PERMS,
        "raw_stability": {
            "median": float(df["raw_stability"].median()),
            "mean": float(df["raw_stability"].mean()),
            "iqr_25": float(df["raw_stability"].quantile(0.25)),
            "iqr_75": float(df["raw_stability"].quantile(0.75)),
            "min": float(df["raw_stability"].min()),
            "max": float(df["raw_stability"].max()),
        },
        "adjusted_stability": {
            "median": float(df["adjusted_stability"].median()),
            "mean": float(df["adjusted_stability"].mean()),
            "iqr_25": float(df["adjusted_stability"].quantile(0.25)),
            "iqr_75": float(df["adjusted_stability"].quantile(0.75)),
            "min": float(df["adjusted_stability"].min()),
            "max": float(df["adjusted_stability"].max()),
        },
        "null_threshold": {
            "median": float(df["null_threshold"].median()),
        },
    }
    summary_path = out / f"summary_rank_{rank}_{dataset}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    print("\n" + "=" * 60)
    print(f"WITHIN-MODEL COS²-PERM STABILITY (rank={rank}, dataset={dataset})")
    print("=" * 60)
    print(
        f"  {summary['n_models']} models, {n_seeds} seeds each, "
        f"{summary['n_seed_pairs_per_model']} pairs per model"
    )
    print("\n  RAW within-model cos²-perm:")
    print(f"    median = {summary['raw_stability']['median']:.3f}")
    print(
        f"    IQR    = [{summary['raw_stability']['iqr_25']:.3f}, "
        f"{summary['raw_stability']['iqr_75']:.3f}]"
    )
    print(
        f"    range  = [{summary['raw_stability']['min']:.3f}, "
        f"{summary['raw_stability']['max']:.3f}]"
    )
    print(
        "\n  NULL-ADJUSTED within-model cos²-perm "
        "(directly comparable to universality):"
    )
    print(f"    median = {summary['adjusted_stability']['median']:.3f}")
    print(
        f"    IQR    = [{summary['adjusted_stability']['iqr_25']:.3f}, "
        f"{summary['adjusted_stability']['iqr_75']:.3f}]"
    )
    print(
        f"    range  = [{summary['adjusted_stability']['min']:.3f}, "
        f"{summary['adjusted_stability']['max']:.3f}]"
    )
    print(f"\n  Null threshold (median): {summary['null_threshold']['median']:.3f}")


if __name__ == "__main__":
    main()
