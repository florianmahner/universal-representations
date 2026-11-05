#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from src.stability import factorization_stability, reference_seed_index

MANUSCRIPT_SIGMA_GRID = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1.0"]

logger = logging.getLogger(__name__)


def compute_model_sigma(
    model: str,
    sigma: str,
    rank: int,
    embeddings_dir: Path,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Score one model x sigma cell by cross-seed stability and explained variance."""
    seeds = list(seeds)
    base = embeddings_dir / model / f"rank_{rank}" / f"sigma_{sigma}"

    ws, evars, loaded_seeds = [], [], []
    missing = []
    for seed in seeds:
        seed_dir = base / f"seed_{seed}"
        npz_path = seed_dir / "w.npz"
        npy_path = seed_dir / "w.npy"
        if npz_path.exists():
            with np.load(npz_path) as archive:
                w = archive["w"]
        elif npy_path.exists():
            w = np.load(npy_path)
        else:
            missing.append(f"seed {seed}: W")
            continue
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            missing.append(f"seed {seed}: metrics.json")
            continue
        evar = float(json.loads(metrics_path.read_text()).get("final_evar", np.nan))
        if not np.isfinite(evar):
            missing.append(f"seed {seed}: finite final_evar")
            continue
        if w.ndim != 2 or w.shape[1] != rank:
            raise ValueError(
                f"{seed_dir}: expected W with rank {rank}, got shape {w.shape}"
            )
        ws.append(w)
        evars.append(evar)
        loaded_seeds.append(seed)

    if missing or loaded_seeds != seeds:
        details = ", ".join(missing) if missing else f"loaded seeds {loaded_seeds}"
        raise FileNotFoundError(
            f"Incomplete five-seed grid for model={model}, sigma={sigma}: {details}"
        )

    stab = factorization_stability(ws)
    mean_stability = float(stab.mean())
    mean_evar = float(np.mean(evars))
    harmonic = 2 * mean_stability * mean_evar / (mean_stability + mean_evar) if (mean_stability > 0 and mean_evar > 0) else 0.0
    ref_seed = loaded_seeds[reference_seed_index(ws)]

    return {
        "model": model,
        "sigma_mult": sigma,
        "n_seeds": len(ws),
        "stability": mean_stability,
        "evar": mean_evar,
        "harmonic": harmonic,
        "best_seed": ref_seed,
    }


def list_models_with_sigma_data(
    rank: int,
    sigmas: list[str],
    embeddings_dir: Path,
    metadata_path: Path,
) -> list[str]:
    """Return every model in the metadata table that has a full sigma grid on disk."""
    meta = pd.read_csv(metadata_path)
    models = sorted(set(meta["feature_uid"].astype(str)))
    missing = [
        embeddings_dir / model / f"rank_{rank}" / f"sigma_{sigma}"
        for model in models
        for sigma in sigmas
        if not (
            embeddings_dir / model / f"rank_{rank}" / f"sigma_{sigma}"
        ).is_dir()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Incomplete model × sigma directory grid: {len(missing)} missing "
            f"(first: {preview})"
        )
    return models


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rank = cfg.factorization.rank
    seeds = list(cfg.factorization.seeds)
    sigmas = list(dict.fromkeys(cfg.factorization.sigma_grid))
    embeddings_dir = Path(cfg.data_root) / "embeddings" / cfg.dataset
    meta_dir = Path(cfg.data_root) / "meta" / cfg.dataset
    meta_dir.mkdir(parents=True, exist_ok=True)

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank_out = out / cfg.dataset / f"rank_{rank}"
    rank_out.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset=%s  Sigma grid for rank %d: %s", cfg.dataset, rank, sigmas)
    models = list_models_with_sigma_data(
        rank, sigmas, embeddings_dir, meta_dir / "model_metadata.csv"
    )
    logger.info("Found %d models for rank %d", len(models), rank)
    if not models:
        raise FileNotFoundError(
            f"No complete eight-sigma model directories found in {embeddings_dir}"
        )

    tasks = [(m, s) for m in models for s in sigmas]
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(compute_model_sigma)(m, s, rank, embeddings_dir, seeds) for m, s in tasks
    )
    df_all = pd.DataFrame(results)
    df_all.to_csv(rank_out / "stability_evar_all.csv", index=False)
    logger.info("Saved stability_evar_all.csv (%d rows)", len(df_all))

    best_idx = df_all.groupby("model")["harmonic"].idxmax()
    df_best = df_all.loc[
        best_idx,
        [
            "model",
            "sigma_mult",
            "best_seed",
            "n_seeds",
            "stability",
            "evar",
            "harmonic",
        ],
    ].copy()
    df_best = df_best.sort_values("model").reset_index(drop=True)
    df_best["stability_protocol"] = "all_pairwise_aligned_fisher_z"
    df_best["sigma_grid"] = "|".join(sigmas)
    df_best["manuscript_grid_complete"] = sigmas == MANUSCRIPT_SIGMA_GRID

    output_csv = meta_dir / f"bandwidth_selection_rank_{rank}.csv"
    df_best.to_csv(output_csv, index=False)
    logger.info("Saved %s (%d models)", output_csv, len(df_best))

    counts = df_best["sigma_mult"].value_counts().sort_index()
    logger.info("Sigma distribution:\n%s", counts.to_string())
    logger.info("Mean stability=%.3f  evar=%.3f  harmonic=%.3f",
                df_best["stability"].mean(), df_best["evar"].mean(), df_best["harmonic"].mean())


if __name__ == "__main__":
    main()
