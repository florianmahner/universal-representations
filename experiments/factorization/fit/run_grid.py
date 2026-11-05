#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.factorization import get_srf_class
from src.metrics import explained_variance, median_sigma_heuristic

LOG = logging.getLogger(__name__)

BLOCK_SIZE = 2048
SIGMA_SUBSAMPLE = None
SKIP_EXISTING = True


def rbf_similarity(features: np.ndarray, sigma: float, block_size: int) -> np.ndarray:
    """Build an RBF similarity matrix with one row block of temporary storage."""
    n_items = features.shape[0]
    squared_norms = np.einsum("ij,ij->i", features, features)
    similarity = np.empty((n_items, n_items), dtype=np.float64)
    scale = -1.0 / (2.0 * sigma * sigma)
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        distances = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - 2.0 * (features[start:stop] @ features.T)
        )
        np.maximum(distances, 0.0, out=distances)
        distances *= scale
        np.exp(distances, out=distances)
        similarity[start:stop] = distances
    return similarity


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    feature_dir = Path(cfg.data_root) / "features" / cfg.dataset
    models = list(cfg.models) or sorted(p.stem for p in feature_dir.glob("*.npy"))
    if not models:
        raise FileNotFoundError(f"no feature files in {feature_dir}")
    for model in models:
        fit_model(cfg, feature_dir / f"{model}.npy", model)


def fit_model(cfg: DictConfig, feature_path: Path, model: str) -> None:
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    # Match the factorizer's numerical contract and make the kernel invariant
    # to whether an extractor happened to serialize float16/32/64 features.
    features = np.asarray(np.load(feature_path), dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"Expected a finite 2D feature matrix, got {features.shape}")

    sigma_features = features
    if SIGMA_SUBSAMPLE is not None:
        if SIGMA_SUBSAMPLE < 2:
            raise ValueError("sigma_subsample must be at least 2")
        rng = np.random.default_rng(0)
        n_sigma = min(SIGMA_SUBSAMPLE, len(features))
        sigma_indices = rng.choice(len(features), n_sigma, replace=False)
        sigma_features = features[sigma_indices]
        LOG.warning(
            "Estimating sigma from %d/%d rows; this does not reproduce the "
            "manuscript's full-dataset median",
            n_sigma,
            len(features),
        )
    median_sigma = float(median_sigma_heuristic(sigma_features))
    rank = cfg.factorization.rank
    output_base = Path(cfg.data_root) / "embeddings" / cfg.dataset / model / f"rank_{rank}"
    relative_feature_path = feature_path.as_posix()

    SRF = get_srf_class()

    for sigma_token in map(str, cfg.factorization.sigma_grid):
        sigma = median_sigma * float(sigma_token)
        expected = [
            output_base / f"sigma_{sigma_token}" / f"seed_{seed}" / "w.npz"
            for seed in cfg.factorization.seeds
        ]
        if SKIP_EXISTING and all(
            path.exists() and path.with_name("metrics.json").exists()
            for path in expected
        ):
            LOG.info("sigma=%s: all seeds already exist", sigma_token)
            continue

        kernel_start = time.perf_counter()
        similarity = rbf_similarity(features, sigma, BLOCK_SIZE)
        kernel_seconds = time.perf_counter() - kernel_start

        for seed, w_path in zip(cfg.factorization.seeds, expected):
            metrics_path = w_path.with_name("metrics.json")
            if SKIP_EXISTING and w_path.exists() and metrics_path.exists():
                continue

            estimator = SRF(
                rank=rank,
                max_outer=cfg.factorization.max_outer,
                max_inner=cfg.factorization.max_inner,
                tol=cfg.factorization.tol,
                init="random_sqrt",
                random_state=seed,
                verbose=0,
            )
            fit_start = time.perf_counter()
            estimator.fit(similarity)
            fit_seconds = time.perf_counter() - fit_start

            w_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(w_path, w=estimator.w_)
            metrics = {
                "model": model,
                "dataset": cfg.dataset,
                "feature_path": relative_feature_path,
                "n_samples": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "rank": rank,
                "median_sigma": median_sigma,
                "sigma_mult": sigma_token,
                "sigma": sigma,
                "seed": seed,
                "max_outer": cfg.factorization.max_outer,
                "max_inner": cfg.factorization.max_inner,
                "tol": cfg.factorization.tol,
                "n_iter": int(estimator.n_iter_),
                "final_evar": explained_variance(similarity, estimator.w_),
                "kernel_time_s": kernel_seconds,
                "fit_time_s": fit_seconds,
            }
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            LOG.info(
                "sigma=%s seed=%d evar=%.4f",
                sigma_token,
                seed,
                metrics["final_evar"],
            )


if __name__ == "__main__":
    main()
