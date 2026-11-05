#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import stats

from experiments.universality.shared.core import max_correlation
from experiments.universality.shared.utils import load_components_for_rank
from src.cka import center_gram
from src.metrics import gaussian_kernel_similarity


logger = logging.getLogger(__name__)


def compute_centered_kernel(features: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute centered RBF kernel and its Frobenius norm.

    This nonlinear CKA validation uses a median-heuristic RBF independently
    within each feature space.
    """
    gram = gaussian_kernel_similarity(features, features)
    centered = center_gram(gram).astype(np.float32)
    norm = float(np.linalg.norm(centered))
    return centered, norm


def _compute_kernel_for_model(
    model: str, feature_path: Path,
) -> tuple[str, np.ndarray, float]:
    """Load features and compute centered kernel for one model."""
    features = np.load(feature_path)
    centered, norm = compute_centered_kernel(features)
    return model, centered, norm


def precompute_kernels(
    models: list[str], feature_dir: Path, n_jobs: int = 1,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Precompute centered kernels and norms for all models."""
    logger.info(f"Computing {len(models)} kernels with {n_jobs} parallel jobs")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_compute_kernel_for_model)(m, feature_dir / f"{m}.npy") for m in models
    )

    kernels: list[np.ndarray] = []
    norms = []
    loaded_models: list[str] = []
    for r in results:
        model, centered, norm = r
        kernels.append(centered)
        norms.append(norm)
        loaded_models.append(model)

    return kernels, np.array(norms), loaded_models


def compute_cka_matrix(
    kernels: list[np.ndarray], norms: np.ndarray
) -> np.ndarray:
    """Compute pairwise CKA from precomputed centered kernels and norms."""
    m = len(kernels)
    cka_mat = np.eye(m, dtype=np.float32)

    n_pairs = m * (m - 1) // 2
    done = 0
    for i in range(m):
        ki = kernels[i].ravel()
        for j in range(i + 1, m):
            kj = kernels[j].ravel()
            val = float(ki.dot(kj)) / (norms[i] * norms[j])
            cka_mat[i, j] = val
            cka_mat[j, i] = val
            done += 1
            if done % 500 == 0:
                logger.info(f"CKA pairs: {done}/{n_pairs}")

    logger.info(f"CKA matrix complete: {m}x{m}")
    return cka_mat


def compute_universality_pairwise(
    components: dict[str, np.ndarray], models: list[str]
) -> np.ndarray:
    """Symmetrized pairwise universality from the W matrices.

    For pair (A, B): mean_k max_j |corr(w_{A,k}, w_{B,j})|, symmetrized.
    """
    m = len(models)
    raw = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                raw[i, j] = 1.0
                continue
            scores = max_correlation(components[models[i]], components[models[j]])
            raw[i, j] = float(scores.mean())
    logger.info(f"Universality pairwise matrix complete: {m}x{m}")
    return (raw + raw.T) / 2.0


def model_level_summary(
    cka_mat: np.ndarray,
    univ_mat: np.ndarray,
    models: list[str],
) -> pd.DataFrame:
    """Per-model mean CKA and mean universality, excluding the diagonal."""
    m = len(models)
    mask = ~np.eye(m, dtype=bool)
    return pd.DataFrame([
        {
            "model": model,
            "mean_cka": float(cka_mat[i][mask[i]].mean()),
            "mean_universality": float(univ_mat[i][mask[i]].mean()),
        }
        for i, model in enumerate(models)
    ])

def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    meta = pd.read_csv(
        Path(cfg.data_root) / "meta" / "things22k" / "model_metadata.csv", comment="#"
    )
    models = list(cfg.models) or sorted(meta["feature_uid"].astype(str).tolist())
    logger.info(f"Models: {len(models)}, rank: {cfg.factorization.rank}, dataset: {cfg.dataset}")

    logger.info("Phase 1: Precomputing centered RBF kernels...")
    kernels, norms, loaded_models = precompute_kernels(
        models, Path(cfg.data_root) / "features" / cfg.dataset, n_jobs=cfg.n_jobs
    )
    logger.info(f"Loaded {len(loaded_models)}/{len(models)} models")

    logger.info("Phase 2: Computing pairwise CKA matrix...")
    cka_mat = compute_cka_matrix(kernels, norms)

    del kernels
    del norms

    logger.info("Phase 3: Computing pairwise universality...")
    components = load_components_for_rank(
        rank=cfg.factorization.rank, models=loaded_models, dataset=cfg.dataset
    )
    loaded_models = [m for m in loaded_models if m in components]
    univ_mat = compute_universality_pairwise(components, loaded_models)

    logger.info("Phase 4: Saving outputs...")
    np.save(out / "cka_matrix.npy", cka_mat)

    with open(out / "model_names.json", "w") as f:
        json.dump(loaded_models, f, indent=2)

    summary_df = model_level_summary(cka_mat, univ_mat, loaded_models)
    summary_df.to_csv(out / "cka_vs_universality.csv", index=False)

    r_pearson, p_pearson = stats.pearsonr(summary_df["mean_cka"], summary_df["mean_universality"])
    r_spearman, p_spearman = stats.spearmanr(summary_df["mean_cka"], summary_df["mean_universality"])
    logger.info(f"Pearson r={r_pearson:.3f} (p={p_pearson:.2e})")
    logger.info(f"Spearman rho={r_spearman:.3f} (p={p_spearman:.2e})")
    logger.info(f"Outputs saved to {out}")


if __name__ == "__main__":
    main()
