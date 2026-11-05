from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import pairwise_distances

from src.data import ThingsDataset

log = logging.getLogger(__name__)


def model_sigma(
    data: ThingsDataset,
    model: str,
    rank: int,
    embedding_dir: Path,
) -> float:
    """Read the Gaussian bandwidth used when fitting NMF for this model."""
    sel = data.bandwidth_selection(rank)
    row = sel[sel["model"] == model].iloc[0]
    sigma_mult = row["sigma_mult"]
    best_seed = int(row["best_seed"])
    mpath = (
        embedding_dir / model / f"rank_{rank}" / f"sigma_{sigma_mult}"
        / f"seed_{best_seed}" / "metrics.json"
    )
    meta = json.loads(mpath.read_text())
    return float(meta["sigma"])


def accumulate_block_stats(
    features: np.ndarray,
    W: np.ndarray,
    sigma: float,
    block_size: int = 512,
) -> tuple[np.ndarray, float, float]:
    """Stream S @ W and scalar sums over row blocks without materializing S.

    Returns
    -------
    SW : (N, r) float64
        S @ W, where S is the RBF kernel with bandwidth sigma.
    sum_S : float
        Sum over all (i, j) of S_ij (diagonal included).
    sum_S_sq : float
        Sum over all (i, j) of S_ij^2 (diagonal included).
    """
    n = features.shape[0]
    r = W.shape[1]
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    SW = np.zeros((n, r), dtype=np.float64)
    sum_S = 0.0
    sum_S_sq = 0.0

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        d_block = pairwise_distances(features[start:stop], features)
        np.square(d_block, out=d_block)
        d_block *= -inv_2s2
        np.exp(d_block, out=d_block)
        S_block = d_block

        SW[start:stop] = S_block @ W
        sum_S += float(S_block.sum())
        sum_S_sq += float((S_block ** 2).sum())

    return SW, sum_S, sum_S_sq


def loo_importance_for_model(
    model: str,
    features: np.ndarray,
    W: np.ndarray,
    sigma: float,
) -> dict:
    """Compute R^2_full and per-dimension delta_r2 for one model."""
    n, r = W.shape

    SW, sum_S, sum_S_sq = accumulate_block_stats(features, W, sigma)
    WtSW = W.T @ SW
    WtW = W.T @ W
    WtW_sq = WtW @ WtW

    row_norm_sq = (W * W).sum(axis=1)
    W_sq = W * W
    W_fourth = W_sq * W_sq

    sum_S_offdiag = sum_S - n
    sum_S_sq_offdiag = sum_S_sq - n
    sum_S_upper = 0.5 * sum_S_offdiag
    sum_S_sq_upper = 0.5 * sum_S_sq_offdiag
    n_pairs = 0.5 * n * (n - 1)
    mean_upper = sum_S_upper / n_pairs

    ss_total_upper = sum_S_sq_upper - n_pairs * mean_upper * mean_upper

    S_minus_WWt_F2 = (
        sum_S_sq - 2.0 * float(np.trace(WtSW)) + float(np.trace(WtW_sq))
    )
    diag_R = 1.0 - row_norm_sq
    diag_R_sq_sum = float(np.sum(diag_R * diag_R))
    ss_resid_full_upper = 0.5 * (S_minus_WWt_F2 - diag_R_sq_sum)

    r2_full = 1.0 - ss_resid_full_upper / ss_total_upper

    wtSw = np.diag(WtSW)
    w_WWT_w = np.diag(WtW_sq)
    w_norm2 = np.diag(WtW)

    wR_w = wtSw - w_WWT_w
    sum_Rii_w2 = (diag_R[:, None] * W_sq).sum(axis=0)
    sum_w4 = W_fourth.sum(axis=0)

    delta_ssr_upper = (
        wR_w + 0.5 * w_norm2 * w_norm2 - sum_Rii_w2 - 0.5 * sum_w4
    )
    delta_r2 = delta_ssr_upper / ss_total_upper

    return {
        "model": model,
        "r2_full": r2_full,
        "delta_r2": delta_r2,
        "n_pairs_upper": int(n_pairs),
        "sigma": sigma,
    }


def _process_one(
    model: str,
    rank: int,
    embedding_dir: Path,
) -> list[dict] | None:
    """Worker function. Re-instantiates ThingsDataset to stay pickleable."""
    data = ThingsDataset()
    try:
        features = data.load_embedding(model).astype(np.float64)
        W = data.load_components(model, source="full_fit", rank=rank).astype(np.float64)
        sigma = model_sigma(data, model, rank, embedding_dir)
    except Exception as e:
        log.warning("skipping %s: %s", model, e)
        return None

    t0 = perf_counter()
    res = loo_importance_for_model(model, features, W, sigma)
    dt = perf_counter() - t0

    records = []
    for k, dr2 in enumerate(res["delta_r2"]):
        records.append({
            "model": model,
            "dimension": k,
            "rank": rank,
            "sigma": res["sigma"],
            "r2_full": res["r2_full"],
            "delta_r2": float(dr2),
            "elapsed_s": dt,
        })
    return records


def run_all(
    data: ThingsDataset,
    rank: int,
    n_jobs: int,
    embedding_dir: Path,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    component_models = data.list_component_models()
    embedding_models = set(data.list_embedding_models())
    models = [
        m for m in component_models
        if m in embedding_models and rank in data.list_available_ranks(m)
    ]
    if model_names:
        models = [m for m in models if m in set(model_names)]

    log.info("running true LOO for %d models at rank %d (n_jobs=%d)", len(models), rank, n_jobs)

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one)(m, rank, embedding_dir) for m in models
    )

    rows = [rec for bundle in results if bundle for rec in bundle]
    return pd.DataFrame(rows)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    data = ThingsDataset()
    df = run_all(
        data,
        rank=rank,
        n_jobs=cfg.n_jobs,
        embedding_dir=Path(cfg.data_root) / "embeddings" / cfg.dataset,
        model_names=list(cfg.models),
    )

    out_path = out / f"loo_importance_full_rank_{rank}.csv"
    df.to_csv(out_path, index=False)
    log.info("saved %d rows → %s", len(df), out_path)

    mean_r2 = df.groupby("model")["r2_full"].first().mean()
    log.info("mean R^2_full across models = %.4f", mean_r2)


if __name__ == "__main__":
    main()
