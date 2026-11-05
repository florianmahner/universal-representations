from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.universality.metric.compute.run import compute_universality_for_model
from experiments.universality.shared import load_components_for_rank, load_universality

N_NULL_PERMS = 1000

logger = logging.getLogger(__name__)


def load_w(name: str, sel: pd.DataFrame, embeddings: Path, rank: int) -> np.ndarray:
    row = sel[sel["model"] == name].iloc[0]
    p = (embeddings / name / f"rank_{rank}" /
         f"sigma_{row['sigma_mult']}" / f"seed_{int(row['best_seed'])}" / "w.npz")
    return np.load(p)["w"]


def score_one(name: str, W: np.ndarray, ref: dict[str, np.ndarray],
              n_null_perms: int, seed: int) -> list[dict]:
    rows = compute_universality_for_model(name, {**ref, name: W},
                                          n_null_perms=n_null_perms, null_seed=seed)
    return [r for r in rows if r["metric"] == "cos2_perm_adj"]


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank = cfg.factorization.rank
    universality_csv = out / f"universality_pseudo_rank_{rank}.csv"

    sel = pd.read_csv(out / f"bandwidth_selection_pseudo_rank_{rank}.csv")
    names = sorted(sel["model"])
    W = {n: load_w(n, sel, out / "embeddings", rank) for n in names}
    ref = load_components_for_rank(rank=rank, dataset=cfg.dataset)
    logger.info(f"{len(names)} pseudo-models vs {len(ref)} reference models")

    results = Parallel(n_jobs=min(cfg.n_jobs, len(names)), verbose=5)(
        delayed(score_one)(n, W[n], ref, N_NULL_PERMS, cfg.seed + i)
        for i, n in enumerate(names)
    )
    df = pd.DataFrame([r for batch in results for r in batch])
    df = df.merge(sel[["model", "stability", "evar", "harmonic"]], on="model", how="left")
    df.to_csv(universality_csv, index=False)

    trained = load_universality(rank=rank, dataset=cfg.dataset).groupby("model")["universality"].mean()
    summary = df.groupby("model")["universality"].mean().rename("mean_universality").to_frame()
    summary["trained_percentile"] = [
        float((trained < u).mean() * 100) for u in summary["mean_universality"]
    ]
    summary.to_csv(out / "summary_pseudo.csv")
    logger.info(
        f"wrote {universality_csv}\ntrained models: mean={trained.mean():.3f} "
        f"min={trained.min():.3f} max={trained.max():.3f}\n{summary.to_string()}"
    )


if __name__ == "__main__":
    main()
