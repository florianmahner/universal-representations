from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from experiments.universality.downstream_predictions.things_behavior.triplets.run import (
    get_exemplary_indices,
    logit_3afc,
    triplet_accuracy_from_sim,
)

from . import data as D

UNTRAINED = False  # set True to score the untrained networks instead of the depth sweep

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_triplets(data_root: Path = Path("data")) -> np.ndarray:
    """Train + validation triplets, identical to the published analysis."""
    triplet_dir = data_root / "behavior" / "things_triplets"
    parts = [np.loadtxt(triplet_dir / n, dtype=int) for n in ("trainset.txt", "validationset.txt")]
    triplets = np.concatenate(parts, axis=0)
    assert triplets.shape[1] == 3 and triplets.min() >= 0 and triplets.max() < 1854
    log.info(f"Loaded {len(triplets)} triplets (train + validation)")
    return triplets


def _one(base: str, depth: str, exemplary_idx: np.ndarray, triplets: np.ndarray) -> dict:
    w = D.load_w(base, depth)[exemplary_idx]
    acc = triplet_accuracy_from_sim(cosine_similarity(w), triplets)
    return {"base": base, "depth": depth, "triplet_accuracy": acc}


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    triplets = load_triplets(Path(cfg.data_root))
    exemplary_idx = get_exemplary_indices()

    if UNTRAINED:
        jobs = [(b, "untrained") for b in D.untrained_models()]
    else:
        models = list(cfg.models) or D.base_models()
        jobs = [(b, d) for d in D.DEPTHS for b in models]
    log.info(f"Triplet accuracy for {len(jobs)} representations")

    rows = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(_one)(b, d, exemplary_idx, triplets) for b, d in jobs
    )
    df = pd.DataFrame(rows)
    df["logit_accuracy"] = logit_3afc(df["triplet_accuracy"].to_numpy())
    df = D.merge_all(df)

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    out_csv = out / "scores" / "triplet" / ("untrained.csv" if UNTRAINED else "across_depth.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv} ({len(df)} rows)")

    for depth, g in df.groupby("depth"):
        log.info(f"  {D.DEPTH_NAME[depth]:12s} acc {g.triplet_accuracy.mean():.4f} "
                 f"[{g.triplet_accuracy.min():.3f}, {g.triplet_accuracy.max():.3f}]")


if __name__ == "__main__":
    main()
