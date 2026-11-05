from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from experiments.rebuttal.universality_across_layers.factorization.factorize import rbf_frugal
from src.metrics import explained_variance, median_sigma_heuristic
from src.stability import factorization_stability, reference_seed_index

SIGMAS = ["0.3", "0.4", "0.5", "0.6"]  # the grid these controls were run on

log = logging.getLogger(__name__)


def sigma0_for(path: Path, subsample: int = 4000) -> float:
    """Median-distance bandwidth from a fixed random subsample of the rows."""
    feats = np.load(path)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(feats), min(subsample, len(feats)), replace=False)
    return float(median_sigma_heuristic(feats[idx].astype(np.float64)))


def one(
    name: str,
    path: Path,
    sigma_mult: str,
    sigma0: float,
    seeds: list[int],
    rank: int,
    embeddings: Path,
) -> list[dict]:
    """Recompute stability and explained variance for one (model, sigma) from saved fits."""
    ws = []
    for seed in seeds:
        p = embeddings / name / f"rank_{rank}" / f"sigma_{sigma_mult}" / f"seed_{seed}" / "w.npz"
        if not p.exists():
            log.warning(f"missing {p}")
            return []
        ws.append(np.load(p)["w"])
    S = rbf_frugal(np.load(path).astype(np.float64), float(sigma_mult) * sigma0)
    evars = [explained_variance(S, w) for w in ws]
    stab = float(np.nanmean(factorization_stability(ws)))
    evar = float(np.nanmean(evars))
    harm = (2 * stab * evar / (stab + evar)) if (stab + evar) > 0 else 0.0
    log.info(f"{name} sigma={sigma_mult}: stab={stab:.3f} evar={evar:.3f} harm={harm:.3f}")
    return [dict(model=name, sigma_mult=sigma_mult, best_seed=int(seeds[reference_seed_index(ws)]),
                 stability=stab, evar=evar, harmonic=harm)]


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s')
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank = cfg.factorization.rank
    seeds = list(cfg.factorization.seeds)

    reprs = sorted((p.stem, p) for p in (out / "features").glob("*.npy"))
    if cfg.models:
        reprs = [(n, p) for n, p in reprs if n in cfg.models]
    if not reprs:
        raise SystemExit(f"no features in {out / 'features'} — run build_features first")

    sigma0 = {n: sigma0_for(p) for n, p in reprs}
    tasks = [(n, p, sm) for n, p in reprs for sm in map(str, SIGMAS)]
    # Each task holds one n x n kernel, so n_jobs trades speed against memory.
    res = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(one)(n, p, sm, sigma0[n], seeds, rank, out / "embeddings")
        for n, p, sm in tasks
    )
    df = pd.concat([pd.DataFrame(r) for r in res if r], ignore_index=True)
    df.to_csv(out / f"sigma_grid_pseudo_rank_{rank}.csv", index=False)
    best = df.loc[df.groupby('model')['harmonic'].idxmax()].reset_index(drop=True)
    best.to_csv(out / f"bandwidth_selection_pseudo_rank_{rank}.csv", index=False)
    log.info(f"wrote {out / f'bandwidth_selection_pseudo_rank_{rank}.csv'}\n{best.to_string(index=False)}")


if __name__ == '__main__':
    main()
