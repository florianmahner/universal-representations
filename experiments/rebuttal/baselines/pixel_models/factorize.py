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

logger = logging.getLogger(__name__)

# Rows subsampled to estimate the median bandwidth; the pixel-scale pseudo-model
# features are too large for a full pairwise distance matrix.
SIGMA_SUBSAMPLE = 4000


def fit_one_sigma(name: str, feat_path: Path, embeddings_dir: Path, sigma_mult: str,
                  sigma0: float, seeds: list[int], rank: int, max_outer: int,
                  max_inner: int, tol: float) -> list[dict]:
    """Build the kernel once for this sigma, fit all seeds, return selection metrics."""
    from src.factorization import get_srf_class

    SRF = get_srf_class()

    sigma = sigma0 * float(sigma_mult)
    feats = np.load(feat_path).astype(np.float64)
    ws, evars = {}, []
    S = None
    for seed in seeds:
        out_dir = embeddings_dir / name / f"rank_{rank}" / f"sigma_{sigma_mult}" / f"seed_{seed}"
        wpath = out_dir / "w.npz"
        if S is None:
            S = rbf_frugal(feats, sigma)
        if wpath.exists():
            # Resume: skip the fit but recompute the real explained variance,
            # otherwise the harmonic selection below degenerates to the first
            # sigma in the grid.
            ws[seed] = np.load(wpath)["w"]
            evars.append(explained_variance(S, ws[seed]))
            continue
        est = SRF(rank=rank, max_outer=max_outer, max_inner=max_inner, tol=tol,
                   init="random_sqrt", random_state=seed)
        est.fit(S)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(wpath, w=est.w_)
        ws[seed] = est.w_
        evars.append(explained_variance(S, est.w_))
    w_list = [ws[s] for s in seeds]
    stab = float(np.nanmean(factorization_stability(w_list)))
    evar_mean = float(np.nanmean(evars))
    best_seed = int(seeds[reference_seed_index(w_list)])
    harm = (2 * stab * evar_mean / (stab + evar_mean)) if (stab + evar_mean) > 0 else 0.0
    logger.info(f"{name} sigma={sigma_mult}: stab={stab:.3f} evar={evar_mean:.3f} harm={harm:.3f}")
    return [dict(model=name, sigma_mult=sigma_mult, best_seed=best_seed,
                 stability=stab, evar=evar_mean, harmonic=harm)]


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    reprs = sorted((p.stem, p) for p in (out / "features").glob("*.npy"))
    if cfg.models:
        reprs = [(n, p) for n, p in reprs if n in cfg.models]
    if not reprs:
        raise SystemExit(f"no features in {out / 'features'} — run build_features first")

    rank = cfg.factorization.rank
    sigmas = [str(s) for s in SIGMAS]
    seeds = list(cfg.factorization.seeds)
    logger.info(f"{len(reprs)} pseudo-models x {len(sigmas)} sigmas x {len(seeds)} seeds")

    rng = np.random.default_rng(0)
    sigma0 = {}
    for name, path in reprs:
        feats = np.load(path)
        idx = rng.choice(len(feats), min(SIGMA_SUBSAMPLE, len(feats)), replace=False)
        sigma0[name] = float(median_sigma_heuristic(feats[idx].astype(np.float64)))

    tasks = [(n, p, sm) for n, p in reprs for sm in sigmas]
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(fit_one_sigma)(n, p, out / "embeddings", sm, sigma0[n], seeds, rank,
                               cfg.factorization.max_outer, cfg.factorization.max_inner,
                               cfg.factorization.tol)
        for n, p, sm in tasks
    )
    df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)
    best = df.loc[df.groupby("model")["harmonic"].idxmax()].reset_index(drop=True)
    sel_csv = out / f"bandwidth_selection_pseudo_rank_{rank}.csv"
    best.to_csv(sel_csv, index=False)
    logger.info(f"wrote {sel_csv}\n{best.to_string(index=False)}")


if __name__ == "__main__":
    main()
