"""Random-feature control: factorize Gaussian noise matrices with the standard
pipeline and score them against the reference model set.

These "models" never saw an image, so their universality is the floor that any
image-driven representation has to beat.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from experiments.rebuttal.universality_across_layers.factorization.factorize import rbf_frugal
from experiments.universality.metric.compute.run import compute_universality_for_model
from experiments.universality.shared import load_components_for_rank
from src.factorization import get_snmf_class
from src.metrics import explained_variance, median_sigma_heuristic
from src.stability import factorization_stability, reference_seed_index

SIGMAS = ["0.3", "0.4", "0.5", "0.6"]  # the grid these controls were run on

log = logging.getLogger(__name__)

N_PERMS = 1000  # permutation null for the universality metric
SIGMA_SUBSAMPLE = 4000  # rows used by the median-distance bandwidth heuristic


def make_features(path: Path, seed: int, shape: tuple[int, int]) -> Path:
    """Write (once) a Gaussian random feature matrix for one baseline model."""
    if not path.exists():
        rng = np.random.default_rng(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rng.standard_normal(shape).astype(np.float32))
    return path


def fit_unit(
    cfg: DictConfig, name: str, feat_path: Path, sigma_mult: str, emb_dir: Path
) -> dict:
    """One (model, sigma) unit: one kernel, all seed fits, one selection row."""
    SNMF = get_snmf_class()
    rank = cfg.factorization.rank

    feats = np.load(feat_path).astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(feats), SIGMA_SUBSAMPLE, replace=False)
    sigma = float(median_sigma_heuristic(feats[idx])) * float(sigma_mult)
    similarity = rbf_frugal(feats, sigma)

    ws, evars = [], []
    for seed in cfg.factorization.seeds:
        w_path = (
            emb_dir / name / f"rank_{rank}" / f"sigma_{sigma_mult}" / f"seed_{seed}" / "w.npz"
        )
        if w_path.exists():
            w = np.load(w_path)["w"]
        else:
            est = SNMF(
                rank=rank,
                max_outer=cfg.factorization.max_outer,
                max_inner=cfg.factorization.max_inner,
                tol=cfg.factorization.tol,
                init="random_sqrt",
                random_state=seed,
            )
            est.fit(similarity)
            w_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(w_path, w=est.w_)
            w = est.w_
        ws.append(w)
        evars.append(explained_variance(similarity, w))

    stability = float(np.nanmean(factorization_stability(ws)))
    evar = float(np.nanmean(evars))
    harmonic = (2 * stability * evar / (stability + evar)) if (stability + evar) > 0 else 0.0
    best_seed = int(cfg.factorization.seeds[reference_seed_index(ws)])
    log.info(
        f"{name} sigma={sigma_mult}: stab={stability:.3f} evar={evar:.3f} harm={harmonic:.3f}"
    )
    return {
        "model": name,
        "sigma_mult": sigma_mult,
        "best_seed": best_seed,
        "stability": stability,
        "evar": evar,
        "harmonic": harmonic,
    }


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    # Five noise "models", each the size of a THINGS penultimate feature matrix.
    names = [f"random__seed{k}__pen" for k in range(5)]
    feats = {
        name: make_features(out / "features" / f"{name}.npy", 1000 + k, (22248, 512))
        for k, name in enumerate(names)
    }

    sigmas = [str(s) for s in SIGMAS]
    units = [(n, feats[n], sm) for n in names for sm in sigmas]
    rows = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
        delayed(fit_unit)(cfg, n, p, sm, out / "embeddings") for n, p, sm in units
    )
    sel = pd.DataFrame(rows)
    best = sel.loc[sel.groupby("model").harmonic.idxmax()].reset_index(drop=True)
    best.to_csv(out / f"bandwidth_selection_random_rank_{rank}.csv", index=False)
    log.info(f"selection:\n{best.to_string(index=False)}")

    ref = load_components_for_rank(rank=rank, dataset=cfg.dataset)
    all_rows = []
    for i, (_, r) in enumerate(best.iterrows()):
        w = np.load(
            out / "embeddings" / r.model / f"rank_{rank}" / f"sigma_{r.sigma_mult}" /
            f"seed_{int(r.best_seed)}" / "w.npz"
        )["w"]
        comp = {**ref, r.model: w}
        rows = compute_universality_for_model(
            r.model,
            comp,
            n_null_perms=N_PERMS,
            # offset keeps the null seeds clear of the real models' seed range
            null_seed=cfg.seed + 1000 + i,
        )
        all_rows += [x for x in rows if x["metric"] == "cos2_perm_adj"]
    uni = pd.DataFrame(all_rows)
    uni.to_csv(out / f"universality_random_rank_{rank}.csv", index=False)
    model_level = uni.groupby("model").universality.agg(["mean", "std", "min", "max"])
    model_level.to_csv(out / "universality_random_model_level.csv")
    log.info(f"\nRANDOM BASELINE universality (cos2_perm_adj):\n{model_level.round(4).to_string()}")
    log.info(f"grand mean: {uni.universality.mean():.4f}")


if __name__ == "__main__":
    main()
