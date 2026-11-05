from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.rebuttal.universality_across_layers.factorization.factorize import rbf_frugal
from src.metrics import explained_variance, median_sigma_heuristic
from src.stability import factorization_stability, reference_seed_index

SIGMAS = ["0.3", "0.4", "0.5", "0.6"]  # the grid these controls were run on

logger = logging.getLogger(__name__)


def sigma0_for(path: Path, subsample: int = 4000) -> float:
    """Median-distance bandwidth from a fixed random subsample of the rows.

    The subsample rule is identical to the one factorize.py used when the fits
    were produced, so the kernel rebuilt here matches the one that was factorized.
    """
    feats = np.load(path)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(feats), min(subsample, len(feats)), replace=False)
    return float(median_sigma_heuristic(feats[idx].astype(np.float64)))


def select_unit(
    name: str,
    sigma_mult: str,
    features: Path,
    embeddings: Path,
    units: Path,
    rank: int,
    seeds: list[int],
) -> dict:
    """Recompute stability and explained variance for one (repr, sigma) from saved fits."""
    unit_json = units / f"{name}__{sigma_mult}.json"
    if unit_json.exists():
        return json.loads(unit_json.read_text())

    ws = [
        np.load(embeddings / name / f"rank_{rank}" / f"sigma_{sigma_mult}" / f"seed_{s}" / "w.npz")["w"]
        for s in seeds
    ]
    stability = float(np.nanmean(factorization_stability(ws)))

    feat_path = features / f"{name}.npy"
    sigma = sigma0_for(feat_path) * float(sigma_mult)
    feats = np.load(feat_path).astype(np.float64)
    S = rbf_frugal(feats, sigma)
    # explained_variance normalizes by the CENTERED sum of squares of the kernel
    # entries, which is the quantity the factorizer reports as final_evar.
    evar = float(np.nanmean([explained_variance(S, w) for w in ws]))
    del S, feats

    harmonic = (2 * stability * evar / (stability + evar)) if (stability + evar) > 0 else 0.0
    row = dict(model=name, sigma_mult=sigma_mult, stability=stability,
               evar=evar, harmonic=harmonic)
    units.mkdir(parents=True, exist_ok=True)
    unit_json.write_text(json.dumps(row))
    logger.info(f"{name} sigma={sigma_mult}: stab={stability:.3f} evar={evar:.3f} "
                f"harm={harmonic:.3f}")
    return row


def finalize(
    embeddings: Path,
    units: Path,
    sel_csv: Path,
    sigmas: list[str],
    rank: int,
    seeds: list[int],
) -> None:
    """Aggregate all unit stats, pick best sigma per repr and its best seed."""
    rows = [json.loads(p.read_text()) for p in units.glob("*.json")]
    df = pd.DataFrame(rows)
    reprs = sorted(d.name for d in embeddings.iterdir() if d.is_dir())
    missing = [(r, sm) for r in reprs for sm in sigmas
               if not (units / f"{r}__{sm}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} units not yet computed, e.g. {missing[:5]}")

    best = df.loc[df.groupby("model").harmonic.idxmax()].reset_index(drop=True)
    best_seeds = []
    for _, r in best.iterrows():
        ws = [
            np.load(embeddings / r.model / f"rank_{rank}" / f"sigma_{r.sigma_mult}" / f"seed_{s}" / "w.npz")["w"]
            for s in seeds
        ]
        best_seeds.append(int(reference_seed_index(ws)))
    best["best_seed"] = best_seeds
    best = best[["model", "sigma_mult", "best_seed", "stability", "evar", "harmonic"]]
    best.to_csv(sel_csv, index=False)
    logger.info(f"wrote {sel_csv} ({len(best)} reprs)\n"
                f"sigma distribution:\n{best.sigma_mult.value_counts().to_string()}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    features = Path("experiments/rebuttal/universality_across_layers/feature_extraction/outputs/features")
    embeddings = out / "embeddings"
    units = out / "selection_units"

    rank = cfg.factorization.rank
    sigmas = [str(s) for s in SIGMAS]
    seeds = list(cfg.factorization.seeds)

    reprs = list(cfg.models) or sorted(d.name for d in embeddings.iterdir() if d.is_dir())
    for name in reprs:
        for sigma_mult in sigmas:
            select_unit(name, sigma_mult, features, embeddings, units, rank, seeds)
    finalize(embeddings, units, out / f"bandwidth_selection_full_rank_{rank}.csv",
             sigmas, rank, seeds)


if __name__ == "__main__":
    main()
