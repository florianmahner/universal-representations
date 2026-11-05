#!/usr/bin/env python3
"""Provenance artifacts for the layer-wise fits: a metrics.json beside every
stored factorization, one JSON per representation (selected bandwidth, seed and
universality), and the model-level universality table."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def write_fit_metrics(cfg: DictConfig, out: Path, features: Path) -> int:
    """Per-fit metrics.json, same schema as the paper's, from stored artifacts."""
    rank = cfg.factorization.rank
    embeddings = out / "embeddings"
    units = {(j["model"], j["sigma_mult"]): j
             for j in (json.loads(p.read_text())
                       for p in (out / "selection_units").glob("*.json"))}
    n = 0
    for (name, sm), unit in units.items():
        feat = features / f"{name}.npy"
        shape = np.load(feat, mmap_mode="r").shape if feat.exists() else (None, None)
        for seed in cfg.factorization.seeds:
            d = embeddings / name / f"rank_{rank}" / f"sigma_{sm}" / f"seed_{seed}"
            if not (d / "w.npz").exists():
                continue
            (d / "metrics.json").write_text(json.dumps({
                "model": name, "rank": rank,
                "n_samples": int(shape[0]) if shape[0] else None,
                "feature_dim": int(shape[1]) if shape[1] else None,
                "feature_dir": str(features),
                "kernel": "rbf", "sigma_mult": float(sm),
                "init": "random_sqrt", "random_state": seed,
                "tol": cfg.factorization.tol,
                "max_outer": cfg.factorization.max_outer,
                "max_inner": cfg.factorization.max_inner,
                "mean_evar_over_seeds": unit["evar"],
                "mean_stability_over_seeds": unit["stability"],
                "harmonic": unit["harmonic"],
            }, indent=2))
            n += 1
    return n


def write_per_model(rank: int, out: Path, features: Path, uni_csv: Path,
                    sel: pd.DataFrame) -> int:
    """One JSON per representation: selection + universality + provenance."""
    per_model = out / "per_model"
    per_model.mkdir(parents=True, exist_ok=True)
    uni = pd.read_csv(uni_csv) if uni_csv.exists() else pd.DataFrame()
    untrained_map = Path("experiments/rebuttal/universality_across_layers/spec/untrained_map.csv")
    twin = {}
    if untrained_map.exists():
        m = pd.read_csv(untrained_map)
        twin = dict(zip(m.feature_uid, m.untrained_repr))
    units = [json.loads(p.read_text()) for p in (out / "selection_units").glob("*.json")]
    per_sigma = pd.DataFrame(units).groupby("model")

    n = 0
    for _, r in sel.iterrows():
        name = r.model
        untrained = name.startswith("untrained__")
        base, depth = ((name[len("untrained__"):-len("__pen")], "pen") if untrained
                       else name.rsplit("__", 1))
        feat = features / f"{name}.npy"
        w = (out / "embeddings" / name / f"rank_{rank}" / f"sigma_{r.sigma_mult}" /
             f"seed_{int(r.best_seed)}" / "w.npz")
        rec = {
            "repr": name,
            "role": "untrained" if untrained else "trained",
            "base_model": base,
            "depth": depth,
            "rank": rank,
            "feature_shape": list(np.load(feat, mmap_mode="r").shape) if feat.exists() else None,
            "selected": {"sigma_mult": r.sigma_mult, "seed": int(r.best_seed),
                         "stability": r.stability, "evar": r.evar, "harmonic": r.harmonic},
            # recorded relative to this module directory, i.e. "outputs/embeddings/..."
            "w_path": str(w.relative_to(out.parent)),
            "per_sigma": per_sigma.get_group(name)[
                ["sigma_mult", "stability", "evar", "harmonic"]].to_dict("records"),
        }
        if not untrained:
            rec["untrained_twin"] = twin.get(base)
        if len(uni):
            u = uni[uni["repr"] == name]
            rec["universality"] = {
                sc: {"mean": float(g.universality.mean()),
                     "median": float(g.universality.median()),
                     "std": float(g.universality.std()),
                     "n_dimensions": int(len(g))}
                for sc, g in u.groupby("scoring")}
        (per_model / f"{name}.json").write_text(json.dumps(rec, indent=2))
        n += 1
    return n


def write_model_level(rank: int, out: Path, uni_csv: Path, sel: pd.DataFrame) -> None:
    """Model-level universality table (one row per repr x scoring)."""
    if not uni_csv.exists():
        return
    uni = pd.read_csv(uni_csv)
    agg = (uni.groupby(["repr", "scoring", "role", "base", "depth"], dropna=False)
           .universality.agg(["mean", "median", "std", "min", "max", "count"])
           .reset_index().rename(columns={"count": "n_dimensions"}))
    agg = agg.merge(sel.rename(columns={"model": "repr"})[
        ["repr", "sigma_mult", "best_seed", "stability", "evar", "harmonic"]],
        on="repr", how="left")
    dest = out / f"universality_model_level_rank_{rank}.csv"
    agg.to_csv(dest, index=False)
    log.info(f"wrote {dest} ({len(agg)} rows)")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank = cfg.factorization.rank
    features = Path("experiments/rebuttal/universality_across_layers/feature_extraction/outputs/features")
    uni_csv = (Path("experiments/rebuttal/universality_across_layers/scoring/outputs")
               / f"universality_full_rank_{rank}.csv")
    sel = pd.read_csv(out / f"bandwidth_selection_full_rank_{rank}.csv")
    log.info(f"per-fit metrics.json written: {write_fit_metrics(cfg, out, features)}")
    log.info(f"per-model JSON written: {write_per_model(rank, out, features, uni_csv, sel)}")
    write_model_level(rank, out, uni_csv, sel)


if __name__ == "__main__":
    main()
