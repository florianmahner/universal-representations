#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .analysis import build_cluster_profiles, compute_provenance_by_tier, load_data
from .plot import make_plots
from .stats import run_all_stats

N_BOOT = 5000
N_PERM = 1000

log = logging.getLogger(__name__)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    min_size = 3

    log.info("Loading data (rank=%d)", cfg.factorization.rank)
    W_all, dim_df, ds = load_data(rank=cfg.factorization.rank)
    categories = np.array(ds.all_categories)

    log.info("Building cluster profiles (min_size=%d)", min_size)
    profiles, prototypes = build_cluster_profiles(
        W_all, dim_df, categories, min_size=min_size,
    )

    log.info("Computing provenance by tier")
    provenance = compute_provenance_by_tier(dim_df, profiles)

    profiles.to_csv(out / "cluster_characterization.csv", index=False)
    provenance.to_csv(out / "provenance_by_tier.csv", index=False)
    np.savez(out / "prototypes.npz", **prototypes)

    params = {
        "rank": cfg.factorization.rank,
        "min_size": min_size,
        "n_clusters_profiled": len(profiles),
        "n_models": int(dim_df["model"].nunique()),
        "n_dimensions": len(dim_df),
    }
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    log.info("Saved outputs to %s", out)

    log.info("Running statistical tests")
    all_stats = run_all_stats(
        profiles, provenance=provenance, n_boot=N_BOOT, n_perm=N_PERM,
    )
    with open(out / "statistical_tests.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    log.info("Saved statistical tests to %s", out / "statistical_tests.json")

    log.info("Generating figures")
    make_plots(profiles, provenance, prototypes, ds, out, stats=all_stats)

    log.info("Done. %d clusters profiled.", len(profiles))


if __name__ == "__main__":
    main()
