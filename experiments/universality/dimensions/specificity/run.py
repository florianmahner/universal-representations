#!/usr/bin/env python3
"""Group specificity of NMF dimensions: within-group vs between-group matching."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.data import ThingsDataset
from src.plotting import setup_plotting
from experiments.universality.shared import load_components_for_rank
from experiments.universality.dimensions.specificity.analysis import (
    build_grouping,
    compute_all_specificity,
    summarize_by_group,
    run_permutation_test,
    run_bootstrap_ci,
    compute_effect_sizes,
)
from experiments.universality.dimensions.specificity.plot import (
    plot_specificity_distribution,
    plot_within_vs_between_scatter,
    plot_dimension_profiles,
    plot_permutation_results,
)

MIN_GROUP_SIZE = 3
N_PERMUTATIONS = 1000

log = logging.getLogger(__name__)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    setup_plotting()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    grouping_name = "model_class"
    meta = ThingsDataset().model_info.copy()
    grouping, description = build_grouping(
        meta, pd.DataFrame(), grouping_name, "metadata",
    )
    log.info(description)
    log.info(f"Models with group labels: {len(grouping)}")
    log.info(f"Rank: {rank}")

    components = load_components_for_rank(
        rank=rank, models=list(cfg.models) or None, dataset=cfg.dataset,
    )
    results = compute_all_specificity(
        components, grouping, MIN_GROUP_SIZE, "group",
    )
    results["rank"] = rank
    results.to_csv(out / "dimension_specificity.csv", index=False)

    summary = summarize_by_group(results)
    summary.to_csv(out / "group_summary.csv", index=False)
    log.info(f"\nGroup summary (rank={rank}):\n{summary.to_string(index=False)}")

    log.info(f"\nRunning permutation test ({N_PERMUTATIONS} permutations)...")
    perm_df = run_permutation_test(
        components, grouping, MIN_GROUP_SIZE, N_PERMUTATIONS,
        seed=cfg.seed, between_pool="group",
    )
    perm_df.to_csv(out / "permutation_test.csv", index=False)
    log.info(f"Permutation results:\n{perm_df.to_string(index=False)}")

    log.info("Computing bootstrap CIs...")
    boot_df = run_bootstrap_ci(results, seed=cfg.seed)
    boot_df.to_csv(out / "bootstrap_ci.csv", index=False)
    log.info(f"Bootstrap CIs:\n{boot_df.to_string(index=False)}")

    effects = compute_effect_sizes(results)
    log.info(f"Effect size: {effects}")

    stats = {
        "grouping_name": grouping_name,
        "grouping_type": "metadata",
        "description": description,
        "primary_rank": rank,
        "ranks": [rank],
        "n_models": int(results["model"].nunique()),
        "n_groups": int(results["group"].nunique()),
        "n_dimensions": int(results["dimension"].nunique()),
        "min_group_size": MIN_GROUP_SIZE,
        "between_pool": "group",
        "n_permutations": N_PERMUTATIONS,
        **{f"effect_{k}": v for k, v in effects.items()},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))

    log.info("Generating figures...")
    plot_specificity_distribution(results, figures, grouping_name)
    plot_within_vs_between_scatter(results, figures, grouping_name)
    plot_dimension_profiles(results, figures)
    if not perm_df.empty:
        plot_permutation_results(perm_df, figures, grouping_name)

    log.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
