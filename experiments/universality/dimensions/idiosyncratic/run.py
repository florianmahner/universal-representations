#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .compute import (
    classify_dimensions,
    compute_correlation_profiles,
    compute_profile_stats,
    compute_thresholds,
    compute_within_model_structure,
    identify_sharing_taxonomy,
    load_and_merge,
)
from .plots import (
    plot_correlation_profiles,
    plot_idiosyncratic_by_group,
    plot_idiosyncratic_fraction,
    plot_noise_vs_signal,
    plot_noise_vs_signal_by_task,
    plot_noise_vs_signal_log,
    plot_profile_classification,
    plot_quadrant_composition,
    plot_subgroup_sharing,
    plot_top_idiosyncratic_images,
    plot_within_model_reconstruction,
)

log = logging.getLogger(__name__)


def run_analysis(rank: int, out: Path) -> dict:
    """Split every model's dimensions into universal and idiosyncratic parts."""

    # Phase 1: Load and merge
    log.info("Phase 1: Loading and merging data...")
    df = load_and_merge(rank=rank)
    thresholds = compute_thresholds(df)
    df = classify_dimensions(df, thresholds)
    df.to_csv(out / "merged_dims.csv", index=False)
    log.info(f"  Thresholds: universality={thresholds['universality']:.4f}, "
             f"importance={thresholds['importance']:.4f}")

    # Quadrant summary
    quadrant_counts = df["quadrant"].value_counts()
    for q, n in quadrant_counts.items():
        log.info(f"  {q}: {n} dimensions ({n/len(df)*100:.1f}%)")

    # Phase 2: Correlation profiles
    log.info("Phase 2: Computing correlation profiles...")
    profiles, models_order = compute_correlation_profiles(rank=rank)
    np.save(out / "correlation_profiles.npy", profiles)
    (out / "model_order.json").write_text(json.dumps(models_order, indent=2))

    log.info("  Computing profile statistics...")
    profile_stats = compute_profile_stats(profiles, models_order, df)
    profile_stats.to_csv(out / "profile_stats.csv", index=False)

    log.info("  Identifying subgroup sharing taxonomy...")
    sharing_df = identify_sharing_taxonomy(profiles, models_order, df)
    sharing_df.to_csv(out / "sharing_taxonomy.csv", index=False)

    # Phase 3: Within-model structure
    log.info("Phase 3: Computing within-model structure...")
    within_df = compute_within_model_structure(df, rank=rank)
    within_df.to_csv(out / "within_model_stats.csv", index=False)

    params = {
        "rank": rank,
        "n_models": int(df["model"].nunique()),
        "n_dimensions": len(df),
        "thresholds": thresholds,
        "quadrant_counts": {k: int(v) for k, v in quadrant_counts.items()},
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))

    log.info("Analysis complete.")
    return {
        "df": df,
        "thresholds": thresholds,
        "profiles": profiles,
        "models_order": models_order,
        "profile_stats": profile_stats,
        "sharing_df": sharing_df,
        "within_df": within_df,
    }


def run_plots(results: dict, out: Path) -> None:
    """Generate all figures from the analysis results."""
    log.info("Generating figures...")

    df = results["df"]
    thresholds = results["thresholds"]

    # Analysis 1: Noise vs signal
    log.info("  Analysis 1: Noise vs signal...")
    plot_noise_vs_signal(df, thresholds, out)
    plot_noise_vs_signal_log(df, thresholds, out)
    plot_noise_vs_signal_by_task(df, thresholds, out)
    plot_quadrant_composition(df, out)

    # Analysis 2: Task-specificity
    log.info("  Analysis 2: Task-specificity...")
    plot_idiosyncratic_by_group(df, "train_task", out)
    plot_idiosyncratic_by_group(df, "model_class", out)
    plot_top_idiosyncratic_images(df, out)

    # Analysis 3: Correlation profiles
    log.info("  Analysis 3: Correlation profiles...")
    plot_correlation_profiles(
        results["profiles"], results["models_order"],
        df, results["profile_stats"], out,
    )
    plot_profile_classification(results["profile_stats"], df, out)
    plot_subgroup_sharing(results["sharing_df"], df, out)

    # Analysis 4: Within-model structure
    log.info("  Analysis 4: Within-model structure...")
    plot_within_model_reconstruction(results["within_df"], out)
    plot_idiosyncratic_fraction(results["within_df"], out)

    log.info(f"All figures saved to {out}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    results = run_analysis(cfg.factorization.rank, out)
    run_plots(results, out)


if __name__ == "__main__":
    main()
