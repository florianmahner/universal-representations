#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from .compute import (
    load_base_data,
    compute_category_selectivity,
    compute_within_model_correlations,
    compute_category_bias,
    load_importance_data,
    compute_importance_universality,
)
from .plot import make_all_plots

log = logging.getLogger(__name__)


def run_analysis(rank: int, out: Path) -> dict:
    """Run all analyses and save outputs."""
    log.info("Loading base data (rank=%d)...", rank)
    components, universality, categories, model_info = load_base_data(rank=rank)

    # Category selectivity (eta-squared)
    log.info("Computing category selectivity (eta^2)...")
    selectivity = compute_category_selectivity(components, universality, categories)

    # Reconstruction importance must be the manuscript's true leave-one-out
    # delta R².  Merge it before writing the shared Figure 4 input so a
    # loading-norm proxy cannot be mislabeled as reconstruction importance.
    log.info("Computing importance-universality orthogonality...")
    importance_df = load_importance_data(rank)
    imp_df, idio_frac, imp_stats = compute_importance_universality(
        importance_df, selectivity, model_info,
    )
    selectivity = selectivity.merge(
        imp_df[["model", "dimension", "importance"]],
        on=["model", "dimension"],
        how="left",
        validate="one_to_one",
    )
    if selectivity["importance"].isna().any():
        raise ValueError("Missing delta R² after merging reconstruction importance")
    selectivity.to_csv(out / "category_selectivity.csv", index=False)
    imp_df.to_csv(out / "importance_universality.csv", index=False)
    idio_frac.to_csv(out / "idiosyncratic_fraction.csv", index=False)

    # Within-model correlations
    within_stats = compute_within_model_correlations(
        selectivity, "universality", "eta_squared",
    )
    log.info(
        "  Within-model eta^2~universality: median rho=%.3f, %d/%d positive",
        within_stats["median_rho"], int(within_stats["frac_positive"] * within_stats["n_models"]),
        within_stats["n_models"],
    )
    c_eff_within = compute_within_model_correlations(
        selectivity, "universality", "c_eff",
    )
    eta_c_eff_within = compute_within_model_correlations(
        selectivity, "eta_squared", "c_eff",
    )
    c_eff_excess_within = compute_within_model_correlations(
        selectivity, "universality", "c_eff_excess",
    )
    eta_c_eff_excess_within = compute_within_model_correlations(
        selectivity, "eta_squared", "c_eff_excess",
    )
    log.info(
        "  Within-model c_eff~universality: median rho=%.3f, %d/%d positive",
        c_eff_within["median_rho"],
        int(c_eff_within["frac_positive"] * c_eff_within["n_models"]),
        c_eff_within["n_models"],
    )
    log.info(
        "  Within-model c_eff_excess~universality: median rho=%.3f, %d/%d positive",
        c_eff_excess_within["median_rho"],
        int(c_eff_excess_within["frac_positive"] * c_eff_excess_within["n_models"]),
        c_eff_excess_within["n_models"],
    )

    # Category bias
    log.info("Computing per-category universality bias...")
    cat_bias = compute_category_bias(components, universality, categories)
    cat_bias.to_csv(out / "category_bias.csv", index=False)

    # Variance decomposition: SS_between and SS_within vs universality
    log.info("Computing variance decomposition stats...")
    ss_between_within = compute_within_model_correlations(
        selectivity, "universality", "ss_between",
    )
    ss_within_within = compute_within_model_correlations(
        selectivity, "universality", "ss_within",
    )

    # Normalize to % of total for quartile display
    valid = selectivity.dropna(subset=["universality", "eta_squared"])
    valid = valid.assign(
        ss_total=valid["ss_between"] + valid["ss_within"],
        ss_between_pct=lambda d: d["ss_between"] / (d["ss_between"] + d["ss_within"]) * 100,
        ss_within_pct=lambda d: d["ss_within"] / (d["ss_between"] + d["ss_within"]) * 100,
    )
    q_col = pd.qcut(valid["universality"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    variance_stats = {
        "ss_between": {
            "within_model": ss_between_within,
            "by_quartile": {
                q: float(grp["ss_between_pct"].mean())
                for q, grp in valid.assign(q=q_col).groupby("q", observed=True)
            },
        },
        "ss_within": {
            "within_model": ss_within_within,
            "by_quartile": {
                q: float(grp["ss_within_pct"].mean())
                for q, grp in valid.assign(q=q_col).groupby("q", observed=True)
            },
        },
    }

    # Aggregate stats
    rho_overall, p_overall = spearmanr(valid["universality"], valid["eta_squared"])

    n_cats = len(np.unique(categories))
    n_items = len(categories)

    # Joint eta^2 x c_eff quadrants (median split). Together they distinguish:
    #   high eta^2, low c_eff  : category-specific factor (e.g. "dogs")
    #   high eta^2, high c_eff : broad semantic factor (e.g. "many animals")
    #   low eta^2, high c_eff  : broad non-category visual factor
    #   low eta^2, low c_eff   : narrow non-category factor (idiosyncratic)
    valid_ce = selectivity.dropna(
        subset=["universality", "eta_squared", "c_eff", "c_eff_excess"],
    )
    eta_med = float(valid_ce["eta_squared"].median())
    ceff_med = float(valid_ce["c_eff"].median())
    ceff_ex_med = float(valid_ce["c_eff_excess"].median())
    high_eta = valid_ce["eta_squared"] >= eta_med
    high_ceff = valid_ce["c_eff"] >= ceff_med
    high_ceff_ex = valid_ce["c_eff_excess"] >= ceff_ex_med
    rho_ce_overall, p_ce_overall = spearmanr(
        valid_ce["universality"], valid_ce["c_eff"],
    )
    rho_eta_ce, p_eta_ce = spearmanr(
        valid_ce["eta_squared"], valid_ce["c_eff"],
    )
    rho_ce_ex_overall, p_ce_ex_overall = spearmanr(
        valid_ce["universality"], valid_ce["c_eff_excess"],
    )
    rho_eta_ce_ex, p_eta_ce_ex = spearmanr(
        valid_ce["eta_squared"], valid_ce["c_eff_excess"],
    )

    all_stats = {
        "rank": rank,
        "n_models": len(components),
        "n_dimensions": len(selectivity),
        "n_categories": n_cats,
        "n_items": n_items,
        "eta_squared": {
            "overall_rho": float(rho_overall),
            "overall_p": float(p_overall),
            "within_model": within_stats,
            "null_expected": float((n_cats - 1) / (n_items - 1)),
            "mean": float(valid["eta_squared"].mean()),
            "std": float(valid["eta_squared"].std()),
            "by_quartile": {
                q: float(grp["eta_squared"].mean())
                for q, grp in valid.assign(
                    q=pd.qcut(valid["universality"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
                ).groupby("q", observed=True)
            },
        },
        "c_eff": {
            "overall_rho_universality": float(rho_ce_overall),
            "overall_p_universality": float(p_ce_overall),
            "within_model_universality": c_eff_within,
            "overall_rho_eta_squared": float(rho_eta_ce),
            "overall_p_eta_squared": float(p_eta_ce),
            "within_model_eta_squared": eta_c_eff_within,
            "n_categories_max": n_cats,
            "mean": float(valid_ce["c_eff"].mean()),
            "std": float(valid_ce["c_eff"].std()),
            "median": ceff_med,
            "p25": float(valid_ce["c_eff"].quantile(0.25)),
            "p75": float(valid_ce["c_eff"].quantile(0.75)),
            "by_quartile": {
                q: float(grp["c_eff"].mean())
                for q, grp in valid_ce.assign(
                    q=pd.qcut(valid_ce["universality"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
                ).groupby("q", observed=True)
            },
            "joint_eta_quadrants": {
                "eta_thresh": eta_med,
                "c_eff_thresh": ceff_med,
                "category_specific":  int((high_eta & ~high_ceff).sum()),
                "broad_semantic":     int((high_eta & high_ceff).sum()),
                "broad_noncategory":  int((~high_eta & high_ceff).sum()),
                "narrow_noncategory": int((~high_eta & ~high_ceff).sum()),
            },
        },
        "c_eff_excess": {
            "overall_rho_universality": float(rho_ce_ex_overall),
            "overall_p_universality": float(p_ce_ex_overall),
            "within_model_universality": c_eff_excess_within,
            "overall_rho_eta_squared": float(rho_eta_ce_ex),
            "overall_p_eta_squared": float(p_eta_ce_ex),
            "within_model_eta_squared": eta_c_eff_excess_within,
            "n_categories_max": n_cats,
            "mean": float(valid_ce["c_eff_excess"].mean()),
            "std": float(valid_ce["c_eff_excess"].std()),
            "median": ceff_ex_med,
            "p25": float(valid_ce["c_eff_excess"].quantile(0.25)),
            "p75": float(valid_ce["c_eff_excess"].quantile(0.75)),
            "by_quartile": {
                q: float(grp["c_eff_excess"].mean())
                for q, grp in valid_ce.assign(
                    q=pd.qcut(valid_ce["universality"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
                ).groupby("q", observed=True)
            },
            "joint_eta_quadrants": {
                "eta_thresh": eta_med,
                "c_eff_excess_thresh": ceff_ex_med,
                "category_specific":  int((high_eta & ~high_ceff_ex).sum()),
                "broad_semantic":     int((high_eta & high_ceff_ex).sum()),
                "broad_noncategory":  int((~high_eta & high_ceff_ex).sum()),
                "narrow_noncategory": int((~high_eta & ~high_ceff_ex).sum()),
            },
        },
        "variance_decomposition": variance_stats,
        "category_bias": {
            "max_bias": float(cat_bias["universality_bias"].max()),
            "min_bias": float(cat_bias["universality_bias"].min()),
            "mean_bias": float(cat_bias["universality_bias"].mean()),
            "frac_positive": float((cat_bias["universality_bias"] > 0).mean()),
        },
        "importance_universality": imp_stats,
    }
    (out / "stats.json").write_text(json.dumps(all_stats, indent=2, default=str))
    log.info("Saved stats.json")

    return {
        "selectivity": selectivity,
        "within_stats": within_stats,
        "variance_stats": variance_stats,
        "imp_df": imp_df,
        "idio_frac": idio_frac,
        "imp_stats": imp_stats,
        "components": components,
        "universality": universality,
        "categories": categories,
    }


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    results = run_analysis(cfg.factorization.rank, out)

    log.info("Generating figures...")
    make_all_plots(
        selectivity=results["selectivity"],
        within_stats=results["within_stats"],
        variance_stats=results["variance_stats"],
        components=results["components"],
        categories=results["categories"],
        output_dir=out,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
