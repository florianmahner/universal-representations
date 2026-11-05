#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.plotting import setup_plotting
from experiments.universality.shared import load_universality

from .contrasts.plot import generate_all_deepnsd_conwell, generate_new_contrasts
from .contrasts.uniform_tests.run import (
    apply_corrections,
    run_clip_vit_convergence,
    run_contrast,
)
from .data import COMPARE_COLS, CONTRAST_DESCRIPTIONS, load_merged
from .regression.analysis import run_all as run_regression

N_BOOTSTRAP = 10_000


def generate_report(
    df,
    contrast_stats: list[dict],
    regression_results: dict,
    rank: int,
    contrast_prefix: str,
    regression_prefix: str,
) -> str:
    """Generate a markdown report with all results."""
    lines = []
    lines.append("# Controlled Analysis: Model Properties and Universality")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"This report examines how specific model properties (architecture, training "
        f"objective, training data, scale) causally relate to representational "
        f"universality across {len(df)} deep neural network models "
        f"(NMF rank = {rank}). "
        f"We use two complementary approaches:"
    )
    lines.append("")
    lines.append(
        "1. **Controlled contrasts**: Subsets of models where only one property varies, "
        "isolating its effect on universality."
    )
    lines.append(
        "2. **Multiple regression**: Simultaneous estimation of each property's unique "
        "contribution while controlling for all others."
    )
    lines.append("")
    lines.append(
        "The contrast design follows Conwell et al.'s model typology from the DeepNSD "
        "project, which curates controlled model subsets where confounding variables "
        "are held constant."
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Part 1: Controlled Contrasts")
    lines.append("")
    lines.append(
        "Each contrast below isolates a single model property while holding others "
        "approximately constant. Two-group contrasts use Welch's t, multi-group "
        "contrasts use one-way ANOVA, and singleton-only contrasts are descriptive. "
        "P-values use the manuscript's Bonferroni denominator of all ten "
        "pre-specified contrast rows (singleton-only designs remain descriptive). "
        "Effect sizes use Hedges' g or partial eta-squared with stratified-bootstrap "
        "95% confidence intervals."
    )
    lines.append("")

    testable_stats = sorted(
        [s for s in contrast_stats
         if np.isfinite(s.get("p_value", float("nan")))],
        key=lambda s: s["p_value"],
    )
    descriptive_stats = [
        s for s in contrast_stats
        if not np.isfinite(s.get("p_value", float("nan")))
    ]
    sorted_stats = testable_stats

    for stat in testable_stats + descriptive_stats:
        col = stat["contrast"]
        label = stat["label"]
        desc = CONTRAST_DESCRIPTIONS.get(col, "")
        short_name = col.replace("compare_", "")

        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"**Design**: {desc}")
        lines.append("")
        lines.append(f"**Sample**: n = {stat['n_total']} models across {stat['n_groups']} groups.")
        lines.append("")

        lines.append("| Group | n | Mean | SD |")
        lines.append("|-------|---|------|-----|")
        for g in sorted(stat["group_means"].keys(), key=lambda k: stat["group_means"][k], reverse=True):
            lines.append(
                f"| {g} | {stat['group_sizes'][g]} | "
                f"{stat['group_means'][g]:.4f} | {stat['group_stds'][g]:.4f} |"
            )
        lines.append("")

        if not np.isfinite(stat.get("p_value", float("nan"))):
            lines.append(
                "**Note**: All groups are singletons (n=1 each), so no inferential "
                "statistics are possible. Values shown are descriptive only."
            )
            lines.append("")
        else:
            p = stat["p_bonferroni"]
            p_str = f"p_Bonf = {p:.4f}" if p >= 0.0001 else "p_Bonf < 0.0001"

            lines.append(
                f"**{stat['test']}**: statistic = {stat['statistic']:.3f}, "
                f"raw p = {stat['p_value']:.4f}, {p_str}; "
                f"{stat['effect_size_type']} = {stat['effect_size']:.3f}, "
                f"95% CI [{stat['effect_ci_low']:.3f}, "
                f"{stat['effect_ci_high']:.3f}]"
            )
            if np.isfinite(stat.get("omega_squared", float("nan"))):
                lines.append(
                    f"Results-paragraph estimator: omega-squared = "
                    f"{stat['omega_squared']:.3f}, 95% bootstrap CI "
                    f"[{stat['omega_ci_low']:.3f}, {stat['omega_ci_high']:.3f}]."
                )
            lines.append("")

        if "pairwise" in stat and stat["pairwise"]:
            lines.append("<details><summary>Post-hoc pairwise comparisons (BH-FDR corrected)</summary>")
            lines.append("")
            lines.append("| Group A | Group B | t | p (raw) | p (FDR) |")
            lines.append("|---------|---------|---|---------|---------|")
            for pw in sorted(stat["pairwise"], key=lambda x: x["p_raw"]):
                lines.append(
                    f"| {pw['group_a']} | {pw['group_b']} | {pw['t']:.2f} | "
                    f"{pw['p_raw']:.4f} | {pw['p_fdr']:.4f} |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append(f"![{label}]({contrast_prefix}/{short_name}.pdf)")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Part 2: Regression Analysis")
    lines.append("")
    lines.append(
        "While controlled contrasts isolate individual factors, regression analysis "
        "estimates the unique contribution of each property while controlling for all "
        "others simultaneously. This addresses the remaining confounds within each "
        "contrast (e.g., model size differences within the architecture contrast)."
    )
    lines.append("")

    corr_df = regression_results.get("correlations")
    if corr_df is not None and not corr_df.empty:
        lines.append("### Pairwise Correlations")
        lines.append("")
        lines.append("| Predictor | n | Pearson r | p | Spearman rho | p |")
        lines.append("|-----------|---|-----------|---|--------------|---|")
        for _, row in corr_df.iterrows():
            pp = f"{row['pearson_p']:.4f}" if row['pearson_p'] >= 0.0001 else "<.0001"
            sp = f"{row['spearman_p']:.4f}" if row['spearman_p'] >= 0.0001 else "<.0001"
            lines.append(
                f"| {row['predictor']} | {row['n']} | {row['pearson_r']:.3f} | "
                f"{pp} | {row['spearman_rho']:.3f} | {sp} |"
            )
        lines.append("")
        lines.append(f"![Correlation Matrix]({regression_prefix}/correlation_matrix.pdf)")
        lines.append("")

    lines.append("### Continuous Predictors")
    lines.append("")
    lines.append(f"![Scatter Plots]({regression_prefix}/scatter_continuous.pdf)")
    lines.append("")

    reg_result = regression_results.get("regression", {})
    if "error" not in reg_result and reg_result:
        lines.append("### Multiple Regression")
        lines.append("")
        lines.append(
            f"**Model fit**: R\u00b2 = {reg_result['r_squared']:.3f}, "
            f"adjusted R\u00b2 = {reg_result['adj_r_squared']:.3f}, "
            f"n = {reg_result['n']}"
        )
        lines.append("")
        coef_df = reg_result["coefficients"]
        lines.append("| Predictor | Beta | SE | t | p |")
        lines.append("|-----------|------|-----|---|---|")
        for _, row in coef_df.iterrows():
            sig = " *" if row["p"] < 0.05 else ""
            pp = f"{row['p']:.4f}" if row['p'] >= 0.0001 else "<.0001"
            lines.append(
                f"| {row['predictor']}{sig} | {row['beta']:.4f} | "
                f"{row['se']:.4f} | {row['t']:.2f} | {pp} |"
            )
        lines.append("")
        lines.append(f"![Regression Coefficients]({regression_prefix}/regression_coefficients.pdf)")
        lines.append("")

    partial_df = regression_results.get("partial")
    if partial_df is not None and not partial_df.empty:
        lines.append("### Partial Correlations (Unique Variance)")
        lines.append("")
        lines.append(
            "Partial correlations measure the association between each predictor "
            "and universality after removing the variance shared with all other "
            "predictors. This reveals each factor's *unique* explanatory power."
        )
        lines.append("")
        lines.append("| Predictor | Partial r | Partial r\u00b2 | p |")
        lines.append("|-----------|-----------|------------|---|")
        for _, row in partial_df.iterrows():
            sig = " *" if row["p_value"] < 0.05 else ""
            pp = f"{row['p_value']:.4f}" if row['p_value'] >= 0.0001 else "<.0001"
            lines.append(
                f"| {row['predictor']}{sig} | {row['partial_r']:.3f} | "
                f"{row['partial_r_squared']:.3f} | {pp} |"
            )
        lines.append("")
        lines.append(f"![Partial Correlations]({regression_prefix}/partial_correlations.pdf)")
        lines.append("")

    vp_df = regression_results.get("variance_partitioning")
    if vp_df is not None and not vp_df.empty:
        lines.append("### Variance Partitioning")
        lines.append("")
        lines.append(
            "Hierarchical regression decomposes the total explained variance into "
            "unique and shared contributions of three predictor groups."
        )
        lines.append("")
        lines.append("| Predictor Group | R\u00b2 alone | R\u00b2 unique | R\u00b2 shared |")
        lines.append("|-----------------|----------|-----------|-----------|")
        for _, row in vp_df.iterrows():
            lines.append(
                f"| {row['predictor_group']} | {row['r2_alone']:.3f} | "
                f"{row['r2_incremental']:.3f} | {row['r2_shared']:.3f} |"
            )
        lines.append("")
        lines.append(f"![Variance Partitioning]({regression_prefix}/variance_partitioning.pdf)")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Discussion")
    lines.append("")

    sig_contrasts = [s for s in sorted_stats if s["p_bonferroni"] < 0.05]
    nonsig_contrasts = [s for s in sorted_stats if s["p_bonferroni"] >= 0.05]

    if sig_contrasts:
        biggest = sig_contrasts[0]
        lines.append("### Significant Effects")
        lines.append("")
        lines.append(f"Of the {len(sorted_stats)} testable controlled contrasts, "
                     f"**{len(sig_contrasts)}** remained significant after Bonferroni correction:")
        lines.append("")
        for s in sig_contrasts:
            p_str = (
                f"p_Bonf = {s['p_bonferroni']:.4f}"
                if s["p_bonferroni"] >= 0.0001
                else "p_Bonf < 0.0001"
            )
            lines.append(
                f"- **{s['label']}** (n={s['n_total']}): {p_str}"
            )
        lines.append("")

        lines.append(
            f"The most significant effect was **{biggest['label']}**. "
        )
        if biggest["contrast"] == "compare_goal_contrastive":
            lines.append(
                "Contrastive self-supervised methods (PIRL, SimCLR, MoCo, SwAV, "
                "BarlowTwins, DeepCluster) produce substantially more universal "
                "representations than non-contrastive methods (JigSaw, RotNet, ClusterFit), "
                "even when architecture and training data are held constant."
            )
        elif biggest["contrast"] == "compare_goal_slip":
            lines.append(
                "Among ViT models of matched scale, SimCLR produces the most universal "
                "representations, followed by SLIP and then CLIP. This suggests that "
                "pure visual contrastive learning yields more shared structure than "
                "language-guided objectives."
            )
        elif biggest["contrast"] == "compare_architecture":
            lines.append(
                "Among ImageNet-trained models, CNNs produce more universal representations "
                "than Transformers and MLP-Mixers, potentially reflecting the inductive "
                "biases of convolutional architectures."
            )
        lines.append("")

    if nonsig_contrasts:
        lines.append("### Non-Significant Effects")
        lines.append("")
        lines.append(f"**{len(nonsig_contrasts)}** testable contrasts did not reach significance:")
        lines.append("")
        for s in nonsig_contrasts:
            lines.append(
                f"- **{s['label']}** (n={s['n_total']}): "
                f"p_Bonf = {s['p_bonferroni']:.4f}"
            )
        lines.append("")

    if descriptive_stats:
        lines.append("### Descriptive-Only Contrasts")
        lines.append("")
        lines.append(
            f"**{len(descriptive_stats)}** contrasts have only singleton groups "
            "(n=1 per level), precluding inferential statistics. These include: "
            + ", ".join(s["label"] for s in descriptive_stats)
            + ". While no formal tests are possible, the rank ordering of "
            "universality across conditions is informative:"
        )
        lines.append("")
        for s in descriptive_stats:
            top = max(s["group_means"], key=s["group_means"].get)
            bot = min(s["group_means"], key=s["group_means"].get)
            spread = s["group_means"][top] - s["group_means"][bot]
            lines.append(
                f"- **{s['label']}**: range = {spread:.4f} "
                f"(highest: {top} = {s['group_means'][top]:.4f}, "
                f"lowest: {bot} = {s['group_means'][bot]:.4f})"
            )
        lines.append("")

    if "error" not in reg_result and reg_result:
        lines.append("### Regression Insights")
        lines.append("")
        lines.append(
            f"The multiple regression model explained **{reg_result['r_squared']*100:.1f}%** "
            f"of variance in universality (adjusted R\u00b2 = {reg_result['adj_r_squared']:.3f}), "
            f"based on n = {reg_result['n']} models with complete data."
        )
        lines.append("")

        sig_coefs = reg_result["coefficients"]
        sig_coefs = sig_coefs[(sig_coefs["predictor"] != "intercept") & (sig_coefs["p"] < 0.05)]
        if len(sig_coefs) > 0:
            lines.append(f"{len(sig_coefs)} predictor(s) reached significance:")
            lines.append("")
            for _, row in sig_coefs.iterrows():
                direction = "positively" if row["beta"] > 0 else "negatively"
                lines.append(f"- **{row['predictor']}** ({direction} associated, "
                             f"beta = {row['beta']:.4f}, p = {row['p']:.4f})")
            lines.append("")

    if vp_df is not None and not vp_df.empty:
        groups_only = vp_df[vp_df["predictor_group"] != "Full Model"]
        biggest_unique = groups_only.loc[groups_only["r2_incremental"].idxmax()]
        total_shared = groups_only["r2_shared"].sum()
        lines.append(
            f"Variance partitioning revealed that **{biggest_unique['predictor_group']}** "
            f"contributed the most unique variance "
            f"(R\u00b2 unique = {biggest_unique['r2_incremental']:.3f}). "
            f"The total shared variance ({total_shared:.3f}) indicates moderate "
            f"collinearity between predictor groups, as expected given that architecture "
            f"and training objective choices are correlated in the model ecosystem."
        )
        lines.append("")

    lines.append("### Limitations")
    lines.append("")
    lines.append(
        "1. **Observational design**: While the controlled contrasts approximate "
        "experimental manipulation, they rely on the assumption that models within "
        "a contrast group are truly matched on all properties except the target. "
        "Residual confounds may remain."
    )
    lines.append(
        "2. **Sample sizes**: Some contrasts have very small "
        "samples, limiting both statistical power and generalizability."
    )
    lines.append(
        "3. **Multiple comparisons**: All ten pre-specified contrast rows define "
        "the primary Bonferroni family, including rows that can only be reported "
        "descriptively because their groups are singletons. A testable-only "
        "Bonferroni correction is retained as a labeled sensitivity analysis."
    )
    lines.append(
        "4. **Model ecosystem bias**: The model zoo is not a random sample from "
        "all possible neural networks. It over-represents certain architectures "
        "(ResNets, ViTs) and training regimes (ImageNet classification)."
    )
    lines.append(
        "5. **Universality metric**: Results depend on the specific operationalization "
        "of universality (cross-model similarity of NMF components). Different "
        "universality measures might yield different patterns."
    )
    lines.append("")

    lines.append("### Conclusion")
    lines.append("")
    lines.append(
        "By combining controlled contrasts with regression analysis, we provide "
        "complementary evidence about which model properties shape representational "
        "universality. The contrast-based approach offers cleaner causal "
        "interpretations for individual factors, while regression quantifies their "
        "relative importance in a unified framework."
    )
    lines.append("")

    return "\n".join(lines)


def _serialize_stats(contrast_stats: list[dict]) -> list[dict]:
    """Make contrast stats JSON-serializable."""
    out = []
    for s in contrast_stats:
        s_copy = dict(s)
        for k, v in s_copy.items():
            if isinstance(v, (np.integer,)):
                s_copy[k] = int(v)
            elif isinstance(v, (np.floating,)):
                s_copy[k] = float(v)
        out.append(s_copy)
    return out


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    setup_plotting()

    rank = cfg.factorization.rank
    print("Controlled Analysis: Model Properties and Universality")

    df = load_merged(rank=rank)
    print(f"  Loaded {len(df)} models with universality scores (rank={rank})")

    contrast_dir = Path(__file__).parent / "contrasts" / "outputs" / f"rank_{rank}"
    stats_dir = (
        Path(__file__).parent / "contrasts" / "stats" / "outputs" / f"rank_{rank}"
    )
    regression_dir = (
        Path(__file__).parent / "regression" / "outputs" / f"rank_{rank}"
    )
    for d in (contrast_dir, stats_dir, regression_dir):
        d.mkdir(parents=True, exist_ok=True)

    df.to_csv(contrast_dir / "merged_data.csv", index=False)

    print(f"\n  Part 1: Controlled Contrasts (rank={rank})")
    table = apply_corrections(
        pd.DataFrame(
            [
                run_contrast(df, col, n_bootstrap=N_BOOTSTRAP, seed=cfg.seed + index)
                for index, col in enumerate(COMPARE_COLS)
            ]
        )
    )
    table.to_csv(stats_dir / "manuscript_contrast_results.csv", index=False)
    clip_convergence = run_clip_vit_convergence(
        df, n_bootstrap=N_BOOTSTRAP, seed=cfg.seed
    )
    (stats_dir / "clip_vit_convergence.json").write_text(
        json.dumps(clip_convergence, indent=2)
    )
    contrast_stats = table.to_dict("records")

    uni_dim = load_universality(rank=rank, dataset=cfg.dataset)
    generate_all_deepnsd_conwell(uni_dim, df, contrast_dir)
    generate_new_contrasts(uni_dim, df, contrast_dir)

    (stats_dir / "stats.json").write_text(
        json.dumps(_serialize_stats(contrast_stats), indent=2, default=str)
    )

    print(f"\n  Part 2: Regression Analysis (rank={rank})")
    regression_results = run_regression(df, output_dir=regression_dir)

    report = generate_report(
        df,
        contrast_stats,
        regression_results,
        rank=rank,
        contrast_prefix=".",
        regression_prefix=f"../../../regression/outputs/rank_{rank}",
    )
    (contrast_dir / "report.md").write_text(report)

    print("\nDone! Outputs saved to:")
    print(f"  Contrasts:  {contrast_dir}")
    print(f"  Regression: {regression_dir}")


if __name__ == "__main__":
    main()
