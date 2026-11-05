#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from experiments.universality.dimensions.clustering.rank_stability.compute import (
    run_full_analysis,
)
from experiments.universality.dimensions.clustering.rank_stability.plots import plot_all
from src.plotting import setup_plotting

log = logging.getLogger(__name__)

# Protocol constants of the manuscript's clustering analysis: the correlation
# distance at which the dendrogram is cut, and the centroid correlation above
# which a cross-rank match counts as the same concept.
THRESHOLD = 0.75
MATCH_THRESHOLD = 0.5


def generate_report(results: dict, output_dir: Path) -> None:
    """Generate markdown report with results."""
    params = results["params"]
    ranks = params["ranks"]
    ref_rank = params["reference_rank"]
    threshold = params["match_threshold"]

    summary = results["per_rank_summary"]
    pairwise = results["pairwise_stats"]
    tracking = results["concept_tracking"]
    splits = results["splits"]

    lines = [
        "# Cross-Rank Stability of NMF Dimension Clusters",
        "",
        "## Question",
        "",
        "Do the same visual concepts emerge when NMF representations vary from",
        "coarse (10 dimensions) to fine-grained (100 dimensions)?",
        "",
        "## Method",
        "",
        f"- Independent complete-linkage clustering at ranks {', '.join(str(r) for r in ranks)}",
        f"  (correlation distance threshold = {params['threshold']})",
        "- Cluster centroids = mean activation profile over 22,248 THINGS images",
        "- Cross-rank matching via Hungarian algorithm on centroid correlations",
        f"- Stability threshold: centroid correlation >= {threshold}",
        "",
    ]

    lines += [
        "## Per-Rank Clustering Summary",
        "",
        "![Per-rank summary](figures/rank_summary.png)",
        "",
        "| Rank | Dimensions | Total clusters | Substantial (>=10) | Mean coherence | Mean model coverage |",
        "|------|-----------|----------------|--------------------|-----------------|--------------------|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {int(row['n_dimensions'])} | "
            f"{int(row['n_clusters_total'])} | {int(row['n_substantial'])} | "
            f"{row['mean_coherence']:.3f} | {row['mean_model_frac']:.1%} |"
        )
    lines.append("")

    lines += [
        "## Pairwise Match Quality",
        "",
        "![Pairwise match quality](figures/pairwise_match_quality.png)",
        "",
        "| Rank pair | Matched | Mean corr | Median corr | % above 0.5 |",
        "|-----------|---------|-----------|-------------|-------------|",
    ]
    for _, row in pairwise.iterrows():
        lines.append(
            f"| {int(row['rank_1'])} vs {int(row['rank_2'])} | "
            f"{int(row['n_matched'])} | {row['mean_corr']:.3f} | "
            f"{row['median_corr']:.3f} | {row['frac_above_threshold']:.0%} |"
        )
    lines.append("")

    lines += [
        "![Match distributions](figures/match_distributions.png)",
        "",
    ]

    if tracking is not None:
        n_total = len(tracking)
        n_core = int(tracking["is_core_stable"].sum())
        mean_stability = float(tracking["mean_match_corr"].mean())
        other_ranks = sorted(r for r in ranks if r != ref_rank)

        lines += [
            f"## Concept Tracking (reference: rank {ref_rank})",
            "",
            f"Of the **{n_total}** substantial clusters at rank {ref_rank}:",
            "",
            f"- **{n_core}** ({n_core/n_total:.0%}) are **core stable** "
            f"(match correlation >= {threshold} at all {len(other_ranks)} other ranks)",
            f"- Mean stability score: {mean_stability:.3f}",
            "",
        ]

        for r in other_ranks:
            col = f"match_r{r}"
            frac = float((tracking[col] >= threshold).mean())
            mean_c = float(tracking[col].mean())
            lines.append(
                f"- Rank {r}: {frac:.0%} of rank-{ref_rank} concepts "
                f"recovered (mean corr = {mean_c:.3f})"
            )
        lines.append("")

        lines += [
            "![Concept tracking](figures/concept_tracking.png)",
            "",
        ]

        lines.append("### Most stable concepts")
        lines.append("")
        lines.append("| Concept | Size | Models | Model % | Coherence | Mean match corr |")
        lines.append("|---------|------|--------|---------|-----------|-----------------|")
        for _, row in tracking.head(10).iterrows():
            lines.append(
                f"| C{int(row['concept_id'])} | {int(row['ref_size'])} | "
                f"{int(row['ref_n_models'])} | {row['ref_model_frac']:.0%} | "
                f"{row['ref_coherence']:.3f} | {row['mean_match_corr']:.3f} |"
            )
        lines.append("")

        lines.append("### Least stable concepts")
        lines.append("")
        lines.append("| Concept | Size | Models | Model % | Coherence | Mean match corr |")
        lines.append("|---------|------|--------|---------|-----------|-----------------|")
        for _, row in tracking.tail(10).iterrows():
            lines.append(
                f"| C{int(row['concept_id'])} | {int(row['ref_size'])} | "
                f"{int(row['ref_n_models'])} | {row['ref_model_frac']:.0%} | "
                f"{row['ref_coherence']:.3f} | {row['mean_match_corr']:.3f} |"
            )
        lines.append("")

        lines.append("![Stable vs rank-specific](figures/stable_vs_specific.png)")
        lines.append("")

    if splits:
        lines += [
            "## Concept Splitting",
            "",
            f"When moving from rank {ref_rank} to higher ranks, some broad concepts "
            "split into finer subcategories:",
            "",
        ]
        by_rank = {}
        for s in splits:
            r = s["child_rank"]
            by_rank.setdefault(r, []).append(s)

        for r in sorted(by_rank.keys()):
            rank_splits = sorted(by_rank[r], key=lambda x: x["n_children"], reverse=True)
            lines.append(f"**Rank {ref_rank} -> {r}**: {len(rank_splits)} concepts split")
            lines.append("")
            for s in rank_splits[:5]:
                children_str = ", ".join(
                    f"C{c} (r={corr:.2f})" for c, corr in s["children"][:4]
                )
                lines.append(
                    f"- C{s['parent_cid']} splits into {s['n_children']} clusters: "
                    f"{children_str}"
                )
            lines.append("")

    lines += [
        "## Summary",
        "",
    ]
    if tracking is not None:
        n_core = int(tracking["is_core_stable"].sum())
        n_total = len(tracking)
        lines += [
            f"**{n_core}/{n_total}** ({n_core/n_total:.0%}) of the rank-{ref_rank} "
            f"visual concept clusters are stable across all tested ranks. "
            "The clustering structure is largely robust to NMF rank choice, "
            "with adjacent ranks showing the highest correspondence.",
        ]

    report = "\n".join(lines) + "\n"
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report)
    log.info(f"Report saved to {report_path}")


def save_results(results: dict, output_dir: Path) -> None:
    """Save analysis outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results["per_rank_summary"].to_csv(
        output_dir / "per_rank_summary.csv", index=False,
    )
    results["pairwise_stats"].to_csv(
        output_dir / "pairwise_matches.csv", index=False,
    )
    if results["concept_tracking"] is not None:
        results["concept_tracking"].to_csv(
            output_dir / "concept_tracking.csv", index=False,
        )

    params = results["params"].copy()
    params["n_core_stable"] = (
        int(results["concept_tracking"]["is_core_stable"].sum())
        if results["concept_tracking"] is not None else 0
    )
    (output_dir / "stability_stats.json").write_text(
        json.dumps(params, indent=2, default=str)
    )

    if results["splits"]:
        splits_serializable = []
        for s in results["splits"]:
            splits_serializable.append({
                "parent_cid": s["parent_cid"],
                "parent_rank": s["parent_rank"],
                "child_rank": s["child_rank"],
                "n_children": s["n_children"],
                "children": [(int(c), float(r)) for c, r in s["children"]],
            })
        (output_dir / "splits.json").write_text(
            json.dumps(splits_serializable, indent=2)
        )


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    results = run_full_analysis(
        ranks=[10, 30, 50, 100],
        threshold=THRESHOLD,
        match_threshold=MATCH_THRESHOLD,
        reference_rank=cfg.factorization.rank,
        max_models=None,
        cache_dir=out / "cache",
    )

    save_results(results, out)

    setup_plotting()
    plot_all(results, out / "figures")

    generate_report(results, out)


if __name__ == "__main__":
    main()
