#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.colors import GRAY_DARK, GRAY_PALE, INDIGO, ROSE
from src.plotting import PANEL_LABEL_FONT, PAPER_WIDTH, paper_style

EXPECTED_DIMENSIONS = 1_059
N_BOOTSTRAP = 10_000


def analysis_specs() -> tuple[dict[str, Any], ...]:
    """The two alignment axes, each pairing an LLM score with a human vote share."""
    return (
        {
            "analysis": "semantic_alignment",
            "axis": "semantic",
            "llm_measure": "semantic_strength",
            "human_measure": "human_semantic_or_both_fraction",
            "human_definition": "(n_semantic + n_both) / n_raters",
            "title": "Shared meaning",
            "x_label": "LLM score: images share a meaning",
            "y_label": "Humans choosing Semantic or Both",
            "color": INDIGO,
        },
        {
            "analysis": "visual_alignment",
            "axis": "visual",
            "llm_measure": "visual_strength",
            "human_measure": "human_visual_or_both_fraction",
            "human_definition": "(n_visual + n_both) / n_raters",
            "title": "Shared appearance",
            "x_label": "LLM score: images share a visual feature",
            "y_label": "Humans choosing Visual or Both",
            "color": ROSE,
        },
    )


def _require_columns(
    frame: pd.DataFrame, required: set[str], source: Path
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _require_expected_dimensions(
    frame: pd.DataFrame, source: Path, expected: int = EXPECTED_DIMENSIONS
) -> None:
    if len(frame) != expected:
        raise ValueError(f"Expected {expected:,} rows in {source}, found {len(frame):,}")
    n_unique = int(frame["dim_id"].nunique(dropna=False))
    if n_unique != expected:
        duplicates = frame.loc[
            frame["dim_id"].duplicated(keep=False), "dim_id"
        ].astype(str)
        examples = ", ".join(duplicates.drop_duplicates().head(5))
        raise ValueError(
            f"Expected {expected:,} unique dim_id values in {source}, found "
            f"{n_unique:,}. Duplicate examples: {examples}"
        )


def _numeric_equal(
    left: pd.Series, right: pd.Series, *, atol: float = 1e-12
) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
            atol=atol,
            rtol=0.0,
            equal_nan=True,
        )
    )


def load_and_validate(
    output_dir: Path, human_ratings: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the completed run and independently verify its human-data join."""
    dimension_path = output_dir / "dimension_summary.csv"
    summary_path = output_dir / "summary.json"
    if not dimension_path.exists():
        raise FileNotFoundError(f"Missing LLM summary: {dimension_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {summary_path}")
    if not human_ratings.exists():
        raise FileNotFoundError(f"Missing canonical human ratings: {human_ratings}")

    run_summary = json.loads(summary_path.read_text())
    llm = pd.read_csv(dimension_path)
    human = pd.read_csv(human_ratings)

    llm_required = {
        "dim_id",
        "model",
        "semantic_strength",
        "visual_strength",
        "n_classifications",
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    }
    human_required = {
        "dim_id",
        "model",
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    }
    _require_columns(llm, llm_required, dimension_path)
    _require_columns(human, human_required, human_ratings)
    _require_expected_dimensions(llm, dimension_path)
    _require_expected_dimensions(human, human_ratings)

    if int(run_summary.get("n_dimensions", -1)) != EXPECTED_DIMENSIONS:
        raise ValueError(
            "summary.json does not describe the expected completed experiment: "
            f"n_dimensions={run_summary.get('n_dimensions')!r}"
        )
    if (
        int(run_summary.get("dimensions_with_all_classifications", -1))
        != EXPECTED_DIMENSIONS
    ):
        raise ValueError(
            "The LLM classification run is incomplete: "
            f"{run_summary.get('dimensions_with_all_classifications')!r}/"
            f"{EXPECTED_DIMENSIONS:,} dimensions have all classifications"
        )

    score_columns = ["semantic_strength", "visual_strength"]
    if llm[score_columns].isna().any().any():
        missing = llm[score_columns].isna().sum()
        raise ValueError(f"LLM scores contain missing values:\n{missing}")
    for column in score_columns:
        values = pd.to_numeric(llm[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise ValueError(f"{column} must be numeric and bounded by [0, 1]")

    count_columns = [
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    ]
    for source_name, frame in (("LLM summary", llm), ("human ratings", human)):
        numeric_counts = frame[count_columns].apply(pd.to_numeric, errors="coerce")
        if numeric_counts.isna().any().any():
            raise ValueError(f"{source_name} contains missing or non-numeric vote counts")
        if (numeric_counts < 0).any().any():
            raise ValueError(f"{source_name} contains negative vote counts")
        if not np.equal(
            numeric_counts.to_numpy(), np.floor(numeric_counts.to_numpy())
        ).all():
            raise ValueError(f"{source_name} contains non-integer vote counts")
        category_total = numeric_counts[
            ["n_semantic", "n_visual", "n_both", "n_neither"]
        ].sum(axis=1)
        mismatch = category_total != numeric_counts["n_raters"]
        if mismatch.any():
            examples = frame.loc[mismatch, "dim_id"].astype(str).head(5).tolist()
            raise ValueError(
                f"{source_name} has {int(mismatch.sum())} inconsistent vote "
                f"denominators. Examples: {examples}"
            )
        if (numeric_counts["n_raters"] <= 0).any():
            raise ValueError(f"{source_name} contains zero-rater dimensions")

    joined = human[
        [
            "dim_id",
            "model",
            "n_raters",
            "n_semantic",
            "n_visual",
            "n_both",
            "n_neither",
        ]
    ].merge(
        llm[
            [
                "dim_id",
                "model",
                "semantic_strength",
                "visual_strength",
                "n_classifications",
            ]
        ],
        on="dim_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_human", "_llm"),
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any() or len(joined) != EXPECTED_DIMENSIONS:
        examples = joined.loc[unmatched, ["dim_id", "_merge"]].head(5).to_dict(
            orient="records"
        )
        raise ValueError(
            "Human-to-LLM join is not complete and one-to-one: "
            f"{int(unmatched.sum())} unmatched rows; examples={examples}"
        )
    if not (joined["model_human"].astype(str) == joined["model_llm"].astype(str)).all():
        raise ValueError("Source-model identities disagree between human and LLM files")

    # Verify that the human fields already carried by dimension_summary.csv are
    # exactly the canonical values, then use the independent join for analysis.
    crosscheck = llm[["dim_id", *count_columns]].merge(
        human[["dim_id", *count_columns]],
        on="dim_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_llm", "_human"),
    )
    for column in count_columns:
        if not _numeric_equal(
            crosscheck[f"{column}_llm"], crosscheck[f"{column}_human"]
        ):
            raise ValueError(
                f"Human field {column} in dimension_summary.csv disagrees with "
                f"{human_ratings}"
            )

    joined = joined.drop(columns="_merge").rename(
        columns={"model_human": "source_model"}
    )
    joined = joined.drop(columns="model_llm")
    joined["human_semantic_or_both_fraction"] = (
        joined["n_semantic"] + joined["n_both"]
    ) / joined["n_raters"]
    joined["human_visual_or_both_fraction"] = (
        joined["n_visual"] + joined["n_both"]
    ) / joined["n_raters"]

    support_columns = [
        "human_semantic_or_both_fraction",
        "human_visual_or_both_fraction",
    ]
    if not joined[support_columns].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all():
        raise ValueError("Derived human vote fractions must be bounded by [0, 1]")

    validation = {
        "expected_dimensions": EXPECTED_DIMENSIONS,
        "dimension_summary_rows": int(len(llm)),
        "dimension_summary_unique_dimensions": int(llm["dim_id"].nunique()),
        "canonical_human_rows": int(len(human)),
        "canonical_human_unique_dimensions": int(human["dim_id"].nunique()),
        "joined_dimensions": int(len(joined)),
        "unmatched_dimensions": 0,
        "duplicate_dimensions": 0,
        "vote_denominator_mismatches": 0,
        "human_fields_match_canonical_file": True,
        "dimensions_with_all_classifications": int(
            run_summary["dimensions_with_all_classifications"]
        ),
        "source_models": int(joined["source_model"].nunique()),
        "min_human_raters_per_dimension": int(joined["n_raters"].min()),
        "max_human_raters_per_dimension": int(joined["n_raters"].max()),
        "total_human_ratings": int(joined["n_raters"].sum()),
    }
    return joined, validation


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return math.nan, math.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def cluster_bootstrap_correlations(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample source models and recompute dimension-level correlations."""
    specs = analysis_specs()
    cluster_indices = [
        indices.to_numpy(dtype=int)
        for _, indices in frame.groupby("source_model", sort=True).groups.items()
    ]
    if len(cluster_indices) < 2:
        raise ValueError("At least two source-model clusters are required")

    x_values = {
        spec["analysis"]: frame[spec["llm_measure"]].to_numpy(dtype=float)
        for spec in specs
    }
    y_values = {
        spec["analysis"]: frame[spec["human_measure"]].to_numpy(dtype=float)
        for spec in specs
    }
    samples = {spec["analysis"]: np.empty(n_bootstrap, dtype=float) for spec in specs}
    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_indices)

    for bootstrap_index in range(n_bootstrap):
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        sampled_rows = np.concatenate(
            [cluster_indices[index] for index in sampled_clusters]
        )
        for spec in specs:
            key = spec["analysis"]
            result = spearmanr(
                x_values[key][sampled_rows], y_values[key][sampled_rows]
            )
            samples[key][bootstrap_index] = float(result.statistic)

    for key, values in samples.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite bootstrap correlation for {key}")
    return samples


def compute_results(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    bootstrap = cluster_bootstrap_correlations(
        frame, n_bootstrap=n_bootstrap, seed=seed
    )
    rows: list[dict[str, Any]] = []
    for spec in analysis_specs():
        rho, p_value = _spearman(
            frame[spec["llm_measure"]], frame[spec["human_measure"]]
        )
        samples = bootstrap[spec["analysis"]]
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "analysis": spec["analysis"],
                "axis": spec["axis"],
                "llm_measure": spec["llm_measure"],
                "human_measure": spec["human_measure"],
                "human_measure_definition": spec["human_definition"],
                "n_dimensions": int(len(frame)),
                "n_source_models": int(frame["source_model"].nunique()),
                "spearman_rho": rho,
                "spearman_p_value": p_value,
                "cluster_bootstrap_ci_low": float(ci_low),
                "cluster_bootstrap_ci_high": float(ci_high),
                "confidence_level": 0.95,
                "bootstrap_cluster": "source_model",
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def compute_binned_results(
    frame: pd.DataFrame, *, n_bins: int = 10
) -> pd.DataFrame:
    """Return descriptive equal-count bins used for the figure overlay."""
    rows: list[dict[str, Any]] = []
    for spec in analysis_specs():
        subset = frame[
            ["dim_id", spec["llm_measure"], spec["human_measure"]]
        ].copy()
        subset["bin"] = pd.qcut(
            subset[spec["llm_measure"]],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        grouped = subset.groupby("bin", observed=True)
        for bin_index, group in grouped:
            human_values = group[spec["human_measure"]].astype(float)
            rows.append(
                {
                    "analysis": spec["analysis"],
                    "axis": spec["axis"],
                    "bin": int(bin_index) + 1,
                    "n_dimensions": int(len(group)),
                    "llm_score_min": float(group[spec["llm_measure"]].min()),
                    "llm_score_mean": float(group[spec["llm_measure"]].mean()),
                    "llm_score_max": float(group[spec["llm_measure"]].max()),
                    "human_vote_fraction_mean": float(human_values.mean()),
                    "human_vote_fraction_sd": float(human_values.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.11,
        label,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_FONT,
        fontweight="bold",
        va="top",
    )


def _quiet_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color=GRAY_PALE, linewidth=0.35)
    axis.set_axisbelow(True)


def plot_alignment(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    binned: pd.DataFrame,
    figure_dir: Path,
) -> list[str]:
    """Render a spacious two-panel continuous human-alignment figure."""
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 2.68))
    fig.subplots_adjust(
        left=0.095,
        right=0.988,
        bottom=0.265,
        top=0.845,
        wspace=0.43,
    )

    for panel_index, (axis, spec) in enumerate(zip(axes, analysis_specs())):
        color = spec["color"]
        axis.scatter(
            frame[spec["llm_measure"]],
            frame[spec["human_measure"]],
            s=4.0,
            color=color,
            alpha=0.105,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        trend = binned[binned["analysis"] == spec["analysis"]].sort_values("bin")
        axis.plot(
            trend["llm_score_mean"],
            trend["human_vote_fraction_mean"],
            color=color,
            linewidth=1.15,
            marker="o",
            markersize=3.3,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.75,
            zorder=3,
        )

        row = results.loc[results["analysis"] == spec["analysis"]].iloc[0]
        axis.text(
            0.04,
            0.96,
            (
                f"How closely scores track human votes: "
                f"{row['spearman_rho']:.2f}\n"
                rf"95% interval: {row['cluster_bootstrap_ci_low']:.2f}–"
                rf"{row['cluster_bootstrap_ci_high']:.2f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=GRAY_DARK,
        )
        axis.set(
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
            xticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            yticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            xlabel=spec["x_label"],
            ylabel=spec["y_label"],
            title=f"{spec['title']} (n={len(frame):,})",
        )
        axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _quiet_axis(axis)
        _panel_label(axis, chr(ord("a") + panel_index))

    fig.text(
        0.5,
        0.035,
        (
            "Faint dots are individual dimensions; connected circles average "
            "ten equal-sized groups. A rising line means that higher LLM "
            "scores go with more human votes for the matching label.\n"
            "A human vote of Both counts in both panels."
        ),
        ha="center",
        va="bottom",
        color=GRAY_DARK,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for suffix in ("pdf", "png"):
        path = figure_dir / f"human_llm_continuous_alignment.{suffix}"
        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
        )
        files.append(str(path))
    plt.close(fig)
    return files


def run_analysis(
    output_dir: Path,
    human_ratings: Path,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    frame, validation = load_and_validate(output_dir, human_ratings)
    results = compute_results(frame, n_bootstrap=n_bootstrap, seed=seed)
    binned = compute_binned_results(frame)

    results_path = output_dir / "human_alignment_results.csv"
    binned_path = output_dir / "human_alignment_binned.csv"
    json_path = output_dir / "human_alignment_results.json"
    figure_files = plot_alignment(
        frame, results, binned, output_dir / "figures"
    )

    results.to_csv(results_path, index=False)
    binned.to_csv(binned_path, index=False)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "continuous_llm_human_alignment",
        "source_dimension_summary": str(output_dir / "dimension_summary.csv"),
        "source_human_ratings": str(human_ratings),
        "method": (
            "Dimension-level Spearman correlation between continuous LLM "
            "strength and human axis-support vote fractions; percentile 95% "
            "confidence intervals from a source-model cluster bootstrap."
        ),
        "validation": validation,
        "results": results.to_dict(orient="records"),
        "files": [
            str(results_path),
            str(json_path),
            str(binned_path),
            *figure_files,
        ],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    manifest = run_analysis(
        out,
        Path(cfg.data_root) / "ratings" / "dim_ratings_aggregated_qc_attn_catch.csv",
        n_bootstrap=N_BOOTSTRAP,
        seed=cfg.seed,
    )
    print(
        "Validated "
        f"{manifest['validation']['joined_dimensions']:,} dimensions from "
        f"{manifest['validation']['source_models']:,} source models."
    )
    for row in manifest["results"]:
        print(
            f"{row['axis'].capitalize()}: Spearman rho="
            f"{row['spearman_rho']:.4f}, model-bootstrap 95% CI "
            f"[{row['cluster_bootstrap_ci_low']:.4f}, "
            f"{row['cluster_bootstrap_ci_high']:.4f}]"
        )
    print(f"Wrote results and figures under {out}")


if __name__ == "__main__":
    main()
