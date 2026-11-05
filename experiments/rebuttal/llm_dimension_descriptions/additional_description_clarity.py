#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import rankdata, spearmanr

from src.colors import GRAY_DARK, GRAY_PALE, INDIGO
from src.plotting import PANEL_LABEL_FONT, PAPER_WIDTH, paper_style

HUMAN_LABEL_ORDER = ["semantic", "visual", "both", "neither"]
N_BOOTSTRAP = 5000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, failing with the line number on malformed input."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_number}: {error}"
                ) from error
            record["_record_index"] = line_number
            records.append(record)
    return records


def _midrank_percentile(values: pd.Series) -> pd.Series:
    """Return empirical midrank percentiles in the open interval (0, 1)."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Cannot rank-normalize missing or non-numeric values")
    n_values = len(numeric)
    if n_values == 0:
        return numeric.astype(float)
    ranks = numeric.rank(method="average")
    return (ranks - 0.5) / n_values


def validate_human_template(template_path: Path) -> dict[str, Any]:
    """Verify the question wording and direction of the original 1--7 scale."""
    question = "How easy was it to see what these images have in common?"
    anchors = {"1": "very easy", "7": "very difficult"}
    if not template_path.exists():
        raise FileNotFoundError(f"Human task template not found: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")
    required_fragments = {
        "question": question,
        "low_anchor": "Very easy",
        "high_anchor": "Very difficult",
        "numeric_direction": "1 = very easy, 7 = very difficult",
    }
    missing = [
        name
        for name, fragment in required_fragments.items()
        if fragment.lower() not in template_text.lower()
    ]
    if missing:
        raise ValueError(
            "Could not verify the human interpretability scale in "
            f"{template_path}; missing: {', '.join(missing)}"
        )
    return {
        "template_path": str(template_path.resolve()),
        "question_verified": True,
        "scale_direction_verified": True,
        "question": question,
        "anchors": anchors,
    }


def load_description_scores(
    descriptions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load, validate, calibrate, and aggregate stage-1 description scores."""
    if not descriptions_path.exists():
        raise FileNotFoundError(f"Description records not found: {descriptions_path}")

    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(descriptions_path):
        if record.get("stage") != "description":
            continue
        response = record.get("response")
        if not isinstance(response, dict):
            continue
        row = {
            "dim_id": record.get("dim_id"),
            "description_generator": record.get("requested_model"),
            "resolved_model": record.get("resolved_model"),
            "prompt_version": record.get("prompt_version"),
            "image_mode": record.get("image_mode"),
            "created_at_utc": record.get("created_at_utc", ""),
            "_record_index": record["_record_index"],
            "description_coherence_raw": response.get("coherence"),
            "no_clear_pattern_raw": response.get(
                "no_clear_pattern_probability"
            ),
        }
        rows.append(row)

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError(f"No valid stage-1 records found in {descriptions_path}")

    prompt_versions = sorted(long["prompt_version"].dropna().unique().tolist())
    image_modes = sorted(long["image_mode"].dropna().unique().tolist())
    if len(prompt_versions) != 1:
        raise ValueError(
            "Description records contain multiple prompt versions "
            f"{prompt_versions}; expected exactly one"
        )
    if len(image_modes) != 1:
        raise ValueError(
            "Description records contain multiple image modes "
            f"{image_modes}; expected exactly one"
        )

    required_identity = ["dim_id", "description_generator"]
    if long[required_identity].isna().any().any():
        raise ValueError("Description records contain missing dimension/model IDs")

    # Append-only logs may contain repeated successful calls.  Use the latest
    # call for each measurement key, consistent with the experiment summarizer.
    long = (
        long.sort_values(["created_at_utc", "_record_index"])
        .drop_duplicates(
            [
                "dim_id",
                "description_generator",
                "prompt_version",
                "image_mode",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )
    raw_columns = [
        "description_coherence_raw",
        "no_clear_pattern_raw",
    ]
    for column in raw_columns:
        long[column] = pd.to_numeric(long[column], errors="coerce")
        invalid = long[column].isna() | ~long[column].between(0, 1)
        if invalid.any():
            examples = long.loc[
                invalid, ["dim_id", "description_generator", column]
            ].head()
            raise ValueError(
                f"{column} must be numeric and in [0, 1]. Examples:\n"
                f"{examples.to_string(index=False)}"
            )

    long["description_coherence_percentile"] = long.groupby(
        "description_generator", group_keys=False
    )["description_coherence_raw"].transform(_midrank_percentile)
    long["no_clear_pattern_percentile"] = long.groupby(
        "description_generator", group_keys=False
    )["no_clear_pattern_raw"].transform(_midrank_percentile)

    aggregate = (
        long.groupby("dim_id", as_index=False)
        .agg(
            n_description_generators=(
                "description_generator",
                "nunique",
            ),
            description_coherence_raw=(
                "description_coherence_raw",
                "mean",
            ),
            no_clear_pattern_raw=("no_clear_pattern_raw", "mean"),
            description_coherence_percentile=(
                "description_coherence_percentile",
                "mean",
            ),
            no_clear_pattern_percentile=(
                "no_clear_pattern_percentile",
                "mean",
            ),
            description_coherence_generator_sd=(
                "description_coherence_percentile",
                "std",
            ),
            no_clear_pattern_generator_sd=(
                "no_clear_pattern_percentile",
                "std",
            ),
        )
        .sort_values("dim_id")
        .reset_index(drop=True)
    )

    generator_counts = (
        long.groupby("description_generator")["dim_id"]
        .nunique()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    diagnostics = {
        "n_records": int(len(long)),
        "n_dimensions": int(long["dim_id"].nunique()),
        "n_generators": int(long["description_generator"].nunique()),
        "records_per_generator": generator_counts,
        "prompt_version": prompt_versions[0],
        "image_mode": image_modes[0],
        "dimensions_by_generator_count": {
            str(int(key)): int(value)
            for key, value in aggregate["n_description_generators"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return long, aggregate, diagnostics


def load_human_difficulty(human_trials_path: Path) -> pd.DataFrame:
    """Validate and aggregate the original human interpretability ratings."""
    if not human_trials_path.exists():
        raise FileNotFoundError(
            f"Human trial ratings not found: {human_trials_path}"
        )
    trials = pd.read_csv(human_trials_path)
    required = {
        "participant",
        "dim_id",
        "task_type",
        "category",
        "interpretability",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(
            "Human trial file is missing required columns: " + ", ".join(missing)
        )

    trials = trials[trials["task_type"] == "real"].copy()
    if trials.empty:
        raise ValueError("Human trial file contains no real trials")
    if trials[["participant", "dim_id"]].isna().any().any():
        raise ValueError("Human trial file contains missing participant/dimension IDs")
    if trials.duplicated(["participant", "dim_id"]).any():
        raise ValueError(
            "Human trial file contains duplicate participant-by-dimension rows"
        )

    trials["interpretability"] = pd.to_numeric(
        trials["interpretability"], errors="coerce"
    )
    invalid_scale = (
        trials["interpretability"].isna()
        | ~trials["interpretability"].between(1, 7)
        | (trials["interpretability"] % 1 != 0)
    )
    if invalid_scale.any():
        raise ValueError(
            "Interpretability ratings must be integer values from 1 to 7"
        )
    trials["category"] = trials["category"].str.lower().replace(
        {"mixed": "both"}
    )
    unknown = sorted(set(trials["category"]) - set(HUMAN_LABEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected human categories: {unknown}")

    difficulty = (
        trials.groupby("dim_id", as_index=False)
        .agg(
            n_human_ratings=("interpretability", "size"),
            human_difficulty_mean=("interpretability", "mean"),
            human_difficulty_median=("interpretability", "median"),
            human_difficulty_sd=("interpretability", "std"),
        )
        .sort_values("dim_id")
        .reset_index(drop=True)
    )
    category_counts = (
        trials.groupby(["dim_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=HUMAN_LABEL_ORDER, fill_value=0)
        .rename(columns=lambda value: f"raw_n_{value}")
        .reset_index()
    )
    return difficulty.merge(
        category_counts, on="dim_id", how="left", validate="one_to_one"
    )


def assemble_dimension_data(
    summary_path: Path,
    description_scores: pd.DataFrame,
    human_difficulty: pd.DataFrame,
    *,
    min_generators: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the three sources and enforce one-row-per-dimension integrity."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Dimension summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {
        "dim_id",
        "model",
        "universality",
        "majority_label",
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(
            "Dimension summary is missing required columns: " + ", ".join(missing)
        )
    if summary["dim_id"].duplicated().any():
        raise ValueError("Dimension summary has duplicate dim_id rows")

    optional_summary_columns = [
        "majority_frac",
        "pairwise_agreement",
        "n_descriptions",
    ]
    ordered_summary_columns = [
        "dim_id",
        "model",
        "universality",
        "majority_label",
        *[
            column
            for column in optional_summary_columns
            if column in summary.columns
        ],
        "n_raters",
        "n_semantic",
        "n_visual",
        "n_both",
        "n_neither",
    ]
    summary = summary[ordered_summary_columns].copy()
    summary["universality"] = pd.to_numeric(
        summary["universality"], errors="coerce"
    )
    if summary[["model", "universality", "majority_label"]].isna().any().any():
        raise ValueError(
            "Dimension summary contains missing model, universality, or labels"
        )
    summary["human_label"] = (
        summary["majority_label"].str.lower().replace({"mixed": "both"})
    )
    unknown = sorted(set(summary["human_label"]) - set(HUMAN_LABEL_ORDER))
    if unknown:
        raise ValueError(f"Unexpected majority labels: {unknown}")

    joined = summary.merge(
        description_scores,
        on="dim_id",
        how="left",
        validate="one_to_one",
        indicator="_description_join",
    )
    missing_descriptions = joined.loc[
        joined["_description_join"] != "both", "dim_id"
    ].tolist()
    joined = joined.drop(columns="_description_join")
    joined = joined.merge(
        human_difficulty,
        on="dim_id",
        how="left",
        validate="one_to_one",
        indicator="_human_join",
    )
    missing_human = joined.loc[joined["_human_join"] != "both", "dim_id"].tolist()
    joined = joined.drop(columns="_human_join")

    below_minimum = joined.loc[
        joined["n_description_generators"].fillna(0) < min_generators,
        "dim_id",
    ].tolist()
    if missing_descriptions or missing_human or below_minimum:
        raise ValueError(
            "Incomplete analysis join: "
            f"{len(missing_descriptions)} dimensions lack description scores, "
            f"{len(missing_human)} lack human difficulty, and "
            f"{len(below_minimum)} have fewer than {min_generators} generators"
        )

    eligible = (
        joined["n_description_generators"].fillna(0) >= min_generators
    ) & joined["human_difficulty_mean"].notna()
    joined = joined[eligible].copy()
    if joined.empty:
        raise ValueError("No dimensions remain after completeness filters")

    # This establishes that the raw human file and dimension summary use the
    # same QC population—not merely overlapping dimension IDs.
    joined["n_raters"] = pd.to_numeric(joined["n_raters"], errors="raise")
    rating_count_matches = (
        joined["n_raters"].astype(int)
        == joined["n_human_ratings"].astype(int)
    )
    vote_match_columns: list[str] = []
    for label in HUMAN_LABEL_ORDER:
        summary_column = f"n_{label}"
        raw_column = f"raw_n_{label}"
        joined[summary_column] = pd.to_numeric(
            joined[summary_column], errors="raise"
        )
        matches = (
            joined[summary_column].astype(int)
            == joined[raw_column].astype(int)
        )
        vote_match_columns.append(f"_votes_match_{label}")
        joined[f"_votes_match_{label}"] = matches
    votes_match = joined[vote_match_columns].all(axis=1)
    if not rating_count_matches.all() or not votes_match.all():
        bad = joined.loc[
            ~(rating_count_matches & votes_match), "dim_id"
        ].head(10)
        raise ValueError(
            "Human trial counts do not reproduce the dimension summary for: "
            + ", ".join(bad)
        )
    joined = joined.drop(
        columns=[
            *vote_match_columns,
            *[f"raw_n_{label}" for label in HUMAN_LABEL_ORDER],
        ]
    )

    joined = joined.sort_values(["model", "dim_id"]).reset_index(drop=True)
    diagnostics = {
        "summary_dimensions": int(len(summary)),
        "analysis_dimensions": int(len(joined)),
        "analysis_source_models": int(joined["model"].nunique()),
        "missing_description_dimensions": int(len(missing_descriptions)),
        "missing_human_dimensions": int(len(missing_human)),
        "dimensions_below_minimum_generators": int(len(below_minimum)),
        "minimum_generators_required": int(min_generators),
        "human_rating_rows": int(joined["n_human_ratings"].sum()),
        "human_counts_exactly_reproduced": True,
        "human_label_counts": {
            label: int((joined["human_label"] == label).sum())
            for label in HUMAN_LABEL_ORDER
        },
    }
    return joined, diagnostics


def _spearman(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    """Spearman correlation without emitting constant-input warnings."""
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _seed_for(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    cluster_column: str,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, int]:
    """Percentile CI for Spearman rho by resampling whole source models."""
    data = frame[[x_column, y_column, cluster_column]].dropna().reset_index(
        drop=True
    )
    clusters = data[cluster_column].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return math.nan, math.nan, 0
    cluster_indices = {
        cluster: data.index[data[cluster_column] == cluster].to_numpy()
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    estimates.fill(np.nan)
    for bootstrap_index in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = np.concatenate(
            [cluster_indices[cluster] for cluster in sampled_clusters]
        )
        estimates[bootstrap_index] = _spearman(
            data.loc[sampled_indices, x_column],
            data.loc[sampled_indices, y_column],
        )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return math.nan, math.nan, 0
    tail = (1 - ci_level) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return float(low), float(high), int(estimates.size)


def compute_correlations(
    dimension_data: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Compute overall/category correlations and clustered intervals."""
    group_display = {
        "overall": "All dimensions",
        "semantic": "Human: Semantic",
        "visual": "Human: Visual",
        "both": "Human: Both",
        "neither": "Human: Neither",
    }
    metric_display = {
        "description_coherence_percentile": "Description coherence",
        "no_clear_pattern_percentile": "No clear pattern",
    }
    outcome_display = {
        "universality": "Universality",
        "human_difficulty_mean": "Human interpretability difficulty",
    }
    rows: list[dict[str, Any]] = []
    for group in ["overall", *HUMAN_LABEL_ORDER]:
        if group == "overall":
            subset = dimension_data
        else:
            subset = dimension_data[
                dimension_data["human_label"] == group
            ]
        for outcome in outcome_display:
            for metric in metric_display:
                usable = subset[
                    [metric, outcome, "model"]
                ].dropna()
                scipy_result = spearmanr(
                    usable[metric],
                    usable[outcome],
                )
                rho = float(scipy_result.statistic)
                p_value = float(scipy_result.pvalue)
                ci_low, ci_high, n_valid = cluster_bootstrap_spearman(
                    usable,
                    x_column=metric,
                    y_column=outcome,
                    cluster_column="model",
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=_seed_for(seed, group, outcome, metric),
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_display": outcome_display[outcome],
                        "human_group": group,
                        "human_group_display": group_display[group],
                        "description_metric": metric,
                        "description_metric_display": metric_display[metric],
                        "n_dimensions": int(len(usable)),
                        "n_source_models": int(usable["model"].nunique()),
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "ci_level": ci_level,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_unit": "source_model",
                        "n_bootstrap_requested": int(n_bootstrap),
                        "n_bootstrap_valid": n_valid,
                    }
                )
    return pd.DataFrame(rows)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.23,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_FONT,
        fontweight="bold",
        va="top",
    )


def _quiet_correlation_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.grid(axis="x", color=GRAY_PALE, linewidth=0.45)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.axvline(0, color=GRAY_DARK, linewidth=0.6, zorder=1)


def plot_correlations(
    correlations: pd.DataFrame,
    dimension_data: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Create a standalone, plain-language two-panel relationship figure."""
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH, 2.72))
    fig.subplots_adjust(
        left=0.17,
        right=0.985,
        bottom=0.23,
        top=0.88,
        wspace=0.58,
    )

    group_order = correlations["human_group"].drop_duplicates().tolist()
    outcomes = correlations["outcome"].drop_duplicates().tolist()
    y_positions = np.arange(len(group_order), dtype=float)
    y_labels = []
    for group in group_order:
        n_dimensions = (
            len(dimension_data)
            if group == "overall"
            else int((dimension_data["human_label"] == group).sum())
        )
        group_name = (
            "All dimensions"
            if group == "overall"
            else f"Humans chose {group.capitalize()}"
        )
        y_labels.append(f"{group_name}\n(n={n_dimensions:,})")

    for axis, outcome, panel in zip(
        axes, outcomes, ["a", "b"], strict=True
    ):
        subset = (
            correlations[
                (correlations["outcome"] == outcome)
                & (
                    correlations["description_metric"]
                    == "description_coherence_percentile"
                )
            ]
            .set_index("human_group")
            .loc[group_order]
        )
        x = subset["spearman_rho"].to_numpy()
        low = subset["ci_low"].to_numpy()
        high = subset["ci_high"].to_numpy()

        # The original human scale runs from easy (1) to difficult (7).
        # Reverse its sign here so both panels read naturally: farther right
        # always means a stronger positive match with the quantity named.
        if outcome == "human_difficulty_mean":
            x, low, high = -x, -high, -low

        axis.errorbar(
            x,
            y_positions,
            xerr=np.vstack([x - low, high - x]),
            fmt="o",
            markersize=3.6,
            markerfacecolor=INDIGO,
            markeredgecolor=INDIGO,
            markeredgewidth=0.7,
            color=INDIGO,
            ecolor=INDIGO,
            elinewidth=0.8,
            capsize=1.8,
            capthick=0.8,
            linestyle="none",
            zorder=3,
        )
        for row_index, (value, upper) in enumerate(
            zip(x, high, strict=True)
        ):
            axis.text(
                upper + 0.018,
                y_positions[row_index],
                f"{value:.2f}",
                va="center",
                ha="left",
                color=GRAY_DARK,
                fontsize=4.5,
            )

        axis.set_yticks(y_positions, labels=y_labels)
        axis.invert_yaxis()
        if outcome == "universality":
            axis.set_xlim(-0.05, 0.82)
            axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
            axis.set_xlabel(
                "Strength of this relationship\n"
                "(0 = none; 1 = perfect)"
            )
            axis.set_title(
                "Clearer patterns recur more consistently\n"
                "across vision models",
                pad=9,
            )
        else:
            axis.set_xlim(-0.12, 0.62)
            axis.set_xticks([-0.1, 0.0, 0.2, 0.4, 0.6])
            axis.set_xlabel(
                "Strength of this relationship\n"
                "(0 = none; 1 = perfect)"
            )
            axis.set_title(
                "Clearer patterns match human ease ratings",
                pad=9,
            )
        _quiet_correlation_axis(axis)
        _panel_label(axis, panel)

    fig.text(
        0.17,
        0.025,
        (
            "Here, “clearer” means that one shared property consistently "
            "explains the 64 images. “Recurs” means that a similar dimension "
            "appears in other vision models. Human ease reverses the original "
            "1–7 difficulty scale. Horizontal lines show 95% uncertainty "
            "intervals."
        ),
        ha="left",
        va="bottom",
        fontsize=4.35,
        color=GRAY_DARK,
        wrap=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"description_clarity.{suffix}"
        fig.savefig(
            path,
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
        )
        paths.append(str(path))
    plt.close(fig)
    return paths


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_analysis(out: Path, seed: int) -> dict[str, Any]:
    analysis_dir = out / "additional_analyses"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    template_path = Path("experiments/human_ratings/template.html")
    human_trials = Path(
        "experiments/human_ratings/outputs/parsed_real_trials_qc_attn.csv"
    )
    human_template_validation = validate_human_template(template_path)

    description_long, description_scores, description_diagnostics = (
        load_description_scores(out / "descriptions.jsonl")
    )
    human_difficulty = load_human_difficulty(human_trials.resolve())
    dimension_data, join_diagnostics = assemble_dimension_data(
        out / "dimension_summary.csv",
        description_scores,
        human_difficulty,
    )
    correlations = compute_correlations(dimension_data, seed=seed)
    figure_paths = plot_correlations(
        correlations, dimension_data, analysis_dir
    )

    generator_path = analysis_dir / "description_clarity_generator_scores.csv"
    dimension_path = analysis_dir / "description_clarity_dimensions.csv"
    correlation_path = analysis_dir / "description_clarity_correlations.csv"
    summary_path = analysis_dir / "description_clarity_summary.json"

    generator_columns = [
        "dim_id",
        "description_generator",
        "resolved_model",
        "prompt_version",
        "image_mode",
        "description_coherence_raw",
        "no_clear_pattern_raw",
        "description_coherence_percentile",
        "no_clear_pattern_percentile",
    ]
    description_long[generator_columns].sort_values(
        ["description_generator", "dim_id"]
    ).to_csv(generator_path, index=False)
    dimension_data.to_csv(dimension_path, index=False)
    correlations.to_csv(correlation_path, index=False)

    overall = correlations[correlations["human_group"] == "overall"]
    key_results: dict[str, dict[str, dict[str, float | int]]] = {}
    for _, row in overall.iterrows():
        key_results.setdefault(row["outcome"], {})[row["description_metric"]] = {
            "n_dimensions": int(row["n_dimensions"]),
            "spearman_rho": float(row["spearman_rho"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "p_value": float(row["p_value"]),
        }

    ci_level = float(correlations["ci_level"].iloc[0])
    n_bootstrap = int(correlations["n_bootstrap_requested"].iloc[0])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "stage_1_description_clarity",
        "sources": {
            "descriptions_jsonl": str(
                (out / "descriptions.jsonl").resolve()
            ),
            "dimension_summary_csv": str(
                (out / "dimension_summary.csv").resolve()
            ),
            "human_trials_csv": str(human_trials.resolve()),
            "human_task_template": str(template_path.resolve()),
        },
        "measurement": {
            "description_stage": "stage-1 multimodal describers",
            "score_calibration": (
                "Empirical midrank percentile within requested description "
                "generator, followed by the mean across available generators "
                "for each dimension"
            ),
            "description_metrics": {
                "description_coherence_percentile": (
                    "Higher means the description generator reported a more "
                    "coherent shared pattern relative to its other dimensions"
                ),
                "no_clear_pattern_percentile": (
                    "Higher means the description generator assigned greater "
                    "probability to there being no clear shared pattern, "
                    "relative to its other dimensions"
                ),
            },
            "human_question": human_template_validation["question"],
            "human_scale_anchors": human_template_validation["anchors"],
            "human_aggregation": (
                "Mean of QC-passing human ratings for each dimension; "
                "correlations are Spearman"
            ),
            "uncertainty": (
                f"{ci_level:.0%} percentile interval from "
                f"{n_bootstrap:,} source-model cluster bootstrap samples"
            ),
            "interpretation_boundary": (
                "These are associations. Coherence and no-clear-pattern are "
                "self-reports from the description generators, not independent "
                "human validation that a proposed description is correct."
            ),
        },
        "description_diagnostics": description_diagnostics,
        "join_diagnostics": join_diagnostics,
        "human_template_validation": human_template_validation,
        "key_results_overall": key_results,
        "outputs": {
            "generator_scores_csv": str(generator_path),
            "dimension_data_csv": str(dimension_path),
            "correlations_csv": str(correlation_path),
            "figures": figure_paths,
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    summary = run_analysis(out, seed=cfg.seed)
    print(
        json.dumps(
            {
                "analysis": summary["analysis"],
                "n_dimensions": summary["join_diagnostics"][
                    "analysis_dimensions"
                ],
                "n_source_models": summary["join_diagnostics"][
                    "analysis_source_models"
                ],
                "key_results_overall": summary["key_results_overall"],
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
