#!/usr/bin/env python3
"""Summarize the IPCL dataset-domain control: tables, statistics, report, figure."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from experiments.universality.shared import load_universality
from src.colors import GRAY, INDIGO, ROSE, SAND, TEAL
from src.plotting import paper_style

N_RESAMPLES = 10000


def violin_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    palette: dict[str, str],
) -> pd.DataFrame:
    """Box + strip of per-dimension universality; returns per-model means."""
    sub = df[df[group_col].isin(group_order)]

    sns.boxplot(
        data=sub, x=group_col, y="universality", order=group_order,
        hue=group_col, palette=palette, legend=False,
        width=0.45, linewidth=0.6, fliersize=0, ax=ax,
    )
    for patch in ax.patches:
        patch.set_alpha(0.3)

    sns.stripplot(
        data=sub, x=group_col, y="universality", order=group_order,
        hue=group_col, palette=palette, legend=False,
        size=1.2, alpha=0.25, jitter=0.10, ax=ax, zorder=5,
    )

    model_means = df.groupby([group_col, "model"])["universality"].mean().reset_index()

    ax.set_xlabel("")
    ax.set_ylabel("")
    return model_means


def section_mean(
    ax: plt.Axes,
    model_means: pd.DataFrame,
    group_col: str,
    groups: list[str],
    x_positions: tuple[float, float],
) -> None:
    """Draw section mean +/- SD as dashed line and shaded band."""
    sec = model_means[model_means[group_col].isin(groups)]
    per_model = sec.groupby("model")["universality"].mean()
    mu = per_model.mean()
    sd = per_model.std()

    x0, x1 = x_positions
    ax.hlines(mu, x0, x1, colors=GRAY, ls=":", lw=0.8, zorder=2)
    ax.fill_between(
        [x0, x1], mu - sd, mu + sd,
        color=GRAY, alpha=0.10, zorder=1,
    )


def random_subset_sd_test(
    target_model_means: np.ndarray,
    reference_model_means: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Compare four-model SD with random four-model subsets of the reference.

    Both tails are saved.  The lower tail is the paper's existing
    within-group-convergence test; the upper tail asks whether the four domains
    are unusually dispersed.
    """
    target = np.asarray(target_model_means, dtype=float)
    reference = np.asarray(reference_model_means, dtype=float)
    if len(target) < 2:
        raise ValueError("at least two target models are required")
    if len(reference) < len(target):
        raise ValueError("reference pool is too small")
    observed_sd = float(target.std(ddof=1))
    rng = np.random.default_rng(seed)
    null_sds = np.empty(n_resamples)
    for index in range(n_resamples):
        null_sds[index] = rng.choice(reference, size=len(target), replace=False).std(ddof=1)
    lower = float((np.count_nonzero(null_sds <= observed_sd) + 1) / (n_resamples + 1))
    upper = float((np.count_nonzero(null_sds >= observed_sd) + 1) / (n_resamples + 1))
    two_sided = float(min(1.0, 2.0 * min(lower, upper)))
    result = {
        "name": "random-subset test on within-group SD",
        "unit_of_analysis": "model",
        "target_group": "four AlexNet-GN IPCL checkpoints (one per training dataset)",
        "null_pool": "162 paper reference models",
        "n_target_models": int(len(target)),
        "n_reference_models": int(len(reference)),
        "observed_mean": float(target.mean()),
        "observed_sd": observed_sd,
        "observed_range": float(target.max() - target.min()),
        "null_sd_mean": float(null_sds.mean()),
        "null_sd_median": float(np.median(null_sds)),
        "null_sd_5pct": float(np.percentile(null_sds, 5)),
        "null_sd_95pct": float(np.percentile(null_sds, 95)),
        "observed_sd_percentile": float(np.mean(null_sds <= observed_sd) * 100),
        "p_one_sided_convergence_lower_tail": lower,
        "p_one_sided_divergence_upper_tail": upper,
        "p_two_sided_dispersion": two_sided,
        "n_resamples": n_resamples,
        "seed": seed,
        "sampling": "without replacement",
        "note": (
            "One checkpoint per domain: the test describes model-level spread and does "
            "not isolate a causal dataset-domain effect from dataset size/composition. "
            "The heterogeneous 162-model null varies architecture, objective, and data; "
            "it follows the paper's random-group spread test and is not a "
            "domain-matched causal null."
        ),
    }
    return result, null_sds


def build_per_model(
    per_dimension: pd.DataFrame,
    reference_means: pd.Series,
    out: Path,
    rank: int,
) -> pd.DataFrame:
    summary = (
        per_dimension.groupby(["model", "training_dataset"], sort=False)["universality"]
        .agg(
            mean_universality="mean",
            dimension_sd="std",
            median_universality="median",
            min_universality="min",
            max_universality="max",
            n_dimensions="size",
        )
        .reset_index()
    )
    summary["reference_percentile"] = [
        float(np.mean(reference_means.to_numpy() < value) * 100)
        for value in summary["mean_universality"]
    ]
    selection = pd.read_csv(out / "meta" / f"bandwidth_selection_rank_{rank}.csv")
    return summary.merge(selection, on=["model", "training_dataset"], how="left")


def plot_control(
    per_dimension: pd.DataFrame,
    order: list[str],
    figure_dir: Path,
) -> None:
    """Match the box-and-strip style of the paper's controlled comparisons."""
    paper_style()
    palette = dict(zip(order, [ROSE, TEAL, INDIGO, SAND]))
    plot_data = per_dimension.copy()
    plot_data["group"] = plot_data["training_dataset"]

    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    model_means = violin_panel(ax, plot_data, "group", order, palette)
    section_mean(ax, model_means, "group", order, (-0.4, len(order) - 0.6))

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylim(-0.02, 0.52)
    ax.set_yticks(np.arange(0.0, 0.51, 0.1))
    ax.set_ylabel("Universality", labelpad=2)
    ax.set_xlabel("")
    ax.set_title("AlexNet-GN IPCL", color="black", fontweight="medium", pad=8)

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "ipcl_dataset_domain.pdf", facecolor="white")
    fig.savefig(figure_dir / "ipcl_dataset_domain.png", facecolor="white", dpi=300)
    plt.close(fig)


def write_report(per_model: pd.DataFrame, stats: dict[str, object], out: Path) -> None:
    test = stats["random_subset_sd_test"]
    lower = test["p_one_sided_convergence_lower_tail"]
    upper = test["p_one_sided_divergence_upper_tail"]
    two_sided = test["p_two_sided_dispersion"]
    if two_sided < 0.05 and upper < lower:
        headline = "The four dataset-domain checkpoints are more dispersed than random four-model reference subsets."
    elif two_sided < 0.05 and lower < upper:
        headline = "The four dataset-domain checkpoints are more convergent than random four-model reference subsets."
    else:
        headline = "The four dataset-domain checkpoints have typical model-level spread relative to random four-model reference subsets."

    table = per_model[
        [
            "training_dataset",
            "mean_universality",
            "median_universality",
            "dimension_sd",
            "reference_percentile",
            "sigma_mult",
            "best_seed",
            "stability",
            "evar",
            "selected_at_grid_boundary",
        ]
    ].copy()
    numeric = [
        "mean_universality",
        "median_universality",
        "dimension_sd",
        "reference_percentile",
        "stability",
        "evar",
    ]
    table[numeric] = table[numeric].round(3)
    headers = list(table.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join(map(str, row)) + " |")
    markdown_table = "\n".join(table_lines)

    provenance_path = out / "meta" / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    srf = provenance.get("srf_runtime", {})
    srf_version = str(srf.get("version") or "unknown")
    report = f"""# IPCL dataset-domain control

## Result

{headline}

The model-level SD across the four IPCL checkpoints is {test['observed_sd']:.4f}
(random-subset median {test['null_sd_median']:.4f}, 5th–95th percentile
{test['null_sd_5pct']:.4f}–{test['null_sd_95pct']:.4f}). The transparent
two-sided Monte Carlo p-value is p={two_sided:.4f}. For transparency, its
component tails are p={lower:.4f} for unusually small spread (convergence) and
p={upper:.4f} for unusually large spread (divergence), using
{test['n_resamples']:,} random subsets. Directional tails are descriptive; the
two-sided value governs the result headline. This is a standalone exploratory
test and is not adjusted together with the paper's pre-existing contrast
family.

The four-checkpoint mean is {test['observed_mean']:.4f}, with a model-mean
range of {test['observed_range']:.4f}. Their observed SD is at the
{test['observed_sd_percentile']:.1f}th percentile of the random-four null.
Across the fixed reference panel, the mean of model means is
{stats['reference']['grand_mean_of_model_means']:.4f} (SD
{stats['reference']['sd_of_model_means']:.4f}).

## Model summaries

{markdown_table}

## Design

- Architecture: AlexNet-GN for all four checkpoints.
- Objective: IPCL for all four checkpoints.
- Feature layer: 4,096-dimensional post-BN+ReLU fc7 backbone representation,
  exposed through an identity fc8 (not the 128-dimensional projection head).
- Training datasets: ImageNet-1K, OpenImages, Places-256, and VGGFace2.
- Evaluation images: the same 22,248 THINGS images used in the paper.
- Factorization: rank 50, model-specific median-distance RBF bandwidth, the
  full eight-multiplier grid, five seeds, and harmonic-mean selection from
  stability and explained variance.
- Factorizer runtime: bundled SRF {srf_version}; source and solver hashes
  are saved in `meta/provenance.json`.
- Universality: squared cosine with one-to-one Hungarian matching and the
  paper's row-permutation null correction (1,000 permutations by default).
- Reference: the fixed 162-model rank-50 paper panel. The four IPCL targets are
  not added as source models.
- Inferential unit: model. Per-dimension values are shown descriptively only.
- Figure: box plots and dots show the 50 dimensions from each checkpoint. The
  dotted line and gray band show the mean ± 1 SD of the four IPCL model means,
  matching the paper's controlled-comparison figure. They do not summarize the
  162-model reference panel.

## What this does and doesn't show

This is not a clean causal test of the training dataset. We have one released
checkpoint per dataset, so dataset size and composition, checkpoint identity,
and possibly some training settings change together. The analysis asks a
narrower question: do the paper's results look similar for IPCL models trained
on a wider range of visual data?

The evaluation still uses the object-focused THINGS images. The random-four
comparison also draws from a mixed collection of models rather than four
perfectly matched controls. It tells us whether the four IPCL results are
unusually spread out relative to the Figure 3 reference panel; it does not
show that training dataset is the cause.

## Outputs

- `per_dimension.csv`: 50 primary metric values per target.
- `per_model.csv`: one summary row per target checkpoint.
- `stats.json`: model-unit random-subset test and reference-panel statistics.
- `meta/`: input/reference provenance and factorization QA.
- `figures/ipcl_dataset_domain.pdf` and `.png`: paper-ready control figure.
"""
    (out / "REPORT.md").write_text(report)


def write_artifact_manifest(out: Path, rank: int) -> None:
    """Hash the shareable result package after every final artifact is written."""
    paths = [
        out / "REPORT.md",
        out / "params.json",
        out / "per_dimension.csv",
        out / "per_model.csv",
        out / "random_subset_sd_null.npz",
        out / "stats.json",
        out / "figures" / "ipcl_dataset_domain.pdf",
        out / "figures" / "ipcl_dataset_domain.png",
        out / "universality" / "per_dimension_all_metrics.csv",
        out / "meta" / "ipcl_model_metadata.csv",
        out / "meta" / f"factorization_metrics_rank_{rank}.csv",
        out / "meta" / "factorization_run_config.json",
        out / "meta" / "input_manifest.csv",
        out / "meta" / "median_sigma.csv",
        out / "meta" / "provenance.json",
        out / "meta" / "reference_manifest.csv",
        out / "meta" / "scoring_run_config.json",
        out / "meta" / f"bandwidth_selection_rank_{rank}.csv",
        out / "meta" / f"stability_by_dimension_rank_{rank}.csv",
        out / "meta" / f"stability_evar_all_rank_{rank}.csv",
    ]
    paths.extend(sorted((out / "universality" / "by_model").glob("*.csv")))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot finalize artifact manifest; missing: {missing}")
    artifacts = {}
    for path in paths:
        relative = str(path.relative_to(out))
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    (out / "run_complete.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank = cfg.factorization.rank

    targets = list(cfg.models) or [
        "ipcl_alexnet_gn_imagenet1k",
        "ipcl_alexnet_gn_openimagesv6",
        "ipcl_alexnet_gn_places2",
        "ipcl_alexnet_gn_vggface2",
    ]
    per_dimension = pd.read_csv(out / "per_dimension.csv")
    if set(per_dimension["model"]) != set(targets):
        raise ValueError(
            f"per_dimension targets {sorted(set(per_dimension['model']))} "
            f"!= planned {sorted(targets)}"
        )
    counts = per_dimension.groupby("model")["dimension"].nunique()
    if not (counts == rank).all():
        raise ValueError(f"expected {rank} dimensions per target:\n{counts}")
    if not per_dimension["universality"].between(0, 1).all():
        raise ValueError("universality values outside [0, 1]")

    training_dataset = per_dimension.drop_duplicates("model").set_index("model")["training_dataset"]
    order = [training_dataset[model] for model in targets]

    reference = load_universality(rank=rank, metric="cos2_perm_adj", dataset=cfg.dataset)
    reference_means = reference.groupby("model")["universality"].mean()
    if len(reference_means) != 162:
        raise ValueError(f"expected 162 reference model means, got {len(reference_means)}")
    per_model = build_per_model(per_dimension, reference_means, out, rank)
    per_model.to_csv(out / "per_model.csv", index=False)

    test, null_sds = random_subset_sd_test(
        per_model["mean_universality"].to_numpy(),
        reference_means.to_numpy(),
        N_RESAMPLES,
        cfg.seed,
    )
    np.savez_compressed(out / "random_subset_sd_null.npz", null_sds=null_sds)
    stats = {
        "analysis": "IPCL dataset-domain control",
        "primary_metric": "cos2_perm_adj",
        "rank": rank,
        "reference": {
            "n_models": int(len(reference_means)),
            "grand_mean_of_model_means": float(reference_means.mean()),
            "sd_of_model_means": float(reference_means.std(ddof=1)),
            "min_model_mean": float(reference_means.min()),
            "max_model_mean": float(reference_means.max()),
        },
        "target_descriptive": {
            "n_models": int(len(per_model)),
            "mean_of_model_means": float(per_model["mean_universality"].mean()),
            "sd_of_model_means": float(per_model["mean_universality"].std(ddof=1)),
            "min_model_mean": float(per_model["mean_universality"].min()),
            "max_model_mean": float(per_model["mean_universality"].max()),
        },
        "random_subset_sd_test": test,
        "dimension_level_inference": "none (dimensions are not treated as independent models)",
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    params = {
        "design": {
            "models": targets,
            "training_datasets": order,
            "architecture": "AlexNet-GN",
            "objective": "IPCL",
            "rank": rank,
            "reference_dataset": cfg.dataset,
        },
        "analysis": {"n_resamples": N_RESAMPLES, "seed": cfg.seed},
    }
    for name in ("factorization_run_config", "scoring_run_config", "provenance"):
        path = out / "meta" / f"{name}.json"
        if path.exists():
            params[name] = json.loads(path.read_text())
    (out / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")

    plot_control(per_dimension, order, out / "figures")
    write_report(per_model, stats, out)
    write_artifact_manifest(out, rank)
    print(per_model[["training_dataset", "mean_universality", "reference_percentile"]].to_string(index=False))
    print(json.dumps(test, indent=2))
    print(f"wrote tables, statistics, report, and figures under {out}")


if __name__ == "__main__":
    main()
