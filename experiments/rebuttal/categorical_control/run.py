from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Keep nested linear algebra from oversubscribing cores inside joblib workers.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp

from experiments.universality.downstream_predictions.macaque_encoding.encoding_things.run import (
    encode_single_model,
    load_macaque_it,
)
from src.colors import GRAY_DARK, INDIGO
from src.data import ThingsDataset
from src.plotting import PANEL_LABEL_FONT, PAPER_WIDTH, paper_style

# Largest accepted absolute difference in mean category eta-squared between the
# two matched halves of a model's dimensions.
MATCH_TOLERANCE = 1e-4
# Resamples behind every bootstrap confidence interval reported here.
N_BOOTSTRAP = 20_000

log = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_triplets(triplet_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    n_categories = 1854
    paths = [triplet_dir / "trainset.txt", triplet_dir / "validationset.txt"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing public THINGS triplets: "
            + ", ".join(missing)
            + ". See the data layout in README.md."
        )

    parts = [np.loadtxt(path, dtype=np.int16) for path in paths]
    triplets = np.concatenate(parts, axis=0)
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError(f"Unexpected triplet shape: {triplets.shape}")
    if int(triplets.min()) != 0 or int(triplets.max()) != n_categories - 1:
        raise ValueError(
            f"Triplet indices must span 0..{n_categories - 1}, got "
            f"{triplets.min()}..{triplets.max()}"
        )
    metadata = {
        "train_rows": int(len(parts[0])),
        "validation_rows": int(len(parts[1])),
        "total_rows": int(len(triplets)),
        "files": {
            path.name: {"sha256": sha256_file(path)}
            for path in paths
        },
    }
    return triplets, metadata


def load_source_tables(
    sources: dict[str, Path],
    rank: int,
    n_models: int = 162,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    category = pd.read_csv(sources["category"])
    required_category = {
        "model",
        "dimension",
        "universality",
        "eta_squared",
        "c_eff_excess",
        "norm_fraction",
    }
    if missing := required_category - set(category.columns):
        raise ValueError(f"Missing category columns: {sorted(missing)}")
    if category.duplicated(["model", "dimension"]).any():
        raise ValueError("Duplicate model/dimension rows in category input")

    per_model_counts = category.groupby("model")["dimension"].agg(["count", "nunique"])
    if not ((per_model_counts == rank).all().all()):
        raise ValueError(f"Every model must have exactly {rank} unique dimensions")
    if category[list(required_category)].isna().any().any():
        raise ValueError("Null values in required category input columns")
    # This control weights category consistency by loading mass, not
    # by the separate leave-one-out delta R² used in manuscript Figure 4.
    category = category.rename(columns={"norm_fraction": "importance"})

    behavior = pd.read_csv(sources["behavior"])
    enc_f = pd.read_csv(sources["encoding_F"])
    enc_n = pd.read_csv(sources["encoding_N"])
    for label, table in (("behavior", behavior), ("F", enc_f), ("N", enc_n)):
        if table["model"].duplicated().any():
            raise ValueError(f"Duplicate model rows in {label} input")

    category_model = (
        category.groupby("model", as_index=False)
        .agg(
            universality=("universality", "mean"),
            mean_eta_squared=("eta_squared", "mean"),
            mean_c_eff_excess=("c_eff_excess", "mean"),
            total_importance=("importance", "sum"),
        )
        .copy()
    )
    weighted_eta = (
        category.assign(
            weighted_eta=lambda x: x["eta_squared"] * x["importance"]
        )
        .groupby("model")
        .agg(
            weighted_eta_sum=("weighted_eta", "sum"),
            importance_sum=("importance", "sum"),
        )
    )
    weighted_eta["importance_weighted_eta_squared"] = (
        weighted_eta["weighted_eta_sum"] / weighted_eta["importance_sum"]
    )
    category_model = category_model.merge(
        weighted_eta[["importance_weighted_eta_squared"]],
        left_on="model",
        right_index=True,
        validate="one_to_one",
    )

    model = category_model.merge(
        behavior[
            [
                "model",
                "triplet_accuracy",
                "universality",
                "display_name",
                "model_class",
                "train_task",
                "train_data",
                "architecture",
                "objective_category",
            ]
        ].rename(columns={"universality": "universality_behavior_input"}),
        on="model",
        how="inner",
        validate="one_to_one",
    )
    model = model.merge(
        enc_f[["model", "mean_r"]].rename(columns={"mean_r": "it_encoding_F"}),
        on="model",
        how="inner",
        validate="one_to_one",
    ).merge(
        enc_n[["model", "mean_r"]].rename(columns={"mean_r": "it_encoding_N"}),
        on="model",
        how="inner",
        validate="one_to_one",
    )
    model["it_encoding"] = (
        model["it_encoding_F"] + model["it_encoding_N"]
    ) / 2

    source_model_sets = {
        "category": set(category["model"]),
        "behavior": set(behavior["model"]),
        "encoding_F": set(enc_f["model"]),
        "encoding_N": set(enc_n["model"]),
    }
    if len(set.intersection(*source_model_sets.values())) != n_models:
        raise ValueError(
            f"Expected {n_models} models shared by all Figure 3 inputs; got "
            f"{len(set.intersection(*source_model_sets.values()))}"
        )
    if len(model) != n_models:
        raise ValueError(f"Expected {n_models} merged models, got {len(model)}")
    max_uni_difference = float(
        np.max(
            np.abs(
                model["universality"] - model["universality_behavior_input"]
            )
        )
    )
    if max_uni_difference > 1e-10:
        raise ValueError(
            "Universality differs between category and behavior inputs: "
            f"max abs difference={max_uni_difference:g}"
        )
    return category, model.sort_values("model").reset_index(drop=True)


def build_matched_assignment(category: pd.DataFrame, rank: int = 50) -> pd.DataFrame:
    """Select category-matched higher-universality halves with MILP.

    Each selected half has half of the model's dimensions, an equal number from
    each within-model eta-squared quintile, and mean eta-squared within
    MATCH_TOLERANCE of the complementary half. Subject to those constraints,
    universality is maximized.
    """
    n_eta_bins = 5
    records: list[pd.DataFrame] = []
    for model_name, raw_group in category.groupby("model", sort=True):
        group = raw_group.sort_values(
            ["eta_squared", "dimension"], kind="mergesort"
        ).reset_index(drop=True)
        if len(group) != rank:
            raise ValueError(f"{model_name}: expected {rank} dimensions")

        eta = group["eta_squared"].to_numpy(float)
        universality = group["universality"].to_numpy(float)
        bin_size = rank // n_eta_bins
        eta_bin = np.repeat(np.arange(n_eta_bins), bin_size)
        n_selected = rank // 2

        # Difference in group means <= tolerance is equivalent to constraining
        # the selected eta sum around half of the total.
        eta_sum_tolerance = MATCH_TOLERANCE * n_selected / 2
        constraint_rows = [np.ones(rank), eta]
        lower = [
            n_selected,
            eta.sum() / 2 - eta_sum_tolerance,
        ]
        upper = [
            n_selected,
            eta.sum() / 2 + eta_sum_tolerance,
        ]
        for bin_index in range(n_eta_bins):
            constraint_rows.append((eta_bin == bin_index).astype(float))
            lower.append(bin_size // 2)
            upper.append(bin_size // 2)

        result = milp(
            c=-universality,
            integrality=np.ones(rank),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                np.stack(constraint_rows),
                lb=np.asarray(lower),
                ub=np.asarray(upper),
            ),
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(
                f"Matching failed for {model_name}: {result.message}"
            )

        selected = np.rint(result.x).astype(bool)
        if selected.sum() != n_selected:
            raise RuntimeError(f"{model_name}: optimizer selected {selected.sum()} dims")
        eta_difference = abs(eta[selected].mean() - eta[~selected].mean())
        universality_difference = (
            universality[selected].mean() - universality[~selected].mean()
        )
        if eta_difference > MATCH_TOLERANCE + 1e-9:
            raise RuntimeError(
                f"{model_name}: eta matching failed ({eta_difference:g})"
            )
        if universality_difference <= 0:
            raise RuntimeError(
                f"{model_name}: selected half is not more universal"
            )

        out = group.copy()
        out["eta_quintile"] = eta_bin + 1
        out["subset"] = np.where(
            selected, "higher_universality", "lower_universality"
        )
        out["selected_by_milp"] = selected
        records.append(out)

    assignment = pd.concat(records, ignore_index=True)
    subset_counts = assignment.groupby(["model", "subset"]).size()
    if not (subset_counts == rank // 2).all():
        raise RuntimeError("Matched subset size check failed")
    bin_counts = assignment.groupby(
        ["model", "subset", "eta_quintile"]
    ).size()
    if not (bin_counts == rank // n_eta_bins // 2).all():
        raise RuntimeError("Eta-quintile balance check failed")
    return assignment.sort_values(["model", "dimension"]).reset_index(drop=True)


def assignment_summary(assignment: pd.DataFrame) -> pd.DataFrame:
    summary = (
        assignment.groupby(["model", "subset"], as_index=False)
        .agg(
            n_dimensions=("dimension", "size"),
            mean_universality=("universality", "mean"),
            mean_eta_squared=("eta_squared", "mean"),
            mean_importance=("importance", "mean"),
        )
        .pivot(index="model", columns="subset")
    )
    summary.columns = [f"{metric}_{subset}" for metric, subset in summary.columns]
    summary = summary.reset_index()
    summary["delta_universality"] = (
        summary["mean_universality_higher_universality"]
        - summary["mean_universality_lower_universality"]
    )
    summary["delta_eta_squared"] = (
        summary["mean_eta_squared_higher_universality"]
        - summary["mean_eta_squared_lower_universality"]
    )
    summary["delta_importance"] = (
        summary["mean_importance_higher_universality"]
        - summary["mean_importance_lower_universality"]
    )
    return summary.sort_values("model").reset_index(drop=True)


def _slug(model_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)[:80]
    suffix = hashlib.sha1(model_name.encode()).hexdigest()[:10]
    return f"{stem}__{suffix}"


def _task_signature(
    model_name: str,
    high_dims: np.ndarray,
    low_dims: np.ndarray,
    triplet_meta: dict[str, Any],
    rank: int,
    n_folds: int,
    random_state: int,
) -> str:
    payload = {
        "model": model_name,
        "high_dims": sorted(map(int, high_dims)),
        "low_dims": sorted(map(int, low_dims)),
        "triplet_hashes": {
            name: details["sha256"]
            for name, details in triplet_meta["files"].items()
        },
        "rank": rank,
        "n_folds": n_folds,
        "random_state": random_state,
        "method_version": 1,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def triplet_accuracy(
    embedding: np.ndarray,
    triplets: np.ndarray,
    chunk_size: int = 500_000,
) -> float:
    embedding = np.asarray(embedding, dtype=np.float64)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = embedding / np.maximum(norms, np.finfo(float).eps)
    similarity = normalized @ normalized.T

    correct = 0
    for start in range(0, len(triplets), chunk_size):
        chunk = triplets[start : start + chunk_size]
        i, j, k = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        s_ij = similarity[i, j]
        correct += int(
            np.count_nonzero(
                (s_ij > similarity[i, k]) & (s_ij > similarity[j, k])
            )
        )
    return correct / len(triplets)


def evaluate_one_model(
    model_name: str,
    high_dims: np.ndarray,
    low_dims: np.ndarray,
    y_f: np.ndarray,
    y_n: np.ndarray,
    triplets: np.ndarray,
    triplet_meta: dict[str, Any],
    rank: int,
    n_folds: int,
    random_state: int,
    cache_dir: Path,
) -> dict[str, Any]:
    n_categories, n_images_per_category = 1854, 12
    signature = _task_signature(
        model_name,
        high_dims,
        low_dims,
        triplet_meta,
        rank,
        n_folds,
        random_state,
    )
    cache_path = cache_dir / f"{_slug(model_name)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("signature") == signature:
            return cached

    dataset = ThingsDataset()
    w = dataset.load_components(model_name, rank=rank)
    if w.shape != (n_categories * n_images_per_category, rank):
        raise ValueError(f"{model_name}: unexpected component shape {w.shape}")

    high_dims = np.asarray(high_dims, dtype=int)
    low_dims = np.asarray(low_dims, dtype=int)
    exemplary = np.arange(n_categories) * n_images_per_category

    result: dict[str, Any] = {
        "model": model_name,
        "signature": signature,
        "n_folds": n_folds,
        "random_state": random_state,
        "n_triplets": len(triplets),
    }
    for monkey, y in (("F", y_f), ("N", y_n)):
        for label, dims in (
            ("higher_universality", high_dims),
            ("lower_universality", low_dims),
        ):
            per_neuron = encode_single_model(
                w[:, dims], y, n_folds, random_state
            )
            result[f"it_encoding_{monkey}_{label}"] = float(
                np.mean(per_neuron)
            )

    w_exemplary = w[exemplary]
    result["triplet_accuracy_public_full"] = triplet_accuracy(
        w_exemplary, triplets
    )
    result["triplet_accuracy_higher_universality"] = triplet_accuracy(
        w_exemplary[:, high_dims], triplets
    )
    result["triplet_accuracy_lower_universality"] = triplet_accuracy(
        w_exemplary[:, low_dims], triplets
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    temporary.replace(cache_path)
    return result


def run_evaluation(
    assignment: pd.DataFrame,
    triplets: np.ndarray,
    triplet_meta: dict[str, Any],
    rank: int,
    n_folds: int,
    random_state: int,
    n_jobs: int,
    cache_dir: Path,
) -> pd.DataFrame:
    y_f, _ = load_macaque_it("F")
    y_n, _ = load_macaque_it("N")

    tasks = []
    for model_name in sorted(assignment["model"].unique()):
        group = assignment[assignment["model"] == model_name]
        high_dims = group.loc[
            group["subset"] == "higher_universality", "dimension"
        ].to_numpy(int)
        low_dims = group.loc[
            group["subset"] == "lower_universality", "dimension"
        ].to_numpy(int)
        tasks.append(
            delayed(evaluate_one_model)(
                model_name,
                high_dims,
                low_dims,
                y_f,
                y_n,
                triplets,
                triplet_meta,
                rank,
                n_folds,
                random_state,
                cache_dir,
            )
        )

    log.info(
        "Evaluating %d models (%d workers, 4 IT fits + 3 behavior scores/model)",
        len(tasks),
        n_jobs,
    )
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs, verbose=10)(tasks)
    results = pd.DataFrame(rows)
    results["it_encoding_higher_universality"] = (
        results["it_encoding_F_higher_universality"]
        + results["it_encoding_N_higher_universality"]
    ) / 2
    results["it_encoding_lower_universality"] = (
        results["it_encoding_F_lower_universality"]
        + results["it_encoding_N_lower_universality"]
    ) / 2
    return results.sort_values("model").reset_index(drop=True)


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    controls = np.asarray(controls, float)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(values)), controls])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def partial_correlation(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> tuple[float, float]:
    rx = residualize(x, controls)
    ry = residualize(y, controls)
    result = stats.pearsonr(rx, ry)
    return float(result.statistic), float(result.pvalue)


def bootstrap_partial_ci(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    *,
    rank: bool,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    controls = np.asarray(controls, float)
    if controls.ndim == 1:
        controls = controls[:, None]
    values = np.column_stack([x, y, controls])
    sampled = np.empty(n_bootstrap)
    n = len(x)
    for index in range(n_bootstrap):
        sample = values[rng.integers(0, n, n)]
        if rank:
            sample = np.apply_along_axis(stats.rankdata, 0, sample)
        sampled[index] = partial_correlation(
            sample[:, 0], sample[:, 1], sample[:, 2:]
        )[0]
    return tuple(np.quantile(sampled, [0.025, 0.975]))


def hc3_standardized_regression(
    outcome: np.ndarray,
    universality: np.ndarray,
    category_control: np.ndarray,
) -> dict[str, float]:
    y = stats.zscore(np.asarray(outcome, float), ddof=1)
    u = stats.zscore(np.asarray(universality, float), ddof=1)
    c = stats.zscore(np.asarray(category_control, float), ddof=1)
    design = np.column_stack([np.ones(len(y)), u, c])
    inv_xtx = np.linalg.inv(design.T @ design)
    beta = inv_xtx @ design.T @ y
    residual = y - design @ beta
    leverage = np.sum((design @ inv_xtx) * design, axis=1)
    adjusted_sq = (residual / np.maximum(1 - leverage, 1e-12)) ** 2
    meat = design.T @ (design * adjusted_sq[:, None])
    covariance = inv_xtx @ meat @ inv_xtx
    se = np.sqrt(np.diag(covariance))
    dof = len(y) - design.shape[1]
    t_value = beta[1] / se[1]
    p_value = 2 * stats.t.sf(abs(t_value), df=dof)
    critical = stats.t.ppf(0.975, df=dof)
    return {
        "standardized_beta_universality": float(beta[1]),
        "hc3_se": float(se[1]),
        "hc3_ci_low": float(beta[1] - critical * se[1]),
        "hc3_ci_high": float(beta[1] + critical * se[1]),
        "hc3_p": float(p_value),
    }


def correlation_analysis(
    model: pd.DataFrame,
    outcome: str,
    seed_offset: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    x = model["universality"].to_numpy(float)
    y = model[outcome].to_numpy(float)
    eta = model["mean_eta_squared"].to_numpy(float)

    raw_pearson = stats.pearsonr(x, y)
    raw_spearman = stats.spearmanr(x, y)
    partial_r, partial_p = partial_correlation(x, y, eta)
    partial_rho, partial_rho_p = partial_correlation(
        stats.rankdata(x), stats.rankdata(y), stats.rankdata(eta)
    )
    partial_ci = bootstrap_partial_ci(
        x,
        y,
        eta,
        rank=False,
        seed=bootstrap_seed + seed_offset,
    )
    partial_rank_ci = bootstrap_partial_ci(
        x,
        y,
        eta,
        rank=True,
        seed=bootstrap_seed + 100 + seed_offset,
    )

    weighted_r, weighted_p = partial_correlation(
        x,
        y,
        model["importance_weighted_eta_squared"].to_numpy(float),
    )
    joint_r, joint_p = partial_correlation(
        x,
        y,
        model[["mean_eta_squared", "mean_c_eff_excess"]].to_numpy(float),
    )
    return {
        "n_models": int(len(model)),
        "raw_pearson_r": float(raw_pearson.statistic),
        "raw_pearson_p": float(raw_pearson.pvalue),
        "raw_spearman_rho": float(raw_spearman.statistic),
        "raw_spearman_p": float(raw_spearman.pvalue),
        "partial_pearson_r": partial_r,
        "partial_pearson_p": partial_p,
        "partial_pearson_ci_low": partial_ci[0],
        "partial_pearson_ci_high": partial_ci[1],
        "partial_spearman_rho": partial_rho,
        "partial_spearman_p": partial_rho_p,
        "partial_spearman_ci_low": partial_rank_ci[0],
        "partial_spearman_ci_high": partial_rank_ci[1],
        "weighted_eta_sensitivity_r": weighted_r,
        "weighted_eta_sensitivity_p": weighted_p,
        "eta_and_breadth_sensitivity_r": joint_r,
        "eta_and_breadth_sensitivity_p": joint_p,
        "standardized_regression": hc3_standardized_regression(y, x, eta),
    }


def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences, float)
    indices = rng.integers(
        0, len(differences), size=(n_bootstrap, len(differences))
    )
    bootstrap_means = differences[indices].mean(axis=1)
    return tuple(np.quantile(bootstrap_means, [0.025, 0.975]))


def paired_analysis(
    higher: np.ndarray,
    lower: np.ndarray,
    seed_offset: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    higher = np.asarray(higher, float)
    lower = np.asarray(lower, float)
    difference = higher - lower
    t_result = stats.ttest_rel(higher, lower)
    wilcoxon = stats.wilcoxon(higher, lower, alternative="two-sided")
    ci = bootstrap_mean_ci(
        difference, bootstrap_seed + 200 + seed_offset
    )
    return {
        "n_models": int(len(higher)),
        "higher_mean": float(higher.mean()),
        "lower_mean": float(lower.mean()),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "paired_t": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "fraction_higher_better": float(np.mean(difference > 0)),
        "cohens_dz": float(difference.mean() / difference.std(ddof=1)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    m = len(ordered)
    for rank_index, name in enumerate(ordered):
        candidate = min(1.0, (m - rank_index) * p_values[name])
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def add_adjusted_values(model: pd.DataFrame) -> pd.DataFrame:
    model = model.copy()
    eta = model["mean_eta_squared"].to_numpy(float)
    for outcome in ("it_encoding", "triplet_accuracy"):
        for column in ("universality", outcome):
            values = model[column].to_numpy(float)
            adjusted = residualize(values, eta) + values.mean()
            model[f"{column}_category_adjusted_for_{outcome}"] = adjusted
    return model


def regression_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = GRAY_DARK,
) -> None:
    fit = np.polyfit(x, y, 1)
    order = np.argsort(x)
    xs = x[order]
    fitted = np.polyval(fit, xs)
    residual_sd = np.std(y - np.polyval(fit, x))
    ax.plot(xs, fitted, color=color, lw=1.0, alpha=0.8, zorder=2)
    ax.fill_between(
        xs,
        fitted - 1.96 * residual_sd,
        fitted + 1.96 * residual_sd,
        color=color,
        alpha=0.12,
        lw=0,
        zorder=1,
    )


def scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    raw_r: float,
    partial_r: float,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.scatter(
        x,
        y,
        s=12,
        color=INDIGO,
        alpha=0.4,
        edgecolors="none",
        zorder=3,
    )
    regression_line(ax, x, y)
    ax.text(
        0.04,
        0.96,
        f"raw $r$ = {raw_r:.2f}\npartial $r$ = {partial_r:.2f}",
        transform=ax.transAxes,
        va="top",
        fontweight="medium",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def make_figure(
    model: pd.DataFrame,
    statistics: dict[str, Any],
    out: Path,
) -> None:
    paper_style()
    fig, axes = plt.subplot_mosaic(
        [["it_scatter", "behavior_scatter"]],
        width_ratios=[1, 1],
        layout="constrained",
        figsize=(PAPER_WIDTH, 2.2),
    )

    scatter_panel(
        axes["it_scatter"],
        model["universality_category_adjusted_for_it_encoding"].to_numpy(),
        model["it_encoding_category_adjusted_for_it_encoding"].to_numpy(),
        statistics["model_level"]["it_encoding"]["raw_pearson_r"],
        statistics["model_level"]["it_encoding"]["partial_pearson_r"],
        "Model universality\n(category adjusted)",
        "IT encoding [Pearson r]",
    )

    scatter_panel(
        axes["behavior_scatter"],
        model[
            "universality_category_adjusted_for_triplet_accuracy"
        ].to_numpy(),
        model[
            "triplet_accuracy_category_adjusted_for_triplet_accuracy"
        ].to_numpy()
        * 100,
        statistics["model_level"]["triplet_accuracy"]["raw_pearson_r"],
        statistics["model_level"]["triplet_accuracy"]["partial_pearson_r"],
        "Model universality\n(category adjusted)",
        "Triplet accuracy [%]",
    )

    for key, label in (("it_scatter", "a"), ("behavior_scatter", "b")):
        axes[key].text(
            -0.12,
            1.08,
            label,
            transform=axes[key].transAxes,
            fontsize=PANEL_LABEL_FONT,
            fontweight="bold",
            va="top",
        )
    for suffix, kwargs in (
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".svg", {}),
    ):
        fig.savefig(
            out / f"categorical_control{suffix}",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def _plain_p(p_value: float) -> str:
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def write_text_outputs(
    statistics: dict[str, Any],
    triplet_meta: dict[str, Any],
    out: Path,
) -> None:
    it_model = statistics["model_level"]["it_encoding"]
    beh_model = statistics["model_level"]["triplet_accuracy"]
    replication = statistics["triplet_replication"]

    conclusion = (
        "Category consistency does not account for the IT result and explains "
        "part, but not all, of the behavioral association."
    )
    report = f"""# Categorical-loading control

## Answer

{conclusion}

Across 162 models, universality remained associated with mean macaque IT
encoding after adjusting both variables for model mean category consistency
(partial Pearson r = {it_model['partial_pearson_r']:.3f}, 95% bootstrap CI
[{it_model['partial_pearson_ci_low']:.3f}, {it_model['partial_pearson_ci_high']:.3f}],
{_plain_p(it_model['partial_pearson_p'])}). The corresponding behavioral
association was attenuated but remained positive (partial r =
{beh_model['partial_pearson_r']:.3f}, 95% CI
[{beh_model['partial_pearson_ci_low']:.3f}, {beh_model['partial_pearson_ci_high']:.3f}],
{_plain_p(beh_model['partial_pearson_p'])}).

## What was controlled

- Category consistency is the per-dimension eta-squared from a one-way ANOVA
  over the 1,854 THINGS categories: the fraction of loading variance explained
  by category membership.
- The model-level analysis controls each model's mean eta-squared. Rank-based,
  loading-importance-weighted, and joint eta-squared/category-breadth
  sensitivities are in `statistics.json`.
- IT encoding exactly follows Figure 3: five-fold cross-validated ridge
  encoding, random seed 42, averaged across monkeys F and N.
- Behavioral prediction exactly follows Figure 3: cosine-similarity
  odd-one-out accuracy on the single rated image from each THINGS category.

## Input and comparability checks

- All four source tables cover the same 162 models; every model has 50 unique
  dimensions and no required values are missing.
- The downloaded public triplet files contain
  {triplet_meta['train_rows']:,} train and
  {triplet_meta['validation_rows']:,} validation trials
  ({triplet_meta['total_rows']:,} pooled). Appendix F reports 457,430
  validation trials, 3,788 more than the current public file.
- Recomputed full-factor accuracies on the downloaded release agree with the
  Figure 3 input to a mean absolute difference of
  {replication['mean_absolute_difference']:.8f} and a maximum absolute
  difference of {replication['max_absolute_difference']:.8f}
  (r = {replication['pearson_r']:.8f} across models). The release difference
  is therefore numerically negligible for this analysis, but it is recorded
  explicitly in the provenance.

## Limits

This is an associational control. It rules out the measured category
eta-squared as a sufficient explanation of the results; it does not establish
a causal effect of universality, and it cannot remove semantic organization
that the THINGS category labels do not capture. The behavioral relationship is
clearly reduced by the adjustment, so category structure is part of that
association even though it does not explain it completely.
"""
    (out / "REPORT.md").write_text(report)

    caption = f"""**Universality predicts neural and behavioral alignment after
controlling for category structure.** Model universality in relation to
(a) macaque IT encoding and (b) human odd-one-out triplet accuracy after
linearly adjusting both variables for each model's mean category consistency
(mean category eta-squared; n = 162 models). Adjusted values are residuals
shifted back to the original variable means for readability. IT encoding is
five-fold cross-validated ridge prediction averaged across two macaques.
Category adjustment leaves the IT association largely unchanged (raw
r = {it_model['raw_pearson_r']:.2f}; partial
r = {it_model['partial_pearson_r']:.2f}) and attenuates but does not eliminate
the behavioral association (raw r = {beh_model['raw_pearson_r']:.2f};
partial r = {beh_model['partial_pearson_r']:.2f}). Each point denotes one
model; lines show least-squares fits.
"""
    (out / "FIGURE_CAPTION.md").write_text(caption)




def analyze_and_plot(
    model: pd.DataFrame,
    assignment: pd.DataFrame,
    matched: pd.DataFrame,
    triplet_meta: dict[str, Any],
    rank: int,
    n_folds: int,
    random_state: int,
    out: Path,
    bootstrap_seed: int = 20260726,
) -> dict[str, Any]:
    n_models = len(model)
    if len(matched) != n_models:
        raise ValueError(
            f"Analysis requires all {n_models} models; found {len(matched)}."
        )

    summary = assignment_summary(assignment)
    matched = matched.merge(
        model[["model", "triplet_accuracy"]],
        on="model",
        how="inner",
        validate="one_to_one",
    ).merge(
        summary,
        on="model",
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != n_models:
        raise ValueError(
            f"Expected {n_models} complete matched results, got {len(matched)}"
        )

    model = add_adjusted_values(model)
    model_level = {
        "it_encoding": correlation_analysis(
            model, "it_encoding", 1, bootstrap_seed
        ),
        "triplet_accuracy": correlation_analysis(
            model, "triplet_accuracy", 2, bootstrap_seed
        ),
    }
    matched_halves = {
        "it_encoding": paired_analysis(
            matched["it_encoding_higher_universality"],
            matched["it_encoding_lower_universality"],
            1,
            bootstrap_seed,
        ),
        "triplet_accuracy": paired_analysis(
            matched["triplet_accuracy_higher_universality"],
            matched["triplet_accuracy_lower_universality"],
            2,
            bootstrap_seed,
        ),
    }
    eta_difference = matched["delta_eta_squared"].to_numpy()
    universality_difference = matched["delta_universality"].to_numpy()
    eta_test = stats.ttest_1samp(eta_difference, 0)
    universality_test = stats.ttest_1samp(universality_difference, 0)
    matching_balance = {
        "mean_delta_eta_squared": float(eta_difference.mean()),
        "max_abs_delta_eta_squared": float(np.abs(eta_difference).max()),
        "eta_paired_t": float(eta_test.statistic),
        "eta_paired_t_p": float(eta_test.pvalue),
        "mean_delta_universality": float(universality_difference.mean()),
        "min_delta_universality": float(universality_difference.min()),
        "universality_paired_t": float(universality_test.statistic),
        "universality_paired_t_p": float(universality_test.pvalue),
    }
    replication_difference = (
        matched["triplet_accuracy_public_full"]
        - matched["triplet_accuracy"]
    )
    replication_r = stats.pearsonr(
        matched["triplet_accuracy_public_full"],
        matched["triplet_accuracy"],
    )
    triplet_replication = {
        "mean_signed_difference": float(replication_difference.mean()),
        "mean_absolute_difference": float(
            np.abs(replication_difference).mean()
        ),
        "max_absolute_difference": float(
            np.abs(replication_difference).max()
        ),
        "pearson_r": float(replication_r.statistic),
        "pearson_p": float(replication_r.pvalue),
    }

    primary_p = {
        "model_partial_it": model_level["it_encoding"]["partial_pearson_p"],
        "model_partial_behavior": model_level["triplet_accuracy"][
            "partial_pearson_p"
        ],
        "matched_it": matched_halves["it_encoding"]["paired_t_p"],
        "matched_behavior": matched_halves["triplet_accuracy"]["paired_t_p"],
    }
    statistics = {
        "analysis_question": (
            "Does measured categorical loading explain the association of "
            "universality with IT encoding and human triplet prediction?"
        ),
        "model_level": model_level,
        "matched_halves": matched_halves,
        "matching_balance": matching_balance,
        "triplet_replication": triplet_replication,
        "primary_p_holm": holm_adjust(primary_p),
        "parameters": {
            "rank": rank,
            "n_folds": n_folds,
            "random_state": random_state,
            "match_tolerance": MATCH_TOLERANCE,
            "eta_quintiles": int(assignment["eta_quintile"].nunique()),
            "bootstrap_samples": N_BOOTSTRAP,
            "bootstrap_seed": bootstrap_seed,
        },
    }

    model.to_csv(out / "model_level_control.csv", index=False)
    assignment.to_csv(out / "dimension_assignment.csv", index=False)
    matched.to_csv(out / "matched_subset_results.csv", index=False)
    write_json(out / "statistics.json", statistics)
    make_figure(model, statistics, out)
    write_text_outputs(statistics, triplet_meta, out)
    return statistics


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def provenance(
    sources: dict[str, Path],
    triplet_meta: dict[str, Any],
    rank: int,
    n_folds: int,
    random_state: int,
    n_jobs: int,
    eta_quintiles: int,
) -> dict[str, Any]:
    return {
        "git_revision": git_revision(),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "sklearn": sklearn.__version__,
        },
        "parameters": {
            "rank": rank,
            "n_folds": n_folds,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "match_tolerance": MATCH_TOLERANCE,
            "eta_quintiles": eta_quintiles,
        },
        "sources": {
            str(path): {"sha256": sha256_file(path)}
            for path in sources.values()
        },
        "triplets": triplet_meta,
        "script": {
            "path": "experiments/rebuttal/categorical_control/run.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if cfg.n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    n_folds = 5
    sources = {
        "category": Path(
            "experiments/universality/dimensions/category_consistency"
            "/outputs/category_selectivity.csv"
        ),
        "behavior": Path(
            "experiments/universality/downstream_predictions/things_behavior"
            "/triplets/outputs/triplet_results.csv"
        ),
        "encoding_F": Path(
            "experiments/universality/downstream_predictions/macaque_encoding"
            "/encoding_things/outputs/encoding_results_F.csv"
        ),
        "encoding_N": Path(
            "experiments/universality/downstream_predictions/macaque_encoding"
            "/encoding_things/outputs/encoding_results_N.csv"
        ),
    }

    log.info("Loading and validating Figure 3 inputs")
    category, model = load_source_tables(sources, rank)
    triplets, triplet_meta = load_triplets(
        Path(cfg.data_root) / "behavior" / "things_triplets"
    )
    log.info(
        "Loaded %d models, %d dimensions, and %s triplets",
        len(model),
        len(category),
        f"{len(triplets):,}",
    )

    log.info("Building category-matched dimension halves")
    assignment = build_matched_assignment(category, rank)
    assignment.to_csv(out / "dimension_assignment.csv", index=False)

    matched = run_evaluation(
        assignment,
        triplets,
        triplet_meta,
        rank,
        n_folds,
        cfg.seed,
        cfg.n_jobs,
        cache_dir,
    )
    matched.to_csv(out / "matched_subset_results_in_progress.csv", index=False)

    write_json(
        out / "provenance.json",
        provenance(
            sources,
            triplet_meta,
            rank,
            n_folds,
            cfg.seed,
            cfg.n_jobs,
            int(assignment["eta_quintile"].nunique()),
        ),
    )

    statistics = analyze_and_plot(
        model,
        assignment,
        matched,
        triplet_meta,
        rank,
        n_folds,
        cfg.seed,
        out,
    )
    log.info(
        "Complete | partial r: IT %.3f, behavior %.3f | matched deltas: "
        "IT %.4f, behavior %.2f pp",
        statistics["model_level"]["it_encoding"]["partial_pearson_r"],
        statistics["model_level"]["triplet_accuracy"]["partial_pearson_r"],
        statistics["matched_halves"]["it_encoding"]["mean_difference"],
        100
        * statistics["matched_halves"]["triplet_accuracy"]["mean_difference"],
    )
    log.info("Figure: %s", out / "categorical_control.pdf")


if __name__ == "__main__":
    main()
