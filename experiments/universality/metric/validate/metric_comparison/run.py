#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from experiments.universality.shared import load_components_for_rank, load_model_info
from experiments.universality.shared.core import (
    best_match_indices,
    cos2_greedy,
    cos2_matrix,
    cos2_permutation,
    l2_normalize_columns,
    max_correlation,
)
from src.colors import INDIGO, ROSE, TEAL
from src.plotting import create_figure, save_figure, setup_plotting

N_PERMS = 1000

logger = logging.getLogger(__name__)


def compute_pairwise_metrics(all_w: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all metrics for every ordered model pair using core functions.

    Returns metric -> (n_models, n_models, rank). Diagonal is NaN.
    """
    metric_fns = {
        "max_corr": max_correlation,
        "cos2_greedy": cos2_greedy,
        "cos2_perm": cos2_permutation,
    }
    n_models, _, rank = all_w.shape
    logger.info(f"Computing pairwise metrics for {n_models} models")

    metrics = {name: np.full((n_models, n_models, rank), np.nan) for name in metric_fns}

    for i in range(n_models):
        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i + 1}/{n_models}")
        for j in range(n_models):
            if i == j:
                continue
            for name, fn in metric_fns.items():
                metrics[name][i, j] = fn(all_w[j], all_w[i])

    return metrics


def _null_cos2_worker(args: tuple) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Null distribution for one (target, source) pair via row permutation.

    Returns (target_idx, source_idx, greedy_p95, perm_p95).
    """
    target_idx, source_idx, norm_path, n_perms, seed = args
    all_norm = np.load(norm_path, mmap_mode="r")
    w_t = np.ascontiguousarray(all_norm[target_idx], dtype=np.float32)
    w_s = np.ascontiguousarray(all_norm[source_idx], dtype=np.float32)
    w_t_T = w_t.T

    rng = np.random.default_rng(seed)
    n_items, rank = w_t.shape
    null_greedy = np.empty((n_perms, rank), dtype=np.float32)
    null_perm = np.empty((n_perms, rank), dtype=np.float32)

    for b in range(n_perms):
        perm = rng.permutation(n_items)
        # Data is pre-L2-normalized (via core.l2_normalize_columns).
        # Row permutation preserves column norms, so cos² = (t.T @ s)² is valid.
        c2 = (w_t_T @ w_s[perm]) ** 2
        null_greedy[b] = c2.max(axis=1)
        row_ind, col_ind = linear_sum_assignment(-c2)
        null_perm[b] = c2[row_ind, col_ind]

    return (
        target_idx,
        source_idx,
        np.percentile(null_greedy, 95, axis=0).astype(np.float64),
        np.percentile(null_perm, 95, axis=0).astype(np.float64),
    )


def compute_cos2_null(
    all_w: np.ndarray,
    scratch_dir: Path,
    n_jobs: int = -2,
    n_perms: int = N_PERMS,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Pair-specific null percentiles for cos2_greedy and cos2_perm.

    Returns metric -> ``(n_models, n_models, rank)``.  Every off-diagonal
    source–target pair is independently calibrated as specified in the paper.
    Diagonal entries are NaN.
    """
    n_models, _, rank = all_w.shape
    all_norm = np.stack([l2_normalize_columns(all_w[i]) for i in range(n_models)])
    norm_path = scratch_dir / "_all_norm.npy"
    np.save(norm_path, all_norm)
    del all_norm

    pairs = []
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            pair_seed = int(
                np.random.SeedSequence([seed, i, j, n_perms]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            pairs.append((i, j, str(norm_path), n_perms, pair_seed))

    n_workers = n_jobs if n_jobs > 0 else max(1, mp.cpu_count() + 1 + n_jobs)
    logger.info(
        f"Computing cos² null for all {len(pairs)} ordered pairs, "
        f"{n_perms} perms, {n_workers} workers"
    )

    greedy_null = np.full((n_models, n_models, rank), np.nan)
    perm_null = np.full((n_models, n_models, rank), np.nan)
    with mp.Pool(n_workers) as pool:
        for done, (ti, si, g95, p95) in enumerate(
            pool.imap_unordered(_null_cos2_worker, pairs, chunksize=8)
        ):
            greedy_null[ti, si] = g95
            perm_null[ti, si] = p95
            if (done + 1) % 500 == 0:
                logger.info(f"  Null progress: {done + 1}/{len(pairs)}")

    norm_path.unlink(missing_ok=True)
    return {"cos2_greedy": greedy_null, "cos2_perm": perm_null}


def _null_adjust(raw: np.ndarray, threshold: np.ndarray) -> float:
    """Null-calibrate: (raw - p95) / (1 - p95), clipped to [0, 1], then averaged."""
    adjusted = (raw - threshold) / np.maximum(1 - threshold, 1e-8)
    return float(np.clip(adjusted, 0, 1).mean())


def aggregate_to_dataframe(
    model_names: list[str],
    metrics: dict[str, np.ndarray],
    cos2_null: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Collapse pairwise per-dimension scores to per-(model, dimension) universality.

    Each value is the mean across all other models.
    """
    n_models = len(model_names)
    rank = next(iter(metrics.values())).shape[2]

    rows = []
    for i, name in enumerate(model_names):
        others = np.ones(n_models, dtype=bool)
        others[i] = False
        for d in range(rank):
            row: dict[str, object] = {"model": name, "dimension": d}
            for metric_name, arr in metrics.items():
                row[metric_name] = float(arr[i, others, d].mean())
            if cos2_null is not None:
                for key in ("cos2_greedy", "cos2_perm"):
                    if key in metrics and key in cos2_null:
                        row[f"{key}_adj"] = _null_adjust(
                            metrics[key][i, others, d],
                            cos2_null[key][i, others, d],
                        )
            rows.append(row)

    return pd.DataFrame(rows)


def compute_collision_stats(
    all_w: np.ndarray,
    model_names: list[str],
) -> pd.DataFrame:
    """Count many-to-one collisions for greedy methods.

    Under greedy matching, multiple target dims can pick the same source dim.
    collision_rate = fraction of target dims mapped to a non-unique source dim.
    """
    n_models, _, rank = all_w.shape
    rows = []
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            w_source, w_target = all_w[j], all_w[i]

            corr_best = best_match_indices(w_source, w_target)
            cos2_best = cos2_matrix(w_source, w_target).argmax(axis=1)

            for method, best in [("max_corr", corr_best), ("cos2_greedy", cos2_best)]:
                n_unique = len(np.unique(best))
                rows.append({
                    "target": model_names[i],
                    "source": model_names[j],
                    "method": method,
                    "n_unique": n_unique,
                    "rank": rank,
                    "collision_rate": 1 - n_unique / rank,
                })
    return pd.DataFrame(rows)


def _label(metric: str, short: bool = False) -> str:
    labels = {
        "max_corr": ("Max |corr|", "Max |correlation| (baseline)"),
        "cos2_greedy": ("Cos² greedy", "Cosine² (greedy)"),
        "cos2_perm": ("Cos² perm", "Cosine² (permutation)"),
        "cos2_greedy_adj": ("Cos² greedy adj", "Cosine² greedy (adjusted)"),
        "cos2_perm_adj": ("Cos² perm adj", "Cosine² perm (adjusted)"),
    }
    pair = labels.get(metric)
    return pair[0 if short else 1] if pair else metric


def plot_scatter_baseline(df: pd.DataFrame, output_dir: Path):
    setup_plotting()
    alternatives = [c for c in df.columns if c not in ("model", "dimension", "max_corr")]
    for alt in alternatives:
        fig, ax = create_figure("single")
        ax.scatter(
            df["max_corr"], df[alt],
            s=2, alpha=0.25, color=INDIGO, rasterized=True,
        )
        rho, _ = spearmanr(df["max_corr"], df[alt])
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax.set(
            xlim=(0, 1), ylim=(min(df[alt].min() - 0.02, 0), 1),
            xlabel="Max |correlation| (baseline)", ylabel=_label(alt),
            title=f"ρ = {rho:.3f}",
        )
        ax.set_aspect("equal")
        save_figure(fig, output_dir / f"scatter_baseline_vs_{alt}")
        plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    setup_plotting()
    metric_cols = [c for c in df.columns if c not in ("model", "dimension")]
    n = len(metric_cols)
    rho_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(df[metric_cols[i]], df[metric_cols[j]])
            rho_matrix[i, j] = rho_matrix[j, i] = r

    fig, ax = create_figure("square")
    labels = [_label(c, short=True) for c in metric_cols]
    im = ax.imshow(rho_matrix, vmin=0.5, vmax=1, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rho_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Metric agreement (per-dimension)")
    save_figure(fig, output_dir / "metric_correlation_heatmap")
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    setup_plotting()
    metric_cols = [c for c in df.columns if c not in ("model", "dimension")]
    df_long = df[metric_cols].melt(var_name="metric", value_name="universality")
    df_long["metric"] = df_long["metric"].map(lambda x: _label(x, short=True))

    fig, ax = create_figure("wide")
    sns.violinplot(
        data=df_long, x="metric", y="universality", ax=ax,
        inner="box", color=INDIGO, alpha=0.7, cut=0,
    )
    ax.set(xlabel="", ylabel="Universality")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    save_figure(fig, output_dir / "metric_distributions")
    plt.close(fig)


def plot_null_calibration_effect(df: pd.DataFrame, output_dir: Path):
    setup_plotting()
    for col in df.columns:
        if not col.endswith("_adj"):
            continue
        raw_col = col.replace("_adj", "")
        if raw_col not in df.columns:
            continue
        fig, ax = create_figure("single")
        ax.scatter(
            df[raw_col], df[col],
            s=2, alpha=0.25, color=ROSE, rasterized=True,
        )
        rho, _ = spearmanr(df[raw_col], df[col])
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax.set(
            xlabel=f"{_label(raw_col)} (raw)",
            ylabel=f"{_label(raw_col)} (null-adjusted)",
            title=f"Null calibration effect — ρ = {rho:.3f}",
        )
        save_figure(fig, output_dir / f"null_calibration_{raw_col}")
        plt.close(fig)


def plot_greedy_vs_permutation(df: pd.DataFrame, output_dir: Path):
    if "cos2_greedy" not in df.columns or "cos2_perm" not in df.columns:
        return
    setup_plotting()
    fig, ax = create_figure("single")
    ax.scatter(
        df["cos2_greedy"], df["cos2_perm"],
        s=2, alpha=0.25, color=TEAL, rasterized=True,
    )
    rho, _ = spearmanr(df["cos2_greedy"], df["cos2_perm"])
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Cosine² (greedy)", ylabel="Cosine² (permutation)",
        title=f"ρ = {rho:.3f}",
    )
    ax.set_aspect("equal")
    save_figure(fig, output_dir / "greedy_vs_permutation")
    plt.close(fig)


def plot_collision_comparison(collision_df: pd.DataFrame, output_dir: Path):
    setup_plotting()
    method_labels = {"max_corr": "Max |correlation|", "cos2_greedy": "Cosine²"}
    df = collision_df.copy()
    df["method"] = df["method"].map(method_labels)

    fig, ax = create_figure("single")
    sns.violinplot(
        data=df, x="method", y="collision_rate", hue="method", ax=ax,
        inner="box", palette=[INDIGO, ROSE], cut=0, legend=False,
    )
    ax.set(
        xlabel="Greedy matching method",
        ylabel="Collision rate",
        title="Fraction of source dims never selected as best match",
    )
    for i, method in enumerate(df["method"].unique()):
        med = df[df["method"] == method]["collision_rate"].median()
        ax.text(i, med + 0.02, f"median = {med:.2f}", ha="center", fontsize=7)
    save_figure(fig, output_dir / "collision_rate_comparison")
    plt.close(fig)


def plot_model_level_agreement(df: pd.DataFrame, output_dir: Path):
    setup_plotting()
    model_means = df.groupby("model").mean(numeric_only=True).reset_index()
    alternatives = [c for c in model_means.columns if c not in ("model", "dimension", "max_corr")]
    for alt in alternatives:
        fig, ax = create_figure("single")
        ax.scatter(
            model_means["max_corr"], model_means[alt],
            s=12, alpha=0.6, color=INDIGO,
        )
        rho, _ = spearmanr(model_means["max_corr"], model_means[alt])
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
        ax.set(
            xlabel="Max |correlation| (baseline, model mean)",
            ylabel=f"{_label(alt)} (model mean)",
            title=f"Model-level agreement — ρ = {rho:.3f}",
        )
        save_figure(fig, output_dir / f"model_level_baseline_vs_{alt}")
        plt.close(fig)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    models = list(cfg.models) or load_model_info()["feature_uid"].astype(str).tolist()
    components = load_components_for_rank(
        rank=cfg.factorization.rank, models=models, dataset=cfg.dataset,
    )
    model_names = list(components.keys())
    all_w = np.stack([components[name] for name in model_names])
    logger.info(f"Loaded {all_w.shape[0]} models, shape {all_w.shape}")
    del components

    metrics = compute_pairwise_metrics(all_w)
    logger.info("Pairwise metrics done")

    cos2_null = compute_cos2_null(
        all_w, out, n_jobs=cfg.n_jobs, n_perms=N_PERMS, seed=cfg.seed,
    )
    logger.info("Cos² null calibration done")

    df = aggregate_to_dataframe(model_names, metrics, cos2_null)
    df.to_csv(out / "metrics_comparison.csv", index=False)
    logger.info(f"Saved metrics_comparison.csv ({len(df)} rows)")

    collision_df = compute_collision_stats(all_w, model_names)
    collision_df.to_csv(out / "collision_stats.csv", index=False)
    logger.info(f"Saved collision_stats.csv ({len(collision_df)} rows)")

    params = {
        "rank": cfg.factorization.rank,
        "n_models": len(model_names),
        "n_perms": N_PERMS,
        "seed": cfg.seed,
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))

    df = pd.read_csv(out / "metrics_comparison.csv")
    logger.info(f"Loaded {len(df)} rows for plotting")

    plot_scatter_baseline(df, out)
    plot_correlation_heatmap(df, out)
    plot_distributions(df, out)
    plot_greedy_vs_permutation(df, out)
    plot_model_level_agreement(df, out)

    if any(c.endswith("_adj") for c in df.columns):
        plot_null_calibration_effect(df, out)

    plot_collision_comparison(pd.read_csv(out / "collision_stats.csv"), out)

    logger.info(f"All plots saved to {out}")


if __name__ == "__main__":
    main()
