#!/usr/bin/env python3
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from experiments.universality.shared import load_components_for_rank, load_model_info
from experiments.universality.shared.core import (
    cos2_perm_null_thresholds,
    cos2_permutation,
    null_adjust,
)
from src.colors import GRAY, GRAY_DARK, TEAL
from src.plotting import CLASS_COLORS, create_figure, save_figure, setup_plotting

N_PERMS = 1000

log = logging.getLogger(__name__)


def compute_pairwise_cos2(
    components: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Precompute all pairwise cos2_perm scores.

    Returns (models, scores) where scores[i, j, d] is the cos2_perm score
    for target dim d of model i matched against model j. Diagonal is zero.
    """
    models = list(components.keys())
    n = len(models)
    rank = next(iter(components.values())).shape[1]
    scores = np.zeros((n, n, rank))
    for i in range(n):
        if (i + 1) % 20 == 0:
            log.info(f"  Pairwise: {i + 1}/{n}")
        for j in range(n):
            if i != j:
                scores[i, j] = cos2_permutation(components[models[j]], components[models[i]])
    return models, scores


def compute_null_thresholds(
    components: dict[str, np.ndarray],
    models: list[str],
    n_perms: int = N_PERMS,
    seed: int = 42,
    n_jobs: int = -2,
) -> np.ndarray:
    """Precompute a null threshold for every ordered model pair.

    Returns ``(n_models, n_models, rank)``.  Entry ``[i, j]`` is ordered by
    dimensions of target model ``i`` and calibrates source model ``j``.
    """
    n = len(models)
    rank = components[models[0]].shape[1]
    log.info(
        "  Computing null thresholds for %d ordered pairs (%d perms per pair)",
        n * (n - 1),
        n_perms,
    )

    def _threshold_for_target(i: int) -> np.ndarray:
        source_indices = [j for j in range(n) if j != i]
        pair_thresholds = cos2_perm_null_thresholds(
            components[models[i]],
            [components[models[j]] for j in source_indices],
            n_perms=n_perms,
            seed=seed + i * n,
        )
        result = np.full((n, rank), np.nan)
        result[source_indices] = pair_thresholds
        return result

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_threshold_for_target)(i) for i in range(n)
    )
    return np.stack(results)


def model_universality_from_mask(
    scores: np.ndarray, mask: np.ndarray, thresholds: np.ndarray,
) -> np.ndarray:
    """Compute model-level null-adjusted universality from pairwise scores with a mask.

    For each target i, adjusts every source score using that ordered model
    pair's null threshold, then averages across selected sources.
    """
    n, _, rank = scores.shape
    result = np.zeros(n)
    for i in range(n):
        valid = mask[i]
        if valid.any():
            raw = scores[i, valid]
            adj = null_adjust(raw, thresholds[i, valid])
            result[i] = adj.mean()
    return result


def get_model_families(models: list[str]) -> np.ndarray:
    """Return model_class (Convolutional, Transformer, MLP-Mixer, Hybrid) for each model."""
    meta = load_model_info()
    uid_to_class = dict(zip(meta["feature_uid"].astype(str), meta["model_class"]))
    return np.array([uid_to_class.get(m, "unknown") for m in models])


def run_bootstrap(
    scores: np.ndarray,
    thresholds: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 1000,
    subsample_frac: float = 0.2,
) -> dict:
    """Random subsample bootstrap: draw subsample_frac of models, measure rank stability."""
    n = scores.shape[0]
    full_mask = ~np.eye(n, dtype=bool)
    full_universality = model_universality_from_mask(scores, full_mask, thresholds)

    k = max(2, int(n * subsample_frac))
    log.info(f"  Bootstrap: {subsample_frac:.0%} subsample = {k} of {n} models")

    rank_correlations = np.zeros(n_boot)
    model_scores = np.full((n_boot, n), np.nan)

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            log.info(f"  Bootstrap {b + 1}/{n_boot}")
        idx = rng.choice(n, size=k, replace=False)
        sub_mask = np.zeros((n, n), dtype=bool)
        for i in idx:
            for j in idx:
                if i != j:
                    sub_mask[i, j] = True

        sub_universality = model_universality_from_mask(scores, sub_mask, thresholds)
        model_scores[b, idx] = sub_universality[idx]
        rho, _ = spearmanr(full_universality[idx], sub_universality[idx])
        rank_correlations[b] = rho

    median_rho = np.median(rank_correlations)
    rep_idx = np.argmin(np.abs(rank_correlations - median_rho))

    return {
        "rank_correlations": rank_correlations,
        "model_scores": model_scores,
        "full_universality": full_universality,
        "subsample_size": k,
        "representative_scores": model_scores[rep_idx],
        "representative_rho": rank_correlations[rep_idx],
    }


def plot_bootstrap_rank_stability(
    rank_corrs: np.ndarray, subsample_size: int, total: int,
) -> plt.Figure:
    fig, ax = create_figure("single")
    ax.hist(rank_corrs, bins=30, color=TEAL, edgecolor="white", alpha=0.8)
    median = np.median(rank_corrs)
    ax.axvline(median, color=TEAL, ls="--", lw=1.5, label=f"median = {median:.3f}")
    fraction = 100 * subsample_size / total
    ax.set_xlabel(
        f"Spearman ρ (full set vs. {fraction:.0f}% subsample, "
        f"n={subsample_size} of {total})"
    )
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    return fig


def plot_bootstrap_ci(
    full_universality: np.ndarray,
    model_scores: np.ndarray,
    families: np.ndarray,
) -> plt.Figure:
    order = np.argsort(full_universality)[::-1]
    ci_lo = np.nanpercentile(model_scores, 2.5, axis=0)
    ci_hi = np.nanpercentile(model_scores, 97.5, axis=0)

    fig, ax = create_figure("wide")
    x = np.arange(len(order))
    for i, idx in enumerate(order):
        color = CLASS_COLORS.get(families[idx], GRAY)
        ax.plot([i, i], [ci_lo[idx], ci_hi[idx]], color=color, alpha=0.4, lw=0.8)
    colors = [CLASS_COLORS.get(families[idx], GRAY) for idx in order]
    ax.scatter(x, full_universality[order], s=4, color=colors, zorder=3, alpha=0.8)

    for fam in ["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"]:
        if fam in families:
            ax.scatter([], [], color=CLASS_COLORS.get(fam), s=20, label=fam)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlabel("Model (sorted by universality)")
    ax.set_ylabel("Universality")
    ax.set_xticks([])
    return fig


def run_leave_family_out(
    scores: np.ndarray,
    thresholds: np.ndarray,
    families: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = scores.shape[0]
    full_mask = ~np.eye(n, dtype=bool)
    full_universality = model_universality_from_mask(scores, full_mask, thresholds)

    between_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and families[i] != families[j]:
                between_mask[i, j] = True

    lfo_universality = model_universality_from_mask(scores, between_mask, thresholds)
    has_peers = between_mask.any(axis=1)
    lfo_universality[~has_peers] = np.nan
    return full_universality, lfo_universality


def plot_lfo_scatter(df: pd.DataFrame) -> plt.Figure:
    df = df.dropna(subset=["universality_lfo"])
    fig, ax = create_figure("single")
    for fam in ["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"]:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub["universality_full"], sub["universality_lfo"],
            s=14, alpha=0.6, color=CLASS_COLORS.get(fam, GRAY),
            label=f"{fam} (n={len(sub)})", edgecolors="none",
        )
    lims = [
        min(df["universality_full"].min(), df["universality_lfo"].min()) - 0.02,
        max(df["universality_full"].max(), df["universality_lfo"].max()) + 0.02,
    ]
    ax.plot(lims, lims, ls="--", color=GRAY, lw=0.8, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Universality (full model set)")
    ax.set_ylabel("Universality (leave-family-out)")
    rho, _ = spearmanr(df["universality_full"], df["universality_lfo"])
    r = np.corrcoef(df["universality_full"], df["universality_lfo"])[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.3f}\nρ = {rho:.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    return fig


def plot_lfo_delta_by_family(df: pd.DataFrame) -> plt.Figure:
    fig, ax = create_figure("single")
    known_families = ["Convolutional", "Transformer", "MLP-Mixer", "Hybrid"]
    order = [f for f in known_families if f in df["family"].values]
    df = df[df["family"].isin(order)]
    palette = {f: CLASS_COLORS.get(f, GRAY) for f in order}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.boxplot(
            data=df, x="family", y="delta", order=order, hue="family",
            palette=palette, legend=False, width=0.5, fliersize=0,
            boxprops=dict(alpha=0.3),
            medianprops=dict(color=GRAY_DARK, linewidth=1.5), ax=ax,
        )
        sns.stripplot(
            data=df, x="family", y="delta", order=order,
            color=GRAY_DARK, alpha=0.4, size=3, jitter=0.15, ax=ax,
        )

    ax.axhline(0, color=GRAY, ls="--", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Δ universality (leave-family-out − full)")
    labels = [f"{f}\n(n={len(df[df['family'] == f])})" for f in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    return fig


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    )
    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    log.info("Loading components...")
    components = load_components_for_rank(
        rank=cfg.factorization.rank,
        models=list(cfg.models) or None,
        dataset=cfg.dataset,
    )
    log.info(f"Loaded {len(components)} models")

    log.info("Computing pairwise cos2_perm scores...")
    models, scores = compute_pairwise_cos2(components)
    families = get_model_families(models)
    fam_counts = dict(zip(*np.unique(families, return_counts=True)))
    log.info(f"Model class distribution: {fam_counts}")

    log.info("Computing null thresholds...")
    thresholds = compute_null_thresholds(
        components, models, n_perms=N_PERMS, seed=cfg.seed, n_jobs=cfg.n_jobs,
    )

    log.info("Running bootstrap...")
    rng = np.random.default_rng(cfg.seed)
    boot = run_bootstrap(scores, thresholds, rng)

    rank_corrs = boot["rank_correlations"]
    log.info(
        f"Rank stability: median ρ = {np.median(rank_corrs):.3f}, "
        f"95%% range = [{np.percentile(rank_corrs, 2.5):.3f}, "
        f"{np.percentile(rank_corrs, 97.5):.3f}]"
    )

    boot_df = pd.DataFrame({
        "model": models,
        "family": families,
        "universality_full": boot["full_universality"],
        "boot_mean": np.nanmean(boot["model_scores"], axis=0),
        "boot_lo": np.nanpercentile(boot["model_scores"], 2.5, axis=0),
        "boot_hi": np.nanpercentile(boot["model_scores"], 97.5, axis=0),
        "boot_n": np.sum(~np.isnan(boot["model_scores"]), axis=0),
    })
    boot_df.to_csv(out / "bootstrap_results.csv", index=False)
    np.save(out / "bootstrap_rank_correlations.npy", rank_corrs)
    log.info(
        f"  Median per-iteration ρ = {np.median(rank_corrs):.3f}, "
        f"mean boot samples per model = {boot_df['boot_n'].mean():.0f}"
    )

    fig = plot_bootstrap_rank_stability(rank_corrs, boot["subsample_size"], len(models))
    save_figure(fig, out / "bootstrap_rank_stability")

    fig = plot_bootstrap_ci(boot["full_universality"], boot["model_scores"], families)
    save_figure(fig, out / "bootstrap_ci")

    log.info("Computing leave-family-out universality...")
    full_u, lfo_u = run_leave_family_out(scores, thresholds, families)

    lfo_df = pd.DataFrame({
        "model": models,
        "family": families,
        "universality_full": full_u,
        "universality_lfo": lfo_u,
        "delta": lfo_u - full_u,
    })
    lfo_df.to_csv(out / "leave_family_out_results.csv", index=False)

    rho, _ = spearmanr(
        lfo_df.dropna()["universality_full"], lfo_df.dropna()["universality_lfo"]
    )
    log.info(f"Leave-family-out: ρ = {rho:.3f}")

    fig = plot_lfo_scatter(lfo_df)
    save_figure(fig, out / "leave_family_out_scatter")

    fig = plot_lfo_delta_by_family(lfo_df)
    save_figure(fig, out / "leave_family_out_delta")

    log.info(f"All outputs saved to {out}")


if __name__ == "__main__":
    main()
