"""Which NMF dimensions drive macaque IT encoding, and are they the universal ones?"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import stats

from experiments.universality.downstream_predictions.macaque_encoding.encoding_things.run import (
    compute_noise_ceiling,
    encode_single_model,
    load_macaque_it,
)
from experiments.universality.shared import load_components_for_rank, load_universality
from src.colors import GRAY, GRAY_DARK, GRAY_LIGHT, INDIGO, PURPLE, ROSE
from src.plotting import (
    create_figure,
    plt,
    save_figure,
    setup_plotting,
)

N_FOLDS = 5

log = logging.getLogger(__name__)


def ablation_single_model(
    model_name: str,
    w: np.ndarray,
    y: np.ndarray,
    noise_ceil: np.ndarray,
    n_folds: int,
    random_state: int,
) -> list[dict]:
    """LOO dimension ablation for one model's neural encoding."""
    rank = w.shape[1]
    r_full = encode_single_model(w, y, n_folds, random_state)
    r_full_corr = r_full / noise_ceil
    mean_r_full = float(np.mean(r_full))
    mean_r_corr_full = float(np.mean(np.clip(r_full_corr, -1, 1)))

    results = []
    for d in range(rank):
        w_abl = w.copy()
        w_abl[:, d] = 0.0
        r_abl = encode_single_model(w_abl, y, n_folds, random_state)
        r_abl_corr = r_abl / noise_ceil
        mean_r_abl = float(np.mean(r_abl))
        mean_r_corr_abl = float(np.mean(np.clip(r_abl_corr, -1, 1)))
        results.append({
            "model": model_name,
            "dimension": d,
            "mean_r_full": mean_r_full,
            "mean_r_ablated": mean_r_abl,
            "importance": mean_r_full - mean_r_abl,
            "mean_r_corrected_full": mean_r_corr_full,
            "mean_r_corrected_ablated": mean_r_corr_abl,
            "importance_corrected": mean_r_corr_full - mean_r_corr_abl,
        })
    return results


def subset_single_model(
    model_name: str,
    w: np.ndarray,
    y: np.ndarray,
    noise_ceil: np.ndarray,
    n_folds: int,
    random_state: int,
    uni_per_dim: pd.DataFrame,
) -> dict:
    """Encoding from universal vs specific dimension subsets for one model."""
    rank = w.shape[1]
    model_uni = uni_per_dim[uni_per_dim["model"] == model_name].sort_values("dimension")
    if len(model_uni) != rank:
        return {}

    uni_order = model_uni.sort_values("universality", ascending=False)["dimension"].values
    top_half = uni_order[: rank // 2]
    bottom_half = uni_order[rank // 2 :]

    r_full = encode_single_model(w, y, n_folds, random_state)
    mean_r_full = float(np.mean(r_full))

    w_top = np.zeros_like(w)
    w_top[:, top_half] = w[:, top_half]
    r_top = encode_single_model(w_top, y, n_folds, random_state)

    w_bot = np.zeros_like(w)
    w_bot[:, bottom_half] = w[:, bottom_half]
    r_bot = encode_single_model(w_bot, y, n_folds, random_state)

    r_top_corr = r_top / noise_ceil
    r_bot_corr = r_bot / noise_ceil

    return {
        "model": model_name,
        "mean_r_full": mean_r_full,
        "mean_r_universal_half": float(np.mean(r_top)),
        "mean_r_specific_half": float(np.mean(r_bot)),
        "mean_r_corrected_universal": float(np.mean(np.clip(r_top_corr, -1, 1))),
        "mean_r_corrected_specific": float(np.mean(np.clip(r_bot_corr, -1, 1))),
    }


def cumulative_single_model(
    model_name: str,
    w: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
    uni_per_dim: pd.DataFrame,
    n_random: int = 5,
) -> list[dict]:
    """Cumulative encoding curve: universality order vs random."""
    rank = w.shape[1]
    model_uni = uni_per_dim[uni_per_dim["model"] == model_name].sort_values("dimension")
    if len(model_uni) != rank:
        return []

    uni_order = model_uni.sort_values("universality", ascending=False)["dimension"].values

    results = []
    for k in range(1, rank + 1):
        dims_uni = uni_order[:k]
        w_sub = np.zeros_like(w)
        w_sub[:, dims_uni] = w[:, dims_uni]
        r_uni = encode_single_model(w_sub, y, n_folds, random_state)
        mean_r_uni = float(np.mean(r_uni))

        acc_random = []
        rng = np.random.RandomState(random_state)
        for _ in range(n_random):
            dims_rand = rng.choice(rank, k, replace=False)
            w_rand = np.zeros_like(w)
            w_rand[:, dims_rand] = w[:, dims_rand]
            r_rand = encode_single_model(w_rand, y, n_folds, random_state)
            acc_random.append(float(np.mean(r_rand)))

        results.append({
            "model": model_name,
            "k": k,
            "mean_r_universality_order": mean_r_uni,
            "mean_r_random_mean": np.mean(acc_random),
            "mean_r_random_std": np.std(acc_random),
        })
    return results


def run_ablation(
    components: dict[str, np.ndarray],
    monkey: str,
    n_folds: int,
    n_jobs: int,
    random_state: int,
) -> pd.DataFrame:
    """Run LOO dimension ablation for all models."""
    y, reliab = load_macaque_it(monkey)
    noise_ceil = compute_noise_ceiling(reliab)
    models = sorted(components)
    rank = next(iter(components.values())).shape[1]
    log.info(f"Monkey {monkey}: ablation for {len(models)} models × {rank} dimensions")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(ablation_single_model)(
            m, components[m], y, noise_ceil, n_folds, random_state,
        )
        for m in models
    )
    return pd.DataFrame([r for batch in results for r in batch])


def run_subsets(
    components: dict[str, np.ndarray],
    monkey: str,
    n_folds: int,
    n_jobs: int,
    random_state: int,
    uni_per_dim: pd.DataFrame,
) -> pd.DataFrame:
    """Run universal vs specific half encoding for all models."""
    y, reliab = load_macaque_it(monkey)
    noise_ceil = compute_noise_ceiling(reliab)
    models = sorted(components)
    log.info(f"Monkey {monkey}: subset comparison for {len(models)} models")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(subset_single_model)(
            m, components[m], y, noise_ceil, n_folds, random_state, uni_per_dim,
        )
        for m in models
    )
    return pd.DataFrame([r for r in results if r])


def run_cumulative(
    components: dict[str, np.ndarray],
    monkey: str,
    n_folds: int,
    n_jobs: int,
    random_state: int,
    uni_per_dim: pd.DataFrame,
) -> pd.DataFrame:
    """Run cumulative encoding curve for all models."""
    y, _ = load_macaque_it(monkey)
    models = sorted(components)
    log.info(f"Monkey {monkey}: cumulative curve for {len(models)} models")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(cumulative_single_model)(
            m, components[m], y, n_folds, random_state, uni_per_dim,
        )
        for m in models
    )
    return pd.DataFrame([r for batch in results for r in batch])


def plot_importance_vs_universality(
    ablation_df: pd.DataFrame,
    uni_df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
) -> None:
    """Scatter: dimension universality vs neural importance."""
    merged = ablation_df.merge(
        uni_df[["model", "dimension", "universality"]],
        on=["model", "dimension"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ylim_lo = merged["importance"].quantile(0.005)
    ylim_hi = merged["importance"].quantile(0.995)
    margin = (ylim_hi - ylim_lo) * 0.1
    r_all, p_all = stats.pearsonr(merged["universality"], merged["importance"])
    rho_all, rho_p_all = stats.spearmanr(merged["universality"], merged["importance"])
    ax.scatter(
        merged["universality"], merged["importance"],
        alpha=0.04, s=4, c=ROSE, rasterized=True,
    )
    # Binned means (model-agnostic summary)
    n_bins = 20
    merged["uni_bin"] = pd.cut(merged["universality"], bins=n_bins)
    binned = merged.groupby("uni_bin", observed=True)[["universality", "importance"]].mean()
    ax.scatter(
        binned["universality"], binned["importance"],
        s=40, c=PURPLE, edgecolors="white", linewidth=0.5,
        zorder=3, label=f"Binned mean (n={n_bins})",
    )
    merged.drop(columns="uni_bin", inplace=True)
    # Lowess trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw_result = lowess(merged["importance"], merged["universality"], frac=0.3)
    ax.plot(lw_result[:, 0], lw_result[:, 1], "-", color=PURPLE, lw=2, zorder=2)
    ax.set_ylim(ylim_lo - margin, ylim_hi + margin)
    ax.set_xlabel("Dimension universality")
    ax.set_ylabel("Neural importance (encoding drop)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(
        0.05, 0.82,
        f"r = {r_all:.3f}, p = {p_all:.1e}\nrho = {rho_all:.3f}",
        transform=ax.transAxes, va="top", fontsize=7,
    )

    ax = axes[1]
    per_model_r = []
    for _, grp in merged.groupby("model"):
        if len(grp) > 5:
            r_m, _ = stats.pearsonr(grp["universality"], grp["importance"])
            per_model_r.append(r_m)
    ax.hist(per_model_r, bins=25, color=ROSE, edgecolor="white", alpha=0.8)
    mean_r = np.mean(per_model_r)
    ax.axvline(mean_r, color=PURPLE, lw=2, ls="--",
               label=f"Mean r={mean_r:.3f}")
    t_stat, t_p = stats.ttest_1samp(per_model_r, 0)
    ax.set_xlabel("Within-model r(universality, importance)")
    ax.set_ylabel("Count")
    ax.text(
        0.05, 0.95,
        f"t = {t_stat:.1f}, p = {t_p:.1e}\nn = {len(per_model_r)} models",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_dir / f"universality_vs_neural_importance_{monkey}")
    log.info(f"Saved universality_vs_neural_importance_{monkey}")


def plot_subset_comparison(
    subset_df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
) -> None:
    """Paired comparison: universal half vs specific half encoding."""
    import seaborn as sns

    sub = subset_df.dropna()
    t_stat, t_p = stats.ttest_rel(sub["mean_r_universal_half"], sub["mean_r_specific_half"])
    diff_mean = (sub["mean_r_universal_half"] - sub["mean_r_specific_half"]).mean()

    melted = pd.melt(
        sub, id_vars=["model"],
        value_vars=["mean_r_universal_half", "mean_r_specific_half"],
        var_name="subset", value_name="encoding_r",
    )
    melted["subset"] = melted["subset"].map({
        "mean_r_universal_half": "Universal\n(top 25 dims)",
        "mean_r_specific_half": "Specific\n(bottom 25 dims)",
    })

    fig, ax = plt.subplots(figsize=(3.5, 4))
    sns.boxplot(
        data=melted, x="subset", y="encoding_r", ax=ax,
        width=0.5, palette=[ROSE, INDIGO],
        fliersize=2, linewidth=1,
    )
    sns.stripplot(
        data=melted, x="subset", y="encoding_r", ax=ax,
        size=3, alpha=0.3, color=GRAY_DARK, jitter=0.15,
    )
    for model in sub["model"]:
        row = sub[sub["model"] == model]
        ax.plot(
            [0, 1],
            [row["mean_r_universal_half"].values[0], row["mean_r_specific_half"].values[0]],
            color="gray", alpha=0.06, lw=0.5,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Mean encoding r")
    ax.set_title(
        f"Monkey {monkey}: universal dims encode IT better\n"
        f"(+{diff_mean:.4f}, t={t_stat:.1f}, p={t_p:.1e})"
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"universal_vs_specific_encoding_{monkey}")
    log.info(f"Saved universal_vs_specific_encoding_{monkey}")


def plot_cumulative_curve(
    cum_df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
) -> None:
    """Cumulative encoding curve: universality order vs random order."""
    agg = cum_df.groupby("k").agg(
        uni_mean=("mean_r_universality_order", "mean"),
        uni_se=("mean_r_universality_order", "sem"),
        rand_mean=("mean_r_random_mean", "mean"),
        rand_se=("mean_r_random_mean", "sem"),
    ).reset_index()

    fig, ax = create_figure("single")
    ax.fill_between(
        agg["k"],
        agg["uni_mean"] - agg["uni_se"],
        agg["uni_mean"] + agg["uni_se"],
        alpha=0.15, color=ROSE,
    )
    ax.plot(agg["k"], agg["uni_mean"], "-", color=ROSE, lw=2,
            label="Universality order")
    ax.fill_between(
        agg["k"],
        agg["rand_mean"] - agg["rand_se"],
        agg["rand_mean"] + agg["rand_se"],
        alpha=0.15, color=GRAY,
    )
    ax.plot(agg["k"], agg["rand_mean"], "-", color=GRAY, lw=2,
            label="Random order")
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Mean encoding r")
    ax.set_title(f"Monkey {monkey}")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir / f"cumulative_encoding_{monkey}")
    log.info(f"Saved cumulative_encoding_{monkey}")


def analyze_clusters(
    ablation_df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
    rank: int,
    dataset: str,
) -> None:
    """Rank semantic clusters by neural predictivity."""
    clusters_csv = Path(
        "experiments/universality/dimensions/clustering/outputs/dimension_clusters.csv"
    )
    profiles_csv = Path(
        "experiments/universality/dimensions/clustering/outputs/cluster_profiles.csv"
    )
    if not clusters_csv.exists() or not profiles_csv.exists():
        log.warning("Clustering outputs not found, skipping cluster analysis")
        return

    clusters = pd.read_csv(clusters_csv)
    profiles = pd.read_csv(profiles_csv)
    profiles["cluster"] = profiles["cluster"].str.lstrip("C").astype(int)

    merged = ablation_df.merge(clusters, on=["model", "dimension"], how="inner")
    cluster_stats = merged.groupby("cluster")["importance"].agg(
        mean_importance="mean",
        median_importance="median",
        std_importance="std",
        n_dims="count",
    ).reset_index()

    cluster_stats = cluster_stats.merge(
        profiles[["cluster", "n_models", "model_pct", "coherence", "top_categories"]],
        on="cluster", how="left",
    )
    # Fill missing categories from W matrices
    missing = cluster_stats[cluster_stats["top_categories"].isna()]
    if len(missing) > 0:
        log.info(f"Computing categories for {len(missing)} unlabeled clusters...")
        missing_members = clusters[clusters["cluster"].isin(missing["cluster"])]
        cat_map = _get_cluster_categories(missing_members, rank)
        cluster_stats.loc[cluster_stats["top_categories"].isna(), "top_categories"] = (
            cluster_stats.loc[cluster_stats["top_categories"].isna(), "cluster"].map(cat_map)
        )

    cluster_stats = cluster_stats.sort_values("mean_importance", ascending=False)
    out_csv = output_dir / f"cluster_neural_predictivity_{monkey}.csv"
    cluster_stats.to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv} ({len(cluster_stats)} clusters)")

    top = cluster_stats.head(20)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(top) + 1.5))
    y_pos = np.arange(len(top))

    uni_df = load_universality(rank, dataset=dataset)
    cluster_uni = (
        merged.merge(uni_df[["model", "dimension", "universality"]], on=["model", "dimension"])
        .groupby("cluster")["universality"].mean()
    )
    top = top.merge(cluster_uni.rename("mean_universality"), on="cluster", how="left")

    norm = plt.Normalize(top["mean_universality"].min(), top["mean_universality"].max())
    cmap = plt.cm.RdYlBu
    colors = cmap(norm(top["mean_universality"].values))

    ax.barh(y_pos, top["mean_importance"], color=colors, edgecolor="white", linewidth=0.3)
    labels = []
    for _, row in top.iterrows():
        cats = row["top_categories"]
        short = ", ".join(cats.split(", ")[:3]) if isinstance(cats, str) else "unlabeled"
        labels.append(f"{row['cluster']} ({row['n_dims']}d): {short}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean neural importance (encoding drop)")
    ax.set_title(f"Monkey {monkey}: top 20 clusters by neural predictivity")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Mean universality", fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_dir / f"cluster_neural_predictivity_{monkey}")
    log.info(f"Saved cluster_neural_predictivity_{monkey}")


def _get_cluster_categories(
    cluster_members: pd.DataFrame,
    rank: int,
    components: dict[str, np.ndarray] | None = None,
    n_top: int = 5,
) -> dict[int, str]:
    """Compute top categories for clusters missing profile labels.

    For each cluster, averages its member W vectors across categories and
    returns the top-n category names by mean activation.
    """
    from src.data import ThingsDataset
    ds = ThingsDataset()
    categories = ds.categories
    n_per_cat = 12
    n_cats = len(set(categories))

    if components is None:
        components = load_components_for_rank(rank)

    cluster_ids = cluster_members["cluster"].unique()
    result = {}
    for cid in cluster_ids:
        members = cluster_members[cluster_members["cluster"] == cid]
        cat_scores = np.zeros(n_cats)
        count = 0
        for _, row in members.iterrows():
            model, dim = row["model"], int(row["dimension"])
            if model not in components:
                continue
            w = components[model][:, dim]
            # Average activation per category (reshape 22248 -> 1854 x 12, mean over exemplars)
            cat_scores += w.reshape(n_cats, n_per_cat).mean(axis=1)
            count += 1
        if count == 0:
            continue
        cat_scores /= count
        top_idx = np.argsort(cat_scores)[::-1][:n_top]
        unique_cats = list(dict.fromkeys(categories))
        top_names = [unique_cats[i] for i in top_idx]
        result[cid] = ", ".join(top_names)
    return result


def _analyze_dissociation(
    merged: pd.DataFrame,
    monkey: str,
    output_dir: Path,
    rank: int,
) -> None:
    """Deeper brain-behavior dissociation: which clusters are neural- vs behavior-specific?"""
    clusters_csv = Path(
        "experiments/universality/dimensions/clustering/outputs/dimension_clusters.csv"
    )
    profiles_csv = Path(
        "experiments/universality/dimensions/clustering/outputs/cluster_profiles.csv"
    )
    if not clusters_csv.exists() or not profiles_csv.exists():
        log.warning("Clustering outputs not found, skipping dissociation analysis")
        return

    clusters = pd.read_csv(clusters_csv)
    profiles = pd.read_csv(profiles_csv)
    profiles["cluster"] = profiles["cluster"].str.lstrip("C").astype(int)

    mc = merged.merge(clusters, on=["model", "dimension"], how="inner")

    # Per-cluster mean neural and behavioral importance
    cluster_agg = mc.groupby("cluster").agg(
        neural=("neural_importance", "mean"),
        behavioral=("behavioral_importance", "mean"),
        universality=("universality", "mean"),
        n_dims=("neural_importance", "count"),
    ).reset_index()
    cluster_agg = cluster_agg.merge(
        profiles[["cluster", "top_categories"]], on="cluster", how="left",
    )

    # Fill missing categories from W matrices
    missing = cluster_agg[cluster_agg["top_categories"].isna()]
    if len(missing) > 0:
        log.info(f"Computing categories for {len(missing)} unlabeled clusters...")
        missing_members = clusters[clusters["cluster"].isin(missing["cluster"])]
        cat_map = _get_cluster_categories(missing_members, rank)
        cluster_agg.loc[cluster_agg["top_categories"].isna(), "top_categories"] = (
            cluster_agg.loc[cluster_agg["top_categories"].isna(), "cluster"].map(cat_map)
        )

    # Z-score both to make them comparable
    cluster_agg["neural_z"] = (
        (cluster_agg["neural"] - cluster_agg["neural"].mean()) / cluster_agg["neural"].std()
    )
    cluster_agg["behavioral_z"] = (
        (cluster_agg["behavioral"] - cluster_agg["behavioral"].mean())
        / cluster_agg["behavioral"].std()
    )
    cluster_agg["dissociation"] = cluster_agg["neural_z"] - cluster_agg["behavioral_z"]

    cluster_agg = cluster_agg.sort_values("dissociation")
    out_csv = output_dir / f"cluster_dissociation_{monkey}.csv"
    cluster_agg.to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv} ({len(cluster_agg)} clusters)")

    r_cb, p_cb = stats.pearsonr(cluster_agg["neural"], cluster_agg["behavioral"])
    log.info(f"Cluster-level neural vs behavioral: r={r_cb:.3f}, p={p_cb:.1e}")

    # Plot: neural-z vs behavioral-z per cluster, labeled by semantics
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(7.5, 3.5))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.1, 1], wspace=0.75,
        left=0.08, right=0.98, top=0.95, bottom=0.14,
    )

    ax = fig.add_subplot(gs[0])
    sc = ax.scatter(
        cluster_agg["behavioral_z"], cluster_agg["neural_z"],
        c=cluster_agg["universality"], cmap="RdYlBu",
        s=10, alpha=0.6, edgecolors="white", linewidth=0.2,
    )
    ax.axhline(0, color=GRAY_LIGHT, lw=0.5, zorder=0)
    ax.axvline(0, color=GRAY_LIGHT, lw=0.5, zorder=0)
    # Tight axis limits: crop empty lower-left quadrant
    bz = cluster_agg["behavioral_z"]
    nz = cluster_agg["neural_z"]
    ax.set_xlim(bz.min() - 0.5, bz.max() + 0.5)
    ax.set_ylim(nz.min() - 0.5, nz.max() + 0.5)
    diag_lo = max(bz.min(), nz.min()) - 1
    diag_hi = max(bz.max(), nz.max()) + 1
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "--", color=GRAY, lw=0.6, zorder=0)
    ax.set_xlabel("Behavioral importance (z-scored)")
    ax.set_ylabel("Neural importance (z-scored)")
    ax.text(
        0.04, 0.96, f"r = {r_cb:.3f}, p = {p_cb:.1e}\nn = {len(cluster_agg)} clusters",
        transform=ax.transAxes, va="top", fontsize=6.5,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, aspect=30, shrink=0.8)
    cbar.set_label("Mean universality", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Label extremes
    label_idx = set()
    for col in ["dissociation", "neural_z", "behavioral_z"]:
        label_idx.update(cluster_agg.nlargest(4, col).index)
        label_idx.update(cluster_agg.nsmallest(4, col).index)
    for idx in label_idx:
        row = cluster_agg.loc[idx]
        cats = row["top_categories"]
        label = ", ".join(cats.split(", ")[:2]) if isinstance(cats, str) else ""
        if not label:
            continue
        ax.annotate(
            label, (row["behavioral_z"], row["neural_z"]),
            fontsize=5, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    # Right panel: top neural-specific vs behavior-specific clusters
    ax = fig.add_subplot(gs[1])
    n_show = 8
    neural_specific = cluster_agg.tail(n_show).copy()
    behav_specific = cluster_agg.head(n_show).copy()
    show = pd.concat([neural_specific, behav_specific])

    labels = []
    for _, row in show.iterrows():
        cats = row["top_categories"]
        short = ", ".join(cats.split(", ")[:3]) if isinstance(cats, str) else str(row["cluster"])
        labels.append(short)

    y_pos = np.arange(len(show))
    colors = [INDIGO] * n_show + [ROSE] * n_show
    ax.barh(y_pos, show["dissociation"], color=colors, height=0.6,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY_DARK, lw=0.8)
    ax.set_xlabel("Dissociation (neural z - behavioral z)")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=INDIGO, label="Neural-specific"),
            Patch(color=ROSE, label="Behavior-specific"),
        ],
        fontsize=6, loc="lower right", frameon=False,
    )

    save_figure(fig, output_dir / f"brain_behavior_dissociation_{monkey}")
    log.info(f"Saved brain_behavior_dissociation_{monkey}")


def analyze_brain_behavior(
    ablation_df: pd.DataFrame,
    monkey: str,
    output_dir: Path,
    rank: int,
    dataset: str,
) -> None:
    """Compare neural importance with behavioral importance per dimension."""
    behav_csv = Path(
        "experiments/universality/downstream_predictions/things_behavior"
        "/dimension_prediction/outputs/dimension_ablation.csv"
    )
    if not behav_csv.exists():
        log.warning("Behavioral ablation not found, skipping brain-behavior comparison")
        return

    behav = pd.read_csv(behav_csv)
    behav = behav.rename(columns={"importance": "behavioral_importance"})

    merged = ablation_df[["model", "dimension", "importance"]].rename(
        columns={"importance": "neural_importance"}
    ).merge(
        behav[["model", "dimension", "behavioral_importance"]],
        on=["model", "dimension"],
    )

    uni = load_universality(rank, dataset=dataset)
    merged = merged.merge(
        uni[["model", "dimension", "universality"]],
        on=["model", "dimension"],
    )

    out_csv = output_dir / f"brain_behavior_comparison_{monkey}.csv"
    merged.to_csv(out_csv, index=False)
    log.info(f"Saved {out_csv} ({len(merged)} dimensions)")

    r, p = stats.pearsonr(merged["neural_importance"], merged["behavioral_importance"])
    rho, rho_p = stats.spearmanr(merged["neural_importance"], merged["behavioral_importance"])
    log.info(f"Brain-behavior: r={r:.3f} p={p:.1e}, rho={rho:.3f} p={rho_p:.1e}")

    r_uni_neural, _ = stats.pearsonr(merged["universality"], merged["neural_importance"])
    r_uni_behav, _ = stats.pearsonr(merged["universality"], merged["behavioral_importance"])
    log.info(f"Universality -> neural importance: r={r_uni_neural:.3f}")
    log.info(f"Universality -> behavioral importance: r={r_uni_behav:.3f}")

    xlim = (
        merged["behavioral_importance"].quantile(0.005),
        merged["behavioral_importance"].quantile(0.995),
    )
    ylim = (
        merged["neural_importance"].quantile(0.005),
        merged["neural_importance"].quantile(0.995),
    )
    xmargin = (xlim[1] - xlim[0]) * 0.08
    ymargin = (ylim[1] - ylim[0]) * 0.08

    fig, ax = create_figure("square")
    sc = ax.scatter(
        merged["behavioral_importance"], merged["neural_importance"],
        c=merged["universality"], cmap="RdYlBu", s=4, alpha=0.15, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Universality", fontsize=8)

    z = np.polyfit(merged["behavioral_importance"], merged["neural_importance"], 1)
    x_line = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY, lw=1.5)

    ax.set_xlim(xlim[0] - xmargin, xlim[1] + xmargin)
    ax.set_ylim(ylim[0] - ymargin, ylim[1] + ymargin)
    ax.set_xlabel("Behavioral importance (triplet accuracy drop)")
    ax.set_ylabel("Neural importance (encoding drop)")
    ax.text(
        0.05, 0.95,
        f"r = {r:.3f}, p = {p:.1e}\nrho = {rho:.3f}\nn = {len(merged)}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    save_figure(fig, output_dir / f"brain_behavior_comparison_{monkey}")
    log.info(f"Saved brain_behavior_comparison_{monkey}")

    _analyze_dissociation(merged, monkey, output_dir, rank)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    uni_per_dim = load_universality(rank, dataset=cfg.dataset)

    components = load_components_for_rank(
        rank, models=list(cfg.models) or None, dataset=cfg.dataset,
    )
    components = {m: w for m, w in components.items() if not m.startswith("macaque")}
    log.info(f"Loaded {len(components)} models at rank {rank}")

    ablation_by_monkey: dict[str, pd.DataFrame] = {}
    subset_by_monkey: dict[str, pd.DataFrame] = {}

    for monkey in ("F", "N"):
        log.info(f"\nMonkey {monkey}: LOO dimension ablation")
        ablation_df = run_ablation(components, monkey, N_FOLDS, cfg.n_jobs, cfg.seed)
        ablation_df.to_csv(out / f"dimension_ablation_{monkey}.csv", index=False)
        log.info(f"Saved dimension_ablation_{monkey}.csv ({len(ablation_df)} rows)")

        log.info(f"\nMonkey {monkey}: subset comparison")
        subset_df = run_subsets(
            components, monkey, N_FOLDS, cfg.n_jobs, cfg.seed, uni_per_dim,
        )
        subset_df.to_csv(out / f"universal_vs_specific_subsets_{monkey}.csv", index=False)
        log.info(f"Saved universal_vs_specific_subsets_{monkey}.csv ({len(subset_df)} rows)")

        log.info(f"\nMonkey {monkey}: cumulative curve")
        cum_df = run_cumulative(
            components, monkey, N_FOLDS, cfg.n_jobs, cfg.seed, uni_per_dim,
        )
        cum_df.to_csv(out / f"cumulative_by_universality_order_{monkey}.csv", index=False)
        log.info(f"Saved cumulative_by_universality_order_{monkey}.csv ({len(cum_df)} rows)")

        mean_imp = ablation_df.groupby("model")["importance"].mean()
        log.info(f"Mean importance: {mean_imp.mean():.5f} (std={mean_imp.std():.5f})")

        merged = ablation_df.merge(
            uni_per_dim[["model", "dimension", "universality"]],
            on=["model", "dimension"],
        )
        r, p = stats.pearsonr(merged["universality"], merged["importance"])
        log.info(f"Universality vs neural importance (all dims): r={r:.3f}, p={p:.1e}")

        plot_importance_vs_universality(ablation_df, uni_per_dim, monkey, out)

        sub = subset_df.dropna()
        t, t_p = stats.ttest_rel(sub["mean_r_universal_half"], sub["mean_r_specific_half"])
        diff = (sub["mean_r_universal_half"] - sub["mean_r_specific_half"]).mean()
        log.info(f"Universal vs specific half: diff={diff:.5f}, t={t:.2f}, p={t_p:.1e}")
        plot_subset_comparison(subset_df, monkey, out)

        plot_cumulative_curve(cum_df, monkey, out)

        analyze_clusters(ablation_df, monkey, out, rank, cfg.dataset)
        analyze_brain_behavior(ablation_df, monkey, out, rank, cfg.dataset)

        ablation_by_monkey[monkey] = ablation_df
        subset_by_monkey[monkey] = subset_df

    # Combined monkey analysis: average importance across F and N
    log.info("\nCombined monkey analysis (mean of F and N)")
    combined = ablation_by_monkey["F"][["model", "dimension", "importance"]].merge(
        ablation_by_monkey["N"][["model", "dimension", "importance"]],
        on=["model", "dimension"], suffixes=("_F", "_N"),
    )
    combined["importance"] = (combined["importance_F"] + combined["importance_N"]) / 2
    combined["mean_r_full"] = np.nan
    combined["mean_r_ablated"] = np.nan
    combined.to_csv(out / "dimension_ablation_combined.csv", index=False)
    log.info(f"Saved dimension_ablation_combined.csv ({len(combined)} rows)")

    merged = combined.merge(
        uni_per_dim[["model", "dimension", "universality"]],
        on=["model", "dimension"],
    )
    r, p = stats.pearsonr(merged["universality"], merged["importance"])
    log.info(f"Combined universality vs importance (all dims): r={r:.3f}, p={p:.1e}")

    plot_importance_vs_universality(combined, uni_per_dim, "combined", out)
    analyze_clusters(combined, "combined", out, rank, cfg.dataset)
    analyze_brain_behavior(combined, "combined", out, rank, cfg.dataset)

    # Combined subset: average across monkeys
    combined_sub = subset_by_monkey["F"].merge(
        subset_by_monkey["N"], on="model", suffixes=("_F", "_N"),
    )
    combined_sub["mean_r_universal_half"] = (
        combined_sub["mean_r_universal_half_F"] + combined_sub["mean_r_universal_half_N"]
    ) / 2
    combined_sub["mean_r_specific_half"] = (
        combined_sub["mean_r_specific_half_F"] + combined_sub["mean_r_specific_half_N"]
    ) / 2
    combined_sub["mean_r_full"] = (
        combined_sub["mean_r_full_F"] + combined_sub["mean_r_full_N"]
    ) / 2
    t, t_p = stats.ttest_rel(
        combined_sub["mean_r_universal_half"], combined_sub["mean_r_specific_half"],
    )
    diff = (
        combined_sub["mean_r_universal_half"] - combined_sub["mean_r_specific_half"]
    ).mean()
    log.info(f"Combined universal vs specific: diff={diff:.5f}, t={t:.2f}, p={t_p:.1e}")
    plot_subset_comparison(combined_sub, "combined", out)


if __name__ == "__main__":
    main()
