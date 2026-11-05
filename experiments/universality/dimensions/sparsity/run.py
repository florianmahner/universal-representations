#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats as sp_stats

from experiments.universality.shared import (
    load_components_for_rank,
    load_model_info,
    load_universality,
)
from src.colors import GRAY, INDIGO, ROSE
from src.plotting import CLASS_COLORS, create_figure, save_figure, setup_plotting


def hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer's sparsity: 0 = perfectly dense, 1 = perfectly sparse."""
    n = len(x)
    l1 = np.abs(x).sum()
    l2 = np.sqrt((x ** 2).sum())
    if l2 == 0:
        return 0.0
    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient of absolute values. 0 = equal, 1 = maximal inequality."""
    a = np.abs(x)
    if a.sum() == 0:
        return 0.0
    sorted_a = np.sort(a)
    n = len(sorted_a)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_a) / (n * np.sum(sorted_a))) - (n + 1) / n


def tail_weight_ratio(x: np.ndarray, q: float = 0.95) -> float:
    """Fraction of total L1 mass in the top (1-q) fraction of entries."""
    a = np.abs(x)
    total = a.sum()
    if total == 0:
        return 0.0
    threshold = np.quantile(a, q)
    return a[a >= threshold].sum() / total


def component_variance(x: np.ndarray) -> float:
    """Variance of component loadings (proxy for signal strength)."""
    return float(np.var(x))


def component_l2(x: np.ndarray) -> float:
    """L2 norm of component."""
    return float(np.sqrt((x ** 2).sum()))


# The distributional statistics this analysis relates to universality; the last
# two are the signal-strength controls partialled out in Part 3.
MEASURES = {
    "hoyer_sparsity": hoyer_sparsity,
    "gini": gini_coefficient,
    "kurtosis": lambda x: float(sp_stats.kurtosis(x, fisher=True)),
    "skewness": lambda x: float(sp_stats.skew(x)),
    "tail_weight_95": tail_weight_ratio,
    "variance": component_variance,
    "l2_norm": component_l2,
}


def load_data(rank: int, dataset: str, models: list[str] | None = None) -> pd.DataFrame:
    """Load universality scores and compute distributional measures for all components."""
    universality_df = load_universality(rank=rank, dataset=dataset)
    models = list(models or []) or sorted(universality_df["model"].unique())

    print(f"Computing distributional measures for {len(models)} models...")
    rows = []
    missing_models = []
    for model_idx, m in enumerate(models, start=1):
        # Stream one model at a time. Keeping every 22k x 50 matrix resident
        # simultaneously creates a multi-gigabyte memory peak.
        components = load_components_for_rank(rank=rank, models=[m], dataset=dataset)
        if m not in components:
            missing_models.append(m)
            continue
        W = components[m]
        for dim in range(W.shape[1]):
            col = W[:, dim]
            row = {"model": m, "dimension": dim}
            for name, func in MEASURES.items():
                row[name] = func(col)
            rows.append(row)
        if model_idx % 20 == 0 or model_idx == len(models):
            print(f"  processed {model_idx}/{len(models)} models")

    if missing_models:
        raise FileNotFoundError(
            "Missing selected components for: " + ", ".join(missing_models)
        )

    measures_df = pd.DataFrame(rows)
    df = measures_df.merge(universality_df, on=["model", "dimension"], how="inner")

    print(
        f"  {len(df)} dimension-model pairs across "
        f"{df['model'].nunique()} loaded models"
    )
    return df


def load_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Merge model metadata (architecture, objective, etc.) onto the dataframe."""
    mi = load_model_info()
    meta_cols = ["name", "model_class", "train_task", "train_data"]
    meta = mi[meta_cols].rename(columns={"name": "model"})
    return df.merge(meta, on="model", how="left")


def correlate_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations between each distributional measure and universality."""
    measure_names = [m for m in MEASURES if m not in ("variance", "l2_norm")]
    rows = []
    for m in measure_names:
        r, p = sp_stats.spearmanr(df[m], df["universality"])
        rows.append({"measure": m, "spearman_r": r, "p_value": p, "n": len(df)})
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False, key=abs)


def per_model_correlations(df: pd.DataFrame, measure: str = "hoyer_sparsity") -> np.ndarray:
    """Compute within-model Spearman r between a measure and universality."""
    rs = []
    for _, grp in df.groupby("model"):
        r, _ = sp_stats.spearmanr(grp[measure], grp["universality"])
        rs.append(r)
    return np.array(rs)


def partial_correlation(
    df: pd.DataFrame, x_col: str, y_col: str, covariates: list[str]
) -> tuple[float, float]:
    """Partial Spearman correlation between x and y controlling for covariates."""
    def _residualize(target, controls, data):
        X = data[controls].values
        X = np.column_stack([np.ones(len(X)), X])
        y = data[target].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    res_x = _residualize(x_col, covariates, df)
    res_y = _residualize(y_col, covariates, df)
    r, p = sp_stats.spearmanr(res_x, res_y)
    return r, p


def binned_analysis(
    df: pd.DataFrame, measure: str, n_bins: int = 10
) -> pd.DataFrame:
    """Compute mean universality in quantile bins of a distributional measure."""
    df = df.copy()
    df["bin"] = pd.qcut(df[measure], n_bins, labels=False, duplicates="drop")
    edges = pd.qcut(df[measure], n_bins, duplicates="drop").cat.categories
    centers = [(e.left + e.right) / 2 for e in edges]

    binned = df.groupby("bin")["universality"].agg(["mean", "std", "count"])
    binned["se"] = binned["std"] / np.sqrt(binned["count"])
    binned["center"] = centers[: len(binned)]
    return binned.reset_index()


def model_level_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to model-level means."""
    measure_cols = list(MEASURES.keys())
    agg = {m: "mean" for m in measure_cols}
    agg["universality"] = "mean"
    return df.groupby("model").agg(agg).reset_index()


def metadata_interaction_analysis(
    df: pd.DataFrame, measure: str = "hoyer_sparsity"
) -> pd.DataFrame:
    """Compute sparsity-universality correlation per architecture family."""
    rows = []
    for group, grp in df.groupby("model_class"):
        if len(grp) < 20 or not group or group == "":
            continue
        r, p = sp_stats.spearmanr(grp[measure], grp["universality"])
        n_models = grp["model"].nunique()
        rows.append({
            "group": group,
            "spearman_r": r,
            "p_value": p,
            "n_dimensions": len(grp),
            "n_models": n_models,
        })
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False, key=abs)


def model_level_metadata_interaction(
    model_df: pd.DataFrame, measure: str = "hoyer_sparsity"
) -> pd.DataFrame:
    """Model-level correlation within each architecture family."""
    mi = load_model_info()
    meta = mi[["name", "model_class"]].rename(columns={"name": "model"})
    mdf = model_df.merge(meta, on="model", how="left")

    rows = []
    for group, grp in mdf.groupby("model_class"):
        if len(grp) < 5:
            continue
        r, p = sp_stats.spearmanr(grp[measure], grp["universality"])
        rows.append({
            "group": group,
            "spearman_r": r,
            "p_value": p,
            "n_models": len(grp),
        })
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False, key=abs)


def robustness_taskonomy(
    df: pd.DataFrame,
    model_df: pd.DataFrame,
    measure: str = "hoyer_sparsity",
) -> dict:
    """Robustness check: CNN correlations with and without Taskonomy models."""
    mi = load_model_info()
    meta = mi[["name", "model_class", "train_task"]].rename(columns={"name": "model"})

    # Dimension-level
    df_m = df.merge(meta, on="model", how="left")
    cnns_dim = df_m[df_m["model_class"] == "Convolutional"]
    is_taskonomy_dim = cnns_dim["train_task"].str.contains("taskonomy", na=False)

    r_all_dim, p_all_dim = sp_stats.spearmanr(cnns_dim[measure], cnns_dim["universality"])
    r_notask_dim, p_notask_dim = sp_stats.spearmanr(
        cnns_dim.loc[~is_taskonomy_dim, measure],
        cnns_dim.loc[~is_taskonomy_dim, "universality"],
    )
    r_task_dim, p_task_dim = sp_stats.spearmanr(
        cnns_dim.loc[is_taskonomy_dim, measure],
        cnns_dim.loc[is_taskonomy_dim, "universality"],
    )

    # Model-level
    mdf = model_df.merge(meta, on="model", how="left")
    cnns_mod = mdf[mdf["model_class"] == "Convolutional"]
    is_taskonomy_mod = cnns_mod["train_task"].str.contains("taskonomy", na=False)

    r_all_mod, p_all_mod = sp_stats.spearmanr(cnns_mod[measure], cnns_mod["universality"])
    r_notask_mod, p_notask_mod = sp_stats.spearmanr(
        cnns_mod.loc[~is_taskonomy_mod, measure],
        cnns_mod.loc[~is_taskonomy_mod, "universality"],
    )
    r_task_mod, p_task_mod = sp_stats.spearmanr(
        cnns_mod.loc[is_taskonomy_mod, measure],
        cnns_mod.loc[is_taskonomy_mod, "universality"],
    )

    return {
        "dimension_level": {
            "all_cnns": {"r": r_all_dim, "p": p_all_dim, "n": len(cnns_dim)},
            "excl_taskonomy": {"r": r_notask_dim, "p": p_notask_dim, "n": int((~is_taskonomy_dim).sum())},
            "taskonomy_only": {"r": r_task_dim, "p": p_task_dim, "n": int(is_taskonomy_dim.sum())},
        },
        "model_level": {
            "all_cnns": {"r": r_all_mod, "p": p_all_mod, "n": len(cnns_mod)},
            "excl_taskonomy": {"r": r_notask_mod, "p": p_notask_mod, "n": int((~is_taskonomy_mod).sum())},
            "taskonomy_only": {"r": r_task_mod, "p": p_task_mod, "n": int(is_taskonomy_mod.sum())},
        },
        "n_taskonomy": int(is_taskonomy_mod.sum()),
        "n_other_cnn": int((~is_taskonomy_mod).sum()),
        # Store dataframes for plotting
        "_cnns_mod": cnns_mod,
        "_is_taskonomy_mod": is_taskonomy_mod,
    }


def plot_measure_correlation_overview(corr_df: pd.DataFrame, out: Path):
    """Bar chart of Spearman r for each distributional measure vs universality."""
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = [ROSE if p < 0.05 else GRAY for p in corr_df["p_value"]]
    ax.barh(corr_df["measure"], corr_df["spearman_r"], color=colors, edgecolor="white")
    ax.set_xlabel("Spearman r with universality")
    ax.set_title("Distributional measures vs. universality (dimension-level)")
    ax.axvline(0, color="black", linewidth=0.5)
    for i, (_, row) in enumerate(corr_df.iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
        ax.text(
            row["spearman_r"] + 0.005 * np.sign(row["spearman_r"]),
            i, sig, va="center", fontsize=7,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, out / "measure_overview", formats=("pdf", "png"))


def plot_pooled_scatter(df: pd.DataFrame, out: Path, measure: str = "hoyer_sparsity"):
    """Pooled scatter of a measure vs universality with LOWESS trend."""
    r, p = sp_stats.spearmanr(df[measure], df["universality"])

    fig, ax = create_figure("single")
    ax.scatter(
        df[measure], df["universality"],
        s=4, alpha=0.15, color=GRAY, rasterized=True,
    )
    sns.regplot(
        x=measure, y="universality", data=df,
        scatter=False, lowess=True, color=ROSE,
        ax=ax, line_kws={"linewidth": 2},
    )
    ax.set_xlabel(measure.replace("_", " ").title())
    ax.set_ylabel("Universality")
    ax.set_title(f"All dimensions (n={len(df):,})")
    ax.annotate(
        f"Spearman r = {r:.3f}\np = {p:.1e}",
        xy=(0.03, 0.95), xycoords="axes fraction",
        fontsize=7, va="top", ha="left",
    )
    save_figure(fig, out / f"scatter_{measure}", formats=("pdf", "png"))


def plot_multi_scatter(df: pd.DataFrame, out: Path):
    """Grid of scatter plots for all distributional measures vs universality."""
    measure_names = [m for m in MEASURES if m not in ("variance", "l2_norm")]
    n = len(measure_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, m in enumerate(measure_names):
        ax = axes[i]
        r, p = sp_stats.spearmanr(df[m], df["universality"])
        ax.scatter(
            df[m], df["universality"],
            s=3, alpha=0.1, color=GRAY, rasterized=True,
        )
        sns.regplot(
            x=m, y="universality", data=df,
            scatter=False, lowess=False, color=ROSE,
            ax=ax, line_kws={"linewidth": 2},
        )
        ax.set_xlabel(m.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel("Universality", fontsize=8)
        ax.annotate(
            f"r = {r:.3f}, p = {p:.1e}",
            xy=(0.03, 0.95), xycoords="axes fraction",
            fontsize=7, va="top", ha="left",
        )
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributional measures vs. universality", fontsize=10, y=1.01)
    fig.tight_layout()
    save_figure(fig, out / "scatter_grid", formats=("pdf", "png"))


def plot_per_model_histogram(per_model_rs: np.ndarray, out: Path, measure: str = "hoyer_sparsity"):
    """Histogram of within-model Spearman correlations."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(per_model_rs, bins=30, color=INDIGO, edgecolor="white", linewidth=0.3)
    ax.axvline(
        np.median(per_model_rs), color=ROSE, linewidth=1.5,
        linestyle="--", label=f"Median = {np.median(per_model_rs):.3f}",
    )
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(f"Spearman r ({measure.replace('_', ' ')} vs universality)")
    ax.set_ylabel("Number of models")
    ax.set_title("Within-model correlation")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, out / f"per_model_hist_{measure}", formats=("pdf", "png"))


def plot_binned_means(binned: pd.DataFrame, out: Path, measure: str = "hoyer_sparsity"):
    """Binned mean universality with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.errorbar(
        binned["center"], binned["mean"],
        yerr=1.96 * binned["se"],
        fmt="o-", markersize=5, color=INDIGO,
        capsize=3, linewidth=1.2, capthick=0.8,
    )
    ax.set_xlabel(f"{measure.replace('_', ' ').title()} (bin center)")
    ax.set_ylabel("Mean universality")
    ax.set_title("Universality by sparsity decile (95% CI)")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, out / f"binned_{measure}", formats=("pdf", "png"))


def plot_model_level_scatter(model_df: pd.DataFrame, out: Path, measure: str = "hoyer_sparsity"):
    """Model-level scatter of mean measure vs mean universality."""
    r, p = sp_stats.spearmanr(model_df[measure], model_df["universality"])

    fig, ax = create_figure("single")
    ax.scatter(
        model_df[measure], model_df["universality"],
        s=14, alpha=0.6, color=INDIGO,
    )
    sns.regplot(
        x=measure, y="universality", data=model_df,
        scatter=False, lowess=True, color=ROSE,
        ax=ax, line_kws={"linewidth": 2},
    )
    ax.set_xlabel(f"Mean {measure.replace('_', ' ')}")
    ax.set_ylabel("Mean universality")
    ax.set_title(f"Model-level (n={len(model_df)})")
    ax.annotate(
        f"Spearman r = {r:.3f}\np = {p:.1e}",
        xy=(0.03, 0.95), xycoords="axes fraction",
        fontsize=7, va="top", ha="left",
    )
    save_figure(fig, out / f"model_level_{measure}", formats=("pdf", "png"))


def plot_model_level_by_architecture(model_df: pd.DataFrame, out: Path, measure: str = "hoyer_sparsity"):
    """Model-level scatter colored by architecture family."""
    mi = load_model_info()
    meta = mi[["name", "model_class"]].rename(columns={"name": "model"})
    mdf = model_df.merge(meta, on="model", how="left")
    mdf = mdf[mdf["model_class"].isin(CLASS_COLORS)]

    fig, ax = plt.subplots(figsize=(5, 4))
    for arch, color in CLASS_COLORS.items():
        sub = mdf[mdf["model_class"] == arch]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[measure], sub["universality"],
            s=18, alpha=0.6, color=color, label=f"{arch} (n={len(sub)})",
        )
    ax.set_xlabel(f"Mean {measure.replace('_', ' ')}")
    ax.set_ylabel("Mean universality")
    ax.set_title("Model-level by architecture family")
    ax.legend(fontsize=7, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, out / "model_level_by_architecture", formats=("pdf", "png"))


def plot_metadata_interaction(
    df: pd.DataFrame, out: Path, measure: str = "hoyer_sparsity"
):
    """Per-architecture scatter + LOWESS of measure vs universality."""
    families = [f for f in CLASS_COLORS if f in df["model_class"].unique()]

    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 3.5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        sub = df[df["model_class"] == fam]
        r, p = sp_stats.spearmanr(sub[measure], sub["universality"])
        ax.scatter(
            sub[measure], sub["universality"],
            s=4, alpha=0.15, color=GRAY, rasterized=True,
        )
        sns.regplot(
            x=measure, y="universality", data=sub,
            scatter=False, lowess=True, color=CLASS_COLORS[fam],
            ax=ax, line_kws={"linewidth": 2},
        )
        ax.set_xlabel(measure.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel("Universality" if ax == axes[0] else "", fontsize=8)
        ax.set_title(f"{fam} (n={sub['model'].nunique()} models)", fontsize=9)
        ax.annotate(
            f"r = {r:.3f}\np = {p:.1e}",
            xy=(0.03, 0.95), xycoords="axes fraction",
            fontsize=7, va="top", ha="left",
        )
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)

    fig.tight_layout()
    save_figure(fig, out / "interaction_architecture", formats=("pdf", "png"))


def plot_partial_vs_raw(
    raw_corrs: pd.DataFrame, partial_results: list[dict], out: Path
):
    """Compare raw vs partial correlations for key measures."""
    partial_df = pd.DataFrame(partial_results)
    merged = raw_corrs.merge(partial_df, on="measure")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(merged))
    w = 0.35
    ax.barh(x - w / 2, merged["spearman_r"], w, label="Raw", color=INDIGO)
    ax.barh(x + w / 2, merged["spearman_r_partial"], w, label="Partial (controlling variance + L2)", color=ROSE)
    ax.set_yticks(x)
    ax.set_yticklabels(merged["measure"])
    ax.set_xlabel("Spearman r with universality")
    ax.set_title("Raw vs. partial correlations")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, out / "partial_vs_raw", formats=("pdf", "png"))


def plot_robustness_taskonomy(robustness: dict, out: Path, measure: str = "hoyer_sparsity"):
    """Scatter of CNN models: Taskonomy highlighted as distinct group."""
    cnns = robustness["_cnns_mod"]
    is_task = robustness["_is_taskonomy_mod"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: all CNNs with Taskonomy highlighted
    ax = axes[0]
    ax.scatter(
        cnns.loc[~is_task, measure], cnns.loc[~is_task, "universality"],
        s=18, alpha=0.6, color=INDIGO, label=f"Other CNNs (n={robustness['n_other_cnn']})",
    )
    ax.scatter(
        cnns.loc[is_task, measure], cnns.loc[is_task, "universality"],
        s=18, alpha=0.6, color=ROSE, marker="^",
        label=f"Taskonomy (n={robustness['n_taskonomy']})",
    )
    r_all = robustness["model_level"]["all_cnns"]
    ax.set_xlabel(f"Mean {measure.replace('_', ' ')}")
    ax.set_ylabel("Mean universality")
    ax.set_title(f"All CNNs (r = {r_all['r']:.3f})")
    ax.legend(fontsize=7, loc="upper right")
    sns.despine(ax=ax)

    # Right: CNNs excluding Taskonomy with trend
    ax = axes[1]
    other = cnns[~is_task]
    ax.scatter(
        other[measure], other["universality"],
        s=18, alpha=0.6, color=INDIGO,
    )
    sns.regplot(
        x=measure, y="universality", data=other,
        scatter=False, lowess=True, color=INDIGO,
        ax=ax, line_kws={"linewidth": 2},
    )
    r_notask = robustness["model_level"]["excl_taskonomy"]
    ax.set_xlabel(f"Mean {measure.replace('_', ' ')}")
    ax.set_ylabel("Mean universality")
    ax.set_title(f"Excl. Taskonomy (r = {r_notask['r']:.3f}, p = {r_notask['p']:.4f})")
    sns.despine(ax=ax)

    fig.suptitle("Robustness: CNN model-level correlation", fontsize=10, y=1.02)
    fig.tight_layout()
    save_figure(fig, out / "robustness_taskonomy", formats=("pdf", "png"))


def plot_example_distributions(df: pd.DataFrame, out: Path, rank: int, dataset: str):
    """Show loading distributions at different sparsity levels."""
    quantiles = {
        "Very sparse\n(95th pct)": 0.95,
        "Median\nsparsity": 0.50,
        "Dense\n(5th pct)": 0.05,
    }
    examples = {}
    for label, q in quantiles.items():
        target = df["hoyer_sparsity"].quantile(q)
        idx = (df["hoyer_sparsity"] - target).abs().idxmin()
        examples[label] = df.loc[idx]

    outlier_idx = df["hoyer_sparsity"].idxmax()
    examples["Most sparse\n(outlier)"] = df.loc[outlier_idx]

    example_models = sorted({str(row["model"]) for row in examples.values()})
    components = load_components_for_rank(rank=rank, models=example_models, dataset=dataset)

    fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 2.5))
    for ax, (label, row) in zip(axes, examples.items()):
        W = components[row["model"]]
        loadings = W[:, int(row["dimension"])]
        sorted_loadings = np.sort(loadings)[::-1]

        ax.bar(
            range(len(sorted_loadings)), sorted_loadings,
            width=1.0, color=INDIGO, linewidth=0,
        )
        ax.set_title(
            f"{label}\n(s={row['hoyer_sparsity']:.2f}, u={row['universality']:.2f})",
            fontsize=7,
        )
        ax.set_xlabel("Image (sorted)", fontsize=7)
        ax.set_ylabel("Loading", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, len(sorted_loadings))
        sns.despine(ax=ax)

    fig.tight_layout()
    save_figure(fig, out / "example_distributions", formats=("pdf", "png"))
    return examples


def plot_measure_intercorrelation(df: pd.DataFrame, out: Path):
    """Correlation matrix between distributional measures."""
    measure_cols = [m for m in MEASURES]
    corr = df[measure_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax, square=True,
        annot_kws={"fontsize": 7},
        xticklabels=[m.replace("_", "\n") for m in measure_cols],
        yticklabels=[m.replace("_", " ") for m in measure_cols],
    )
    ax.set_title("Inter-correlation of distributional measures")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(fig, out / "measure_intercorrelation", formats=("pdf", "png"))


def plot_nonlinearity(df: pd.DataFrame, out: Path, measure: str = "hoyer_sparsity", n_bins: int = 20):
    """Fine-grained binned plot to reveal nonlinear relationships."""
    binned = binned_analysis(df, measure, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(
        binned["center"], binned["mean"],
        yerr=1.96 * binned["se"],
        fmt="o-", markersize=4, color=INDIGO,
        capsize=2, linewidth=1, capthick=0.6,
    )
    ax.set_xlabel(f"{measure.replace('_', ' ').title()}")
    ax.set_ylabel("Mean universality")
    ax.set_title(f"Universality by {measure.replace('_', ' ')} (20 bins, 95% CI)")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, out / f"nonlinearity_{measure}", formats=("pdf", "png"))


def generate_report(
    df: pd.DataFrame,
    model_df: pd.DataFrame,
    corr_overview: pd.DataFrame,
    per_model_rs: np.ndarray,
    partial_results: list[dict],
    dim_interaction: pd.DataFrame,
    model_interaction: pd.DataFrame,
    robustness: dict,
    examples: dict,
    rank: int,
) -> str:
    """Generate the full markdown report."""
    n_models = df["model"].nunique()
    n_dims = len(df)
    L = []  # lines

    # Title
    L.append("# Score Distribution and Universality")
    L.append("")
    L.append("## Overview")
    L.append("")
    L.append(
        f"This report examines how the distributional properties of NMF component loadings "
        f"relate to universality across {n_models} deep neural network models. "
        f"Each model is factorized into {rank} NMF components, yielding {n_dims:,} "
        f"dimension-model pairs. We ask: **what kind of loading distributions produce "
        f"components that are shared across models?**"
    )
    L.append("")
    L.append("We analyze several distributional measures:")
    L.append("")
    L.append("- **Hoyer sparsity**: Ratio of L1/L2 norms, normalized to [0, 1]. Higher = sparser.")
    L.append("- **Gini coefficient**: Inequality of absolute loadings. Higher = more concentrated.")
    L.append("- **Kurtosis**: Tailedness of the distribution (Fisher). Higher = heavier tails.")
    L.append("- **Skewness**: Asymmetry of the distribution. Positive = right-skewed (few high values).")
    L.append("- **Tail weight (95th pct)**: Fraction of total L1 mass in the top 5% of entries.")
    L.append("")
    L.append("We also include **variance** and **L2 norm** as control variables to separate "
             "distributional shape from signal strength.")
    L.append("")

    # Part 1: Measure overview
    L.append("---")
    L.append("")
    L.append("## Part 1: Which Distributional Properties Predict Universality?")
    L.append("")
    L.append("### Correlation Overview")
    L.append("")
    L.append("Spearman correlations between each distributional measure and universality, "
             "pooled across all dimension-model pairs:")
    L.append("")
    L.append("| Measure | Spearman r | p | n |")
    L.append("|---------|-----------|---|---|")
    for _, row in corr_overview.iterrows():
        p_str = f"{row['p_value']:.1e}" if row["p_value"] < 0.0001 else f"{row['p_value']:.4f}"
        L.append(f"| {row['measure']} | {row['spearman_r']:.3f} | {p_str} | {row['n']:,} |")
    L.append("")
    L.append("![Measure Overview](outputs/measure_overview.png)")
    L.append("")

    # Inter-correlation
    L.append("### Inter-correlation of Measures")
    L.append("")
    L.append("Many of these measures are highly correlated (e.g., Hoyer sparsity and Gini both "
             "capture concentration). The heatmap below shows their Spearman inter-correlations:")
    L.append("")
    L.append("![Measure Intercorrelation](outputs/measure_intercorrelation.png)")
    L.append("")

    # Scatter grid
    L.append("### Scatter Plots")
    L.append("")
    L.append("![Scatter Grid](outputs/scatter_grid.png)")
    L.append("")

    # Part 2: Sparsity deep-dive
    L.append("---")
    L.append("")
    L.append("## Part 2: Sparsity and Universality (Detailed)")
    L.append("")
    L.append("We focus on Hoyer sparsity as the primary measure, as it has the clearest "
             "interpretation and is widely used in the NMF literature.")
    L.append("")

    # Pooled
    r_pooled, p_pooled = sp_stats.spearmanr(df["hoyer_sparsity"], df["universality"])
    L.append("### Dimension-Level (Pooled)")
    L.append("")
    L.append(f"Across all {n_dims:,} dimension-model pairs, Hoyer sparsity and universality "
             f"show a Spearman correlation of **r = {r_pooled:.3f}** (p = {p_pooled:.1e}).")
    L.append("")
    L.append("![Pooled Scatter](outputs/scatter_hoyer_sparsity.png)")
    L.append("")

    # Within-model
    med_r = np.median(per_model_rs)
    mean_r = np.mean(per_model_rs)
    frac_pos = (per_model_rs > 0).mean()
    L.append("### Within-Model Consistency")
    L.append("")
    L.append(
        f"The pooled correlation could be driven by between-model differences. To test whether "
        f"the relationship holds *within* individual models, we compute per-model Spearman "
        f"correlations (across {rank} dimensions each)."
    )
    L.append("")
    L.append(f"- Median r = {med_r:.3f}")
    L.append(f"- Mean r = {mean_r:.3f}")
    L.append(f"- Fraction of models with positive r: {frac_pos:.1%}")
    L.append("")
    L.append("![Per-Model Histogram](outputs/per_model_hist_hoyer_sparsity.png)")
    L.append("")

    # Binned
    L.append("### Binned Analysis (Nonlinearity)")
    L.append("")
    L.append("To probe the functional form of the relationship, we bin dimensions by sparsity "
             "decile and compute mean universality in each bin:")
    L.append("")
    L.append("![Binned Means](outputs/binned_hoyer_sparsity.png)")
    L.append("")
    L.append("A finer-grained (20-bin) version reveals any threshold or plateau effects:")
    L.append("")
    L.append("![Nonlinearity](outputs/nonlinearity_hoyer_sparsity.png)")
    L.append("")

    # Model-level
    r_model, p_model = sp_stats.spearmanr(
        model_df["hoyer_sparsity"], model_df["universality"]
    )
    L.append("### Model-Level Aggregation")
    L.append("")
    L.append(
        f"Aggregating to model-level means (one point per model), the Spearman correlation "
        f"between mean sparsity and mean universality is **r = {r_model:.3f}** "
        f"(p = {p_model:.1e}, n = {len(model_df)})."
    )
    L.append("")
    L.append("![Model-Level Scatter](outputs/model_level_hoyer_sparsity.png)")
    L.append("")

    # Part 3: Controlling for signal strength
    L.append("---")
    L.append("")
    L.append("## Part 3: Controlling for Signal Strength")
    L.append("")
    L.append(
        "A sparse component might be more universal simply because it captures a strong, "
        "well-defined signal. To probe this possibility, we compute partial correlations between each "
        "distributional measure and universality, controlling for component variance and L2 norm."
    )
    L.append("")
    L.append("| Measure | Raw r | Partial r | Partial p |")
    L.append("|---------|-------|-----------|-----------|")
    partial_df = pd.DataFrame(partial_results)
    for _, prow in partial_df.iterrows():
        raw_r = corr_overview.loc[corr_overview["measure"] == prow["measure"], "spearman_r"].values[0]
        p_str = f"{prow['p_value']:.1e}" if prow["p_value"] < 0.0001 else f"{prow['p_value']:.4f}"
        L.append(f"| {prow['measure']} | {raw_r:.3f} | {prow['spearman_r_partial']:.3f} | {p_str} |")
    L.append("")
    L.append("![Partial vs Raw](outputs/partial_vs_raw.png)")
    L.append("")

    # Part 4: Metadata interactions
    L.append("---")
    L.append("")
    L.append("## Part 4: Interaction with Model Properties")
    L.append("")
    L.append(
        "Does the sparsity-universality relationship depend on model architecture? "
        "We compute separate correlations within each architecture family."
    )
    L.append("")

    # Dimension-level by architecture
    L.append("### Dimension-Level by Architecture")
    L.append("")
    L.append("| Architecture | Spearman r | p | n dimensions | n models |")
    L.append("|-------------|-----------|---|-------------|----------|")
    for _, row in dim_interaction.iterrows():
        p_str = f"{row['p_value']:.1e}" if row["p_value"] < 0.0001 else f"{row['p_value']:.4f}"
        L.append(
            f"| {row['group']} | {row['spearman_r']:.3f} | {p_str} | "
            f"{row['n_dimensions']:,} | {row['n_models']} |"
        )
    L.append("")
    L.append("![Interaction by Architecture](outputs/interaction_architecture.png)")
    L.append("")

    # Model-level by architecture
    L.append("### Model-Level by Architecture")
    L.append("")
    L.append("| Architecture | Spearman r | p | n models |")
    L.append("|-------------|-----------|---|----------|")
    for _, row in model_interaction.iterrows():
        p_str = f"{row['p_value']:.1e}" if row["p_value"] < 0.0001 else f"{row['p_value']:.4f}"
        L.append(f"| {row['group']} | {row['spearman_r']:.3f} | {p_str} | {row['n_models']} |")
    L.append("")
    L.append("![Model-Level by Architecture](outputs/model_level_by_architecture.png)")
    L.append("")

    # Part 4b: Robustness — Taskonomy models
    L.append("### Robustness: Taskonomy Models")
    L.append("")
    L.append(
        f"We test whether the {robustness['n_taskonomy']} Taskonomy models account for the CNN "
        "association by reporting CNN correlations with and without them. Taskonomy models are "
        "ResNet-50s trained on non-standard visual tasks such as autoencoding, denoising, and "
        "depth estimation."
    )
    L.append("")

    dim_stats = robustness["dimension_level"]
    mod_stats = robustness["model_level"]

    L.append("| CNN subset | Dimension-level r | Model-level r | n models |")
    L.append("|-----------|------------------|--------------|----------|")
    for label, dk, mk in [
        ("All CNNs", "all_cnns", "all_cnns"),
        ("Excl. Taskonomy", "excl_taskonomy", "excl_taskonomy"),
        ("Taskonomy only", "taskonomy_only", "taskonomy_only"),
    ]:
        d = dim_stats[dk]
        m = mod_stats[mk]
        d_p = f"p = {d['p']:.1e}" if d['p'] < 0.0001 else f"p = {d['p']:.4f}"
        m_p = f"p = {m['p']:.1e}" if m['p'] < 0.0001 else f"p = {m['p']:.4f}"
        L.append(
            f"| {label} | {d['r']:.3f} ({d_p}) | {m['r']:.3f} ({m_p}) | {m['n']} |"
        )
    L.append("")
    cnn_all = mod_stats["all_cnns"]
    cnn_excl = mod_stats["excl_taskonomy"]
    excl_direction = "positive" if cnn_excl["r"] > 0 else "negative"
    excl_significance = (
        "statistically significant"
        if cnn_excl["p"] < 0.05
        else "not statistically significant"
    )
    L.append(
        f"Excluding Taskonomy changes the CNN model-level correlation from r = {cnn_all['r']:.3f} "
        f"to r = {cnn_excl['r']:.3f}. The remaining association is {excl_direction} and "
        f"{excl_significance} (p = {cnn_excl['p']:.4f}). Thus Taskonomy strengthens the overall "
        "CNN pattern, but the exclusion test does not justify attributing the full association "
        "to Taskonomy alone."
    )
    L.append("")
    L.append("![Robustness: Taskonomy](outputs/robustness_taskonomy.png)")
    L.append("")

    # Part 5: Examples
    L.append("---")
    L.append("")
    L.append("## Part 5: Example Loading Distributions")
    L.append("")
    L.append("To build intuition, we show sorted loading distributions for components at "
             "different sparsity levels:")
    L.append("")
    for label, row in examples.items():
        L.append(
            f"- **{label.replace(chr(10), ' ')}**: model={row['model']}, dim={int(row['dimension'])}, "
            f"sparsity={row['hoyer_sparsity']:.3f}, universality={row['universality']:.3f}"
        )
    L.append("")
    L.append("![Example Distributions](outputs/example_distributions.png)")
    L.append("")

    # Discussion
    L.append("---")
    L.append("")
    L.append("## Discussion")
    L.append("")

    partial_df_disc = pd.DataFrame(partial_results)

    strongest_raw = corr_overview.loc[corr_overview["spearman_r"].abs().idxmax()]
    strongest_partial = partial_df_disc.loc[
        partial_df_disc["spearman_r_partial"].abs().idxmax()
    ]
    n_significant_raw = int((corr_overview["p_value"] < 0.05).sum())

    L.append("### Key Findings")
    L.append("")

    L.append("**1. Distribution shape is more informative than pooled Hoyer sparsity.**")
    L.append("")
    L.append(
        f"At the pooled dimension level, {n_significant_raw} of the five distributional measures have "
        f"p < 0.05. The strongest raw association by absolute magnitude is "
        f"{strongest_raw['measure']} (r = {strongest_raw['spearman_r']:.3f}), whereas Hoyer "
        f"sparsity itself is near zero (r = {r_pooled:.3f}, p = {p_pooled:.4f}). After controlling "
        f"for variance and L2 norm, the strongest association is "
        f"{strongest_partial['measure']} (partial r = "
        f"{strongest_partial['spearman_r_partial']:.3f}). These results distinguish overall "
        "sparsity from higher-order distribution shape."
    )
    L.append("")

    L.append("**2. Within-model and between-model sparsity associations have opposite signs.**")
    L.append("")
    L.append(
        f"Within individual models, {frac_pos:.1%} show a positive sparsity-universality "
        f"correlation (median r = {med_r:.3f}). In contrast, model-level means are negatively "
        f"associated (r = {r_model:.3f}, p = {p_model:.1e}), while the pooled dimension-level "
        f"association is near zero (r = {r_pooled:.3f}). The pooled result therefore masks a "
        "strong within-model pattern that is opposed by between-model differences."
    )
    L.append("")

    # Architecture interaction narrative
    if not dim_interaction.empty and not model_interaction.empty:
        L.append("**3. Architecture changes the magnitude and direction of the association.**")
        L.append("")
        L.append("The dimension-level sparsity-universality correlation varies dramatically by architecture:")
        L.append("")
        for _, row in dim_interaction.iterrows():
            L.append(f"- {row['group']}: r = {row['spearman_r']:.3f} (dimension-level)")
        L.append("")
        L.append("At the model level:")
        L.append("")
        for _, row in model_interaction.iterrows():
            L.append(f"- {row['group']}: r = {row['spearman_r']:.3f} (model-level)")
        L.append("")
        # Robustness insight
        r_cnn_excl = robustness["model_level"]["excl_taskonomy"]
        L.append(
            "Transformer dimensions show a positive pooled association, while CNN dimensions show "
            "a small negative association. At the model level, the strongest association is the "
            "negative CNN result; the Transformer and MLP-Mixer estimates are not statistically "
            f"significant. Excluding Taskonomy attenuates the CNN model-level estimate to "
            f"r = {r_cnn_excl['r']:.3f} (p = {r_cnn_excl['p']:.4f}) without reversing its sign."
        )
        L.append("")

    binned_disc = df.copy()
    binned_disc["sparsity_decile"] = pd.qcut(
        binned_disc["hoyer_sparsity"], 10, labels=False, duplicates="drop"
    )
    binned_disc = (
        binned_disc.groupby("sparsity_decile", observed=True)
        .agg(
            mean_sparsity=("hoyer_sparsity", "mean"),
            mean_universality=("universality", "mean"),
        )
        .reset_index()
    )
    peak_bin = binned_disc.loc[binned_disc["mean_universality"].idxmax()]
    low_bin = binned_disc.iloc[0]
    high_bin = binned_disc.iloc[-1]

    L.append("**4. The binned relationship is inverted-U-shaped rather than monotonic.**")
    L.append("")
    L.append(
        f"Mean universality rises from {low_bin['mean_universality']:.3f} in the lowest sparsity "
        f"decile to {peak_bin['mean_universality']:.3f} near mean Hoyer sparsity "
        f"{peak_bin['mean_sparsity']:.3f}, then falls to {high_bin['mean_universality']:.3f} in "
        "the highest decile. This nonlinearity helps explain why the pooled rank correlation is "
        "near zero despite strong positive within-model correlations."
    )
    L.append("")

    L.append("### Interpretation")
    L.append("")
    L.append(
        "The scale-dependent reversal means that two questions must be kept separate. Within a "
        "given model, relatively sparse dimensions tend to be more universal. Across models, "
        "models with higher mean sparsity tend to have lower mean universality. Architecture, "
        "training objective, and other model-level factors may contribute to this between-model "
        "association, so neither correlation should be interpreted causally."
    )
    L.append("")
    L.append(
        "Hoyer sparsity captures overall concentration, whereas kurtosis and skewness capture "
        "tail shape and asymmetry. Their stronger pooled associations indicate that components "
        "with similar overall sparsity can still differ meaningfully in universality depending "
        "on the detailed shape of their loading distributions."
    )
    L.append("")

    L.append("### Limitations")
    L.append("")
    L.append(
        "1. **Correlational design**: All relationships are observational. Sparsity and "
        "universality may both be consequences of a third factor (e.g., the underlying visual "
        "statistics that a component captures)."
    )
    L.append(
        "2. **NMF non-negativity constraint**: NMF enforces non-negative loadings, which "
        "already biases toward sparse, right-skewed distributions. The observed relationships "
        "may partly reflect NMF's optimization landscape rather than fundamental properties "
        "of the representations."
    )
    L.append(
        f"3. **Rank dependence**: All analyses use rank-{rank} factorizations. The "
        "sparsity-universality relationship may differ at other ranks."
    )
    L.append(
        "4. **Architecture confounds**: The architecture interaction analysis is observational "
        "— differences between CNN and Transformer correlations could be driven by "
        "confounding factors (training data, objective, model scale) rather than architecture per se."
    )
    L.append("")

    return "\n".join(L)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rank = cfg.factorization.rank

    print("=" * 60)
    print("Score Distribution and Universality Analysis")
    print("=" * 60)

    # Load data
    df = load_data(rank, cfg.dataset, list(cfg.models))
    df_meta = load_metadata(df)
    model_df = model_level_aggregation(df)

    # Save merged data
    df.to_csv(out / "dimension_data.csv", index=False)
    model_df.to_csv(out / "model_data.csv", index=False)

    # Analysis
    print("\n--- Measure correlations ---")
    corr_overview = correlate_measures(df)
    corr_overview.to_csv(out / "measure_correlations.csv", index=False)
    print(corr_overview.to_string(index=False))

    print("\n--- Per-model correlations (hoyer_sparsity) ---")
    per_model_rs = per_model_correlations(df, "hoyer_sparsity")
    print(
        f"Median r = {np.median(per_model_rs):.3f}, "
        f"Mean r = {np.mean(per_model_rs):.3f}, "
        f"Fraction positive: {(per_model_rs > 0).mean():.1%}"
    )

    print("\n--- Partial correlations (controlling variance + L2) ---")
    partial_results = []
    for m in [m for m in MEASURES if m not in ("variance", "l2_norm")]:
        r_partial, p_partial = partial_correlation(
            df, m, "universality", ["variance", "l2_norm"]
        )
        partial_results.append({
            "measure": m,
            "spearman_r_partial": r_partial,
            "p_value": p_partial,
        })
        print(f"  {m}: partial r = {r_partial:.3f} (p = {p_partial:.1e})")
    pd.DataFrame(partial_results).to_csv(out / "partial_correlations.csv", index=False)

    print("\n--- Metadata interaction (dimension-level) ---")
    dim_interaction = metadata_interaction_analysis(df_meta, "hoyer_sparsity")
    dim_interaction.to_csv(out / "interaction_dimension.csv", index=False)
    print(dim_interaction.to_string(index=False))

    print("\n--- Metadata interaction (model-level) ---")
    model_interaction = model_level_metadata_interaction(model_df, "hoyer_sparsity")
    model_interaction.to_csv(out / "interaction_model.csv", index=False)
    print(model_interaction.to_string(index=False))

    print("\n--- Robustness: Taskonomy models ---")
    robustness = robustness_taskonomy(df, model_df, "hoyer_sparsity")
    for level in ("dimension_level", "model_level"):
        print(f"  {level}:")
        for subset, stats in robustness[level].items():
            print(f"    {subset}: r={stats['r']:.3f}, p={stats['p']:.4f}, n={stats['n']}")

    # Plots
    print("\n--- Generating plots ---")

    plot_measure_correlation_overview(corr_overview, out)
    print("  measure_overview")

    plot_measure_intercorrelation(df, out)
    print("  measure_intercorrelation")

    plot_multi_scatter(df, out)
    print("  scatter_grid")

    plot_pooled_scatter(df, out, "hoyer_sparsity")
    print("  scatter_hoyer_sparsity")

    plot_per_model_histogram(per_model_rs, out, "hoyer_sparsity")
    print("  per_model_hist")

    binned = binned_analysis(df, "hoyer_sparsity")
    plot_binned_means(binned, out, "hoyer_sparsity")
    print("  binned_hoyer_sparsity")

    plot_nonlinearity(df, out, "hoyer_sparsity", n_bins=20)
    print("  nonlinearity")

    plot_model_level_scatter(model_df, out, "hoyer_sparsity")
    print("  model_level_scatter")

    plot_model_level_by_architecture(model_df, out, "hoyer_sparsity")
    print("  model_level_by_architecture")

    plot_metadata_interaction(df_meta, out, "hoyer_sparsity")
    print("  interaction_architecture")

    plot_partial_vs_raw(corr_overview, partial_results, out)
    print("  partial_vs_raw")

    plot_robustness_taskonomy(robustness, out, "hoyer_sparsity")
    print("  robustness_taskonomy")

    examples = plot_example_distributions(df, out, rank, cfg.dataset)
    print("  example_distributions")

    # Report
    print("\n--- Generating report ---")
    report = generate_report(
        df, model_df, corr_overview, per_model_rs,
        partial_results, dim_interaction, model_interaction, robustness, examples, rank,
    )
    # Beside run.py, so the "outputs/*.png" links inside the report resolve.
    report_path = Path(__file__).parent / "report.md"
    report_path.write_text(report)
    print(f"  Report written to {report_path}")

    print(f"\nDone! All outputs saved to {out}")


if __name__ == "__main__":
    main()
