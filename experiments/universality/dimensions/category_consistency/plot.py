from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import ThingsDataset
from src.colors import GRAY_DARK, INDIGO, PURPLE, ROSE
from src.plotting import setup_plotting, save_figure

log = logging.getLogger(__name__)


def _format_p(p: float) -> str:
    if p == 0 or p < 1e-300:
        return "p < 10$^{-300}$"
    if p < 0.001:
        return f"p < 10$^{{{int(np.floor(np.log10(p)))}}}$"
    return f"p = {p:.3f}"


def _annotate_rho(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    rho, p = spearmanr(x[valid], y[valid])
    ax.text(
        0.97, 0.97, f"$\\rho$ = {rho:.2f}\n{_format_p(p)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
    )


def _binned_means(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, n_bins: int = 10,
    color: str = PURPLE,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x_v, y_v = x[valid], y[valid]
    bin_edges = np.percentile(x_v, np.linspace(0, 100, n_bins + 1))
    bin_centers, bin_means, bin_sems = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (x_v >= lo) & (x_v < hi) if i < n_bins - 1 else (x_v >= lo) & (x_v <= hi)
        if mask.sum() < 5:
            continue
        bin_centers.append(x_v[mask].mean())
        bin_means.append(y_v[mask].mean())
        bin_sems.append(y_v[mask].std() / np.sqrt(mask.sum()))
    ax.errorbar(
        bin_centers, bin_means, yerr=bin_sems,
        fmt="o-", color=color, markersize=4, lw=1.2, capsize=2, zorder=5,
    )


def _decile_stats(
    df: pd.DataFrame, x_col: str, y_col: str, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute decile-binned means and SEMs for y_col grouped by x_col."""
    valid = df.dropna(subset=[x_col, y_col])
    x, y = valid[x_col].values, valid[y_col].values
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    centers, means, sems = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & (x < hi) if i < n_bins - 1 else (x >= lo) & (x <= hi)
        if mask.sum() < 5:
            continue
        centers.append(x[mask].mean())
        means.append(y[mask].mean())
        sems.append(y[mask].std() / np.sqrt(mask.sum()))
    return np.array(centers), np.array(means), np.array(sems)


def plot_category_selectivity(
    selectivity: pd.DataFrame, within_stats: dict,
    variance_stats: dict, output_dir: Path,
) -> None:
    """Category selectivity vs universality.

    Panel a: Variance decomposition crossover (between vs within % of total).
    Panel b: Hexbin of eta-squared vs universality with binned means.
    """
    setup_plotting()
    valid = selectivity.dropna(subset=["universality", "eta_squared"]).copy()
    valid["ss_total"] = valid["ss_between"] + valid["ss_within"]
    valid["between_pct"] = valid["ss_between"] / valid["ss_total"] * 100
    valid["within_pct"] = valid["ss_within"] / valid["ss_total"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Panel a: Variance decomposition crossover
    ax = axes[0]
    for col, label, color in [
        ("between_pct", "Between-category", INDIGO),
        ("within_pct", "Within-category", ROSE),
    ]:
        c, m, s = _decile_stats(valid, "universality", col)
        ax.fill_between(c, m - s, m + s, alpha=0.15, color=color)
        ax.plot(c, m, "o-", color=color, markersize=4, lw=1.5, label=label)
    ax.axhline(50, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("Universality")
    ax.set_ylabel("% of total variance")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="center right")

    ss_b_rho = variance_stats["ss_between"]["within_model"]["median_rho"]
    ss_w_rho = variance_stats["ss_within"]["within_model"]["median_rho"]
    ax.text(
        0.03, 0.03,
        f"$\\rho_{{between}}$ = {ss_b_rho:.2f}\n$\\rho_{{within}}$ = {ss_w_rho:.2f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Panel b: Hexbin of eta-squared
    ax = axes[1]
    x, y = valid["universality"].values, valid["eta_squared"].values
    ax.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1, linewidths=0.2)
    _binned_means(ax, x, y)
    _annotate_rho(ax, x, y)
    ax.set_xlabel("Universality")
    ax.set_ylabel("Category selectivity ($\\eta^2$)")
    ax.set_title("b", loc="left", fontweight="bold", fontsize=10)

    n_cats, n_items = 1854, 22248
    null_eta = (n_cats - 1) / (n_items - 1)
    ax.axhline(null_eta, color=GRAY_DARK, ls=":", lw=0.8, alpha=0.4)
    ax.text(0.03, null_eta + 0.01, "null", fontsize=7, color=GRAY_DARK,
            transform=ax.get_yaxis_transform(), alpha=0.6)

    med_rho = within_stats["median_rho"]
    frac = within_stats["frac_positive"]
    ax.text(
        0.03, 0.97,
        f"Within-model\n$\\rho$ = {med_rho:.2f}\n{frac:.0%} positive",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "category_selectivity")


def plot_effective_categories(
    selectivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Effective number of categories (C_eff_excess) — complementary to eta^2.

    Uses the baseline-corrected version: per-dimension category-mean loadings
    minus their min before normalizing. Without this correction the NMF
    baseline floor dominates and C_eff sits near n_cats for almost every
    dimension. The excess version reflects how concentrated the *above-floor*
    loading mass is — the meaningful question.

    Panel a: C_eff_excess vs universality hexbin with binned trend.
    Panel b: joint eta^2 x C_eff_excess with median-split quadrants.
    """
    setup_plotting()
    valid = selectivity.dropna(
        subset=["universality", "eta_squared", "c_eff_excess"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Long-tailed: most dims have small excess C_eff but a few extend far up.
    # Clip at p99 for visualization; binned means and ρ use full data.
    ce_all = valid["c_eff_excess"].values
    y_max = float(np.percentile(ce_all, 99)) * 1.05
    y_min = 0.0

    # Panel a: C_eff_excess vs universality
    ax = axes[0]
    x, y = valid["universality"].values, valid["c_eff_excess"].values
    mask = (y >= y_min) & (y <= y_max)
    ax.hexbin(x[mask], y[mask], gridsize=30, cmap="Blues",
              mincnt=1, linewidths=0.2)
    _binned_means(ax, x, y, color=PURPLE)
    _annotate_rho(ax, x, y)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Universality")
    ax.set_ylabel("Effective # categories ($C^{\\mathrm{eff}}_{\\mathrm{excess}}$)")
    ax.set_title("a", loc="left", fontweight="bold", fontsize=10)

    # Panel b: joint eta^2 x C_eff_excess
    ax = axes[1]
    eta = valid["eta_squared"].values
    ce = valid["c_eff_excess"].values
    mask = (ce >= y_min) & (ce <= y_max)
    ax.hexbin(eta[mask], ce[mask], gridsize=30, cmap="Blues",
              mincnt=1, linewidths=0.2)
    _annotate_rho(ax, eta, ce)
    eta_med = float(np.median(eta))
    ce_med = float(np.median(ce))
    ax.axvline(eta_med, color=GRAY_DARK, ls=":", lw=0.8, alpha=0.5)
    ax.axhline(ce_med, color=GRAY_DARK, ls=":", lw=0.8, alpha=0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Category selectivity ($\\eta^2$)")
    ax.set_ylabel("Effective # categories ($C^{\\mathrm{eff}}_{\\mathrm{excess}}$)")
    ax.set_title("b", loc="left", fontweight="bold", fontsize=10)

    eta_lo, eta_hi = float(eta.min()), float(eta.max())
    quad_x_lo = eta_lo + 0.05 * (eta_med - eta_lo)
    quad_x_hi = eta_med + 0.05 * (eta_hi - eta_med)
    quad_y_lo = y_min + 0.05 * (ce_med - y_min)
    quad_y_hi = y_max - 0.05 * (y_max - ce_med)
    fs = 6.5
    ax.text(quad_x_hi, quad_y_lo, "category-\nspecific",
            fontsize=fs, color=GRAY_DARK, ha="left", va="bottom")
    ax.text(quad_x_hi, quad_y_hi, "broad\nsemantic",
            fontsize=fs, color=GRAY_DARK, ha="left", va="top")
    ax.text(quad_x_lo, quad_y_hi, "broad\nnon-category",
            fontsize=fs, color=GRAY_DARK, ha="left", va="top")
    ax.text(quad_x_lo, quad_y_lo, "narrow\nnon-category",
            fontsize=fs, color=GRAY_DARK, ha="left", va="bottom")

    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "effective_categories")


def plot_exemplar_dimensions(
    selectivity: pd.DataFrame,
    components: dict[str, np.ndarray],
    categories: np.ndarray,
    output_dir: Path,
    n_universal: int = 3,
    n_idiosyncratic: int = 3,
    n_images: int = 8,
) -> None:
    """Image grids contrasting universal vs idiosyncratic dimensions.

    Selects dimensions with the highest/lowest eta-squared from diverse models,
    showing top-loading images (one per unique category) for each.
    """
    setup_plotting()
    ds = ThingsDataset()
    valid = selectivity.dropna(subset=["universality", "eta_squared"])

    # Select universal examples: highest eta^2, from diverse models
    universal_pool = valid.nlargest(200, "eta_squared")
    universal_sel = []
    used_models = set()
    for _, row in universal_pool.iterrows():
        if row["model"] not in used_models:
            universal_sel.append(row)
            used_models.add(row["model"])
            if len(universal_sel) >= n_universal:
                break

    # Select idiosyncratic examples: lowest eta^2, exclude taskonomy
    idio_pool = valid[~valid["model"].str.contains("taskonomy")]
    idio_pool = idio_pool.nsmallest(200, "eta_squared")
    idio_sel = []
    used_models = set()
    for _, row in idio_pool.iterrows():
        if row["model"] not in used_models:
            idio_sel.append(row)
            used_models.add(row["model"])
            if len(idio_sel) >= n_idiosyncratic:
                break

    selected = universal_sel + idio_sel
    n_rows = len(selected)
    if n_rows == 0:
        log.warning("No exemplar dimensions found")
        return

    fig, axes = plt.subplots(
        n_rows, n_images,
        figsize=(1.2 * n_images, 1.5 * n_rows),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, dim_info in enumerate(selected):
        model = dim_info["model"]
        d = int(dim_info["dimension"])
        W = components[model]
        w_col = W[:, d]

        # Top-loading images (one per unique category)
        sorted_idx = np.argsort(-w_col)
        top_idx = []
        seen_cats = set()
        for idx in sorted_idx:
            cat = categories[idx]
            if cat not in seen_cats:
                top_idx.append(idx)
                seen_cats.add(cat)
                if len(top_idx) >= n_images:
                    break

        top_cats = [categories[i] for i in top_idx[:3]]
        is_universal = row < n_universal

        for col in range(n_images):
            ax = axes[row, col]
            if col < len(top_idx):
                try:
                    img = ds.get_all_image(top_idx[col])
                    ax.imshow(img)
                except Exception:
                    pass
            ax.axis("off")

            # Color-coded left edge
            edge_color = INDIGO if is_universal else ROSE
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.patch.set_edgecolor(edge_color)
            ax.patch.set_linewidth(2)
            ax.patch.set_visible(True)

        # Row label on the left
        label = f"$\\eta^2$={dim_info['eta_squared']:.2f}  {', '.join(top_cats)}"
        axes[row, 0].set_title(label, fontsize=7, loc="left", pad=3)

    # Section headers
    fig.text(0.01, 1.02, "Universal", fontsize=9, fontweight="bold",
             color=INDIGO, va="bottom")
    fig.text(0.01, (n_rows - n_universal) / n_rows - 0.01, "Idiosyncratic",
             fontsize=9, fontweight="bold", color=ROSE, va="top")

    fig.tight_layout(h_pad=0.8)
    save_figure(fig, output_dir / "figures" / "exemplar_dimensions")


def make_all_plots(
    selectivity: pd.DataFrame,
    within_stats: dict,
    variance_stats: dict,
    components: dict[str, np.ndarray],
    categories: np.ndarray,
    output_dir: Path,
    **_kwargs,
) -> None:
    """Generate all figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fig 1: category selectivity...")
    plot_category_selectivity(selectivity, within_stats, variance_stats, output_dir)

    log.info("Fig 2: effective number of categories (C_eff)...")
    plot_effective_categories(selectivity, output_dir)

    log.info("Fig 3: exemplar dimensions...")
    plot_exemplar_dimensions(selectivity, components, categories, output_dir)

    log.info("All figures saved to %s", fig_dir)
