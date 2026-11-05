import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import ThingsDataset
from src.colors import CYCLE, GREEN, INDIGO, ROSE, WINE
from src.plotting import create_figure, save_figure, setup_plotting

PALETTE_CATEGORICAL = CYCLE + [GREEN, WINE]

log = logging.getLogger(__name__)


def plot_coverage_distribution(
    profiles: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Ranked curve of cluster model coverage."""
    fig, ax = create_figure("wide")

    coverages = profiles["model_coverage"].sort_values(ascending=False).values
    ax.plot(
        np.arange(1, len(coverages) + 1),
        coverages * 100,
        color=INDIGO,
        linewidth=1.5,
    )

    for thresh in [5, 15]:
        n_above = int((coverages * 100 >= thresh).sum())
        ax.axhline(thresh, color=ROSE, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.annotate(
            f"{thresh}% ({n_above})",
            xy=(len(coverages) * 0.75, thresh + 1),
            fontsize=6.5,
            color=ROSE,
        )

    ax.set_xlabel("Cluster rank")
    ax.set_ylabel("Model coverage (%)")
    save_figure(fig, figures_dir / "coverage_distribution", formats=("pdf",))


def _fmt_p(p: float) -> str:
    """Format p-value for annotation."""
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def plot_metrics_vs_coverage(
    profiles: pd.DataFrame,
    figures_dir: Path,
    stats: dict | None = None,
) -> None:
    """Scatter plots of cluster metrics vs model coverage (2x2 panel)."""
    metrics = [
        ("mean_universality", "Mean universality"),
        ("gini", "Gini coefficient"),
        ("coherence", "Within-cluster coherence"),
        ("task_entropy", "Task provenance entropy"),
    ]

    corr_stats = stats.get("correlations", {}) if stats else {}
    tier_stats = stats.get("tier_differences", {}) if stats else {}

    fig, axes = create_figure("full_width", nrows=2, ncols=2)
    cov = profiles["model_coverage"].values * 100
    sizes = np.clip(profiles["size"].values, 3, 100) * 0.5

    for ax, (col, label) in zip(axes.flat, metrics):
        y = profiles[col].values
        ax.scatter(cov, y, s=sizes, alpha=0.35, color=INDIGO, edgecolors="none")

        cs = corr_stats.get(col, {})
        if cs:
            r = cs["spearman_rho"]
            p = cs["spearman_p"]
            ax.annotate(
                f"$\\rho$ = {r:.2f}, {_fmt_p(p)}",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=6.5,
            )
        else:
            valid = ~(np.isnan(cov) | np.isnan(y))
            if valid.sum() > 5:
                r = np.corrcoef(cov[valid], y[valid])[0, 1]
                ax.annotate(
                    f"r = {r:.2f}",
                    xy=(0.95, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=7,
                )

        ts = tier_stats.get(col, {})
        if ts:
            kw = ts.get("kruskal_wallis", {})
            if kw and not np.isnan(kw.get("p", float("nan"))):
                ax.annotate(
                    f"KW: H = {kw['H']:.1f}, {_fmt_p(kw['p'])}",
                    xy=(0.95, 0.87),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=5.5,
                    color=ROSE,
                )

        if col == "gini":
            valid = ~(np.isnan(cov) | np.isnan(y))
            order = np.argsort(cov[valid])
            cov_sorted = cov[valid][order]
            y_sorted = y[valid][order]
            window = max(len(y_sorted) // 15, 20)
            q10, q50, q90 = [], [], []
            centers = []
            for start in range(0, len(y_sorted) - window + 1, window // 2):
                chunk = y_sorted[start : start + window]
                q10.append(np.percentile(chunk, 10))
                q50.append(np.percentile(chunk, 50))
                q90.append(np.percentile(chunk, 90))
                centers.append(cov_sorted[start + window // 2])
            ax.fill_between(
                centers, q10, q90,
                alpha=0.15, color=ROSE, linewidth=0,
            )
            ax.plot(centers, q50, color=ROSE, linewidth=1, alpha=0.7)

            gini_var = stats.get("gini_variance_collapse", {}) if stats else {}
            lev = gini_var.get("levene", {})
            if lev and not np.isnan(lev.get("p", float("nan"))):
                ax.annotate(
                    f"Levene: W = {lev['W']:.1f}, {_fmt_p(lev['p'])}",
                    xy=(0.95, 0.79),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=5.5,
                    color=ROSE,
                )

        ax.set_xlabel("Model coverage (%)")
        ax.set_ylabel(label)

    fig.tight_layout()
    save_figure(fig, figures_dir / "metrics_vs_coverage", formats=("pdf",))


def plot_provenance_by_tier(
    provenance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Stacked bar chart of training task composition by coverage tier."""
    task_prov = provenance[provenance["grouping"] == "task"]
    if len(task_prov) == 0:
        log.warning("No task provenance data; skipping plot")
        return

    from .analysis import DEFAULT_TIER_LABELS

    tier_order = DEFAULT_TIER_LABELS + ["baseline"]
    tiers_present = [t for t in tier_order if t in task_prov["tier"].unique()]

    baseline = task_prov[task_prov["tier"] == "baseline"].sort_values(
        "fraction", ascending=False
    )
    top_tasks = baseline["category"].head(7).tolist()

    fig, ax = create_figure("wide")
    x = np.arange(len(tiers_present))
    bottom = np.zeros(len(tiers_present))

    task_labels = {
        "classification": "Classification",
        "taskonomy": "Taskonomy",
        "selfsupervised": "Self-supervised",
        "bit_expert": "BiT Expert",
        "slip": "SLIP",
        "seer": "SEER",
        "clip": "CLIP",
    }

    for i, task in enumerate(top_tasks):
        fracs = []
        for tier in tiers_present:
            match = task_prov[(task_prov["tier"] == tier) & (task_prov["category"] == task)]
            fracs.append(float(match["fraction"].iloc[0]) if len(match) > 0 else 0.0)
        fracs = np.array(fracs)
        label = task_labels.get(task, task)
        ax.bar(x, fracs, bottom=bottom, color=PALETTE_CATEGORICAL[i], label=label, width=0.65)
        bottom += fracs

    other_fracs = np.maximum(1.0 - bottom, 0)
    ax.bar(x, other_fracs, bottom=bottom, color="#cccccc", label="Other", width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers_present, fontsize=7)
    ax.set_ylabel("Fraction of dimensions")
    ax.legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.tight_layout()
    save_figure(fig, figures_dir / "provenance_by_tier", formats=("pdf",))


def plot_exemplar_grids(
    profiles: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
    ds: ThingsDataset,
    figures_dir: Path,
    n_universal: int = 6,
    n_idiosyncratic: int = 6,
    n_images: int = 8,
) -> None:
    """Image grids comparing universal and idiosyncratic cluster exemplars."""
    from .analysis import top_unique_categories

    categories = np.array(ds.all_categories)

    universal = profiles.head(n_universal)

    idio_pool = profiles[
        (profiles["model_coverage"] < 0.05) & (profiles["size"] >= 5)
    ]
    # Stratify by dominant task: pick the most coherent cluster per task family
    exemplars = []
    for _, group in idio_pool.groupby("dominant_task"):
        exemplars.append(group.nlargest(1, "coherence"))
    idiosyncratic = (
        pd.concat(exemplars)
        .sort_values("coherence", ascending=False)
        .head(n_idiosyncratic)
    )

    n_univ = len(universal)
    n_idio = len(idiosyncratic)
    n_rows = n_univ + 1 + n_idio  # +1 for separator

    fig, axes = plt.subplots(
        n_rows,
        n_images,
        figsize=(1.5 * n_images, 1.4 * n_rows),
        gridspec_kw={"hspace": 0.5, "wspace": 0.05},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    def fill_row(row_idx: int, cluster_row: pd.Series) -> None:
        cid = str(cluster_row["cluster"])
        proto = prototypes.get(cid)
        if proto is None:
            log.warning("No prototype for cluster %s", cid)
            return

        top_idx, top_cats = top_unique_categories(proto, categories, n=n_images)

        for col in range(min(n_images, len(top_idx))):
            ax = axes[row_idx, col]
            ax.imshow(ds.get_all_image(top_idx[col]))
            ax.axis("off")

            if col == 0:
                cov_pct = cluster_row["model_coverage"] * 100
                task = cluster_row["dominant_task"]
                task_frac = cluster_row["dominant_task_frac"] * 100
                detail = cluster_row.get("detail_tasks", "")
                if isinstance(detail, str) and detail:
                    task_info = detail
                else:
                    task_info = f"{task} ({task_frac:.0f}%)"
                top_cat = cluster_row["top_categories"].split(", ")[0]
                label = f"C{cid} | {cov_pct:.0f}% | {task_info}\n{top_cat}"
                ax.set_title(label, fontsize=5, loc="left", pad=2)
            else:
                ax.set_title(top_cats[col], fontsize=4.5, pad=1)
        for col in range(len(top_idx), n_images):
            axes[row_idx, col].axis("off")

    for i, (_, row) in enumerate(universal.iterrows()):
        fill_row(i, row)

    sep_idx = n_univ
    for col in range(n_images):
        axes[sep_idx, col].axis("off")
    axes[sep_idx, 0].text(
        0,
        0.5,
        "Idiosyncratic (coverage < 5%)",
        fontsize=7,
        fontweight="bold",
        va="center",
        transform=axes[sep_idx, 0].transAxes,
    )

    for i, (_, row) in enumerate(idiosyncratic.iterrows()):
        fill_row(sep_idx + 1 + i, row)

    save_figure(fig, figures_dir / "exemplar_grids", formats=("pdf",))


def make_plots(
    profiles: pd.DataFrame,
    provenance: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
    ds: ThingsDataset,
    figures_dir: Path,
    stats: dict | None = None,
) -> None:
    """Generate all figures."""
    setup_plotting()
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_coverage_distribution(profiles, figures_dir)
    plot_metrics_vs_coverage(profiles, figures_dir, stats=stats)
    plot_provenance_by_tier(provenance, figures_dir)
    plot_exemplar_grids(profiles, prototypes, ds, figures_dir)

    log.info("Saved all figures to %s", figures_dir)
