from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from src.colors import INDIGO, ROSE, TEAL
from src.plotting import create_figure, save_figure, setup_plotting

log = logging.getLogger(__name__)


def fig_r2_distribution(dim_df: pd.DataFrame, out: Path) -> None:
    """R² distribution split by universality tertile."""
    df = dim_df.dropna(subset=["universality"]).copy()
    df["univ_tertile"] = pd.qcut(df["universality"], 3, labels=["low", "mid", "high"])
    tertile_colors = {
        "low": INDIGO,
        "mid": TEAL,
        "high": ROSE,
    }

    fig, ax = create_figure("single")
    for label in ["low", "mid", "high"]:
        subset = df.loc[df["univ_tertile"] == label, "spose_r2"]
        sns.kdeplot(
            subset, ax=ax, color=tertile_colors[label], linewidth=1.2,
            label=f"{label} universality (n={len(subset)})",
            fill=True, alpha=0.15,
        )
    med = dim_df["spose_r2"].median()
    ax.axvline(med, color="0.4", linewidth=0.8, linestyle="--",
               label=f"overall median = {med:.3f}")
    ax.set_xlabel("Cross-validated $R^2$ (SPoSE → NMF dim)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save_figure(fig, out / "r2_distribution")


def fig_r2_vs_universality(dim_df: pd.DataFrame, out: Path) -> None:
    """Dim-level scatter: universality vs SPoSE R²."""
    df = dim_df.dropna(subset=["universality"])
    fig, ax = create_figure("single")
    ax.scatter(df["universality"], df["spose_r2"],
               s=3, alpha=0.15, c=INDIGO, edgecolors="none",
               rasterized=True)
    r, _ = spearmanr(df["universality"], df["spose_r2"])
    ax.set_xlabel("Universality")
    ax.set_ylabel("SPoSE $R^2$")
    ax.set_title(f"$\\rho$ = {r:.2f}", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / "r2_vs_universality")


def fig_universality_by_quality(
    dim_df: pd.DataFrame,
    spose_meta: pd.DataFrame,
    out: Path,
) -> None:
    """Two panels: raw and normalized quality-specific R² vs universality."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    quality_colors = {
        "semantic": INDIGO,
        "visual": ROSE,
        "mix visual-semantic": TEAL,
        "unclear": "0.5",
    }
    panels = [
        ("semantic", "spose_r2_semantic"),
        ("visual", "spose_r2_visual"),
        ("mix visual-semantic", "spose_r2_mix"),
        ("unclear", "spose_r2_unclear"),
    ]
    panels = [(q, c) for q, c in panels if c in dim_df.columns]
    if not panels:
        log.warning("No quality-specific R² columns found, skipping")
        return

    quality_counts = spose_meta["quality"].value_counts()
    univ = dim_df["universality"].values
    valid = ~np.isnan(univ)
    x = univ[valid]
    r2_total = dim_df["spose_r2"].values[valid]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=False)

    # Left: raw R²
    ax = axes[0]
    for q, col in panels:
        y = dim_df[col].values[valid]
        smooth = lowess(y, x, frac=0.3)
        rho, _ = spearmanr(x, y)
        n = quality_counts.get(q, 0)
        ax.plot(smooth[:, 0], smooth[:, 1], color=quality_colors[q],
                linewidth=1.5, label=f"{q} ({n}, $\\rho$={rho:.2f})")
    ax.set_xlabel("Universality")
    ax.set_ylabel("SPoSE $R^2$")
    ax.legend(fontsize=5.5, frameon=False)

    # Right: normalized R² (fraction of total R²)
    ax = axes[1]
    safe_total = np.where(r2_total > 0.01, r2_total, np.nan)
    for q, col in panels:
        y_raw = dim_df[col].values[valid]
        y_norm = y_raw / safe_total
        mask = ~np.isnan(y_norm)
        smooth = lowess(y_norm[mask], x[mask], frac=0.3)
        rho, _ = spearmanr(x[mask], y_norm[mask])
        n = quality_counts.get(q, 0)
        ax.plot(smooth[:, 0], smooth[:, 1], color=quality_colors[q],
                linewidth=1.5, label=f"{q} ({n}, $\\rho$={rho:.2f})")
    ax.set_xlabel("Universality")
    ax.set_ylabel("$R^2_\\mathrm{quality}$ / $R^2_\\mathrm{all}$")
    ax.legend(fontsize=5.5, frameon=False)

    fig.tight_layout()
    save_figure(fig, out / "universality_by_quality")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    setup_plotting()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    dim_df = pd.read_csv(out / "dim_r2.csv")
    spose_meta = pd.read_csv(
        Path(cfg.data_root)
        / "embeddings" / "things" / "human_behavior" / "dimension_metadata.csv"
    )
    log.info("Loaded: %d dims, %d SPoSE metadata rows", len(dim_df), len(spose_meta))

    log.info("Generating figures")
    fig_r2_distribution(dim_df, out)
    fig_r2_vs_universality(dim_df, out)
    fig_universality_by_quality(dim_df, spose_meta, out)

    plt.close("all")
    log.info("Done. Figures in %s", out)


if __name__ == "__main__":
    main()
