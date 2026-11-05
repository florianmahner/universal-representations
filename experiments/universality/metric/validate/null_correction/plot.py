#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

from src.colors import ROSE
from src.plotting import plt, setup_plotting

logger = logging.getLogger(__name__)


def plot_confound(df: pd.DataFrame, out_dir: Path) -> None:
    """Raw vs corrected universality against explained variance."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    valid = df.dropna(subset=["evar"])

    for ax, y_col, title in zip(
        axes,
        ["raw_mean", "corrected_mean"],
        ["Raw universality", "Corrected universality"],
    ):
        ax.scatter(
            valid["evar"], valid[y_col],
            alpha=0.5, s=20, c=ROSE, edgecolors="none",
        )

        r, p = stats.pearsonr(valid["evar"], valid[y_col])
        ax.set_xlabel("Explained variance (evar)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.1e}")

    fig.suptitle("Null correction removes evar confound", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confound.pdf")
    plt.savefig(out_dir / "confound.png", dpi=200)
    plt.close()
    logger.info("Saved confound figure")


def plot_before_after(df: pd.DataFrame, out_dir: Path) -> None:
    """Raw vs corrected universality scatter with the identity line."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        df["raw_mean"], df["corrected_mean"],
        alpha=0.5, s=20, c=ROSE, edgecolors="none",
    )

    lims = [
        min(df["raw_mean"].min(), df["corrected_mean"].min()) - 0.01,
        max(df["raw_mean"].max(), df["corrected_mean"].max()) + 0.01,
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Raw universality")
    ax.set_ylabel("Corrected universality")
    ax.set_title("Raw vs corrected universality")

    plt.tight_layout()
    plt.savefig(out_dir / "before_after.pdf")
    plt.savefig(out_dir / "before_after.png", dpi=200)
    plt.close()
    logger.info("Saved before_after figure")


def plot_correction_magnitude(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean pair-specific null threshold vs explained variance."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    valid = df.dropna(subset=["evar"])

    ax.scatter(
        valid["evar"], valid["null_mean"],
        alpha=0.5, s=20, c=ROSE, edgecolors="none",
    )

    r, p = stats.pearsonr(valid["evar"], valid["null_mean"])
    ax.set_xlabel("Explained variance (evar)")
    ax.set_ylabel("Mean 95th-percentile null threshold")
    ax.set_title(f"Chance threshold vs evar\nr = {r:.3f}, p = {p:.1e}")

    plt.tight_layout()
    plt.savefig(out_dir / "correction_magnitude.pdf")
    plt.savefig(out_dir / "correction_magnitude.png", dpi=200)
    plt.close()
    logger.info("Saved correction_magnitude figure")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    per_model_df = pd.read_csv(out / "per_model.csv")
    logger.info(
        "Loaded %d models (%s, rank %d)",
        len(per_model_df),
        cfg.dataset,
        cfg.factorization.rank,
    )

    plot_confound(per_model_df, out)
    plot_before_after(per_model_df, out)
    plot_correction_magnitude(per_model_df, out)

    logger.info("All figures saved to %s", out)


if __name__ == "__main__":
    main()
