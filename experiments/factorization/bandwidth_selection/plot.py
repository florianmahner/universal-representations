from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.colors import CYCLE, GRAY_DARK, INDIGO, PURPLE, ROSE
from src.plotting import (
    create_figure,
    save_figure,
    setup_plotting,
)

logger = logging.getLogger(__name__)


def fig1_evar_stability(df: pd.DataFrame, sigmas: list[float], out_dir: Path) -> None:
    """Twin-axis: stability and explained variance vs sigma, per-model lines."""
    setup_plotting()
    fig, ax = create_figure("single")
    ax2 = ax.twinx()

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("sigma_mult")
        ax.plot(sub["sigma_mult"], sub["stability"], color=INDIGO, alpha=0.07, lw=0.4)
        ax2.plot(sub["sigma_mult"], sub["evar"], color=ROSE, alpha=0.07, lw=0.4)

    agg = df.groupby("sigma_mult").agg(
        stab_mean=("stability", "mean"), stab_sem=("stability", "sem"),
        evar_mean=("evar", "mean"), evar_sem=("evar", "sem"),
    ).reset_index()

    ax.errorbar(
        agg["sigma_mult"], agg["stab_mean"], yerr=1.96 * agg["stab_sem"],
        marker="o", color=INDIGO, lw=2, zorder=10, markersize=5,
        label="Stability",
    )
    ax2.errorbar(
        agg["sigma_mult"], agg["evar_mean"], yerr=1.96 * agg["evar_sem"],
        marker="s", color=ROSE, lw=2, zorder=10, markersize=5,
        label="Explained variance",
    )

    ax.set_ylabel("Factorization stability", color=INDIGO)
    ax.tick_params(axis="y", labelcolor=INDIGO)
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Explained variance", color=ROSE)
    ax2.tick_params(axis="y", labelcolor=ROSE)
    ax2.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Sigma multiplier")
    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    save_figure(fig, out_dir / "fig1_evar_stability")


def fig2_harmonic_mean(df: pd.DataFrame, sigmas: list[float], out_dir: Path) -> None:
    """Harmonic mean of (stability, evar) across sigma, per-model lines."""
    setup_plotting()
    fig, ax = create_figure("single")

    sigma_colors = {s: CYCLE[i % len(CYCLE)] for i, s in enumerate(sigmas)}
    best = df.loc[df.groupby("model")["harmonic"].idxmax()]
    model_opt = best.set_index("model")["sigma_mult"].to_dict()

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("sigma_mult")
        ax.plot(
            sub["sigma_mult"], sub["harmonic"],
            color=sigma_colors[model_opt.get(model, 0.5)], alpha=0.12, lw=0.5,
        )

    agg = df.groupby("sigma_mult")["harmonic"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(
        agg["sigma_mult"], agg["mean"], yerr=1.96 * agg["sem"],
        marker="o", color=GRAY_DARK, lw=2.5, zorder=10, markersize=6,
    )

    peak_idx = agg["mean"].idxmax()
    peak_s = agg.loc[peak_idx, "sigma_mult"]
    peak_v = agg.loc[peak_idx, "mean"]
    ax.annotate(
        f"σ = {peak_s}×", (peak_s, peak_v),
        textcoords="offset points", xytext=(12, 8), fontsize=8,
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2),
        color=PURPLE,
    )

    ax.set_xlabel("Sigma multiplier")
    ax.set_ylabel("H(stability, explained variance)")
    ax.set_xscale("log")
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.set_ylim(0, 1.02)
    save_figure(fig, out_dir / "fig2_harmonic_mean")


def _plot_sigma_panel(ax: object, df: pd.DataFrame, show_legend: bool = False) -> None:
    """Plot stability/evar curves for one rank on a single axis."""
    sigmas = sorted(df["sigma_mult"].unique())

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("sigma_mult")
        ax.plot(sub["sigma_mult"], sub["stability"], color=INDIGO, alpha=0.05, lw=0.3)
        ax.plot(sub["sigma_mult"], sub["evar"], color=ROSE, alpha=0.05, lw=0.3)

    agg = df.groupby("sigma_mult").agg(
        stab_mean=("stability", "mean"), stab_sem=("stability", "sem"),
        evar_mean=("evar", "mean"), evar_sem=("evar", "sem"),
    ).reset_index()

    ax.errorbar(
        agg["sigma_mult"], agg["stab_mean"], yerr=1.96 * agg["stab_sem"],
        marker="o", color=INDIGO, zorder=10,
        label="Stability" if show_legend else None,
    )
    ax.errorbar(
        agg["sigma_mult"], agg["evar_mean"], yerr=1.96 * agg["evar_sem"],
        marker="s", color=ROSE, zorder=10,
        label="Expl. variance" if show_legend else None,
    )

    agg_h = df.groupby("sigma_mult")["harmonic"].agg(["mean"]).reset_index()
    peak_idx = agg_h["mean"].idxmax()
    peak_s = agg_h.loc[peak_idx, "sigma_mult"]
    ax.axvline(peak_s, color=PURPLE, ls="--", lw=0.6, alpha=0.7, zorder=5)
    ax.text(
        0.97, 0.03, f"σ* = {peak_s}×",
        transform=ax.transAxes, ha="right", va="bottom",
        color=PURPLE,
    )

    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("Bandwidth multiplier")


def fig_overview(out_dir: Path) -> None:
    """Supplementary overview: 2×3 grid of THINGS ranks + ObjectNet rank 50."""
    from src.plotting import paper_style, paper_figure

    paper_style()

    panels = [
        ("THINGS, r = 10",    out_dir / "things22k" / "rank_10" / "stability_evar_all.csv"),
        ("THINGS, r = 30",    out_dir / "things22k" / "rank_30" / "stability_evar_all.csv"),
        ("THINGS, r = 50",    out_dir / "things22k" / "rank_50" / "stability_evar_all.csv"),
        ("THINGS, r = 100",   out_dir / "things22k" / "rank_100" / "stability_evar_all.csv"),
        ("THINGS, r = 200",   out_dir / "things22k" / "rank_200" / "stability_evar_all.csv"),
        ("ObjectNet, r = 50", out_dir / "objectnet" / "rank_50" / "stability_evar_all.csv"),
    ]

    fig, axes = paper_figure(2, 3, aspect=0.75)

    for idx, (title, csv_path) in enumerate(panels):
        ax = axes.flat[idx]
        if not csv_path.exists():
            ax.set_visible(False)
            logger.warning("Missing %s", csv_path)
            continue
        df = pd.read_csv(csv_path)
        _plot_sigma_panel(ax, df, show_legend=(idx == 0))
        ax.set_title(title)

    axes.flat[0].set_ylabel("Score")
    axes.flat[3].set_ylabel("Score")
    axes.flat[0].legend(loc="lower left")

    for ax in axes.flat:
        if ax.get_subplotspec().colspan.start > 0:
            ax.set_yticklabels([])

    save_figure(fig, out_dir / "overview_bandwidth_selection")
    logger.info("Saved overview figure to %s", out_dir)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    fig_overview(out)

    rank_dir = out / cfg.dataset / f"rank_{cfg.factorization.rank}"
    csv = rank_dir / "stability_evar_all.csv"
    if not csv.exists():
        logger.warning(
            "Missing %s; skipping per-rank plots (run compute_selection first)", csv
        )
        return
    sigmas = [float(s) for s in cfg.factorization.sigma_grid]
    df = pd.read_csv(csv)
    fig1_evar_stability(df, sigmas, rank_dir)
    fig2_harmonic_mean(df, sigmas, rank_dir)
    logger.info("Saved plots to %s", rank_dir)


if __name__ == "__main__":
    main()
