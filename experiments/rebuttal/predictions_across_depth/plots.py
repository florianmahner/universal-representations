from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

from experiments.universality.downstream_predictions.macaque_encoding.encoding_things.run import (
    partial_corr,
)
from src.colors import GRAY_DARK
from src.plotting import setup_plotting

from . import data as D

log = logging.getLogger(__name__)

# score column, axis label and chance level of each prediction target
# (1/3 for the three alternative forced choice triplet task, 0 for encoding r);

TASKS = {
    "triplet": dict(
        csv="scores/triplet/across_depth.csv",
        untrained_csv="scores/triplet/untrained.csv",
        score="triplet_accuracy",
        label="THINGS triplet accuracy",
        chance=1 / 3,
    ),
    # monkeys F and N averaged into one score, as the paper reports it
    "macaque": dict(
        csv="scores/macaque/across_depth.csv",
        untrained_csv="scores/macaque/untrained.csv",
        score="mean_r",
        label="Macaque IT encoding",
        chance=0.0,
    ),
}


def _save(fig, path: Path, dpi: int = 600) -> None:
    """Vector PDF plus a crisp high-dpi PNG, written from the same figure so the
    two can never disagree. Local because the shared `save_figure` pins dpi=300."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def _p_text(p: float) -> str:
    if p < 0.001:
        return "$p$ < .001"
    return f"$p$ = {p:.3f}".replace("0.", ".")


def _stats_rows(df: pd.DataFrame, score: str, task: str, uni_col: str,
                axis: str) -> pd.DataFrame:
    rows = []
    for depth, g in df.groupby("depth"):
        g = g.dropna(subset=[uni_col, score])
        if len(g) < 5:
            continue
        r, p = stats.pearsonr(g[uni_col], g[score])
        rho, rho_p = stats.spearmanr(g[uni_col], g[score])
        row = dict(task=task, axis=axis, depth=depth,
                   depth_name=D.DEPTH_NAME[depth], n=len(g),
                   pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=rho_p,
                   score_mean=g[score].mean(), universality_mean=g[uni_col].mean())
        if "top1_accuracy" in g and g.top1_accuracy.notna().sum() >= 10:
            h = g.dropna(subset=["top1_accuracy"])
            pc = partial_corr(h[score].to_numpy(), h[uni_col].to_numpy(),
                              h.top1_accuracy.to_numpy())
            row |= dict(partial_r_given_accuracy=pc.statistic,
                        partial_p_given_accuracy=pc.pvalue, n_partial=len(h))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scatter_by_depth(df: pd.DataFrame, score: str, label: str, task: str,
                          uni_col: str, axis: str, out: Path) -> None:
    """Universality vs prediction, one panel per depth — the published scatter."""
    depths = [d for d in D.DEPTHS if (df.depth == d).any()]
    fig, axes = plt.subplots(1, len(depths), figsize=(1.55 * len(depths) + 0.5, 2.1),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, depth in zip(axes, depths):
        g = df[df.depth == depth].dropna(subset=[uni_col, score])
        ax.scatter(g[uni_col], g[score], s=9, alpha=0.65, color=D.COLOR[depth],
                   edgecolors="white", linewidth=0.25)
        if len(g) >= 3:
            z = np.polyfit(g[uni_col], g[score], 1)
            xs = np.linspace(g[uni_col].min(), g[uni_col].max(), 100)
            ax.plot(xs, np.polyval(z, xs), "--", color=GRAY_DARK, lw=0.9)
            r, _ = stats.pearsonr(g[uni_col], g[score])
            ax.text(0.05, 0.95, f"$r$ = {r:.2f}", transform=ax.transAxes,
                    va="top", fontsize=7)
        ax.set_title(D.DEPTH_NAME[depth], fontsize=8)
    axes[0].set_ylabel(label)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    fig.supxlabel("Universality", fontsize=8, y=0.02)
    path = out / task / f"across_depth_{axis}"
    _save(fig, path)
    log.info(f"Saved {path}")


def plot_untrained(df_tr: pd.DataFrame, df_un: pd.DataFrame, score: str,
                   label: str, chance: float, task: str, out: Path) -> None:
    """Untrained networks against the trained penultimate cohort.

    Left: where the untrained networks fall on the prediction axis. Right: does
    universality predict the target among untrained networks too, on the
    within-cohort axis (untrained vs other untrained).
    """
    tr = df_tr[df_tr.depth == "pen"].dropna(subset=[score, "universality_cohort"])
    un = df_un.dropna(subset=[score, "universality_cohort"])
    if un.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(1.55 * 2 + 0.5, 2.1), sharey=True)

    ax = axes[0]
    for g, c, name in [(tr, D.COLOR["pen"], "Trained"), (un, D.COLOR["untrained"], "Untrained")]:
        ax.scatter(g.universality_cohort, g[score], s=11, alpha=0.65, color=c,
                   edgecolors="white", linewidth=0.25, label=name)
    if chance:
        ax.axhline(chance, ls=":", color=GRAY_DARK, lw=0.8, zorder=0)
    ax.set_xlabel("Universality")
    ax.set_ylabel(label)

    ax = axes[1]
    ax.scatter(un.universality_cohort, un[score], s=11, alpha=0.75,
               color=D.COLOR["untrained"], edgecolors="white", linewidth=0.25)
    if len(un) >= 3:
        z = np.polyfit(un.universality_cohort, un[score], 1)
        xs = np.linspace(un.universality_cohort.min(), un.universality_cohort.max(), 100)
        ax.plot(xs, np.polyval(z, xs), "--", color=GRAY_DARK, lw=0.9)
        r, p = stats.pearsonr(un.universality_cohort, un[score])
        ax.text(0.04, 0.96, f"$r$ = {r:.2f}, {_p_text(p)}",
                transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("Universality")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=7, columnspacing=1.4, handletextpad=0.3)
    path = out / task / "untrained"
    _save(fig, path)
    log.info(f"Saved {path}")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    del cfg  # the figures are fully specified by the score tables triplets.py and macaque.py wrote
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup_plotting()
    plt.rcParams.update({
        "axes.titleweight": "normal", "axes.titlesize": 8,
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None,   # fixed canvas: identical height across the set
    })
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    # dataframe column holding the universality score -> name of its scoring axis
    axis_of = {"universality": "vs_ref162", "universality_cohort": "within_cohort"}
    all_stats = []

    for task, spec in TASKS.items():
        path = out / spec["csv"]
        if not path.exists():
            log.warning(f"Missing {path}, skipping {task}")
            continue
        df = pd.read_csv(path)
        score, label = spec["score"], spec["label"]

        task_stats = []
        for uni_col, axis in axis_of.items():
            s = _stats_rows(df, score, task, uni_col, axis)
            task_stats.append(s)
            for _, row in s.iterrows():
                log.info(f"{task:10s} {axis:14s} {row.depth_name:12s} "
                         f"r={row.pearson_r:+.3f} p={row.pearson_p:.2e} n={int(row.n)}")
            plot_scatter_by_depth(df, score, label, task, uni_col, axis, out)

        all_stats.append(pd.concat(task_stats, ignore_index=True))

        un_path = out / spec["untrained_csv"]
        if un_path.exists():
            plot_untrained(df, pd.read_csv(un_path), score, label, spec["chance"], task, out)
        else:
            log.warning(f"No untrained results at {un_path}")

    if all_stats:
        stats_path = out / "stats" / "prediction_by_depth.csv"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_stats, ignore_index=True).to_csv(stats_path, index=False)
        log.info(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
