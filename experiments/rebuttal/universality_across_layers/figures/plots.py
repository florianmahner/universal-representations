from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

METRIC = "Universality (cos² perm-adjusted)"
# colorblind-safe sequential blues for depth, consistent across all figures
COLOR = {"d010": "#9ecae1", "d050": "#4292c6", "d090": "#08519c", "pen": "#08306b",
         "untrained": "#d95f02", "trained": "#7f7f7f"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}")
    plt.close(fig)
    log.info(f"wrote {folder.name}/{name}.pdf / .png")


def _load(dataset: str, rank: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    uni = pd.read_csv(
        Path("experiments/rebuttal/universality_across_layers/scoring/outputs")
        / f"universality_full_rank_{rank}.csv"
    )
    paper = pd.read_csv(
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    return uni, paper[paper.metric == "cos2_perm_adj"]


def fig1(uni: pd.DataFrame, paper: pd.DataFrame, out: Path) -> None:
    """Per-dimension universality within same-depth cohorts, plus untrained.

    Each group is scored against models of its OWN kind: the depth cohorts
    against the other 161 models at that depth, the untrained networks against
    each other. Depth 1.0 is the paper's published penultimate result (162
    models scored against each other) - the same within-cohort comparison.
    """
    groups = [
        ("Untrained", uni.loc[uni.scoring == "cohort_untrained", "universality"].to_numpy(),
         COLOR["untrained"]),
        ("Early", uni.loc[uni.scoring == "cohort_d010", "universality"].to_numpy(), COLOR["d010"]),
        ("Middle", uni.loc[uni.scoring == "cohort_d050", "universality"].to_numpy(), COLOR["d050"]),
        ("Late", uni.loc[uni.scoring == "cohort_d090", "universality"].to_numpy(), COLOR["d090"]),
        ("Penultimate", paper.universality.to_numpy(), COLOR["pen"]),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    parts = ax.violinplot([g[1] for g in groups], positions=range(len(groups)),
                          showextrema=False, widths=0.78)
    for body, (_, _, color) in zip(parts["bodies"], groups):
        body.set_facecolor(color)
        body.set_alpha(0.9)
        body.set_edgecolor("none")
    for i, (_, vals, _) in enumerate(groups):
        ax.hlines(np.median(vals), i - 0.26, i + 0.26, color="black", lw=1.5, zorder=3)
    ax.set_xticks(range(len(groups)),
                  [f"{lab}\n(n = {len(vals) // 50})" for lab, vals, _ in groups])
    ax.set_xlabel("Layer")
    ax.set_ylabel(METRIC)
    ax.set_title("Universality within matched cohorts", pad=12)
    ax.axvline(0.5, color="0.85", lw=0.8, ls=":", zorder=0)   # separates untrained
    _save(fig, out / "universality_across_depth", "within_depth_cohort_distributions")


def fig2(uni: pd.DataFrame, paper: pd.DataFrame, out: Path) -> None:
    """Model-level universality across depth, against the penultimate reference set."""
    v = uni[(uni.scoring == "vs_ref162") & (uni.role == "trained")]
    m = v.groupby(["base", "depth"]).universality.mean().unstack()[["d010", "d050", "d090"]]
    m["pen"] = paper.groupby("model").universality.mean().reindex(m.index)
    m = m.dropna()
    x = [0.1, 0.5, 0.9, 1.0]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for _, row in m.iterrows():
        ax.plot(x, row.to_numpy(), color=COLOR["trained"], alpha=0.12, lw=0.6, zorder=1)
    mean, sem = m.mean(), m.std() / np.sqrt(len(m))
    ax.errorbar(x, mean.to_numpy(), yerr=sem.to_numpy(), color=COLOR["pen"], lw=2,
                marker="o", ms=5, capsize=3, zorder=3, label=f"Mean ± SEM (n = {len(m)})")
    ax.set_xlabel("Layer")
    ax.set_ylabel(METRIC)
    ax.set_title("Model universality across layers")
    ax.set_xticks(x, ["Early", "Middle", "Late", "Penultimate"])
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out / "universality_across_depth", "model_trajectories_across_depth")


def fig3(uni: pd.DataFrame, paper: pd.DataFrame, out: Path) -> None:
    """Ridgeline: universality of each depth against the penultimate reference set.

    Every representation is scored against the same fixed reference: the 162
    models' penultimate-layer dimensions from the paper. Depth 1.0 is that
    reference set measured against itself (the paper's own published values).
    """
    from scipy.stats import gaussian_kde

    v = uni[uni.scoring == "vs_ref162"].groupby("repr").universality.mean()
    rows = [
        ("Untrained", "", v[v.index.str.startswith("untrained__")].to_numpy(),
         COLOR["untrained"]),
        ("Early", "", v[v.index.str.endswith("__d010")].to_numpy(), COLOR["d010"]),
        ("Middle", "", v[v.index.str.endswith("__d050")].to_numpy(), COLOR["d050"]),
        ("Late", "", v[v.index.str.endswith("__d090")].to_numpy(), COLOR["d090"]),
        ("Penultimate", "",
         paper.groupby("model").universality.mean().to_numpy(), COLOR["pen"]),
    ]
    lo = max(0.0, min(r[2].min() for r in rows) - 0.015)
    hi = max(r[2].max() for r in rows) + 0.015
    grid = np.linspace(lo, hi, 400)
    step, height = 1.0, 0.8

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, (label, sub, vals, color) in enumerate(rows):
        base = (len(rows) - 1 - i) * step
        dens = gaussian_kde(vals)(grid)
        dens = dens / dens.max() * height
        ax.fill_between(grid, base, base + dens, color=color, alpha=0.9, lw=0)
        ax.plot(grid, base + dens, color="white", lw=0.8)
        ax.vlines(vals.mean(), base, base + height, color="black", lw=1.3, zorder=5)
        # mean value inside its own row, next to the line - never overlaps the row above
        ax.text(vals.mean() + 0.004, base + height * 0.55, f"{vals.mean():.3f}",
                ha="left", va="center", fontsize=7.5, color="black", zorder=6)
        # row labels in the left margin, one baseline per row
        ax.text(-0.012, base + height * 0.45, label, transform=ax.get_yaxis_transform(),
                ha="right", va="baseline", fontsize=9)
        note = f"n = {len(vals)}" + (f", {sub}" if sub else "")
        ax.text(-0.012, base + height * 0.10, note, transform=ax.get_yaxis_transform(),
                ha="right", va="baseline", fontsize=7, color="0.45")

    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.12, (len(rows) - 1) * step + height + 0.28)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Model-level {METRIC.lower()}")
    ax.set_title("Universality measured against penultimate dimensions", pad=14)
    fig.subplots_adjust(left=0.22)
    _save(fig, out / "universality_across_depth", "depths_vs_penultimate_reference")


def fig4(uni: pd.DataFrame, paper: pd.DataFrame, out: Path) -> None:
    """Learned structure vs architecture: three comparisons."""
    v = uni.groupby(["scoring", "repr"]).universality.mean().reset_index()
    groups = [
        ("Untrained\nvs penultimate reference", v[(v.scoring == "vs_ref162")
                                           & v.repr.str.startswith("untrained__")], COLOR["untrained"]),
        ("Untrained\namong themselves", v[v.scoring == "cohort_untrained"], COLOR["untrained"]),
        ("Trained (late layer)\namong themselves", v[v.scoring == "cohort_d090"], COLOR["d090"]),
    ]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i, (label, g, color) in enumerate(groups):
        vals = g.universality.to_numpy()
        ax.scatter(rng.normal(i, 0.07, len(vals)), vals, s=10, alpha=0.55,
                   color=color, edgecolor="none", zorder=2)
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color="black", lw=1.8, zorder=3)
        ax.text(i, ax.get_ylim()[1], f"n = {len(vals)}", ha="center", va="bottom",
                fontsize=7, color="0.35")
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylabel(f"Model-level {METRIC.lower()}")
    ax.set_title("Untrained networks share structure, but not the learned one")
    _save(fig, out / "untrained_baselines", "learned_vs_architectural_structure")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    uni, paper = _load(cfg.dataset, cfg.factorization.rank)
    for f in (fig1, fig2, fig3, fig4):
        f(uni, paper, out)


if __name__ == "__main__":
    main()
