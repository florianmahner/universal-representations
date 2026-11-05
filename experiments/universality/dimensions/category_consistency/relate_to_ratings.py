from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.colors import GRAY_DARK, INDIGO, ROSE, TEAL
from src.plotting import save_figure, setup_plotting

# The four response options of the human dimension-rating study.
LABEL_ORDER = ["semantic", "both", "visual", "neither"]


def load(selectivity_csv: Path, ratings_csv: Path) -> pd.DataFrame:
    sel = pd.read_csv(selectivity_csv)
    sel["dim_id"] = sel["model"] + "__d" + sel["dimension"].astype(str)
    rat = pd.read_csv(ratings_csv)
    df = sel.merge(
        rat[["dim_id", "majority_label", "majority_frac", "pairwise_agreement",
             "n_raters"]],
        on="dim_id", how="inner",
    )
    return df


def _mw(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney U + rank-biserial effect size."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rb = 1 - 2 * u / (len(a) * len(b))   # rank-biserial in [-1, 1]
    return {"u": float(u), "p": float(p), "rank_biserial": float(rb),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "n_a": int(len(a)), "n_b": int(len(b))}


def _auc(df: pd.DataFrame, metric: str, pos: str, neg: str,
         higher_is_pos: bool = True) -> float:
    """ROC AUC for distinguishing label `pos` from `neg` using `metric`.
    Reports AUC for the better direction; values < 0.5 inverted."""
    sub = df[df["majority_label"].isin([pos, neg])]
    y = (sub["majority_label"] == pos).astype(int).values
    s = sub[metric].values
    if higher_is_pos:
        auc = roc_auc_score(y, s)
    else:
        auc = roc_auc_score(y, -s)
    return float(auc)


def stats_table(df: pd.DataFrame) -> dict:
    """Compute label discrimination stats for eta^2 and c_eff."""
    by_label = {
        lab: df[df["majority_label"] == lab]
        for lab in LABEL_ORDER
    }
    n_by_label = {lab: len(g) for lab, g in by_label.items()}

    out: dict = {"n_by_label": n_by_label}

    # Pairwise comparisons that matter for the hypothesis
    pairs = [
        ("semantic", "visual"),    # core test
        ("semantic", "neither"),
        ("visual",   "neither"),
        ("both",     "neither"),
        ("semantic", "both"),
    ]
    out["mannwhitney"] = {}
    for metric in ("eta_squared", "c_eff"):
        out["mannwhitney"][metric] = {}
        for pos, neg in pairs:
            a = by_label[pos][metric].values
            b = by_label[neg][metric].values
            if len(a) >= 5 and len(b) >= 5:
                out["mannwhitney"][metric][f"{pos}_vs_{neg}"] = _mw(a, b)

    # AUC: can the metric distinguish semantic vs visual?
    # Direction by hypothesis: eta^2 higher for semantic, c_eff lower for semantic.
    out["auc_semantic_vs_visual"] = {
        "eta_squared": _auc(df, "eta_squared", "semantic", "visual",
                            higher_is_pos=True),
        "c_eff": _auc(df, "c_eff", "semantic", "visual",
                      higher_is_pos=False),
    }

    # Per-label medians (sanity)
    out["medians"] = {
        metric: {lab: float(np.median(by_label[lab][metric]))
                 for lab in LABEL_ORDER if len(by_label[lab]) > 0}
        for metric in ("eta_squared", "c_eff")
    }
    return out


def plot(df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    setup_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
    palette = [TEAL, INDIGO, ROSE, GRAY_DARK]   # aligned with LABEL_ORDER

    # Panel a: eta^2 by label
    ax = axes[0]
    sns.violinplot(
        data=df, x="majority_label", y="eta_squared",
        order=LABEL_ORDER, palette=palette,
        inner=None, cut=0, linewidth=0.6, ax=ax,
    )
    sns.stripplot(
        data=df, x="majority_label", y="eta_squared",
        order=LABEL_ORDER, color=GRAY_DARK,
        size=1.5, alpha=0.35, jitter=0.25, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Category selectivity ($\\eta^2$)")
    auc = summary["auc_semantic_vs_visual"]["eta_squared"]
    ax.set_title(f"a   AUC$_{{sem|vis}}$ = {auc:.2f}",
                 loc="left", fontsize=9)

    # Panel b: c_eff by label (clip y to p99 for legibility)
    ax = axes[1]
    y_clip = float(np.percentile(df["c_eff"], 99))
    df_clip = df[df["c_eff"] <= y_clip]
    sns.violinplot(
        data=df_clip, x="majority_label", y="c_eff",
        order=LABEL_ORDER, palette=palette,
        inner=None, cut=0, linewidth=0.6, ax=ax,
    )
    sns.stripplot(
        data=df_clip, x="majority_label", y="c_eff",
        order=LABEL_ORDER, color=GRAY_DARK,
        size=1.5, alpha=0.35, jitter=0.25, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Effective # categories ($C^{\\mathrm{eff}}$)")
    auc = summary["auc_semantic_vs_visual"]["c_eff"]
    ax.set_title(f"b   AUC$_{{sem|vis}}$ = {auc:.2f}",
                 loc="left", fontsize=9)

    # Panel c: joint scatter, color by label
    ax = axes[2]
    for lab, color in zip(LABEL_ORDER, palette):
        sub = df[df["majority_label"] == lab]
        ax.scatter(
            sub["eta_squared"], sub["c_eff"],
            c=color, s=10, alpha=0.55,
            edgecolors="white", linewidths=0.3, label=lab.capitalize(),
        )
    ax.set_xlabel("Category selectivity ($\\eta^2$)")
    ax.set_ylabel("Effective # categories ($C^{\\mathrm{eff}}$)")
    ax.set_ylim(top=y_clip)
    ax.legend(fontsize=7, frameon=False, loc="lower left",
              markerscale=1.2, handletextpad=0.3)
    ax.set_title("c", loc="left", fontsize=9)

    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "ratings_vs_metrics")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    df = load(
        out / "category_selectivity.csv",
        Path(cfg.data_root) / "ratings" / "dim_ratings_aggregated_qc_attn_catch.csv",
    )
    print(f"Merged dimensions: {len(df)}")
    print(df["majority_label"].value_counts().to_string())

    summary = stats_table(df)

    print("\n=== medians ===")
    for metric, by_lab in summary["medians"].items():
        print(f"  {metric}:")
        for lab, val in by_lab.items():
            n = summary["n_by_label"][lab]
            print(f"    {lab:<9s} (n={n:3d})  {val:.3f}" if metric == "eta_squared"
                  else f"    {lab:<9s} (n={n:3d})  {val:.1f}")

    print("\n=== Mann-Whitney (rank-biserial effect size) ===")
    for metric, comps in summary["mannwhitney"].items():
        print(f"  {metric}:")
        for name, r in comps.items():
            print(f"    {name:<25s}  rb={r['rank_biserial']:+.3f}  p={r['p']:.2e}")

    print("\n=== AUC (semantic vs visual) ===")
    for metric, auc in summary["auc_semantic_vs_visual"].items():
        print(f"  {metric:<12s}  AUC={auc:.3f}")

    plot(df, summary, out)
    print(f"\nFigure saved -> {out / 'figures' / 'ratings_vs_metrics.pdf'}")


if __name__ == "__main__":
    main()
