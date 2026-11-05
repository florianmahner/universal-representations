from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from scipy import stats

from src.colors import GRAY, GRAY_DARK, GRAY_LIGHT, INDIGO, PURPLE, ROSE, TEAL
from src.plotting import (
    create_figure,
    plt,
    save_figure,
    setup_plotting,
)

log = logging.getLogger(__name__)

OBJ_COLORS = {
    "Self-Supervised": TEAL,
    "Vision-Language": INDIGO,
    "Supervised": ROSE,
    "Untrained": GRAY,
}
OBJ_ORDER = ["Self-Supervised", "Vision-Language", "Supervised"]


def load_data() -> pd.DataFrame:
    """Load and merge universality, behavior, and encoding data."""
    behavior = pd.read_csv(
        Path(
            "experiments/universality/downstream_predictions/things_behavior"
            "/triplets/outputs/triplet_results.csv"
        )
    )

    encoding_dir = Path(
        "experiments/universality/downstream_predictions/macaque_encoding"
        "/encoding_things/outputs"
    )
    enc_parts = []
    for monkey in ["F", "N"]:
        p = encoding_dir / f"encoding_results_{monkey}.csv"
        if p.exists():
            enc = pd.read_csv(p)[["model", "mean_r"]].rename(
                columns={"mean_r": f"encoding_{monkey}"}
            )
            enc_parts.append(enc)
    encoding = enc_parts[0]
    for ep in enc_parts[1:]:
        encoding = encoding.merge(ep, on="model", how="outer")
    encoding["encoding"] = encoding[[c for c in encoding.columns if c != "model"]].mean(axis=1)

    df = behavior.merge(encoding, on="model", how="left")
    clip_diet_labels = {
        "clip_vit_l_14": "OpenAI WIT-400M",
        "openclip_vit_l_14_laion400m": "LAION-400M",
        "openclip_vit_l_14_laion2b": "LAION-2B",
        "openclip_vit_l_14_datacomp_xl": "DataComp-XL",
        "openclip_vit_l_14_dfn2b": "DFN-2B",
        "openclip_vit_l_14_metaclip_400m": "MetaCLIP-400M",
        "openclip_vit_l_14_metaclip_fullcc": "MetaCLIP-FullCC",
    }
    df["clip_label"] = df["model"].map(clip_diet_labels)
    df["is_clip_diet"] = df["clip_label"].notna()
    return df


def _scatter_with_regression(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    highlight_clip: bool = True,
) -> tuple[float, float, float]:
    """Scatter colored by objective_category with regression line.

    Returns (pearson_r, spearman_rho, p_value).
    """
    sub = df.dropna(subset=[x, y])

    for cat in OBJ_ORDER:
        mask = sub["objective_category"] == cat
        if not mask.any():
            continue
        ax.scatter(
            sub.loc[mask, x],
            sub.loc[mask, y],
            s=18,
            alpha=0.55,
            color=OBJ_COLORS[cat],
            edgecolors="white",
            linewidth=0.3,
            label=cat,
            zorder=2,
        )

    if highlight_clip:
        clip = sub[sub["is_clip_diet"]]
        if len(clip) > 0:
            ax.scatter(
                clip[x],
                clip[y],
                s=40,
                facecolors="none",
                edgecolors=PURPLE,
                linewidth=1.2,
                zorder=3,
                label="CLIP ViT-L/14 (diet)",
            )

    z = np.polyfit(sub[x], sub[y], 1)
    x_line = np.linspace(sub[x].min(), sub[x].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY_DARK, lw=1, zorder=1)

    r, p = stats.pearsonr(sub[x], sub[y])
    rho, _ = stats.spearmanr(sub[x], sub[y])
    return r, rho, p


def plot_universality_vs_behavior(df: pd.DataFrame, output_dir: Path) -> None:
    """Universality vs human triplet accuracy."""
    fig, ax = create_figure("single")
    r, rho, p = _scatter_with_regression(ax, df, "universality", "triplet_accuracy")

    ax.set_xlabel("Universality")
    ax.set_ylabel("Triplet accuracy")
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}, p = {p:.0e}\nρ = {rho:.2f}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    save_figure(fig, output_dir / "universality_vs_behavior")


def plot_universality_vs_encoding(df: pd.DataFrame, output_dir: Path) -> None:
    """Universality vs macaque IT encoding (averaged across monkeys)."""
    fig, ax = create_figure("single")
    r, rho, p = _scatter_with_regression(ax, df, "universality", "encoding")

    ax.set_xlabel("Universality")
    ax.set_ylabel("Macaque IT encoding (mean r)")
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}, p = {p:.0e}\nρ = {rho:.2f}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    save_figure(fig, output_dir / "universality_vs_encoding")


def plot_behavior_vs_encoding(df: pd.DataFrame, output_dir: Path) -> None:
    """Human triplet accuracy vs macaque IT encoding, showing CLIP dissociation."""
    fig, ax = create_figure("single")
    r, rho, p = _scatter_with_regression(ax, df, "encoding", "triplet_accuracy")

    ax.set_xlabel("Macaque IT encoding (mean r)")
    ax.set_ylabel("Triplet accuracy")
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}, p = {p:.0e}\nρ = {rho:.2f}",
        transform=ax.transAxes, va="top", fontsize=7,
    )
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    save_figure(fig, output_dir / "behavior_vs_encoding")


def plot_category_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin + strip for universality, behavior, encoding by objective category."""
    sub = df[df["objective_category"].isin(OBJ_ORDER)].copy()

    metrics = [
        ("universality", "Universality"),
        ("triplet_accuracy", "Triplet accuracy"),
        ("encoding", "Macaque IT encoding (mean r)"),
    ]

    for col, ylabel in metrics:
        valid = sub.dropna(subset=[col])
        if len(valid) < 10:
            continue

        fig, ax = create_figure("single")

        sns.violinplot(
            data=valid,
            x="objective_category",
            y=col,
            hue="objective_category",
            order=OBJ_ORDER,
            hue_order=OBJ_ORDER,
            palette=OBJ_COLORS,
            inner=None,
            linewidth=0.5,
            saturation=0.3,
            legend=False,
            ax=ax,
            cut=0,
        )
        sns.stripplot(
            data=valid,
            x="objective_category",
            y=col,
            hue="objective_category",
            order=OBJ_ORDER,
            hue_order=OBJ_ORDER,
            palette=OBJ_COLORS,
            size=3,
            alpha=0.6,
            jitter=0.2,
            legend=False,
            ax=ax,
        )

        clip_in_valid = valid[valid["is_clip_diet"]]
        if len(clip_in_valid) > 0:
            vl_idx = OBJ_ORDER.index("Vision-Language")
            ax.scatter(
                [vl_idx] * len(clip_in_valid),
                clip_in_valid[col],
                s=40,
                facecolors="none",
                edgecolors=PURPLE,
                linewidth=1.2,
                zorder=5,
                label="CLIP ViT-L/14 (diet)",
            )
            ax.legend(fontsize=6, loc="upper right")

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(OBJ_ORDER)))
        ax.set_xticklabels([c.replace("-", "-\n") for c in OBJ_ORDER], fontsize=7)

        fname = f"category_{col}"
        save_figure(fig, output_dir / fname)
        log.info(f"Saved {fname}")


def plot_rank_shift(df: pd.DataFrame, output_dir: Path) -> None:
    """Bump chart showing how model ranks shift between the three metrics."""
    sub = df.dropna(subset=["universality", "triplet_accuracy", "encoding"]).copy()
    sub["rank_univ"] = sub["universality"].rank(ascending=False)
    sub["rank_behavior"] = sub["triplet_accuracy"].rank(ascending=False)
    sub["rank_encoding"] = sub["encoding"].rank(ascending=False)

    fig, ax = create_figure("single")

    for _, row in sub.iterrows():
        color = OBJ_COLORS.get(row["objective_category"], GRAY_LIGHT)
        alpha = 0.08
        lw = 0.5
        zorder = 1
        if row["is_clip_diet"]:
            color = PURPLE
            alpha = 0.8
            lw = 1.5
            zorder = 3

        ax.plot(
            [0, 1, 2],
            [row["rank_univ"], row["rank_behavior"], row["rank_encoding"]],
            color=color,
            alpha=alpha,
            lw=lw,
            zorder=zorder,
        )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Universality", "Behavior", "Encoding"], fontsize=8)
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()

    ax.plot([], [], color=PURPLE, lw=1.5, label="CLIP ViT-L/14 (diet)")
    for cat in OBJ_ORDER:
        ax.plot([], [], color=OBJ_COLORS[cat], lw=1, alpha=0.5, label=cat)
    ax.legend(fontsize=6, loc="lower left")
    save_figure(fig, output_dir / "rank_shift")


def plot_clip_diet_detail(df: pd.DataFrame, output_dir: Path) -> None:
    """Focused bar chart of CLIP diet models across all three metrics."""
    clip = df[df["is_clip_diet"]].copy()
    if len(clip) == 0:
        return

    clip = clip.sort_values("triplet_accuracy", ascending=False)

    metrics = [
        ("universality", "Universality", INDIGO),
        ("triplet_accuracy", "Triplet accuracy", ROSE),
        ("encoding", "Macaque IT encoding", TEAL),
    ]

    for col, title, color in metrics:
        if col not in clip.columns or clip[col].isna().all():
            continue
        fig, ax = create_figure("single")
        y_pos = np.arange(len(clip))
        ax.barh(y_pos, clip[col], color=color, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clip["clip_label"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(title)

        vals = clip[col]
        spread = vals.max() - vals.min()
        mean = vals.mean()
        cv = vals.std() / mean * 100
        ax.text(
            0.95, 0.05,
            f"range: {spread:.3f}\nCV: {cv:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=GRAY_DARK,
        )
        save_figure(fig, output_dir / f"clip_diet_{col}")
        log.info(f"Saved clip_diet_{col}")


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics to log."""
    sub = df.dropna(subset=["universality", "triplet_accuracy", "encoding"])
    n = len(sub)
    log.info(f"\n{'='*60}")
    log.info(f"Models with all three metrics: {n}")

    pairs = [
        ("universality", "triplet_accuracy", "Universality vs Behavior"),
        ("universality", "encoding", "Universality vs Encoding"),
        ("triplet_accuracy", "encoding", "Behavior vs Encoding"),
    ]
    for x, y, label in pairs:
        r, p = stats.pearsonr(sub[x], sub[y])
        rho, _ = stats.spearmanr(sub[x], sub[y])
        log.info(f"  {label}: r={r:.3f} (p={p:.1e}), rho={rho:.3f}")

    log.info("\nBy objective category:")
    for cat in OBJ_ORDER:
        c = sub[sub["objective_category"] == cat]
        log.info(
            f"  {cat:20s} (n={len(c):3d}): "
            f"univ={c['universality'].mean():.3f}  "
            f"behav={c['triplet_accuracy'].mean():.3f}  "
            f"enc={c['encoding'].mean():.3f}"
        )

    clip = sub[sub["is_clip_diet"]]
    if len(clip) > 0:
        log.info(
            f"\n  CLIP diet (n={len(clip)}): "
            f"univ={clip['universality'].mean():.3f} (rank ~{int(sub['universality'].rank(ascending=False)[clip.index].mean())}), "
            f"behav={clip['triplet_accuracy'].mean():.3f} (rank ~{int(sub['triplet_accuracy'].rank(ascending=False)[clip.index].mean())}), "
            f"enc={clip['encoding'].mean():.3f} (rank ~{int(sub['encoding'].rank(ascending=False)[clip.index].mean())})"
        )


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup_plotting()
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    df = load_data()
    if cfg.models:
        df = df[df["model"].isin(list(cfg.models))].reset_index(drop=True)
    log.info(f"Loaded {len(df)} models")

    print_summary(df)

    plot_universality_vs_behavior(df, out)
    plot_universality_vs_encoding(df, out)
    plot_behavior_vs_encoding(df, out)
    plot_category_comparison(df, out)
    plot_rank_shift(df, out)
    plot_clip_diet_detail(df, out)

    log.info(f"\nAll figures saved to {out}")


if __name__ == "__main__":
    main()
