#!/usr/bin/env python3
"""Predict THINGS odd-one-out triplets from NMF embeddings and raw features."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from src.colors import GRAY, GRAY_DARK, INDIGO, PURPLE, ROSE, TEAL
from src.data import ThingsDataset
from src.plotting import CLASS_COLORS, plt, save_figure, setup_plotting

logger = logging.getLogger(__name__)

N_CATEGORIES = 1854  # THINGS concepts; triplet indices run over these
N_IMAGES_PER_CATEGORY = 12  # 22k feature rows are 12 images per concept


def load_triplets(triplet_dir: Path) -> np.ndarray:
    """Load train + validation triplets as (n_triplets, 3) int array."""
    parts = []
    for name in ("trainset.txt", "validationset.txt"):
        t = np.loadtxt(triplet_dir / name, dtype=int)
        parts.append(t)
    triplets = np.concatenate(parts, axis=0)
    assert triplets.shape[1] == 3
    assert triplets.min() >= 0 and triplets.max() < N_CATEGORIES
    logger.info(f"Loaded {len(triplets)} triplets (train + validation)")
    return triplets


def get_exemplary_indices() -> np.ndarray:
    """Return indices of the first (exemplary) image per category in the 22k array."""
    return np.arange(N_CATEGORIES) * N_IMAGES_PER_CATEGORY


def triplet_accuracy_from_sim(s: np.ndarray, triplets: np.ndarray) -> float:
    """Compute odd-one-out accuracy from a precomputed similarity matrix.

    Model is correct if sim(i,j) > max(sim(i,k), sim(j,k)).
    """
    ii = triplets[:, 0]
    jj = triplets[:, 1]
    kk = triplets[:, 2]
    s_ij = s[ii, jj]
    s_ik = s[ii, kk]
    s_jk = s[jj, kk]
    correct = (s_ij > s_ik) & (s_ij > s_jk)
    return float(correct.mean())


def compute_single_model(
    model_name: str,
    dataset: ThingsDataset,
    rank: int,
    exemplary_idx: np.ndarray,
    triplets: np.ndarray,
    feature_dir: Path,
) -> dict:
    """Compute triplet accuracy from both NMF embedding and raw features."""
    w = dataset.load_components(model_name, rank=rank)
    w_ex = w[exemplary_idx]
    s_emb = cosine_similarity(w_ex)
    acc_embedding = triplet_accuracy_from_sim(s_emb, triplets)

    feat_path = feature_dir / f"{model_name}.npy"
    if feat_path.exists():
        feats = np.load(feat_path)[exemplary_idx]
        s_raw = cosine_similarity(feats)
        acc_raw = triplet_accuracy_from_sim(s_raw, triplets)
    else:
        acc_raw = np.nan

    return {
        "model": model_name,
        "triplet_accuracy": acc_embedding,
        "triplet_accuracy_raw": acc_raw,
    }


def load_universality_scores(dataset: str, rank: int) -> pd.DataFrame:
    """Per-model mean universality from the table written by the metric experiment."""
    path = (
        Path("experiments/universality/shared/data")
        / f"universality_{dataset}_rank_{rank}.csv"
    )
    univ = pd.read_csv(path)
    univ = univ[univ["metric"] == "cos2_perm_adj"]
    return univ.groupby("model")["universality"].mean().reset_index()


def run_all(
    dataset: ThingsDataset,
    rank: int,
    n_jobs: int,
    triplet_dir: Path,
    feature_dir: Path,
    requested_models: list[str],
) -> pd.DataFrame:
    """Run triplet accuracy for all DNN models."""
    triplets = load_triplets(triplet_dir)
    exemplary_idx = get_exemplary_indices()

    models = [m for m in dataset.list_component_models() if not m.startswith("macaque")]
    if requested_models:
        models = [m for m in models if m in set(requested_models)]
    logger.info(f"Computing triplet accuracy for {len(models)} models...")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_single_model)(
            m, dataset, rank, exemplary_idx, triplets, feature_dir
        )
        for m in models
    )
    return pd.DataFrame(results)


def plot_universality_vs_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of universality vs triplet accuracy."""
    setup_plotting()

    sub = df.dropna(subset=["universality", "triplet_accuracy"])
    r, p = stats.pearsonr(sub["universality"], sub["triplet_accuracy"])
    rho, rho_p = stats.spearmanr(sub["universality"], sub["triplet_accuracy"])

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, color in CLASS_COLORS.items():
        mask = sub["model_class"] == cls
        if mask.any():
            ax.scatter(
                sub.loc[mask, "universality"], sub.loc[mask, "triplet_accuracy"],
                alpha=0.6, s=30, c=color, edgecolors="white", linewidth=0.3,
                label=f"{cls} (n={mask.sum()})",
            )
    ax.legend(fontsize=8, loc="lower right")

    z = np.polyfit(sub["universality"], sub["triplet_accuracy"], 1)
    x_line = np.linspace(sub["universality"].min(), sub["universality"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY_DARK, lw=1.5)

    ax.set_xlabel("Universality (mean across dimensions)")
    ax.set_ylabel("Triplet accuracy")
    ax.set_title(f"r={r:.3f}, p={p:.1e} | ρ={rho:.3f}, p={rho_p:.1e}")
    fig.tight_layout()
    fig.savefig(output_dir / "universality_vs_triplet_accuracy.pdf")
    plt.close(fig)
    logger.info("Saved universality_vs_triplet_accuracy.pdf")


def plot_model_ranking(df: pd.DataFrame, output_dir: Path, top_n: int = 40) -> None:
    """Horizontal bar chart of top/bottom models by triplet accuracy."""
    setup_plotting()

    default_color = GRAY_DARK

    sorted_df = df.sort_values("triplet_accuracy", ascending=False).reset_index(drop=True)
    top = sorted_df.head(top_n // 2)
    bottom = sorted_df.tail(top_n // 2)
    show = pd.concat([top, bottom]).reset_index(drop=True)
    show["rank_label"] = [f"#{i+1}" for i in range(len(top))] + [
        f"#{len(sorted_df) - len(bottom) + i + 1}" for i in range(len(bottom))
    ]

    bar_colors = [
        CLASS_COLORS.get(row.get("model_class", ""), default_color)
        for _, row in show.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(show) + 1.5))
    y_pos = np.arange(len(show))
    ax.barh(y_pos, show["triplet_accuracy"], color=bar_colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{row['rank_label']} {row['model']}" for _, row in show.iterrows()],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Triplet accuracy (odd-one-out)")
    ax.set_title(f"THINGS triplet prediction — top/bottom {top_n // 2}", fontweight="bold")

    for cls, color in CLASS_COLORS.items():
        if (show["model_class"] == cls).any():
            ax.barh([], [], color=color, label=cls)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "model_ranking.pdf")
    plt.close(fig)
    logger.info("Saved model_ranking.pdf")


def logit_3afc(p: np.ndarray) -> np.ndarray:
    """3AFC logit: l(p) = ln((3p - 1) / (3(1 - p))). Chance at p=1/3 maps to -inf."""
    return np.log((3 * p - 1) / (3 * (1 - p)))


def plot_logit_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    """Universality vs 3AFC logit-transformed triplet accuracy."""
    setup_plotting()

    sub = df.dropna(subset=["universality", "triplet_accuracy"]).copy()
    sub["logit_acc"] = logit_3afc(sub["triplet_accuracy"].values)
    sub["logit_acc_raw"] = logit_3afc(sub["triplet_accuracy_raw"].values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for acc_col, label, ax in [
        ("logit_acc", "NMF embedding", axes[0]),
        ("logit_acc_raw", "Raw features", axes[1]),
    ]:
        valid = sub.dropna(subset=[acc_col])
        if len(valid) < 3:
            continue
        r, p = stats.pearsonr(valid["universality"], valid[acc_col])
        rho, rho_p = stats.spearmanr(valid["universality"], valid[acc_col])

        for cls, color in CLASS_COLORS.items():
            mask = valid["model_class"] == cls
            if mask.any():
                ax.scatter(
                    valid.loc[mask, "universality"],
                    valid.loc[mask, acc_col],
                    alpha=0.6, s=30, c=color, edgecolors="white", linewidth=0.3,
                    label=f"{cls} (n={mask.sum()})",
                )

        z = np.polyfit(valid["universality"], valid[acc_col], 1)
        x_line = np.linspace(valid["universality"].min(), valid["universality"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY_DARK, lw=1.5)

        ax.set_xlabel("Universality (mean across dimensions)")
        ax.set_ylabel("Triplet accuracy (3AFC logit)")
        ax.set_title(f"{label}: r={r:.3f}, p={p:.1e} | ρ={rho:.3f}")
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "universality_vs_logit_accuracy.pdf")
    plt.close(fig)
    logger.info("Saved universality_vs_logit_accuracy")


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Side-by-side: raw accuracy vs logit-transformed, to show linearization."""
    setup_plotting()

    sub = df.dropna(subset=["universality", "triplet_accuracy"]).copy()
    sub["logit_acc"] = logit_3afc(sub["triplet_accuracy"].values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for acc_col, ylabel, title, ax in [
        ("triplet_accuracy", "Triplet accuracy (p)", "Raw accuracy", axes[0]),
        ("logit_acc", "Triplet accuracy (3AFC logit)", "3AFC logit", axes[1]),
    ]:
        r, p = stats.pearsonr(sub["universality"], sub[acc_col])

        for cls, color in CLASS_COLORS.items():
            mask = sub["model_class"] == cls
            if mask.any():
                ax.scatter(
                    sub.loc[mask, "universality"],
                    sub.loc[mask, acc_col],
                    alpha=0.6, s=30, c=color, edgecolors="white", linewidth=0.3,
                    label=f"{cls} (n={mask.sum()})",
                )

        z = np.polyfit(sub["universality"], sub[acc_col], 1)
        x_line = np.linspace(sub["universality"].min(), sub["universality"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color=GRAY_DARK, lw=1.5)

        ax.set_xlabel("Universality")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (r={r:.3f}, p={p:.1e})")
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_raw_vs_logit.pdf")
    plt.close(fig)
    logger.info("Saved accuracy_raw_vs_logit")


def plot_embedding_vs_raw(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: NMF embedding accuracy vs raw feature accuracy per model."""
    setup_plotting()

    sub = df.dropna(subset=["triplet_accuracy", "triplet_accuracy_raw"])
    r, p = stats.pearsonr(sub["triplet_accuracy_raw"], sub["triplet_accuracy"])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for cls, color in CLASS_COLORS.items():
        mask = sub["model_class"] == cls
        if mask.any():
            ax.scatter(
                sub.loc[mask, "triplet_accuracy_raw"],
                sub.loc[mask, "triplet_accuracy"],
                alpha=0.6, s=30, c=color, edgecolors="white", linewidth=0.3,
                label=f"{cls} (n={mask.sum()})",
            )

    lims = [
        min(sub["triplet_accuracy_raw"].min(), sub["triplet_accuracy"].min()) - 0.01,
        max(sub["triplet_accuracy_raw"].max(), sub["triplet_accuracy"].max()) + 0.01,
    ]
    ax.plot(lims, lims, "--", color="gray", lw=1, alpha=0.5, label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Triplet accuracy (raw features)")
    ax.set_ylabel("Triplet accuracy (NMF embedding)")
    ax.set_title(f"r={r:.3f}, p={p:.1e}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_vs_raw_accuracy.pdf")
    plt.close(fig)
    logger.info("Saved embedding_vs_raw_accuracy.pdf")


def plot_top_models(df: pd.DataFrame, output_dir: Path, top_n: int = 25) -> None:
    """Bar chart of top models colored by objective category, highlighting CLIP diet."""
    setup_plotting()
    top = df.sort_values("triplet_accuracy", ascending=False).head(top_n).reset_index(drop=True)

    cat_colors = {
        "Vision-Language": INDIGO,
        "Supervised": ROSE,
        "Self-Supervised": TEAL,
    }
    bar_colors = [
        cat_colors.get(row.get("objective_category", ""), GRAY)
        for _, row in top.iterrows()
    ]
    clip_diet = top["model"].str.contains("openclip_vit_l_14")
    edge_colors = [PURPLE if c else "white" for c in clip_diet]
    linewidths = [1.5 if c else 0.3 for c in clip_diet]

    fig, ax = plt.subplots(figsize=(5.5, 6))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["triplet_accuracy"], color=bar_colors,
            edgecolor=edge_colors, linewidth=linewidths)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [row.get("display_name", row["model"]) for _, row in top.iterrows()],
        fontsize=6.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Triplet accuracy (odd-one-out)")

    for cat, color in cat_colors.items():
        ax.barh([], [], color=color, label=cat)
    ax.plot([], [], color=PURPLE, linewidth=2, label="OpenCLIP diet contrast")
    ax.legend(fontsize=7, loc="lower right")
    save_figure(fig, output_dir / "top_models_highlight")
    logger.info("Saved top_models_highlight")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    dataset = ThingsDataset()
    uni = load_universality_scores(cfg.dataset, cfg.factorization.rank)
    meta = dataset.model_info[
        ["name", "display_name", "model_class", "train_task", "train_data",
         "feature_uid", "architecture", "objective_category"]
    ].copy()

    csv_path = out / "triplet_results.csv"

    results = run_all(
        dataset,
        cfg.factorization.rank,
        cfg.n_jobs,
        Path(cfg.data_root) / "behavior" / "things_triplets",
        Path(cfg.data_root) / "features" / cfg.dataset,
        list(cfg.models),
    )
    results = results.merge(uni, on="model", how="left")
    results = results.merge(meta, left_on="model", right_on="name", how="left")
    results = results.drop(columns=["name"], errors="ignore")
    results.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    df = pd.read_csv(csv_path)

    sub = df.dropna(subset=["universality", "triplet_accuracy"])
    r, p = stats.pearsonr(sub["universality"], sub["triplet_accuracy"])
    logger.info(f"Universality vs embedding accuracy: r={r:.3f}, p={p:.1e} (n={len(sub)})")
    logger.info(f"Embedding accuracy range: {sub['triplet_accuracy'].min():.3f} – {sub['triplet_accuracy'].max():.3f}")

    logit_acc = logit_3afc(sub["triplet_accuracy"].values)
    r_logit, p_logit = stats.pearsonr(sub["universality"], logit_acc)
    logger.info(f"Universality vs logit(embedding): r={r_logit:.3f}, p={p_logit:.1e}")

    sub_raw = df.dropna(subset=["universality", "triplet_accuracy_raw"])
    if len(sub_raw) > 0:
        r_raw, p_raw = stats.pearsonr(sub_raw["universality"], sub_raw["triplet_accuracy_raw"])
        logger.info(f"Universality vs raw accuracy: r={r_raw:.3f}, p={p_raw:.1e} (n={len(sub_raw)})")
        logger.info(f"Raw accuracy range: {sub_raw['triplet_accuracy_raw'].min():.3f} – {sub_raw['triplet_accuracy_raw'].max():.3f}")

    setup_plotting()
    plot_universality_vs_accuracy(df, out)
    plot_logit_accuracy(df, out)
    plot_accuracy_comparison(df, out)
    plot_model_ranking(df, out)
    plot_embedding_vs_raw(df, out)
    plot_top_models(df, out)


if __name__ == "__main__":
    main()
