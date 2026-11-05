#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from experiments.universality.shared.utils import (
    load_components_for_rank,
    load_universality,
)
from src.data import ThingsDataset
from src.colors import GRAY_DARK, ROSE
from src.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def _lowess_line(x: np.ndarray, y: np.ndarray, frac: float = 0.3):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    result = lowess(y, x, frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


# Panel A: Sparsity Goldilocks


def load_sparsity_data() -> pd.DataFrame:
    path = Path(
        "experiments/universality/dimensions/sparsity/outputs/dimension_data.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Sparsity data not found: {path}\n"
            "Run: python -m experiments.universality.dimensions.sparsity.run"
        )
    return pd.read_csv(path)


def plot_panel_a(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Universality vs Hoyer sparsity with density and LOWESS."""
    fig, ax = create_figure("single")

    ax.hexbin(
        df["hoyer_sparsity"], df["universality"],
        gridsize=40, cmap="Blues", mincnt=1, alpha=0.8, linewidths=0.2,
    )

    x_low, y_low = _lowess_line(
        df["hoyer_sparsity"].values, df["universality"].values, frac=0.3
    )
    ax.plot(x_low, y_low, color=ROSE, lw=2, label="LOWESS")

    rho, p = sp_stats.spearmanr(df["hoyer_sparsity"], df["universality"])
    ax.text(
        0.97, 0.97, f"$\\rho$ = {rho:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color=GRAY_DARK,
    )

    ax.set_xlabel("Hoyer sparsity")
    ax.set_ylabel("Universality")
    ax.legend(fontsize=7, loc="upper left")

    save_figure(fig, output_dir / "panel_a_sparsity")
    logger.info(f"Panel A: Spearman rho={rho:.4f}, p={p:.2e}")
    return fig


# Panel B: Text predictability


def compute_text_embeddings(
    category_names: list[str], cache_path: Path,
) -> np.ndarray:
    """Embed category names with sentence-transformers (all-mpnet-base-v2)."""
    if cache_path.exists():
        logger.info(f"Loading cached text embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    logger.info("Computing text embeddings with all-mpnet-base-v2...")
    model = SentenceTransformer("all-mpnet-base-v2")
    names_clean = [name.replace("_", " ") for name in category_names]
    embeddings = model.encode(names_clean, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info(f"Cached text embeddings to {cache_path}")
    return embeddings


def compute_text_r2(
    components: dict[str, np.ndarray],
    text_emb_path: Path,
    ds: ThingsDataset,
) -> pd.DataFrame:
    """Cross-validated R² predicting each NMF dim from text embeddings."""
    cats = ds.categories
    unique_cats = sorted(set(cats))
    text_emb = np.load(text_emb_path)
    cat_name_to_emb = dict(zip(unique_cats, text_emb))
    text_image = np.array([cat_name_to_emb[c] for c in cats])

    text_pca = PCA(n_components=10, random_state=42).fit_transform(
        StandardScaler().fit_transform(text_image)
    )

    ridge = Ridge(alpha=1.0)
    records = []
    n_total = sum(W.shape[1] for W in components.values())
    done = 0
    for model_name, W in components.items():
        for dim_idx in range(W.shape[1]):
            scores = cross_val_score(ridge, text_pca, W[:, dim_idx], cv=5, scoring="r2")
            records.append({
                "model": model_name,
                "dimension": dim_idx,
                "r2_text": max(np.mean(scores), 0.0),
            })
            done += 1
            if done % 500 == 0:
                logger.info(f"  Text R²: {done}/{n_total}")

    return pd.DataFrame(records)


def run_panel_b(
    rank: int, output_dir: Path, dataset: str = "things22k",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compute text R² per dimension, merge with universality, plot scatter."""
    text_r2_path = output_dir / "text_r2_scores.csv"
    if text_r2_path.exists():
        text_r2 = pd.read_csv(text_r2_path)
        logger.info(f"Loaded cached text R² from {text_r2_path}")
    else:
        ds = ThingsDataset()
        unique_cats = sorted(set(ds.categories))
        text_emb_path = output_dir / "text_embeddings.npy"
        compute_text_embeddings(unique_cats, text_emb_path)

        components = load_components_for_rank(rank, dataset=dataset)
        logger.info(f"Computing text R² for {len(components)} models...")
        text_r2 = compute_text_r2(components, text_emb_path, ds)
        text_r2.to_csv(text_r2_path, index=False)

    univ_df = load_universality(rank, dataset=dataset)
    df = text_r2.merge(univ_df[["model", "dimension", "universality"]], on=["model", "dimension"])
    logger.info(f"Panel B: {len(df)} dimensions")

    fig = plot_panel_b(df, output_dir)
    return df, fig


def plot_panel_b(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Universality vs text R² scatter with LOWESS."""
    fig, ax = create_figure("single")

    ax.hexbin(
        df["universality"], df["r2_text"],
        gridsize=40, cmap="Blues", mincnt=1, alpha=0.8, linewidths=0.2,
    )

    x_low, y_low = _lowess_line(
        df["universality"].values, df["r2_text"].values, frac=0.3,
    )
    ax.plot(x_low, y_low, color=ROSE, lw=2, label="LOWESS")

    rho, p = sp_stats.spearmanr(df["universality"], df["r2_text"])
    ax.text(
        0.97, 0.97, f"$\\rho$ = {rho:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color=GRAY_DARK,
    )

    ax.set_xlabel("Universality")
    ax.set_ylabel("Text predictability ($R^2$)")
    ax.legend(fontsize=7, loc="upper left")

    save_figure(fig, output_dir / "panel_b_text")
    logger.info(f"Panel B: Spearman rho={rho:.4f}, p={p:.2e}")
    return fig


# Panel C: Importance–Universality Orthogonality


def compute_importance(
    components: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute within-model fractional importance = ||w_k||^2 / sum_j ||w_j||^2."""
    records = []
    for model_name, W in components.items():
        sq_norms = np.sum(W ** 2, axis=0)
        total = sq_norms.sum()
        fracs = sq_norms / total if total > 0 else np.zeros_like(sq_norms)
        for dim_idx, frac in enumerate(fracs):
            records.append({
                "model": model_name,
                "dimension": dim_idx,
                "importance": frac,
                "sq_norm": sq_norms[dim_idx],
            })
    return pd.DataFrame(records)


def run_panel_c(
    rank: int, output_dir: Path, dataset: str = "things22k",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Full Panel C pipeline: importance computation + plot."""
    components = load_components_for_rank(rank, dataset=dataset)
    imp_df = compute_importance(components)

    univ_df = load_universality(rank, dataset=dataset)
    imp_df = imp_df.merge(
        univ_df[["model", "dimension", "universality"]], on=["model", "dimension"]
    )

    imp_df.to_csv(output_dir / "importance_scores.csv", index=False)
    logger.info(f"Saved importance scores: {len(imp_df)} rows")

    fig = plot_panel_c(imp_df, output_dir)
    return imp_df, fig


def plot_panel_c(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Internal importance vs universality."""
    fig, ax = create_figure("single")

    ax.hexbin(
        df["importance"], df["universality"],
        gridsize=40, cmap="Blues", mincnt=1, alpha=0.8, linewidths=0.2,
    )

    x_low, y_low = _lowess_line(
        df["importance"].values, df["universality"].values, frac=0.3
    )
    ax.plot(x_low, y_low, color=ROSE, lw=2, label="LOWESS")

    rho, p = sp_stats.spearmanr(df["importance"], df["universality"])
    ax.text(
        0.97, 0.97, f"$\\rho$ = {rho:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color=GRAY_DARK,
    )

    ax.set_xlabel("Internal importance (frac. variance)")
    ax.set_ylabel("Universality")
    ax.legend(fontsize=7, loc="upper left")

    save_figure(fig, output_dir / "panel_c_importance")
    logger.info(f"Panel C: Spearman rho={rho:.4f}, p={p:.2e}")
    return fig


# Combined 3-panel figure


def plot_combined(
    sparsity_df: pd.DataFrame,
    gradient_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    output_dir: Path,
) -> plt.Figure:
    """Combined A+B+C as a 1x3 full-width figure."""
    fig, axes = create_figure("full_width", ncols=3, pad_left=0.5, pad_right=0.1)
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.55)

    # Panel A: hexbin scatter
    ax_a.hexbin(
        sparsity_df["hoyer_sparsity"], sparsity_df["universality"],
        gridsize=35, cmap="Blues", mincnt=1, alpha=0.8, linewidths=0.2,
    )
    x_low, y_low = _lowess_line(
        sparsity_df["hoyer_sparsity"].values, sparsity_df["universality"].values
    )
    ax_a.plot(x_low, y_low, color=ROSE, lw=2)
    rho_a, _ = sp_stats.spearmanr(sparsity_df["hoyer_sparsity"], sparsity_df["universality"])
    ax_a.text(
        0.97, 0.97, f"$\\rho$ = {rho_a:.3f}",
        transform=ax_a.transAxes, ha="right", va="top", fontsize=8, color=GRAY_DARK,
    )
    ax_a.set_xlabel("Hoyer sparsity")
    ax_a.set_ylabel("Universality")
    ax_a.set_title("A", fontsize=10, loc="left", fontweight="bold")

    # Panel B: text R² scatter
    ax_b.hexbin(
        gradient_df["universality"], gradient_df["r2_text"],
        gridsize=35, cmap="Blues", mincnt=1, alpha=0.8, linewidths=0.2,
    )
    x_low, y_low = _lowess_line(
        gradient_df["universality"].values, gradient_df["r2_text"].values,
    )
    ax_b.plot(x_low, y_low, color=ROSE, lw=2)
    rho_b, _ = sp_stats.spearmanr(gradient_df["universality"], gradient_df["r2_text"])
    ax_b.text(
        0.97, 0.97, f"$\\rho$ = {rho_b:.3f}",
        transform=ax_b.transAxes, ha="right", va="top", fontsize=8, color=GRAY_DARK,
    )
    ax_b.set_xlabel("Universality")
    ax_b.set_ylabel("Text predictability ($R^2$)")
    ax_b.set_title("B", fontsize=10, loc="left", fontweight="bold")

    # Panel C: hexbin scatter
    ax_c.hexbin(
        imp_df["importance"], imp_df["universality"],
        gridsize=35, cmap="Blues", mincnt=1, alpha=0.8, linewidths=0.2,
    )
    x_low, y_low = _lowess_line(imp_df["importance"].values, imp_df["universality"].values)
    ax_c.plot(x_low, y_low, color=ROSE, lw=2)
    rho_c, _ = sp_stats.spearmanr(imp_df["importance"], imp_df["universality"])
    ax_c.text(
        0.97, 0.97, f"$\\rho$ = {rho_c:.3f}",
        transform=ax_c.transAxes, ha="right", va="top", fontsize=8, color=GRAY_DARK,
    )
    ax_c.set_xlabel("Internal importance")
    ax_c.set_ylabel("Universality")
    ax_c.set_title("C", fontsize=10, loc="left", fontweight="bold")

    save_figure(fig, output_dir / "combined_characterization")
    return fig


# Main


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    logger.info("── Panel A: Sparsity Goldilocks Zone ──")
    sparsity_df = load_sparsity_data()
    plot_panel_a(sparsity_df, out)

    logger.info("── Panel B: Abstraction Gradient ──")
    gradient_df, _ = run_panel_b(cfg.factorization.rank, out, dataset=cfg.dataset)

    logger.info("── Panel C: Importance–Universality Orthogonality ──")
    imp_df, _ = run_panel_c(cfg.factorization.rank, out, dataset=cfg.dataset)

    logger.info("── Combined Figure ──")
    plot_combined(sparsity_df, gradient_df, imp_df, out)

    logger.info(f"All outputs saved to {out}")


if __name__ == "__main__":
    main()
