#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf

from src.colors import INDIGO, TEAL
from src.plotting import save_figure, setup_plotting


def load_w(outputs: Path, dataset: str, model: str, rank: int) -> np.ndarray:
    """W matrix of the sigma/seed selected for this model by the aligned pipeline."""
    selection = pd.read_csv(outputs / "meta" / f"bandwidth_selection_rank_{rank}.csv")
    row = selection[selection["model"] == model].iloc[0]
    w_path = (
        outputs / "embeddings" / dataset / model / f"rank_{rank}"
        / f"sigma_{row['sigma_mult']}" / f"seed_{int(row['best_seed'])}" / "w.npz"
    )
    return np.load(w_path)["w"]


def dim_universality(outputs: Path, model: str, rank: int) -> pd.DataFrame:
    per_dim = pd.read_csv(outputs / "universality" / f"per_dimension_rank_{rank}.csv")
    return per_dim[
        (per_dim["model"] == model) & (per_dim["metric"] == "cos2_perm_adj")
    ].sort_values("dimension")


def render_image(path: Path, size: int = 96) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img)


def panel_for_variant(
    w: np.ndarray,
    udf: pd.DataFrame,
    items: pd.DataFrame,
    image_root: Path,
    color: str,
    ax_grid,
    n_dims_shown: int,
    n_top_images: int,
) -> None:
    sorted_dims = udf.sort_values("universality", ascending=False)
    top = sorted_dims.head(n_dims_shown)["dimension"].astype(int).tolist()
    bot = sorted_dims.tail(n_dims_shown)["dimension"].astype(int).tolist()
    rows = list(zip(top, ["top"] * len(top))) + list(zip(bot, ["bot"] * len(bot)))

    for r, (d, kind) in enumerate(rows):
        u = float(udf[udf["dimension"] == d]["universality"].iloc[0])
        order = np.argsort(w[:, d])[::-1][:n_top_images]
        side = "Universal" if kind == "top" else "Idiosyncratic"
        ax_grid[r, 0].text(
            0.0, 0.5, f"{side}\nDim {d}\nu={u:.2f}",
            ha="left", va="center", fontsize=7,
            color=color if kind == "top" else "#888888",
            transform=ax_grid[r, 0].transAxes,
        )
        ax_grid[r, 0].axis("off")
        for c, idx in enumerate(order):
            ax_grid[r, c + 1].imshow(
                render_image(image_root / items.iloc[int(idx)]["path"])
            )
            ax_grid[r, c + 1].set_xticks([])
            ax_grid[r, c + 1].set_yticks([])
            for spine in ax_grid[r, c + 1].spines.values():
                spine.set_visible(False)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    setup_plotting()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rank = cfg.factorization.rank
    n_top_images = 6
    n_dims_shown = 4
    variants = ["base_model", "alignet"]
    colors = {"base_model": INDIGO, "alignet": TEAL}
    labels = {"base_model": "Base", "alignet": "AligNet"}

    items = pd.read_csv(
        Path(cfg.data_root) / "meta" / "things22k" / "image_info_22k.csv"
    )

    n_rows = 2 * n_dims_shown
    n_cols = 1 + n_top_images

    fig, axes = plt.subplots(
        n_rows, n_cols * len(variants),
        figsize=(1.0 * n_cols * len(variants), 1.0 * n_rows),
        gridspec_kw={
            "wspace": 0.05,
            "hspace": 0.15,
            "width_ratios": ([0.5] + [1] * n_top_images) * len(variants),
        },
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for vi, variant in enumerate(variants):
        sub_axes = axes[:, vi * n_cols:(vi + 1) * n_cols]
        sub_axes[0, 1].set_title(
            labels[variant], fontsize=10, color=colors[variant],
            loc="left", pad=8, fontweight="medium",
        )
        model = f"alignet_vit_b_{variant}"
        panel_for_variant(
            load_w(out, cfg.dataset, model, rank),
            dim_universality(out, model, rank),
            items,
            Path(cfg.data_root),
            colors[variant],
            sub_axes,
            n_dims_shown,
            n_top_images,
        )

    fig.suptitle(
        "ViT-B: top vs bottom universal NMF dimensions (top-activating THINGS images)",
        fontsize=11, y=0.995,
    )

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, figure_dir / "vit_b_dimensions", formats=("pdf", "png"))


if __name__ == "__main__":
    main()
