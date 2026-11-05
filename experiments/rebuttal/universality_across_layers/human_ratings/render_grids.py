"""Render the 8x8 image grids shown to raters in the layer-wise rating study.

Grid construction is identical to the penultimate-layer study: for each selected
dimension, the top 64 images by weight with at most one image per THINGS
category, pasted as an 8x8 grid of 88px thumbnails.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

log = logging.getLogger(__name__)


def top_images_one_per_category(w_vec: np.ndarray, categories: list[str],
                                n: int) -> list[int]:
    """Top-n image indices by weight, max one image per THINGS category."""
    selected, seen = [], set()
    for idx in np.argsort(w_vec)[::-1]:
        cat = categories[idx]
        if cat in seen:
            continue
        seen.add(cat)
        selected.append(int(idx))
        if len(selected) >= n:
            break
    return selected


def render_grid(indices: list[int], paths: list[Path], out_path: Path,
                grid_size: int, thumb_px: int, gap_px: int) -> None:
    side = grid_size * thumb_px + (grid_size - 1) * gap_px
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    for i, idx in enumerate(indices):
        img = Image.open(paths[idx]).convert("RGB").resize(
            (thumb_px, thumb_px), Image.LANCZOS)
        row, col = divmod(i, grid_size)
        canvas.paste(img, (col * (thumb_px + gap_px), row * (thumb_px + gap_px)))
    canvas.save(out_path, format="PNG")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    # The penultimate study used gapless grids; the gap was added for
    # readability and set gap_px = 0 to reproduce that layout exactly.
    grid_size, thumb_px, gap_px = 8, 88, 4
    rank = cfg.factorization.rank
    embeddings = Path("experiments/rebuttal/universality_across_layers/factorization/outputs/embeddings")

    info = pd.read_csv(Path(cfg.data_root) / "meta" / "things22k" / "image_info_22k.csv")
    categories = info.category.tolist()
    paths = [Path(cfg.data_root) / "images" / "things" / c / f
             for c, f in zip(info.category, info.filename)]
    sel = pd.read_csv(
        Path("experiments/rebuttal/universality_across_layers/factorization/outputs")
        / f"bandwidth_selection_full_rank_{rank}.csv"
    ).set_index("model")

    for layer in ("early", "middle", "late"):
        reps = pd.read_csv(
            Path("experiments/rebuttal/universality_across_layers/human_ratings/outputs")
            / f"representatives_{layer}.csv"
        )
        layer_dir = out / layer
        layer_dir.mkdir(parents=True, exist_ok=True)

        n_done = 0
        for repr_name, group in reps.groupby("repr"):
            r = sel.loc[repr_name]
            w = np.load(embeddings / repr_name / f"rank_{rank}" / f"sigma_{r.sigma_mult}" /
                        f"seed_{int(r.best_seed)}" / "w.npz")["w"]
            for _, row in group.iterrows():
                out_path = layer_dir / f"{row.dim_id}.png"
                if out_path.exists():
                    continue
                idx = top_images_one_per_category(w[:, int(row.dimension)], categories,
                                                  grid_size * grid_size)
                render_grid(idx, paths, out_path, grid_size, thumb_px, gap_px)
                n_done += 1
                if n_done % 50 == 0:
                    log.info(f"{n_done} grids rendered")
        log.info(f"{layer}: {n_done} new grids -> {layer_dir} "
                 f"(total {len(list(layer_dir.glob('*.png')))})")


if __name__ == "__main__":
    main()
