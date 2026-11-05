#!/usr/bin/env python3
"""Render the four high-resolution quadrant panels for each rated dimension.

Each panel set reproduces the 8x8 overview grid that human raters saw, split
into four 4x4 quadrants so a vision model can read the images at full size.
The reconstruction is verified pixel-for-pixel against the cached overview
grid before anything is written.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageChops

from .run import PANEL_FILENAMES, atomic_write_json, load_dimensions


def parse_dimension_id(dim_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<model>.+)__d(?P<dimension>\d+)", dim_id)
    if not match:
        raise ValueError(f"Invalid dimension ID: {dim_id}")
    return match.group("model"), int(match.group("dimension"))


def load_selection_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def load_image_metadata(
    path: Path, images_root: Path
) -> tuple[list[Path], list[str]]:
    """Read the image catalog in row order, resolved against the image root."""
    image_paths = []
    categories = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_paths.append(images_root / row["category"] / row["filename"])
            categories.append(row["category"])
    return image_paths, categories


def load_weights(
    model: str,
    selection_rows: dict[str, dict[str, str]],
    embeddings_dir: Path,
    rank: int,
) -> tuple[np.ndarray, Path]:
    if model not in selection_rows:
        raise KeyError(f"No rank-{rank} sigma selection for {model}")
    row = selection_rows[model]
    directory = (
        embeddings_dir
        / model
        / f"rank_{rank}"
        / f"sigma_{row['sigma_mult']}"
        / f"seed_{int(row['best_seed'])}"
    )
    npz_path = directory / "w.npz"
    npy_path = directory / "w.npy"
    if npz_path.is_file():
        return np.load(npz_path)["w"], npz_path
    if npy_path.is_file():
        return np.load(npy_path), npy_path
    raise FileNotFoundError(f"No W matrix in {directory}")


def select_category_diverse_indices(
    weights: np.ndarray,
    categories: list[str],
    count: int = 64,
) -> list[int]:
    selected = []
    seen = set()
    for raw_index in np.argsort(-weights):
        index = int(raw_index)
        category = categories[index]
        if category in seen:
            continue
        selected.append(index)
        seen.add(category)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only found {len(selected)} category-diverse images")
    return selected


def render_montage(
    indices: list[int],
    image_paths: list[Path],
    rows: int,
    columns: int,
    thumbnail_pixels: int,
) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (columns * thumbnail_pixels, rows * thumbnail_pixels),
        (255, 255, 255),
    )
    for position, index in enumerate(indices):
        with Image.open(image_paths[index]) as source:
            tile = source.convert("RGB").resize(
                (thumbnail_pixels, thumbnail_pixels),
                Image.Resampling.LANCZOS,
            )
        row, column = divmod(position, columns)
        canvas.paste(tile, (column * thumbnail_pixels, row * thumbnail_pixels))
    return canvas


def quadrant_positions() -> list[list[int]]:
    return [
        [row * 8 + column for row in range(0, 4) for column in range(0, 4)],
        [row * 8 + column for row in range(0, 4) for column in range(4, 8)],
        [row * 8 + column for row in range(4, 8) for column in range(0, 4)],
        [row * 8 + column for row in range(4, 8) for column in range(4, 8)],
    ]


def prepare_dimension(
    *,
    dim_id: str,
    weights: np.ndarray,
    matrix_path: Path,
    image_paths: list[Path],
    categories: list[str],
    grid_dir: Path,
    output_dir: Path,
    thumbnail_pixels: int = 224,
) -> None:
    model, dimension = parse_dimension_id(dim_id)
    target = output_dir / dim_id
    expected = [target / filename for filename in PANEL_FILENAMES]
    manifest_path = target / "manifest.json"
    if manifest_path.is_file() and all(path.is_file() for path in expected):
        print(f"{dim_id}: already prepared")
        return
    if dimension >= weights.shape[1]:
        raise IndexError(f"{dim_id}: dimension outside W shape {weights.shape}")
    if weights.shape[0] != len(image_paths):
        raise ValueError(
            f"{dim_id}: W rows {weights.shape[0]} != metadata rows {len(image_paths)}"
        )

    selected = select_category_diverse_indices(weights[:, dimension], categories)
    cached_grid = Image.open(grid_dir / f"{dim_id}.png").convert("RGB")
    recreated_grid = render_montage(selected, image_paths, 8, 8, 88)
    if ImageChops.difference(cached_grid, recreated_grid).getbbox() is not None:
        raise ValueError(f"{dim_id}: reconstructed selection differs from cached grid")

    target.mkdir(parents=True, exist_ok=True)
    positions_by_panel = quadrant_positions()
    for filename, positions in zip(PANEL_FILENAMES, positions_by_panel):
        indices = [selected[position] for position in positions]
        panel = render_montage(
            indices,
            image_paths,
            rows=4,
            columns=4,
            thumbnail_pixels=thumbnail_pixels,
        )
        panel.save(target / filename, format="PNG", optimize=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dim_id": dim_id,
        "model": model,
        "dimension": dimension,
        "matrix_path": str(matrix_path),
        "overview_grid": str(grid_dir / f"{dim_id}.png"),
        "overview_thumbnail_pixels": 88,
        "panel_thumbnail_pixels": thumbnail_pixels,
        "panel_layout": "four 4x4 quadrants in reading order",
        "selection_verified_against_cached_grid": True,
        "selected_images": [
            {
                "grid_position": position,
                "source_index": index,
                "category": categories[index],
                "path": str(image_paths[index]),
            }
            for position, index in enumerate(selected)
        ],
    }
    atomic_write_json(manifest_path, manifest)
    print(f"{dim_id}: wrote four {thumbnail_pixels * 4}px panels")


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load("config.yaml"), OmegaConf.from_cli())
    out = Path(__file__).parent / "outputs" / "panel_cache_dimensions"
    out.mkdir(parents=True, exist_ok=True)
    rank = cfg.factorization.rank
    grid_dir = Path("experiments/universality/notebooks/grid_cache_dimensions")

    dimensions = load_dimensions(
        Path(cfg.data_root) / "ratings" / "dim_ratings_aggregated_qc_attn_catch.csv"
    )
    image_paths, categories = load_image_metadata(
        Path(cfg.data_root) / "meta" / "things22k" / "image_info_22k.csv",
        Path(cfg.data_root) / "images" / "things",
    )
    selection_rows = load_selection_rows(
        Path(cfg.data_root) / "meta" / cfg.dataset / f"bandwidth_selection_rank_{rank}.csv"
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in dimensions:
        model, _ = parse_dimension_id(row["dim_id"])
        grouped[model].append(row["dim_id"])

    for model, dim_ids in grouped.items():
        weights, matrix_path = load_weights(
            model,
            selection_rows,
            Path(cfg.data_root) / "embeddings" / cfg.dataset,
            rank,
        )
        for dim_id in dim_ids:
            prepare_dimension(
                dim_id=dim_id,
                weights=weights,
                matrix_path=matrix_path,
                image_paths=image_paths,
                categories=categories,
                grid_dir=grid_dir,
                output_dir=out,
            )


if __name__ == "__main__":
    main()
