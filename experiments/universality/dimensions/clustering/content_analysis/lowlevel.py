from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

log = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "outputs" / "image_stats_cache.npz"


def compute_image_stats(image_paths: list[Path], target_size: int = 128) -> dict[str, np.ndarray]:
    """Compute low-level statistics for each image.

    Returns dict with keys: brightness, edge_density, color_entropy, spatial_freq.
    Each value is shape (n_images,).
    """
    n = len(image_paths)
    stats = {
        "brightness": np.zeros(n),
        "edge_density": np.zeros(n),
        "color_entropy": np.zeros(n),
        "spatial_freq": np.zeros(n),
    }

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB").resize((target_size, target_size))
            arr = np.array(img, dtype=np.float32) / 255.0

            # brightness: mean luminance
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            stats["brightness"][i] = gray.mean()

            # edge density: fraction of strong Sobel edges
            sx = ndimage.sobel(gray, axis=0)
            sy = ndimage.sobel(gray, axis=1)
            edges = np.hypot(sx, sy)
            stats["edge_density"][i] = edges.mean()

            # color entropy: entropy of the hue histogram
            hsv_approx = arr.reshape(-1, 3)
            hist, _ = np.histogram(hsv_approx[:, 0] - hsv_approx[:, 2], bins=32, density=True)
            hist = hist[hist > 0]
            stats["color_entropy"][i] = -np.sum(hist * np.log2(hist + 1e-10))

            # spatial frequency: mean power in high frequencies (FFT)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(np.abs(fft))
            cy, cx = target_size // 2, target_size // 2
            Y, X = np.ogrid[:target_size, :target_size]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            high_freq_mask = dist > target_size // 4
            stats["spatial_freq"][i] = fft_shift[high_freq_mask].mean() / (fft_shift.mean() + 1e-10)

        except Exception as e:
            log.warning("Failed to process image %d (%s): %s", i, path, e)

        if (i + 1) % 5000 == 0:
            log.info("Processed %d/%d images", i + 1, n)

    return stats


def load_or_compute_image_stats(image_paths: list[Path]) -> dict[str, np.ndarray]:
    """Load cached stats or compute them."""
    if CACHE_PATH.exists():
        log.info("Loading cached image stats from %s", CACHE_PATH)
        data = np.load(CACHE_PATH)
        return {k: data[k] for k in data.files}

    log.info("Computing image statistics for %d images...", len(image_paths))
    stats = compute_image_stats(image_paths)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **stats)
    log.info("Cached image stats to %s", CACHE_PATH)
    return stats


def _permutation_null(proto: np.ndarray, stat_matrix: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Compute null distribution of max |r| across stats under permutation.

    Returns shape (n_perm,) array of max |r| values from permuted prototypes.
    """
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(proto)
        # correlate permuted prototype with each stat, take max |r|
        rs = np.abs(np.corrcoef(perm, stat_matrix)[0, 1:])
        null_max[i] = rs.max()
    return null_max


def analyze_lowlevel(
    profiles: pd.DataFrame,
    prototypes: dict[str, np.ndarray],
    image_paths: list[Path],
    n_perm: int = 500,
) -> pd.DataFrame:
    """Correlate each cluster prototype with low-level image features.

    Returns DataFrame with per-cluster correlations for each statistic,
    a summary max_lowlevel_r column, and permutation-corrected z-scores
    that account for the mechanical relationship between prototype sparsity
    and correlation variance.
    """
    stats = load_or_compute_image_stats(image_paths)
    stat_names = list(stats.keys())

    # stack stats into matrix for vectorized permutation test
    stat_matrix = np.stack([stats[name] for name in stat_names])  # (4, n_images)

    rng = np.random.default_rng(42)
    records = []
    for idx, (_, row) in enumerate(profiles.iterrows()):
        cid = str(row["cluster"])
        proto = prototypes.get(cid)
        if proto is None:
            continue

        entry = {"cluster": cid, "model_coverage": row["model_coverage"]}

        # actual correlations
        correlations = []
        for i, name in enumerate(stat_names):
            r = float(np.corrcoef(proto, stats[name])[0, 1])
            entry[f"r_{name}"] = r
            correlations.append(abs(r))

        actual_max_r = max(correlations) if correlations else 0.0
        entry["max_lowlevel_r"] = actual_max_r

        # permutation test: shuffle prototype to get null distribution
        # of max |r| expected for a vector with this sparsity pattern
        null_max = _permutation_null(proto, stat_matrix, n_perm, rng)
        null_mean = null_max.mean()
        null_std = null_max.std()
        entry["max_lowlevel_z"] = (actual_max_r - null_mean) / null_std if null_std > 0 else 0.0
        entry["max_lowlevel_p"] = float((null_max >= actual_max_r).mean())
        entry["null_mean"] = null_mean
        entry["null_std"] = null_std

        records.append(entry)

        if (idx + 1) % 200 == 0:
            log.info("Permutation tests: %d/%d clusters", idx + 1, len(profiles))

    result = pd.DataFrame(records)
    n_sig = (result["max_lowlevel_p"] < 0.05).sum()
    log.info("Low-level analysis: %d clusters, max |r| range [%.3f, %.3f], "
             "%d significant after permutation correction (p<0.05)",
             len(result), result["max_lowlevel_r"].min(), result["max_lowlevel_r"].max(), n_sig)
    return result
