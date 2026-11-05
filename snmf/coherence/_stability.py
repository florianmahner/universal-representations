from __future__ import annotations

import numpy as np


def _select_spectral_cutoff(
    coherence: np.ndarray,
    sampling_grid: np.ndarray,
    leakage_quantile: float,
    min_cutoff: int = 2,
    min_segment_size: int = 2,
) -> tuple[int, np.ndarray]:
    coherence_median = np.median(coherence, axis=2)
    leakage = _leakage_profile(
        coherence_median,
        sampling_grid,
        leakage_quantile,
    )
    spectral_cutoff = _find_leakage_changepoint(
        leakage,
        min_cutoff=min_cutoff,
        min_segment_size=min_segment_size,
    )
    return spectral_cutoff, leakage


def _leakage_profile(
    coherence_median: np.ndarray,
    sampling_grid: np.ndarray,
    leakage_quantile: float,
) -> np.ndarray:
    threshold = np.quantile(sampling_grid, leakage_quantile)
    selected_grid = np.flatnonzero(sampling_grid >= threshold)
    if selected_grid.size == 0:
        selected_grid = np.array([len(sampling_grid) - 1])
    fractions = sampling_grid[selected_grid]
    weights = fractions / np.maximum(1.0 - fractions, 1e-12)
    leakage = (1.0 - coherence_median[:, selected_grid]) * weights[np.newaxis, :]
    return np.median(leakage, axis=1).astype(np.float64)


def _find_leakage_changepoint(
    leakage: np.ndarray,
    min_cutoff: int,
    min_segment_size: int,
) -> int:
    n_dimensions = len(leakage)
    if n_dimensions < 2 * min_segment_size:
        return n_dimensions
    scores = np.full(n_dimensions - 1, -np.inf)
    for split in range(min_segment_size - 1, n_dimensions - min_segment_size):
        scores[split] = _score_mean_gap(leakage[: split + 1], leakage[split + 1 :])
    first_split = max(min_cutoff - 1, 0) if min_cutoff <= n_dimensions - 1 else 0
    return int(first_split + np.nanargmax(scores[first_split:])) + 1


def _score_mean_gap(left: np.ndarray, right: np.ndarray) -> float:
    n_left, n_right = len(left), len(right)
    within_segment_sse = ((left - left.mean()) ** 2).sum() + (
        (right - right.mean()) ** 2
    ).sum()
    pooled_variance = within_segment_sse / max(n_left + n_right - 2, 1)
    if pooled_variance <= 1e-12:
        return np.inf
    harmonic_size = n_left * n_right / (n_left + n_right)
    mean_gap_squared = (right.mean() - left.mean()) ** 2
    return harmonic_size * mean_gap_squared / pooled_variance
