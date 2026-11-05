from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import isotonic_regression

from .._common import (
    RandomStateLike,
    n_jobs_for_tasks,
    observation_mask,
    replace_missing_with_nan,
    symmetrize_observations,
)
from ._coherence import _coherence, _top_eigenpairs
from ._stability import _select_spectral_cutoff

_MAX_SAMPLING_FRACTION = 0.95


@dataclass(frozen=True, slots=True)
class CVCalibration:
    spectral_cutoff: int
    sampling_fraction: float
    eigvals: np.ndarray
    leakage: np.ndarray
    sampling_grid: np.ndarray
    signal_loss_raw: np.ndarray
    signal_loss_monotone: np.ndarray
    detectability_floor: float
    n_features_in: int


def calibrate_cross_validation(
    similarity_matrix: np.ndarray,
    *,
    max_eigenpairs: int | None = None,
    sampling_grid: np.ndarray | None = None,
    signal_loss_tolerance: float = 0.10,
    leakage_min_fraction: float = 0.85,
    n_bootstrap: int = 50,
    random_state: RandomStateLike = 0,
    n_jobs: int | None = -1,
    missing_values: float | None = np.nan,
) -> CVCalibration:
    (
        sampling_grid,
        signal_loss_tolerance,
        leakage_min_fraction,
        n_bootstrap,
    ) = _validate_args(
        sampling_grid,
        signal_loss_tolerance,
        leakage_min_fraction,
        n_bootstrap,
    )

    similarity, observed_entries = _observed_similarity(
        similarity_matrix,
        missing_values,
    )
    eigvals, eigvecs = _top_eigenpairs_for_calibration(
        similarity,
        max_eigenpairs,
    )

    coherence, retained_mass = _estimate_coherence(
        similarity,
        observed_entries,
        eigvecs,
        sampling_grid,
        n_bootstrap,
        random_state,
        n_jobs,
    )

    return _calibration_from_stability_estimate(
        eigvals,
        coherence,
        retained_mass,
        sampling_grid,
        signal_loss_tolerance,
        leakage_min_fraction,
        len(similarity_matrix),
    )


def _calibration_from_stability_estimate(
    eigvals: np.ndarray,
    coherence: np.ndarray,
    retained_mass: np.ndarray,
    sampling_grid: np.ndarray,
    signal_loss_tolerance: float,
    leakage_min_fraction: float,
    n_features: int,
) -> CVCalibration:
    spectral_cutoff, leakage = _calibrate_spectral_cutoff(
        coherence,
        sampling_grid,
        leakage_min_fraction,
    )
    signal_loss_raw = _signal_loss_by_sampling_probability(
        retained_mass,
        eigvals,
        spectral_cutoff,
    )
    signal_loss_monotone = _monotone_signal_loss(signal_loss_raw)
    detectability_floor = _detectability_floor(eigvals, spectral_cutoff)
    sampling_fraction = _sampling_fraction_from_signal_loss(
        sampling_grid,
        signal_loss_monotone,
        detectability_floor,
        signal_loss_tolerance,
    )

    return CVCalibration(
        spectral_cutoff=int(spectral_cutoff),
        sampling_fraction=float(sampling_fraction),
        eigvals=eigvals,
        leakage=leakage,
        sampling_grid=sampling_grid,
        signal_loss_raw=signal_loss_raw,
        signal_loss_monotone=signal_loss_monotone,
        detectability_floor=detectability_floor,
        n_features_in=int(n_features),
    )


def _top_eigenpairs_for_calibration(
    similarity: np.ndarray,
    max_eigenpairs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_eigenpairs = _eigenpair_count(max_eigenpairs, similarity.shape[0])
    return _top_eigenpairs(similarity, n_eigenpairs)


def _estimate_coherence(
    similarity: np.ndarray,
    observed_entries: np.ndarray,
    eigenvecs: np.ndarray,
    sampling_grid: np.ndarray,
    n_bootstrap: int,
    random_state: RandomStateLike,
    n_jobs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_worker_jobs = n_jobs_for_tasks(n_jobs, n_tasks=sampling_grid.size)
    return _coherence(
        similarity,
        observed_entries,
        eigenvecs,
        sampling_grid,
        n_bootstrap,
        random_state,
        n_worker_jobs,
    )


def _calibrate_spectral_cutoff(
    coherence: np.ndarray,
    sampling_grid: np.ndarray,
    leakage_min_fraction: float,
) -> tuple[int, np.ndarray]:
    spectral_cutoff, leakage = _select_spectral_cutoff(
        coherence,
        sampling_grid,
        leakage_min_fraction,
    )
    return spectral_cutoff, leakage


def _observed_similarity(
    similarity_matrix: np.ndarray,
    missing_values: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = replace_missing_with_nan(similarity_matrix, missing_values)
    similarity = symmetrize_observations(similarity)
    observed_entries = observation_mask(similarity)
    np.fill_diagonal(observed_entries, True)
    return np.nan_to_num(similarity, nan=0.0), observed_entries


def _eigenpair_count(max_eigenpairs: int | None, n_features: int) -> int:
    if max_eigenpairs is None:
        max_eigenpairs = max(min(n_features // 4, 100), 2)
    return max(1, min(int(max_eigenpairs), n_features))


def _prepare_sampling_grid(sampling_grid: np.ndarray | None) -> np.ndarray:
    if sampling_grid is None:
        sampling_grid = np.linspace(0.05, 0.95, 20)
    sampling_grid = np.asarray(sampling_grid, dtype=np.float64)
    if sampling_grid.ndim != 1 or sampling_grid.size == 0:
        raise ValueError("sampling_grid must be a non-empty one-dimensional array")
    if not np.all((0.0 < sampling_grid) & (sampling_grid < 1.0)):
        raise ValueError("sampling_grid values must be in (0, 1)")
    return np.sort(sampling_grid)


def _sampling_fraction_from_signal_loss(
    sampling_grid: np.ndarray,
    signal_loss_monotone: np.ndarray,
    detectability_floor: float,
    signal_loss_tolerance: float,
) -> float:
    sampling_fraction = _smallest_p_below_tolerance(
        sampling_grid,
        signal_loss_monotone,
        signal_loss_tolerance,
    )
    sampling_fraction = max(sampling_fraction, detectability_floor)
    sampling_fraction = min(sampling_fraction, _MAX_SAMPLING_FRACTION)
    return float(sampling_fraction)


def _signal_loss_by_sampling_probability(
    retained_mass: np.ndarray,
    eigvals: np.ndarray,
    spectral_cutoff: int,
) -> np.ndarray:
    median_mass = np.median(retained_mass, axis=2)
    ref_mass = max(float(eigvals[:spectral_cutoff].sum()), 1e-12)
    return 1.0 - median_mass[spectral_cutoff - 1] / ref_mass


def _monotone_signal_loss(signal_loss_raw: np.ndarray) -> np.ndarray:
    return isotonic_regression(signal_loss_raw, increasing=False).x


def _smallest_p_below_tolerance(
    sampling_grid: np.ndarray,
    signal_loss_monotone: np.ndarray,
    signal_loss_tolerance: float,
) -> float:
    if signal_loss_monotone[-1] >= signal_loss_tolerance:
        return float(sampling_grid[-1])
    if signal_loss_monotone[0] <= signal_loss_tolerance:
        return float(sampling_grid[0])
    index = int(
        np.searchsorted(
            -signal_loss_monotone,
            -signal_loss_tolerance,
            side="left",
        )
    )
    index = max(min(index, len(signal_loss_monotone) - 1), 1)
    loss_before = signal_loss_monotone[index - 1]
    loss_after = signal_loss_monotone[index]
    p_before = sampling_grid[index - 1]
    p_after = sampling_grid[index]
    if loss_before == loss_after:
        return float(p_before)
    weight = (loss_before - signal_loss_tolerance) / (loss_before - loss_after)
    return float(p_before + weight * (p_after - p_before))


def _detectability_floor(eigvals: np.ndarray, spectral_cutoff: int) -> float:
    lam_k = (
        float(eigvals[spectral_cutoff - 1])
        if spectral_cutoff >= 1
        else float(eigvals[0])
    )
    lam_kp1 = float(eigvals[min(spectral_cutoff, len(eigvals) - 1)])
    if lam_k <= 1e-12:
        return 1.0
    gap_squared = (lam_kp1 / lam_k) ** 2
    return float(np.clip(gap_squared / (1.0 + gap_squared), 0.0, 1.0))


def _validate_args(
    sampling_grid: np.ndarray | None,
    signal_loss_tolerance: float,
    leakage_min_fraction: float,
    n_bootstrap: int,
) -> tuple[np.ndarray, float, float, int]:
    sampling_grid = _prepare_sampling_grid(sampling_grid)
    signal_loss_tolerance = float(signal_loss_tolerance)
    leakage_min_fraction = float(leakage_min_fraction)
    n_bootstrap = int(n_bootstrap)

    if not (0.0 < signal_loss_tolerance < 1.0):
        raise ValueError("signal_loss_tolerance must be in (0, 1)")
    if not (0.0 < leakage_min_fraction < 1.0):
        raise ValueError("leakage_min_fraction must be in (0, 1)")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}")
    return sampling_grid, signal_loss_tolerance, leakage_min_fraction, n_bootstrap
