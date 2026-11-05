from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

from ._common import (
    RandomStateLike,
    blas_limits_for_workers,
    n_jobs_for_tasks,
    observation_mask,
    replace_missing_with_nan,
    seed_stream,
    symmetrize_observations,
    validate_n_jobs,
)
from .coherence import CVCalibration, calibrate_cross_validation
from .model import SNMF

_RESERVED_SNMF_KWARGS = frozenset({"rank", "bounds", "missing_values", "random_state"})


@dataclass(frozen=True, slots=True)
class CVResult:
    model_rank: int
    fold_scores: pd.DataFrame
    rank_scores: pd.DataFrame
    spectral_cutoff: int | None
    sampling_fraction: float
    candidate_ranks: tuple[int, ...]
    calibration: CVCalibration | None = None


@dataclass(frozen=True, slots=True)
class _CVSplit:
    repeat: int
    fold: int
    train_mask: np.ndarray
    validation_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _CVJob:
    repeat: int
    fold: int
    rank: int
    split_seed: int
    fit_seed: int


def cross_val_score(
    similarity_matrix: np.ndarray,
    ranks: Sequence[int],
    sampling_fraction: float | None = None,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: RandomStateLike = 0,
    n_jobs: int | None = -1,
    threads_per_worker: int | str | None = None,
    missing_values: float | None = np.nan,
    snmf_kwargs: Mapping[str, object] | None = None,
) -> CVResult:
    _check_args(n_folds, n_repeats, n_jobs)
    fit_kwargs = _checked_snmf_kwargs(snmf_kwargs)

    s = _prepare_similarity(similarity_matrix, missing_values)
    n = s.shape[0]
    ranks = _validate_ranks(ranks, n)
    bounds = _observed_bounds(s)
    sampling_fraction, calibration = _cv_sampling_fraction(
        s, sampling_fraction, random_state, n_jobs
    )

    fold_fraction = get_fold_fraction(sampling_fraction, n_folds, n)
    fold_scores = _cross_validate_ranks(
        s,
        ranks,
        fold_fraction,
        n_folds,
        n_repeats,
        random_state,
        n_jobs,
        threads_per_worker,
        bounds,
        fit_kwargs,
    )
    rank_scores = _summarize(fold_scores)

    return CVResult(
        model_rank=_best_rank(rank_scores),
        fold_scores=fold_scores,
        rank_scores=rank_scores,
        spectral_cutoff=_spectral_cutoff(calibration),
        sampling_fraction=_training_fraction(fold_fraction, n_folds),
        candidate_ranks=ranks,
        calibration=calibration,
    )


def _prepare_similarity(
    similarity_matrix: np.ndarray,
    missing_values: float | None,
) -> np.ndarray:
    return symmetrize_observations(
        replace_missing_with_nan(similarity_matrix, missing_values)
    )


def _check_args(n_folds: int, n_repeats: int, n_jobs: int | None) -> None:
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    validate_n_jobs(n_jobs)


def _checked_snmf_kwargs(snmf_kwargs: Mapping[str, object] | None) -> dict:
    kwargs = dict(snmf_kwargs) if snmf_kwargs else {}
    reserved = _RESERVED_SNMF_KWARGS & kwargs.keys()
    if reserved:
        raise ValueError(
            f"snmf_kwargs must not contain {sorted(reserved)}; these are set per fit"
        )
    return kwargs


def _validate_ranks(ranks: Sequence[int], n: int) -> tuple[int, ...]:
    ranks = tuple(dict.fromkeys(int(r) for r in ranks))
    if not ranks:
        raise ValueError("ranks must contain at least one value")
    if any(r < 1 for r in ranks):
        raise ValueError("ranks must be positive")
    if any(r > n for r in ranks):
        raise ValueError(f"ranks must be at most n_features={n}")
    return ranks


def _validate_fraction(fraction: float) -> float:
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("sampling_fraction must be in (0, 1)")
    return fraction


def _observed_bounds(s: np.ndarray) -> tuple[float, float]:
    observed = observation_mask(s)
    values = s[observed & np.isfinite(s)]
    if values.size == 0:
        raise ValueError("similarity_matrix has no finite observed entries")
    return float(values.min()), float(values.max())


def _cv_sampling_fraction(
    s: np.ndarray,
    sampling_fraction: float | None,
    random_state: RandomStateLike,
    n_jobs: int | None,
) -> tuple[float, CVCalibration | None]:
    if sampling_fraction is not None:
        return _validate_fraction(sampling_fraction), None
    calibration = calibrate_cross_validation(
        s, random_state=random_state, n_jobs=n_jobs
    )
    return _validate_fraction(calibration.sampling_fraction), calibration


def _spectral_cutoff(calibration: CVCalibration | None) -> int | None:
    return None if calibration is None else int(calibration.spectral_cutoff)


def _cv_seeds(
    random_state: RandomStateLike, n_repeats: int, n_folds: int, n_ranks: int
) -> tuple[np.ndarray, np.ndarray]:
    seeds = seed_stream(random_state)
    split_seeds = np.fromiter(seeds, dtype=np.uint32, count=n_repeats)
    fit_seeds = np.fromiter(seeds, dtype=np.uint32, count=n_repeats * n_folds * n_ranks)
    return split_seeds, fit_seeds.reshape(n_repeats, n_folds, n_ranks)


def _observed_pair_ids(
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates and flat triu ids of observed upper-triangle entries.

    The ids match np.flatnonzero(observed[np.triu_indices(n, k=1)]) in
    values, order and dtype without materializing the O(n^2) triu index
    arrays.
    """
    n = observed.shape[0]
    rows, cols = np.nonzero(np.triu(observed, k=1))
    ids = rows * (2 * n - rows - 1) // 2 + (cols - rows - 1)
    return ids, rows, cols


def _split_fold_ids(
    observed: np.ndarray,
    fold_fraction: float,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Regenerate one repeat's entrywise split in pair-id space.

    Replicates the exact RandomState call sequence of the original
    mask-based implementation (choice over observed pair ids, then a
    permutation of the sorted sample), so the resulting folds are
    bit-identical to the dense-mask splits for the same seed.
    """
    ids, rows, cols = _observed_pair_ids(observed)
    if ids.size == 0:
        raise ValueError("cross-validation needs observed off-diagonal entries")
    rng = check_random_state(int(seed))
    n_keep = max(1, int(fold_fraction * ids.size))
    kept = rng.choice(ids, size=n_keep, replace=False)
    sampled = np.sort(kept)
    if sampled.size < n_folds:
        raise ValueError(
            f"cross-validation needs at least {n_folds} fold-sampled "
            f"off-diagonal entries; got {sampled.size}"
        )
    folds = np.array_split(rng.permutation(sampled), n_folds)
    return ids, rows, cols, sampled, folds


def _ids_to_mask(
    target_ids: np.ndarray,
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n: int,
) -> np.ndarray:
    pos = np.searchsorted(ids, target_ids)
    mask = np.zeros((n, n), dtype=bool)
    mask[rows[pos], cols[pos]] = True
    mask[cols[pos], rows[pos]] = True
    return mask


def _entrywise_splits(
    observed: np.ndarray,
    fold_fraction: float,
    n_folds: int,
    split_seeds: np.ndarray,
) -> list[_CVSplit]:
    """Dense-mask splits, kept for inspection and tests.

    The scoring path regenerates splits per job via _split_fold_ids and
    never materializes these masks.
    """
    n = observed.shape[0]
    splits: list[_CVSplit] = []
    for repeat, seed in enumerate(split_seeds):
        ids, rows, cols, sampled, folds = _split_fold_ids(
            observed, fold_fraction, n_folds, int(seed)
        )
        for fold, fold_ids in enumerate(folds):
            validation = _ids_to_mask(np.sort(fold_ids), ids, rows, cols, n)
            train_ids = np.setdiff1d(sampled, fold_ids, assume_unique=True)
            train = _ids_to_mask(train_ids, ids, rows, cols, n)
            diag = np.diag_indices_from(train)
            train[diag] = observed[diag]
            splits.append(_CVSplit(repeat, fold, train, validation))
    return splits


def get_fold_fraction(sampling_fraction: float, n_folds: int, n: int) -> float:
    cap = _max_fold_fraction(n)
    requested = sampling_fraction * n_folds / (n_folds - 1)
    if requested > cap:
        warnings.warn(
            f"Requested fold fraction {requested:.3f} exceeds the cap {cap:.3f}. "
            f"Each fold will train at {_training_fraction(cap, n_folds):.3f} "
            f"instead of {sampling_fraction:.3f}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return min(requested, cap)


def _training_fraction(fold_fraction: float, n_folds: int) -> float:
    return float(fold_fraction * (n_folds - 1) / n_folds)


def _max_fold_fraction(
    n: int, floor: float = 0.95, holdout_budget: int = 2000
) -> float:
    n_pairs = n * (n - 1) / 2
    if n_pairs <= 0:
        return floor
    return float(max(floor, 1.0 - holdout_budget / n_pairs))


def _cross_validate_ranks(
    s: np.ndarray,
    ranks: tuple[int, ...],
    fold_fraction: float,
    n_folds: int,
    n_repeats: int,
    random_state: RandomStateLike,
    n_jobs: int | None,
    threads_per_worker: int | str | None,
    bounds: tuple[float, float],
    fit_kwargs: dict,
) -> pd.DataFrame:
    split_seeds, fit_seeds = _cv_seeds(random_state, n_repeats, n_folds, len(ranks))
    jobs = [
        _CVJob(repeat, fold, rank, int(split_seeds[repeat]), int(fit_seeds[repeat, fold, i]))
        for repeat in range(n_repeats)
        for fold in range(n_folds)
        for i, rank in enumerate(ranks)
    ]
    return _score_jobs(
        s, jobs, fold_fraction, n_folds, n_jobs, threads_per_worker, bounds, fit_kwargs
    )


def _score_jobs(
    s: np.ndarray,
    jobs: list[_CVJob],
    fold_fraction: float,
    n_folds: int,
    n_jobs: int | None,
    threads_per_worker: int | str | None,
    bounds: tuple[float, float],
    fit_kwargs: dict,
) -> pd.DataFrame:
    n_jobs = n_jobs_for_tasks(n_jobs, len(jobs))
    limits = blas_limits_for_workers(threads_per_worker, n_jobs)

    if n_jobs == 1:
        scores = [
            _score_job(s, job, fold_fraction, n_folds, bounds, fit_kwargs, limits)
            for job in jobs
        ]
    else:
        scores = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_score_job)(
                s, job, fold_fraction, n_folds, bounds, fit_kwargs, limits
            )
            for job in jobs
        )

    return pd.DataFrame(
        [
            {
                "repeat": job.repeat,
                "fold": job.fold,
                "candidate_rank": job.rank,
                "val_mse": score,
            }
            for job, score in zip(jobs, scores)
        ]
    )


def _materialize_split(
    s: np.ndarray,
    job: _CVJob,
    fold_fraction: float,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild this job's train matrix and validation pair coordinates.

    Splits are regenerated deterministically from the job's split seed,
    so nothing but the seed travels from the parent to the worker.
    """
    observed = observation_mask(s)
    ids, rows, cols, sampled, folds = _split_fold_ids(
        observed, fold_fraction, n_folds, job.split_seed
    )
    val_ids = np.sort(folds[job.fold])
    train_ids = np.setdiff1d(sampled, val_ids, assume_unique=True)

    train_pos = np.searchsorted(ids, train_ids)
    train_rows = rows[train_pos]
    train_cols = cols[train_pos]
    train = np.full(s.shape, np.nan, dtype=np.float64)
    train[train_rows, train_cols] = s[train_rows, train_cols]
    train[train_cols, train_rows] = s[train_cols, train_rows]
    diag = np.flatnonzero(np.diagonal(observed))
    train[diag, diag] = s[diag, diag]

    val_pos = np.searchsorted(ids, val_ids)
    return train, rows[val_pos], cols[val_pos]


def _score_job(
    s: np.ndarray,
    job: _CVJob,
    fold_fraction: float,
    n_folds: int,
    bounds: tuple[float, float],
    fit_kwargs: dict,
    blas_limits: int | None,
) -> float:
    limit = threadpool_limits(limits=blas_limits) if blas_limits else nullcontext()
    with limit:
        train, val_rows, val_cols = _materialize_split(s, job, fold_fraction, n_folds)
        est = SNMF(
            rank=job.rank,
            bounds=bounds,
            missing_values=np.nan,
            random_state=job.fit_seed,
            **fit_kwargs,
        )
        est.fit(train)
        s_hat = est.reconstruct()
        return _pairs_mse(s, s_hat, val_rows, val_cols)


def _validation_pairs(validation_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(validation_mask)
    upper = rows < cols
    return rows[upper], cols[upper]


def _pairs_mse(
    s: np.ndarray, s_hat: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> float:
    s_val = s[rows, cols]
    s_hat_val = s_hat[rows, cols]
    scored = np.isfinite(s_val) & np.isfinite(s_hat_val)
    if not scored.any():
        return float("nan")
    residual = s_val[scored] - s_hat_val[scored]
    return float(np.mean(residual * residual))


def _validation_mse(
    s: np.ndarray, s_hat: np.ndarray, validation_mask: np.ndarray
) -> float:
    rows, cols = _validation_pairs(validation_mask)
    return _pairs_mse(s, s_hat, rows, cols)


def _summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    rank_scores = (
        fold_scores.groupby("candidate_rank", as_index=False)
        .agg(
            val_mse_mean=("val_mse", "mean"),
            val_mse_std=("val_mse", "std"),
            n_fold_scores=("val_mse", "count"),
        )
        .sort_values("candidate_rank", kind="stable")
        .reset_index(drop=True)
    )
    rank_scores["val_mse_sem"] = rank_scores["val_mse_std"] / np.sqrt(
        rank_scores["n_fold_scores"]
    )
    return rank_scores


def _best_rank(rank_scores: pd.DataFrame) -> int:
    valid = rank_scores["val_mse_mean"].notna()
    if not valid.any():
        raise ValueError("all cross-validation scores are NaN")
    best = rank_scores.loc[valid, "val_mse_mean"].idxmin()
    return int(rank_scores.loc[best, "candidate_rank"])
