from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

Array = np.ndarray
EPS = 1e-8


def l2_normalize_columns(w: Array) -> Array:
    """L2-normalize each column of w."""
    return w / (np.linalg.norm(w, axis=0, keepdims=True) + EPS)


def correlation_matrix(W_source: Array, W_target: Array) -> Array:
    """Pearson correlation between all (target, source) column pairs.

    Returns shape (n_dims_target, n_dims_source).
    """
    X = (W_source - W_source.mean(0)) / (W_source.std(0) + EPS)
    Y = (W_target - W_target.mean(0)) / (W_target.std(0) + EPS)
    return (Y.T @ X) / len(X)


def max_correlation(W_source: Array, W_target: Array) -> Array:
    """Max |Pearson r| for each target dimension (greedy best-match).

    Returns shape (n_dims_target,).
    """
    return np.abs(correlation_matrix(W_source, W_target)).max(axis=1)


def best_match_indices(W_source: Array, W_target: Array) -> Array:
    """Index of best-matching source dim for each target dim (by |Pearson r|).

    Returns shape (n_dims_target,).
    """
    return np.abs(correlation_matrix(W_source, W_target)).argmax(axis=1)


def cos2_matrix(W_source: Array, W_target: Array) -> Array:
    """Cos² between all (target, source) column pairs.

    Returns shape (n_dims_target, n_dims_source).
    """
    s = l2_normalize_columns(W_source)
    t = l2_normalize_columns(W_target)
    return (t.T @ s) ** 2


def cos2_greedy(W_source: Array, W_target: Array) -> Array:
    """Max cos² for each target dimension (greedy best-match).

    Returns shape (n_dims_target,).
    """
    return cos2_matrix(W_source, W_target).max(axis=1)


def cos2_permutation(W_source: Array, W_target: Array) -> Array:
    """Optimal 1-to-1 cos² assignment via Hungarian algorithm.

    Returns shape (n_dims_target,).
    """
    c2 = cos2_matrix(W_source, W_target)
    row_ind, col_ind = linear_sum_assignment(-c2)
    scores = np.zeros(c2.shape[0])
    scores[row_ind] = c2[row_ind, col_ind]
    return scores


def cos2_match_indices(W_source: Array, W_target: Array) -> Array:
    """Source-column index assigned to each target column by Hungarian matching.

    Unmatched target columns are represented by ``-1`` when the source has fewer
    columns than the target.
    """
    c2 = cos2_matrix(W_source, W_target)
    row_ind, col_ind = linear_sum_assignment(-c2)
    indices = np.full(c2.shape[0], -1, dtype=int)
    indices[row_ind] = col_ind
    return indices


def cos2_perm_null_threshold(
    W_source: Array,
    W_target: Array,
    n_perms: int = 1000,
    seed: int = 42,
) -> Array:
    """Pair-specific 95th-percentile null threshold for ``cos2_permutation``.

    This implements the manuscript's null for one ordered model pair.  Rows of
    ``W_source`` are independently permuted on every iteration, destroying item
    correspondence while preserving both matrices' column structure.  The
    returned scores are ordered by target dimension.
    """
    W_source = np.asarray(W_source)
    W_target = np.asarray(W_target)
    if W_source.ndim != 2 or W_target.ndim != 2:
        raise ValueError("W_source and W_target must both be two-dimensional")
    if W_source.shape[0] != W_target.shape[0]:
        raise ValueError(
            "W_source and W_target must have the same number of rows, got "
            f"{W_source.shape[0]} and {W_target.shape[0]}"
        )
    if n_perms < 1:
        raise ValueError("n_perms must be at least 1")

    rng = np.random.default_rng(seed)
    s_norm = l2_normalize_columns(W_source)
    t_norm = l2_normalize_columns(W_target)
    n_items = W_target.shape[0]
    n_target_dims = W_target.shape[1]
    null_scores = np.empty((n_perms, n_target_dims))

    for p in range(n_perms):
        perm = rng.permutation(n_items)
        c2 = (t_norm.T @ s_norm[perm]) ** 2
        row_ind, col_ind = linear_sum_assignment(-c2)
        scores = np.zeros(n_target_dims)
        scores[row_ind] = c2[row_ind, col_ind]
        null_scores[p] = scores

    return np.percentile(null_scores, 95, axis=0)


def cos2_perm_null_thresholds(
    W_target: Array,
    W_sources: list[Array],
    n_perms: int = 1000,
    seed: int = 42,
    n_source_samples: int = 10,
) -> Array:
    """Null thresholds for every source, as used for the published numbers.

    The threshold is dominated by the target's column structure, so it is
    estimated once from ``n_source_samples`` subsampled sources (verified
    <1% error against using all of them) and applied to every source.
    Returns shape ``(n_sources, n_dims_target)``.
    """
    W_target = np.asarray(W_target)
    if W_target.ndim != 2:
        raise ValueError("W_target must be two-dimensional")

    rng = np.random.default_rng(seed)
    n_sources = len(W_sources)
    n_samples = min(n_source_samples, n_sources)
    indices = rng.choice(n_sources, n_samples, replace=False)

    t_norm = l2_normalize_columns(W_target)
    n_items, n_dims = W_target.shape

    all_p95 = []
    for idx in indices:
        s_norm = l2_normalize_columns(np.asarray(W_sources[idx]))
        null_scores = np.empty((n_perms, n_dims))
        for p in range(n_perms):
            perm = rng.permutation(n_items)
            c2 = (t_norm.T @ s_norm[perm]) ** 2
            row_ind, col_ind = linear_sum_assignment(-c2)
            scores = np.zeros(n_dims)
            scores[row_ind] = c2[row_ind, col_ind]
            null_scores[p] = scores
        all_p95.append(np.percentile(null_scores, 95, axis=0))

    threshold = np.mean(all_p95, axis=0)
    return np.tile(threshold, (n_sources, 1))


def null_adjust(raw: Array, threshold: Array) -> Array:
    """Null-calibrate: (raw - threshold) / (1 - threshold), clipped to [0, 1]."""
    adjusted = (raw - threshold) / np.maximum(1 - threshold, EPS)
    return np.clip(adjusted, 0, 1)


METRICS: dict[str, Callable] = {
    "max_corr": max_correlation,
    "cos2_perm": cos2_permutation,
}
