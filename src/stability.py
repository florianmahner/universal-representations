from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.stats import average_pearson_r


def _correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation matrix between (n, p) and (n, q)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    a_n = np.sqrt((a_c**2).sum(axis=0, keepdims=True))
    b_n = np.sqrt((b_c**2).sum(axis=0, keepdims=True))
    return (a_c.T @ b_c) / (a_n.T @ b_n + 1e-12)


def _hungarian_match(corr_matrix: np.ndarray) -> np.ndarray:
    """1-to-1 column permutation maximizing sum of correlations."""
    _, col_ind = linear_sum_assignment(-corr_matrix)
    return col_ind


def _pairwise_correlation_matrices(
    w_matrices: list[np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """Compute each unordered seed-pair correlation matrix exactly once."""
    return {
        (i, j): _correlation_matrix(w_matrices[i], w_matrices[j])
        for i in range(len(w_matrices))
        for j in range(i + 1, len(w_matrices))
    }


def _central_index_from_pairwise(
    pairwise: dict[tuple[int, int], np.ndarray],
    n_seeds: int,
) -> int:
    """Choose the seed with highest mean optimally matched correlation."""
    scores = np.zeros(n_seeds, dtype=np.float64)
    for (i, j), corr in pairwise.items():
        row_ind, col_ind = linear_sum_assignment(-corr)
        matched_mean = corr[row_ind, col_ind].mean()
        scores[i] += matched_mean
        scores[j] += matched_mean
    scores /= n_seeds - 1
    return int(np.argmax(scores))


def reference_seed_index(w_matrices: list[np.ndarray]) -> int:
    """Return the most central seed under pairwise Hungarian-aligned correlation."""
    if not w_matrices:
        raise ValueError("At least one W matrix is required")
    ws = [np.asarray(w, dtype=np.float64) for w in w_matrices]
    shape = ws[0].shape
    if any(w.shape != shape for w in ws):
        raise ValueError("All W matrices must have the same shape")
    if len(ws) == 1:
        return 0

    pairwise = _pairwise_correlation_matrices(ws)
    return _central_index_from_pairwise(pairwise, len(ws))


def align_seeds_to_reference(
    w_matrices: list[np.ndarray],
    reference_index: int,
) -> list[np.ndarray]:
    """Reorder every seed so its columns correspond to the reference seed."""
    if not w_matrices:
        return []
    ws = [np.asarray(w, dtype=np.float64) for w in w_matrices]
    if not 0 <= reference_index < len(ws):
        raise ValueError(f"reference_index={reference_index} is out of range")
    shape = ws[0].shape
    if any(w.shape != shape for w in ws):
        raise ValueError("All W matrices must have the same shape")

    reference = ws[reference_index]
    aligned: list[np.ndarray] = []
    for index, w in enumerate(ws):
        if index == reference_index:
            aligned.append(w)
            continue
        permutation = _hungarian_match(_correlation_matrix(reference, w))
        aligned.append(w[:, permutation])
    return aligned


def factorization_stability(
    w_matrices: list[np.ndarray],
) -> np.ndarray:
    """Per-dimension stability of NMF factorization across independent seeds.

    Measures how consistently each dimension is recovered across independent
    NMF fits. Aligns all seeds to the most representative reference via
    Hungarian matching, then computes per-dimension Pearson r (Fisher-z
    averaged across seed pairs).

    Parameters
    ----------
    w_matrices : list of (n_items, rank) arrays
        W matrices from independent NMF fits (different random seeds).

    Returns
    -------
    stability : (rank,) array
        Per-dimension stability in [0, 1]. NaN if fewer than 2 matrices.
    """
    if len(w_matrices) == 0:
        return np.array([], dtype=np.float64)
    if len(w_matrices) < 2:
        return np.full(w_matrices[0].shape[1], np.nan)

    ws = [np.asarray(w, dtype=np.float64) for w in w_matrices]
    n_seeds = len(ws)
    rank = ws[0].shape[1]

    pairwise_corr = {}
    pairwise_perm = {}
    for i in range(n_seeds):
        for j in range(n_seeds):
            if i == j:
                continue
            c = _correlation_matrix(ws[i], ws[j])
            perm = _hungarian_match(c)
            pairwise_corr[(i, j)] = c[np.arange(rank), perm].sum()
            pairwise_perm[(i, j)] = perm

    avg_corr_sum = np.zeros(n_seeds)
    for i in range(n_seeds):
        for j in range(n_seeds):
            if i == j:
                continue
            avg_corr_sum[i] += pairwise_corr[(i, j)]
    avg_corr_sum /= n_seeds - 1
    ref_idx = int(np.argmax(avg_corr_sum))

    per_seed_r = np.zeros((n_seeds - 1, rank))
    idx = 0
    for i in range(n_seeds):
        if i == ref_idx:
            continue
        perm = pairwise_perm[(ref_idx, i)]
        per_seed_r[idx] = np.diag(
            _correlation_matrix(ws[ref_idx], ws[i][:, perm])
        )
        idx += 1

    return average_pearson_r(per_seed_r, axis=0)


def dimension_reliability(
    w_matrices: list[np.ndarray],
    n_splits: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """Cross-validated split-half reliability per dimension across NMF seeds.

    For each random split of objects into halves A and B:
      - Match dimensions between all seed pairs using A-half (argmax |corr|)
      - Evaluate matched dimensions on held-out B-half
    This prevents overfitting in the dimension matching step.

    Parameters
    ----------
    w_matrices : list of (n_objects, rank) arrays
        W matrices from independent NMF fits (different random seeds).
    n_splits : int
        Number of random object splits.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    reliability : (rank,) array
        Cross-validated per-dimension reliability (Spearman-Brown corrected).
        NaN if fewer than 2 matrices.
    """
    if len(w_matrices) < 2:
        if len(w_matrices) == 0:
            return np.array([], dtype=np.float64)
        return np.full(w_matrices[0].shape[1], np.nan)

    emb = np.stack([np.asarray(w, dtype=np.float64) for w in w_matrices])
    n_runs, n_objects, n_dims = emb.shape
    rng = np.random.default_rng(random_state)

    pairs_i, pairs_j = np.triu_indices(n_runs, k=1)
    n_pairs = len(pairs_i)

    all_split_reliabilities = np.zeros((n_splits, n_dims))

    for s in range(n_splits):
        mask_a = rng.random(n_objects) < 0.5
        mask_b = ~mask_a

        emb_a = _normalize_cols(emb[:, mask_a])
        emb_b = _normalize_cols(emb[:, mask_b])

        pair_per_dim = np.zeros((n_pairs, n_dims))
        for p in range(n_pairs):
            i, j = pairs_i[p], pairs_j[p]
            corr_a = emb_a[i].T @ emb_a[j]
            best_j = np.argmax(np.abs(corr_a), axis=1)
            corr_b = emb_b[i].T @ emb_b[j]
            pair_per_dim[p] = np.abs(corr_b[np.arange(n_dims), best_j])

        z = np.arctanh(np.clip(pair_per_dim, -0.999, 0.999))
        all_split_reliabilities[s] = np.tanh(np.mean(z, axis=0))

    z_splits = np.arctanh(np.clip(all_split_reliabilities, -0.999, 0.999))
    raw = np.tanh(np.mean(z_splits, axis=0))
    return 2 * raw / (1 + raw)


def _normalize_cols(x: np.ndarray) -> np.ndarray:
    """Center and L2-normalize columns per matrix in a batch.

    (batch, n_samples, k) -> (batch, n_samples, k)
    After normalization, col_i.T @ col_j = Pearson r.
    """
    x_c = x - x.mean(axis=-2, keepdims=True)
    norms = np.sqrt(np.sum(x_c**2, axis=-2, keepdims=True))
    return x_c / np.maximum(norms, 1e-12)
