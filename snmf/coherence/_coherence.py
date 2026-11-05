from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

from .._common import RandomStateLike, as_seed_sequence


def _coherence(
    similarity: np.ndarray,
    observed_entries: np.ndarray,
    reference_eigvecs: np.ndarray,
    sampling_probabilities: np.ndarray,
    n_bootstrap: int,
    random_state: RandomStateLike,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    parent_seed = _parent_seed(random_state)

    def run(p_index, sampling_probability):
        return _coherence_at_sampling_probability(
            similarity,
            observed_entries,
            reference_eigvecs,
            float(sampling_probability),
            int(p_index),
            parent_seed,
            n_bootstrap,
        )

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(run)(i, pr) for i, pr in enumerate(sampling_probabilities)
    )

    coherence, retained_spectral_mass = zip(*results)
    return np.stack(coherence, axis=1), np.stack(retained_spectral_mass, axis=1)


def _coherence_at_sampling_probability(
    similarity: np.ndarray,
    observed_entries: np.ndarray,
    reference_eigvecs: np.ndarray,
    sampling_probability: float,
    p_index: int,
    parent_seed: int,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray]:
    triu = np.triu_indices(similarity.shape[0], k=1)

    def run(replicate_index):
        return _coherence_from_subsample(
            similarity,
            observed_entries,
            reference_eigvecs,
            sampling_probability,
            _replicate_seed(parent_seed, p_index, replicate_index),
            triu,
        )

    with threadpool_limits(limits=1):
        subsamples = [run(i) for i in range(n_bootstrap)]

    coherence, retained_spectral_mass = zip(*subsamples)
    return np.stack(coherence, axis=1), np.stack(retained_spectral_mass, axis=1)


def _coherence_from_subsample(
    similarity: np.ndarray,
    observed_entries: np.ndarray,
    reference_eigvecs: np.ndarray,
    sampling_probability: float,
    seed: int,
    triu: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sub_similarity = _subsample_similarity(
        similarity,
        observed_entries,
        sampling_probability,
        rng,
        triu,
    )
    _, sub_eigvecs = _top_eigenpairs(
        sub_similarity,
        reference_eigvecs.shape[1],
    )
    coherence = _subspace_overlap(sub_eigvecs, reference_eigvecs)
    retained_spectral_mass = _retained_spectral_mass(
        similarity,
        sub_eigvecs,
    )
    return coherence, retained_spectral_mass


def _subsample_similarity(
    similarity: np.ndarray,
    observed_entries: np.ndarray,
    sampling_probability: float,
    rng: np.random.Generator,
    triu: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    triu_i, triu_j = triu
    samples = rng.random(triu_i.size) < sampling_probability
    samples &= observed_entries[triu_i, triu_j]

    rows = triu_i[samples]
    cols = triu_j[samples]
    values = similarity[rows, cols] / sampling_probability

    n = similarity.shape[0]
    sub_similarity = np.zeros((n, n), dtype=np.float64)
    sub_similarity[rows, cols] = values
    sub_similarity[cols, rows] = values
    np.fill_diagonal(sub_similarity, np.diag(similarity))
    return sub_similarity


def _subspace_overlap(
    sample_eigenvectors: np.ndarray,
    reference_eigenvectors: np.ndarray,
) -> np.ndarray:
    squared_overlap = (sample_eigenvectors.T @ reference_eigenvectors) ** 2
    dimensions = np.arange(squared_overlap.shape[0])
    cumulative_overlap = np.cumsum(squared_overlap, axis=0)[
        dimensions,
        dimensions,
    ]
    return np.clip(cumulative_overlap, 0.0, 1.0)


def _retained_spectral_mass(
    similarity: np.ndarray,
    sample_eigenvectors: np.ndarray,
) -> np.ndarray:
    mass_by_dimension = np.einsum(
        "ij,ij->j",
        sample_eigenvectors,
        similarity @ sample_eigenvectors,
    )
    return np.cumsum(mass_by_dimension)


def _top_eigenpairs(
    a: np.ndarray,
    k: int,
    v0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if k < a.shape[0]:
        try:
            values, vectors = eigsh(a, k=k, which="LA", tol=1e-6, v0=v0)
            return _top_k_descending(values, vectors, k)
        except Exception:
            pass
    return _top_k_descending(*eigh(a), k=k)


def _top_k_descending(
    values: np.ndarray,
    vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)[::-1][:k]
    return values[order], vectors[:, order]


def _parent_seed(random_state: RandomStateLike) -> int:
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return int(random_state or 0) & 0xFFFFFFFF
    return int(as_seed_sequence(random_state).generate_state(1)[0]) & 0xFFFFFFFF


def _replicate_seed(parent_seed: int, p_index: int, replicate_index: int) -> int:
    return (
        int(parent_seed) + 1_000_003 * int(p_index) + 9176 * int(replicate_index)
    ) & 0xFFFFFFFF
