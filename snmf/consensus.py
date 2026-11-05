# License: BSD-3-Clause

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from ._common import blas_limits_for_workers, n_jobs_for_tasks


def l2_norm_columns(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def hungarian_match(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    sim = ref.T @ target
    _, col_ind = linear_sum_assignment(-sim)
    return col_ind


def align_embeddings(embeddings: np.ndarray, reference_idx: int = 0) -> np.ndarray:
    n_runs = embeddings.shape[0]
    norms = np.array([l2_norm_columns(e) for e in embeddings])
    ref = norms[reference_idx]

    aligned = np.empty_like(embeddings)
    for i in range(n_runs):
        if i == reference_idx:
            aligned[i] = embeddings[i]
        else:
            perm = hungarian_match(ref, norms[i])
            aligned[i] = embeddings[i][:, perm]

    return aligned


def pairwise_agreement(aligned: np.ndarray) -> np.ndarray:
    n_runs, _, rank = aligned.shape
    scores = np.zeros(n_runs)

    for j in range(rank):
        col = aligned[:, :, j]
        norms = np.linalg.norm(col, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        col_normed = col / norms
        sim = col_normed @ col_normed.T
        np.fill_diagonal(sim, 0.0)
        scores += sim.sum(axis=1) / (n_runs - 1)

    scores /= rank
    return scores


class EnsembleFit(BaseEstimator, TransformerMixin):
    """Fit the base estimator multiple times and stack the embeddings.

    Each run uses a different random seed, producing embeddings that
    differ only in column permutation. The stacked output has shape
    (n_samples, rank * n_runs) and can be passed to a consensus method.

    Parameters
    ----------
    base_estimator : BaseEstimator
        Estimator with fit/transform (e.g. SNMF)
    n_runs : int, default=50
        Number of independent runs
    random_state : int, default=0
        Base random seed (run i uses random_state + i)
    n_jobs : int, default=-1
        Number of parallel jobs

    Attributes
    ----------
    embeddings_ : ndarray of shape (n_samples, rank * n_runs)
        Horizontally stacked embeddings from all runs
    estimators_ : list of BaseEstimator
        Fitted estimators from each run
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_runs: int = 50,
        random_state: int = 0,
        n_jobs: int = -1,
        threads_per_worker: int | str | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_runs = n_runs
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.threads_per_worker = threads_per_worker

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> EnsembleFit:
        """Fit n_runs independent estimators in parallel.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : Ignored

        Returns
        -------
        self : EnsembleFit
        """

        n_jobs = n_jobs_for_tasks(self.n_jobs, self.n_runs)
        limits = blas_limits_for_workers(self.threads_per_worker, n_jobs)

        def _fit_one(seed):
            # Limit BLAS threads per worker so parallel runs do not
            # oversubscribe the machine
            limit = threadpool_limits(limits=limits) if limits else nullcontext()
            with limit:
                est = clone(self.base_estimator)
                est.set_params(random_state=seed)
                return est, est.fit_transform(x)

        if n_jobs == 1:
            results = [_fit_one(self.random_state + i) for i in range(self.n_runs)]
        else:
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_fit_one)(self.random_state + i) for i in range(self.n_runs)
            )

        estimators, embeddings = zip(*results)
        self.estimators_ = list(estimators)
        self.embeddings_ = np.hstack(embeddings)
        return self

    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return the stacked embeddings (x is ignored).

        Parameters
        ----------
        x : Ignored
        y : Ignored

        Returns
        -------
        embeddings : ndarray of shape (n_samples, rank * n_runs)
        """
        check_is_fitted(self, "embeddings_")
        return self.embeddings_


class AlignedConsensus(BaseEstimator, TransformerMixin):
    """Consensus via Hungarian alignment for symmetric NMF.

    Symmetric NMF produces the same dimensions across runs but in
    different order (permutation ambiguity, no rotation). This class
    aligns dimensions via the Hungarian algorithm and returns the
    most central run — the one closest to the element-wise median
    across all aligned runs.

    Parameters
    ----------
    rank : int
        Number of dimensions in each embedding

    Attributes
    ----------
    aligned_embeddings_ : ndarray of shape (n_runs, n_samples, rank)
        Embeddings after Hungarian alignment
    consensus_median_ : ndarray of shape (n_samples, rank)
        Element-wise median across aligned runs
    centrality_scores_ : ndarray of shape (n_runs,)
        Frobenius distance of each run to the consensus median
    agreement_scores_ : ndarray of shape (n_runs,)
        Mean pairwise cosine similarity per run (higher = more stable)
    selected_run_idx_ : int
        Index of the most central run

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from snmf import SNMF
    >>> from snmf.consensus import EnsembleFit, AlignedConsensus
    >>>
    >>> pipeline = Pipeline([
    ...     ('ensemble', EnsembleFit(SNMF(rank=10), n_runs=50)),
    ...     ('consensus', AlignedConsensus(rank=10))
    ... ])
    >>> embedding = pipeline.fit_transform(similarity_matrix)
    """

    def __init__(self, rank: int):
        self.rank = rank

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> AlignedConsensus:
        """Align embeddings and select the most central run.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_runs * rank)
            Stacked embeddings from EnsembleFit
        y : Ignored

        Returns
        -------
        self : AlignedConsensus
        """
        x = np.asarray(x)
        n_samples, n_total = x.shape
        n_runs = n_total // self.rank

        if n_total % self.rank != 0:
            raise ValueError(
                f"Number of columns ({n_total}) not divisible by rank ({self.rank})"
            )

        # (n_samples, n_runs, rank) -> (n_runs, n_samples, rank)
        embeddings = x.reshape(n_samples, n_runs, self.rank).transpose(1, 0, 2)

        self.aligned_embeddings_ = align_embeddings(embeddings, reference_idx=0)
        self.consensus_median_ = np.median(self.aligned_embeddings_, axis=0)

        diffs = self.aligned_embeddings_ - self.consensus_median_[np.newaxis]
        self.centrality_scores_ = np.sqrt(np.einsum("ijk,ijk->i", diffs, diffs))

        self.agreement_scores_ = pairwise_agreement(self.aligned_embeddings_)
        self.selected_run_idx_ = int(np.argmin(self.centrality_scores_))

        return self

    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Return the most central run.

        Parameters
        ----------
        x : Ignored
        y : Ignored

        Returns
        -------
        w : ndarray of shape (n_samples, rank)
        """
        check_is_fitted(self, "aligned_embeddings_")
        return self.aligned_embeddings_[self.selected_run_idx_]
