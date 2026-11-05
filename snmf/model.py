# License: BSD-3-Clause

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections import defaultdict, deque
from collections.abc import Iterator

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from tqdm import trange
from sklearn.utils.validation import (
    check_is_fitted,
    check_symmetric,
    validate_data,
)
from sklearn.utils._param_validation import Interval, StrOptions, Integral, Real

from ._bsum import admm_step_, bsum_step, update_w
from ._common import is_nan_marker, observation_mask
from ._steps import PROGRESS_WINDOW, Step


logger = logging.getLogger(__name__)


def _initialize_w(
    x: np.ndarray,
    rank: int,
    method: str = "random_sqrt",
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Initialize the embedding matrix W.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix (used for scaling)
    rank : int
        Number of dimensions
    method : {'random', 'random_sqrt'}, default='random_sqrt'
        Initialization strategy:

        - 'random' : Uniform [0, 0.1], simple baseline
        - 'random_sqrt' : Uniform scaled by sqrt(X.mean() / rank)
    random_state : int, RandomState or None
        Controls random number generation

    Returns
    -------
    w : ndarray of shape (n_samples, rank)
        Non-negative embedding matrix
    """
    rng = np.random.RandomState(random_state)
    n_samples = x.shape[0]

    if method == "random":
        w = 0.1 * rng.rand(n_samples, rank)
    elif method == "random_sqrt":
        avg = np.sqrt(x.mean() / rank)
        w = rng.rand(n_samples, rank) * avg
    else:
        raise ValueError(
            f"Invalid initialization method '{method}'. "
            f"Choose from: 'random', 'random_sqrt'"
        )

    return w


def _validate_bounds(val) -> None:
    if val is None:
        return
    lo, hi = val
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("Lower bound cannot be greater than upper bound")


def _complete_steps(
    x: np.ndarray, w: np.ndarray, max_inner: int
) -> Iterator[tuple[np.ndarray, Step]]:
    """Yield one BSUM iteration at a time on complete data."""
    ss_x = np.einsum("ij,ij->", x, x)
    while True:
        w = update_w(x, w, max_iter=max_inner)
        yield w, bsum_step(x, w, ss_x)


def _missing_steps(
    x: np.ndarray,
    observed_mask: np.ndarray,
    w: np.ndarray,
    rho: float,
    bounds: tuple[float | None, float | None] | None,
    max_inner: int,
) -> Iterator[tuple[np.ndarray, Step]]:
    """Yield one ADMM iteration at a time on partially observed data.

    An auxiliary matrix V and dual variables decouple the data-fitting
    term from the factorization. Each iteration solves the W subproblem
    on the current target, then admm_step_ updates V, the duals and the
    next target in one pass.
    """
    bound_min, bound_max = bounds if bounds is not None else (None, None)

    lam = np.zeros_like(x)
    v = x.copy()
    x_hat = np.empty_like(x)
    # First solver target: lam / rho + v with lam = 0 is v
    target = v.copy()
    # The compiled kernel reads the mask as a uint8 buffer
    obs = observed_mask.view(np.uint8)

    while True:
        w = update_w(target, w, max_iter=max_inner)
        np.dot(w, w.T, out=x_hat)
        yield w, admm_step_(x, obs, x_hat, v, lam, target, rho, bound_min, bound_max)


class SNMF(TransformerMixin, BaseEstimator):
    """Symmetric nonnegative matrix factorization of a similarity matrix.

    Factorizes a symmetric similarity matrix S into WW' where W >= 0.
    Handles missing entries and can enforce bounds on the reconstructed
    similarities.

    The objective is:
        min_{W>=0, V} ||M * (S - V)||^2_F + rho/2 * ||V - WW'||^2_F
    where M is the observation mask.

    Parameters
    ----------
    rank : int, default=10
        Number of dimensions in the embedding
    rho : float, default=3.0
        Penalty weight controlling how closely WW' must match V
    max_outer : int, default=100
        Maximum number of outer optimization iterations; the stall rule
        usually stops far earlier
    max_inner : int, default=10
        Maximum iterations for the W update per outer iteration
    tol : float, default=1e-3
        Relative convergence tolerance. Fitting stops when the relative
        fit ||X - WW'||_F / ||X||_F over observed entries falls below tol
        or improves by less than tol across the last 10 iterations
        (Xu et al., 2012). With missing data, the ADMM splitting must
        additionally agree: ||V - WW'||_F <= tol * max(||V||_F, ||WW'||_F).
    verbose : int, default=0
        Whether to print optimization progress
    init : str, default='random_sqrt'
        Embedding initialization method ('random', 'random_sqrt')
    random_state : int or None, default=None
        Random seed for reproducible initialization
    missing_values : float or None, default=np.nan
        Sentinel value marking missing entries in the similarity matrix
    bounds : tuple of (float, float) or None, default=(None, None)
        (lower, upper) bounds on the reconstructed similarities.
        For example, (0, 1) for similarities normalized to [0, 1].
    check_input : bool, default=True
        Whether to verify that the input matrix and missing mask are
        symmetric. Set to False for large matrices where the input is
        known to be valid, to avoid allocating temporary arrays during
        the symmetry check.

    Attributes
    ----------
    w_ : ndarray of shape (n_samples, rank)
        Learned embedding matrix
    components_ : ndarray of shape (n_samples, rank)
        Alias for w_ (sklearn compatibility)
    n_iter_ : int
        Number of outer iterations performed
    history_ : dict
        Optimization metrics per iteration

    Examples
    --------
    >>> from snmf import SNMF
    >>> model = SNMF(rank=10, random_state=42)
    >>> w = model.fit_transform(similarity_matrix)
    >>> reconstruction = w @ w.T

    >>> # With missing data
    >>> similarity_matrix[mask] = np.nan
    >>> model = SNMF(rank=10, missing_values=np.nan)
    >>> w = model.fit_transform(similarity_matrix)
    """

    _parameter_constraints = {
        "rank": [Interval(Integral, 1, None, closed="left")],
        "rho": [Interval(Real, 0.0, None, closed="neither")],
        "max_outer": [Interval(Integral, 1, None, closed="left")],
        "max_inner": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "init": [StrOptions({"random", "random_sqrt"})],
        "verbose": [Interval(Integral, 0, None, closed="left")],
        "random_state": ["random_state"],  # sklearn's special validator
        "missing_values": [None, Real, np.nan],
        "bounds": [None, tuple],
        "check_input": ["boolean"],
    }

    def __init__(
        self,
        rank: int = 10,
        rho: float = 3.0,
        max_outer: int = 100,
        max_inner: int = 10,
        tol: float = 1e-3,
        verbose: int = 0,
        init: str = "random_sqrt",
        random_state: int | None = None,
        missing_values: float | None = np.nan,
        bounds: tuple[float, float] | None = (None, None),
        check_input: bool = True,
    ) -> None:
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.missing_values = missing_values
        self.bounds = bounds
        self.check_input = check_input

    def _progress_bar(self):
        return trange(
            1,
            self.max_outer + 1,
            disable=not self.verbose,
            desc="SNMF",
            mininterval=10.0 if not sys.stderr.isatty() else 0.1,
        )

    def _fit_loop(self, steps: Iterator[tuple[np.ndarray, Step]]) -> SNMF:
        """Drive the outer iterations; steps yields one measurement each."""
        history = defaultdict(list)

        recent = deque(maxlen=PROGRESS_WINDOW)
        pbar = self._progress_bar()
        for i, (w, step) in zip(pbar, steps):
            earlier = recent[0] if recent else None
            metrics = step.metrics(self._total_ss)
            metrics["converged"] = step.converged(earlier, self.tol)
            for key, value in metrics.items():
                history[key].append(value)

            pbar.set_postfix(
                rec_error=f"{metrics['rec_error']:.3f}",
                evar=f"{metrics['evar']:.3f}",
            )

            if metrics["converged"]:
                logger.info("Converged at iteration %d", i)
                break
            recent.append(step)
        else:
            warnings.warn(
                f"Maximum number of outer iterations ({self.max_outer}) "
                "reached without convergence. Increase max_outer or tol.",
                ConvergenceWarning,
            )

        self.w_ = w
        self.components_ = w
        self.n_iter_ = i
        self.history_ = history

        return self

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Validate the similarity matrix and derive the observation stats.

        Allocates only the one contract copy of x and the observation
        mask; all statistics are computed without materializing the
        observed entries.
        """
        self._validate_params()
        _validate_bounds(self.bounds)

        x = validate_data(
            self,
            x,
            reset=True,
            ensure_all_finite=(
                "allow-nan" if is_nan_marker(self.missing_values) else True
            ),
            ensure_2d=True,
            dtype=np.float64,
            copy=True,
        )

        self._observed_mask = observation_mask(x, self.missing_values)
        if not np.any(self._observed_mask):
            raise ValueError(
                "No observed entries found in the data. All values are missing."
            )

        if is_nan_marker(self.missing_values) or self.missing_values is None:
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x[~self._observed_mask] = 0.0
        if self.check_input:
            check_symmetric(self._observed_mask, raise_exception=True)
            x = check_symmetric(x, raise_exception=True, tol=1e-10)

        # Zero-filled missing entries drop out of the sums, so the
        # variance identity avoids materializing the observed values
        n_observed = np.count_nonzero(self._observed_mask)
        observed_mean = np.sum(x) / n_observed
        observed_sumsq = np.einsum("ij,ij->", x, x)
        self._total_ss = observed_sumsq - n_observed * observed_mean**2

        if self.verbose:
            n_threads = os.environ.get("OMP_NUM_THREADS", "1")
            logger.info(
                "Using %s BLAS thread(s). For faster fitting, set "
                "OMP_NUM_THREADS before importing snmf "
                "(e.g. export OMP_NUM_THREADS=4)",
                n_threads,
            )

        return x

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> SNMF:
        """Fit the model to a similarity matrix.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values should be marked
            according to the ``missing_values`` parameter.
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        self : SNMF
            Fitted estimator.
        """
        x = self._validate_input(x)
        w = _initialize_w(x, self.rank, self.init, self.random_state)

        if self._observed_mask.all():
            steps = _complete_steps(x, w, self.max_inner)
        else:
            steps = _missing_steps(
                x, self._observed_mask, w, self.rho, self.bounds, self.max_inner
            )
        return self._fit_loop(steps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return the learned embedding matrix W.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Ignored, present for sklearn API compatibility.

        Returns
        -------
        w : ndarray of shape (n_samples, rank)
            Embedding matrix
        """
        check_is_fitted(self)
        return self.w_

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and return the learned embedding matrix W.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        w : ndarray of shape (n_samples, rank)
            Embedding matrix
        """
        return self.fit(x, y).transform(x)

    def reconstruct(self, w: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct the similarity matrix as WW'.

        Parameters
        ----------
        w : ndarray of shape (n_samples, rank) or None
            Embedding matrix. If None, uses the fitted embedding.

        Returns
        -------
        s_hat : ndarray of shape (n_samples, n_samples)
            Reconstructed similarity matrix
        """
        if w is None:
            check_is_fitted(self)
            w = self.w_

        return w @ w.T

    def score(self, x: np.ndarray, y: np.ndarray | None = None) -> float:
        """Negative MSE on observed entries (higher is better).

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values should be marked
            according to the ``missing_values`` parameter.
        y : Ignored
            Not used, present for sklearn API compatibility.

        Returns
        -------
        score : float
            Negative mean squared reconstruction error on observed entries.
        """
        check_is_fitted(self)

        x = validate_data(
            self,
            x,
            reset=False,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite="allow-nan"
            if is_nan_marker(self.missing_values)
            else True,
        )
        observed_mask = observation_mask(x, self.missing_values)
        s_hat = self.reconstruct()
        mse = np.mean((x[observed_mask] - s_hat[observed_mask]) ** 2)
        return -mse
