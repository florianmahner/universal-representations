import math

import numpy as np

from ._steps import AdmmStep, BsumStep

# Imported only when the compiled extension (_bsum.pyx) is absent; slower but
# numerically identical.
BACKEND = "python"

# Element-change threshold at which a W sweep stops early (Lin, 2007)
INNER_TOL = 1e-6


def bsum_step(x, w, ss_x=None):
    """Measure the fit of WW' to x without forming the n x n product.

    ss_x is <X, X>, constant across iterations; pass it to skip the
    full-matrix reduction.
    """
    if ss_x is None:
        ss_x = np.einsum("ij,ij->", x, x)
    xw = x @ w
    wtw = w.T @ w
    return BsumStep(
        ss_x=ss_x,
        ss_xw=np.einsum("ij,ij->", xw, w),
        ss_wwt=np.einsum("ij,ij->", wtw, wtw),
    )


def _quartic_root(a: float, b: float, c: float, d: float) -> float:
    """Find x >= 0 minimizing g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d*x.

    Key subroutine in the BSUM algorithm for updating individual
    elements of the factor matrix W. Equation (11) in paper.

    Parameters
    ----------
    a, b, c, d : float
        Polynomial coefficients

    Returns
    -------
    root : float
        Non-negative minimizer of g(x)
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    bb = b * b
    a2 = a * a
    p = (3.0 * a * c - bb) / (3.0 * a2)

    q = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)

    if c > bb / (3.0 * a):
        delta = math.sqrt(q * q * 0.25 + p * p * p / 27.0)
        root = math.cbrt(q * 0.5 - delta) + math.cbrt(q * 0.5 + delta)
    else:
        tmp = b * bb / (27.0 * a2 * a) - d / a
        root = math.cbrt(tmp)

    return max(0.0, root)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product accumulated sequentially, matching the scalar Cython
    summation order so both solvers produce identical results.

    Parameters
    ----------
    a, b : ndarray of shape (n,)
        Input vectors

    Returns
    -------
    result : float
        Dot product of a and b
    """
    return sum(x * y for x, y in zip(a, b))


def update_w(
    m: np.ndarray,
    w0: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """Block successive upper-bound minimization (BSUM) for SymNMF.

    Solves the W-subproblem by minimizing ||M - WW'||^2_F element-wise
    via quartic surrogate functions. See TABLE 1 in Shi et al. (2016),
    "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative
    Matrix Factorization".

    We adopt `m` instead of `x` here for the similarity to align with the
    notation of the original derivations in the paper.

    Parameters
    ----------
    m : ndarray of shape (n, n)
        Target symmetric similarity matrix to factorize
    w0 : ndarray of shape (n, r)
        Initial embedding matrix
    max_iter : int, default=100
        Maximum number of iterations

    Returns
    -------
    w : ndarray of shape (n, r)
        Optimized embedding matrix
    """
    w = w0.copy()
    n, r = w.shape
    wtw = w.T @ w
    diag = np.einsum("ij,ij->i", w, w)
    a = 4.0

    for _ in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = w[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag[i] - m[i, i]) + wtw[j, j] + old * old)
                d = 4.0 * _dot(w[i], wtw[:, j]) - 4.0 * _dot(m[i], w[:, j])
                new = _quartic_root(a, b, c, d)

                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # steps 7 - 13 in TABLE 1
                diag[i] += new * new - old * old
                update_row = delta * w[i]
                wtw[j, :] += update_row
                wtw[:, j] += update_row
                wtw[j, j] += delta * delta
                w[i, j] = new

        if max_delta < INNER_TOL:
            break

    return w


def update_v_(
    observed_mask: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    lam: np.ndarray,
    rho: float,
    bound_min: float,
    bound_max: float,
    v: np.ndarray,
) -> None:
    """Update the auxiliary variable V for missing data handling.

    V is an auxiliary variable that decouples the data-fitting term
    from the factorization constraint.

    For unobserved entries: v_ij = x_hat_ij - lam_ij / rho.
    For observed entries: v_ij = (x_ij + rho * x_hat_ij - lam_ij) / (1 + rho).

    Parameters
    ----------
    observed_mask : ndarray of shape (n, n)
        Binary mask, True where similarities are observed
    x : ndarray of shape (n, n)
        Similarity matrix
    x_hat : ndarray of shape (n, n)
        Current reconstruction WW'
    lam : ndarray of shape (n, n)
        Dual variables enforcing agreement between V and WW'
    rho : float
        Penalty weight for the constraint V = WW'
    bound_min : float or None
        Lower bound on V entries
    bound_max : float or None
        Upper bound on V entries
    v : ndarray of shape (n, n)
        Auxiliary variable, updated in place
    """
    v[:] = lam
    v /= -rho
    v += x_hat

    v[observed_mask] = (
        x[observed_mask] + rho * x_hat[observed_mask] - lam[observed_mask]
    ) / (1.0 + rho)

    if bound_min is not None or bound_max is not None:
        np.clip(v, bound_min, bound_max, out=v)

    # Enforce symmetry
    v += v.T
    v *= 0.5


def update_lambda_(
    lam: np.ndarray, v: np.ndarray, x_hat: np.ndarray, rho: float
) -> None:
    """Update dual variables via dual ascent: lam += rho * (V - WW').

    Parameters
    ----------
    lam : ndarray of shape (n, n)
        Dual variables, updated in place
    v : ndarray of shape (n, n)
        Auxiliary variable
    x_hat : ndarray of shape (n, n)
        Current reconstruction WW'
    rho : float
        Penalty weight for the constraint V = WW'
    """
    lam += rho * (v - x_hat)


def admm_step_(
    x: np.ndarray,
    obs: np.ndarray,
    x_hat: np.ndarray,
    v: np.ndarray,
    lam: np.ndarray,
    target: np.ndarray,
    rho: float,
    bound_min: float | None = None,
    bound_max: float | None = None,
) -> tuple:
    """Update V, the dual variables and the next solver target in place.

    Performs the V update, dual ascent and solver-target computation of
    one ADMM iteration.

    Parameters
    ----------
    x : ndarray of shape (n, n)
        Similarity matrix, zero at unobserved entries
    obs : uint8 ndarray of shape (n, n)
        Observation mask (viewed as uint8 for the compiled signature)
    x_hat : ndarray of shape (n, n)
        Current reconstruction WW'
    v, lam, target : ndarray of shape (n, n)
        Auxiliary variable, dual variables and next solver target,
        all updated in place
    rho : float
        Penalty weight for the constraint V = WW'
    bound_min, bound_max : float or None
        Bounds on V entries

    Returns
    -------
    step : AdmmStep
        Measured inner products of the updated iteration state
    """
    observed_mask = obs.view(bool)
    v_old = v.copy()
    update_v_(observed_mask, x, x_hat, lam, rho, bound_min, bound_max, v)

    primal_residual = v - x_hat
    update_lambda_(lam, v, x_hat, rho)
    np.divide(lam, rho, out=target)
    target += v

    v_step = v - v_old
    return AdmmStep(
        rho=rho,
        data_fit=np.sum((x[observed_mask] - v[observed_mask]) ** 2),
        primal=np.einsum("ij,ij->", primal_residual, primal_residual),
        lagrangian=np.einsum("ij,ij->", lam, primal_residual),
        reconstruction=np.sum((x[observed_mask] - x_hat[observed_mask]) ** 2),
        ss_x=np.sum(x[observed_mask] ** 2),
        dual_step=np.einsum("ij,ij->", v_step, v_step),
        v=np.einsum("ij,ij->", v, v),
        x_hat=np.einsum("ij,ij->", x_hat, x_hat),
        lam=np.einsum("ij,ij->", lam, lam),
    )
