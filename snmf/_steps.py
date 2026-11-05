from __future__ import annotations

from typing import NamedTuple

import numpy as np


def _relative_fit(reconstruction: float, ss_x: float) -> float:
    """||X - WW'||_F / ||X||_F over observed entries (zero when X is)."""
    if ss_x > 0:
        return np.sqrt(reconstruction / ss_x)
    return 0.0


# Iterations the convergence test measures progress over (scikit-learn's
# NMF checks every 10 iterations the same way)
PROGRESS_WINDOW = 10


def _fit_stalled(f: float, f_earlier: float | None, tol: float) -> bool:
    """Stopping rule of Xu et al. (2012), Eq. 20, measured over
    PROGRESS_WINDOW iterations: the relative fit is below tol or improved
    by less than tol across the window."""
    if f <= tol:
        return True
    if f_earlier is None:
        return False
    return abs(f_earlier - f) <= tol * max(1.0, f)


def _explained_variance(reconstruction: float, total_ss: float) -> float:
    """Fraction of observed variance explained by the reconstruction."""
    if total_ss > 0:
        return 1.0 - reconstruction / total_ss
    return 0.0


class BsumStep(NamedTuple):
    """One complete-data BSUM iteration: measured inner products."""

    ss_x: float
    """<X, X>"""
    ss_xw: float
    """<XW, W>"""
    ss_wwt: float
    """<W'W, W'W>"""

    @property
    def reconstruction(self) -> float:
        """<X - WW', X - WW'>, clamped at zero."""
        return max(0.0, self.ss_x - 2.0 * self.ss_xw + self.ss_wwt)

    @property
    def rec_error(self) -> float:
        """||X - WW'||_F."""
        return np.sqrt(self.reconstruction)

    @property
    def relative_fit(self) -> float:
        """||X - WW'||_F / ||X||_F."""
        return _relative_fit(self.reconstruction, self.ss_x)

    def converged(self, earlier: BsumStep | None, tol: float) -> bool:
        f_earlier = earlier.relative_fit if earlier is not None else None
        return _fit_stalled(self.relative_fit, f_earlier, tol)

    def metrics(self, total_ss: float) -> dict[str, float]:
        """Metrics recorded in history_ for this iteration."""
        return {
            "rec_error": self.rec_error,
            "evar": _explained_variance(self.reconstruction, total_ss),
        }


class AdmmStep(NamedTuple):
    """One ADMM iteration: measured inner products and the rho used."""

    rho: float
    """Penalty weight the step was computed with"""
    data_fit: float
    """<M * (X - V), M * (X - V)> over observed entries"""
    primal: float
    """<V - WW', V - WW'>"""
    lagrangian: float
    """<Lambda, V - WW'>"""
    reconstruction: float
    """<M * (X - WW'), M * (X - WW')> over observed entries"""
    ss_x: float
    """<M * X, M * X> over observed entries"""
    dual_step: float
    """<V - V_prev, V - V_prev>"""
    v: float
    """<V, V>"""
    x_hat: float
    """<WW', WW'>"""
    lam: float
    """<Lambda, Lambda>"""

    @property
    def penalty(self) -> float:
        """rho/2 penalty term of the objective."""
        return (self.rho / 2.0) * self.primal

    @property
    def total_objective(self) -> float:
        """Value of the augmented Lagrangian objective."""
        return self.data_fit + self.penalty + self.lagrangian

    @property
    def rec_error(self) -> float:
        """||M * (X - WW')||_F over observed entries."""
        return np.sqrt(self.reconstruction)

    @property
    def primal_residual(self) -> float:
        """||V - WW'||_F."""
        return np.sqrt(self.primal)

    @property
    def dual_residual(self) -> float:
        """rho * ||V - V_prev||_F."""
        return self.rho * np.sqrt(self.dual_step)

    @property
    def relative_fit(self) -> float:
        """||M * (X - WW')||_F / ||M * X||_F over observed entries."""
        return _relative_fit(self.reconstruction, self.ss_x)

    def converged(self, earlier: AdmmStep | None, tol: float) -> bool:
        """Stalled relative fit plus relative agreement of the ADMM
        splitting: ||V - WW'||_F <= tol * max(||V||_F, ||WW'||_F)."""
        if earlier is None:
            return False
        agree = self.primal_residual <= tol * np.sqrt(max(self.v, self.x_hat))
        return agree and _fit_stalled(self.relative_fit, earlier.relative_fit, tol)

    def metrics(self, total_ss: float) -> dict[str, float]:
        """Metrics recorded in history_ for this iteration."""
        return {
            "total_objective": self.total_objective,
            "data_fit": self.data_fit,
            "penalty": self.penalty,
            "lagrangian": self.lagrangian,
            "rec_error": self.rec_error,
            "evar": _explained_variance(self.reconstruction, total_ss),
            "primal_residual": self.primal_residual,
            "dual_residual": self.dual_residual,
        }


Step = BsumStep | AdmmStep
