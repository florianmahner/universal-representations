# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# Cython implementations of the block successive upper bound minimization
# algorithm for symmetric nonnegative matrix factorization.
# See Shi et al. (2016) for more details.
# Reference: Shi et al. (2016) "Inexact Block Coordinate Descent Methods For
# Symmetric Nonnegative Matrix Factorization".
# Link: https://arxiv.org/abs/1608.02649
#
# The update_w solvers call the factorization target `m` to match the
# notation of the original derivations in the paper.

import numpy as np
cimport numpy as np

from ._steps import AdmmStep, BsumStep
from libc.math cimport fabs, sqrt, cbrt
from cython.parallel cimport prange
from libc.string cimport memcpy
from scipy.linalg.cython_blas cimport ddot, daxpy, dgemv, dgemm

np.import_array()

BACKEND = "cython"

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

ctypedef np.float64_t DTYPE_t


cdef inline double _quartic_root(double a, double b, double c, double d) nogil:
    """Solve cubic resolvent of the quartic subproblem (Shi et al. 2016, Eq. 11)."""
    cdef double bb  = b * b
    cdef double a2  = a * a
    cdef double p   = (3.0 * a * c - bb) / (3.0 * a2)
    cdef double q   = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)
    cdef double root, delta, tmp

    if c > bb / (3.0 * a):
        delta = sqrt(q * q * 0.25 + p * p * p / 27.0)
        root  = cbrt(q * 0.5 - delta) + cbrt(q * 0.5 + delta)
    else:
        tmp   = b * bb / (27.0 * a2 * a) - d / a
        root  = cbrt(tmp)

    return 0.0 if root < 0.0 else root


cdef inline double _dot_product(double[:] a, double[:] b) nogil:
    cdef int n = a.shape[0]
    cdef int i
    cdef double result = 0.0
    for i in range(n):
        result += a[i] * b[i]
    return result


# ---------------------------------------------------------------------------
# Scalar Cython (per-row mw_i precomputation)
# ---------------------------------------------------------------------------

cpdef np.ndarray[DTYPE_t, ndim=2] update_w_scalar(double[:, ::1] m,
                                                   double[:, ::1] w0,
                                                   int max_iter=100):
    cdef double tol = INNER_TOL
    cdef int n = w0.shape[0]
    cdef int r = w0.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] wtw = np.dot(w.T, w)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", w, w)
    cdef np.ndarray[DTYPE_t, ndim=1] mw_i = np.empty(r)

    cdef double[:, :] w_view = w
    cdef double[:, :] m_view = m
    cdef double[:, :] wtw_view = wtw
    cdef double[:] diag_view = diag
    cdef double[:] mw_i_view = mw_i

    cdef double a = 4.0
    cdef int it, i, j, k
    cdef double max_delta, old, b, c, d, new, delta, update_val, m_ik

    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                mw_i_view[j] = 0.0
            for k in range(n):
                m_ik = m_view[i, k]
                for j in range(r):
                    mw_i_view[j] += m_ik * w_view[k, j]

            for j in range(r):
                old = w_view[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag_view[i] - m_view[i, i]) + wtw_view[j, j] + old * old)

                # wtw_view[:, j] is a strided column slice of a C-contiguous array,
                # creating a temporary memoryview per j-iteration. Not performance-
                # critical since update_w is the scalar fallback, not the fast path.
                d = 4.0 * _dot_product(w_view[i, :], wtw_view[:, j]) - 4.0 * mw_i_view[j]
                new = _quartic_root(a, b, c, d)
                delta = new - old

                if fabs(delta) > max_delta:
                    max_delta = fabs(delta)

                diag_view[i] += new * new - old * old

                for k in range(r):
                    update_val = delta * w_view[i, k]
                    wtw_view[j, k] += update_val
                    wtw_view[k, j] += update_val

                wtw_view[j, j] += delta * delta

                w_view[i, j] = new

        if max_delta < tol:
            break

    return w


# ---------------------------------------------------------------------------
# BLAS-2 dgemv variant
# ---------------------------------------------------------------------------

cpdef np.ndarray[DTYPE_t, ndim=2] update_w_blas(double[:, ::1] m,
                                                 double[:, ::1] w0,
                                                 int max_iter=100):
    cdef double tol = INNER_TOL
    cdef Py_ssize_t n = w0.shape[0]
    cdef int r = w0.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] wtw = np.dot(w.T, w)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", w, w)
    cdef np.ndarray[DTYPE_t, ndim=1] mw_i_buf = np.empty(r)

    # Raw C pointers — all arrays are C-contiguous
    cdef double* w_p = <double*>w.data
    cdef double* wtw_p = <double*>wtw.data
    cdef double* diag_p = <double*>diag.data
    cdef double* mwi_p = <double*>mw_i_buf.data
    cdef double* m_p = &m[0, 0]

    # BLAS parameters
    cdef char trans_n = b'N'
    cdef int blas_n = n
    cdef int blas_r = r
    cdef int inc_1 = 1
    cdef double alpha_1 = 1.0
    cdef double beta_0 = 0.0

    cdef int it, i, j, k
    cdef Py_ssize_t ir, jr, in_off
    cdef double max_delta, old_val, b_coef, c_coef, d_coef
    cdef double new_val, delta_val, dot_val

    for it in range(max_iter):
        max_delta = 0.0

        for i in range(n):
            ir = i * r
            in_off = i * n

            # mw_i = M[i,:] @ W = W^T @ M[i,:] via BLAS dgemv
            # C row-major W(n,r) = Fortran col-major A(r,n) with lda=r, where A = W^T
            # y = A @ x: trans='N', m=r, n=n, lda=r
            dgemv(&trans_n, &blas_r, &blas_n, &alpha_1,
                  w_p, &blas_r,
                  &m_p[in_off], &inc_1,
                  &beta_0, mwi_p, &inc_1)

            for j in range(r):
                jr = j * r
                old_val = w_p[ir + j]

                b_coef = 12.0 * old_val
                c_coef = 4.0 * ((diag_p[i] - m_p[in_off + i]) + wtw_p[jr + j] + old_val * old_val)

                # W[i,:] · WtW[j,:] via BLAS ddot (both contiguous rows of length r)
                dot_val = ddot(&blas_r, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                d_coef = 4.0 * dot_val - 4.0 * mwi_p[j]
                new_val = _quartic_root(4.0, b_coef, c_coef, d_coef)
                delta_val = new_val - old_val

                if fabs(delta_val) > max_delta:
                    max_delta = fabs(delta_val)

                diag_p[i] += new_val * new_val - old_val * old_val

                # WtW[j,:] += delta * W[i,:] via BLAS daxpy
                daxpy(&blas_r, &delta_val, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                # WtW[:,j] += delta * W[i,:] (strided column update)
                for k in range(r):
                    wtw_p[k * r + j] += delta_val * w_p[ir + k]

                wtw_p[jr + j] += delta_val * delta_val

                w_p[ir + j] = new_val

        if max_delta < tol:
            break

    return w


# ---------------------------------------------------------------------------
# BLAS-3 blocked dsymm/dgemm variant
# ---------------------------------------------------------------------------

cpdef np.ndarray[DTYPE_t, ndim=2] update_w_blas_blocked(double[:, ::1] m,
                                                         double[:, ::1] w0,
                                                         int max_iter=100,
                                                         int block_size=0):
    cdef double tol = INNER_TOL
    cdef Py_ssize_t n = w0.shape[0]
    cdef int r = w0.shape[1]

    if block_size <= 0:
        block_size = min(50, max(1, n // 10))

    cdef np.ndarray[DTYPE_t, ndim=2] w = np.array(w0, copy=True)
    cdef np.ndarray[DTYPE_t, ndim=2] wtw = np.dot(w.T, w)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", w, w)

    # MW buffer: precomputed M @ W, corrected per block
    cdef np.ndarray[DTYPE_t, ndim=2] mw = np.empty((n, r), dtype=np.float64)

    # Per-row delta buffer: stores delta_W for each row in current block
    cdef np.ndarray[DTYPE_t, ndim=2] delta_w = np.empty((block_size, r), dtype=np.float64)

    # Snapshot of W[i,:] before processing row i (for computing per-row delta)
    cdef np.ndarray[DTYPE_t, ndim=1] w_old_row = np.empty(r, dtype=np.float64)

    # Raw C pointers
    cdef double* w_p = <double*>w.data
    cdef double* wtw_p = <double*>wtw.data
    cdef double* diag_p = <double*>diag.data
    cdef double* mw_p = <double*>mw.data
    cdef double* m_p = &m[0, 0]
    cdef double* dw_p = <double*>delta_w.data
    cdef double* w_old_p = <double*>w_old_row.data

    # BLAS parameters
    cdef char trans_n = b'N'
    cdef int inc_1 = 1
    cdef double alpha_1 = 1.0
    cdef double beta_0 = 0.0
    cdef double beta_1 = 1.0

    # BLAS dimension variables
    cdef int blas_n = n
    cdef int blas_r = r

    cdef Py_ssize_t it, i, j, k, bi, prev
    cdef Py_ssize_t ir, jr, in_off
    cdef Py_ssize_t block_start, block_end, block_len
    cdef int remaining, blas_remaining, blas_block_len
    cdef double max_delta, old_val, b_coef, c_coef, d_coef
    cdef double new_val, delta_val, dot_val
    cdef double m_ik

    for it in range(max_iter):
        max_delta = 0.0

        # Precompute MW = M @ W via dgemm (BLAS-3). dgemm is used instead of
        # dsymm although M is symmetric: gemm kernels are far better
        # optimized and threaded in common BLAS builds, while symm rarely
        # realizes its theoretical half-read advantage.
        # C row-major (n,r) maps to Fortran col-major (r,n):
        # C(r,n) = W^T(r,n) @ M^T(n,n), and M^T = M by symmetry
        dgemm(&trans_n, &trans_n, &blas_r, &blas_n, &blas_n, &alpha_1,
              w_p, &blas_r,
              m_p, &blas_n,
              &beta_0, mw_p, &blas_r)

        block_start = 0
        while block_start < n:
            block_end = block_start + block_size
            if block_end > n:
                block_end = n
            block_len = block_end - block_start

            for bi in range(block_len):
                i = block_start + bi
                ir = i * r
                in_off = i * n

                # Intra-block correction: correct MW[i,:] for changes from rows
                # block_start..block_start+bi-1 (earlier rows in this block)
                for prev in range(bi):
                    m_ik = m_p[in_off + block_start + prev]
                    daxpy(&blas_r, &m_ik, &dw_p[prev * r], &inc_1, &mw_p[ir], &inc_1)

                # Save W[i,:] before j-loop modifies it
                memcpy(w_old_p, &w_p[ir], r * sizeof(double))

                for j in range(r):
                    jr = j * r
                    old_val = w_p[ir + j]

                    b_coef = 12.0 * old_val
                    c_coef = 4.0 * ((diag_p[i] - m_p[in_off + i]) + wtw_p[jr + j] + old_val * old_val)

                    dot_val = ddot(&blas_r, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                    d_coef = 4.0 * dot_val - 4.0 * mw_p[ir + j]
                    new_val = _quartic_root(4.0, b_coef, c_coef, d_coef)
                    delta_val = new_val - old_val

                    if fabs(delta_val) > max_delta:
                        max_delta = fabs(delta_val)

                    diag_p[i] += new_val * new_val - old_val * old_val

                    daxpy(&blas_r, &delta_val, &w_p[ir], &inc_1, &wtw_p[jr], &inc_1)

                    for k in range(r):
                        wtw_p[k * r + j] += delta_val * w_p[ir + k]

                    wtw_p[jr + j] += delta_val * delta_val
                    w_p[ir + j] = new_val

                # Store delta for this row: delta_w[bi,:] = W_new[i,:] - W_old[i,:]
                for j in range(r):
                    dw_p[bi * r + j] = w_p[ir + j] - w_old_p[j]

            # Inter-block correction: correct MW for all remaining rows using dgemm
            remaining = n - block_end
            if remaining > 0:
                blas_remaining = remaining
                blas_block_len = block_len
                dgemm(&trans_n, &trans_n, &blas_r, &blas_remaining, &blas_block_len,
                      &alpha_1,
                      dw_p, &blas_r,
                      &m_p[block_end * n + block_start], &blas_n,
                      &beta_1,
                      &mw_p[block_end * r], &blas_r)

            block_start = block_end

        if max_delta < tol:
            break

    return w



# ---------------------------------------------------------------------------
# ADMM iteration update (missing-data path)
# ---------------------------------------------------------------------------

def admm_step_(double[:, ::1] x,
               unsigned char[:, ::1] obs,
               double[:, ::1] x_hat,
               double[:, ::1] v,
               double[:, ::1] lam,
               double[:, ::1] target,
               double rho,
               bound_min=None,
               bound_max=None):
    """Update V, the dual variables and the next solver target in place.

    Performs the V update, dual ascent and solver-target computation of
    one ADMM iteration in a single pass over the matrix, processing rows
    in parallel with OpenMP.

    All matrices must be symmetric; v, lam and target stay symmetric.

    Returns
    -------
    step : AdmmStep
        Measured inner products of the updated iteration state
    """
    cdef bint use_min = bound_min is not None
    cdef bint use_max = bound_max is not None
    cdef double bmin = bound_min if use_min else 0.0
    cdef double bmax = bound_max if use_max else 0.0

    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i, j

    cdef double data_fit = 0.0
    cdef double primal = 0.0
    cdef double lagrangian = 0.0
    cdef double reconstruction = 0.0
    cdef double dual_step = 0.0
    cdef double norm_v_sq = 0.0
    cdef double norm_x_hat_sq = 0.0
    cdef double norm_lam_sq = 0.0
    cdef double norm_x_sq = 0.0

    cdef double denom = 1.0 + rho
    cdef double xh_ij, xh_ji, raw_ij, raw_ji, v_new
    cdef double prim_ij, prim_ji, lam_ij, lam_ji, d

    for i in prange(n, nogil=True, schedule='guided'):
        # --- diagonal element (symmetrization is a no-op there) ---
        xh_ij = x_hat[i, i]
        if obs[i, i]:
            raw_ij = ((x[i, i] + rho * xh_ij) - lam[i, i]) / denom
        else:
            raw_ij = lam[i, i] / (-rho) + xh_ij
        if use_min and raw_ij < bmin:
            raw_ij = bmin
        if use_max and raw_ij > bmax:
            raw_ij = bmax
        v_new = raw_ij

        d = v_new - v[i, i]
        dual_step += d * d
        v[i, i] = v_new

        prim_ij = v_new - xh_ij
        primal += prim_ij * prim_ij
        lam_ij = lam[i, i] + rho * prim_ij
        lam[i, i] = lam_ij
        lagrangian += lam_ij * prim_ij

        if obs[i, i]:
            d = x[i, i] - v_new
            data_fit += d * d
            d = x[i, i] - xh_ij
            reconstruction += d * d
            norm_x_sq += x[i, i] * x[i, i]

        norm_v_sq += v_new * v_new
        norm_x_hat_sq += xh_ij * xh_ij
        norm_lam_sq += lam_ij * lam_ij
        target[i, i] = lam_ij / rho + v_new

        # --- strict upper triangle, mirrored to the lower ---
        for j in range(i + 1, n):
            xh_ij = x_hat[i, j]
            xh_ji = x_hat[j, i]

            if obs[i, j]:
                raw_ij = ((x[i, j] + rho * xh_ij) - lam[i, j]) / denom
                raw_ji = ((x[j, i] + rho * xh_ji) - lam[j, i]) / denom
            else:
                raw_ij = lam[i, j] / (-rho) + xh_ij
                raw_ji = lam[j, i] / (-rho) + xh_ji

            if use_min:
                if raw_ij < bmin:
                    raw_ij = bmin
                if raw_ji < bmin:
                    raw_ji = bmin
            if use_max:
                if raw_ij > bmax:
                    raw_ij = bmax
                if raw_ji > bmax:
                    raw_ji = bmax

            v_new = 0.5 * (raw_ij + raw_ji)

            d = v_new - v[i, j]
            dual_step += d * d
            d = v_new - v[j, i]
            dual_step += d * d
            v[i, j] = v_new
            v[j, i] = v_new

            prim_ij = v_new - xh_ij
            prim_ji = v_new - xh_ji
            primal += prim_ij * prim_ij + prim_ji * prim_ji

            lam_ij = lam[i, j] + rho * prim_ij
            lam_ji = lam[j, i] + rho * prim_ji
            lam[i, j] = lam_ij
            lam[j, i] = lam_ji
            lagrangian += lam_ij * prim_ij + lam_ji * prim_ji

            if obs[i, j]:
                d = x[i, j] - v_new
                data_fit += d * d
                d = x[j, i] - v_new
                data_fit += d * d
                d = x[i, j] - xh_ij
                reconstruction += d * d
                d = x[j, i] - xh_ji
                reconstruction += d * d
                norm_x_sq += x[i, j] * x[i, j] + x[j, i] * x[j, i]

            norm_v_sq += 2.0 * (v_new * v_new)
            norm_x_hat_sq += xh_ij * xh_ij + xh_ji * xh_ji
            norm_lam_sq += lam_ij * lam_ij + lam_ji * lam_ji

            target[i, j] = lam_ij / rho + v_new
            target[j, i] = lam_ji / rho + v_new

    return AdmmStep(
        rho=rho,
        data_fit=data_fit,
        primal=primal,
        lagrangian=lagrangian,
        reconstruction=reconstruction,
        ss_x=norm_x_sq,
        dual_step=dual_step,
        v=norm_v_sq,
        x_hat=norm_x_hat_sq,
        lam=norm_lam_sq,
    )


# The blocked BLAS-3 variant is the production solver
update_w = update_w_blas_blocked
