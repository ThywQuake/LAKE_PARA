"""Batched 2x2 block-tridiagonal solver (Fortran `MATRIXPROGONKA`).

Fortran: `numerics_mod.f90:108-172`.  The system is

    A(k) y(k-1) + C(k) y(k) + B(k) y(k+1) = D(k),  k = 1..N

with 2x2 blocks and 2-vectors.  Python arrays are 0-based:
`A,B,C` shape (..., N, 2, 2), `D` shape (..., N, 2).
"""

import numpy as np


def matrixprogonka(A, B, C, D, n=None):
    A, B, C, D = (np.asarray(x, dtype=float) for x in (A, B, C, D))
    N = A.shape[-3] if n is None else n
    lead = A.shape[:-3]
    alpha = np.zeros(lead + (N, 2, 2))
    beta = np.zeros(lead + (N, 2))

    inv = np.linalg.inv
    y = np.zeros(lead + (N, 2))
    if N == 1:
        y[..., 0, :] = np.einsum('...ij,...j->...i', inv(C[..., 0, :, :]), D[..., 0, :])
        return y
    # Python alpha[j] == Fortran alpha(j+1).  Fortran alpha(2) is from row 1.
    alpha[..., 1, :, :] = inv(C[..., 0, :, :]) @ B[..., 0, :, :]
    beta[..., 1, :] = np.einsum('...ij,...j->...i', inv(C[..., 0, :, :]), D[..., 0, :])
    for j in range(2, N):
        den = C[..., j - 1, :, :] - A[..., j - 1, :, :] @ alpha[..., j - 1, :, :]
        x32 = inv(den)
        alpha[..., j, :, :] = x32 @ B[..., j - 1, :, :]
        rhs = D[..., j - 1, :] + np.einsum('...ij,...j->...i', A[..., j - 1, :, :],
                                            beta[..., j - 1, :])
        beta[..., j, :] = np.einsum('...ij,...j->...i', x32, rhs)
    # Final block Y(N).
    j = N - 1
    den = C[..., j, :, :] - A[..., j, :, :] @ alpha[..., j, :, :]
    rhs = D[..., j, :] + np.einsum('...ij,...j->...i', A[..., j, :, :], beta[..., j, :])
    y[..., j, :] = np.einsum('...ij,...j->...i', inv(den), rhs)
    for j in range(N - 1, 0, -1):
        y[..., j - 1, :] = np.einsum('...ij,...j->...i', alpha[..., j, :, :],
                                     y[..., j, :]) + beta[..., j, :]
    return y


def as_dense(A, B, C, n=None):
    """Dense (2N, 2N) matrix of the block system, for tests."""
    N = A.shape[-3] if n is None else n
    M = np.zeros((2 * N, 2 * N))
    M[0:2, 0:2] = C[0]
    M[0:2, 2:4] = -B[0]
    for k in range(1, N):
        M[2 * k:2 * k + 2, 2 * (k - 1):2 * k] = -A[k]
        M[2 * k:2 * k + 2, 2 * k:2 * k + 2] = C[k]
        if k + 1 < N:
            M[2 * k:2 * k + 2, 2 * (k + 1):2 * (k + 2)] = -B[k]
    return M
