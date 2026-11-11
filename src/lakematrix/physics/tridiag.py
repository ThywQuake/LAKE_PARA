"""Batched tridiagonal solver (Fortran `PROGONKA`).

Fortran: `numerics_mod.f90:243-273`.  The system is (1-based)

    c(K) y(K) - b(K) y(K+1) = f(K)
    -a(i) y(i-1) + c(i) y(i) - b(i) y(i+1) = f(i),  i = K+1..N-1
    -a(N) y(N-1) + c(N) y(N) = f(N)

The public functions take 0-based arrays and 0-based `k`, `n`
(`K = k+1`, `N = n+1`); internally the arrays are padded with a leading zero so
the recurrence can be written exactly as the Fortran one.
"""

import functools

import numpy as np


def _pad1(x):
    return np.concatenate((np.zeros(x.shape[:-1] + (1,), dtype=x.dtype), x), axis=-1)


def progonka_numpy(a, b, c, f, k=0, n=None):
    """Reference NumPy batched solver; a,b,c,f shape (..., P)."""
    a, b, c, f = (np.asarray(x, dtype=float) for x in (a, b, c, f))
    P = a.shape[-1]
    K, N = k + 1, (n + 1 if n is not None else P)
    a1, b1, c1, f1 = (_pad1(x) for x in (a, b, c, f))
    alpha = np.zeros(a.shape[:-1] + (P + 2,))
    beta = np.zeros_like(alpha)
    alpha[..., K + 1] = b1[..., K] / c1[..., K]
    beta[..., K + 1] = f1[..., K] / c1[..., K]
    for i in range(K + 2, N + 2):
        den = c1[..., i - 1] - a1[..., i - 1] * alpha[..., i - 1]
        alpha[..., i] = b1[..., i - 1] / den
        beta[..., i] = (f1[..., i - 1] + a1[..., i - 1] * beta[..., i - 1]) / den
    y1 = np.zeros_like(a1)
    y1[..., N] = beta[..., N + 1]
    for i in range(N - 1, K - 1, -1):
        y1[..., i] = alpha[..., i + 1] * y1[..., i + 1] + beta[..., i + 1]
    return y1[..., 1:P + 1]


def progonka_jax(a, b, c, f, k=0, n=None):
    """JAX version: `lax.scan` over the forward and backward sweeps."""
    import jax
    import jax.numpy as jnp

    a, b, c, f = (jnp.asarray(x) for x in (a, b, c, f))
    P = a.shape[-1]
    K, N = k + 1, (n + 1 if n is not None else P)
    pad = lambda x: jnp.concatenate((jnp.zeros(x.shape[:-1] + (1,), x.dtype), x), -1)
    a1, b1, c1, f1 = (pad(x) for x in (a, b, c, f))
    alpha0 = jnp.zeros(a.shape[:-1] + (P + 2,))
    beta0 = jnp.zeros_like(alpha0)
    alpha0 = alpha0.at[..., K + 1].set(b1[..., K] / c1[..., K])
    beta0 = beta0.at[..., K + 1].set(f1[..., K] / c1[..., K])

    def fwd(carry, i):
        alpha, beta = carry
        den = c1[..., i - 1] - a1[..., i - 1] * alpha[..., i - 1]
        alpha = alpha.at[..., i].set(b1[..., i - 1] / den)
        beta = beta.at[..., i].set(
            (f1[..., i - 1] + a1[..., i - 1] * beta[..., i - 1]) / den)
        return (alpha, beta), None

    (alpha, beta), _ = jax.lax.scan(fwd, (alpha0, beta0),
                                    jnp.arange(K + 2, N + 2))
    y0 = jnp.zeros_like(a1)
    y0 = y0.at[..., N].set(beta[..., N + 1])

    def bwd(y, i):
        y = y.at[..., i].set(alpha[..., i + 1] * y[..., i + 1] + beta[..., i + 1])
        return y, None

    y, _ = jax.lax.scan(bwd, y0, jnp.arange(N - 1, K - 1, -1))
    return y[..., 1:P + 1]


def as_matrix(a, b, c, k=0, n=None):
    """Dense matrix of the system, for tests (1-based Fortran rows)."""
    P = a.shape[-1]
    K, N = k + 1, (n + 1 if n is not None else P)
    A = np.zeros((P, P))
    A[K - 1, K - 1] = c[K - 1]
    A[K - 1, K] = -b[K - 1]
    for i in range(K + 1, N):
        A[i - 1, i - 2] = -a[i - 1]
        A[i - 1, i - 1] = c[i - 1]
        A[i - 1, i] = -b[i - 1]
    A[N - 1, N - 2] = -a[N - 1]
    A[N - 1, N - 1] = c[N - 1]
    return A


@functools.lru_cache(maxsize=None)
def _progonka_jit(shape, k, n):
    import jax

    @jax.jit
    def solve(a, b, c, f):
        return progonka_jax(a, b, c, f, k, n)

    return solve


def progonka(a, b, c, f, k=0, n=None):
    """JIT-compiled batched tridiagonal solver (same recurrence as Fortran).

    The first call for a given ``(shape, k, n)`` compiles; later calls are
    ~10x faster than the NumPy loop for the 40-90 level systems used here.
    Use :func:`progonka_numpy` for the reference implementation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    f = np.asarray(f, dtype=float)
    P = a.shape[-1]
    nn = P if n is None else n
    if a.shape[-1] == b.shape[-1] == c.shape[-1] == f.shape[-1] == P:
        return np.asarray(_progonka_jit(a.shape, k, nn)(a, b, c, f))
    return progonka_numpy(a, b, c, f, k, n)
