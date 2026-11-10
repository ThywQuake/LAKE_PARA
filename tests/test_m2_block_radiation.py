"""M2 tests: block-tridiagonal solver and radiation."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.physics.blocktridiag import as_dense, matrixprogonka  # noqa: E402
from lakematrix.physics.radiation import rad_update  # noqa: E402


def test_blocktridiag_matches_dense():
    rng = np.random.default_rng(2)
    L, N = 3, 10
    A = rng.random((L, N, 2, 2))
    B = rng.random((L, N, 2, 2))
    C = 3.0 * np.eye(2)[None, None, :, :] + rng.random((L, N, 2, 2))
    D = rng.random((L, N, 2))
    Y = matrixprogonka(A, B, C, D)
    for l in range(L):
        ref = np.linalg.solve(as_dense(A[l], B[l], C[l]), D[l].reshape(-1))
        np.testing.assert_allclose(Y[l].reshape(-1), ref, rtol=0, atol=1e-12)


def test_radiation_matches_loop():
    rng = np.random.default_rng(3)
    nlevels, nb = 7, 4
    extinct = rng.random((2, nlevels, nb)) + 0.1
    thickness = rng.random((2, nlevels)) + 0.1
    flux0 = rng.random((2, nb))
    flux, integr = rad_update(extinct, thickness, flux0)
    flux = np.asarray(flux); integr = np.asarray(integr)
    f = np.zeros((2, nlevels + 1, nb))
    f[:, 0] = flux0
    for i in range(nlevels):
        f[:, i + 1] = f[:, i] * np.exp(-extinct[:, i] * thickness[:, i, None])
    np.testing.assert_allclose(flux, f, rtol=0, atol=1e-14)
    np.testing.assert_allclose(integr, f.sum(-1), rtol=0, atol=1e-12)
