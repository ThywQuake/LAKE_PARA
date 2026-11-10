"""M2 tests: tridiagonal solver and density against NumPy / the Fortran probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.physics.tridiag import as_matrix, progonka, progonka_jax  # noqa: E402

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-dens" / "probe" / "1"


def _system(L=4, P=40, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((L, P))
    b = rng.random((L, P))
    c = 2.0 + rng.random((L, P)) + a + b
    f = rng.random((L, P))
    return a, b, c, f


def test_tridiag_matches_dense():
    a, b, c, f = _system()
    y = progonka(a, b, c, f)
    yj = np.asarray(progonka_jax(a, b, c, f))
    for l in range(a.shape[0]):
        ref = np.linalg.solve(as_matrix(a[l], b[l], c[l]), f[l])
        np.testing.assert_allclose(y[l], ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(yj[l], ref, rtol=0, atol=1e-12)


def test_tridiag_inner_k_n():
    a, b, c, f = _system(P=30)
    k, n = 5, 24
    y = progonka(a, b, c, f, k, n)
    yj = np.asarray(progonka_jax(a, b, c, f, k, n))
    for l in range(a.shape[0]):
        A = as_matrix(a[l], b[l], c[l], k, n)[k:n + 1, k:n + 1]
        ref = np.linalg.solve(A, f[l, k:n + 1])
        np.testing.assert_allclose(y[l, k:n + 1], ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(yj[l, k:n + 1], ref, rtol=0, atol=1e-12)


@pytest.mark.skipif(not PROBE.exists(), reason="density probe not generated")
def test_density_matches_fortran():
    from lakematrix.physics.density import water_dens_ts
    from probe import read_probe

    for tag in ("DENS_975", "DENS_1192"):
        inp = read_probe(PROBE / f"{tag}_in.bin")
        out = read_probe(PROBE / f"{tag}_out.bin")
        row = water_dens_ts(inp["Tw1"], inp["Sal1"])
        np.testing.assert_array_equal(row, out["row"])


def test_density_batch_consistency():
    from lakematrix.physics.density import water_dens_ts

    rng = np.random.default_rng(1)
    T = 5.0 + rng.random((3, 41)) * 20.0
    S = rng.random((3, 41)) * 1e-3
    batch = water_dens_ts(T, S)
    for l in range(3):
        np.testing.assert_array_equal(batch[l], water_dens_ts(T[l], S[l]))
