"""B1: JAX-vectorised case-1 T_DIFF matches the NumPy reference."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402


def test_jax_tdiff_matches_numpy_and_batch():
    from lakematrix.jaxbatch.diffusion import jitted_t_diff_water
    from lakematrix.physics.tsolver import t_diff_water as np_tdiff

    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    st, sp = ss.state, ss.soil_params
    L, M, ns, nc = 4, case.M, case.ns, case.nsoilcols

    def rep(x):
        return np.repeat(np.asarray(x)[None, ...], L, axis=0)

    Tw1 = rep(np.r_[0, st.Tw1[0]])
    Tsoil1 = rep(ss.soil.Tsoil1)
    lamw = rep(np.r_[0, st.lamw[0]])
    area_int = rep(np.r_[0, ss.bathym.area_int[0]])
    area_half = rep(np.r_[0, ss.bathym.area_half[0]])
    ddz = rep(np.r_[0, grid.ddz])
    lamsoil = rep(ss.soil.lamsoil)
    csoil = rep(ss.soil.csoil)
    rosoil = rep(ss.soil.rosoil)
    dzs = rep(sp.dzs)
    rad = rep(np.asarray(ss.integr))
    dzeta = rep(np.r_[0, grid.dzeta_int])
    ddz05 = rep(grid.ddz05)
    h1 = np.full(L, float(st.h1[0]))
    zero = np.zeros(L)
    P = np.zeros((L, M + 2))
    Tsurf = np.full(L, 20.0)

    jtw, jts = jitted_t_diff_water(
        Tsurf, 20.0, Tw1, Tsoil1, lamw, area_int, area_half, h1, ddz,
        zero, zero, P, P, lamsoil, csoil, rosoil, dzs, rad, dzeta,
        ddz05, M, ns, nc, 0.0, None)
    ntw, nts = np_tdiff(
        Tsurf[0], 20.0, Tw1[0], Tsoil1[0], lamw[0], area_int[0],
        area_half[0], h1[0], ddz[0], 0.0, 0.0, P[0], P[0], lamsoil[0],
        csoil[0], rosoil[0], dzs[0], rad[0], dzeta[0], ddz05[0], M, ns,
        nc, 0.0, None)
    np.testing.assert_allclose(np.asarray(jtw)[0], ntw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jts)[0], nts, rtol=0, atol=1e-12)
    # all batch members see the same state
    np.testing.assert_array_equal(np.asarray(jtw), np.broadcast_to(np.asarray(jtw)[0], np.asarray(jtw).shape))
