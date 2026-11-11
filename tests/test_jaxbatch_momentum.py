"""B3: JAX batched momentum_solver matches the NumPy reference (case 1).

The reference state comes from ``build_step_state`` on the real
``ref/v1/summer60d`` case (``ss.mom`` / ``ss.mom_params``); the comparison is
``u2``, ``v2`` and the updated ``k2`` for one and four lakes in the batch, at
1e-10 absolute, plus every scalar diagnostic the kernel returns.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.momentum import (MomentumParams, MomentumState,  # noqa: E402
                                         momentum_solver as np_momentum_solver)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402

UWIND = 5.0        # m/s, exercises the wind-stress terms
VWIND = 1.5
CDMW = 1.27e-3

# diagnostics returned by the JAX kernel and by MomentumResult
SCALARS = ("taux", "tauy", "tau_air", "tau_gr", "tau_wav", "tau_sbl",
           "u_star", "velfrict_bot", "velfrict_ice", "wind10", "C10", "ufr",
           "coef1", "coef2", "Cz_gr", "Cz_i", "rp", "windwork", "u", "v",
           "urel", "vrel", "urels", "vrels", "wr")
FIELDS = ("k2_old", "E1", "eps1", "uv", "Ri", "knum", "veg_sw",
          "lsu_water", "lsv_water")
VECTORS = ("Force_x", "Force_y")


def _case():
    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    dt = float(np.atleast_1d(case.dt)[0])
    return ss, dt


def _rep(x, L):
    return np.repeat(np.asarray(x, dtype=float)[None, ...], L, axis=0)


def _batched(ms, mp, L):
    """Turn one MomentumState/MomentumParams into batched dicts."""
    state = dict(
        u1=_rep(ms.u1, L), v1=_rep(ms.v1, L), E1=_rep(ms.E1, L),
        eps1=_rep(ms.eps1, L), k2=_rep(ms.k2, L), k2_old=_rep(ms.k2_old, L),
        row=_rep(ms.row, L), knum=_rep(ms.knum, L),
        veg_sw=_rep(ms.veg_sw, L), Tw1=_rep(ms.Tw1, L),
        Sal1=_rep(ms.Sal1, L), lsu_water=_rep(ms.lsu_water, L),
        lsv_water=_rep(ms.lsv_water, L),
        h1=np.full(L, ms.h1), dhw=np.full(L, ms.dhw),
        dhw0=np.full(L, ms.dhw0), l1=np.full(L, ms.l1),
        ls1=np.full(L, ms.ls1), alphax=np.full(L, ms.alphax),
        alphay=np.full(L, ms.alphay),
        velfrict_ice=np.full(L, ms.velfrict_ice))
    params = dict(
        ddz=_rep(mp.ddz, L), ddz05=_rep(mp.ddz05, L),
        dzeta_int=_rep(mp.dzeta_int, L),
        dzeta_05int=_rep(mp.dzeta_05int, L),
        area_int=_rep(mp.area_int, L), area_half=_rep(mp.area_half, L),
        Lx=_rep(mp.Lx, L), Ly=_rep(mp.Ly, L),
        Lx_half=_rep(mp.Lx_half, L), Ly_half=_rep(mp.Ly_half, L),
        depth_area=_rep(mp.depth_area, L), kor=np.full(L, mp.kor),
        uwind=np.full(L, UWIND), vwind=np.full(L, VWIND),
        cdmw=np.full(L, CDMW), zref=np.full(L, mp.zref),
        fetch=np.full(L, mp.fetch))
    return state, params


def _start_state():
    """The step-1 state of ``summer60d`` with the initial-momentum forcing."""
    ss, dt = _case()
    ms, mp = ss.mom, ss.mom_params
    # NOTE: `build_step_state` stores `row` as a 1-based array of length M+2
    # (`water_dens_ts(np.r_[0, Tw1[l]], ...)`: index 0 is a spurious density at
    # T=0/S=0), while `MomentumState.row` is documented as 0-based (M+1) for
    # the solver.  `step.step_case1_to_solver` sets the 0-based version
    # (`water_dens_ts(st.Tw1[l], st.Sal1[l])`), so drop the pad here rather
    # than touching the shared builder.
    ms = MomentumState(**{**ms.__dict__, "row": np.asarray(ms.row)[1:]})
    mp = MomentumParams(**{**mp.__dict__, "uwind": UWIND, "vwind": VWIND,
                           "cdmw": CDMW})
    return ms, mp, dt


def _check(L, dt, ms, mp, knum_keys=("u2", "v2", "k2"), **static):
    """Compare the batched kernel (eager and jitted) with NumPy for L lakes."""
    from lakematrix.jaxbatch.momentum import (jitted_momentum_solver,
                                              momentum_solver)

    ref = np_momentum_solver(ms, mp, dt)
    state, params = _batched(ms, mp, L)
    out = momentum_solver(state, params, dt, **static)
    jout = jitted_momentum_solver(state, params, dt, **static)

    expected = {"u2": ref.u2, "v2": ref.v2, "k2": ref.state.k2}
    for key in knum_keys:
        expect = np.tile(np.asarray(expected[key]), (L, 1))
        got = np.asarray(out[key])
        np.testing.assert_allclose(got, expect, rtol=0, atol=1e-10)
        # The jitted kernel must meet the same 1e-10 tolerance; it may differ
        # from the eager one in the last bits (XLA turns x**-1 into a division
        # and fuses multiply-adds), so compare the two relatively.
        np.testing.assert_allclose(np.asarray(jout[key]), expect,
                                   rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(jout[key]), got, rtol=1e-12,
                                   atol=0)
        # all batch members see the same state -> identical results
        np.testing.assert_array_equal(
            got, np.broadcast_to(got[0], got.shape))
    return ref, out, jout


def test_jax_momentum_matches_numpy_one_lake():
    ms, mp, dt = _start_state()
    _check(1, dt, ms, mp)


def test_jax_momentum_matches_numpy_four_lakes():
    ms, mp, dt = _start_state()
    _check(4, dt, ms, mp)


def test_jax_momentum_second_step_with_k2_old():
    """k2_old != 0 exercises the implicit vertical-diffusion coefficients."""
    ms, mp, dt = _start_state()
    ref = np_momentum_solver(ms, mp, dt)
    ms = ref.state
    ms.u1 = np.array(ref.u2)
    ms.v1 = np.array(ref.v2)
    ms.firstcall = False
    _check(4, dt, ms, mp)


def test_jax_momentum_diagnostics_match_numpy():
    """Every scalar/array diagnostic of MomentumResult, not just u2/v2/k2."""
    ms, mp, dt = _start_state()
    ref, out, _ = _check(1, dt, ms, mp)
    for key in SCALARS:
        np.testing.assert_allclose(np.asarray(out[key]).reshape(-1),
                                   np.asarray(getattr(ref, key)).reshape(-1),
                                   rtol=0, atol=1e-10, err_msg=key)
    for key in VECTORS:
        np.testing.assert_allclose(np.asarray(out[key])[0],
                                   np.asarray(getattr(ref, key)),
                                   rtol=0, atol=1e-10, err_msg=key)
    for key in FIELDS:
        np.testing.assert_allclose(np.asarray(out[key])[0],
                                   np.asarray(getattr(ref.state, key)),
                                   rtol=0, atol=1e-10, err_msg=key)
    np.testing.assert_allclose(np.asarray(out["discharge"])[0],
                               np.asarray(ref.discharge), rtol=0, atol=1e-10)
    # Fortran leaves b0/tau_i undefined outside 0 < l1 <= L0; the NumPy
    # reference returns None there and the kernel returns 0 (documented).
    assert ref.b0 is None and ref.tau_i is None
    np.testing.assert_array_equal(np.asarray(out["b0"]), np.zeros(1))
    np.testing.assert_array_equal(np.asarray(out["tau_i"]), np.zeros(1))
    assert out["xbot"] == 0.0 and out["horvisc_on"] == 0


@pytest.mark.parametrize("variant", ["no_lateral_friction", "slopes",
                                     "galperin", "momflxpart"])
def test_jax_momentum_branch_variants(variant):
    """``jnp.where`` branches and the optional switch paths of Turbpar=2."""
    ms, mp, dt = _start_state()
    if variant == "no_lateral_friction":
        depth_area = np.asarray(mp.depth_area, dtype=float).copy()
        depth_area[0, 1] = -999.0          # Fortran: skip the lsu/lsv block
        mp = MomentumParams(**{**mp.__dict__, "depth_area": depth_area})
        # keep non-zero lateral-friction state to prove it is carried over
        ms = MomentumState(**{**ms.__dict__,
                              "lsu_water": np.full(ms.u1.size, 7.0e-4),
                              "lsv_water": np.full(ms.u1.size, -3.0e-4)})
        _check(4, dt, ms, mp)
    elif variant == "slopes":
        ms = MomentumState(**{**ms.__dict__, "alphax": 1.0e-4,
                              "alphay": -2.0e-4})
        ref, out, _ = _check(4, dt, ms, mp)
        # the slope terms are actually used (u2 would differ without them)
        ms0 = MomentumState(**{**ms.__dict__, "alphax": 0.0, "alphay": 0.0})
        ref0 = np_momentum_solver(ms0, mp, dt)
        assert np.abs(ref.u2 - ref0.u2).max() > 1e-8
        np.testing.assert_allclose(np.asarray(out["u2"])[0], ref.u2,
                                   rtol=0, atol=1e-10)
    else:
        static = {"galperin": {"stabfunc": 3},
                  "momflxpart": {"momflxpart": 0.4}}[variant]
        mp = MomentumParams(**{**mp.__dict__, **static})
        _check(4, dt, ms, mp, **static)
