"""B3: JAX batched case-1 ``SOILFORLAKE`` matches the NumPy reference.

The batch kernel is exercised on the deepest soil column of the real
``summer60d`` setup.  ``Tsoil2`` is not hand-made: it is taken from a NumPy
``soilforlake`` call (the first call warms up the phase-iteration work array,
the second call produces the reference from the pristine state).  A second,
synthetic cold/icy snapshot uses the same soil geometry and coefficients but
exercises the melting/freezing and excess-water branches that the warm initial
state never reaches.
"""

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402

ATOL = 1e-12
DT = 20.0


def _case():
    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    return case, build_step_state(case, grid)


def _params(ss):
    sp = ss.soil_params
    return dict(
        ns=int(sp.ns), nsoilcols=int(sp.nsoilcols),
        POR=np.asarray(sp.POR), rosdry=np.asarray(sp.rosdry),
        dzs=np.asarray(sp.dzs), dzss=np.asarray(sp.dzss),
        zsoil=np.asarray(sp.zsoil),
        # UNFRWAT coefficients: not in the geo/type list but required by the
        # reference (they come from the same SoilParams object).
        WLM0=np.asarray(sp.WLM0), WLM7=np.asarray(sp.WLM7),
        nmeltpoint=int(sp.nmeltpoint),
        soilbotflx=float(sp.soilbotflx),
        pressure=float(ss.soil.pressure),
    )


def _batch(soil, L):
    """One dict of leading-lake-axis arrays; every lake gets the same state."""
    def rep(x):
        return np.repeat(np.asarray(x, dtype=float)[None, ...], L, axis=0)

    return dict(
        Tsoil1=rep(soil.Tsoil1), Tsoil3=rep(soil.Tsoil3),
        wl1=rep(soil.wl1), wl4=rep(soil.wl4),
        wi1=rep(soil.wi1), wi2=rep(soil.wi2),
        Sals1=rep(soil.Sals1),
        csoil=rep(soil.csoil), rosoil=rep(soil.rosoil),
        lammoist=rep(soil.lammoist), filtr=rep(soil.filtr),
        wa=rep(soil.wa), methgenmh=rep(soil.methgenmh),
        h1=np.full(L, soil.h1),
    )


def _np_soilforlake(*args, **kwargs):
    from lakematrix.physics.soil import soilforlake
    return soilforlake(*args, **kwargs)


def _reference(ss, *, tweak=None, dt=DT):
    """NumPy reference, the pristine input state and the shared ``Tsoil2``.

    The first NumPy call starts from ``Tsoil2 = Tsoil1`` of the deepest column
    and supplies a realistic post-``T_DIFF``/phase-relaxed ``Tsoil2``; the
    second call then runs from the pristine soil state with that input.
    """
    nc = ss.soil_params.nsoilcols
    base = copy.deepcopy(ss.soil)
    if tweak is not None:
        tweak(base)

    warm = copy.deepcopy(base)
    warm.Tsoil2 = np.array(warm.Tsoil1[:, nc], copy=True)
    warm = _np_soilforlake(warm, ss.soil_params, dt, is_=nc)
    Tsoil2 = np.array(warm.Tsoil2, copy=True)

    ref = copy.deepcopy(base)
    ref.Tsoil2 = np.array(Tsoil2, copy=True)
    ref = _np_soilforlake(ref, ss.soil_params, dt, is_=nc)
    return base, ref, Tsoil2


def _check(ss, L, *, tweak=None):
    from lakematrix.jaxbatch.soilforlake import soilforlake as jax_soilforlake

    base, ref, Tsoil2 = _reference(ss, tweak=tweak)
    state = _batch(base, L)
    T2 = np.repeat(Tsoil2[None, ...], L, axis=0)
    out = jax_soilforlake(state, _params(ss), T2, DT)

    for key in ("Tsoil1", "Tsoil3", "wl1", "wl4", "wi1", "wi2", "Sals1",
                "wa", "methgenmh"):
        want = np.asarray(getattr(ref, key), dtype=float)
        got = np.asarray(out[key], dtype=float)
        assert got.shape == (L,) + want.shape
        for lake in range(L):
            np.testing.assert_allclose(got[lake], want, rtol=0, atol=ATOL,
                                       err_msg=f"{key}, lake {lake}")
    # The phase-iteration work value is part of the reference output too.
    np.testing.assert_allclose(
        np.asarray(out["Tsoil2"]),
        np.broadcast_to(np.asarray(ref.Tsoil2), (L, ref.Tsoil2.size)),
        rtol=0, atol=ATOL)
    for key in ("wl2", "wl3"):
        np.testing.assert_allclose(np.asarray(out[key])[0],
                                   np.asarray(getattr(ref, key)),
                                   rtol=0, atol=ATOL, err_msg=key)
    np.testing.assert_allclose(np.asarray(out["dhwfsoil"]),
                               np.full(L, ref.dhwfsoil), rtol=0, atol=ATOL)
    if L > 1:
        for key in ("Tsoil1", "Tsoil3", "wl1", "wl4", "wi1", "wi2",
                    "Tsoil2", "wa", "dhwfsoil"):
            np.testing.assert_array_equal(
                np.asarray(out[key]),
                np.broadcast_to(np.asarray(out[key])[:1],
                                np.asarray(out[key]).shape))
    return out, ref


def test_jax_soilforlake_open_water_L1():
    ss = _case()[1]
    _check(ss, 1)


def test_jax_soilforlake_open_water_L4():
    ss = _case()[1]
    _check(ss, 4)


def _freezing_snapshot(soil):
    """Cold, ice-free state: ``Tsoil2 < 0`` forces the freezing branch."""
    nc = soil.Tsoil1.shape[1] - 1
    soil.wl1[1:, nc] = 0.55
    soil.wi1[1:, nc] = 0.0
    soil.Tsoil1[1:, nc] = -4.0


def _icy_snapshot(soil):
    """Partially frozen column with a top-to-bottom temperature gradient."""
    nc = soil.Tsoil1.shape[1] - 1
    T = soil.Tsoil1[1:, nc].copy()
    T[:3] = -8.0
    T[3:6] = -2.0
    T[6:] = 2.0
    soil.Tsoil1[1:, nc] = T
    soil.wl1[1:, nc] = 0.30
    soil.wi1[1:, nc] = 0.45
    soil.Sals1[1:, nc] = 0.003


def test_jax_soilforlake_phase_branches_L4():
    ss = _case()[1]
    _check(ss, 4, tweak=_freezing_snapshot)
    _check(ss, 4, tweak=_icy_snapshot)


def test_jax_soilforlake_jit_wrapper_matches_numpy():
    from lakematrix.jaxbatch.soilforlake import make_soilforlake

    ss = _case()[1]
    ref, Tsoil2 = _reference(ss)[1:]
    L = 4
    state = _batch(ss.soil, L)
    T2 = np.repeat(Tsoil2[None, ...], L, axis=0)
    kernel = make_soilforlake(ss.soil_params.ns,
                              ss.soil_params.nsoilcols,
                              nmeltpoint=ss.soil_params.nmeltpoint)
    out = kernel(state, _params(ss), T2, DT)
    for key in ("Tsoil1", "wl1", "wi1", "Tsoil3", "wl4", "wi2"):
        np.testing.assert_allclose(
            np.asarray(out[key])[0], np.asarray(getattr(ref, key)),
            rtol=0, atol=ATOL, err_msg=key)
    np.testing.assert_allclose(np.asarray(out["dhwfsoil"]),
                               np.full(L, ref.dhwfsoil), rtol=0, atol=ATOL)
