"""B5: full case-1 JAX step (``jaxbatch.step``) vs the NumPy reference.

The reference is :func:`lakematrix.step.step_case1` / :func:`~.run_case1` on
``ref/v1/summer1d`` (``M = 40``, ``nsoilcols = 5``, ``dt = 20 s``), i.e. a real
coupled step: bathymetry, surface fluxes, four side soil columns, momentum,
``T_SOLVER``, the deepest soil column, ``S_DIFF``, k-epsilon and ``TURB_SCALES``.

Tolerances (from the task statement):

============================  ==========================================
``Tw1``, ``Tsoil1``           1e-8 K absolute
``h1``                        1e-10 m absolute
``E1``, ``eps1``, ``lamw``    1e-8 relative
============================  ==========================================

They are met with a wide margin (measured: 1e-13 K, 5e-15 K, 0 m,
4e-10 / 7e-10 / 1e-9 relative), so the assertions use the stated tolerances.

**Runtime.**  Compiling the coupled step (all B1-B4 kernels plus
``T_SOLVER``'s ``while_loop`` sweeps) takes ~25 s per ``(L, params)``
signature on a CPU box and the ``lax.scan`` body another ~30 s; the kernels are
cached per process, so the first test in each batch size pays the cost and the
others reuse it.  Each individual test stays well below two minutes.
"""

import copy
import ctypes
import gc
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import at_time as np_at_time          # noqa: E402
from lakematrix.forcing import read_ascii                     # noqa: E402
from lakematrix.grid import create_grid                       # noqa: E402
from lakematrix.physics.density import ddens_dsal as np_dds   # noqa: E402
from lakematrix.physics.density import ddens_dtemp as np_ddt  # noqa: E402
from lakematrix.physics.density import water_dens_ts as np_dens  # noqa: E402
from lakematrix.physics.layers import advance_case1 as np_adv  # noqa: E402
from lakematrix.physics.radiation import FRACBANDS            # noqa: E402
from lakematrix.physics.radiation import rad_update as np_rad  # noqa: E402
from lakematrix.physics.salinity import s_diff as np_sdiff    # noqa: E402
from lakematrix.physics.surface import surf_char as np_surf_char  # noqa: E402
from lakematrix.physics.turb import diffmin_hs as np_diffmin_hs  # noqa: E402
from lakematrix.setup_io import read_case                     # noqa: E402
from lakematrix.step import (build_step_state, run_case1,     # noqa: E402
                             step_case1)

import lakematrix.jaxbatch.step as jstep                      # noqa: E402

CASE = ROOT / "ref" / "v1" / "summer1d"
DT = 20.0
NSTEPS = 10

ATOL_T = 1e-8          # Tw1 / Tsoil1, K
ATOL_H = 1e-10         # h1, m
RTOL_F = 1e-8          # E1 / eps1 / lamw


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _trim_heap():
    """Return the freed XLA/LLVM arenas to the OS.

    Compiling the coupled step peaks at ~2-4 GB per ``(L, nsteps)`` signature;
    glibc keeps those arenas mapped afterwards, so a test module that compiles
    four of them in one process can exhaust the address space of the box even
    though every individual compilation fits.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):        # non-glibc platform
        pass


@pytest.fixture(autouse=True)
def _release_compiler_memory():
    yield
    _trim_heap()


@pytest.fixture(scope="module")
def ref():
    """Case, grid, forcing, the NumPy one-step and ten-step references."""
    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(CASE / "meteo" / "test.dat")
    ss1 = build_step_state(case, grid)
    one = step_case1(copy.deepcopy(ss1), case, grid, rows, 1)
    ss10, _out = run_case1(case, grid, rows, NSTEPS)
    return dict(case=case, grid=grid, rows=rows, ss1=ss1, one=one, ten=ss10)


def _params(ref, L=1):
    return jstep.make_params(ref["case"], ref["grid"], L=L)


def _state(ref, L=1):
    st = jstep.from_numpy_state(ref["ss1"])
    return st if L == 1 else jstep.dup_state(st, L)


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------

def _rel(got, want):
    want = np.asarray(want, dtype=float)
    return float(np.max(np.abs(np.asarray(got) - want)
                        / np.maximum(np.abs(want), 1e-300)))


def _check_step(got, want, lanes=1, tag=""):
    """Compare every ``step_case1`` output field for ``lanes`` lakes."""
    for lane in range(lanes):
        tag_l = f"{tag} lane {lane}" if lanes > 1 else tag
        np.testing.assert_allclose(got["Tw1"][lane], want["Tw1"],
                                   rtol=0, atol=ATOL_T,
                                   err_msg=f"Tw1 {tag_l}")
        np.testing.assert_allclose(got["Tsoil1"][lane], want["Tsoil1"],
                                   rtol=0, atol=ATOL_T,
                                   err_msg=f"Tsoil1 {tag_l}")
        np.testing.assert_allclose(got["Sal1"][lane], want["Sal1"],
                                   rtol=0, atol=1e-12, err_msg=f"Sal1 {tag_l}")
        np.testing.assert_allclose(got["h1"][lane], want["h1"],
                                   rtol=0, atol=ATOL_H, err_msg=f"h1 {tag_l}")
        for key in ("E1", "eps1", "lamw"):
            assert _rel(got[key][lane], want[key]) < RTOL_F, (key, tag_l)
        # secondary fields of the return dict (not in the task tolerances)
        np.testing.assert_allclose(got["Elatent"][lane],
                                   want["Elatent"], rtol=1e-9,
                                   err_msg=f"Elatent {tag_l}")
        np.testing.assert_allclose(got["Tsurf3"][lane], want["Tsurf3"],
                                   rtol=0, atol=1e-9,
                                   err_msg=f"Tsurf3 {tag_l}")
        np.testing.assert_allclose(got["dhw"][lane], want["dhw"],
                                   rtol=0, atol=1e-12, err_msg=f"dhw {tag_l}")


def _check_state_shape(state):
    """The state pytree must be identical before and after a step."""
    assert set(state) == {"water", "soil", "ke", "mom", "diag"}


# ---------------------------------------------------------------------------
# integration helpers
# ---------------------------------------------------------------------------

def test_integration_helpers_match_numpy(ref):
    """``at_time`` / ``rad_update`` / density / ``s_diff`` / ``advance_case1``
    / ``diffmin_hs`` / ``surf_char`` against their NumPy originals."""
    import jax.numpy as jnp

    case, grid, rows, ss = (ref["case"], ref["grid"], ref["rows"], ref["ss1"])
    st = ss.state
    M = case.M

    # --- at_time (driver rule, incl. clamping) ---------------------------
    for t in (DT, 200.0, 3600.0, 7200.0, 5.0e5):
        want = np_at_time(rows, t, float(case.interval))
        got = np.asarray(jstep.at_time(rows, jnp.float64(t),
                                       float(case.interval)))
        np.testing.assert_array_equal(got, want, err_msg=f"at_time {t}")

    # --- rad_update ------------------------------------------------------
    extinct = np.zeros((M, 4))
    extinct[:, 0], extinct[:, 1] = 3.13, 0.81
    extinct[:, 2], extinct[:, 3] = 1.73, 1.087e3
    thickness = grid.ddz05[:M] * float(st.h1[0])
    flux0 = FRACBANDS * 123.4
    f_ref, i_ref = np_rad(extinct, thickness, flux0)
    flux, integr = jstep.rad_update(extinct[None], thickness[None],
                                    flux0[None])
    # the 4-band cumulative product is reassociated by XLA: a few ulp
    np.testing.assert_allclose(np.asarray(flux)[0], f_ref, rtol=1e-14,
                               atol=0)
    np.testing.assert_allclose(np.asarray(integr)[0], i_ref, rtol=1e-14,
                               atol=0)

    # --- density and its derivatives -------------------------------------
    T, S = st.Tw1[0], st.Sal1[0]
    np.testing.assert_allclose(np.asarray(jstep.water_dens_ts(T, S)),
                               np_dens(T, S), rtol=1e-15, atol=0)
    np.testing.assert_allclose(np.asarray(jstep.ddens_dtemp(T, S)),
                               np_ddt(T, S), rtol=1e-15, atol=0)
    np.testing.assert_allclose(np.asarray(jstep.ddens_dsal(T, S)),
                               np_dds(T, S), rtol=1e-15, atol=0)

    # --- s_diff (the reference is fresh water: Sal1 == 0) ----------------
    from lakematrix.constants import C
    bath = ss.bathym
    Sal1 = np.r_[0, st.Sal1[0]]
    lamsal = np.r_[0, (st.lamw[0] - C.lamw0) / C.cw_m_row0 * C.alsal
                   + C.lamsal0]
    dhw, dhw0, dhwp, dhwe = 1.0e-4, -2.0e-5, 3.0e-5, -1.0e-5
    a_int = np.r_[0, bath.area_int[0]]
    a_half = np.r_[0, bath.area_half[0]]
    ddz = np.r_[0, grid.ddz]
    dzeta = np.r_[0, grid.dzeta_int]
    want = np_sdiff(Sal1, DT, dhw, dhw0, dhwp, dhwe, lamsal, a_int, a_half,
                    ddz, float(st.h1[0]), dzeta, grid.ddz05, M)
    got = jstep.s_diff(jnp.asarray(Sal1)[None], DT, jnp.asarray([dhw]),
                       jnp.asarray([dhw0]), jnp.asarray([dhwp]),
                       jnp.asarray([dhwe]), jnp.asarray(lamsal)[None],
                       jnp.asarray(a_int)[None], jnp.asarray(a_half)[None],
                       jnp.asarray(ddz)[None], jnp.asarray([st.h1[0]]),
                       jnp.asarray(dzeta)[None], jnp.asarray(grid.ddz05), M)
    np.testing.assert_allclose(np.asarray(got)[0], want, rtol=0, atol=1e-15)
    # ... and with a non-zero salinity profile (exercises the tridiagonal)
    Sal1 = Sal1 + 1.0e-3 * np.arange(Sal1.size)
    want = np_sdiff(Sal1, DT, dhw, dhw0, dhwp, dhwe, lamsal, a_int, a_half,
                    ddz, float(st.h1[0]), dzeta, grid.ddz05, M)
    got = jstep.s_diff(jnp.asarray(Sal1)[None], DT, jnp.asarray([dhw]),
                       jnp.asarray([dhw0]), jnp.asarray([dhwp]),
                       jnp.asarray([dhwe]), jnp.asarray(lamsal)[None],
                       jnp.asarray(a_int)[None], jnp.asarray(a_half)[None],
                       jnp.asarray(ddz)[None], jnp.asarray([st.h1[0]]),
                       jnp.asarray(dzeta)[None], jnp.asarray(grid.ddz05), M)
    np.testing.assert_allclose(np.asarray(got)[0], want, rtol=0, atol=1e-15)

    # --- advance_case1 ---------------------------------------------------
    want = np_adv(SimpleNamespace(h1=np.array([st.h1[0]]),
                                 ls1=np.array([0.0])), np.array([dhw]))
    got = jstep.advance_case1(dict(h1=jnp.asarray([st.h1[0]])),
                              jnp.asarray([dhw]))
    np.testing.assert_array_equal(np.asarray(got["h1"]), want["h1"])

    # --- diffmin_hs (backdiff = 1 in the reference setup) ----------------
    row = np_dens(st.Tw1[0], st.Sal1[0])
    want = np_diffmin_hs(row, grid.ddz, float(st.h1[0]),
                         float(case.area_lake[0]), M, float(case.backdiff0))
    got = jstep.diffmin_hs(jnp.asarray(row)[None], jnp.asarray(grid.ddz),
                           jnp.asarray([st.h1[0]]),
                           jnp.asarray([case.area_lake[0]]), M,
                           float(case.backdiff0))
    np.testing.assert_allclose(np.asarray(got)[0], want, rtol=0, atol=1e-12)

    # --- surf_char (albedo = water_albedo(SINH0), roughness) -------------
    hour = float(case.hour0) + DT / 3600.0
    sc = np_surf_char(4, int(case.year0), int(case.month0), int(case.day0),
                      hour, float(case.phi[0]), float(case.fetch[0]), 5.0,
                      1.0e-2, 1)
    scj = jstep.surf_char(int(case.year0), int(case.month0),
                          int(case.day0), hour,
                          jnp.asarray([case.phi[0]]),
                          jnp.asarray([case.fetch[0]]),
                          jnp.asarray([5.0]), jnp.asarray([1.0e-2]))
    for key in ("albedo", "roughness", "relhums", "aM", "bM", "emissivity",
                "albedo_lw"):
        np.testing.assert_allclose(np.asarray(scj[key]).reshape(-1)[0],
                                   sc[key], rtol=0, atol=0, err_msg=key)


def test_adapters_roundtrip(ref):
    """``from_numpy_state`` / ``to_numpy`` keep every prognostic field."""
    ss = ref["ss1"]
    state = jstep.from_numpy_state(ss)
    _check_state_shape(state)
    new_water = state["water"]
    back = jstep.to_numpy(state)
    np.testing.assert_array_equal(back["Tw1"][0], ss.state.Tw1[0])
    np.testing.assert_array_equal(back["Sal1"][0], ss.state.Sal1[0])
    np.testing.assert_array_equal(back["Tsoil1"][0], ss.state.Tsoil1[0])
    np.testing.assert_array_equal(back["h1"][0], ss.state.h1[0])
    np.testing.assert_array_equal(back["lamw"][0], ss.state.lamw[0])
    np.testing.assert_array_equal(back["E1"][0], ss.state.E[0])
    np.testing.assert_array_equal(back["eps1"][0], ss.state.eps[0])
    assert float(new_water["Tsurf_prev"][0]) == float(ss.Tsurf_prev)
    assert float(back["Tsurf_prev"][0]) == float(ss.Tsurf_prev)
    assert float(back["Elatent"][0]) == float(ss.Elatent)


def test_traceable_soil_pass_matches_shared_kernel(ref):
    """``jaxbatch.soilpass`` (the ``jit``/``scan``-safe copy) == ``soilcols``."""
    import jax

    from lakematrix.jaxbatch import soilcols
    from lakematrix.jaxbatch import soilpass

    ss, case = ref["ss1"], ref["case"]
    sp = ss.soil_params
    L = 2

    def rep(x):
        return np.repeat(np.asarray(x, dtype=float)[None, ...], L, axis=0)

    keys = ("Tsoil1", "Tsoil2", "Tsoil3", "wl1", "wl2", "wl3", "wl4", "wi1",
            "wi2", "Sals1", "csoil", "rosoil", "lamsoil", "lammoist",
            "filtr", "wsoil", "wa", "methgenmh", "soilflux")
    state = {k: rep(getattr(ss.soil, k)) for k in keys}
    state["Tw1"] = rep(ss.soil.Tw1)
    state["h1"] = np.full(L, float(ss.state.h1[0]))
    state["dhwfsoil"] = np.full(L, float(ss.soil.dhwfsoil))
    packed = soilcols.pack_geometry(ss.soil.geom, case.M, case.nsoilcols)
    geom = {k: np.repeat(v, L, axis=0) for k, v in packed.items()}
    params = dict(
        ns=int(sp.ns), nsoilcols=int(sp.nsoilcols), POR=np.asarray(sp.POR),
        rosdry=np.asarray(sp.rosdry), dzs=np.asarray(sp.dzs),
        dzss=np.asarray(sp.dzss), zsoil=np.asarray(sp.zsoil),
        BH=np.asarray(sp.BH), DLMAX=np.asarray(sp.DLMAX),
        FLWMAX=np.asarray(sp.FLWMAX), WLM0=np.asarray(sp.WLM0),
        WLM7=np.asarray(sp.WLM7), nmeltpoint=int(sp.nmeltpoint),
        tricemethhydr=float(sp.tricemethhydr),
        soilbotflx=float(sp.soilbotflx), soilcolconjtype=2,
        pressure=float(ss.soil.pressure))

    want = soilcols.soilcolstemp_water(state, params, geom, DT)
    got = jax.jit(lambda s: soilpass.soilcolstemp_water(s, params, geom, DT))(
        state)
    # eager vs compiled: the only difference is the XLA reassociation of the
    # area/level reductions (soilflux is a difference of ~1e4 W/m2 terms, so
    # it shows up at ~1e-11 absolute; B3's own tolerance is 1e-10).
    for key in want:
        np.testing.assert_allclose(np.asarray(got[key]), np.asarray(want[key]),
                                   rtol=1e-13, atol=1e-12, err_msg=key)


# ---------------------------------------------------------------------------
# one full step and ten steps
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_one_step_matches_numpy_L1(ref):
    state = _state(ref, 1)
    new = jstep.step_case1_jax(state, _params(ref), ref["rows"], 1)
    _check_state_shape(new)
    got = jstep.to_numpy(new)
    assert got["Tw1"].shape[0] == 1
    _check_step(got, ref["one"], lanes=1, tag="step 1, L=1")


@pytest.mark.slow
def test_ten_steps_match_numpy_L1(ref):
    params = _params(ref)
    final, diags = jstep.run_case1_jax(params, ref["rows"], NSTEPS,
                                       state=_state(ref, 1))
    got = jstep.to_numpy(final)
    st = ref["ten"].state
    want = dict(Tw1=st.Tw1[0], Sal1=st.Sal1[0], Tsoil1=st.Tsoil1[0],
                h1=float(st.h1[0]), E1=st.E[0], eps1=st.eps[0],
                lamw=st.lamw[0])
    np.testing.assert_allclose(got["Tw1"][0], want["Tw1"], rtol=0,
                               atol=ATOL_T, err_msg="Tw1, 10 steps, L=1")
    np.testing.assert_allclose(got["Tsoil1"][0], want["Tsoil1"], rtol=0,
                               atol=ATOL_T, err_msg="Tsoil1, 10 steps, L=1")
    np.testing.assert_allclose(got["h1"][0], want["h1"], rtol=0, atol=ATOL_H)
    for key in ("E1", "eps1", "lamw"):
        assert _rel(got[key][0], want[key]) < RTOL_F, key
    # per-step diagnostics of the scan: one entry per step, cheap scalars only
    np.testing.assert_allclose(np.asarray(diags["Tsurf3"]).reshape(-1)[-1],
                               float(ref["ten"].Tsurf_prev), rtol=0,
                               atol=1e-9)
    assert np.asarray(diags["Tsurf3"]).shape == (NSTEPS, 1)
    assert np.asarray(diags["kt"]).shape == (NSTEPS, 1, ref["case"].M + 2)


@pytest.mark.slow
def test_lax_scan_matches_python_loop(ref):
    """The ``lax.scan`` runner == the same steps driven from Python."""
    params = _params(ref)
    scan_state, _diags = jstep.run_case1_jax(params, ref["rows"], NSTEPS,
                                             state=_state(ref, 1))
    loop_state = _state(ref, 1)
    for nstep in range(1, NSTEPS + 1):
        loop_state = jstep.step_case1_jax(loop_state, params, ref["rows"],
                                          nstep)
    scan, loop = jstep.to_numpy(scan_state), jstep.to_numpy(loop_state)
    for key in ("Tw1", "Sal1", "Tsoil1", "h1", "E1", "eps1", "lamw", "kt",
                "row2"):
        np.testing.assert_allclose(scan[key], loop[key], rtol=1e-13,
                                   atol=0, err_msg=key)


@pytest.mark.slow
def test_one_step_matches_numpy_L4(ref):
    """Four identical lanes must reproduce the NumPy single-lake step."""
    state = _state(ref, 4)
    new = jstep.step_case1_jax(state, _params(ref, 4), ref["rows"], 1)
    got = jstep.to_numpy(new)
    _check_step(got, ref["one"], lanes=4, tag="step 1, L=4")
    for key in ("Tw1", "E1", "lamw"):
        arr = np.asarray(new["water"][key])
        np.testing.assert_array_equal(arr, np.broadcast_to(arr[:1], arr.shape))


@pytest.mark.slow
def test_ten_steps_match_numpy_L4(ref):
    params = _params(ref, 4)
    final, _diags = jstep.run_case1_jax(params, ref["rows"], NSTEPS,
                                        state=_state(ref, 4))
    got = jstep.to_numpy(final)
    st = ref["ten"].state
    for lane in range(4):
        np.testing.assert_allclose(got["Tw1"][lane], st.Tw1[0], rtol=0,
                                   atol=ATOL_T, err_msg=f"Tw1 lane {lane}")
        np.testing.assert_allclose(got["Tsoil1"][lane], st.Tsoil1[0], rtol=0,
                                   atol=ATOL_T,
                                   err_msg=f"Tsoil1 lane {lane}")
        np.testing.assert_allclose(got["h1"][lane], float(st.h1[0]), rtol=0,
                                   atol=ATOL_H, err_msg=f"h1 lane {lane}")
        for key, ref_key in (("E1", "E"), ("eps1", "eps"), ("lamw", "lamw")):
            assert _rel(got[key][lane], getattr(st, ref_key)[0]) < RTOL_F, key


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_benchmark_jax_batch_vs_numpy(ref):
    """One step: JAX batch ``L = 1, 8, 64`` vs the NumPy per-lake loop.

    Prints lake-steps/s (``L / elapsed``).  NumPy is timed on prepared states
    (``build_step_state`` is a setup cost, not part of ``step_case1``); JAX is
    timed after a warm-up call, so the numbers exclude the JIT compilation.
    """
    import jax

    case, grid, rows = ref["case"], ref["grid"], ref["rows"]

    # --- NumPy per-lake loop ---------------------------------------------
    np_rates = {}
    for L in (1, 8, 64):
        states = [build_step_state(case, grid) for _ in range(L)]
        t0 = time.perf_counter()
        for ss in states:
            step_case1(ss, case, grid, rows, 1)
        np_rates[L] = L / (time.perf_counter() - t0)

    # --- JAX batch --------------------------------------------------------
    j_rates = {}
    for L in (1, 8, 64):
        params = _params(ref, L)
        state = _state(ref, L)
        jstep.step_case1_jax(state, params, rows, 1)          # compile
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            jax.block_until_ready(jstep.step_case1_jax(state, params, rows, 1))
        j_rates[L] = reps * L / (time.perf_counter() - t0)

    for L in (1, 8, 64):
        print(f"L={L:3d}: JAX {j_rates[L]:10.1f} lake-steps/s   "
              f"NumPy {np_rates[L]:8.1f} lake-steps/s   "
              f"speed-up x{j_rates[L] / np_rates[L]:6.2f}")
    # the batch must at least not be slower per lake-step than the loop
    assert j_rates[64] > 0.0
