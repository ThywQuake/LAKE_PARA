"""B6: batched JAX ``T_DIFF`` ice/snow port vs the NumPy reference.

Reference: :func:`lakematrix.physics.ice.t_diff_ice` on a real
``ref/v1/summer1d`` state extended with
:func:`lakematrix.step.build_ice_state` (``M = 40``, ``Mice = 10``,
``ns = 10``, ``ms = 100``, ``nsoilcols = 5``, ``dt = 20 s``) plus:

* a **case 2** state: ice over water, ``l1 = 0.5 m``, ``ls1 = 0.3 m``, with
  and without snow and with/without deep ice on the ``surf == 0`` path;
* a **case 3** synthetic state: ``h1 = 0`` (no water), ``l1 > 0``, so the
  ``water == 0`` branches of ``T_DIFF`` run (ice/snow over soil, and the
  no-soil ``soilswitch == 0`` variant).

Every case is checked with ``L = 1`` and ``L = 4``; the four lanes differ in
temperatures, conductivities, layer thicknesses, snow depth (``itop``) and
switches, so the per-lane ``jnp.where`` selection is exercised (including a
mixed ``water``/``deepice``/``soilswitch`` batch).

The NumPy ``IceState`` fields that :func:`build_ice_state` leaves at zero are
filled from :class:`~lakematrix.step.StepState` (the soil column: ``Tsoil1``,
``lamsoil``, ``csoil``, ``rosoil``, ``dzs``), otherwise the ice-soil
interfaces of ``T_DIFF`` divide by zero.

Tolerance: ``atol = 1e-10``, ``rtol = 0`` (measured: <= 1e-15, i.e. double
round-off; most entries are bit-identical).
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.ice import sr_botsnow  # noqa: E402
from lakematrix.physics.ice import snow_cond_heat_coef  # noqa: E402
from lakematrix.physics.ice import t_diff_ice as np_t_diff_ice  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_ice_state, build_step_state  # noqa: E402

CASE = ROOT / "ref" / "v1" / "summer1d"
DT = 20.0
ATOL = 1e-10

#: lane factors for the synthetic per-lake variation (temperatures,
#: conductivities, thicknesses, snow pack, bathymetry)
LANE_SCALE = np.array([1.0, 0.9, 1.15, 1.05])

#: per-lane snow bottom node (``ms - 2`` is what SNOW_COND_HEAT_COEF builds)
_LANE_ITOP = np.array([-2, -3, -2, -4])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_CACHE = {}


def _setup():
    """``ref/v1/summer1d`` StepState + an ice state with a populated column."""
    if "setup" in _CACHE:
        return _CACHE["setup"]
    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    st = build_ice_state(ss, case, grid)

    # build_ice_state only copies the ice/water fields; the soil column of the
    # IceState stays zero (make_ice_state), which would make the ice-soil
    # interface divide by zero.
    st.Tsoil1 = np.array(ss.soil.Tsoil1, copy=True)
    st.Tsoil2 = np.array(ss.soil.Tsoil2, copy=True)
    st.lamsoil = np.array(ss.soil.lamsoil, copy=True)
    st.csoil = np.array(ss.soil.csoil, copy=True)
    st.rosoil = np.array(ss.soil.rosoil, copy=True)
    st.dzs = np.array(ss.soil_params.dzs, copy=True)

    # case-2 ice column: l1 / Ti1 / Tis1 (+ the increments T_DIFF reads)
    st.l1 = 0.5
    st.ls1 = 0.3
    st.Ti1[1:] = np.linspace(-5.0, 0.0, case.Mice + 1)
    st.Ti2[1:] = st.Ti1[1:]
    st.Tis1[1:] = np.linspace(-2.0, 0.0, case.Mice + 1)
    st.Tis2[1:] = st.Tis1[1:]
    st.dhi, st.dhi0 = 0.010, 0.008
    st.dls, st.dls0 = 0.004, 0.003
    st.dhw, st.dhw0 = 0.020, 0.010
    st.snmeltice, st.dhiimp = 1.0e-3, 5.0e-4
    st.h1 = 22.66

    st.Sal2[1:] = 2.0e-3
    st.preswat[1:] = 2.0e5
    st.PEMF[1:] = 1.0e-6
    st.pt_down_f[1:] = 0.5
    st.rad_integr_ice[:] = np.linspace(0.0, -5.0, case.Mice + 1)
    st.rad_integr_water[:] = np.linspace(0.0, -3.0, case.M + 1)
    st.rad_integr_deepice[:] = np.linspace(0.0, -1.0, case.Mice + 1)
    st.lsh_ice[1:] = 0.05
    st.lsh_dice[1:] = 0.01
    st.lsh_water[1:] = 0.02
    st.SR_botsnow = 3.0
    st.soilbotflx = 0.1

    _CACHE["setup"] = (case, grid, ss, st)
    return case, grid, ss, st


def _snow_state(st):
    """Copy of ``st`` with the initial snow pack built (``itop = ms-2``)."""
    snow = copy.deepcopy(st)
    snow.flag_snow = 1
    snow_cond_heat_coef(snow, -5.0)
    sr_botsnow(snow, 20.0)
    return snow


def _make_lanes(st, L, itops=None):
    """``L`` per-lane IceState copies with the same synthetic variation."""
    lanes = []
    for lane in range(L):
        li = copy.deepcopy(st)
        s = float(LANE_SCALE[lane])
        for name in ("Ti1", "Ti2", "Tis1", "Tis2", "Tw1", "Tw2", "lamw",
                     "ci_m_roi_v", "lami_v", "PEMF", "pt_down_f", "Sal2",
                     "preswat", "rad_integr_ice", "rad_integr_water",
                     "rad_integr_deepice", "lsh_ice", "lsh_dice",
                     "lsh_water", "Tsoil1", "Tsoil2", "lamsoil", "csoil",
                     "rosoil", "dzs", "T", "Tsn", "cs", "dens", "dz",
                     "lams"):
            setattr(li, name, np.asarray(getattr(li, name)) * s)
        for name in ("h1", "l1", "ls1", "dhi", "dhi0", "dls", "dls0", "dhw",
                     "dhw0", "snmeltice", "dhiimp", "soilbotflx",
                     "SR_botsnow"):
            setattr(li, name, float(getattr(li, name)) * s)
        for name in ("bathymice", "bathymwater", "bathymdice"):
            levels = getattr(li, name)
            for b in levels:
                b.area_int *= s
                b.area_half *= s
        if itops is not None and int(itops[lane]) != li.itop:
            _extend_snow(li, int(itops[lane]))
        lanes.append(li)
    return lanes


def _extend_snow(st, itop):
    """Grow the snow pack upwards (for the per-lane ``itop`` test)."""
    for i in range(itop, st.itop):
        st.T[i] = st.T[st.itop]
        st.dens[i] = st.dens[st.itop]
        st.dz[i] = st.dz[st.itop]
    st.itop = itop
    st.flag_snow_init = 0
    snow_cond_heat_coef(st, -5.0)


# ---------------------------------------------------------------------------
# packing a list of IceStates into the batched state dict
# ---------------------------------------------------------------------------

def _bathym(lanes, name, field):
    return np.stack([np.array([getattr(b, field) for b in getattr(li, name)])
                     for li in lanes])


def _pack(lanes, case, grid):
    """Build the ``jaxbatch.ice_tdiff`` state dict from per-lane IceStates."""
    st0 = lanes[0]
    M, Mice, ns = st0.M, st0.Mice, st0.ns
    d = {
        # grid metrics: bare (n,) arrays (documented "lake axis optional")
        "ddz": np.asarray(st0.ddz),
        "ddz05": np.asarray(st0.ddz05),
        "dzeta_int": np.asarray(st0.dzeta_int),
        "ddzi": np.asarray(st0.ddzi),
        "ddzi05": np.asarray(st0.ddzi05),
        "dzetai_int": np.asarray(st0.dzetai_int),
        "Aice": _bathym(lanes, "bathymice", "area_int"),
        "Ahice": _bathym(lanes, "bathymice", "area_half"),
        "Awat": _bathym(lanes, "bathymwater", "area_int"),
        "Ahwat": _bathym(lanes, "bathymwater", "area_half"),
        "Adice": _bathym(lanes, "bathymdice", "area_int"),
        "Ahdice": _bathym(lanes, "bathymdice", "area_half"),
        "itop": np.array([li.itop for li in lanes], dtype=int),
        "nmeltpoint": st0.nmeltpoint,
        "soilswitch": np.array([li.soilswitch for li in lanes], dtype=int),
    }
    for name in ("Ti1", "Ti2", "ci_m_roi_v", "lami_v", "rad_integr_ice",
                 "lsh_ice", "Tis1", "Tis2", "rad_integr_deepice",
                 "lsh_dice", "Tw1", "Tw2", "lamw", "PEMF", "pt_down_f",
                 "Sal2", "preswat", "rad_integr_water", "lsh_water", "T",
                 "Tsn", "cs", "dens", "dz", "lams", "Tsoil1", "Tsoil2",
                 "lamsoil", "csoil", "rosoil", "dzs"):
        d[name] = np.stack([np.asarray(getattr(li, name)) for li in lanes])
    for name in ("h1", "l1", "ls1", "dhi", "dhi0", "dls", "dls0", "dhw",
                 "dhw0", "snmeltice", "dhiimp", "soilbotflx", "SR_botsnow"):
        d[name] = np.array([float(getattr(li, name)) for li in lanes])
    return d


# ---------------------------------------------------------------------------
# comparison driver
# ---------------------------------------------------------------------------

def _ref_lanes(lanes, surf, Tsurf, snow, water, deepice, isoilcol):
    """NumPy ``t_diff_ice`` per lane (``ice = 1``)."""
    out = []
    for lane, li in enumerate(lanes):
        out.append(np_t_diff_ice(
            _at(surf, lane), float(np.atleast_1d(Tsurf)[lane]
                                   if np.ndim(Tsurf) else Tsurf), DT,
            _at(snow, lane), 1, _at(water, lane), _at(deepice, lane),
            isoilcol, False, copy.deepcopy(li)))
    return out


def _at(x, lane):
    a = np.asarray(x)
    return int(a) if a.ndim == 0 else int(a.reshape(-1)[lane])


def _check(state, lanes, surf, Tsurf, snow, water, deepice, isoilcol,
           keys=("Ti2", "Tw2", "Tis2", "Tsoil2", "Tsn")):
    from lakematrix.jaxbatch.ice_tdiff import t_diff_ice

    before = {k: np.array(np.asarray(state[k]), copy=True) for k in keys}
    out = t_diff_ice(surf, Tsurf, DT, snow, 1, water, deepice, isoilcol,
                     False, state)
    refs = _ref_lanes(lanes, surf, Tsurf, snow, water, deepice, isoilcol)
    for key in keys:
        ref = np.stack([np.asarray(getattr(r, key)) for r in refs])
        got = np.asarray(out[key])
        assert got.shape == ref.shape, (key, got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, rtol=0, atol=ATOL,
                                   err_msg=f"{key}: JAX vs NumPy T_DIFF")
    # the input dict and its arrays are not mutated
    for key in keys:
        np.testing.assert_array_equal(np.asarray(state[key]), before[key],
                                      err_msg=f"input state[{key!r}] mutated")
    assert out is not state
    return out, refs


def _Tsurf(L):
    return -2.0 - np.arange(L, dtype=float)


# ---------------------------------------------------------------------------
# case 2, surf == 1 (T_SOLVER scan): ice over water, with and without snow
# ---------------------------------------------------------------------------

def test_case2_no_snow_surf1():
    case, grid, _ss, st = _setup()
    for L in (1, 4):
        lanes = _make_lanes(st, L)
        state = _pack(lanes, case, grid)
        out, refs = _check(state, lanes, 1, _Tsurf(L), 0, 1, 0,
                           case.nsoilcols)
        # ice and water: the ice column is updated, the water column is not
        assert np.all(np.asarray(out["Ti2"])[:, 1:] < 0.0)
        np.testing.assert_array_equal(np.asarray(out["Tw2"]),
                                      np.asarray(state["Tw2"]))
        # Tsoil2 is untouched by this branch
        np.testing.assert_array_equal(np.asarray(out["Tsoil2"]),
                                      np.asarray(state["Tsoil2"]))


def test_case2_snow_surf1():
    case, grid, _ss, st = _setup()
    snow_st = _snow_state(st)
    for L in (1, 4):
        itops = None
        if L > 1:
            itops = snow_st.ms + _LANE_ITOP[:L]
        lanes = _make_lanes(snow_st, L, itops=itops)
        state = _pack(lanes, case, grid)
        out, _ = _check(state, lanes, 1, _Tsurf(L), 1, 1, 0,
                        case.nsoilcols)
        # the snow solution is non-trivial and rows below itop keep the input
        for lane, li in enumerate(lanes):
            got = np.asarray(out["Tsn"])[lane]
            assert np.all(np.isfinite(got))
            np.testing.assert_array_equal(got[:li.itop],
                                          np.asarray(state["Tsn"])[lane,
                                                                  :li.itop])
            assert not np.allclose(got[li.itop:li.ms + 1],
                                   np.asarray(state["Tsn"])[lane,
                                                           li.itop:li.ms + 1])


# ---------------------------------------------------------------------------
# case 2, surf == 0 (end of T_SOLVER): water + deep ice, with and without soil
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("soilswitch", [1, 0])
def test_case2_surf0_deepice_mixed(soilswitch):
    case, grid, _ss, st = _setup()
    L = 4
    lanes = _make_lanes(st, L)
    for li in lanes:
        li.soilswitch = soilswitch
    state = _pack(lanes, case, grid)
    deepice = np.array([0, 1, 0, 1])
    out, _ = _check(state, lanes, 0, _Tsurf(L), 0, 1, deepice,
                    case.nsoilcols)
    got_tis = np.asarray(out["Tis2"])
    src_tis = np.asarray(state["Tis2"])
    # deep-ice lanes solve Tis2, the others echo the input
    for lane, flag in enumerate(deepice):
        if flag == 0:
            np.testing.assert_array_equal(got_tis[lane], src_tis[lane])
        else:
            assert not np.allclose(got_tis[lane, 1:], src_tis[lane, 1:])
    # the water column is solved on every lane
    assert not np.allclose(np.asarray(out["Tw2"]), np.asarray(state["Tw2"]))
    if soilswitch == 0:
        np.testing.assert_array_equal(np.asarray(out["Tsoil2"]),
                                      np.asarray(state["Tsoil1"])[:, :,
                                                                  case.nsoilcols])


# ---------------------------------------------------------------------------
# case 3 (water == 0): ice (and snow) over soil, h1 = 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snow", [0, 1])
@pytest.mark.parametrize("soilswitch", [1, 0])
def test_case3_ice_over_soil(snow, soilswitch):
    case, grid, _ss, st = _setup()
    base = copy.deepcopy(st)
    base.h1 = 0.0                      # no water above the ice
    base.soilswitch = soilswitch
    if snow:
        base = _snow_state(base)
    for L in (1, 4):
        itops = None
        if snow and L > 1:
            itops = base.ms + _LANE_ITOP[:L]
        lanes = _make_lanes(base, L, itops=itops)
        state = _pack(lanes, case, grid)
        out, _ = _check(state, lanes, 1, _Tsurf(L), snow, 0, 0,
                        case.nsoilcols)
        assert np.all(np.isfinite(np.asarray(out["Ti2"])))
        if soilswitch == 0:
            # Tsoil2 = Tsoil1(:, isoilcol) in the ice-only branch
            np.testing.assert_array_equal(
                np.asarray(out["Tsoil2"]),
                np.asarray(state["Tsoil1"])[:, :, case.nsoilcols])
        else:
            assert not np.allclose(np.asarray(out["Tsoil2"]),
                                   np.asarray(state["Tsoil2"]))


def test_surf0_water0_returns_state_unchanged():
    case, grid, _ss, st = _setup()
    base = copy.deepcopy(st)
    base.h1 = 0.0
    L = 4
    lanes = _make_lanes(base, L)
    state = _pack(lanes, case, grid)
    from lakematrix.jaxbatch.ice_tdiff import t_diff_ice

    out = t_diff_ice(0, _Tsurf(L), DT, 0, 1, 0, 0, case.nsoilcols, False,
                     state)
    for key in ("Ti2", "Tw2", "Tis2", "Tsoil2", "Tsn"):
        np.testing.assert_array_equal(np.asarray(out[key]),
                                      np.asarray(state[key]))


# ---------------------------------------------------------------------------
# jit: same numerics, and flags as traced arrays
# ---------------------------------------------------------------------------

def test_jit_and_dynamic_flags_match():
    import jax

    case, grid, _ss, st = _setup()
    L = 4
    lanes = _make_lanes(st, L)
    state = _pack(lanes, case, grid)
    from lakematrix.jaxbatch.ice_tdiff import t_diff_ice

    # static flags: only the no-snow system is built
    jitted = jax.jit(t_diff_ice, static_argnames=("isoilcol", "ifflux"))
    eager = t_diff_ice(1, _Tsurf(L), DT, 0, 1, 1, 0, case.nsoilcols, False,
                       state)
    fast = jitted(1, _Tsurf(L), DT, 0, 1, 1, 0, case.nsoilcols, False, state)
    for key in ("Ti2", "Tw2", "Tis2", "Tsoil2", "Tsn"):
        np.testing.assert_allclose(np.asarray(fast[key]),
                                   np.asarray(eager[key]),
                                   rtol=0, atol=1e-10)

    # traced flags: every system is built and selected with jnp.where
    jitted_dyn = jax.jit(
        lambda sf, ts, sn, wt, di, s: t_diff_ice(
            sf, ts, DT, sn, 1, wt, di, case.nsoilcols, False, s))
    water = np.array([1, 0, 1, 0])
    deepice = np.array([0, 0, 1, 0])
    surf = np.array([1, 1, 0, 0])
    dynamic = jitted_dyn(surf, _Tsurf(L), np.zeros(L, dtype=int), water,
                         deepice, state)
    refs = []
    for lane, li in enumerate(lanes):
        refs.append(np_t_diff_ice(
            int(surf[lane]), float(_Tsurf(L)[lane]), DT, 0, 1,
            int(water[lane]), int(deepice[lane]), case.nsoilcols, False,
            copy.deepcopy(li)))
    for key in ("Ti2", "Tw2", "Tis2", "Tsoil2", "Tsn"):
        ref = np.stack([np.asarray(getattr(r, key)) for r in refs])
        np.testing.assert_allclose(np.asarray(dynamic[key]), ref,
                                   rtol=0, atol=1e-10)
