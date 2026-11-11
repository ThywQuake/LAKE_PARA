"""B3: JAX batched open-water side soil columns (``SOILCOLSTEMP``) vs NumPy.

The batched kernel is exercised on the real ``summer60d`` state.  Both regular
tests start from a pristine ``build_step_state`` state; the four-lane test gives
every lane its own water depth, ``TSURFSOILCOL`` geometry (``zsoilcols``
recomputed by :func:`lakematrix.grid.zsoilcols` for that depth, which is what
``BATHYMETRY`` does per lake) and its own water/soil temperature offset, so the
lanes really do differ and each one is compared with its own NumPy
``soilcolstemp`` call.  ``tw1`` offsets only touch ``Tw1[1:]`` (index 0 is the
Fortran pad).
"""

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid, zsoilcols  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402

DT = 20.0
ATOL = 1e-12          # temperatures, moisture contents, soil coefficients
# soilflux is a W/m2 difference of two ~1e4 W/m2 terms: the PROGONKA/JAX
# reduction order leaves ~1e-15 relative (measured 8.2e-12 absolute here).
ATOL_FLUX = 1e-10


def _case():
    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    return case, grid, build_step_state(case, grid)


def _params(ss):
    sp = ss.soil_params
    return dict(
        ns=int(sp.ns), nsoilcols=int(sp.nsoilcols),
        POR=np.asarray(sp.POR), rosdry=np.asarray(sp.rosdry),
        dzs=np.asarray(sp.dzs), dzss=np.asarray(sp.dzss),
        zsoil=np.asarray(sp.zsoil),
        BH=np.asarray(sp.BH), DLMAX=np.asarray(sp.DLMAX),
        FLWMAX=np.asarray(sp.FLWMAX),
        WLM0=np.asarray(sp.WLM0), WLM7=np.asarray(sp.WLM7),
        nmeltpoint=int(sp.nmeltpoint),
        tricemethhydr=float(sp.tricemethhydr),
        soilbotflx=float(sp.soilbotflx),
        soilcolconjtype=int(sp.soilcolconjtype),
        pressure=float(ss.soil.pressure),
    )


def _lane_soil(case, grid, ss, h1, dTw, dT, dSal=0.0):
    """Copy of the NumPy soil state for one lane (own ``h1``/``zsoilcols``)."""
    soil = copy.deepcopy(ss.soil)
    soil.h1 = float(h1)
    Tw1 = np.array(ss.soil.Tw1, copy=True)
    Tw1[1:] = Tw1[1:] + dTw
    soil.Tw1 = Tw1
    Tsoil1 = np.array(ss.soil.Tsoil1, copy=True)
    Tsoil1[1:, 1:] = Tsoil1[1:, 1:] + dT
    soil.Tsoil1 = Tsoil1
    if dSal:
        soil.Sals1[1:, 1:] = ss.soil.Sals1[1:, 1:] + dSal
    geom = copy.deepcopy(ss.soil_geom)
    table = case.morphometry[0, :int(case.morphometry_len[0]), :]
    geom.zsoilcols = np.r_[0.0, zsoilcols(soil.h1, table, grid,
                                         case.nsoilcols)]
    soil.geom = geom
    return soil


def _batch_state(soils):
    """One state dict with a leading lake axis, one entry per lane."""
    def stack(attr):
        return np.stack([np.asarray(getattr(s, attr), dtype=float)
                         for s in soils], axis=0)

    keys2d = ("Tsoil1", "Tsoil3", "wl1", "wl4", "wi1", "wi2", "Sals1")
    keys1d = ("Tsoil2", "wl2", "wl3", "wa", "methgenmh", "csoil", "rosoil",
              "lamsoil", "lammoist", "filtr", "wsoil")
    out = {k: stack(k) for k in keys2d + keys1d}
    out["Tw1"] = stack("Tw1")
    out["h1"] = np.array([s.h1 for s in soils], dtype=float)
    out["soilflux"] = stack("soilflux")
    out["dhwfsoil"] = np.array([s.dhwfsoil for s in soils], dtype=float)
    return out


def _batch_geom(soil_geoms, M, ncols):
    from lakematrix.jaxbatch.soilcols import pack_geometry

    packed = [pack_geometry(g, M, ncols) for g in soil_geoms]
    return {k: np.concatenate([p[k] for p in packed], axis=0) for k in packed[0]}


def _check(case, grid, ss, lanes):
    from lakematrix.jaxbatch.soilcols import soilcolstemp_water
    from lakematrix.physics.soil import soilcolstemp, tsurfsoilcol

    sp = ss.soil_params
    M, ncols = case.M, case.nsoilcols
    soils = [_lane_soil(case, grid, ss, *lane) for lane in lanes]
    refs = [soilcolstemp(copy.deepcopy(s), sp, DT) for s in soils]
    tsurf = [tsurfsoilcol(s, sp) for s in soils]

    state = _batch_state(soils)
    geom = _batch_geom([s.geom for s in soils], M, ncols)
    out = soilcolstemp_water(state, _params(ss), geom, DT)

    def cmp(key, ref, atol=ATOL, tag=None):
        got = np.asarray(out[key], dtype=float)
        assert got.shape == (len(soils),) + np.shape(ref), (key, got.shape,
                                                            np.shape(ref))
        np.testing.assert_allclose(got[0], np.asarray(ref, dtype=float),
                                   rtol=0, atol=atol, err_msg=tag or key)

    for lane, (soil, ref, ts) in enumerate(zip(soils, refs, tsurf)):
        tag = f"lane {lane}"
        for key in ("Tsoil1", "wl1", "wi1", "Tsoil3", "wl4", "wi2", "Sals1"):
            np.testing.assert_allclose(
                np.asarray(out[key])[lane], np.asarray(getattr(ref, key)),
                rtol=0, atol=ATOL, err_msg=f"{key}, {tag}")
        for key in ("Tsoil2", "wl2", "wl3", "wa", "methgenmh", "csoil",
                    "rosoil", "lamsoil", "lammoist", "filtr", "wsoil"):
            np.testing.assert_allclose(
                np.asarray(out[key])[lane], np.asarray(getattr(ref, key)),
                rtol=0, atol=ATOL, err_msg=f"{key}, {tag}")
        np.testing.assert_allclose(
            np.asarray(out["soilflux"])[lane], np.asarray(ref.soilflux),
            rtol=0, atol=ATOL_FLUX, err_msg=f"soilflux, {tag}")
        np.testing.assert_allclose(
            np.asarray(out["dhwfsoil"])[lane], ref.dhwfsoil,
            rtol=0, atol=ATOL, err_msg=f"dhwfsoil, {tag}")
        np.testing.assert_allclose(
            np.asarray(out["Tsoilsurf"])[lane], ts, rtol=0, atol=ATOL,
            err_msg=f"Tsoilsurf, {tag}")
        # The deepest column is not touched here (lake.f90 handles it later).
        np.testing.assert_array_equal(
            np.asarray(out["soilflux"])[lane, 0], 0.0)
        np.testing.assert_array_equal(
            np.asarray(out["soilflux"])[lane, ncols], 0.0)
        np.testing.assert_array_equal(
            np.asarray(out["Tsoil1"])[lane][:, 0],
            np.asarray(soil.Tsoil1)[:, 0])
    # shapes of the batch outputs
    assert np.asarray(out["Tsoil1"]).shape == (len(lanes), case.ns + 1,
                                               ncols + 1)
    assert np.asarray(out["soilflux"]).shape == (len(lanes), ncols + 1)
    assert np.asarray(out["Tsoilsurf"]).shape == (len(lanes), ncols + 1)
    return out, refs


def test_jax_soilcolstemp_water_L1():
    case, grid, ss = _case()
    _check(case, grid, ss, [(float(ss.state.h1[0]), 0.0, 0.0)])


def test_jax_soilcolstemp_water_L4():
    case, grid, ss = _case()
    h1 = float(ss.state.h1[0])
    # Different (valid) water depths -> different zsoilcols partitions around
    # the side-column intervals; different Tw1/Tsoil1 -> different per-lane
    # physics.  The reference needs the water to flood all side columns
    # (``zsoilcols(2) <= z(0)``), which for this morphometry means
    # ``h1 > ~18.06 m``; shallower depths would hit TSURFSOILCOL's unset-j1
    # RuntimeError in the NumPy reference.
    lanes = [
        (h1, 0.0, 0.0),
        (20.0, 0.5, 0.3),
        (18.5, -0.75, -0.4),
        (h1 - 0.25, 1.0, 0.8),
    ]
    out, _refs = _check(case, grid, ss, lanes)
    arr = np.asarray(out["Tsoil1"])
    for lane in range(1, len(lanes)):
        assert not np.array_equal(arr[lane], arr[0])


def test_jax_tsurfsoilcol_water_matches_numpy():
    from lakematrix.jaxbatch.soilcols import pack_geometry, tsurfsoilcol_water
    from lakematrix.physics.soil import tsurfsoilcol

    case, grid, ss = _case()
    M, ncols = case.M, case.nsoilcols
    h1s = [float(ss.state.h1[0]), 20.0, 19.0, 22.0]
    soils = [_lane_soil(case, grid, ss, h, 0.25 * k, -0.1 * k)
             for k, h in enumerate(h1s)]
    geom = _batch_geom([s.geom for s in soils], M, ncols)
    Tw1 = np.stack([s.Tw1 for s in soils], axis=0)
    h1 = np.array([s.h1 for s in soils])
    got = tsurfsoilcol_water(
        Tw1, h1, geom["zsoilcols"], geom["ddz"], geom["w_lx"], geom["w_ly"],
        geom["w_lx_half"], geom["w_ly_half"], geom["s_lx"], geom["s_ly"],
        M, ncols)
    assert np.asarray(got).shape == (len(soils), ncols + 1)
    for lane, soil in enumerate(soils):
        want = tsurfsoilcol(soil, ss.soil_params)
        np.testing.assert_allclose(np.asarray(got)[lane], want, rtol=0,
                                   atol=1e-12, err_msg=f"lane {lane}")
    # 1-D geometry arguments are broadcast over the lake axis as well.
    one = pack_geometry(soils[0].geom, M, ncols)
    got1 = tsurfsoilcol_water(
        Tw1[:1], h1[:1], one["zsoilcols"][0], one["ddz"][0],
        one["w_lx"][0], one["w_ly"][0], one["w_lx_half"][0],
        one["w_ly_half"][0], one["s_lx"][0], one["s_ly"][0], M, ncols)
    np.testing.assert_array_equal(np.asarray(got1), np.asarray(got)[:1])


def test_jax_jitted_subkernels_match_numpy():
    """``jitted_tsurfsoilcol_water`` / ``jitted_t_diff_soil`` vs NumPy.

    The whole ``soilcolstemp_water`` pass cannot be jitted yet (see the module
    docstring: ``SOIL_COND_HEAT_COEF`` in ``jaxbatch/soil.py`` is eager-only),
    but both pure sub-kernels are.
    """
    from lakematrix.jaxbatch.soilcols import (jitted_t_diff_soil,
                                              jitted_tsurfsoilcol_water,
                                              tsurfsoilcol_water)
    from lakematrix.physics.soil import (_t_diff_soil, soil_cond_heat_coef,
                                        tsurfsoilcol)

    case, grid, ss = _case()
    sp = ss.soil_params
    M, ncols, ns = case.M, case.nsoilcols, case.ns
    h1 = float(ss.state.h1[0])
    lanes = [(h1, 0.0, 0.0), (20.0, 0.5, 0.3), (18.5, -0.75, -0.4)]
    soils = [_lane_soil(case, grid, ss, *lane) for lane in lanes]
    geom = _batch_geom([s.geom for s in soils], M, ncols)

    # --- TSURFSOILCOL: jitted == eager == NumPy ---------------------------
    Tw1 = np.stack([s.Tw1 for s in soils], axis=0)
    h1s = np.array([s.h1 for s in soils])
    args = (Tw1, h1s, geom["zsoilcols"], geom["ddz"], geom["w_lx"],
            geom["w_ly"], geom["w_lx_half"], geom["w_ly_half"], geom["s_lx"],
            geom["s_ly"], M, ncols)
    eager = np.asarray(tsurfsoilcol_water(*args))
    jitted = np.asarray(jitted_tsurfsoilcol_water(*args))
    # XLA may reassociate the masked area reductions: ~1e-15, not bit-equal.
    np.testing.assert_allclose(jitted, eager, rtol=0, atol=ATOL)
    for lane, soil in enumerate(soils):
        np.testing.assert_allclose(jitted[lane], tsurfsoilcol(soil, sp),
                                   rtol=0, atol=ATOL,
                                   err_msg=f"Tsoilsurf, lane {lane}")

    # --- T_DIFF (side column is = 1): jitted == NumPy ---------------------
    coef = []
    T2 = []
    for lane, soil in enumerate(soils):
        st = copy.deepcopy(soil)
        soil_cond_heat_coef(st, sp, is_=1)
        coef.append((st.csoil, st.rosoil, st.lamsoil))
        T2.append(np.asarray(_t_diff_soil(copy.deepcopy(st), sp, eager[lane, 1],
                                          1, DT)))
    got = jitted_t_diff_soil(
        np.stack([np.asarray(s.Tsoil1)[:, 1] for s in soils]),
        np.stack([c[2] for c in coef]), np.stack([c[0] for c in coef]),
        np.stack([c[1] for c in coef]), np.asarray(sp.dzs), eager[:, 1], DT,
        float(sp.soilbotflx), ns)
    np.testing.assert_allclose(np.asarray(got), np.stack(T2), rtol=0,
                               atol=ATOL, err_msg="Tsoil2 (t_diff_soil)")


def test_jax_soilcolstemp_water_preserves_deep_column_entries():
    """Entries 0 and ``ncols`` of ``soilflux`` are not touched (deepest column
    is handled after ``T_SOLVER`` by the driver, as in ``lake.f90``)."""
    from lakematrix.jaxbatch.soilcols import soilcolstemp_water
    from lakematrix.physics.soil import soilcolstemp

    case, grid, ss = _case()
    ncols = case.nsoilcols
    soil = _lane_soil(case, grid, ss, float(ss.state.h1[0]), 0.0, 0.0)
    ref = soilcolstemp(copy.deepcopy(soil), ss.soil_params, DT)
    state = _batch_state([soil])
    state["soilflux"] = np.array([[1.5] + [0.0] * (ncols - 1) + [2.5]])
    geom = _batch_geom([soil.geom], case.M, ncols)
    out = soilcolstemp_water(state, _params(ss), geom, DT)
    flux = np.asarray(out["soilflux"])
    assert flux[0, 0] == 1.5
    assert flux[0, ncols] == 2.5
    np.testing.assert_allclose(flux[0, 1:ncols], ref.soilflux[1:ncols],
                               rtol=0, atol=ATOL_FLUX)
