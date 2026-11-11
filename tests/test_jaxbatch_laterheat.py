"""B3b: JAX batched LATERHEAT (water branch) matches the NumPy reference."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402


def _state():
    """A real case-1 state from ref/v1/summer60d with a non-zero soil flux."""
    from lakematrix.physics.soil import soilcolstemp

    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    # One NumPy soil column pass gives the reference soilflux (all-zero after
    # build_step_state); it is one call, not a simulation.
    soilcolstemp(ss.soil, ss.soil_params, 20.0)
    return case, grid, ss


def _inputs(ss, ncols, M):
    geom = ss.soil_geom
    return {
        "zsoilcols": np.asarray(geom.zsoilcols[1:])[None, :],    # (1, ncols+1)
        "Aint": np.asarray(ss.bathym.area_int[0])[None, :],      # (1, M+1)
        "Ahalf": np.asarray(ss.bathym.area_half[0])[None, :],    # (1, M)
        "ddz05": np.asarray(geom.ddz05),
        # bathymsoil(1:ncols+1)%area_int, aligned with zsoilcols
        "Asoil": np.array([[geom.bathymsoil[j].area_int
                            for j in range(1, ncols + 2)]]),
    }


def _rep(x, L):
    return np.repeat(np.asarray(x, dtype=float), L, axis=0)


def test_jax_laterheat_water_matches_numpy():
    from lakematrix.jaxbatch.laterheat import jitted_laterheat_water
    from lakematrix.physics.soil import laterheat

    case, _grid, ss = _state()
    st, sp = ss.state, ss.soil_params
    M, ncols = case.M, case.nsoilcols
    flux = np.array(ss.soil.soilflux, copy=True)          # 1-based, (ncols+1,)
    assert np.any(flux != 0.0)
    assert flux.shape == (ncols + 1,)
    h1 = float(st.h1[0])
    inp = _inputs(ss, ncols, M)
    ref = laterheat(ss.soil, sp, flux, onlywater=True).water
    assert ref.shape == (M + 2,)

    # --- L = 1 ------------------------------------------------------------
    out1 = jitted_laterheat_water(inp["zsoilcols"], np.array([h1]),
                                  inp["Aint"], inp["Ahalf"], inp["ddz05"],
                                  flux[None, :], M, ncols, inp["Asoil"])
    assert out1.shape == (1, M + 2)
    np.testing.assert_allclose(np.asarray(out1)[0], ref, rtol=0, atol=1e-12)

    # the 1-based padded Fortran forms are accepted as well
    out1p = jitted_laterheat_water(np.r_[0.0, inp["zsoilcols"][0]][None, :],
                                   np.array([h1]),
                                   np.r_[0.0, inp["Aint"][0]][None, :],
                                   np.r_[0.0, inp["Ahalf"][0]][None, :],
                                   inp["ddz05"], flux[None, :], M, ncols,
                                   np.r_[0.0, inp["Asoil"][0]][None, :])
    np.testing.assert_array_equal(np.asarray(out1p), np.asarray(out1))

    # --- L = 4: each lane has its own soil flux ---------------------------
    scale = np.array([1.0, 2.0, -0.5, 3.25])
    flux4 = scale[:, None] * flux[None, :]
    zsc4, Aint4, Ahalf4, Asoil4 = (_rep(inp["zsoilcols"], 4),
                                   _rep(inp["Aint"], 4),
                                   _rep(inp["Ahalf"], 4),
                                   _rep(inp["Asoil"], 4))
    h14 = np.full(4, h1)
    out4 = np.asarray(jitted_laterheat_water(
        zsc4, h14, Aint4, Ahalf4, inp["ddz05"], flux4, M, ncols, Asoil4))
    for lane in range(4):
        lane_ref = laterheat(ss.soil, sp, flux4[lane], onlywater=True).water
        np.testing.assert_allclose(out4[lane], lane_ref, rtol=0, atol=1e-12)
    # the lanes are independent (different fluxes -> different sources)
    assert not np.allclose(out4[0], out4[1])

    # --- dry lakes (h1 <= small) skip the water branch ---------------------
    h1_mixed = np.array([h1, 0.0, h1, 0.0])
    out_mixed = np.asarray(jitted_laterheat_water(
        zsc4, h1_mixed, Aint4, Ahalf4, inp["ddz05"], flux4, M, ncols, Asoil4))
    np.testing.assert_array_equal(out_mixed[1], np.zeros(M + 2))
    np.testing.assert_array_equal(out_mixed[3], np.zeros(M + 2))
    np.testing.assert_allclose(out_mixed[0], out4[0], rtol=0, atol=0)
    np.testing.assert_allclose(out_mixed[2], out4[2], rtol=0, atol=0)


def test_jax_laterheat_water_varied_depth():
    """Lanes with different water thicknesses: ``offset``, ``zsoilcols`` and
    the layer/column areas all change per lake."""
    from lakematrix.constants import C
    from lakematrix.jaxbatch.laterheat import laterheat_water
    from lakematrix.physics.bathym import bathymetry
    from lakematrix.physics.soil import laterheat, make_soil_state
    from lakematrix.state import init_state
    from lakematrix.step import _zsoil_geom

    case, grid, ss = _state()
    sp = ss.soil_params
    M, ncols = case.M, case.nsoilcols
    h1s = np.array([3.0, 7.5, 16.4, 22.66, 23.9])
    synth = np.array([0.0] + [(-1.0) ** k * 3.0 * (k + 1)
                              for k in range(1, ncols + 1)])

    lanes = []
    for lane, h1 in enumerate(h1s):
        st = init_state(case, grid)
        st.h1[:] = h1
        b = bathymetry(case, grid, st, C.pref)
        geom = _zsoil_geom(case, grid, 0, st, b)
        soil = make_soil_state(sp, geom, h1=h1, pressure=C.pref)
        flux = (1.0 + 0.37 * lane) * synth
        lanes.append({
            "h1": h1, "flux": flux,
            "ref": laterheat(soil, sp, flux, onlywater=True).water,
            "zsc": np.asarray(geom.zsoilcols[1:]),
            "Aint": np.asarray(b.area_int[0]),
            "Ahalf": np.asarray(b.area_half[0]),
            "Asoil": np.array([geom.bathymsoil[j].area_int
                               for j in range(1, ncols + 2)]),
        })

    out = np.asarray(laterheat_water(
        np.stack([ln["zsc"] for ln in lanes]),
        np.array([ln["h1"] for ln in lanes]),
        np.stack([ln["Aint"] for ln in lanes]),
        np.stack([ln["Ahalf"] for ln in lanes]),
        np.asarray(grid.ddz05),
        np.stack([ln["flux"] for ln in lanes]), M, ncols,
        np.stack([ln["Asoil"] for ln in lanes])))
    for lane, ln in enumerate(lanes):
        assert np.any(ln["ref"] != 0.0)
        np.testing.assert_allclose(out[lane], ln["ref"], rtol=0, atol=1e-12)
