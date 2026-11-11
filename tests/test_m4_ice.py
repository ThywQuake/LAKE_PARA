"""M4 smoke test: ice state + T_DIFF ice branch."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_ice_state, build_step_state  # noqa: E402


def test_ice_tdiff_smoke():
    case = read_case(ROOT / "ref" / "v1" / "summer1d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    st = build_ice_state(ss, case, grid)
    st.l1 = 0.5
    st.h1 = 22.16
    st.Ti1[1:] = np.linspace(-5.0, 0.0, case.Mice + 1)
    st.Ti2[1:] = st.Ti1[1:]
    from lakematrix.physics.ice import t_diff_ice
    st = t_diff_ice(1, -2.0, 20.0, 0, 1, 1, 0, case.nsoilcols, False, st)
    assert np.all(np.isfinite(st.Ti2))
    assert np.all(np.isfinite(st.Tw2))
    assert st.Ti2[1] > -10.0


def test_ice_surface_solver_smoke():
    import numpy as np
    from lakematrix.physics.surface import IceSurfaceContext, heatbalance_ice, surf_char
    from lakematrix.physics.tsolver import t_solver

    case = read_case(ROOT / "ref" / "v1" / "summer1d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    ice = build_ice_state(ss, case, grid)
    ice.l1 = 0.5
    ice.h1 = 22.16
    ice.Ti1[1:] = np.linspace(-5.0, 0.0, case.Mice + 1)
    ice.Ti2[1:] = ice.Ti1[1:]
    sc = surf_char(2, 2016, 8, 1, 4.0, 56.0, 1000.0, 1.0, 0.01)
    ctx = IceSurfaceContext(
        ice=ice, snow=0, water=1, dt=20.0, fetch=1000.0, uwind=1.0,
        vwind=0.0, u=0.0, v=0.0, tempair=15.0, pressure=100000.0,
        humair=0.01, zref=2.0, roughness=sc["roughness"],
        relhums=sc["relhums"], aM=sc["aM"], bM=sc["bM"], l1=0.5,
        longwave=300.0, shortwave=0.0, netrad=-1.0, cloud=-1.0,
        albedo=sc["albedo"], albedo_lw=sc["albedo_lw"],
        emissivity=sc["emissivity"], radwater_integr=np.zeros(41),
        radice_integr=np.zeros(2))
    r = t_solver(lambda t: heatbalance_ice(t, ctx), 1, 2, 0, 0.0)
    assert np.isfinite(r["Tsurf3"])
    assert -20.0 < r["Tsurf3"] < 2.0


def test_ice_bathymetry_uses_morphometry():
    """BATHYMETRY interpolates the ice area, not the full lake area."""
    from lakematrix.step import _zsoil_geom

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    st = ss.state
    st.l1[0] = 0.0527
    from lakematrix.physics.bathym import bathymetry
    from lakematrix.constants import C
    b = bathymetry(case, grid, st, C.pref)
    geom = _zsoil_geom(case, grid, 0, st, b)
    top = geom.bathymice[1].area_int
    assert top < float(case.area_lake[0])
    assert geom.bathymdice[1].area_int == float(case.area_lake[0])


def test_case2_one_step_smoke():
    import numpy as np
    from lakematrix.forcing import read_ascii
    from lakematrix.step2 import step_case2

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / "ref" / "v1" / "full425d" / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    st = ss.state
    st.l1[0] = 0.1
    st.Ti1[0] = np.linspace(-5.0, 0.0, case.Mice + 1)
    st.Tw1[0] = np.maximum(st.Tw1[0], 0.0)
    r = step_case2(ss, case, grid, rows, 460620, 0)
    for key in ("Tw1", "Tsoil1", "Ti1", "lamw"):
        assert np.all(np.isfinite(r[key]))
    assert r["l1"] > 0.0


def test_water_freeze_melt_switch():
    from lakematrix.physics.soil import water_freeze_melt
    from lakematrix.constants import C

    # 1 mK below freezing in a 1 m cell cannot freeze (latent heat too large)
    assert not water_freeze_melt(-0.001, 1.0, 0.0, 1)
    # 0.3 K below freezing triggers the "> 0.2 K" branch
    assert water_freeze_melt(-0.3, 1.0, 0.0, 1)
    # symmetric ice -> water check
    assert not water_freeze_melt(0.001, 1.0, 0.0, -1)
    assert water_freeze_melt(0.3, 1.0, 0.0, -1)


def test_layer_case_selects_ice_at_freezing():
    from lakematrix.forcing import read_ascii
    from lakematrix.step2 import layer_case

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / "ref" / "v1" / "full425d" / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    ss.state.Tw1[0][0] = -0.3
    ss.grid_ddz = grid.ddz
    ss.grid_ddzi = grid.ddzi
    assert layer_case(case, ss, 1, rows) == 2


def test_step_any_freezes_surface_water():
    from lakematrix.forcing import read_ascii
    from lakematrix.step2 import step_any

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / "ref" / "v1" / "full425d" / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    ss.state.Tw1[0][0] = -0.3
    r = step_any(ss, case, grid, rows, 1, nstep0=1)
    assert ss.state.layer_case[0] == 2
    assert r["l1"] > 0.0


def test_snow_zero_layer_guard():
    from lakematrix.physics.ice import (diff_coef_snow, make_ice_state,
                                        snow_cond_heat_coef)
    st = make_ice_state(40, 10, 10, 100)
    st.flag_snow_init = 0
    st.itop = 98
    st.dens[98] = 0.0
    st.dz[98] = 0.0
    st.wl[98] = 0.0
    snow_cond_heat_coef(st, 0.0)
    assert np.all(np.isfinite(st.cs[98:100]))
    a = np.zeros(101); b = np.zeros(101); c = np.zeros(101); d = np.zeros(101)
    diff_coef_snow(a, b, c, d, 1, 1, 1, 1, 20.0, st)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
