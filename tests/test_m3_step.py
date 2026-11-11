"""M3 integration test: one case-1 step through T_SOLVER."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state, step_case1_to_solver  # noqa: E402

TS = ROOT / "scratch" / "m0.4" / "thermo" / "probe-ts" / "probe" / "1"


@pytest.mark.skipif(not (TS / "TS_out.bin").exists(), reason="T_SOLVER probe not generated")
def test_step_case1_to_solver_matches_probe():
    from probe import read_probe_sequence

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    rows = read_ascii(ROOT / "ref" / "v1" / "full425d" / "meteo" / "test.dat")
    r = step_case1_to_solver(ss, case, grid, rows, 1)
    ts = {n: float(v) for n, v in read_probe_sequence(TS / "TS_out.bin")}
    assert r["Tsurf3"] == pytest.approx(ts["Tsurf3"], abs=1e-9)
    assert r["Bal3"] == pytest.approx(ts["Bal3"], abs=1e-4)


@pytest.mark.skipif(not (ROOT / "scratch/m0.4/thermo/probe-forcing/probe/2/LAKE_entry.bin").exists(),
                    reason="step-2 probe not generated")
def test_full_case1_step_close_to_probe():
    from probe import read_probe_sequence

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    rows = read_ascii(ROOT / "ref" / "v1" / "full425d" / "meteo" / "test.dat")
    from lakematrix.step import step_case1
    r = step_case1(ss, case, grid, rows, 1)
    p2 = {n: v for n, v in read_probe_sequence(
        ROOT / "scratch/m0.4/thermo/probe-forcing/probe/2/LAKE_entry.bin")}
    assert np.max(np.abs(r["Tw1"] - p2["Tw1"])) < 0.03
    assert np.max(np.abs(r["Tsoil1"] - p2["Tsoil1"])) < 0.07
    assert abs(r["h1"] - float(p2["h1"])) < 1e-3


def test_bathymetry_is_refreshed_each_step():
    """BATHYMETRY must be recomputed with the current h1 (regression)."""
    from lakematrix.constants import C
    from lakematrix.physics.bathym import bathymetry
    from lakematrix.step import step_case1

    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / "ref" / "v1" / "full425d" / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    initial_area = np.array(ss.bathym.area_int, copy=True)
    ss.state.h1[0] = float(ss.state.h1[0]) - 0.5
    step_case1(ss, case, grid, rows, 2, 0)
    fresh = bathymetry(case, grid, ss.state, C.pref)
    assert not np.allclose(ss.bathym.area_int, initial_area)
    np.testing.assert_allclose(ss.bathym.area_int, fresh.area_int, rtol=0,
                               atol=0)
    np.testing.assert_allclose(ss.soil.geom.bathymwater[1].area_int,
                               float(ss.bathym.area_int[0, 0]), rtol=0,
                               atol=0)
