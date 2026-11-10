"""M3 test: MOMENTUM_SOLVER against the step-1 Fortran probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.forcing import at_step, read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.bathym import bathymetry  # noqa: E402
from lakematrix.physics.momentum import (MomentumParams, MomentumState,  # noqa: E402
                                         momentum_solver)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.state import init_state  # noqa: E402

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-mom" / "probe" / "1"
SLM = ROOT / "scratch" / "m0.4" / "thermo" / "probe-slm" / "probe" / "1"


@pytest.mark.skipif(not (PROBE / "MOM_1105_in.bin").exists(), reason="momentum probe not generated")
def test_momentum_matches_fortran():
    from probe import read_probe_sequence

    ins = {n: v for n, v in read_probe_sequence(PROBE / "MOM_1105_in.bin")}
    out = {n: v for n, v in read_probe_sequence(PROBE / "MOM_1105_out.bin")}
    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    st = init_state(case, grid)
    b = bathymetry(case, grid, st, 100140.111111)
    rows = read_ascii("/Users/mac/文档/研究生/LAKE/LAKE3.2/meteo/testlake.dat")
    f = at_step(rows, 1, 20.0, 1.0)
    slm = read_probe_sequence(SLM / "SLM_out.bin")
    params = MomentumParams(
        M=40, ddz=grid.ddz, ddz05=grid.ddz05, dzeta_int=grid.dzeta_int,
        dzeta_05int=grid.dzeta_05int, area_int=b.area_int[0],
        area_half=b.area_half[0], Lx=b.Lx[0], Ly=b.Ly[0],
        Lx_half=b.Lx_half[0], Ly_half=b.Ly_half[0], depth_area=np.zeros((1, 2)),
        kor=float(case.kor[0]), uwind=float(f[3]), vwind=float(f[4]),
        cdmw=float(slm[3][1]), zref=2.0, fetch=1000.0)
    s = MomentumState(
        u1=ins["u1"], v1=ins["v1"], E1=ins["E1"], eps1=ins["eps1"],
        k2=ins["k2"], k2_old=np.zeros(40), row=ins["row"], Ri=ins["Ri"],
        knum=ins["knum"], veg_sw=ins["veg_sw"], u_star=float(ins["u_star"]),
        Tw1=ins["Tw1"], Sal1=ins["Sal1"], lsu_water=np.zeros(41),
        lsv_water=np.zeros(41), uv=ins["uv"], h1=float(ins["h1"]),
        l1=float(ins["l1"]), ls1=float(ins["ls1"]), dhw=float(ins["dhw"]),
        dhw0=float(ins["dhw0"]), alphax=float(ins["alphax"]),
        alphay=float(ins["alphay"]), velfrict_bot=float(ins["velfrict_bot"]),
        velfrict_ice=float(ins["velfrict_ice"]))
    r = momentum_solver(s, params, 20.0)
    np.testing.assert_allclose(r.u2, out["u2"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r.v2, out["v2"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(r.state.k2, out["k2"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r.state.Ri, out["Ri"], rtol=0, atol=1e-5)
