"""M3 test: soil bathymetry + SOILCOLSTEMP + LATERHEAT against the T_DIFF probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.grid import create_grid, zsoilcols  # noqa: E402
from lakematrix.physics.bathym import bathymetry, soil_bathymetry  # noqa: E402
from lakematrix.physics.soil import (SoilGeometry, bathym_levels,  # noqa: E402
                                     comsoilforlake, laterheat,
                                     make_soil_state, soil_cond_heat_coef,
                                     soilcolstemp)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.state import init_state  # noqa: E402

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-td2" / "probe" / "1"


@pytest.mark.skipif(not (PROBE / "TD_in.bin").exists(), reason="T_DIFF probe not generated")
def test_laterheat_matches_probe():
    from probe import read_probe_sequence

    ins = {n: v for n, v in read_probe_sequence(PROBE / "TD_in.bin")[:19]}
    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    st = init_state(case, grid)
    h1 = float(st.h1[0])
    bw = bathymetry(case, grid, st, 100140.111111)
    sp = comsoilforlake(soiltype=5, ns=10, nsoilcols=5, soil_depth=10.0)
    zsc = zsoilcols(h1, case.morphometry[0, :int(case.morphometry_len[0]), :],
                    grid, 5)
    bs = soil_bathymetry(case, grid, st, 0)
    bwater = bathym_levels(41, area_int=bw.area_int[0], area_half=bw.area_half[0],
                           Lx=bw.Lx[0], Ly=bw.Ly[0],
                           Lx_half=bw.Lx_half[0], Ly_half=bw.Ly_half[0])
    bice = bathym_levels(11, area_int=np.full(11, case.area_lake[0]),
                         area_half=np.full(10, case.area_lake[0]))
    geom = SoilGeometry(M=40, Mice=10, zsoilcols=np.r_[0, zsc],
                        ddz=np.r_[0, grid.ddz], ddzi=np.r_[0, grid.ddzi],
                        ddz05=grid.ddz05, ddzi05=grid.ddzi05,
                        z_half=np.r_[0, grid.dzeta_05int * h1],
                        bathymwater=bwater, bathymice=bice,
                        bathymdice=bice, bathymsoil=bs)
    ss = make_soil_state(sp, geom, h1=h1, l1=0.0, hs1=0.0, ls1=0.0,
                         pressure=100140.111111,
                         Tw1=np.r_[0, st.Tw1[0]], Ti1=np.zeros(11),
                         Tis1=np.zeros(11))
    ss.Tsoil1[1:, 1:] = st.Tsoil1[0]
    ss.Sals1[1:, 1:] = st.Sals1[0]
    ss.wl1[1:, 1:] = st.wl1[0]
    ss.wi1[1:, 1:] = st.wi1[0]
    soil_cond_heat_coef(ss, sp, is_=5)
    ss = soilcolstemp(ss, sp, 20.0)
    lh = laterheat(ss, sp, ss.soilflux, onlywater=True)
    np.testing.assert_allclose(lh.water[1:41], ins["lsh_water"][:40],
                               rtol=0, atol=1e-10)
