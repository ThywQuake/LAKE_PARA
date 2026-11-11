"""M4 case-2 transition checks against Fortran probes (scratch/ only)."""

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
from lakematrix.step import build_step_state  # noqa: E402
from lakematrix.step2 import step_any  # noqa: E402
from probe import read_probe_sequence  # noqa: E402

PROBE = ROOT / "scratch" / "m4" / "probe"
CASE = ROOT / "ref" / "v1" / "full425d"


def _case_grid():
    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    return case, grid


def _load(ss, case, step, pre):
    e = {n: v for n, v in read_probe_sequence(PROBE / str(pre) / "LAKE_exit.bin")}
    sf = {n: v for n, v in read_probe_sequence(PROBE / str(pre) / "SOILF_out.bin")}
    ts = {n: v for n, v in read_probe_sequence(PROBE / str(pre) / "TS_out.bin")}
    st = ss.state
    st.Tw1[0] = e["Tw1"]; st.Sal1[0] = e["Sal1"]
    st.Ti1[0] = e["Ti1"]; st.Tis1[0] = e["Tis1"]
    st.Tsoil1[0] = e["Tsoil1"]; st.h1[0] = e["h1"]
    st.l1[0] = e["l1"]; st.ls1[0] = e["ls1"]; st.hs1[0] = e["hs1"]
    st.snowmass[0] = e["snowmass"]
    st.flag_snow[0] = int(e.get("flag_snow", 0))
    st.itop[0] = int(e.get("itop", 0))
    st.lamw[0] = e["lamw"]; st.E[0] = e["E1"]; st.eps[0] = e["eps1"]
    st.time[0] = e["time"]; st.nstep[0] = int(e["nstep"])
    st.layer_case[0] = 1 if e["l1"] == 0 else 2
    ss.soil.Tsoil1[1:, 1:] = e["Tsoil1"]
    ss.soil.wl1[1:, 1:] = sf["wl1"]; ss.soil.wi1[1:, 1:] = sf["wi1"]
    ss.soil.dhwfsoil = float(e["dhwfsoil"])
    ss.Tsurf_prev = float(ts["Tsurf3"])
    ss.surftyp_prev = int(ts["surftyp"])
    mom_tags = sorted(p.name for p in (PROBE / str(step)).glob("MOM_*_in.bin"))
    mom = {n: v for n, v in read_probe_sequence(PROBE / str(step) / mom_tags[0])}
    ss.mom.u1 = np.asarray(mom["u1"]); ss.mom.v1 = np.asarray(mom["v1"])
    ss.mom_u1_old = np.array(ss.mom.u1); ss.mom_v1_old = np.array(ss.mom.v1)
    ss.ke.firstcall = False
    ss.ke.init_keps = True
    from lakematrix.physics.ice import make_ice_state
    ss.ice = make_ice_state(
        case.M, case.Mice, case.ns, 100, geom=ss.soil_geom,
        isoilcol=case.nsoilcols,
        nmeltpoint=int(np.atleast_1d(case.nmeltpoint)[0]),
        soilbotflx=float(np.atleast_1d(case.soilbotflx)[0]))
    ice = ss.ice
    ice.T[1:1 + len(e["T"])] = e["T"]
    ice.Tsn[1:1 + len(e["Tsn"])] = e["Tsn"]
    ice.cs[1:1 + len(e["cs"])] = e["cs"]
    ice.dens[1:1 + len(e["dens"])] = e["dens"]
    ice.dz[1:1 + len(e["dz"])] = e["dz"]
    ice.lams[1:1 + len(e["lams"])] = e["lams"]
    ice.itop = int(e["itop"])
    ice.flag_snow = int(e["flag_snow"])
    ice.flag_snow_init = int(e.get("flag_snow_init", 0))
    ice.snowmass = float(e["snowmass"])
    ice.hs1 = float(e["hs1"])


@pytest.mark.skipif(not (PROBE / "460619" / "LAKE_exit.bin").exists(),
                    reason="M4 transition probes not generated")
def test_freeze_up_step_matches_probe():
    case, grid = _case_grid()
    rows = read_ascii(CASE / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    _load(ss, case, 460620, 460619)
    _ke = {n: v for n, v in read_probe_sequence(PROBE / "460619" / "KE_in.bin")}
    ss.ke.k2[1:] = _ke["k2"]
    r = step_any(ss, case, grid, rows, 460620, nstep0=460620)
    t = {n: v for n, v in read_probe_sequence(PROBE / "460620" / "LAKE_exit.bin")}
    assert np.max(np.abs(r["Tw1"] - t["Tw1"])) < 1e-5
    assert np.max(np.abs(r["Tsoil1"] - t["Tsoil1"])) < 1e-5
    assert abs(r["h1"] - float(t["h1"])) < 1e-6
    assert abs(r["l1"] - float(t["l1"])) < 1e-5


@pytest.mark.skipif(not (PROBE / "545579" / "LAKE_exit.bin").exists(),
                    reason="M4 snow probes not generated")
def test_snow_step_matches_probe():
    case, grid = _case_grid()
    rows = read_ascii(CASE / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    _load(ss, case, 545580, 545579)
    e = {n: v for n, v in read_probe_sequence(PROBE / "545579" / "LAKE_exit.bin")}
    r = step_any(ss, case, grid, rows, 545580, nstep0=545580)
    t = {n: v for n, v in read_probe_sequence(PROBE / "545580" / "LAKE_exit.bin")}
    assert np.max(np.abs(r["Tw1"] - t["Tw1"])) < 1e-4
    assert np.max(np.abs(r["Tsoil1"] - t["Tsoil1"])) < 1e-5
    assert abs(r["l1"] - float(t["l1"])) < 1e-5
    assert abs(r["hs1"] - float(t["hs1"])) < 1e-4
    assert abs(r["snowmass"] - float(t["snowmass"])) < 1e-2
