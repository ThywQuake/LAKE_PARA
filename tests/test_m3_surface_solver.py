"""M3 test: the case-1 HEATBALANCE/T_SOLVER against the step-1 probes."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.surface import WaterSurfaceContext, heatbalance_water  # noqa: E402
from lakematrix.physics.tsolver import t_solver  # noqa: E402

TD = ROOT / "scratch" / "m0.4" / "thermo" / "probe-td2" / "probe" / "1"
SURF = ROOT / "scratch" / "m0.4" / "thermo" / "probe-surf" / "probe" / "1"
SLM = ROOT / "scratch" / "m0.4" / "thermo" / "probe-slm" / "probe" / "1"
SC = ROOT / "scratch" / "m0.4" / "thermo" / "probe-sc" / "probe" / "1"
TS = ROOT / "scratch" / "m0.4" / "thermo" / "probe-ts" / "probe" / "1"
M, NS, ISOIL = 40, 10, 5


def _pad1(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,))
    out[..., 1:] = x
    return out


@pytest.mark.skipif(not (TD / "TD_in.bin").exists(), reason="probes not generated")
def test_heatbalance_and_t_solver_match_fortran():
    from probe import read_probe_sequence

    td = read_probe_sequence(TD / "TD_in.bin")
    hb = read_probe_sequence(SURF / "HB_out.bin")
    slm = read_probe_sequence(SLM / "SLM_in.bin")
    rb = read_probe_sequence(SURF / "RB_in.bin")
    sc = read_probe_sequence(SC / "SC_out.bin")
    ts = read_probe_sequence(TS / "TS_out.bin")

    names = [n for n, _ in td[:19]]
    a = {names[i]: td[i][1] for i in range(19)}
    Tsoil1 = np.zeros((NS + 1, 6))
    Tsoil1[1:, 1:] = a["Tsoil1"]
    grid = create_grid(M, 10, NS, 1e-2, 1e-2, 10.0, 0.02)
    slm_in = {n: float(v) for n, v in slm[:18]}
    rb_in = {n: float(v) for n, v in rb[:12]}
    sc_out = {n: float(v) for n, v in sc}
    ctx = WaterSurfaceContext(
        Tw1=_pad1(a["Tw1"]), Tsoil1=Tsoil1, lamw=_pad1(a["lamw"]),
        area_int=_pad1(a["area_int"]), area_half=_pad1(a["area_half"]),
        h1=float(a["h1"]), ddz=_pad1(a["ddz"]), dhw=float(a["dhw"]),
        dhw0=float(a["dhw0"]), PEMF=_pad1(a["PEMF"]),
        pt_down_f=_pad1(a["pt_down_f"]), lamsoil=_pad1(a["lamsoil"]),
        csoil=_pad1(a["csoil"]), rosoil=_pad1(a["rosoil"]),
        dzs=_pad1(a["dzs"]), radwater_integr=a["RadWater_integr"],
        lsh_water=_pad1(a["lsh_water"]), dzeta_int=_pad1(grid.dzeta_int),
        ddz05=grid.ddz05, M=M, ns=NS, isoilcol=ISOIL, dt=float(a["dt"]),
        fetch=slm_in["fetch"], uwind=slm_in["uwind"], vwind=slm_in["vwind"],
        u=slm_in["u"], v=slm_in["v"], tempair=slm_in["tempair"],
        pressure=slm_in["pressure"], humair=slm_in["humair"],
        zref=slm_in["zref"], roughness=slm_in["roughness"],
        relhums=slm_in["relhums"], aM=slm_in["aM"], bM=slm_in["bM"],
        l1=slm_in["l1"], longwave=rb_in["longwave"],
        shortwave=rb_in["shortwave"], netrad=rb_in["netrad"],
        cloud=rb_in["cloud"], albedo=sc_out["albedo"],
        albedo_lw=sc_out["albedo_lw"], emissivity=sc_out["emissivity"])

    def fn(t):
        HB, state = heatbalance_water(t, ctx)
        return HB, state

    n = len(td) // 19
    for k in range(n):
        T = float(td[k * 19 + names.index("Tsurf")][1])
        HB, _ = fn(T)
        np.testing.assert_allclose(HB, float(hb[k * 10 + 9][1]), rtol=0, atol=1e-9)

    ref = {n: v for n, v in ts}
    r = t_solver(fn, 1, 4, 4, 0.0)
    assert r["Tsurf3"] == pytest.approx(float(ref["Tsurf3"]), abs=1e-10)
    assert r["iter"] == int(ref["iter"])
    assert r["Bal3"] == pytest.approx(float(ref["Bal3"]), abs=1e-10)
