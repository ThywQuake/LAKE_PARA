"""Replay `K_EPSILON` from a `KE_in` probe directory (development helper)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.bathym import bathymetry  # noqa: E402
from lakematrix.physics.kepsilon import (KEpsilonParams, KEpsilonState,  # noqa: E402
                                         k_epsilon)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.state import init_state  # noqa: E402
from probe import read_probe_sequence  # noqa: E402


def _p1(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x) + 1)
    out[1:] = x
    return out


def replay(probe_dir, dt=20.0):
    probe_dir = Path(probe_dir)
    i = {n: v for n, v in read_probe_sequence(probe_dir / "KE_in.bin")}
    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    st0 = init_state(case, grid)
    b = bathymetry(case, grid, st0, 100140.111111)
    M = case.M
    s = KEpsilonState(
        e1=_p1(i["E1"]), eps1=_p1(i["eps1"]), k2=_p1(i["k2"]),
        k2t=_p1(i["k2t"]), k5=_p1(i["k5"]), k2_mid=_p1(i["k2_mid"]),
        k3_mid=_p1(i["k3_mid"]), k5_mid=_p1(i["k5_mid"]), l=_p1(i["l"]),
        tf=_p1(i["tf"]), kt=_p1(i["kt"]), gamt=_p1(i["gamt"]),
        gamu=_p1(i["gamu"]), gamv=_p1(i["gamv"]), row=_p1(i["row"]),
        row2=_p1(i["row2"]), z_half=_p1(i["z_half"]), u1=_p1(i["u1"]),
        v1=_p1(i["v1"]), u2=_p1(i["u2"]), v2=_p1(i["v2"]), um=_p1(i["um"]),
        vm=_p1(i["vm"]), tw1=_p1(i["Tw1"]), tw2=_p1(i["Tw2"]),
        sal1=_p1(i["Sal1"]), sal2=_p1(i["Sal2"]), knum=_p1(i["knum"]),
        veg_sw=_p1(i["veg_sw"]), lamw=_p1(i["lamw"]), lamsal=_p1(i["lamsal"]),
        h1=float(i["h1"]), l1=float(i["l1"]), dhw=float(i["dhw"]),
        dhw0=float(i["dhw0"]), vol=float(i["vol"]), botar=float(i["botar"]),
        b0=float(i["b0"]), tau_air=float(i["tau_air"]),
        tau_wav=float(i["tau_wav"]), tau_i=float(i["tau_i"]),
        tau_gr=float(i["tau_gr"]), uwind=float(i["uwind"]),
        vwind=float(i["vwind"]), fetch=float(i["fetch"]),
        roughness=float(i["roughness"]))
    s.init_arrays(M)
    s.ddz = np.r_[0, grid.ddz]
    s.ddz05 = np.r_[0, grid.ddz05]
    s.dzeta_int = np.r_[0, grid.dzeta_int]
    s.dzeta_05int = np.r_[0, grid.dzeta_05int]
    s.ddz2 = np.zeros(M + 2)
    s.ddz2[1:M] = grid.ddz[:-1] + grid.ddz[1:]
    s.ddz052 = np.zeros(M + 2)
    s.ddz054 = np.zeros(M + 2)
    for k in range(2, M):
        s.ddz052[k] = 0.5 * grid.ddz[k - 2] + grid.ddz[k - 1] + 0.5 * grid.ddz[k]
        s.ddz054[k] = 2.0 * s.ddz052[k]
    s.area_int = np.r_[0, b.area_int[0]]
    s.area_half = np.r_[0, b.area_half[0]]
    return k_epsilon(s, dt, dt_keps=dt, nstep_keps=1, params=KEpsilonParams(M=M)), b


if __name__ == "__main__":
    r, _ = replay(sys.argv[1] if len(sys.argv) > 1 else
                  "scratch/m0.4/thermo/probe-ke2/probe/1")
    print("E2", r.e2[1:6])
    print("eps2", r.eps2[1:6])
