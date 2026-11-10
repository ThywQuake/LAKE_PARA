"""Run the case-1 summer60d case and save daily profiles."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import run_case1  # noqa: E402

case = read_case(ROOT / "ref" / "v1" / "summer60d")
grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                   case.soil_depth, 0.02)
rows = read_ascii("/Users/mac/文档/研究生/LAKE/LAKE3.2/meteo/testlake.dat")
profiles = []


def cb(n, ss, r):
    if n % 4320 == 0:
        profiles.append(dict(n=n, Tw1=np.array(r["Tw1"], copy=True),
                             Tsoil1=np.array(r["Tsoil1"], copy=True),
                             h1=float(r["h1"]), lamw=np.array(r["lamw"], copy=True)))

ss, _ = run_case1(case, grid, rows, 60 * 4320, callback=cb)
np.savez(ROOT / "scratch" / "m3_summer60d.npz",
         n=np.array([p["n"] for p in profiles]),
         Tw1=np.array([p["Tw1"] for p in profiles]),
         Tsoil1=np.array([p["Tsoil1"] for p in profiles]),
         h1=np.array([p["h1"] for p in profiles]),
         lamw=np.array([p["lamw"] for p in profiles]),
         z=grid.dzeta_int * float(ss.state.h1[0]))
print("done", len(profiles))
