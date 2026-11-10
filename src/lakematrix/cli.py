"""Command-line entry point for the case-1 runner."""

import argparse
from pathlib import Path

import numpy as np

from .forcing import read_ascii
from .grid import create_grid
from .output import save_profiles
from .setup_io import read_case
from .step import run_case1

DEFAULT_METEO = "/Users/mac/文档/研究生/LAKE/LAKE3.2/meteo/testlake.dat"


def main(argv=None):
    p = argparse.ArgumentParser(prog="lakematrix")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one case")
    r.add_argument("case")
    r.add_argument("--meteo", default=DEFAULT_METEO)
    r.add_argument("--days", type=float, default=1.0)
    r.add_argument("--out", default="results.nc")
    a = p.parse_args(argv)
    if a.cmd != "run":
        p.error(f"unknown command {a.cmd}")

    case = read_case(a.case)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(a.meteo)
    nsteps = int(round(a.days * 86400.0 / case.dt))
    every = max(1, int(round(86400.0 / case.dt)))
    times, water, soil, h1 = [], [], [], []

    def cb(n, ss, r):
        if n % every == 0 or n == nsteps:
            times.append(n * case.dt)
            water.append(np.array(r["Tw1"], copy=True))
            soil.append(np.array(r["Tsoil1"], copy=True))
            h1.append(float(r["h1"]))

    ss, _ = run_case1(case, grid, rows, nsteps, callback=cb)
    z = grid.dzeta_int * float(ss.state.h1[0])
    out = save_profiles(a.out, times, z, np.array(water), np.array(soil),
                        np.array(h1), attrs=dict(case=Path(a.case).name,
                                                 days=a.days))
    print(out)


if __name__ == "__main__":
    main()
