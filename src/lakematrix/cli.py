"""Command-line entry point for the case-1 runner."""

import argparse
from pathlib import Path

import numpy as np

from .forcing import read_ascii
from .grid import create_grid
from .output import save_profiles
from .setup_io import read_case
from .step import run_case1


def _default_meteo(case_dir):
    """Forcing table shipped with a reference case, if there is one."""
    for rel in ("meteo/test.dat", "meteo/testlake.dat"):
        p = Path(case_dir) / rel
        if p.exists():
            return p
    raise SystemExit(
        f"{case_dir}: no meteo/test.dat; pass --meteo PATH explicitly")


def main(argv=None):
    p = argparse.ArgumentParser(prog="lakematrix")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one case")
    r.add_argument("case")
    r.add_argument("--meteo", default=None,
                   help="forcing table (default: <case>/meteo/test.dat)")
    r.add_argument("--days", type=float, default=1.0)
    r.add_argument("--out", default="results.nc")

    b = sub.add_parser("batch", help="run several cases into one (lake, time) file")
    b.add_argument("cases", nargs="+")
    b.add_argument("--days", type=float, default=1.0)
    b.add_argument("--meteo", nargs="*", default=None)
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--ckpt", default=None)
    b.add_argument("--resume", action="store_true")
    b.add_argument("--out", default="results.nc")
    a = p.parse_args(argv)

    if a.cmd == "batch":
        from .run import run_batch, run_batch_parallel, save_netcdf
        if a.jobs > 1:
            data = run_batch_parallel(a.cases, a.days, meteo_paths=a.meteo,
                                      jobs=a.jobs)
        else:
            data = run_batch(a.cases, a.days, meteo_paths=a.meteo,
                             ckpt=a.ckpt, resume=a.resume)
        out = save_netcdf(a.out, data, names=a.cases)
        print(out)
        return

    if a.cmd != "run":
        p.error(f"unknown command {a.cmd}")

    case = read_case(a.case)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    meteo = Path(a.meteo) if a.meteo else _default_meteo(a.case)
    rows = read_ascii(meteo)
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
