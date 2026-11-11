"""Run one reference case with the full case selector and daily checkpoints.

usage: python tools/ref/run_case.py CASE_DIR DAYS [--out NPZ] [--meteo PATH]
       [--resume] [--ckpt PATH]
"""
import argparse
import pickle
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402
from lakematrix.step2 import step_any  # noqa: E402


def _record(ss, n, every):
    st = ss.state
    return dict(n=n, Tw1=np.array(st.Tw1[0], copy=True),
                Tsoil1=np.array(st.Tsoil1[0], copy=True),
                Ti1=np.array(st.Ti1[0], copy=True),
                h1=float(st.h1[0]), l1=float(st.l1[0]),
                ls1=float(st.ls1[0]), hs1=float(st.hs1[0]),
                snowmass=float(st.snowmass[0]),
                layer_case=int(st.layer_case[0]),
                lamw=np.array(st.lamw[0], copy=True))


def save_npz(out, rec, grid, ss):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             n=np.array([p["n"] for p in rec]),
             Tw1=np.array([p["Tw1"] for p in rec]),
             Tsoil1=np.array([p["Tsoil1"] for p in rec]),
             Ti1=np.array([p["Ti1"] for p in rec]),
             h1=np.array([p["h1"] for p in rec]),
             l1=np.array([p["l1"] for p in rec]),
             ls1=np.array([p["ls1"] for p in rec]),
             hs1=np.array([p["hs1"] for p in rec]),
             snowmass=np.array([p["snowmass"] for p in rec]),
             layer_case=np.array([p["layer_case"] for p in rec]),
             z=grid.dzeta_int * float(ss.state.h1[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("days", type=float)
    ap.add_argument("--meteo", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    case_dir = Path(a.case)
    case = read_case(case_dir)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    meteo = Path(a.meteo) if a.meteo else case_dir / "meteo" / "test.dat"
    rows = read_ascii(meteo)
    nsteps = int(round(a.days * 86400.0 / case.dt))
    every = int(round(86400.0 / case.dt))
    out = Path(a.out) if a.out else ROOT / "scratch" / f"{case_dir.name}_{int(a.days)}d.npz"
    ckpt = Path(a.ckpt) if a.ckpt else out.with_suffix(".ckpt.pkl")

    start, rec = 1, []
    if a.resume and ckpt.exists():
        with ckpt.open("rb") as fh:
            ck = pickle.load(fh)
        ss, start, rec = ck["ss"], ck["n"] + 1, ck["rec"]
        print(f"resume from step {start - 1} ({len(rec)} daily records)", flush=True)
    else:
        ss = build_step_state(case, grid)

    def checkpoint(n):
        tmp = ckpt.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(dict(ss=ss, n=n, rec=rec), fh, protocol=4)
        os.replace(tmp, ckpt)

    for n in range(start, nsteps + 1):
        step_any(ss, case, grid, rows, n, nstep0=1)
        if n % every == 0 or n == nsteps:
            rec.append(_record(ss, n, every))
            print(f"day {n // every}", flush=True)
            checkpoint(n)
            if not np.all(np.isfinite(ss.state.Tw1[0])):
                print(f"NaN at step {n} (day {n / every}); checkpoint is "
                      f"the previous finite day", flush=True)
                break
    save_npz(out, rec, grid, ss)
    print("saved", out)


if __name__ == "__main__":
    main()
