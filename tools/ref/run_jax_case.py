"""Run a reference case with the full JAX step (case 1, or case 3 if h10=0)."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.jaxbatch.step import (from_numpy_state, make_params,  # noqa: E402
                                      run_case1_jax)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402
from lakematrix.step3 import build_case3_state  # noqa: E402
from lakematrix.jaxbatch.step3 import run_case3_jax  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("days", type=float)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    case_dir = Path(a.case)
    case = read_case(case_dir)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(case_dir / "meteo" / "test.dat")
    params = make_params(case, grid)
    case3 = (float(np.atleast_1d(case.h10)[0]) == 0.0
             and float(np.atleast_1d(case.l10)[0]) > 0.0)
    ss = build_case3_state(case, grid) if case3 else build_step_state(case, grid)
    state = from_numpy_state(ss, ss.ice if case3 else None)
    every = int(round(86400.0 / case.dt))
    nsteps = int(round(a.days * 86400.0 / case.dt))
    rec = []
    n = 0
    while n < nsteps:
        k = min(every, nsteps - n)
        if case3:
            state = run_case3_jax(params, rows, k, state, nstep0=n + 1)
        else:
            state, _ = run_case1_jax(params, rows, k, state, nstep0=n + 1,
                                     first=(n == 0))
        n += k
        w = state["water"]
        s = state["soil"]
        if case3:
            rec.append(dict(n=n, Tw1=np.asarray(state["ice"]["Ti1"])[0],
                            Tsoil1=np.asarray(s["Tsoil1"])[0],
                            h1=0.0, lamw=np.zeros_like(w["lamw"][0])))
        else:
            rec.append(dict(n=n,
                            Tw1=np.asarray(w["Tw1"][0]),
                            Tsoil1=np.asarray(s["Tsoil1"][0]),
                            h1=float(np.asarray(w["h1"])[0]),
                            lamw=np.asarray(w["lamw"][0])))
        print(f"day {n // every}", flush=True)
    h1 = float(np.asarray(state["water"]["h1"])[0])
    out = Path(a.out) if a.out else ROOT / "scratch" / f"{case_dir.name}_jax_{int(a.days)}d.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, n=np.array([p["n"] for p in rec]),
             Tw1=np.array([p["Tw1"] for p in rec]),
             Tsoil1=np.array([p["Tsoil1"] for p in rec]),
             h1=np.array([p["h1"] for p in rec]),
             lamw=np.array([p["lamw"] for p in rec]),
             z=grid.dzeta_int * h1)
    print("saved", out)


if __name__ == "__main__":
    main()
