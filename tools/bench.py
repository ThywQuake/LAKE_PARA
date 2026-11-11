"""Measure single-lake / per-process batch throughput (M5 baseline)."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state, step_case1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lakes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--case", default="ref/v1/summer60d")
    a = ap.parse_args()
    case = read_case(ROOT / a.case)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / a.case / "meteo" / "test.dat")
    models = [build_step_state(case, grid) for _ in range(a.lakes)]
    t0 = time.perf_counter()
    for n in range(1, a.steps + 1):
        for ss in models:
            r = step_case1(ss, case, grid, rows, n, 0, init_step=(n == 1))
            st = ss.state
            st.Tw1[0] = r["Tw1"]; st.Sal1[0] = r["Sal1"]
            st.Tsoil1[0] = r["Tsoil1"]; st.h1[0] = r["h1"]
            st.E[0] = r["E1"]; st.eps[0] = r["eps1"]
            st.lamw[0] = r["lamw"]; ss.Elatent = r["Elatent"]
            ss.soil.dhwfsoil = r["dhwfsoil"]
    dt = time.perf_counter() - t0
    print(f"lakes={a.lakes} steps={a.steps} time={dt:.3f}s "
          f"lake-steps/s={a.lakes * a.steps / dt:.1f}")


if __name__ == "__main__":
    main()
