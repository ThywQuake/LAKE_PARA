"""Create the M0.4 reference case set under `lakematrix/ref/v1/`.

Each case is the shipped testlake setup, with `tribheat 0` (the v1 model has no
tributaries) and, for the variants, a changed forcing or driver.  The Fortran
runs themselves are started with `--run` (check build) or later by the probe /
comparison tools.

usage: python3 tools/ref/make_ref_cases.py [--only NAME ...] [--run] [--jobs N]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lakebatch" / "vector" / "rewrite"))

from make_cases import _cold, _rows, make_case  # noqa: E402

OUT = ROOT / "lakematrix" / "ref" / "v1"


def _shift(rows, dk):
    out = []
    for line in rows:
        cols = line.split(",")
        cols[0] = repr(float(cols[0]) + dk)
        out.append(",".join(cols))
    return out


def _set_key(path, key, value):
    lines = path.read_text().splitlines()
    lines = [f"{key} {value}" if l.split()[:1] == [key] else l for l in lines]
    path.write_text("\n".join(lines) + "\n")


def _set_h10_and_scale_table(driver, h10, original_depth):
    """Set h10 and scale the depth column of the `morphometry` table."""
    lines = driver.read_text().splitlines()
    out = []
    i = 0
    scale = h10 / original_depth
    while i < len(lines):
        parts = lines[i].split()
        if parts[:1] == ["morphometry"] and len(parts) >= 2:
            n = int(parts[1])
            out.append(lines[i])
            i += 1
            for _ in range(n):
                cols = lines[i].split()
                if len(cols) >= 2:
                    out.append(f"{float(cols[0]) * scale:.6f}\t{cols[1]}")
                else:
                    out.append(lines[i])
                i += 1
            continue
        if parts[:1] == ["h10"]:
            out.append(f"h10 {h10}")
            i += 1
            continue
        out.append(lines[i])
        i += 1
    driver.write_text("\n".join(out) + "\n")


def build_case(name, dst):
    full = _rows()
    if name == "summer1d":
        make_case(dst, [full[:49]], 1)
    elif name == "summer60d":
        make_case(dst, [full[: 49 + 24 * 60]], 60)
    elif name == "full425d":
        make_case(dst, [full], 425)
    elif name == "cold425d":
        make_case(dst, [_cold(full)], 425)
    elif name == "shallow425d":
        make_case(dst, [full], 425)
        _set_h10_and_scale_table(dst / "setup/testlake_driver.dat", 5.0, 22.66)
    elif name == "shallow05m425d":
        make_case(dst, [full], 425)
        _set_h10_and_scale_table(dst / "setup/testlake_driver.dat", 0.5, 22.66)
    elif name == "shallow1m425d":
        make_case(dst, [full], 425)
        _set_h10_and_scale_table(dst / "setup/testlake_driver.dat", 1.0, 22.66)
    elif name == "warm425d":
        make_case(dst, [_shift(full, 3.0)], 425)
        _set_key(dst / "setup/testlake_driver.dat", "phi", 40.0)
    else:
        raise SystemExit(f"unknown case {name}")
    _set_key(dst / "setup/testlake_setup.dat", "tribheat", 0)


CASES = ["summer1d", "summer60d", "full425d", "cold425d", "shallow425d",
         "shallow05m425d", "shallow1m425d", "warm425d"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=CASES)
    ap.add_argument("--run", action="store_true", help="run the Fortran check build in each case")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    exe = Path(os.environ.get(
        "LAKE_EXE",
        ROOT / "lakebatch" / ("core/build-check-linux" if sys.platform != "darwin"
                              else "core/build-check") / "lake.out",
    ))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        dst = OUT / name
        if dst.exists():
            raise SystemExit(f"{dst} exists; use a new ref directory (no rm -rf)")
        build_case(name, dst)
        print(f"built {dst}")

    if args.run:
        if not exe.exists():
            raise SystemExit(f"check build not found: {exe}")

        def run(name):
            dst = OUT / name
            with (dst / "run.log").open("w") as log:
                rc = subprocess.call([str(exe)], cwd=dst, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "OMP_NUM_THREADS": "1"})
            (dst / "status").write_text(f"{rc}\n")
            print(f"ran {name}: rc={rc}")

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run, args.only))


if __name__ == "__main__":
    main()
