"""Build a copy of the lakebatch Fortran source with the biogeochemistry and
BUBBLE calls commented out, for the M0.4 "does biogeochemistry feed back into
thermodynamics?" experiment.

The copy is made under `lakebatch/scratch/thermo-only/` (never in place).  The
call statements are removed with their continuation lines; the soil calls that
also compute soil temperature are kept, so the experiment isolates the gas /
chemistry / bubble routines.

usage: python3 tools/ref/make_thermo_only.py [--out DIR]
"""

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAKEBATCH = ROOT / "lakebatch"
SRC = LAKEBATCH / "core" / "source"

# Top-level calls in SUBROUTINE LAKE that belong to biogeochemistry or BUBBLE.
DROP = {
    "CHLOROPHYLLA",
    "OXYGEN_PRODCONS",
    "METHANE",
    "ADDOXPRODCONS",
    "OXYGEN",
    "CARBON_DIOXIDE",
    "PHOSPHORUS",
    "METHANE_OXIDATION",
    "BUBBLE_BLOCK",
}

CALL = re.compile(r"^\s*call\s+([A-Za-z_]\w*)", re.I)


def patch_lake(path):
    lines = path.read_text().splitlines()
    out = []
    i = 0
    dropped = []
    while i < len(lines):
        line = lines[i]
        m = CALL.match(line)
        if m and m.group(1).upper() in DROP:
            name = m.group(1)
            start = i + 1
            while True:
                cur = lines[i]
                out.append("!" + cur)
                i += 1
                # A free-form statement continues while the line ends in '&'.
                if not cur.rstrip().endswith("&") or i >= len(lines):
                    break
            dropped.append((name, start))
            continue
        out.append(line)
        i += 1
    path.write_text("\n".join(out) + "\n")
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LAKEBATCH / "scratch" / "thermo-only"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; use a new --out directory (no rm -rf)")
    (out / "build").mkdir(parents=True)

    shutil.copytree(SRC, out / "source")
    dropped = patch_lake(out / "source" / "model" / "lake.f90")

    cmake = (LAKEBATCH / "core" / "build" / "CMakeLists.txt").read_text()
    cmake = cmake.replace('RUNTIME_OUTPUT_DIRECTORY "../"', 'RUNTIME_OUTPUT_DIRECTORY "."')
    old = 'list(APPEND FLAGS_ "-O3")'
    new = 'list(APPEND FLAGS_ "-O3" "-ffp-contract=off")'
    if old not in cmake:
        raise SystemExit("CMakeLists: -O3 flag line not found")
    cmake = cmake.replace(old, new)
    (out / "build" / "CMakeLists.txt").write_text(cmake)

    print(f"copied source to {out/'source'}")
    print(f"commented {len(dropped)} call statements:")
    for name, line in dropped:
        print(f"  {line:5d}  {name}")
    print(f"build: cd {out/'build'} && cmake "
          f"-DCMAKE_POLICY_VERSION_MINIMUM=3.5 "
          f"-DCMAKE_Fortran_FLAGS='-I{out/'source'/'model'}' . && "
          f"cmake --build . --target lake.out --parallel 8")


if __name__ == "__main__":
    main()
