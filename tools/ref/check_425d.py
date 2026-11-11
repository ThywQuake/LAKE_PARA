"""Check a `run_case.py` npz against a Fortran 425-day case (plan section 5)."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from compare import read_water_temp  # noqa: E402


def read_layers(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 18:
            continue
        try:
            rows.append([float(x) for x in p[:18]])
        except ValueError:
            continue
    a = np.array(rows)
    return dict(day=a[:, 4] / 24.0, h1=a[:, 5], l1=a[:, 14], hs1=a[:, 15])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("case")
    a = ap.parse_args()
    d = np.load(a.npz)
    case = Path(a.case)
    ts = case / "results" / "test" / "time_series"
    ref = read_water_temp(ts / "water_temp  1  1.dat")
    lay = read_layers(ts / "layers  1  1.dat")
    days = d["n"] / 4320.0
    z = d["z"]
    ice = d["l1"]
    snow = d["hs1"]
    # profiles
    errs = []
    for k, day in enumerate(days):
        row = min(ref, key=lambda r: abs(r[0] - day))
        mask = (row[2] > -999.0) & (row[1] <= z.max())
        e = np.interp(row[1][mask], z, d["Tw1"][k]) - row[2][mask]
        errs.append(float(np.max(np.abs(e))))
    errs = np.array(errs)
    j = int(np.argmin(np.abs(lay["day"] - days)))
    ice_ref = lay["l1"][j]
    snow_ref = lay["hs1"][j]
    fi = np.where(ice > 0)[0]
    fr = np.where(ice_ref > 0)[0]
    print(f"days checked {len(days)}, last {days[-1]:.2f}")
    print(f"max |dT| over all days: {errs.max():.6f} K")
    print(f"ice max |d|: {np.max(np.abs(ice - ice_ref)):.6f} m")
    print(f"snow max |d|: {np.max(np.abs(snow - snow_ref)):.6f} m")
    if len(fi) and len(fr):
        print(f"freeze-up day ours {days[fi[0]]:.3f} ref {lay['day'][fr[0]]:.3f}")
        print(f"break-up  day ours {days[fi[-1]]:.3f} ref {lay['day'][fr[-1]]:.3f}")


if __name__ == "__main__":
    main()
