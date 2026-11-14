"""Check a `run_case.py` npz against a Fortran 425-day case (plan §5).

Prints pointwise/mean errors, monthly-mean errors and the ice phenology, and
marks the revised (chaos-aware) acceptance criteria.
"""
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


def nearest_index(days, value):
    return int(np.argmin(np.abs(days - value)))


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

    errs, common = [], None
    for k, day in enumerate(days):
        row = min(ref, key=lambda r: abs(r[0] - day))
        mask = (row[2] > -999.0) & (row[1] <= z.max())
        if common is None:
            common = row[1][mask]
        errs.append(np.interp(row[1][mask], z, d["Tw1"][k]) - row[2][mask])
    E = np.array(errs)
    point = np.max(np.abs(E))
    bias = np.max(np.abs(E.mean(axis=0)))
    rmse = np.max(np.sqrt((E ** 2).mean(axis=0)))
    monthly = []
    for lo in range(0, int(days[-1]) + 1, 30):
        sel = (days >= lo) & (days < lo + 30)
        if np.any(sel):
            monthly.append(np.max(np.abs(E[sel].mean(axis=0))))
    month = max(monthly) if monthly else 0.0

    n = len(days)
    ice = d["l1"][:n]
    snow = d["hs1"][:n]
    ice_ref = np.array([lay["l1"][nearest_index(lay["day"], x)] for x in days])
    snow_ref = np.array([lay["hs1"][nearest_index(lay["day"], x)] for x in days])
    ice_err = np.max(np.abs(ice - ice_ref))
    snow_err = np.max(np.abs(snow - snow_ref))

    fi = np.where(ice > 0)[0]
    fr = np.where(ice_ref > 0)[0]
    if len(fi) and len(fr):
        freeze = abs(days[fi[0]] - days[fr[0]])
        breakup = abs(days[fi[-1]] - days[fr[-1]])
    elif len(fi) or len(fr):    # only one of the two ever froze
        freeze = breakup = np.inf
    else:                       # neither froze: phenology agrees
        freeze = breakup = 0.0

    print(f"days {days[0]:.2f}..{days[-1]:.2f}  n={n}")
    print(f"pointwise max   {point:.3f} K   (criterion 5.0)")
    print(f"full-mean bias  {bias:.3f} K   (criterion 0.5)")
    print(f"monthly mean    {month:.3f} K   (criterion 1.5)")
    print(f"profile RMSE    {rmse:.3f} K")
    print(f"ice thickness   {ice_err:.3f} m   (criterion 0.3)")
    print(f"snow thickness  {snow_err:.3f} m   (criterion 0.3)")
    print(f"freeze/break    {freeze:.2f} / {breakup:.2f} days (criterion 7)")
    ok = (point <= 5.0 and bias <= 0.5 and month <= 1.5 and ice_err <= 0.3
          and snow_err <= 0.3 and max(freeze, breakup) <= 7.0)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
