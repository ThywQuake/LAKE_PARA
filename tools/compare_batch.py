"""Compare a `(lake, time, z)` lakematrix NetCDF with several Fortran cases."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from compare import read_water_temp


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
    return dict(day=a[:, 4] / 24.0, h1=a[:, 5], l1=a[:, 14], hs1=a[:, 15],
                ls1=a[:, 16])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nc")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--day", type=float, default=None)
    a = ap.parse_args()
    ds = xr.open_dataset(a.nc)
    day = a.day if a.day is not None else float(ds.time[-1])
    idx = int(np.argmin(np.abs(ds.time.values - day)))
    for l, case in enumerate(a.cases):
        case = Path(case)
        ref = case / "results" / "test" / "time_series" / "water_temp  1  1.dat"
        rows = read_water_temp(ref)
        row = min(rows, key=lambda r: abs(r[0] - day))
        z = ds.z.values
        tw = ds.water_temp.values[l, idx]
        mask = (row[2] > -999.0) & (row[1] <= z.max())
        err = np.interp(row[1][mask], z, tw) - row[2][mask]
        print(f"{case.name}: day {row[0]:.4f} max |dT| {np.max(np.abs(err)):.6f} K")
        lpath = (case / "results" / "test" / "time_series" /
                 "layers  1  1.dat")
        if lpath.exists():
            layers = read_layers(lpath)
            j = int(np.argmin(np.abs(layers["day"] - day)))
            print(f"  h1 {ds.h_water.values[l, idx]:.6f} / {layers['h1'][j]:.6f}"
                  f" m  ice {ds.h_ice.values[l, idx]:.6f} / {layers['l1'][j]:.6f}"
                  f" m  snow {ds.h_snow.values[l, idx]:.6f} /"
                  f" {layers['hs1'][j]:.6f} m"
                  f"  case {int(ds.layer_case.values[l, idx])}")


if __name__ == "__main__":
    main()
