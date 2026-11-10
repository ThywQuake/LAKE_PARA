"""Compare a lakematrix NetCDF profile with a Fortran `water_temp` time series."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def read_water_temp(path):
    lines = Path(path).read_text().splitlines()
    rows = []
    for line in lines[6:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        t = float(parts[4])
        depths, vals = [], []
        for i in range(5, len(parts) - 1, 2):
            d = float(parts[i])
            if d < 0:
                break
            depths.append(d)
            vals.append(float(parts[i + 1]))
        rows.append((t, np.array(depths), np.array(vals)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("nc")
    p.add_argument("ref")
    p.add_argument("--day", type=float, action="append")
    a = p.parse_args()
    ds = xr.open_dataset(a.nc)
    rows = read_water_temp(a.ref)
    days = a.day or [float(ds.time[-1])]
    for day in days:
        row = min(rows, key=lambda r: abs(r[0] - day))
        t = float(ds.time[-1]) if day is None else day
        idx = int(np.argmin(np.abs(ds.time.values - t)))
        z = ds.z.values
        tw = ds.water_temp.values[idx]
        mask = (row[2] > -999.0) & (row[1] <= z.max())
        err = np.interp(row[1][mask], z, tw) - row[2][mask]
        print(f"day {row[0]:.4f} max {np.max(np.abs(err)):.6f} K")
        for d, e in zip(row[1][mask], err):
            print(f"  {d:5.1f} m  {e:+.6f} K")


if __name__ == "__main__":
    main()
