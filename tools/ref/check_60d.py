"""Check the summer60d npz against the revised 60-day criteria (plan §5)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from compare import read_water_temp  # noqa: E402


def main():
    npz = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "scratch/m3_60d.npz")
    ref = read_water_temp(ROOT / "ref/v1/summer60d/results/test/time_series/"
                          "water_temp  1  1.dat")
    d = np.load(npz)
    z = d["z"]
    days = d["n"] / 4320.0
    E = []
    for k, day in enumerate(days):
        row = min(ref, key=lambda r: abs(r[0] - day))
        mask = (row[2] > -999.0) & (row[1] <= z.max())
        E.append(np.interp(row[1][mask], z, d["Tw1"][k]) - row[2][mask])
    E = np.array(E)
    point = np.max(np.abs(E))
    bias = np.max(np.abs(E.mean(axis=0)))
    rmse = np.max(np.sqrt((E ** 2).mean(axis=0)))
    print(f"days {days[0]:.1f}..{days[-1]:.1f} n={len(days)}")
    print(f"pointwise max {point:.4f} K (<=0.5)")
    print(f"time-mean bias {bias:.4f} K (<=0.1)")
    print(f"profile RMSE {rmse:.4f} K (<=0.15)")
    print("PASS" if point <= 0.5 and bias <= 0.1 and rmse <= 0.15 else "FAIL")


if __name__ == "__main__":
    main()
