"""Atmospheric forcing: read the LAKE ASCII tables and interpolate per step.

The driver reads an hourly record at each integer hour and linearly
interpolates between the current and the next record (`driver.f90:1306-1396`).
The model step `nstep` uses the forcing at `t = (nstep + 1) * dt`; this offset
was verified against the probe (`tempair` at steps 1, 2, 3 and 180).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Column order of the classic ASCII forcing file (driver N_* keywords).
COLUMNS = ("tempair", "pressure", "longwave", "shortwave", "uwind", "vwind", "qair", "precip")


@dataclass(frozen=True)
class Forcing:
    rows: np.ndarray       # (T, 8)
    interval: float = 1.0  # hours between records


def read_ascii(path, n_header=0):
    lines = Path(path).read_text(errors="replace").splitlines()[n_header:]
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.replace(",", " ").split()[:8]])
    return np.array(rows, dtype=float)


def at_time(rows, time_s, interval=1.0, hour_sec=3600.0):
    """Interpolate the 8 forcing columns at `time_s` seconds (driver rule)."""
    span = hour_sec * interval
    ti = int(time_s / span)
    ti = min(max(ti, 0), len(rows) - 2)
    t_old = ti * span
    frac = (time_s - t_old) / span
    return rows[ti] + frac * (rows[ti + 1] - rows[ti])


def at_step(rows, nstep, dt, interval=1.0, hour_sec=3600.0):
    """Forcing used by model step `nstep` (1-based).

    The driver's clock is one `dt` ahead of the state clock that `LAKE` sees,
    so step 1 uses `t = 2*dt`.
    """
    return at_time(rows, (nstep + 1) * dt, interval, hour_sec)


def read_cases(paths, n_header=0):
    """Read several forcing files; returns a list of (T_i, 8) arrays."""
    return [read_ascii(p, n_header) for p in paths]
