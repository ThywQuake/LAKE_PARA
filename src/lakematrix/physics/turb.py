"""Turbulence diagnostics: `MIXED_LAYER_CALC` (and later `TURB_SCALES`).

Fortran: `phys_func.f90:1072-1154` (`MIXED_LAYER_CALC`).
"""

import math

import numpy as np

from ..constants import C


def mixed_layer_calc(row, ddz, dzeta_05int, h, M):
    """Return (i_mixed_layer, H_mixed_layer, maxN), 1-based `i`.

    `row` is the density profile (1-based in Fortran; pass the 1-based array
    or the 0-based array with the same first element ordering).
    """
    row = np.asarray(row, dtype=float)
    ddz = np.asarray(ddz, dtype=float)
    dzeta_05int = np.asarray(dzeta_05int, dtype=float)
    # row[0:M+1] is Fortran row(1:M+1); ddz[0:M] is Fortran ddz(1:M)
    bvf = C.g / C.row0 * (row[1:M + 1] - row[0:M]) / (h * ddz[0:M])
    ml = int(np.argmax(bvf)) + 1
    maxN = math.sqrt(max(float(bvf[ml - 1]), 0.0))
    if (row[M] - row[0]) / C.row0 < 1.0e-3:
        ml = M
    H_mixed_layer = float(dzeta_05int[ml - 1]) * h
    return ml, H_mixed_layer, maxN
