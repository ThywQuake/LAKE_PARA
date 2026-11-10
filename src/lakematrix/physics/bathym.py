"""Lake bathymetry (`BATHYMETRY`), water part.

Fortran: `bathym_mod.f90:7-382`.  v1 supports the testlake configuration:
`soilcolconjtype = 2` (linear interpolation of the depth-area table),
`lakeform = 2` (rectangle), and a morphometry table.  The soil-column
geometry (`bathymsoil`, `itop`/`ibot`/`icent`) is added with the soil port.

The water surface is placed relative to the deepest table point:
`work1 = depth - max(depth) + h1 + ls1`; the area at model depth `z` is the
table area at `z + max(depth) - h1 - ls1`.
"""

from dataclasses import dataclass

import numpy as np

from ..constants import C
from ..grid import Grid, interp_profile

FORM_ELLIPSE = 1
FORM_RECTANGLE = 2


@dataclass
class Bathym:
    z_full: np.ndarray   # (L, M+1)
    z_half: np.ndarray   # (L, M)
    preswat: np.ndarray  # (L, M+1)
    area_int: np.ndarray  # (L, M+1)
    area_half: np.ndarray  # (L, M)
    vol: np.ndarray       # (L,)
    botar: np.ndarray     # (L,)
    Lx: np.ndarray        # (L, M+1)
    Ly: np.ndarray
    rad_int: np.ndarray
    Lx_half: np.ndarray   # (L, M)
    Ly_half: np.ndarray


def _lxly(area, lakeform, cellipt):
    if lakeform == FORM_ELLIPSE:
        b = np.sqrt(area / (C.pi * cellipt))
        a = b * cellipt
        return 2.0 * a, 2.0 * b
    if lakeform == FORM_RECTANGLE:
        ly = np.sqrt(area / cellipt)
        return cellipt * ly, ly
    raise NotImplementedError(f"lakeform={lakeform} is not supported")


def bathymetry(params, grid: Grid, state, pressure):
    """Port of the water part of `BATHYMETRY` for all lakes."""
    L, M = params.L, params.M
    h1 = np.asarray(state.h1)
    ls1 = np.asarray(state.ls1)
    z_full = grid.dzeta_int[None, :] * h1[:, None]
    z_half = grid.dzeta_05int[None, :] * h1[:, None]
    preswat = pressure + C.row0 * C.g * z_full

    area_int = np.zeros((L, M + 1))
    area_half = np.zeros((L, M))
    for l in range(L):
        n = int(params.morphometry_len[l])
        if n == 0 or params.morphometry[0, 0, 1] < 0.0:
            area_int[l] = params.area_lake[l]
            area_half[l] = params.area_lake[l]
            continue
        table = params.morphometry[l, :n, :]
        dep = table[:, 0]
        area = table[:, 1]
        work1 = dep - np.max(dep) + h1[l] + ls1[l]
        area_int[l] = interp_profile(work1, area, z_full[l][None, :])[0]
        area_half[l] = interp_profile(work1, area, z_half[l][None, :])[0]

    botar = area_int[:, 0].copy()
    vol = np.sum(area_int * h1[:, None] * grid.ddz05[None, :M + 1], axis=1)

    Lx = np.zeros((L, M + 1))
    Ly = np.zeros((L, M + 1))
    Lx_half = np.zeros((L, M))
    Ly_half = np.zeros((L, M))
    for l in range(L):
        Lx[l], Ly[l] = _lxly(area_int[l], int(params.lakeform[l]), params.cellipt[l])
        Lx_half[l], Ly_half[l] = _lxly(area_half[l], int(params.lakeform[l]), params.cellipt[l])
    rad_int = np.sqrt(area_int / C.pi)
    return Bathym(z_full=z_full, z_half=z_half, preswat=preswat,
                  area_int=area_int, area_half=area_half, vol=vol, botar=botar,
                  Lx=Lx, Ly=Ly, rad_int=rad_int, Lx_half=Lx_half, Ly_half=Ly_half)
