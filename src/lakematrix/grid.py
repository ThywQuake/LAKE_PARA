"""Dimensionless vertical grid and static bathymetry helpers.

Fortran source: `GRID_CREATE` (`lakebatch/core/source/model/init.f90:4-65`),
`DZETA`/`DZETAI` (`dzeta_mod.f90`), the soil grid in `SOIL_COND_HEAT_COEF`
(`soil_mod.f90:71-90`), and the `zsoilcols` part of `INIT_VAR`
(`init_var_mod.f90:200-211`).

Conventions: arrays are 0-based.  `ddz05` and `ddzi05` keep the Fortran
0-based lower bound as index 0, so `ddz05[k]` is Fortran `ddz05(k)`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    M: int
    Mice: int
    ns: int
    ddz: np.ndarray        # (M,)      Fortran ddz(1:M)
    ddz05: np.ndarray      # (M+1,)    Fortran ddz05(0:M)
    dzeta_int: np.ndarray  # (M+1,)    Fortran dzeta_int(1:M+1)
    dzeta_05int: np.ndarray  # (M,)    Fortran dzeta_05int(1:M)
    ddzi: np.ndarray       # (Mice,)
    ddzi05: np.ndarray     # (Mice+1,)
    dzetai_int: np.ndarray  # (Mice+1,)
    zsoil: np.ndarray      # (ns,)
    dzs: np.ndarray        # (ns-1,)
    dzss: np.ndarray       # (ns,)

    @property
    def ddz_last(self):
        return float(self.ddz[-1])


def create_grid(M, Mice, ns, d_surf=1.0e-2, d_bot=1.0e-2,
                soil_depth=10.0, upper_layer=0.02):
    """Create the static grid, matching `GRID_CREATE` and `SOIL_COND_HEAT_COEF`."""
    # NUMERIC_PARAMS::UpperLayer = 0.02 is a default-real literal stored in
    # a real(8) variable: round it to float32 first.
    upper_layer = float(np.float32(upper_layer))

    # Water grid, Burchard (2002), GRID_CREATE lines 14-22.
    i = np.arange(M, dtype=float)
    denom = np.tanh(d_surf) + np.tanh(d_bot)
    zeta1 = (np.tanh(d_surf) + np.tanh((d_surf + d_bot) * i / M - d_surf)) / denom
    zeta2 = (np.tanh(d_surf) + np.tanh((d_surf + d_bot) * (i + 1) / M - d_surf)) / denom
    ddz = zeta2 - zeta1

    ddz05 = np.empty(M + 1)
    ddz05[0] = 0.5 * ddz[0]
    ddz05[1:M] = 0.5 * (ddz[:-1] + ddz[1:])
    ddz05[M] = 0.5 * ddz[-1]

    dzeta_int = np.concatenate(([0.0], np.cumsum(ddz)))
    dzeta_05int = np.empty(M)
    dzeta_05int[0] = 0.5 * ddz[0]
    for k in range(1, M):
        dzeta_05int[k] = dzeta_05int[k - 1] + 0.5 * (ddz[k - 1] + ddz[k])

    # Ice grid: uniform, GRID_CREATE lines 41-52.
    ddzi = np.full(Mice, 1.0 / Mice)
    ddzi05 = np.empty(Mice + 1)
    ddzi05[0] = 0.5 * ddzi[0]
    ddzi05[1:Mice] = 0.5 * (ddzi[:-1] + ddzi[1:])
    ddzi05[Mice] = 0.5 * ddzi[-1]
    dzetai_int = np.concatenate(([0.0], np.cumsum(ddzi)))

    # Soil grid, SOIL_COND_HEAT_COEF lines 71-90.
    zsoil = np.empty(ns)
    zsoil[0] = 0.0
    if upper_layer > 0.0:
        power = np.log(upper_layer / soil_depth) / np.log(1.0 / float(ns - 1))
        for k in range(1, ns):
            # Fortran: 1.*(i-1)/(ns-1) is a default-real division, rounded
            # to float32 before the (double) power.
            base = float(np.float32(float(k) / float(ns - 1)))
            zsoil[k] = base ** power * soil_depth
    else:
        for k in range(1, ns):
            zsoil[k] = float(k) / float(ns - 1) * soil_depth
    dzs = np.diff(zsoil)
    dzss = np.empty(ns)
    dzss[0] = 0.5 * dzs[0]
    dzss[-1] = 0.5 * dzs[-1]
    dzss[1:-1] = 0.5 * (dzs[:-1] + dzs[1:])

    return Grid(M=M, Mice=Mice, ns=ns, ddz=ddz, ddz05=ddz05,
                dzeta_int=dzeta_int, dzeta_05int=dzeta_05int,
                ddzi=ddzi, ddzi05=ddzi05, dzetai_int=dzetai_int,
                zsoil=zsoil, dzs=dzs, dzss=dzss)


def zsoilcols(h10, depth_area, grid, nsoilcols):
    """Lake depth intervals for the tops of the soil columns.

    `INIT_VAR` lines 200-211.  `depth_area` is the `(ndatamax, 2)` bathymetry
    table (depth, area).  Returns `(nsoilcols+1,)`; the last entry is the
    maximum depth in the table.
    """
    if nsoilcols <= 1 or depth_area is None or depth_area[0, 1] < 0.0:
        return None
    work1 = float(np.max(depth_area[:, 0]))
    work = (work1 - h10 * 0.5 * grid.ddz_last) / float(nsoilcols - 1)
    zsc = np.empty(nsoilcols + 1)
    zsc[:nsoilcols] = np.arange(nsoilcols) * work
    zsc[nsoilcols] = work1
    return zsc


def interp_profile(z1, f1, z2):
    """`LININTERPOL` (`init.f90:68-117`) with extrapolation, vectorised.

    `z1`, `f1`: one profile, shape (N,).  `z2`: shape (L, K).  Below the first
    point the Fortran code extrapolates with the first two points; at or above
    the last point it holds the last value.
    """
    z1 = np.asarray(z1, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    out = np.interp(z2, z1, f1)  # linear inside, f1[-1] above, f1[0] below
    below = z2 < z1[0]
    if np.any(below):
        slope = (f1[1] - f1[0]) / (z1[1] - z1[0])
        out = np.where(below, f1[0] + slope * (z2 - z1[0]), out)
    return out
