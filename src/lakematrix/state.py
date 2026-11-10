"""Model state and thermodynamic initialisation (`INIT_VAR`).

State fields are `(lake, ...)` arrays.  Shapes: water `(M+1)`, ice
`(Mice+1)`, soil `(ns, nsoilcols)`; scalars `(lake,)`.

`init_state` copies the thermodynamic part of `init_var_mod.f90:7-500`.  The
soil side columns use a Fortran expression with default-real literals
(`-10.*float(i-1)/float(ns-1)`), i.e. **single precision** before `exp`; the
same expression is reproduced here, because a float64-only version differs by
about 5e-7 K (see NOTES.md).
"""

from dataclasses import dataclass

import numpy as np

from .constants import C
from .grid import Grid, interp_profile, zsoilcols


@dataclass
class State:
    Tw1: np.ndarray        # (L, M+1)
    Ti1: np.ndarray        # (L, Mice+1)
    Tis1: np.ndarray       # (L, Mice+1)
    Sal1: np.ndarray       # (L, M+1)
    Tsoil1: np.ndarray     # (L, ns, nsoilcols)
    Sals1: np.ndarray      # (L, ns, nsoilcols)
    wi1: np.ndarray        # (L, ns, nsoilcols)
    wl1: np.ndarray        # (L, ns, nsoilcols)
    h1: np.ndarray         # (L,)
    l1: np.ndarray
    hs1: np.ndarray
    ls1: np.ndarray
    snowmass: np.ndarray
    snmelt: np.ndarray
    E: np.ndarray          # (L, M+1)
    eps: np.ndarray
    lamw: np.ndarray       # (L, M)
    time: np.ndarray       # (L,)
    nstep: np.ndarray      # (L,) int
    layer_case: np.ndarray  # (L,) int


def _empty(params, grid):
    L, M, Mice, ns, nc = params.L, params.M, params.Mice, params.ns, params.nsoilcols
    z = lambda *shape: np.zeros((L,) + shape, dtype=float)
    return dict(
        Tw1=z(M + 1), Ti1=z(Mice + 1), Tis1=z(Mice + 1), Sal1=z(M + 1),
        Tsoil1=z(ns, nc), Sals1=z(ns, nc), wi1=z(ns, nc), wl1=z(ns, nc),
        h1=np.zeros(L), l1=np.zeros(L), hs1=np.zeros(L), ls1=np.zeros(L),
        snowmass=np.zeros(L), snmelt=np.zeros(L),
        E=z(M + 1), eps=z(M + 1), lamw=z(M),
        time=np.zeros(L), nstep=np.zeros(L, dtype=np.int64),
        layer_case=np.ones(L, dtype=np.int64),
    )


def init_state(params, grid: Grid):
    """Thermodynamic part of `INIT_VAR` for all lakes in `params`."""
    L = params.L
    s = _empty(params, grid)
    s["h1"] = params.h10.copy()
    s["l1"] = params.l10.copy()
    s["hs1"] = params.hs10.copy()
    s["ls1"] = params.ls10.copy()

    # Water temperature and salinity: LININTERPOL of the T_profile.
    nT = int(params.T_profile_len[0])
    prof = params.T_profile[0, :nT, :]
    Tprof, Sprof = prof[:, 1], prof[:, 2]
    z1 = prof[:, 0]
    z_full = grid.dzeta_int[None, :] * s["h1"][:, None]
    for l in range(L):
        s["Tw1"][l] = interp_profile(z1, Tprof, z_full[l][None, :])[0]
        s["Sal1"][l] = interp_profile(z1, Sprof, z_full[l][None, :])[0]

    # Soil temperature: deepest column from T_soilprofile, side columns from
    # the water profile at the mid-depth of the soil column (INIT_VAR 251-295).
    nS = int(params.T_soilprofile_len[0])
    tsoil_prof = params.T_soilprofile[0, :nS, :]
    for l in range(L):
        s["Tsoil1"][l, :, -1] = interp_profile(
            tsoil_prof[:, 0], tsoil_prof[:, 1], grid.zsoil[None, :])[0]
        if params.nsoilcols == 1:
            continue
        morph = params.morphometry[0, :int(params.morphometry_len[0]), :]
        zsc = zsoilcols(s["h1"][l], morph, grid, params.nsoilcols)
        if zsc is None:
            continue
        for j in range(params.nsoilcols - 1):
            work = 0.5 * (zsc[j] + zsc[j + 1])
            k = int(np.argmin(np.abs(z_full[l] - work)))
            tw = s["Tw1"][l, k]
            for i in range(params.ns):
                # Fortran: -10.*float(i-1)/float(ns-1), single precision;
                # exp() of a default-real argument is single precision too.
                arg = np.float32(-10.0) * np.float32(i) / np.float32(params.ns - 1)
                e = np.float32(np.exp(arg))
                s["Tsoil1"][l, i, j] = tw + (params.Tbb0[l] - tw) * np.float64(
                    np.float32(1.0) - e)

    # Salinity in soil: INIT_VAR assumes the same salinity as the bottom water.
    # The soil density/porosity are set in COMSOILFORLAKE; v1 keeps the field
    # in State and fills it when the soil properties are ported (M3).
    s["Sals1"][:] = 0.0

    # TKE/dissipation and background diffusivity (INIT_VAR 169-176, 475-480).
    s["E"][:, :params.M] = 10.0 ** (-5.5)
    s["eps"][:, :params.M] = 1.0e-9
    s["lamw"][:] = 1.0e3

    s["time"][:] = 0.0
    s["nstep"][:] = 0
    s["layer_case"][:] = 1
    return State(**s)
