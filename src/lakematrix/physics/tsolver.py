"""Water-column / soil temperature diffusion (`DIFF_COEF` + `T_DIFF`).

Fortran: `t_solver_mod.f90:219-937`.  This module ports the case-1 path used
by `T_SOLVER`: a Dirichlet surface temperature, a water column, a water-soil
interface, and the soil column.  Ice/snow/deep-ice branches are added in M4.

Internally the arrays are 1-based (index 0 unused), exactly like Fortran, to
keep the coefficient indexing readable.
"""

import numpy as np

LAST_COEFS = {}

from ..constants import C


def _pad1(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,))
    out[..., 1:] = x
    return out


def _progonka_1based(a, b, c, d, n):
    from .tridiag import progonka
    y = progonka(a[1:n + 1], b[1:n + 1], c[1:n + 1], d[1:n + 1])
    out = np.zeros(n + 1)
    out[1:] = y
    return out


def diff_coef_soil(a, b, c, d, n0, n1, m0, m1, dt, lamsoil, csoil, rosoil,
                   dzs, Tsoil1, isoilcol):
    dt_inv = 1.0 / dt
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        lam2 = 0.5 * (lamsoil[j] + lamsoil[j + 1])
        lam1 = 0.5 * (lamsoil[j - 1] + lamsoil[j])
        dzmean = 0.5 * (dzs[j - 1] + dzs[j])
        ai = -0.5 * lam1 / dzs[j - 1]
        bi = -0.5 * lam2 / dzs[j]
        a[i] = ai
        b[i] = bi
        c[i] = ai + bi - dt_inv * (csoil[j] * rosoil[j] * dzmean)
        d[i] = (-Tsoil1[j, isoilcol] * dt_inv * (csoil[j] * rosoil[j] * dzmean)
                + ai * Tsoil1[j - 1, isoilcol]
                - (ai + bi) * Tsoil1[j, isoilcol]
                + bi * Tsoil1[j + 1, isoilcol])


def diff_coef_water(a, b, c, d, n0, n1, m0, m1, dt, Tw1, lamw, area_int,
                    area_half, ddz, h1, dhw, dhw0, PEMF, pt_down_f,
                    rad_integr, lsh_water, dzeta_int, ddz05):
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    cw = C.cw_m_row0
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        aa = 0.5 * (-area_half[j - 1] * lamw[j - 1] / (ddz[j - 1] * h1 * area_int[j])
                    + cw * dhw * dzeta_int[j] * dt_inv05
                    - cw * dhw0 * dt_inv05)
        a[i] = aa + cw * 0.5 * PEMF[j - 1]
        bb = 0.5 * (-area_half[j] * lamw[j] / (ddz[j] * h1 * area_int[j])
                    - cw * dhw * dzeta_int[j] * dt_inv05
                    + cw * dhw0 * dt_inv05)
        b[i] = bb - cw * 0.5 * PEMF[j]
        c[i] = (aa + bb - dt_inv * ddz05[j - 1] * h1 * cw
                + cw * 0.5 * (PEMF[j] - PEMF[j - 1]))
        d[i] = (-Tw1[j] * dt_inv * ddz05[j - 1] * h1 * cw
                + (area_half[j] * rad_integr[j] - area_half[j - 1] * rad_integr[j - 1]) / area_int[j]
                - cw * PEMF[j - 1] * pt_down_f[j - 1]
                + cw * PEMF[j] * pt_down_f[j]
                + aa * Tw1[j - 1] + bb * Tw1[j + 1]
                - (aa + bb) * Tw1[j]
                - lsh_water[j] * ddz05[j - 1] * h1)


def t_diff_water(Tsurf, dt, Tw1, Tsoil1, lamw, area_int, area_half, h1, ddz,
                 dhw, dhw0, PEMF, pt_down_f, lamsoil, csoil, rosoil, dzs,
                 rad_integr, dzeta_int, ddz05, M, ns, isoilcol,
                 soilbotflx=0.0, lsh_water=None):
    """Case-1 `T_DIFF`: water + soil, no ice/snow.  Returns (Tw2, Tsoil2).

    All arrays are 1-based (index 0 unused), matching Fortran; `rad_integr`
    keeps its Fortran 0-based bound (index = Fortran index).
    """
    nn = M + ns
    a = np.zeros(nn + 1); b = np.zeros(nn + 1)
    c = np.zeros(nn + 1); d = np.zeros(nn + 1)
    if lsh_water is None:
        lsh_water = np.zeros(nn + 1)

    b[1] = 0.0; c[1] = 1.0; d[1] = Tsurf
    diff_coef_water(a, b, c, d, 2, M, 2, M, dt, Tw1, lamw, area_int, area_half,
                    ddz, h1, dhw, dhw0, PEMF, pt_down_f, rad_integr,
                    lsh_water, dzeta_int, ddz05)

    cw = C.cw_m_row0
    dt_inv = 1.0 / dt
    xx = area_int[M + 1] / area_half[M]
    a[M + 1] = (0.5 * (-lamw[M] / (ddz[M] * h1)
                       + xx * 0.5 * dt_inv * cw * (dhw - dhw0))
                + xx * 0.5 * cw * PEMF[M])
    b[M + 1] = -0.25 * (lamsoil[1] + lamsoil[2]) / dzs[1]
    cc = 0.5 * dt_inv * (csoil[1] * rosoil[1] * dzs[1] + xx * cw * ddz[M] * h1)
    c[M + 1] = a[M + 1] + b[M + 1] - cc - xx * cw * PEMF[M]
    d[M + 1] = (-Tw1[M + 1] * cc - rad_integr[M]
                - xx * cw * PEMF[M] * pt_down_f[M]
                + a[M + 1] * Tw1[M]
                - (a[M + 1] + b[M + 1]) * Tw1[M + 1]
                + b[M + 1] * Tsoil1[2, isoilcol])

    diff_coef_soil(a, b, c, d, 2, ns - 1, M + 2, M + ns - 1, dt,
                   lamsoil, csoil, rosoil, dzs, Tsoil1, isoilcol)
    c[M + ns] = 1.0
    a[M + ns] = 1.0
    d[M + ns] = -soilbotflx * dzs[ns - 1] / (lamsoil[ns - 1] + lamsoil[ns])

    Temp = _progonka_1based(a, b, c, d, nn)
    LAST_COEFS.update(a=a.copy(), b=b.copy(), c=c.copy(), d=d.copy(),
                      Temp=Temp.copy())
    return Temp[1:M + 2], Temp[M + 1:M + ns + 1]


def t_solver(fn, nstep, surftyp, surftyp_prev, Tsurf_prev, maxiter=20,
             dt_scan=5.0, resid=1.0, templin=None):
    """Surface-temperature scan + Chorde iteration (`T_SOLVER`).

    `fn(Tsurf)` must run `T_DIFF` for that surface temperature and return
    `(heat_balance, state_after_diff)`.  The returned `state` is the one from
    the final `fn(Tsurf3)` call.  `templin` is only needed if the scan reaches
    80 C (the linearised-heat-balance fallback).
    """
    if nstep > 1 and surftyp == surftyp_prev:
        Tsurf2 = Tsurf_prev - 10.0
    else:
        Tsurf2 = -90.0

    iterate = True
    Bal1 = 1.0
    Bal2 = 1.0
    Tsurf1 = Tsurf2
    state1 = state2 = None
    Tsurf3 = Tsurf2
    state3 = None
    while Bal1 * Bal2 > 0.0 and iterate:
        Tsurf1 = Tsurf2
        Tsurf2 = Tsurf2 + dt_scan
        Bal1, state1 = fn(Tsurf1)
        Bal2, state2 = fn(Tsurf2)
        if Tsurf2 > 80.0:
            if templin is None:
                raise RuntimeError("T_SOLVER scan exceeded 80 C and no templin")
            Tsurf3 = templin(Tsurf_prev)
            state3 = state2
            iterate = False

    it = 0
    Bal3 = resid + 1.0
    if Bal1 != 0.0 and Bal2 != 0.0 and iterate:
        while abs(Bal3) > resid:
            it += 1
            if Bal1 != Bal3:
                Bal1, state1 = fn(Tsurf1)
            if Bal2 != Bal3:
                Bal2, state2 = fn(Tsurf2)
            Tsurf3 = (Tsurf1 * Bal2 - Tsurf2 * Bal1) / (Bal2 - Bal1)
            Bal3, state3 = fn(Tsurf3)
            if Bal1 * Bal3 < 0.0:
                Tsurf2, Bal2 = Tsurf3, Bal3
            elif Bal2 * Bal3 < 0.0:
                Tsurf1, Bal1 = Tsurf3, Bal3
            if it > maxiter:
                break
    elif Bal1 == 0.0:
        Tsurf3, state3 = Tsurf1, state1
    elif Bal2 == 0.0:
        Tsurf3, state3 = Tsurf2, state2

    return dict(Tsurf3=Tsurf3, state=state3, iter=it, Bal3=Bal3)
