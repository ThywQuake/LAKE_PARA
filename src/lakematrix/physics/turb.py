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


def _bound_from_below(x, xmin):
    sign = 1.0 if x >= 0.0 else -1.0
    return max(abs(x), xmin) * sign


def turb_scales(row, ddz, ddz2, dzeta_int, dzeta_05int, h1, M, tw2, sal2,
                u2, v2, lamw, lamsal, k_turb_T_flux, T_massflux,
                eflux0_kinem, turb_density_flux, Buoyancy0, tau, kor,
                rad_integr, area_int, Lx, Ly, vol, z_full, ddz05,
                DDENS_DTEMP0, DDENS_DSAL0):
    """`TURB_SCALES` (phys_func.f90:914-1072), 0-based arrays.

    `rad_integr` is the 0-based `RadWater%integr` (index 0..M+1).
    """
    from .density import ddens_dsal, ddens_dtemp

    i_maxN, H, maxN = mixed_layer_calc(row, ddz, dzeta_05int, h1, M)
    g_d_k = C.g / C.Kelvin0
    w_conv_scale = max((Buoyancy0 - (rad_integr[1] - rad_integr[i_maxN])
                        * g_d_k / C.cw_m_row0) * H, 0.0) ** (1.0 / 3.0)
    T_conv_scale = -eflux0_kinem / (w_conv_scale + C.small_value)

    Wedderburn = max(min(C.g * (row[M] - row[0]) * H * H
                         / ((tau + C.small_value) * max(Lx[0], Ly[0])),
                         1.0e2), 1.0e-2)

    zm = x1 = zv = x2 = 0.0
    for i in range(M + 1):
        y = ddz05[i] * h1 * area_int[i]
        x = y * z_full[i]
        zm += row[i] * x
        x1 += row[i] * y
        zv += x
        x2 += y
    zm /= x1
    zv /= x2
    LakeNumber = ((zm - zv) * vol * C.row0 * C.g * 2.0 * H
                  / (zv * (tau + C.small_value) * area_int[0] * max(Lx[0], Ly[0])))
    LakeNumber = max(min(LakeNumber, 1.0e2), 1.0e-2)
    Rossby_rad = math.sqrt(C.g * max(row[M] - row[0], C.small_value) / C.row0 * H) / max(kor, C.small_value)

    # Thermocline thickness
    x1, x2 = 0.0, h1
    u1 = u2t = v1 = v2t = Sal1 = Sal2t = 0.0
    for i in range(M):
        if tw2[i] >= 14.0 and tw2[i + 1] < 14.0:
            y = ddz[i] * h1
            x1 = z_full[i] + (tw2[i] - 14.0) * y / (tw2[i] - tw2[i + 1])
            x = (x1 - z_full[i]) / y
            u1 = u2[i] + x * (u2[i + 1] - u2[i])
            v1 = v2[i] + x * (v2[i + 1] - v2[i])
            Sal1 = sal2[i] + x * (sal2[i + 1] - sal2[i])
        if tw2[i] >= 8.0 and tw2[i + 1] < 8.0:
            y = ddz[i] * h1
            x2 = z_full[i] + (tw2[i] - 8.0) * y / (tw2[i] - tw2[i + 1])
            x = (x2 - z_full[i]) / y
            u2t = u2[i] + x * (u2[i + 1] - u2[i])
            v2t = v2[i] + x * (v2[i + 1] - v2[i])
            Sal2t = sal2[i] + x * (sal2[i + 1] - sal2[i])
    ThermThick = x2 - x1
    ReTherm = math.sqrt((u2t - u1) ** 2 + (v2t - v1) ** 2) * ThermThick / C.niu_wat
    RiTherm = (C.g / C.row0 * (DDENS_DTEMP0 * (8.0 - 14.0) + DDENS_DSAL0 * (Sal2t - Sal1))
               * ThermThick / ((u2t - u1) ** 2 + (v2t - v1) ** 2 + C.small_value))

    Rp = np.zeros(M)
    Rpdens = np.zeros(M)
    for i in range(M):
        Tm = 0.5 * (tw2[i + 1] + tw2[i])
        Sm = 0.5 * (sal2[i + 1] + sal2[i])
        x1 = ddens_dtemp(Tm, Sm)
        x2 = ddens_dsal(Tm, Sm)
        x = _bound_from_below(tw2[i + 1] - tw2[i], C.small_value)
        Rp[i] = -x2 / x1 * (sal2[i + 1] - sal2[i]) / x
        Rpdens[i] = lamsal[i] / lamw[i] * Rp[i]

    return dict(i_maxN=i_maxN, H_mixed_layer=H, maxN=maxN,
                w_conv_scale=w_conv_scale, T_conv_scale=T_conv_scale,
                Wedderburn=Wedderburn, LakeNumber=LakeNumber,
                Rossby_rad=Rossby_rad, ThermThick=ThermThick,
                ReTherm=ReTherm, RiTherm=RiTherm, Rp=Rp, Rpdens=Rpdens)
