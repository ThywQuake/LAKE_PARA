"""Surface radiation balance (`RADBALANCE`).

Fortran: `heatbalsurf_mod.f90:388-508`.  v1 uses `skin=0`, `ifrad=1`
(longwave is provided).  The surface albedo/emissivity are inputs; they are
produced by the surface-characteristics and radiation code (`SURF_CHAR`,
`PBLDAT_*`) which is ported in M3.
"""

import numpy as np

from ..constants import C

SOIL = 1
ICE = 2
SNOW = 3
WATER = 4


def radbalance(Tsurf, radwater_integr, radice_integr, surftyp, *,
               longwave, shortwave, netrad, cloud, humair, pressure,
               tempair, albedo, albedo_lw, emissivity,
               ifrad=1, skin=0, sabs=C.sabs, sigma=C.sigma,
               kelvin0=C.Kelvin0):
    """Return (Radbal, Radbal_surf, ShortBal_toplayer, surfrad, swup)."""
    Tsurf = np.asarray(Tsurf, dtype=float)
    surfrad = emissivity * sigma * (Tsurf + kelvin0) ** 4 if ifrad == 1 else np.zeros_like(Tsurf)
    longwave1 = longwave
    radbal_surf = None

    if surftyp == SOIL:
        radbal = shortwave * (1.0 - albedo) + longwave1 * (1.0 - albedo_lw) - surfrad
        shortbal = shortwave * (1.0 - albedo)
        swup = shortwave * (1.0 - albedo)
    elif surftyp == ICE:
        shortbal = radice_integr[0] - radice_integr[1]
        radbal = shortbal + longwave1 * (1.0 - albedo_lw) - surfrad
        swup = shortwave * (1.0 - albedo)
    elif surftyp == SNOW:
        radbal = longwave1 * (1.0 - albedo_lw) - surfrad
        shortbal = 0.0
        swup = shortwave * (1.0 - albedo)
    elif surftyp == WATER:
        shortbal = shortwave * sabs + radwater_integr[0] - radwater_integr[1]
        radbal = shortbal + longwave1 * (1.0 - albedo_lw) - surfrad
        if skin == 1:
            radbal = shortwave * sabs + longwave1 * (1.0 - albedo_lw) - surfrad
            radbal_surf = None
            swup = shortwave * albedo * (1.0 - sabs)
        else:
            radbal_surf = shortwave * sabs + longwave1 * (1.0 - albedo_lw) - surfrad
            swup = shortwave * albedo * (1.0 - sabs)
    else:
        raise ValueError(f"unknown surftyp {surftyp}")
    return radbal, radbal_surf, shortbal, surfrad, swup


def heatfluxsurf_water(Tw1, Tw2, lamw, area_int, area_half, h1, ddz, dhw0, dt):
    """Water branches of `HEATFLUXSURF` (heatbalsurf_mod.f90:75-170).

    Returns (eflux, dHdt, convect_flux, botflux).  Shapes are (L, ...) except
    `ddz` (M,), `h1`/`dhw0` (L,).
    """
    Tw1 = np.asarray(Tw1); Tw2 = np.asarray(Tw2)
    lamw = np.asarray(lamw); area_int = np.asarray(area_int); area_half = np.asarray(area_half)
    h1 = np.asarray(h1); ddz = np.asarray(ddz); dhw0 = np.asarray(dhw0)
    dt_inv = 1.0 / dt
    eflux = (-0.5 * lamw[:, 0] / area_int[:, 0] * area_half[:, 0]
             * (Tw2[:, 1] + Tw1[:, 1] - Tw2[:, 0] - Tw1[:, 0]) / (ddz[0] * h1))
    dHdt = 0.5 * dt_inv * C.cw_m_row0 * (
        ddz[0] * h1 * (Tw2[:, 0] - Tw1[:, 0])
        + 0.5 * dhw0 * (Tw2[:, 1] + Tw1[:, 1] - Tw2[:, 0] - Tw1[:, 0]))
    convect_flux = np.zeros_like(h1)
    botflux = (-0.5 * lamw[:, -1] / area_int[:, -1] * area_half[:, -1]
               * (Tw2[:, -1] + Tw1[:, -1] - Tw2[:, -2] - Tw1[:, -2]) / (ddz[-1] * h1))
    return eflux, dHdt, convect_flux, botflux


def senslatmom_fluxes(Tsurf, fetch, uwind, vwind, u, v, tempair, pressure,
                      humair, zref, roughness, relhums=1.0,
                      aM=7.6326, bM=241.9, l1=0.0, h1=0.0,
                      relwind=1, PBLpar=3, minwind=C.minwind):
    """PBLpar=3 path of `SENSLATMOM_FLUXES` (heatbalsurf_mod.f90:167-384)."""
    import math

    from .sfcflx import SfcFlx_momsenlat

    if relwind == 1:
        urel, vrel = uwind - u, vwind - v
    else:
        urel, vrel = uwind, vwind
    wr = max(math.sqrt(urel ** 2 + vrel ** 2), minwind)
    wind10 = wr * math.log(10.0 / roughness) / math.log(zref / roughness)
    TET2 = (tempair + C.Kelvin0) * (C.pref / pressure) ** (C.Rd / C.cp)
    TET1 = (Tsurf + C.Kelvin0) * (C.pref / pressure) ** (C.Rd / C.cp)
    ro = pressure / (C.Rd * (tempair + C.Kelvin0))
    esatsurf = C.esatsurf0 * 10.0 ** (aM * Tsurf / (bM + Tsurf))
    humsurf = (C.Rd / C.Rwv) / pressure * esatsurf * relhums

    if PBLpar != 3:
        raise NotImplementedError(f"PBLpar={PBLpar} is not ported; v1 supports 3")
    xx1, yy, zz, xx = SfcFlx_momsenlat(zref, zref, fetch, wr, tempair + C.Kelvin0,
                                       humair, Tsurf + C.Kelvin0, pressure, l1)
    hflux, Elatent, tau = yy, zz, -xx1
    cdmw = tau / (C.roa0 * wr)
    xlew = Elatent
    hw = hflux
    velfrict = math.sqrt(tau / ro)
    return dict(hflux=hflux, Elatent=Elatent, tau=tau, cdmw=cdmw, xlew=xlew,
                velfrict=velfrict, wind10=wind10, ro=ro, humsurf=humsurf,
                wr=wr, TET2=TET2, TET1=TET1)


def heatbalance(dHdt, Radbal, Radbal_surf, eflux, convect_flux, hflux, Elatent):
    """Composition in `HEATBALANCE` (heatbalsurf_mod.f90:8-72, skin=0).

    The individual terms come from `heatfluxsurf`, `senslatmom_fluxes` and
    `radbalance`; this function only applies the Fortran sign convention.
    """
    HB = dHdt - (Radbal - hflux - Elatent) + eflux + convect_flux
    eflux0 = Radbal_surf - hflux - Elatent
    eflux0_kinem = eflux0 / C.cw_m_row0
    return HB, eflux0, eflux0_kinem


def julian_day(year, month, day):
    """`JULIAN_DAY` (time.f90:56-89)."""
    ndaym = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0 and not (year % 100 == 0 and year % 400 != 0):
        ndaym[1] = 29
    return sum(ndaym[:month - 1]) + day


def sinh0(year, month, day, hour, phi):
    """`SINH0` (phys_func.f90:580-613): sine of the solar elevation."""
    nday = julian_day(year, month, day)
    delta = 23.5 * np.pi / 180.0 * np.cos(2.0 * np.pi * (float(nday) - 173.0) / 365.0)
    theta = np.pi * (hour - 12.0) / 12.0
    phi_rad = phi * np.pi / 180.0
    s = (np.sin(phi_rad) * np.sin(delta)
         + np.cos(phi_rad) * np.cos(delta) * np.cos(theta))
    return max(float(s), 0.0)


def water_albedo(sinh):
    """`WATER_ALBEDO` (phys_func.f90:1238-1252)."""
    return 0.05 / (sinh + 0.15)


def _f32(x):
    return float(np.float32(x))


def surf_char(surftyp, year, month, day, hour, phi, fetch, wind,
              velfrict_prev, varalb=1):
    """`SURF_CHAR` (t_solver_mod.f90:940-1034) for all four surface types."""
    from .sfcflx import SfcFlx_roughness

    if surftyp == 1:  # soil
        return dict(albedo=C.albedoofsoil, albedo_lw=C.albedoofsoil_lw,
                    roughness=_f32(0.05), aM=C.aMagw, bM=C.bMagw,
                    emissivity=C.emissivityofsoil, relhums=_f32(0.5),
                    velfrict_prev=velfrict_prev)
    if surftyp == 2:  # ice
        return dict(albedo=C.albedoofice, albedo_lw=C.albedoofice_lw,
                    roughness=_f32(1.0e-5), aM=C.aMagi, bM=C.bMagi,
                    emissivity=C.emissivityofice, relhums=_f32(0.7),
                    velfrict_prev=velfrict_prev)
    if surftyp == 3:  # snow
        return dict(albedo=C.albedoofsnow, albedo_lw=C.albedoofsnow_lw,
                    roughness=_f32(1.0e-3), aM=C.aMagi, bM=C.bMagi,
                    emissivity=C.emissivityofsnow, relhums=_f32(0.7),
                    velfrict_prev=velfrict_prev)
    if surftyp != 4:
        raise ValueError(f"unknown surftyp {surftyp}")

    velfrict_prev = max(velfrict_prev, 0.01)
    roughness = SfcFlx_roughness(fetch, wind, velfrict_prev, 0.0)[2]
    aM, bM = C.aMagw, C.bMagw
    if varalb == 0:
        albedo = C.albedoofwater
    elif varalb == 1:
        albedo = water_albedo(sinh0(year, month, day, hour, phi))
    else:
        raise ValueError(f"unknown varalb {varalb}")
    return dict(albedo=albedo, albedo_lw=C.albedoofwater_lw,
                roughness=roughness, aM=aM, bM=bM,
                emissivity=C.emissivityofwater, relhums=1.0,
                velfrict_prev=velfrict_prev)
