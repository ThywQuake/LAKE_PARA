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


from dataclasses import dataclass


@dataclass
class WaterSurfaceContext:
    """Everything `HEATBALANCE` needs for a case-1 water surface.

    Arrays are 1-based (index 0 unused), as in `tsolver.t_diff_water`.
    """
    Tw1: np.ndarray
    Tsoil1: np.ndarray
    lamw: np.ndarray
    area_int: np.ndarray
    area_half: np.ndarray
    h1: float
    ddz: np.ndarray
    dhw: float
    dhw0: float
    PEMF: np.ndarray
    pt_down_f: np.ndarray
    lamsoil: np.ndarray
    csoil: np.ndarray
    rosoil: np.ndarray
    dzs: np.ndarray
    radwater_integr: np.ndarray   # Fortran 0-based
    lsh_water: np.ndarray
    dzeta_int: np.ndarray
    ddz05: np.ndarray
    M: int
    ns: int
    isoilcol: int
    dt: float
    fetch: float
    uwind: float
    vwind: float
    u: float
    v: float
    tempair: float
    pressure: float
    humair: float
    zref: float
    roughness: float
    relhums: float
    aM: float
    bM: float
    l1: float
    longwave: float
    shortwave: float
    netrad: float
    cloud: float
    albedo: float
    albedo_lw: float
    emissivity: float
    soilbotflx: float = 0.0


def heatbalance_water(Tsurf, ctx: WaterSurfaceContext):
    """Run `T_DIFF` then the `HEATBALANCE` composition for a water surface.

    Returns `(HB, state)` where `state` is a dict with `Tw2`, `Tsoil2`,
    `eflux`, `dHdt`, `convect_flux`, `botflux`, `hflux`, `Elatent`,
    `Radbal`, `Radbal_surf`, `eflux0`, `eflux0_kinem`.
    """
    from .tsolver import t_diff_water

    Tw2, Tsoil2 = t_diff_water(
        Tsurf, ctx.dt, ctx.Tw1, ctx.Tsoil1, ctx.lamw, ctx.area_int,
        ctx.area_half, ctx.h1, ctx.ddz, ctx.dhw, ctx.dhw0, ctx.PEMF,
        ctx.pt_down_f, ctx.lamsoil, ctx.csoil, ctx.rosoil, ctx.dzs,
        ctx.radwater_integr, ctx.dzeta_int, ctx.ddz05, ctx.M, ctx.ns,
        ctx.isoilcol, soilbotflx=ctx.soilbotflx, lsh_water=ctx.lsh_water)
    ef, dh, conv, bot = heatfluxsurf_water(
        ctx.Tw1[1:][None, :], Tw2[None, :], ctx.lamw[1:][None, :],
        ctx.area_int[1:][None, :], ctx.area_half[1:][None, :],
        np.array([ctx.h1]), ctx.ddz[1:], np.array([ctx.dhw0]), ctx.dt)
    slm = senslatmom_fluxes(
        Tsurf, ctx.fetch, ctx.uwind, ctx.vwind, ctx.u, ctx.v, ctx.tempair,
        ctx.pressure, ctx.humair, ctx.zref, ctx.roughness,
        relhums=ctx.relhums, aM=ctx.aM, bM=ctx.bM, l1=ctx.l1, h1=ctx.h1)
    rb, rbs, sb, sr, swup = radbalance(
        Tsurf, ctx.radwater_integr, np.zeros(2), WATER,
        longwave=ctx.longwave, shortwave=ctx.shortwave, netrad=ctx.netrad,
        cloud=ctx.cloud, humair=ctx.humair, pressure=ctx.pressure,
        tempair=ctx.tempair, albedo=ctx.albedo, albedo_lw=ctx.albedo_lw,
        emissivity=ctx.emissivity)
    HB, eflux0, eflux0_kinem = heatbalance(
        float(dh[0]), float(rb), float(rbs), float(ef[0]), float(conv[0]),
        slm["hflux"], slm["Elatent"])
    return HB, dict(Tw2=Tw2, Tsoil2=Tsoil2, eflux=float(ef[0]),
                    dHdt=float(dh[0]), convect_flux=float(conv[0]),
                    botflux=float(bot[0]), hflux=slm["hflux"],
                    Elatent=slm["Elatent"], Radbal=float(rb),
                    Radbal_surf=float(rbs), eflux0=eflux0,
                    eflux0_kinem=eflux0_kinem, cdmw=slm["cdmw"])


def heatfluxsurf_ice(Ti1, Ti2, lami_v, ci_m_roi_v, l1, dhi0, ddzi, area_int,
                     area_half, dt):
    """`HEATFLUXSURF` ice branch; returns (eflux, dHdt, convect_flux)."""
    eflux = (-0.5 * lami_v[1] / area_int[1] * area_half[1]
             * (Ti2[2] + Ti1[2] - Ti2[1] - Ti1[1]) / (ddzi[1] * l1))
    dHdt = 0.5 / dt * ci_m_roi_v[1] * (
        ddzi[1] * l1 * (Ti2[1] - Ti1[1])
        + 0.5 * dhi0 * (Ti2[2] + Ti1[2] - Ti2[1] - Ti1[1]))
    return eflux, dHdt, 0.0


def heatfluxsurf_snow(Tsn, T, cs, dens, dz, lams, itop, dt):
    """`HEATFLUXSURF` snow branch; returns (eflux, dHdt, convect_flux)."""
    eflux = (-(lams[itop] + lams[itop + 1]) * 0.25
             * (Tsn[itop + 1] + T[itop + 1] - Tsn[itop] - T[itop]) / dz[itop])
    dHdt = (0.5 / dt * cs[itop] * dens[itop] * dz[itop]
            * (Tsn[itop] - T[itop]))
    return eflux, dHdt, 0.0
