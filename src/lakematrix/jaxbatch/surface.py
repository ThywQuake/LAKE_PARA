"""JAX (batched) open-water surface: SURF_CHAR, RADBALANCE, HEATBALANCE."""

import jax.numpy as jnp

from ..constants import C
from .diffusion import t_diff_water
from .sfcflx import momsenlat, roughness


def sinh0(year, month, day, hour, phi):
    """SINH0 for scalar arrays; year/month/day are static ints."""
    ndaym = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0 and not (year % 100 == 0 and year % 400 != 0):
        ndaym[1] = 29
    nday = sum(ndaym[:month - 1]) + day
    delta = 23.5 * jnp.pi / 180.0 * jnp.cos(2.0 * jnp.pi * (float(nday) - 173.0) / 365.0)
    theta = jnp.pi * (hour - 12.0) / 12.0
    phi_rad = phi * jnp.pi / 180.0
    s = (jnp.sin(phi_rad) * jnp.sin(delta)
         + jnp.cos(phi_rad) * jnp.cos(delta) * jnp.cos(theta))
    return jnp.maximum(s, 0.0)


def surf_char(year, month, day, hour, phi, fetch, wind, velfrict_prev):
    """Open-water ``SURF_CHAR`` (varalb=1)."""
    velfrict_prev = jnp.maximum(velfrict_prev, 0.01)
    c_z0u_fetch, _, z0u, _, _ = roughness(fetch, wind, velfrict_prev)
    albedo = 0.05 / (sinh0(year, month, day, hour, phi) + 0.15)
    return dict(albedo=albedo, albedo_lw=C.albedoofwater_lw,
                roughness=z0u, aM=C.aMagw, bM=C.bMagw,
                emissivity=C.emissivityofwater, relhums=1.0,
                velfrict_prev=velfrict_prev)


def radbalance(Tsurf, radwater_integr, surftyp, *, longwave, shortwave,
               albedo, albedo_lw, emissivity):
    """Water-only ``RADBALANCE`` (skin=0, ifrad=1)."""
    surfrad = emissivity * C.sigma * (Tsurf + C.Kelvin0) ** 4
    shortbal = shortwave * C.sabs + radwater_integr[..., 0] - radwater_integr[..., 1]
    radbal = shortbal + longwave * (1.0 - albedo_lw) - surfrad
    radbal_surf = shortwave * C.sabs + longwave * (1.0 - albedo_lw) - surfrad
    swup = shortwave * albedo * (1.0 - C.sabs)
    return radbal, radbal_surf, shortbal, surfrad, swup


def senslatmom_fluxes(Tsurf, fetch, uwind, vwind, u, v, tempair, pressure,
                      humair, zref, roughness, relhums=1.0,
                      aM=7.6326, bM=241.9, minwind=C.minwind):
    urel, vrel = uwind - u, vwind - v
    wr = jnp.maximum(jnp.sqrt(urel ** 2 + vrel ** 2), minwind)
    qmom, qsen, qlat, qwat = momsenlat(zref, zref, fetch, wr,
                                       tempair + C.Kelvin0, humair,
                                       Tsurf + C.Kelvin0, pressure)
    tau = -qmom
    return dict(hflux=qsen, Elatent=qlat, tau=tau,
                cdmw=tau / (C.roa0 * wr), xlew=qlat, hw=qsen,
                velfrict=jnp.sqrt(jnp.maximum(tau, 0.0) /
                                  (pressure / (C.Rd * (tempair + C.Kelvin0)))),
                wr=wr)


def heatfluxsurf_water(Tw1, Tw2, lamw, area_int, area_half, h1, ddz, dhw0, dt):
    dt_inv = 1.0 / dt
    eflux = (-0.5 * lamw[..., 0] / area_int[..., 0] * area_half[..., 0]
             * (Tw2[..., 1] + Tw1[..., 1] - Tw2[..., 0] - Tw1[..., 0])
             / (ddz[..., 0] * h1))
    dHdt = 0.5 * dt_inv * C.cw_m_row0 * (
        ddz[..., 0] * h1 * (Tw2[..., 0] - Tw1[..., 0])
        + 0.5 * dhw0 * (Tw2[..., 1] + Tw1[..., 1] - Tw2[..., 0] - Tw1[..., 0]))
    convect_flux = jnp.zeros_like(h1)
    botflux = (-0.5 * lamw[..., -1] / area_int[..., -1] * area_half[..., -1]
               * (Tw2[..., -1] + Tw1[..., -1] - Tw2[..., -2] - Tw1[..., -2])
               / (ddz[..., -1] * h1))
    return eflux, dHdt, convect_flux, botflux


def heatbalance(dHdt, Radbal, Radbal_surf, eflux, convect_flux, hflux, Elatent):
    HB = dHdt - (Radbal - hflux - Elatent) + eflux + convect_flux
    rbs = Radbal if Radbal_surf is None else Radbal_surf
    eflux0 = rbs - hflux - Elatent
    return HB, eflux0, eflux0 / C.cw_m_row0


def heatbalance_water(Tsurf, ctx):
    """JAX ``HEATBALANCE`` for a water surface; ctx is a dict of batched arrays."""
    M = ctx["Tw1"].shape[-1] - 2
    ns = ctx["Tsoil1"].shape[-2] - 1
    isoilcol = ctx["Tsoil1"].shape[-1] - 1
    Tw2, Tsoil2 = t_diff_water(
        Tsurf, ctx["dt"], ctx["Tw1"], ctx["Tsoil1"], ctx["lamw"],
        ctx["area_int"], ctx["area_half"], ctx["h1"], ctx["ddz"],
        ctx["dhw"], ctx["dhw0"], ctx["PEMF"], ctx["pt_down_f"],
        ctx["lamsoil"], ctx["csoil"], ctx["rosoil"], ctx["dzs"],
        ctx["radwater_integr"], ctx["dzeta_int"], ctx["ddz05"],
        M, ns, isoilcol, 0.0, ctx.get("lsh_water"))
    eflux, dHdt, conv, botflux = heatfluxsurf_water(
        ctx["Tw1"][..., 1:], Tw2, ctx["lamw"][..., 1:],
        ctx["area_int"][..., 1:], ctx["area_half"][..., 1:],
        ctx["h1"], ctx["ddz"][..., 1:], ctx["dhw0"], ctx["dt"])
    slm = senslatmom_fluxes(
        Tsurf, ctx["fetch"], ctx["uwind"], ctx["vwind"], ctx["u"], ctx["v"],
        ctx["tempair"], ctx["pressure"], ctx["humair"], ctx["zref"],
        ctx["roughness"])
    rb, rbs, sb, sr, swup = radbalance(
        Tsurf, ctx["radwater_integr"], 4, longwave=ctx["longwave"],
        shortwave=ctx["shortwave"], albedo=ctx["albedo"],
        albedo_lw=ctx["albedo_lw"], emissivity=ctx["emissivity"])
    HB, eflux0, eflux0_kinem = heatbalance(
        dHdt, rb, rbs, eflux, conv, slm["hflux"], slm["Elatent"])
    return HB, dict(Tw2=Tw2, Tsoil2=Tsoil2, eflux=eflux, dHdt=dHdt,
                    convect_flux=conv, botflux=botflux,
                    hflux=slm["hflux"], Elatent=slm["Elatent"],
                    Radbal=rb, Radbal_surf=rbs, eflux0=eflux0,
                    eflux0_kinem=eflux0_kinem, cdmw=slm["cdmw"])
