"""Case-3 (ice only, no water) step for synthetic verification.

There is no stable Fortran case-3 setup in the reference set (a zero water
depth segfaults ``BATHYMETRY``).  This module is a *self-consistency*
reference: it assembles the existing NumPy kernels following the Fortran
case-3 sequence so the JAX case-3 step can be compared against it.
"""

import numpy as np

from .constants import C
from .physics.bathym import bathymetry
from .physics.ice import (ice_properties, make_ice_state, snow_calc,
                          snow_cond_heat_coef, sr_botsnow, t_diff_ice)
from .physics.radiation import FRACBANDS, rad_update
from .physics.soil import (meltpnt, soil_cond_heat_coef, soilforlake)
from .physics.surface import IceSurfaceContext, heatbalance_ice, surf_char
from .physics.tsolver import t_solver
from .step import build_step_state


def build_case3_state(params, grid, l=0):
    """Start from a normal state and move it to ``h1=0, l1=0.2``."""
    ss = build_step_state(params, grid)
    st = ss.state
    st.h1[l] = 0.0
    st.l1[l] = 0.2
    st.Ti1[l] = np.linspace(-4.0, 0.0, params.Mice + 1)
    st.Tis1[l] = 0.0
    st.Tw1[l] = 0.0
    st.Sal1[l] = 0.0
    ss.ice = make_ice_state(
        params.M, params.Mice, params.ns, 100, geom=ss.soil_geom,
        isoilcol=params.nsoilcols,
        nmeltpoint=int(np.atleast_1d(params.nmeltpoint)[0]),
        soilbotflx=float(np.atleast_1d(params.soilbotflx)[0]))
    ss.ice.l1 = 0.2
    ss.ice.h1 = 0.0
    ss.ice.Ti1[1:] = st.Ti1[l]
    ss.ice.Ti2[1:] = st.Ti1[l]
    ss.ice.Tis1[1:] = 0.0
    ss.ice.Tis2[1:] = 0.0
    ss.ice.Tw1[1:] = 0.0
    ss.ice.lamw[1:] = st.lamw[l]
    ss.ice.Tsoil1 = ss.soil.Tsoil1
    ss.ice.Tsoil2 = ss.soil.Tsoil2
    ss.ice.lamsoil = ss.soil.lamsoil
    ss.ice.csoil = ss.soil.csoil
    ss.ice.rosoil = ss.soil.rosoil
    ss.ice.dzs = ss.soil_params.dzs
    ice_properties(ss.ice)
    return ss


def step_case3(ss, params, grid, rows, nstep=1, l=0):
    """One case-3 step (h1=0, l1>0, optional snow) using NumPy kernels."""
    from .forcing import at_time
    st, ice, soil = ss.state, ss.ice, ss.soil
    dt = params.dt
    f = at_time(rows, nstep * dt, float(np.atleast_1d(params.interval)[0]))
    tempair = f[0] - C.Kelvin0
    pressure, longwave, shortwave = f[1], f[2], f[3]
    uwind, vwind, humair, precip = f[4], f[5], f[6], f[7]
    hour = float(np.atleast_1d(params.hour0)[0]) + nstep * dt / 3600.0
    year, month, day = (int(np.atleast_1d(x)[0]) for x in
                        (params.year0, params.month0, params.day0))
    phi = float(np.atleast_1d(params.phi)[l])
    fetch = float(np.atleast_1d(params.fetch)[l])
    snow = int(st.flag_snow[l])
    surftyp = 3 if snow else 2
    sc = surf_char(surftyp, year, month, day, hour, phi, fetch, uwind, 1e-2)
    albedo, albedo_lw, emis = sc["albedo"], sc["albedo_lw"], sc["emissivity"]
    if snow:
        snow_cond_heat_coef(ice, tempair)
        erad = sr_botsnow(ice, shortwave * (1.0 - albedo))
    else:
        erad = shortwave * (1.0 - albedo)
    ex = float(np.atleast_1d(params.extice)[l])
    extinct = np.full((params.Mice, 4), ex)
    _, integr_i = rad_update(extinct, ss.soil_geom.ddzi05[:params.Mice] * ice.l1,
                             np.asarray(FRACBANDS) * erad)
    ice.rad_integr_ice = np.asarray(integr_i)
    ice.rad_integr_water = np.zeros(params.M + 1)
    ice.rad_integr_deepice = np.zeros(params.Mice + 1)

    x = meltpnt(0.0, 0.0, int(np.atleast_1d(params.nmeltpoint)[0]))
    if ice.Ti1[1] > x + C.T_phase_threshold:
        dhw = ((ice.Ti1[1] - x - C.T_phase_threshold) * ice.ci_m_roi_v[1]
               * ice.l1 * ice.ddzi[1] * 0.5 / C.row0_m_Lwi)
        dhihigh = -dhw * C.roi_d_row0
        ice.Ti1[1] = x + C.T_phase_threshold
        dhis = 0.0
    else:
        dhis = -float(ss.Elatent) / (C.roi * C.Liv) * dt
        dhihigh = 0.0
    if precip > 0.0 and tempair < 0.0:
        ice.snowmass = ice.snowmass + precip * dt * C.row0
        if ice.snowmass / C.rofrsn >= 3.0 * C.dzmin:
            ice.flag_snow = 1
            ice.hs1 = ice.snowmass / C.rofrsn + C.small_value
            ice.itop = ice.ms - 2
        dhip = 0.0
    else:
        dhip = precip * C.row0_d_roi * dt
    ice.dhi = dhis + dhihigh + dhip
    ice.dhi0 = ice.dhi

    soil.Tsoil1 = ice.Tsoil1
    soil.h1 = 0.0
    soil.l1 = ice.l1
    soil.ls1 = 0.0
    soil_cond_heat_coef(soil, ss.soil_params, is_=params.nsoilcols)
    ctx = IceSurfaceContext(
        ice=ice, snow=snow, water=0, dt=dt, fetch=fetch, uwind=uwind,
        vwind=vwind, u=0.0, v=0.0, tempair=tempair, pressure=pressure,
        humair=humair, zref=2.0, roughness=sc["roughness"],
        relhums=sc["relhums"], aM=sc["aM"], bM=sc["bM"],
        l1=ice.l1, longwave=longwave, shortwave=shortwave, netrad=-1.0,
        cloud=-1.0, albedo=albedo, albedo_lw=albedo_lw, emissivity=emis,
        radwater_integr=np.zeros(params.M + 1),
        radice_integr=ice.rad_integr_ice)
    r = t_solver(lambda T: heatbalance_ice(T, ctx), nstep, surftyp,
                 ss.surftyp_prev, ss.Tsurf_prev)
    ice = t_diff_ice(0, r["Tsurf3"], dt, snow, 1, 0, 0,
                     params.nsoilcols, False, ice)
    ss.Tsurf_prev = r["Tsurf3"]
    ss.surftyp_prev = surftyp
    ss.Elatent = r["state"]["Elatent"]
    soil.Tsoil2 = r["state"]["Tsoil2"] if "Tsoil2" in r["state"] else ice.Tsoil2
    soil = soilforlake(soil, ss.soil_params, dt, is_=params.nsoilcols)
    if snow:
        ice.T[ice.itop:ice.ms + 1] = ice.Tsn[ice.itop:ice.ms + 1]
        sc_out = snow_calc(ice, dt, ts=float(ice.T[ice.itop]), t2=tempair,
                           snow_sc=float(ice.snowmass),
                           eflux=float(r["state"]["eflux"]), Erad=erad,
                           Elatent=float(r["state"]["Elatent"]),
                           precip=precip, pheat=0.0)
        ice.snowmass = float(sc_out["snow_sc"])
        ice.hs1 = float(sc_out["hsnow"])
    st.l1[l] = ice.l1 + ice.dhi
    st.h1[l] = 0.0
    st.Ti1[l] = ice.Ti2[1:]
    st.Tis1[l] = ice.Tis2[1:]
    st.Tsoil1[l] = soil.Tsoil1[1:, 1:]
    st.snowmass[l] = ice.snowmass
    st.hs1[l] = ice.hs1
    st.flag_snow[l] = ice.flag_snow
    st.itop[l] = ice.itop
    st.time[l] += dt
    st.nstep[l] = nstep
    return dict(Ti1=st.Ti1[l], Tis1=st.Tis1[l], Tsoil1=st.Tsoil1[l],
                l1=st.l1[l], h1=st.h1[l], hs1=st.hs1[l],
                snowmass=st.snowmass[l], Tsurf3=r["Tsurf3"])
