"""Case-2 (ice over water) step, single lake (M4).

The Fortran case is split into 2.1 (snow on ice) and 2.2 (no snow); the call
sequence is the same.  This module implements the no-snow path first, matching
``lake.f90:1613-1877``; the snow branch is added on top of the same sequence.
"""

from types import SimpleNamespace

import numpy as np

from .constants import C
from .physics import surface
from .physics.bathym import bathymetry
from .physics.density import ddens_dtemp, water_dens_ts
from .physics.ice import (ice_increments, ice_increments_nosnow,
                          ice_properties, make_ice_state, snow_calc,
                          snow_cond_heat_coef, sr_botsnow, t_diff_ice)
from .physics.radiation import FRACBANDS, rad_update
from .physics.salinity import s_diff
from .physics.soil import (laterheat, meltpnt, soil_cond_heat_coef,
                           soilcolstemp, soilforlake, water_freeze_melt)
from .physics.surface import (IceSurfaceContext, heatbalance_ice,
                              heatbalance_water, radbalance, surf_char)
from .physics.turb import diffmin_hs, mixed_layer_calc, turb_scales
from .physics.momentum import momentum_solver
from .physics.tsolver import t_solver
from .step import _refresh_geometry


def _pad(x, n):
    out = np.zeros(n + 1)
    out[1:] = np.asarray(x)[:n]
    return out


def sync_ice(ss, params, grid, st, l=0):
    """Fill the persistent `IceState` from the current water/soil state."""
    ice = ss.ice
    if ice is None:
        ice = make_ice_state(
            params.M, params.Mice, params.ns, 100,
            geom=ss.soil_geom, isoilcol=params.nsoilcols,
            nmeltpoint=int(np.atleast_1d(params.nmeltpoint)[0]),
            soilbotflx=float(np.atleast_1d(params.soilbotflx)[0]))
        ss.ice = ice
    ice.ddz = np.array(ss.soil_geom.ddz, copy=True)
    ice.ddz05 = np.array(ss.soil_geom.ddz05, copy=True)
    ice.ddzi = np.array(ss.soil_geom.ddzi, copy=True)
    ice.ddzi05 = np.array(ss.soil_geom.ddzi05, copy=True)
    ice.dzeta_int = np.r_[0.0, grid.dzeta_int]
    ice.dzetai_int = np.r_[0.0, grid.dzetai_int]
    ice.bathymwater = list(ss.soil_geom.bathymwater)
    ice.bathymice = list(ss.soil_geom.bathymice)
    ice.bathymdice = list(ss.soil_geom.bathymdice)
    ice.Ti1[1:] = st.Ti1[l]
    ice.Tis1[1:] = st.Tis1[l]
    ice.Tw1[1:] = st.Tw1[l]
    ice.Tw2[1:] = st.Tw1[l]
    ice.lamw[1:] = st.lamw[l]
    ice.Sal2[1:] = st.Sal1[l]
    ice.h1 = float(st.h1[l])
    ice.l1 = float(st.l1[l])
    ice.ls1 = float(st.ls1[l])
    ice.snowmass = float(st.snowmass[l])
    ice.hs1 = float(st.hs1[l])
    ice.flag_snow = int(np.asarray(st.flag_snow)[l]) if hasattr(st, 'flag_snow') else 0
    ice.dhw = 0.0
    ice.dhw0 = 0.0
    ice.dhi = 0.0
    ice.dhi0 = 0.0
    ice.dls = 0.0
    ice.dls0 = 0.0
    ice.dhwfsoil = float(ss.soil.dhwfsoil)
    ice.dhwtrib = (float(np.atleast_1d(params.h10)[l]) - ice.h1) / 10.0
    ice.PEMF = np.zeros(params.M + 2)
    ice.pt_down_f = np.zeros(params.M + 2)
    ice.Tsoil1 = ss.soil.Tsoil1
    ice.Tsoil2 = ss.soil.Tsoil2
    ice.lamsoil = ss.soil.lamsoil
    ice.csoil = ss.soil.csoil
    ice.rosoil = ss.soil.rosoil
    ice.dzs = ss.soil_params.dzs
    ice.isoilcol = params.nsoilcols
    ice_properties(ice)
    return ice


def _radiation(ice, params, grid, l, Erad, shortwave, Erad_ice=None):
    ex = float(np.atleast_1d(params.extice)[l])
    extinct_ice = np.full((params.Mice, 4), ex)
    if Erad_ice is None:
        Erad_ice = Erad
    ice_flux, ice_int = rad_update(extinct_ice,
                                   grid.ddzi05[:params.Mice] * ice.l1,
                                   np.asarray(FRACBANDS) * Erad_ice)
    ice.rad_integr_ice = np.asarray(ice_int)
    flux0 = np.asarray(ice_flux[-1])
    extinct_water = np.empty((params.M, 4))
    extinct_water[:, 0] = 3.13
    extinct_water[:, 1] = float(np.atleast_1d(params.extwat)[l])
    extinct_water[:, 2] = 1.73
    extinct_water[:, 3] = 1.087e3
    water_flux, water_int = rad_update(extinct_water,
                                       grid.ddz05[:params.M] * ice.h1, flux0)
    ice.rad_integr_water = np.asarray(water_int)
    if ice.ls1 != 0.0:
        dflux, dint = rad_update(extinct_ice,
                                 grid.ddzi05[:params.Mice] * ice.ls1,
                                 np.asarray(water_flux[-1]))
        ice.rad_integr_deepice = np.asarray(dint)
    else:
        ice.rad_integr_deepice = np.zeros(params.Mice + 1)
    return ice_flux, water_flux


def step_case2_to_solver(ss, params, grid, rows, nstep=1, l=0,
                         init_step=False):
    """Case-2 chain through T_SOLVER (no snow)."""
    from .forcing import at_time

    st = ss.state
    dt = params.dt
    _refresh_geometry(ss, params, grid, l, st)
    ice = sync_ice(ss, params, grid, st, l)

    f = at_time(rows, nstep * dt, interval=float(np.atleast_1d(params.interval)[0]))
    tempair, pressure, longwave, shortwave = f[0] - C.Kelvin0, f[1], f[2], f[3]
    uwind, vwind, humair, precip = f[4], f[5], f[6], f[7]
    hour = float(np.atleast_1d(params.hour0)[0]) + nstep * dt / 3600.0
    year = int(np.atleast_1d(params.year0)[0])
    month = int(np.atleast_1d(params.month0)[0])
    day = int(np.atleast_1d(params.day0)[0])
    phi = float(np.atleast_1d(params.phi)[l])
    fetch = float(np.atleast_1d(params.fetch)[l])
    snow = int(ice.flag_snow)
    surftyp = 3 if snow else 2
    sc = surf_char(surftyp, year, month, day, hour, phi, fetch, uwind,
                   1.0e-2, int(np.atleast_1d(params.extwat)[l] * 0 + 1))
    albedo, albedo_lw, emis = sc["albedo"], sc["albedo_lw"], sc["emissivity"]

    row = water_dens_ts(np.r_[0, st.Tw1[l]], np.r_[0, st.Sal1[l]])
    ml = mixed_layer_calc(row, np.r_[0, grid.ddz], grid.dzeta_05int,
                          float(st.h1[l]), params.M)

    Erad = shortwave * (1.0 - albedo)
    if snow:
        snow_cond_heat_coef(ice, tempair)
        Erad_ice = sr_botsnow(ice, Erad)
        inc = ice_increments(ice, dt)
    else:
        Erad_ice = Erad
        inc = ice_increments_nosnow(ice, dt, precip=precip, tempair=tempair,
                                    Elatent=float(ss.Elatent))
    _radiation(ice, params, grid, l, Erad, shortwave, Erad_ice=Erad_ice)
    ice.dhwfsoil = float(ss.soil.dhwfsoil)

    # soil columns at the current water temperature
    ss.soil.Tw1 = np.r_[0, st.Tw1[l]]
    ss.soil.Ti1 = np.r_[0, st.Ti1[l]]
    ss.soil.Tis1 = np.r_[0, st.Tis1[l]]
    ss.soil.h1 = float(st.h1[l])
    ss.soil.l1 = float(st.l1[l])
    ss.soil.ls1 = float(st.ls1[l])
    ss.soil.pressure = pressure
    soil_cond_heat_coef(ss.soil, ss.soil_params, is_=params.nsoilcols)
    ss.soil = soilcolstemp(ss.soil, ss.soil_params, dt)
    lsh = laterheat(ss.soil, ss.soil_params, ss.soil.soilflux, onlywater=False)
    ss.lsh = lsh
    ice.lsh_water = np.asarray(lsh.water)
    ice.lsh_ice = np.asarray(lsh.ice)
    ice.lsh_dice = np.asarray(lsh.dice)

    ctx = IceSurfaceContext(
        ice=ice, snow=snow, water=1, dt=dt, fetch=fetch, uwind=uwind,
        vwind=vwind, u=float(ss.mom.u1[0]), v=float(ss.mom.v1[0]),
        tempair=tempair, pressure=pressure, humair=humair, zref=2.0,
        roughness=sc["roughness"], relhums=sc["relhums"], aM=sc["aM"],
        bM=sc["bM"], l1=float(st.l1[l]), longwave=longwave,
        shortwave=shortwave, netrad=-1.0, cloud=-1.0, albedo=albedo,
        albedo_lw=albedo_lw, emissivity=emis,
        radwater_integr=ice.rad_integr_water,
        radice_integr=ice.rad_integr_ice, isoilcol=params.nsoilcols)

    def fn(t):
        return heatbalance_ice(t, ctx)

    r = t_solver(fn, nstep, surftyp, ss.surftyp_prev, ss.Tsurf_prev)
    ice = t_diff_ice(0, r["Tsurf3"], dt, snow, 1, 1,
                     1 if ice.ls1 != 0.0 else 0, params.nsoilcols, False,
                     r["state"]["ice"])
    ss.Tsurf_prev = r["Tsurf3"]
    ss.surftyp_prev = surftyp
    ss.cdmw = float(r["state"].get("cdmw", ss.cdmw))
    return dict(Tsurf3=r["Tsurf3"], Bal3=r["Bal3"], iter=r["iter"],
                state=dict(ice=ice, Elatent=r["state"]["Elatent"],
                           eflux0_kinem=r["state"]["eflux0_kinem"],
                           cdmw=r["state"].get("cdmw")),
                ice=ice, row=row, ml=ml, sc=sc, inc=inc, integ=None,
                uwind=float(uwind), vwind=float(vwind), fetch=fetch,
                snow=snow, Erad=Erad, tempair=float(tempair),
                precip=float(precip))


def step_case2(ss, params, grid, rows, nstep=1, l=0, init_step=False):
    """Full case-2 step (no snow): finish soil/salinity/turbulence/layers."""
    r = step_case2_to_solver(ss, params, grid, rows, nstep, l,
                             init_step=init_step)
    st = ss.state
    dt = params.dt
    M = params.M

    ice = r["ice"]
    Tw2 = np.asarray(ice.Tw2)
    Tsoil2 = np.asarray(ice.Tsoil2)
    ss.soil.Tsoil2 = Tsoil2
    ss.soil = soilforlake(ss.soil, ss.soil_params, dt, is_=params.nsoilcols)

    Sal1 = np.r_[0, st.Sal1[l]]
    lamsal = np.r_[0, (st.lamw[l] - C.lamw0) / C.cw_m_row0 * C.alsal + C.lamsal0]
    dhwe = -float(r["state"]["Elatent"]) / (C.Lwv * C.row0) * dt
    dhwp = float(ice.dhw0) - dhwe
    Sal2 = s_diff(Sal1, dt, ice.dhw, ice.dhw0, dhwp, dhwe, lamsal,
                  np.r_[0, ss.bathym.area_int[l]],
                  np.r_[0, ss.bathym.area_half[l]], np.r_[0, grid.ddz],
                  float(st.h1[l]), np.r_[0, grid.dzeta_int], grid.ddz05, M)

    row2 = water_dens_ts(Tw2, Sal2)

    if r["snow"]:
        ice.T[ice.itop:ice.ms + 1] = ice.Tsn[ice.itop:ice.ms + 1]
        sc_out = snow_calc(ice, dt, ts=float(ice.T[ice.itop]),
                           t2=float(r["tempair"]),
                           snow_sc=float(ice.snowmass),
                           eflux=float(r["state"].get("eflux", 0.0)),
                           Erad=float(r["Erad"]),
                           Elatent=float(r["state"]["Elatent"]),
                           precip=float(r["precip"]), pheat=0.0)
        ice.snowmass = float(sc_out["snow_sc"])
        ice.snowmass_init = ice.snowmass
        ice.hs1 = float(sc_out["hsnow"])
        ice.snmelt = float(sc_out["snmelt"])
        ice.flag_snow_init = 0

    # momentum (after T_SOLVER, as in lake.f90:1546/1811)
    ss.mom_params.uwind = float(r["uwind"])
    ss.mom_params.vwind = float(r["vwind"])
    ss.mom_params.cdmw = float(r["state"].get("cdmw") or 1.0e-3)
    ss.mom_params.fetch = float(r["fetch"])
    ss.mom.Tw1 = st.Tw1[l]
    ss.mom.Sal1 = st.Sal1[l]
    ss.mom.row = water_dens_ts(st.Tw1[l], st.Sal1[l])
    ss.mom.E1 = st.E[l]
    ss.mom.eps1 = st.eps[l]
    ss.mom.h1 = float(st.h1[l])
    ss.mom.l1 = float(st.l1[l])
    ss.mom.ls1 = float(st.ls1[l])
    ss.mom.dhw = ice.dhw
    ss.mom.dhw0 = ice.dhw0
    ss.mom_u1_old = np.array(ss.mom.u1, copy=True)
    ss.mom_v1_old = np.array(ss.mom.v1, copy=True)
    mom_res = momentum_solver(ss.mom, ss.mom_params, dt)
    ss.mom = mom_res.state
    ss.mom.u1 = np.array(mom_res.u2, copy=True)
    ss.mom.v1 = np.array(mom_res.v2, copy=True)

    ke = ss.ke
    ke.e1 = np.r_[0, st.E[l]]
    ke.eps1 = np.r_[0, st.eps[l]]
    u_old = (np.asarray(ss.mom_u1_old) if ss.mom_u1_old is not None
             else np.asarray(ss.mom.u1))
    v_old = (np.asarray(ss.mom_v1_old) if ss.mom_v1_old is not None
             else np.asarray(ss.mom.v1))
    ke.u1 = np.r_[0, u_old]
    ke.v1 = np.r_[0, v_old]
    ke.um = np.zeros(M + 2)
    ke.vm = np.zeros(M + 2)
    ke.u2 = np.r_[0, np.asarray(mom_res.u2)]
    ke.v2 = np.r_[0, np.asarray(mom_res.v2)]
    ke.tw1 = np.r_[0, st.Tw1[l]]
    ke.tw2 = Tw2
    ke.sal1 = Sal1
    ke.sal2 = Sal2
    ke.row = np.r_[0, ss.mom.row] if np.size(ss.mom.row) == M + 1 else np.asarray(r["row"])
    ke.row2 = np.asarray(row2)
    ke.h1 = float(st.h1[l])
    ke.l1 = float(st.l1[l])
    ke.ls1 = float(st.ls1[l])
    ke.dhw = ice.dhw
    ke.dhw0 = ice.dhw0
    ke.vol = float(ss.bathym.vol[l])
    ke.botar = float(ss.bathym.botar[l])
    b0_val = mom_res.b0 if mom_res.b0 is not None else (1.0 if float(st.l1[l]) > 0.0 else 0.0)
    ke.b0 = float(b0_val)
    ke.Buoyancy0 = (C.g / C.row0 * float(ddens_dtemp(st.Tw1[l][0], 0.0))
                    * float(r["state"]["eflux0_kinem"]))
    ke.tau_air = float(mom_res.tau_air)
    ke.tau_wav = float(mom_res.tau_wav)
    ke.tau_i = float(mom_res.tau_i or 0.0)
    ke.tau_gr = float(mom_res.tau_gr)
    ke.uwind = float(r["uwind"])
    ke.vwind = float(r["vwind"])
    ke.fetch = float(np.atleast_1d(params.fetch)[l])
    ke.roughness = float(r["sc"]["roughness"])
    ke.knum = np.zeros(M + 2)
    ke.veg_sw = np.zeros(M + 2)
    from .physics.kepsilon import k_epsilon
    ke = k_epsilon(ke, dt, dt_keps=dt, nstep_keps=1, params=ss.ke_params)
    ss.ke = ke
    lamw_new = C.lamw0 + np.maximum(ke.kt[1:M + 1], 0.0)

    lay = dict(h1=float(st.h1[l]) + ice.dhw,
               l1=float(st.l1[l]) + ice.dhi,
               ls1=float(st.ls1[l]) + ice.dls,
               hs1=float(ice.hs1), snowmass=float(ice.snowmass),
               flag_snow=int(ice.flag_snow), itop=int(ice.itop))
    return dict(Tw1=Tw2[1:], Sal1=Sal2[1:],
                Tsoil1=ss.soil.Tsoil1[1:, 1:], Ti1=ice.Ti2[1:],
                Tis1=ice.Tis2[1:], h1=lay["h1"], l1=lay["l1"],
                ls1=lay["ls1"], hs1=lay["hs1"],
                snowmass=lay["snowmass"],
                E1=ke.e1[1:], eps1=ke.eps1[1:], lamw=lamw_new,
                kt=ke.kt[1:],
                Elatent=float(r["state"]["Elatent"]),
                dhwfsoil=float(ss.soil.dhwfsoil), dhw=ice.dhw,
                dhw0=ice.dhw0, Tsurf3=r["Tsurf3"], Bal3=r["Bal3"],
                iter=r["iter"], ice=ice, flag_snow=int(ice.flag_snow),
                itop=int(ice.itop))


def run_case2(params, grid, rows, nsteps=180, l=0, callback=None,
              store=True):
    """Run `nsteps` case-2 steps from the current state (no transitions)."""
    ss = build_case2_state(params, grid)
    st = ss.state
    dt = params.dt
    out = []
    for n in range(1, nsteps + 1):
        r = step_case2(ss, params, grid, rows, n, l, init_step=(n == 1))
        st.Tw1[l] = r["Tw1"]; st.Sal1[l] = r["Sal1"]
        st.Tsoil1[l] = r["Tsoil1"]; st.Ti1[l] = r["Ti1"]
        st.Tis1[l] = r["Tis1"]; st.h1[l] = r["h1"]
        st.l1[l] = r["l1"]; st.ls1[l] = r["ls1"]
        st.hs1[l] = r["hs1"]; st.snowmass[l] = r["snowmass"]
        st.flag_snow[l] = r["flag_snow"]; st.itop[l] = r["itop"]
        st.E[l] = r["E1"]; st.eps[l] = r["eps1"]; st.lamw[l] = r["lamw"]
        st.time[l] += dt; st.nstep[l] = n
        ss.Elatent = r["Elatent"]
        ss.soil.dhwfsoil = r["dhwfsoil"]
        if store:
            out.append(r)
        if callback is not None:
            callback(n, ss, r)
    return ss, out


def build_case2_state(params, grid):
    from .step import build_step_state
    ss = build_step_state(params, grid)
    st = ss.state
    # The caller is expected to have filled st.l1/Ti1/Tis1 (probe or a
    # case-1 -> 2 transition); this only allocates the ice storage.
    from .physics.ice import make_ice_state
    ss.ice = make_ice_state(
        params.M, params.Mice, params.ns, 100, geom=ss.soil_geom,
        isoilcol=params.nsoilcols,
        nmeltpoint=int(np.atleast_1d(params.nmeltpoint)[0]),
        soilbotflx=float(np.atleast_1d(params.soilbotflx)[0]))
    ss.Tsurf_prev = -999.0
    ss.surftyp_prev = 0
    return ss


# ---------------------------------------------------------------------------
# Layer-case selection and transitions (lake.f90:881-912, 1246-1281)
# ---------------------------------------------------------------------------

def layer_case(params, ss, nstep, rows, l=0):
    """Return the Fortran ``layer_case`` for the current state and forcing."""
    from .forcing import at_time
    st = ss.state
    h1 = float(st.h1[l])
    l1 = float(st.l1[l])
    Tw1 = np.asarray(st.Tw1[l])
    Ti1 = np.asarray(st.Ti1[l])
    Sal1 = np.asarray(st.Sal1[l])
    pres = np.asarray(ss.bathym.preswat[l])
    nmp = int(np.atleast_1d(params.nmeltpoint)[0])
    lc = 1
    if h1 > 0.0 and l1 > 0.0:
        lc = 2
    elif h1 == 0.0 and l1 > 0.0:
        lc = 3
    elif h1 == 0.0 and l1 == 0.0:
        lc = 4
    if lc == 1:
        mp = meltpnt(float(Sal1[0]), float(pres[0]), nmp)
        if (water_freeze_melt(float(Tw1[0]), 0.5 * float(ss.grid_ddz[0]) * h1,
                              mp, 1)
                and h1 - C.min_ice_thick * C.roi_d_row0 > C.min_water_thick):
            lc = 2
    elif lc == 3:
        mp = meltpnt(0.0, 0.0, nmp)
        if (water_freeze_melt(float(Ti1[-1]), 0.5 * float(ss.grid_ddzi[-1]) * l1,
                              mp, -1)
                and l1 - C.min_water_thick * C.row0_d_roi > C.min_ice_thick):
            lc = 2
    return lc


def create_initial_ice(params, ss, nstep, rows, l=0):
    """Case 2's initial thin ice layer (lake.f90:1246-1263)."""
    from .forcing import at_time
    st = ss.state
    nmp = int(np.atleast_1d(params.nmeltpoint)[0])
    f = at_time(rows, nstep * params.dt,
                interval=float(np.atleast_1d(params.interval)[0]))
    tempair = float(f[0]) - C.Kelvin0
    st.l1[l] = C.min_ice_thick
    st.h1[l] = float(st.h1[l]) - C.min_ice_thick * C.roi_d_row0
    x = meltpnt(float(st.Sal1[l][0]), 0.0, nmp)
    Mice = params.Mice
    if tempair < 0.0:
        st.Ti1[l] = tempair + (np.arange(Mice + 1) / float(Mice)) * (x - tempair)
    else:
        st.Ti1[l] = x - C.T_phase_threshold
    st.Tis1[l] = 0.0
    st.ls1[l] = 0.0
    st.hs1[l] = 0.0
    st.snowmass[l] = 0.0
    st.flag_snow[l] = 0
    st.itop[l] = 0
    ss.ice = None
    ss.surftyp_prev = 0
    ss.Tsurf_prev = -999.0


def step_any(ss, params, grid, rows, nstep, l=0, nstep0=1):
    """One step with the Fortran case selection and 1->2 / 2->1 transitions."""
    from .step import step_case1

    st = ss.state
    ss.grid_ddz = grid.ddz
    ss.grid_ddzi = grid.ddzi
    lc = layer_case(params, ss, nstep, rows, l)
    if lc == 1 and float(st.l1[l]) > 0.0:
        # melt-out: drop the ice/snow state before the open-water step
        st.l1[l] = 0.0
        st.ls1[l] = 0.0
        st.hs1[l] = 0.0
        st.snowmass[l] = 0.0
        st.flag_snow[l] = 0
        st.Ti1[l] = 0.0
        st.Tis1[l] = 0.0
        ss.ice = None
        ss.surftyp_prev = 0
        ss.Tsurf_prev = -999.0
    if lc == 2 and float(st.l1[l]) == 0.0:
        create_initial_ice(params, ss, nstep, rows, l)
    if lc == 1:
        r = step_case1(ss, params, grid, rows, nstep, l,
                       init_step=(nstep == nstep0 and float(st.l1[l]) == 0.0))
        st.Tw1[l] = r["Tw1"]; st.Sal1[l] = r["Sal1"]
        st.Tsoil1[l] = r["Tsoil1"]; st.h1[l] = r["h1"]
        st.E[l] = r["E1"]; st.eps[l] = r["eps1"]; st.lamw[l] = r["lamw"]
        st.l1[l] = 0.0; st.ls1[l] = 0.0; st.hs1[l] = 0.0
        st.snowmass[l] = 0.0; st.flag_snow[l] = 0
        ss.Elatent = r["Elatent"]
        ss.soil.dhwfsoil = r["dhwfsoil"]
        st.layer_case[l] = 1
    elif lc == 2:
        r = step_case2(ss, params, grid, rows, nstep, l)
        st.Tw1[l] = r["Tw1"]; st.Sal1[l] = r["Sal1"]
        st.Tsoil1[l] = r["Tsoil1"]; st.Ti1[l] = r["Ti1"]; st.Tis1[l] = r["Tis1"]
        st.h1[l] = r["h1"]; st.l1[l] = r["l1"]; st.ls1[l] = r["ls1"]
        st.hs1[l] = r["hs1"]; st.snowmass[l] = r["snowmass"]
        st.flag_snow[l] = r["flag_snow"]; st.itop[l] = r["itop"]
        st.E[l] = r["E1"]; st.eps[l] = r["eps1"]; st.lamw[l] = r["lamw"]
        ss.Elatent = r["Elatent"]; ss.soil.dhwfsoil = r["dhwfsoil"]
        st.layer_case[l] = 2
        if st.l1[l] <= 0.0:
            st.l1[l] = 0.0; st.ls1[l] = 0.0; st.hs1[l] = 0.0
            st.snowmass[l] = 0.0; st.flag_snow[l] = 0
            st.Ti1[l] = 0.0; st.Tis1[l] = 0.0
            ss.ice = None
        if st.h1[l] <= 0.0:
            st.h1[l] = 0.0
            st.layer_case[l] = 3
    else:
        raise NotImplementedError(f"layer_case {lc} is not ported (M4)")
    st.time[l] += params.dt
    st.nstep[l] = nstep
    return r


def run_model(params, grid, rows, nsteps=180, l=0, callback=None, store=True,
              nstep0=1):
    """Run with case selection/transitions (case 1 and case 2 for now)."""
    from .step import build_step_state
    ss = build_step_state(params, grid)
    out = []
    for n in range(nstep0, nstep0 + nsteps):
        r = step_any(ss, params, grid, rows, n, l, nstep0=nstep0)
        if store:
            out.append(r)
        if callback is not None:
            callback(n, ss, r)
    return ss, out
