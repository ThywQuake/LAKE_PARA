"""One case-1 (open-water) step, single lake.

This is the integration layer: it wires the validated physics modules in the
order of `lake.f90` case 1.  It is deliberately single-lake and NumPy first;
the batch/JAX layer is M6.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .constants import C
from .physics import surface
from .grid import interp_profile
from .physics.bathym import _lxly, bathymetry, soil_bathymetry
from .physics.density import water_dens_ts
from .physics.kepsilon import KEpsilonParams, KEpsilonState, k_epsilon
from .physics.layers import advance_case1
from .physics.momentum import MomentumParams, MomentumState, momentum_solver
from .physics.radiation import FRACBANDS, rad_update
from .physics.salinity import s_diff
from .physics.soil import (SoilGeometry, bathym_levels, comsoilforlake,
                           laterheat, make_soil_state, soil_cond_heat_coef,
                           soilcolstemp, soilforlake)
from .physics.turb import mixed_layer_calc, turb_scales
from .physics.tsolver import t_solver


@dataclass
class StepState:
    state: object
    soil_params: object
    soil: object
    soil_geom: object
    mom_params: object
    mom: object
    ke_params: object
    ke: object
    integr: np.ndarray
    flux: np.ndarray
    dhw: float = 0.0
    dhw0: float = 0.0
    Elatent: float = 0.0
    eflux0_kinem: float = 0.0
    buo: float = 0.0
    lsh: object = None
    bathym: object = None
    ice: object = None
    lsh: object = None
    Tsurf_prev: float = -999.0
    surftyp_prev: int = 0
    cdmw: float = 1.0e-15
    mom_u1_old: np.ndarray = None
    mom_v1_old: np.ndarray = None
    ke_trace: list = field(default_factory=list)
    trace_ke: bool = False
    ck_soilcol: dict = field(default_factory=dict)
    ck_soilf: dict = field(default_factory=dict)


def _zsoil_geom(params, grid, l, st, bathym):
    h1 = float(st.h1[l])
    table = params.morphometry[l, :int(params.morphometry_len[l]), :]
    from .grid import zsoilcols
    zsc = zsoilcols(h1, table, grid, params.nsoilcols)
    bs = soil_bathymetry(params, grid, st, l)
    if zsc is None or bs is None:
        raise ValueError("case-1 step needs a morphometry table and soil columns")
    nc = params.nsoilcols
    bwater = bathym_levels(params.M + 1, area_int=bathym.area_int[l],
                           area_half=bathym.area_half[l], Lx=bathym.Lx[l],
                           Ly=bathym.Ly[l], Lx_half=bathym.Lx_half[l],
                           Ly_half=bathym.Ly_half[l])
    area = float(np.atleast_1d(params.area_lake)[l])

    def _ice_levels(thickness, total):
        """BATHYMETRY's ice/deep-ice area interpolation (bathym_mod.f90)."""
        n = params.Mice + 1
        if thickness <= C.small_value:
            a = np.full(n, area)
            return bathym_levels(n, area_int=a, area_half=a[:-1],
                                 Lx=np.full(n, _lxly(area, int(params.lakeform[l]),
                                                     float(params.cellipt[l]))[0]),
                                 Ly=np.full(n, _lxly(area, int(params.lakeform[l]),
                                                     float(params.cellipt[l]))[1]),
                                 Lx_half=np.full(n - 1, _lxly(area, int(params.lakeform[l]),
                                                              float(params.cellipt[l]))[0]),
                                 Ly_half=np.full(n - 1, _lxly(area, int(params.lakeform[l]),
                                                              float(params.cellipt[l]))[1]))
        table = params.morphometry[l, :int(params.morphometry_len[l]), :]
        dep = table[:, 0]
        ar = table[:, 1]
        zz = dep - np.max(dep) + total
        pos = np.zeros(n)
        for i in range(1, n):
            pos[i] = pos[i - 1] + grid.ddzi[i - 1] * thickness
        pos_half = np.zeros(n - 1)
        pos_half[0] = 0.5 * grid.ddzi[0] * thickness
        for i in range(1, n - 1):
            pos_half[i] = pos_half[i - 1] + 0.5 * (grid.ddzi[i - 1] + grid.ddzi[i]) * thickness
        a_int = interp_profile(zz, ar, pos[None, :])[0]
        a_half = interp_profile(zz, ar, pos_half[None, :])[0]
        form = int(np.atleast_1d(params.lakeform)[l])
        cell = float(np.atleast_1d(params.cellipt)[l])
        Lx, Ly = _lxly(a_int, form, cell)
        Lxh, Lyh = _lxly(a_half, form, cell)
        return bathym_levels(n, area_int=a_int, area_half=a_half,
                             Lx=Lx, Ly=Ly, Lx_half=Lxh, Ly_half=Lyh)

    bice = _ice_levels(float(st.l1[l]), float(st.h1[l]) + float(st.l1[l]) + float(st.ls1[l]))
    bdice = _ice_levels(float(st.ls1[l]), float(st.ls1[l]))
    z_half = grid.dzeta_05int * h1
    return SoilGeometry(
        M=params.M, Mice=params.Mice, zsoilcols=np.r_[0, zsc],
        ddz=np.r_[0, grid.ddz], ddzi=np.r_[0, grid.ddzi],
        ddz05=grid.ddz05, ddzi05=grid.ddzi05, z_half=np.r_[0, z_half],
        bathymwater=bwater, bathymice=bice, bathymdice=bdice, bathymsoil=bs)


def build_step_state(params, grid, l=0):
    from .state import init_state
    st = init_state(params, grid)
    h1 = float(st.h1[l])
    bathym = bathymetry(params, grid, st, C.pref)
    sp = comsoilforlake(
        soiltype=int(np.atleast_1d(params.soiltype)[0]), ns=params.ns,
        nsoilcols=params.nsoilcols, soil_depth=params.soil_depth,
        tricemethhydr=float(np.atleast_1d(params.tricemethhydr)[0]),
        nmeltpoint=int(np.atleast_1d(params.nmeltpoint)[0]),
        soilbotflx=float(np.atleast_1d(params.soilbotflx)[0]))
    geom = _zsoil_geom(params, grid, l, st, bathym)
    soil = make_soil_state(
        sp, geom, h1=h1, pressure=C.pref,
        Tw1=np.r_[0, st.Tw1[l]], Ti1=np.r_[0, st.Ti1[l]],
        Tis1=np.r_[0, st.Tis1[l]])
    soil.Tsoil1[1:, 1:] = st.Tsoil1[l]
    soil.Sals1[1:, 1:] = st.Sals1[l]
    soil.wl1[1:, 1:] = st.wl1[l]
    soil.wi1[1:, 1:] = st.wi1[l]
    soil_cond_heat_coef(soil, sp, is_=params.nsoilcols)

    row = water_dens_ts(np.r_[0, st.Tw1[l]], np.r_[0, st.Sal1[l]])
    mom_params = MomentumParams(
        M=params.M, ddz=grid.ddz, ddz05=grid.ddz05,
        dzeta_int=grid.dzeta_int, dzeta_05int=grid.dzeta_05int,
        area_int=bathym.area_int[l], area_half=bathym.area_half[l],
        Lx=bathym.Lx[l], Ly=bathym.Ly[l], Lx_half=bathym.Lx_half[l],
        Ly_half=bathym.Ly_half[l], depth_area=params.morphometry[l],
        kor=float(np.atleast_1d(params.kor)[l]),
        uwind=0.0, vwind=0.0, cdmw=1.0e-3, zref=2.0,
        fetch=float(np.atleast_1d(params.fetch)[l]))
    u1 = np.zeros(params.M + 1)
    v1 = np.zeros(params.M + 1)
    us0 = float(np.atleast_1d(params.us0)[l])
    vs0 = float(np.atleast_1d(params.vs0)[l])
    h_ml0 = float(np.atleast_1d(params.h_ML0)[l])
    i_ml = int((params.M + 1) * h_ml0 / h1)
    for ii in range(1, params.M + 2):
        if ii <= max(i_ml, 1):
            u1[ii - 1], v1[ii - 1] = us0, vs0
        else:
            frac = (ii - i_ml) / (params.M + 1 - i_ml)
            u1[ii - 1] = us0 * (1.0 - frac)
            v1[ii - 1] = vs0 * (1.0 - frac)
    mom = MomentumState(
        u1=u1, v1=v1, E1=st.E[l], eps1=st.eps[l],
        k2=np.zeros(params.M + 1), k2_old=np.zeros(params.M),
        row=row, Ri=np.zeros(params.M),
        knum=np.zeros(params.M + 1), veg_sw=np.zeros(params.M + 1),
        u_star=0.0, Tw1=st.Tw1[l], Sal1=st.Sal1[l],
        lsu_water=np.zeros(params.M + 1), lsv_water=np.zeros(params.M + 1),
        uv=np.abs(u1), h1=h1, l1=0.0, ls1=0.0, dhw=0.0, dhw0=0.0,
        alphax=0.0, alphay=0.0, velfrict_bot=0.0, velfrict_ice=0.0)
    M = params.M
    ke = KEpsilonState(
        e1=np.r_[0, st.E[l]], eps1=np.r_[0, st.eps[l]],
        k2=np.zeros(M + 2), k2t=np.zeros(M + 2), k5=np.zeros(M + 2),
        k2_mid=np.zeros(M + 2), k3_mid=np.zeros(M + 2),
        k5_mid=np.zeros(M + 2), l=np.zeros(M + 2), tf=np.zeros(M + 2),
        kt=np.zeros(M + 2), gamt=np.zeros(M + 2), gamu=np.zeros(M + 2),
        gamv=np.zeros(M + 2), row=np.r_[0, row], row2=np.r_[0, row],
        z_half=np.r_[0, grid.dzeta_05int * h1],
        u1=u1, v1=v1, u2=u1, v2=v1, um=u1, vm=v1,
        tw1=np.r_[0, st.Tw1[l]], tw2=np.r_[0, st.Tw1[l]],
        sal1=np.r_[0, st.Sal1[l]], sal2=np.r_[0, st.Sal1[l]],
        knum=np.zeros(M + 2), veg_sw=np.zeros(M + 2),
        lamw=np.r_[0, st.lamw[l]], lamsal=np.zeros(M + 2),
        h1=h1, l1=0.0, dhw=0.0, dhw0=0.0,
        vol=float(bathym.vol[l]), botar=float(bathym.botar[l]))
    ke.init_arrays(M)
    ke.ddz = np.r_[0, grid.ddz]
    ke.ddz05 = grid.ddz05
    ke.ddz2 = np.zeros(M + 2); ke.ddz2[1:M] = grid.ddz[:-1] + grid.ddz[1:]
    ke.ddz052 = np.zeros(M + 2); ke.ddz054 = np.zeros(M + 2)
    for k in range(2, M):
        ke.ddz052[k] = 0.5 * grid.ddz[k - 2] + grid.ddz[k - 1] + 0.5 * grid.ddz[k]
        ke.ddz054[k] = 2.0 * ke.ddz052[k]
    ke.dzeta_int = np.r_[0, grid.dzeta_int]
    ke.dzeta_05int = np.r_[0, grid.dzeta_05int]
    ke.area_int = np.r_[0, bathym.area_int[l]]
    ke.area_half = np.r_[0, bathym.area_half[l]]
    extinct = np.empty((M, 4))
    extinct[:, 0] = 3.13
    extinct[:, 1] = float(np.atleast_1d(params.extwat)[l])
    extinct[:, 2] = 1.73
    extinct[:, 3] = 1.087e3
    flux0 = np.asarray(FRACBANDS) * 0.0
    _, integr = rad_update(extinct, grid.ddz05[:params.M] * h1, flux0)
    return StepState(st, sp, soil, geom, mom_params, mom,
                     KEpsilonParams(M=M), ke, np.asarray(integr),
                     np.asarray(flux0), bathym=bathym)


def _refresh_geometry(ss, params, grid, l, st):
    """Recompute BATHYMETRY for the current water thickness.

    The Fortran calls ``BATHYMETRY`` on every step (shared prologue and again
    inside case 1), so the layer areas/widths must not stay at their initial
    values.  They feed the surface heat balance, the multi-column soil
    interpolation and momentum/k-epsilon.
    """
    bathym = bathymetry(params, grid, st, C.pref)
    ss.bathym = bathym
    geom = _zsoil_geom(params, grid, l, st, bathym)
    ss.soil_geom = geom
    ss.soil.geom = geom
    mp = ss.mom_params
    mp.area_int = bathym.area_int[l]
    mp.area_half = bathym.area_half[l]
    mp.Lx = bathym.Lx[l]
    mp.Ly = bathym.Ly[l]
    mp.Lx_half = bathym.Lx_half[l]
    mp.Ly_half = bathym.Ly_half[l]
    ss.ke.area_int = np.r_[0, bathym.area_int[l]]
    ss.ke.area_half = np.r_[0, bathym.area_half[l]]
    return bathym


def step_case1_to_solver(ss, params, grid, rows, nstep=1, l=0, init_step=False):
    """Case-1 chain through T_SOLVER (diagnostic integration helper)."""
    import numpy as np
    from .forcing import at_time
    from .physics.density import water_dens_ts
    from .physics.surface import (WaterSurfaceContext, heatbalance_water,
                                  radbalance, surf_char)

    st = ss.state
    dt = params.dt
    _refresh_geometry(ss, params, grid, l, st)
    f = at_time(rows, nstep * dt, interval=float(np.atleast_1d(params.interval)[0]))
    tempair, pressure, longwave, shortwave = f[0] - C.Kelvin0, f[1], f[2], f[3]
    uwind, vwind, humair, precip = f[4], f[5], f[6], f[7]
    hour = float(np.atleast_1d(params.hour0)[0]) + nstep * dt / 3600.0
    year = int(np.atleast_1d(params.year0)[0])
    month = int(np.atleast_1d(params.month0)[0])
    day = int(np.atleast_1d(params.day0)[0])
    phi = float(np.atleast_1d(params.phi)[l])
    fetch = float(np.atleast_1d(params.fetch)[l])
    sc = surf_char(4, year, month, day, hour, phi, fetch, uwind, 1.0e-2,
                   int(np.atleast_1d(params.extwat)[l] * 0 + 1))
    albedo, albedo_lw, emis = sc["albedo"], sc["albedo_lw"], sc["emissivity"]

    # shortwave radiation in the water column
    Erad = shortwave * (1.0 - albedo)
    flux0 = np.asarray(FRACBANDS) * Erad
    M = params.M
    extinct = np.empty((M, 4))
    extinct[:, 0] = 3.13
    extinct[:, 1] = float(np.atleast_1d(params.extwat)[l])
    extinct[:, 2] = 1.73
    extinct[:, 3] = 1.087e3
    _, integr = rad_update(extinct, grid.ddz05[:M] * float(st.h1[l]), flux0)

    # soil columns at the current water temperature (the soil fluxes also
    # water-depth increments use the PREVIOUS step's dhwfsoil, as in the
    # Fortran order (dhw before SOILCOLSTEMP)
    dhwp = precip * dt
    dhwe = -ss.Elatent / (C.Lwv * C.row0) * dt
    # no tributary inflow: Fortran's fallback depth relaxation
    dhwtrib = (float(np.atleast_1d(params.h10)[l]) - float(st.h1[l])) / 10.0
    dhw = dhwp + dhwe + float(ss.soil.dhwfsoil) + dhwtrib
    dhw0 = dhwp + dhwe

    # soil columns at the current water temperature
    ss.soil.Tw1 = np.r_[0, st.Tw1[l]]
    ss.soil.Ti1 = np.r_[0, st.Ti1[l]]
    ss.soil.Tis1 = np.r_[0, st.Tis1[l]]
    ss.soil.h1 = float(st.h1[l])
    ss.soil.pressure = pressure
    soil_cond_heat_coef(ss.soil, ss.soil_params, is_=params.nsoilcols)
    ss.soil = soilcolstemp(ss.soil, ss.soil_params, dt)
    lsh = laterheat(ss.soil, ss.soil_params, ss.soil.soilflux, onlywater=True)
    ss.ck_soilcol = dict(Tsoil1=ss.soil.Tsoil1.copy(),
                         Tsoil2=ss.soil.Tsoil2.copy(),
                         dhwfsoil=float(ss.soil.dhwfsoil),
                         h1=float(st.h1[l]), dhw=float(dhw),
                         dhw0=float(dhw0), Elatent=float(ss.Elatent),
                         wl1=ss.soil.wl1.copy(), wi1=ss.soil.wi1.copy(),
                         csoil=ss.soil.csoil.copy(), rosoil=ss.soil.rosoil.copy(),
                         lamsoil=ss.soil.lamsoil.copy(),
                         radwater=np.array(integr, copy=True))

    # momentum
    ss.mom_params.uwind = float(uwind)
    ss.mom_params.vwind = float(vwind)
    ss.mom_params.cdmw = ss.cdmw
    ss.mom_params.fetch = fetch
    if getattr(ss.mom, "u2", None) is not None:
        ss.mom.u1 = np.array(ss.mom.u2)
        ss.mom.v1 = np.array(ss.mom.v2)
    ss.mom.Tw1 = st.Tw1[l]
    ss.mom.Sal1 = st.Sal1[l]
    ss.mom.row = water_dens_ts(st.Tw1[l], st.Sal1[l])
    ss.mom.E1 = st.E[l]
    ss.mom.eps1 = st.eps[l]
    if ss.ke.k2 is not None:
        ss.mom.k2 = np.array(ss.ke.k2[1:M + 2], copy=True)
    if ss.ke.knum is not None:
        ss.mom.knum = np.array(ss.ke.knum[1:M + 2], copy=True)
    if ss.ke.veg_sw is not None:
        ss.mom.veg_sw = np.array(ss.ke.veg_sw[1:M + 2], copy=True)
    ss.mom.h1 = float(st.h1[l])
    ss.mom.dhw = dhw
    ss.mom.dhw0 = dhw0
    ss.mom_u1_old = np.array(ss.mom.u1, copy=True)
    ss.mom_v1_old = np.array(ss.mom.v1, copy=True)
    mom_res = momentum_solver(ss.mom, ss.mom_params, dt)
    ss.mom = mom_res.state
    ss.mom.u1 = np.array(mom_res.u2, copy=True)
    ss.mom.v1 = np.array(mom_res.v2, copy=True)

    Tw1 = np.r_[0, st.Tw1[l]]
    Tsoil1 = ss.soil.Tsoil1
    ctx = WaterSurfaceContext(
        Tw1=Tw1, Tsoil1=Tsoil1, lamw=np.r_[0, st.lamw[l]],
        area_int=np.r_[0, ss.bathym.area_int[l]],
        area_half=np.r_[0, ss.bathym.area_half[l]], h1=float(st.h1[l]),
        ddz=np.r_[0, grid.ddz], dhw=dhw, dhw0=dhw0,
        PEMF=np.zeros(M + 2), pt_down_f=np.zeros(M + 2),
        lamsoil=ss.soil.lamsoil, csoil=ss.soil.csoil,
        rosoil=ss.soil.rosoil, dzs=ss.soil_params.dzs,
        radwater_integr=np.asarray(integr),
        lsh_water=np.r_[0, lsh.water[1:]],
        dzeta_int=np.r_[0, grid.dzeta_int], ddz05=grid.ddz05,
        M=M, ns=params.ns, isoilcol=params.nsoilcols, dt=dt,
        fetch=fetch, uwind=uwind, vwind=vwind,
        u=float(ss.mom.u1[0]), v=float(ss.mom.v1[0]),
        tempair=tempair, pressure=pressure, humair=humair, zref=2.0,
        roughness=sc["roughness"], relhums=sc["relhums"], aM=sc["aM"],
        bM=sc["bM"], l1=0.0, longwave=longwave, shortwave=shortwave,
        netrad=-1.0, cloud=-1.0, albedo=albedo, albedo_lw=albedo_lw,
        emissivity=emis)

    def fn(t):
        return heatbalance_water(t, ctx)

    r = t_solver(fn, nstep, 4, ss.surftyp_prev, ss.Tsurf_prev)
    ss.Tsurf_prev = r["Tsurf3"]
    ss.surftyp_prev = 4
    ss.cdmw = float(r["state"].get("cdmw", ss.cdmw))
    return dict(Tsurf3=r["Tsurf3"], state=r["state"], iter=r["iter"],
                Bal3=r["Bal3"], dhw=dhw, dhw0=dhw0, integr=integr, lsh=lsh,
                sc=sc, f=f, ctx=ctx, mom=mom_res, precip=precip,
                uwind=uwind, vwind=vwind, ss=ss)


def step_case1(ss, params, grid, rows, nstep=1, l=0, init_step=False):
    """Full case-1 step (open water, no ice/snow)."""
    import numpy as np
    from .physics.density import ddens_dtemp, water_dens_ts
    from .physics.turb import diffmin_hs, turb_scales

    r = step_case1_to_solver(ss, params, grid, rows, nstep, l, init_step=init_step)
    st = ss.state
    dt = params.dt
    M, ns, nc = params.M, params.ns, params.nsoilcols
    Tw2 = np.r_[0, r["state"]["Tw2"]]
    Tsoil2 = np.r_[0, r["state"]["Tsoil2"]]

    # deepest soil column
    ss.soil.Tsoil2 = Tsoil2
    ss.soil = soilforlake(ss.soil, ss.soil_params, dt, is_=nc)
    ss.ck_soilf = dict(Tsoil1=ss.soil.Tsoil1.copy(),
                       Tsoil2=ss.soil.Tsoil2.copy(),
                       dhwfsoil=float(ss.soil.dhwfsoil),
                       h1=float(ss.h1 if hasattr(ss, "h1") else st.h1[l]),
                       dhw=float(r["dhw"]), dhw0=float(r["dhw0"]),
                       Elatent=float(r["state"]["Elatent"]))

    # salinity (fresh water)
    Sal1 = np.r_[0, st.Sal1[l]]
    lamsal = np.r_[0, (st.lamw[l] - C.lamw0) / C.cw_m_row0 * C.alsal + C.lamsal0]
    dhwp = r["precip"] * dt
    dhwe = -r["state"]["Elatent"] / (C.Lwv * C.row0) * dt
    Sal2 = s_diff(Sal1, dt, r["dhw"], r["dhw0"], dhwp, dhwe, lamsal,
                  np.r_[0, ss.bathym.area_int[l]],
                  np.r_[0, ss.bathym.area_half[l]], np.r_[0, grid.ddz],
                  float(st.h1[l]), np.r_[0, grid.dzeta_int], grid.ddz05, M)

    preswat = np.r_[0, ss.bathym.preswat[l]]
    row2 = water_dens_ts(Tw2, Sal2)

    # k-epsilon
    ke = ss.ke
    ke.e1 = np.r_[0, st.E[l]]
    ke.eps1 = np.r_[0, st.eps[l]]
    ke.u1 = np.r_[0, ss.mom_u1_old]
    ke.v1 = np.r_[0, ss.mom_v1_old]
    ke.um = np.zeros(M + 2)
    ke.vm = np.zeros(M + 2)
    ke.u2 = np.r_[0, r["mom"].u2]
    ke.v2 = np.r_[0, r["mom"].v2]
    ke.tw1 = np.r_[0, st.Tw1[l]]
    ke.tw2 = Tw2
    ke.sal1 = Sal1
    ke.sal2 = Sal2
    ke.row = np.r_[0, ss.mom.row]
    ke.row2 = row2
    ke.h1 = float(st.h1[l])
    ke.dhw = r["dhw"]
    ke.dhw0 = r["dhw0"]
    ke.vol = float(ss.bathym.vol[l])
    ke.botar = float(ss.bathym.botar[l])
    ke.b0 = 0.0
    ke.Buoyancy0 = (C.g / C.row0
                    * float(ddens_dtemp(st.Tw1[l][0], 0.0))
                    * float(r["state"]["eflux0_kinem"]))
    ke.tau_air = float(r["mom"].tau_air)
    ke.tau_wav = float(r["mom"].tau_wav)
    ke.tau_i = float(r["mom"].tau_i or 0.0) if getattr(r["mom"], "tau_i", None) is not None else 0.0
    ke.tau_gr = float(r["mom"].tau_gr)
    ke.uwind = float(r["uwind"])
    ke.vwind = float(r["vwind"])
    ke.fetch = float(np.atleast_1d(params.fetch)[l])
    ke.roughness = float(r["sc"]["roughness"])
    ke.knum = np.zeros(M + 2)
    ke.veg_sw = np.zeros(M + 2)
    import copy as _copy
    if ss.trace_ke:
        ss.ke_trace.append(_copy.deepcopy(ke))
    ke = k_epsilon(ke, dt, dt_keps=dt, nstep_keps=1, params=ss.ke_params)
    ss.ke = ke

    # eddy diffusivity and turbulence diagnostics
    if int(np.atleast_1d(params.backdiff)[0]) == 1:
        lamw_back = diffmin_hs(ss.mom.row, grid.ddz, float(st.h1[l]),
                               float(np.atleast_1d(params.area_lake)[l]), M,
                               float(np.atleast_1d(params.backdiff0)[0]))
    else:
        lamw_back = np.zeros(M)
    lamw_new = C.lamw0 + np.maximum(ke.kt[1:M + 1], 0.0) + lamw_back
    k_turb = np.array([
        -0.5 * lamw_new[i] / C.cw_m_row0
        * (Tw2[i + 2] + st.Tw1[l][i + 1] - Tw2[i + 1] - st.Tw1[l][i])
        / (grid.ddz[i] * float(st.h1[l]))
        for i in range(M)])
    eflux0_kinem = float(r["state"]["eflux0_kinem"])
    turb_density_flux = float(ddens_dtemp(st.Tw1[l][0], 0.0) * eflux0_kinem)
    ts = turb_scales(
        ss.mom.row, grid.ddz, grid.ddz[:-1] + grid.ddz[1:], grid.dzeta_int,
        grid.dzeta_05int, float(st.h1[l]), M, Tw2[1:], Sal2[1:],
        np.r_[0, r["mom"].u2][1:], np.r_[0, r["mom"].v2][1:],
        lamw_new, lamsal, k_turb, np.zeros(M), eflux0_kinem,
        turb_density_flux, C.g / C.row0 * turb_density_flux,
        float(r["mom"].tau_air - r["mom"].tau_wav),
        float(np.atleast_1d(params.kor)[l]), np.asarray(r["integr"]),
        ss.bathym.area_int[l], ss.bathym.Lx[l], ss.bathym.Ly[l],
        float(ss.bathym.vol[l]), grid.dzeta_int * float(st.h1[l]),
        grid.ddz05, -0.15063272953663126, 0.7778285184375)

    lay = advance_case1(
        SimpleNamespace(h1=np.array([st.h1[l]]), ls1=np.array([0.0])),
        np.array([r["dhw"]]))
    return dict(
        Tw1=np.array(Tw2[1:], copy=True), Sal1=np.array(Sal2[1:], copy=True),
        Tsoil1=np.array(ss.soil.Tsoil1[1:, 1:], copy=True),
        h1=float(lay["h1"][0]), Elatent=float(r["state"]["Elatent"]),
        eflux0_kinem=eflux0_kinem, E1=np.array(ke.e1[1:], copy=True),
        eps1=np.array(ke.eps1[1:], copy=True),
        lamw=np.array(lamw_new, copy=True), kt=np.array(ke.kt[1:], copy=True),
        row2=np.array(row2[1:], copy=True), lsh=r["lsh"],
        Tsurf3=r["Tsurf3"], dhw=r["dhw"], dhw0=r["dhw0"],
        turb=ts, Tsoil3=ss.soil.Tsoil3[1:, 1:], dhwfsoil=ss.soil.dhwfsoil)


def run_case1(params, grid, rows, nsteps=180, l=0, callback=None, store=True):
    """Run `nsteps` case-1 steps from the initial state."""
    ss = build_step_state(params, grid, l)
    st = ss.state
    dt = params.dt
    out = []
    for n in range(1, nsteps + 1):
        r = step_case1(ss, params, grid, rows, n, l, init_step=(n == 1))
        st.Tw1[l] = r["Tw1"]
        st.Sal1[l] = r["Sal1"]
        st.Tsoil1[l] = r["Tsoil1"]
        st.h1[l] = r["h1"]
        st.E[l] = r["E1"]
        st.eps[l] = r["eps1"]
        st.lamw[l] = r["lamw"]
        st.time[l] += dt
        st.nstep[l] = n
        ss.Elatent = r["Elatent"]
        ss.soil.dhwfsoil = r["dhwfsoil"]
        if store:
            out.append(r)
        if callback is not None:
            callback(n, ss, r)
    return ss, out


def build_ice_state(ss, params, grid):
    """Allocate/fill an `IceState` from the current water/soil state (M4)."""
    from .physics.ice import ice_properties, make_ice_state

    st = make_ice_state(params.M, params.Mice, params.ns, 100,
                        geom=ss.soil_geom, isoilcol=params.nsoilcols,
                        nmeltpoint=int(np.atleast_1d(params.nmeltpoint)[0]),
                        soilbotflx=float(np.atleast_1d(params.soilbotflx)[0]))
    st.Ti1[1:] = ss.state.Ti1[0]
    st.Ti2[1:] = ss.state.Ti1[0]
    st.Tis1[1:] = ss.state.Tis1[0]
    st.Tis2[1:] = ss.state.Tis1[0]
    st.Tw1[1:] = ss.state.Tw1[0]
    st.Tw2[1:] = ss.state.Tw1[0]
    st.lamw[1:] = ss.state.lamw[0]
    st.h1 = float(ss.state.h1[0])
    st.l1 = float(ss.state.l1[0])
    st.ls1 = float(ss.state.ls1[0])
    ice_properties(st)
    return st
