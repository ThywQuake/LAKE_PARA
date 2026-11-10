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
from .physics.bathym import bathymetry, soil_bathymetry
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
    bice = bathym_levels(params.Mice + 1, area_int=np.full(params.Mice + 1, area),
                         area_half=np.full(params.Mice, area))
    z_half = grid.dzeta_05int * h1
    return SoilGeometry(
        M=params.M, Mice=params.Mice, zsoilcols=np.r_[0, zsc],
        ddz=np.r_[0, grid.ddz], ddzi=np.r_[0, grid.ddzi],
        ddz05=grid.ddz05, ddzi05=grid.ddzi05, z_half=np.r_[0, z_half],
        bathymwater=bwater, bathymice=bice, bathymdice=bice, bathymsoil=bs)


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

    row = water_dens_ts(np.r_[0, st.Tw1[l]], np.r_[0, st.Sal1[l]],
                        bathym.preswat[l])
    mom_params = MomentumParams(
        M=params.M, ddz=grid.ddz, ddz05=grid.ddz05,
        dzeta_int=grid.dzeta_int, dzeta_05int=grid.dzeta_05int,
        area_int=bathym.area_int[l], area_half=bathym.area_half[l],
        Lx=bathym.Lx[l], Ly=bathym.Ly[l], Lx_half=bathym.Lx_half[l],
        Ly_half=bathym.Ly_half[l], depth_area=params.morphometry[l],
        kor=float(np.atleast_1d(params.kor)[l]),
        uwind=0.0, vwind=0.0, cdmw=1.0e-3, zref=2.0,
        fetch=float(np.atleast_1d(params.fetch)[l]))
    u1 = np.zeros(params.M + 2)
    v1 = np.zeros(params.M + 2)
    u1[1:] = float(np.atleast_1d(params.us0)[l])
    v1[1:] = float(np.atleast_1d(params.vs0)[l])
    mom = MomentumState(
        u1=u1, v1=v1, E1=np.r_[0, st.E[l]], eps1=np.r_[0, st.eps[l]],
        k2=np.zeros(params.M + 2), k2_old=np.zeros(params.M + 1),
        row=np.r_[0, row], Ri=np.zeros(params.M + 1),
        knum=np.zeros(params.M + 2), veg_sw=np.zeros(params.M + 2),
        u_star=0.0, Tw1=np.r_[0, st.Tw1[l]], Sal1=np.r_[0, st.Sal1[l]],
        lsu_water=np.zeros(params.M + 2), lsv_water=np.zeros(params.M + 2),
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
                     np.asarray(flux0))
