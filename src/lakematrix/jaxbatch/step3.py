"""Minimal batched JAX case-3 step (ice only, h1=0).

Self-consistency port of ``lakematrix.step3.step_case3``.  It uses the
validated ice T_DIFF/surface/snow kernels and the existing state adapter from
`jaxbatch.step2`; there is no stable Fortran case-3 reference.
"""

import jax.numpy as jnp

from .step import at_time, _pad
from .step2 import (make_params, from_numpy_state, to_numpy,  # noqa: F401
                   ice_bathymetry)
from .ice_tdiff import t_diff_ice
from .ice_surface import heatbalance_ice, surf_char_ice
from .snowice import (ice_properties, ice_increments_nosnow,
                      snow_calc, snow_cond_heat_coef, sr_botsnow)
from .soilforlake import soilforlake
from .tsolver import t_solver
from ..constants import C
from ..physics.radiation import FRACBANDS


def _radiation(ice, ex, erad):
    from .step import rad_update
    extinct = jnp.full((ice["Ti1"].shape[0], ice["ddzi"].shape[1] - 1, 4), ex)
    _, integr = rad_update(extinct, ice["ddzi"][:, 1:] * ice["l1"][:, None],
                           erad[:, None] * jnp.asarray(FRACBANDS)[None, :])
    return integr


def step_case3_jax(state, params, rows, nstep, first=True):
    """One synthetic case-3 step (no water) for a batch of lakes."""
    p = {**params, "rows": rows}
    ice = {k: jnp.asarray(v) for k, v in state["ice"].items()}
    M, Mice, ns, ncols = p["M"], p["Mice"], p["ns"], p["ncols"]
    L = ice["l1"].shape[0]
    ice.setdefault("ddzi", jnp.broadcast_to(_pad(jnp.asarray(p["ddzi"]))[None, :],
                                            (L, Mice + 1)))
    ice.setdefault("ddz", jnp.broadcast_to(_pad(jnp.asarray(p["ddz"]))[None, :],
                                           (L, M + 1)))
    for k in ("Tsoil1", "lamsoil", "csoil", "rosoil"):
        ice.setdefault(k, jnp.asarray(state["soil"][k]))
    ice.setdefault("dzs", jnp.asarray(p["dzs"])[None, :])
    ib = ice_bathymetry(ice["l1"], ice["l1"], p)
    ice.setdefault("Aice", _pad(ib["area_int"]))
    ice.setdefault("Ahice", _pad(ib["area_half"]))
    ice.setdefault("Adice", jnp.zeros_like(_pad(ib["area_int"])))
    ice.setdefault("Ahdice", jnp.zeros_like(_pad(ib["area_half"])))
    ice.setdefault("Awat", jnp.zeros((L, M + 2)))
    ice.setdefault("Ahwat", jnp.zeros((L, M + 1)))
    ice.setdefault("ddz05", jnp.broadcast_to(jnp.asarray(p["ddz05"])[None, :],
                                            (L, p["ddz05"].shape[0])))
    ice.setdefault("ddzi05", jnp.broadcast_to(jnp.asarray(p["ddzi05"])[None, :],
                                             (L, p["ddzi05"].shape[0])))
    ice.setdefault("dzetai_int", _pad(jnp.asarray(p["dzetai_int"])))
    ice.setdefault("dzeta_int", _pad(jnp.asarray(p["dzeta_int"])))
    ice.setdefault("lamw", _pad(jnp.asarray(state["water"]["lamw"])))
    ice.setdefault("PEMF", jnp.zeros((L, M + 2)))
    ice.setdefault("pt_down_f", jnp.zeros((L, M + 2)))
    ice.setdefault("Sal2", _pad(jnp.asarray(state["water"]["Sal1"])))
    ice.setdefault("preswat", jnp.zeros((L, M + 2)))
    ice.setdefault("Tsoil2", jnp.asarray(state["soil"]["Tsoil2"]))
    ice.setdefault("soilbotflx", jnp.asarray(p["soilbotflx"]))
    ice.setdefault("soilswitch", jnp.asarray(1))
    ice.setdefault("h1", jnp.zeros(L))
    ice.setdefault("ls1", jnp.zeros(L))
    ice.setdefault("dhw", jnp.zeros(L))
    ice.setdefault("dhw0", jnp.zeros(L))
    ice.setdefault("snmeltice", jnp.zeros(L))
    ice.setdefault("dhiimp", jnp.zeros(L))
    dt = p["dt"]
    f = at_time(rows, nstep * dt, p["interval"])
    tempair = f[0] - C.Kelvin0
    pressure, longwave, shortwave = f[1], f[2], f[3]
    uwind, vwind, humair, precip = f[4], f[5], f[6], f[7]
    hour = p["hour0"] + nstep * dt / 3600.0
    snow = ice["flag_snow"].astype(jnp.int32)
    surftyp = jnp.where(snow == 1, 3, 2).astype(jnp.int32)
    sc = surf_char_ice(surftyp, p["year0"], p["month0"], p["day0"], hour,
                       p["phi"], p["fetch"], uwind, jnp.full_like(uwind, 1e-2))
    albedo = sc["albedo"]
    ice = ice_properties(ice)
    ice = snow_cond_heat_coef(ice, tempair)
    erad = jnp.where(snow == 1, sr_botsnow(ice, shortwave * (1 - albedo)),
                     shortwave * (1 - albedo))
    ex = jnp.asarray(p["extice"], dtype=jnp.float64)[:, None]
    ice["rad_integr_ice"] = _radiation(ice, ex, erad)
    ice["rad_integr_water"] = jnp.zeros((ice["l1"].shape[0], M + 1))
    ice["rad_integr_deepice"] = jnp.zeros((ice["l1"].shape[0], Mice + 1))

    x = 0.0
    melt = ice["Ti1"][:, 1] > x + C.T_phase_threshold
    dhw = ((ice["Ti1"][:, 1] - x - C.T_phase_threshold) * ice["ci_m_roi_v"][:, 1]
           * ice["l1"] * ice["ddzi"][:, 1] * 0.5 / C.row0_m_Lwi)
    dhihigh = -dhw * C.roi_d_row0
    dhis = jnp.where(melt, 0.0, -state["water"]["Elatent"] / (C.roi * C.Liv) * dt)
    ice["Ti1"] = ice["Ti1"].at[:, 1].set(
        jnp.where(melt, x + C.T_phase_threshold, ice["Ti1"][:, 1]))
    # precipitation/snow bookkeeping (case-3.2)
    snowmass = ice["snowmass"] + jnp.where((precip > 0) & (tempair < 0),
                                           precip * dt * C.row0, 0.0)
    dhip = jnp.where((precip > 0) & (tempair < 0), 0.0,
                     precip * C.row0_d_roi * dt)
    ice["snowmass"] = snowmass
    ice["dhi"] = jnp.where(melt, dhihigh, dhis) + dhip
    ice["dhi0"] = ice["dhi"]

    ctx = dict(state=ice, dt=dt, fetch=p["fetch"], uwind=uwind,
               vwind=vwind, u=jnp.zeros_like(uwind), v=jnp.zeros_like(uwind),
               tempair=tempair, pressure=pressure, humair=humair, zref=2.0,
               roughness=sc["roughness"], relhums=sc["relhums"], aM=sc["aM"],
               bM=sc["bM"], l1=ice["l1"], longwave=longwave,
               shortwave=shortwave, netrad=-1.0, cloud=-1.0,
               albedo=albedo, albedo_lw=sc["albedo_lw"],
               emissivity=sc["emissivity"], radwater_integr=ice["rad_integr_water"],
               radice_integr=ice["rad_integr_ice"], snow=snow,
               area_int=ice["Aice"], area_half=ice["Ahice"],
               ice=jnp.ones_like(snow), water=jnp.zeros_like(snow),
               deepice=jnp.zeros_like(snow), isoilcol=ncols)
    fn = lambda T, c: heatbalance_ice(T, c, t_diff_ice)
    r = t_solver(fn, nstep, surftyp, state["water"]["surftyp_prev"],
                 state["water"]["Tsurf_prev"], ctx=ctx)
    ice = t_diff_ice(0, r["Tsurf3"], dt, snow, jnp.ones_like(snow),
                     jnp.zeros_like(snow), jnp.zeros_like(snow), ncols,
                     False, r["state"]["ice"])
    soil = dict(state["soil"])
    soil["Tsoil2"] = ice["Tsoil2"]
    soil["h1"] = jnp.zeros(L)
    soil["l1"] = ice["l1"]
    soil["ls1"] = jnp.zeros(L)
    soil_p = {k: p[k] for k in ("ns", "POR", "rosdry", "dzs", "dzss", "zsoil",
                               "nmeltpoint", "WLM0", "WLM7",
                               "soilbotflx")}
    soil_p["pressure"] = pressure
    soil_p["nsoilcols"] = p["ncols"]
    soil = soilforlake(soil, soil_p,
                       ice["Tsoil2"], dt)
    ice["l1"] = ice["l1"] + ice["dhi"]
    out = dict(state)
    out["soil"] = soil
    out["ice"] = ice
    out["water"] = dict(state["water"])
    out["water"]["Tsurf_prev"] = r["Tsurf3"]
    out["water"]["surftyp_prev"] = surftyp
    out["water"]["h1"] = jnp.zeros_like(state["water"]["h1"])
    return out
