"""B5: batched JAX case-1 step plus a ``lax.scan`` runner.

This module assembles the validated B1-B4 kernels into the full open-water
(case 1) time step and runs it over a batch of lakes, mirroring
:func:`lakematrix.step.step_case1` (the NumPy reference) call for call::

    at_time -> BATHYMETRY (water + soil columns)
            -> SURF_CHAR / RAD_UPDATE
            -> SOIL_COND_HEAT_COEF + SOILCOLSTEMP (side columns)
            -> LATERHEAT
            -> dhw / dhw0
            -> MOMENTUM_SOLVER
            -> T_SOLVER (HEATBALANCE + T_DIFF)
            -> SOILFORLAKE (deepest column)
            -> S_DIFF
            -> K_EPSILON
            -> TURB_SCALES (lamw, k_turb)
            -> layer update (h1 += dhw)

Nothing here adds physics: every equation is taken from an existing kernel
(``jaxbatch.bathym``, ``jaxbatch.soil``/``soilcols``/``soilpass``/
``soilforlake``, ``jaxbatch.laterheat``, ``jaxbatch.momentum``,
``jaxbatch.tsolver``/``surface``, ``jaxbatch.kepsilon``, ``jaxbatch.turb``).
The only new code is the small integration helpers that had no JAX version yet
(``at_time``, ``rad_update``, ``s_diff``, ``water_dens_ts``/``ddens_dtemp``/
``ddens_dsal``, ``diffmin_hs``, ``advance_case1``) plus the two adapters
:func:`from_numpy_state` / :func:`to_numpy`.

State / params layout
=====================

Both are plain ``dict``s of arrays whose leading axis is the lake (``L``);
``lax.scan`` carries the state dict unchanged.  ``params`` holds the static
sizes/grid plus the per-lake setup values (see :func:`make_params`).  For
tracing, :func:`_split` separates the hashable/static part (sizes, switches,
the :class:`~lakematrix.grid.Grid`, soil-type scalars) from the array part;
the static part is closed over (and the compiled step cached per signature)
while every array is a real ``jax.jit`` operand.

``state`` (all axes are lake-first)::

    water  Tw1      (L, M+1)     water temperature, 0-based (Fortran 1..M+1)
           Sal1     (L, M+1)     salinity, kg/kg
           h1       (L,)         depth, m
           lamw     (L, M)       molecular+turbulent heat conductivity, W/m/K
           E1       (L, M+1)     TKE, m^2/s^2
           eps1     (L, M+1)     TKE dissipation, m^2/s^3
           u2       (L, M+1)     velocities at the *start* of the step
           v2       (L, M+1)     (== the previous MOMENTUM_SOLVER output)
           Elatent  (L,)         latent heat flux of the previous step, W/m^2
           Tsurf_prev (L,)       T_SOLVER surface temperature of the last step
           surftyp_prev (L,)     int32, T_SOLVER surface-type switch
           cdmw     (L,)         drag coefficient from the last step
    soil   Tsoil1   (L, ns+1, ncols+1)   soil temperature, 1-based padded
           Tsoil2   (L, ns+1)            T_DIFF result of the deepest column
           Tsoil3   (L, ns+1, ncols+1)   phase-changed temperature
           wl1, wl4, wi1, wi2, Sals1 (L, ns+1, ncols+1)
           wl2, wl3      (L, ns+1)       SOILFORLAKE work arrays
           wa, methgenmh (L, ns+1)
           csoil, rosoil, lamsoil, lammoist, filtr, wsoil (L, ns+1)
           soilflux (L, ncols+1)         heat flux into each soil column, W/m2
           dhwfsoil (L,)                 water released/consumed by the soil
    ke     e1, eps1, k2  (L, M+2) 1-based padded K_EPSILON fields
           init_keps    (L,) bool
    mom    k2_old  (L, M)   Ri (L, M)   uv, knum, veg_sw, lsu_water,
           lsv_water (L, M+1)  velfrict_ice, alphax, alphay (L,)
    diag   per-step diagnostics (Tsurf3, dhw, dhw0, lsh, kt, row2, turb, ...)

The shapes follow `lakematrix.step.StepState`: arrays that the Fortran keeps
1-based (soil, k-epsilon) stay padded, the arrays taken from
:class:`lakematrix.state.State` stay 0-based.  :func:`from_numpy_state` builds
this dict from a NumPy ``StepState`` and :func:`to_numpy` converts the
prognostic fields back to the ``step_case1`` return layout.

Scope: ``nsoilcols >= 2``, a morphometry table, ``l1 == ls1 == 0``,
``Turbpar/stabfunc/kepsbc`` of the reference setup, ``eos 5``/``lindens 0``.
Anything outside raises in the kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C
from ..physics.density import (A11, A12, A13, A14, alpha1, alpha2, alpha3,
                               alpha4, alpha5, beta1, beta2, beta3, gamma1)
from ..physics.radiation import FRACBANDS
from ..physics.tridiag import progonka_jax
from . import bathym as _bathym
from .kepsilon import KEpsilonParams
from .kepsilon import k_epsilon as _k_epsilon
from .laterheat import laterheat_water
from .momentum import momentum_solver as _momentum_solver
from .soilforlake import soilforlake as _soilforlake
from .soilpass import soilcolstemp_water
from .surface import surf_char
from .turb import turb_scales as _turb_scales
from .tsolver import make_solver

__all__ = [
    "advance_case1", "at_time", "ddens_dsal", "ddens_dtemp", "diffmin_hs",
    "dup_state", "from_numpy_state", "make_params", "rad_update",
    "run_case1_jax", "s_diff", "step_case1_jax", "to_numpy",
    "water_dens_ts", "SOIL_KEYS",
]

#: soil fields carried between steps (``SoilState`` arrays the kernels read or
#: write; the ``a``/``b``/``c``/``d`` work arrays of the NumPy reference are
#: local to ``SOILFORLAKE`` in the JAX port and need not be carried).
SOIL_KEYS = ("Tsoil1", "Tsoil2", "Tsoil3", "wl1", "wl2", "wl3", "wl4",
             "wi1", "wi2", "Sals1", "wa", "methgenmh", "csoil", "rosoil",
             "lamsoil", "lammoist", "filtr", "wsoil", "soilflux",
             "dhwfsoil")

#: keys of the ``params`` dict read by the batched soil kernels
_SOIL_PARAM_KEYS = ("ns", "ncols", "POR", "rosdry", "dzs", "dzss", "zsoil",
                    "BH", "DLMAX", "FLWMAX", "WLM0", "WLM7", "soilbotflx",
                    "nmeltpoint", "tricemethhydr", "soilcolconjtype")

#: static (Python-level, hashable) part of ``params``
_STATIC_KEYS = ("M", "Mice", "ns", "ncols", "dt", "interval", "year0",
                "month0", "day0", "hour0", "backdiff", "backdiff0", "zref",
                "pref", "soilbotflx", "nmeltpoint", "tricemethhydr",
                "soilcolconjtype")


# ---------------------------------------------------------------------------
# small integration helpers (no JAX version existed before B5)
# ---------------------------------------------------------------------------

def _pad(x):
    """1-based padded form: leading zero on the last axis."""
    return jnp.concatenate((jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype), x),
                           axis=-1)


def at_time(rows, time_s, interval=1.0, hour_sec=3600.0):
    """``forcing.at_time`` for a batched (traced) ``time_s``.

    ``rows`` is the ``(T, 8)`` forcing table.  The driver rule is
    ``ti = int(t/span)`` clipped to ``[0, T-2]`` and a linear interpolation
    between records ``ti`` and ``ti+1`` (``driver.f90:1306-1396``).
    """
    rows = jnp.asarray(rows, dtype=jnp.float64)
    span = hour_sec * interval
    ti = jnp.clip(jnp.floor(time_s / span).astype(jnp.int32), 0,
                  rows.shape[0] - 2)
    frac = (time_s - ti * span) / span
    return rows[ti] + frac * (rows[ti + 1] - rows[ti])


def rad_update(extinct, thickness, flux0):
    """Batched ``RAD_UPDATE`` (Beer-Lambert over the 4 spectral bands).

    ``extinct``: ``(L, M, 4)``; ``thickness``: ``(L, M)``; ``flux0``:
    ``(L, 4)``.  Returns ``(flux, integr)`` with ``flux`` ``(L, M+1, 4)`` and
    ``integr`` ``(L, M+1)`` (``integr[i]`` is the total downward flux at level
    ``i``; ``i = 0`` is the surface).
    """
    extinct = jnp.asarray(extinct, dtype=jnp.float64)
    thickness = jnp.asarray(thickness, dtype=jnp.float64)
    flux0 = jnp.asarray(flux0, dtype=jnp.float64)
    att = jnp.exp(-extinct * thickness[..., None])
    L, _nlev, nb = att.shape
    surf = jnp.broadcast_to(flux0[:, None, :], (L, 1, nb))
    flux = jnp.concatenate((surf, jnp.cumprod(att, axis=1) * flux0[:, None, :]),
                           axis=1)
    return flux, jnp.sum(flux, axis=-1)


def water_dens_ts(temp, sal, row0=C.row0):
    """Batched ``WATER_DENS_TS`` (McCutcheon et al. 1993, eos 5)."""
    temp = jnp.asarray(temp, dtype=jnp.float64)
    sal = jnp.asarray(sal, dtype=jnp.float64)
    sg = sal * 1.0e3
    dens = row0 * (1.0 - (temp + A11) * (temp - A14) ** 2
                   / (A12 * (temp + A13)))
    a = (alpha1 - alpha2 * temp + alpha3 * temp ** 2 - alpha4 * temp ** 3
         + alpha5 * temp ** 4)
    b = -beta1 + beta2 * temp - beta3 * temp ** 2
    c = gamma1
    return dens + a * sg + b * sg ** 1.5 + c * sg ** 2


def ddens_dtemp(temp, sal):
    """Batched ``DDENS_DTEMP`` (phys_func.f90:325-360), eos 5."""
    temp = jnp.asarray(temp, dtype=jnp.float64)
    sal = jnp.asarray(sal, dtype=jnp.float64)
    sg = sal * 1.0e3
    out = (-(C.row0 / A12) * (temp - A14) / ((temp + A13) ** 2)
           * ((temp + A13) * ((temp - A14) + 2.0 * (temp + A11))
              - (temp + A11) * (temp - A14)))
    a = (-alpha2 + 2.0 * alpha3 * temp - 3.0 * alpha4 * temp ** 2
         + 4.0 * alpha5 * temp ** 3)
    b = beta2 - 2.0 * beta3 * temp
    return out + a * sg + b * sg ** 1.5


def ddens_dsal(temp, sal):
    """Batched ``DDENS_DSAL`` (phys_func.f90:363-388), eos 5."""
    temp = jnp.asarray(temp, dtype=jnp.float64)
    sal = jnp.asarray(sal, dtype=jnp.float64)
    sg = sal * 1.0e3
    a = (alpha1 - alpha2 * temp + alpha3 * temp ** 2 - alpha4 * temp ** 3
         + alpha5 * temp ** 4)
    b = -beta1 + beta2 * temp - beta3 * temp ** 2
    c = gamma1
    return (a + 1.5 * b * sg ** 0.5 + 2.0 * c * sg) * 1.0e3


def s_diff(Sal1, dt, dhw, dhw0, dhwp, dhwe, lamsal, area_int, area_half,
           ddz, h1, dzeta_int, ddz05, M, Sflux0=0.0):
    """Batched ``S_DIFF`` water branch (``salinity_mod.f90:14-240``).

    All arrays are 1-based padded (``M+2``) except ``ddz05`` (0-based
    ``(M+1,)``).  Returns the new ``Sal2`` (``M+2``).
    """
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    L = Sal1.shape[0]
    a = jnp.zeros((L, M + 2))
    b = jnp.zeros((L, M + 2))
    c = jnp.zeros((L, M + 2))
    d = jnp.zeros((L, M + 2))
    Sflux0 = Sflux0 + (dhwp + dhwe) / dt * (0.0 - Sal1[:, 1])

    i = jnp.arange(2, M + 1)                       # interior points 2..M
    j = i
    ai = 0.5 * (-area_half[:, j - 1] * lamsal[:, j - 1]
                / (ddz[:, j - 1] * h1[:, None] * area_int[:, j])
                + dhw[:, None] * dzeta_int[:, j] * dt_inv05
                - dhw0[:, None] * dt_inv05)
    bi = 0.5 * (-area_half[:, j] * lamsal[:, j]
                / (ddz[:, j] * h1[:, None] * area_int[:, j])
                - dhw[:, None] * dzeta_int[:, j] * dt_inv05
                + dhw0[:, None] * dt_inv05)
    a = a.at[:, i].set(ai)
    b = b.at[:, i].set(bi)
    c = c.at[:, i].set(ai + bi - dt_inv * ddz05[j - 1] * h1[:, None])
    d = d.at[:, i].set(-Sal1[:, j] * dt_inv * ddz05[j - 1] * h1[:, None]
                       + ai * Sal1[:, j - 1] + bi * Sal1[:, j + 1]
                       - (ai + bi) * Sal1[:, j])

    x = 0.5 * (-area_half[:, 1] * lamsal[:, 1]
               / (h1 * ddz[:, 1] * area_int[:, 1]) + dhw0 / (2.0 * dt))
    c = c.at[:, 1].set(x - ddz[:, 1] * h1 / (2.0 * dt))
    b = b.at[:, 1].set(x)
    d = d.at[:, 1].set(-ddz[:, 1] * h1 * Sal1[:, 1] / (2.0 * dt) - Sflux0
                       + x * (Sal1[:, 2] - Sal1[:, 1]))
    x = 0.5 * (-area_half[:, M] * lamsal[:, M]
               / (ddz[:, M] * h1 * area_int[:, M + 1])
               + (dhw - dhw0) / (2.0 * dt))
    c = c.at[:, M + 1].set(x - ddz[:, M] * h1 / (2.0 * dt))
    a = a.at[:, M + 1].set(x)
    d = d.at[:, M + 1].set(-Sal1[:, M + 1] * ddz[:, M] * h1 / (2.0 * dt)
                           - x * (Sal1[:, M + 1] - Sal1[:, M]))

    sol = progonka_jax(a[:, 1:M + 2], b[:, 1:M + 2], c[:, 1:M + 2],
                       d[:, 1:M + 2])
    sol = jnp.maximum(sol, 0.0)
    out = jnp.zeros((L, M + 2))
    return out.at[:, 1:].set(sol)


def diffmin_hs(row, ddz, h1, area, M, backdiff0=-999.0):
    """Batched ``DIFFMIN_HS`` (phys_func.f90:2066-2110).

    ``row`` is 0-based (``M+1``, Fortran 1..M+1), ``ddz`` 0-based ``(M,)``.
    Returns the background conductivity ``(L, M)`` in W/m/K.
    """
    C1 = float(np.float32(8.17e-4))
    C2 = float(np.float32(0.56))
    C3 = float(np.float32(-0.43))
    xmin = float(np.float32(7.0e-5))
    z = backdiff0 if backdiff0 >= 0.0 else C1
    row = jnp.asarray(row, dtype=jnp.float64)
    h1 = jnp.atleast_1d(jnp.asarray(h1, dtype=jnp.float64))
    area = jnp.atleast_1d(jnp.asarray(area, dtype=jnp.float64))
    y = area[:, None] * 1.0e-6
    x = C.g / C.row0 * (row[:, 1:] - row[:, :-1]) / (h1[:, None] * ddz)
    val = z * y ** C2 * jnp.maximum(x, xmin) ** C3
    val = val * (1.0e-4 * C.cw_m_row0 * 0.5)
    return jnp.where(x <= 0.0, 0.0, val)


def advance_case1(state, dhw, dls=0.0):
    """Batched layer-thickness update of case 1 (``lake.f90:1233-1236``)."""
    h2 = jnp.asarray(state["h1"]) + jnp.asarray(dhw)
    ls2 = jnp.asarray(state.get("ls1", jnp.zeros_like(h2))) + jnp.asarray(dls)
    l2 = jnp.zeros_like(h2)
    return dict(h1=h2, l1=l2, ls1=ls2, hs1=jnp.zeros_like(h2),
                snowmass=jnp.zeros_like(h2))


# ---------------------------------------------------------------------------
# parameters / adapters
# ---------------------------------------------------------------------------

def make_params(case, grid, L=None):
    """Build the ``params`` dict from a NumPy ``Params`` (``read_case``) and a
    :class:`lakematrix.grid.Grid`.

    ``params`` mixes static Python values (sizes, switches, the grid) with
    lake-first arrays (setup values, soil column parameters); :func:`_split`
    separates them for tracing.
    """
    from ..physics.soil import comsoilforlake

    L = int(np.atleast_1d(case.L)[0]) if L is None else int(L)
    sp = comsoilforlake(
        soiltype=int(np.atleast_1d(case.soiltype)[0]), ns=case.ns,
        nsoilcols=case.nsoilcols, soil_depth=case.soil_depth,
        tricemethhydr=float(np.atleast_1d(case.tricemethhydr)[0]),
        nmeltpoint=int(np.atleast_1d(case.nmeltpoint)[0]),
        soilbotflx=float(np.atleast_1d(case.soilbotflx)[0]))

    def rep(x):
        """Per-lake value -> ``(L,) + shape[1:]`` (single-lake input repeated)."""
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            return np.full(L, float(a))
        if a.shape[0] == L:
            return a
        return np.repeat(a[:1], L, axis=0)

    return dict(
        # --- static sizes / switches / grid ------------------------------
        M=int(case.M), Mice=int(case.Mice), ns=int(case.ns),
        ncols=int(case.nsoilcols),
        dt=float(np.atleast_1d(case.dt)[0]),
        interval=float(np.atleast_1d(case.interval)[0]),
        year0=int(np.atleast_1d(case.year0)[0]),
        month0=int(np.atleast_1d(case.month0)[0]),
        day0=int(np.atleast_1d(case.day0)[0]),
        hour0=float(np.atleast_1d(case.hour0)[0]),
        backdiff=int(np.atleast_1d(case.backdiff)[0]),
        backdiff0=float(np.atleast_1d(case.backdiff0)[0]),
        zref=2.0, pref=C.pref, grid=grid,
        soilbotflx=float(np.atleast_1d(case.soilbotflx)[0]),
        nmeltpoint=int(np.atleast_1d(case.nmeltpoint)[0]),
        tricemethhydr=float(np.atleast_1d(case.tricemethhydr)[0]),
        soilcolconjtype=int(sp.soilcolconjtype),
        # --- static grid arrays -----------------------------------------
        dzeta_int=np.asarray(grid.dzeta_int, dtype=float),
        dzeta_05int=np.asarray(grid.dzeta_05int, dtype=float),
        ddz=np.asarray(grid.ddz, dtype=float),
        ddz05=np.asarray(grid.ddz05, dtype=float),
        # --- per-lake setup values --------------------------------------
        area_lake=rep(case.area_lake),
        morphometry=rep(case.morphometry),
        morphometry_len=rep(case.morphometry_len).astype(np.int32),
        lakeform=rep(case.lakeform).astype(np.int32),
        cellipt=rep(case.cellipt),
        fetch=rep(case.fetch), phi=rep(case.phi), extwat=rep(case.extwat),
        kor=rep(case.kor), h10=rep(case.h10),
        # --- soil column parameters -------------------------------------
        POR=np.asarray(sp.POR), rosdry=np.asarray(sp.rosdry),
        dzs=np.asarray(sp.dzs), dzss=np.asarray(sp.dzss),
        zsoil=np.asarray(sp.zsoil),
        BH=np.asarray(sp.BH), DLMAX=np.asarray(sp.DLMAX),
        FLWMAX=np.asarray(sp.FLWMAX),
        WLM0=np.asarray(sp.WLM0), WLM7=np.asarray(sp.WLM7),
    )


def _split(params):
    """``(static, dynamic)``: hashable Python values vs traced arrays."""
    stat = {k: params[k] for k in _STATIC_KEYS if k in params}
    stat["grid"] = params["grid"]
    dyn = {k: v for k, v in params.items() if k not in _STATIC_KEYS
           and k != "grid"}
    return stat, dyn


def _empty_diag(M, L):
    return dict(
        Tsurf3=np.zeros(L), dhw=np.zeros(L), dhw0=np.zeros(L),
        eflux0_kinem=np.zeros(L), lsh=np.zeros((L, M + 2)),
        kt=np.zeros((L, M + 2)), row2=np.zeros((L, M + 2)),
        integr=np.zeros((L, M + 1)), tempair=np.zeros(L),
        pressure=np.zeros(L), humair=np.zeros(L), precip=np.zeros(L),
        longwave=np.zeros(L), shortwave=np.zeros(L), uwind=np.zeros(L),
        vwind=np.zeros(L), albedo=np.zeros(L), roughness=np.zeros(L),
        tau_air=np.zeros(L), tau_wav=np.zeros(L), tau_gr=np.zeros(L),
        u_star=np.zeros(L), Bal3=np.zeros(L), iter=np.zeros(L),
        H_mixed_layer=np.zeros(L), maxN=np.zeros(L),
        i_maxN=np.zeros(L, dtype=np.int32), Wedderburn=np.zeros(L),
        LakeNumber=np.zeros(L), RiTherm=np.zeros(L), ThermThick=np.zeros(L))


def from_numpy_state(ss, params=None, grid=None):
    """Convert a NumPy :class:`lakematrix.step.StepState` into a JAX state.

    ``params``/``grid`` are optional (kept for signature compatibility with the
    task statement); every value is read from ``ss``.
    """
    st, soil, ke, mom = ss.state, ss.soil, ss.ke, ss.mom
    L = int(np.atleast_1d(st.h1).size)
    M = int(np.atleast_1d(st.eps).shape[-1] - 1)

    def rep(x):
        """One lake's array -> ``(L,) + shape`` (``build_step_state`` is
        single-lake; the batch replicates that lake)."""
        a = np.asarray(x, dtype=float)
        return np.repeat(a[None, ...], L, axis=0)

    def lake(x):
        """Array that already carries a (broadcastable) lake axis."""
        a = np.asarray(x, dtype=float)
        return np.broadcast_to(a, (L,) + a.shape[1:]) if a.ndim else np.full(L, float(a))

    def stem(key):
        return lake(getattr(ss, key)) if not isinstance(getattr(ss, key), float) \
            else np.full(L, float(getattr(ss, key)))

    state = {
        "water": dict(
            Tw1=lake(st.Tw1), Sal1=lake(st.Sal1), h1=lake(st.h1),
            lamw=lake(st.lamw), E1=lake(st.E), eps1=lake(st.eps),
            u2=rep(mom.u1), v2=rep(mom.v1),
            Elatent=stem("Elatent"),
            Tsurf_prev=np.full(L, float(ss.Tsurf_prev)),
            surftyp_prev=np.full(L, int(ss.surftyp_prev), dtype=np.int32),
            cdmw=stem("cdmw"),
        ),
        "soil": {k: rep(getattr(soil, k)) for k in SOIL_KEYS},
        "ke": dict(e1=rep(ke.e1), eps1=rep(ke.eps1), k2=rep(ke.k2),
                   init_keps=np.full(L, bool(ke.init_keps))),
        "mom": dict(
            k2_old=rep(mom.k2_old), Ri=rep(mom.Ri),
            lsu_water=rep(mom.lsu_water), lsv_water=rep(mom.lsv_water),
            uv=rep(mom.uv), knum=rep(mom.knum), veg_sw=rep(mom.veg_sw),
            velfrict_ice=np.full(L, float(mom.velfrict_ice)),
            alphax=np.full(L, float(mom.alphax)),
            alphay=np.full(L, float(mom.alphay)),
        ),
        "diag": _empty_diag(M, L),
    }
    return state


def dup_state(state, L):
    """Replicate a single-lake (``L == 1``) state ``L`` times along axis 0."""
    return jax.tree_util.tree_map(
        lambda a: np.repeat(np.asarray(a), L, axis=0), state)


def to_numpy(state, params=None):
    """Extract the ``step_case1`` return fields from a JAX state dict.

    Returns ``Tw1``, ``Sal1``, ``Tsoil1``, ``h1``, ``E1``, ``eps1``, ``lamw``,
    ``kt``, ``row2``, ``Elatent``, ``eflux0_kinem``, ``Tsurf3`` (the surface
    temperature of the *last* evaluated step), ``Tsurf_prev`` (the prognostic
    start value, equal to ``Tsurf3`` after a step), ``dhw``, ``dhw0``,
    ``dhwfsoil`` and the ``TURB_SCALES`` diagnostics (``turb``), all as NumPy
    arrays with the leading lake axis kept.
    """
    w, s, d = state["water"], state["soil"], state["diag"]
    turb = {k: np.asarray(d[k]) for k in
            ("H_mixed_layer", "maxN", "i_maxN", "Wedderburn", "LakeNumber",
             "RiTherm", "ThermThick")}
    return dict(
        Tw1=np.asarray(w["Tw1"]), Sal1=np.asarray(w["Sal1"]),
        Tsoil1=np.asarray(s["Tsoil1"])[:, 1:, 1:],
        h1=np.asarray(w["h1"]),
        E1=np.asarray(w["E1"]), eps1=np.asarray(w["eps1"]),
        lamw=np.asarray(w["lamw"]),
        kt=np.asarray(d["kt"])[:, 1:], row2=np.asarray(d["row2"])[:, 1:],
        Elatent=np.asarray(w["Elatent"]),
        eflux0_kinem=np.asarray(d["eflux0_kinem"]),
        Tsurf3=np.asarray(d["Tsurf3"]),
        Tsurf_prev=np.asarray(w["Tsurf_prev"]),
        dhw=np.asarray(d["dhw"]), dhw0=np.asarray(d["dhw0"]),
        dhwfsoil=np.asarray(s["dhwfsoil"]), turb=turb,
    )


# ---------------------------------------------------------------------------
# one case-1 step
# ---------------------------------------------------------------------------

def _extinct(extwat, M):
    """``extinct(1:M, 1:4)`` of ``lake.f90`` (bands 3.13 / extwat / 1.73 / 1e3)."""
    e = jnp.zeros((extwat.shape[0], M, 4))
    e = e.at[:, :, 0].set(3.13)
    e = e.at[:, :, 1].set(extwat[:, None])
    e = e.at[:, :, 2].set(1.73)
    e = e.at[:, :, 3].set(1.087e3)
    return e


def _geometry(h1, p):
    """``BATHYMETRY`` for the batch: water levels + soil-column geometry."""
    M, ncols = p["M"], p["ncols"]
    bath = _bathym.bathymetry_water(
        h1, 0.0, p["area_lake"], p["morphometry"], p["morphometry_len"],
        p["lakeform"], p["cellipt"], p["pref"], p["dzeta_int"],
        p["dzeta_05int"], p["ddz05"])
    soil_b = _bathym.soil_bathymetry_jax(
        h1, p["morphometry"], p["morphometry_len"], p["lakeform"],
        p["cellipt"], p["grid"], ncols, 0.0)
    col = slice(1, ncols + 2)
    geom = dict(
        zsoilcols=soil_b["zsoilcols"], ddz=_pad(p["ddz"]),
        w_lx=bath["Lx"], w_ly=bath["Ly"],
        w_lx_half=bath["Lx_half"], w_ly_half=bath["Ly_half"],
        s_lx=soil_b["Lx"][:, col], s_ly=soil_b["Ly"][:, col])
    return bath, geom, soil_b["area_int"][:, col]


def _step(state, p, nstep):
    """One case-1 step; returns ``(new_state, diagnostics)``.

    ``p`` is the merged ``{**static, **dynamic}`` params dict of :func:`_split`
    (the static part must be concrete: it is read as Python values).
    """
    M, ns, ncols = p["M"], p["ns"], p["ncols"]
    dt = p["dt"]
    w, soil, ke_st, mom = (state["water"], state["soil"], state["ke"],
                           state["mom"])
    L = w["h1"].shape[0]

    # --- BATHYMETRY with the current water depth -------------------------
    bath, geom, Asoil = _geometry(w["h1"], p)
    Tw1_p = _pad(w["Tw1"])
    Sal1_p = _pad(w["Sal1"])
    lamw_p = _pad(w["lamw"])
    area_int = _pad(bath["area_int"])
    area_half = _pad(bath["area_half"])
    # `surface.heatbalance_water` takes every geometry array lake-batched
    ddz_p = jnp.broadcast_to(_pad(p["ddz"])[None, :], (L, M + 1))
    dzeta_int_p = jnp.broadcast_to(_pad(p["dzeta_int"])[None, :], (L, M + 2))
    ddz05_b = jnp.broadcast_to(jnp.asarray(p["ddz05"])[None, :], (L, M + 1))
    dzs_b = jnp.broadcast_to(jnp.asarray(p["dzs"])[None, :], (L, ns + 1))

    # --- forcing ---------------------------------------------------------
    f = at_time(p["rows"], nstep * dt, p["interval"])
    tempair = f[0] - C.Kelvin0
    pressure, longwave, shortwave = f[1], f[2], f[3]
    uwind, vwind, humair, precip = f[4], f[5], f[6], f[7]
    hour = p["hour0"] + nstep * dt / 3600.0
    phi, fetch = p["phi"], p["fetch"]

    # --- SURF_CHAR + RAD_UPDATE ------------------------------------------
    sc = surf_char(p["year0"], p["month0"], p["day0"], hour, phi, fetch,
                   uwind, jnp.full_like(uwind, 1.0e-2))
    albedo, albedo_lw = sc["albedo"], sc["albedo_lw"]
    emis, roughness = sc["emissivity"], sc["roughness"]
    erad = shortwave * (1.0 - albedo)
    flux0 = jnp.asarray(FRACBANDS)[None, :] * erad[:, None]
    _, integr = rad_update(_extinct(p["extwat"], M),
                           p["ddz05"][:M][None, :] * w["h1"][:, None], flux0)

    # --- water-depth increments (lake.f90 order: dhw before SOILCOLSTEMP)
    dhwp = precip * dt
    dhwe = -w["Elatent"] / (C.Lwv * C.row0) * dt
    dhwtrib = (p["h10"] - w["h1"]) / 10.0
    dhw = dhwp + dhwe + soil["dhwfsoil"] + dhwtrib
    dhw0 = dhwp + dhwe

    # --- side soil columns (SOIL_COND_HEAT_COEF + SOILCOLSTEMP) ----------
    p_soil = {k: p[k] for k in _SOIL_PARAM_KEYS}
    p_soil["nsoilcols"] = ncols
    p_soil["pressure"] = pressure
    soil_in = dict(soil)
    soil_in["Tw1"] = Tw1_p
    soil_in["h1"] = w["h1"]
    soil_out = soilcolstemp_water(soil_in, p_soil, geom, dt)

    # --- LATERHEAT -------------------------------------------------------
    lsh = laterheat_water(geom["zsoilcols"], w["h1"], bath["area_int"],
                          bath["area_half"], p["ddz05"],
                          soil_out["soilflux"], M, ncols, Asoil)

    # --- MOMENTUM --------------------------------------------------------
    row = water_dens_ts(w["Tw1"], w["Sal1"])
    zero = jnp.zeros_like(w["h1"])
    mom_state = dict(
        u1=w["u2"], v1=w["v2"], E1=w["E1"], eps1=w["eps1"],
        k2=ke_st["k2"][:, 1:], k2_old=mom["k2_old"], row=row,
        Ri=mom["Ri"], knum=mom["knum"], veg_sw=mom["veg_sw"], u_star=zero,
        Tw1=w["Tw1"], Sal1=w["Sal1"],
        lsu_water=mom["lsu_water"], lsv_water=mom["lsv_water"],
        uv=mom["uv"], h1=w["h1"], l1=zero, ls1=zero, dhw=dhw, dhw0=dhw0,
        alphax=mom["alphax"], alphay=mom["alphay"], velfrict_bot=zero,
        velfrict_ice=mom["velfrict_ice"])
    mom_par = dict(
        ddz=p["ddz"], ddz05=p["ddz05"], dzeta_int=p["dzeta_int"],
        dzeta_05int=p["dzeta_05int"], area_int=bath["area_int"],
        area_half=bath["area_half"], Lx=bath["Lx"], Ly=bath["Ly"],
        Lx_half=bath["Lx_half"], Ly_half=bath["Ly_half"],
        depth_area=p["morphometry"], kor=p["kor"], uwind=uwind,
        vwind=vwind, cdmw=w["cdmw"], zref=jnp.full_like(w["h1"], p["zref"]),
        fetch=fetch)
    mom_out = _momentum_solver(mom_state, mom_par, dt)

    # --- T_SOLVER (HEATBALANCE + T_DIFF) ---------------------------------
    zeros2 = jnp.zeros((L, M + 2))
    ctx = dict(
        Tw1=Tw1_p, Tsoil1=soil_out["Tsoil1"], lamw=lamw_p,
        area_int=area_int, area_half=area_half, h1=w["h1"], ddz=ddz_p,
        dhw=dhw, dhw0=dhw0, PEMF=zeros2, pt_down_f=zeros2,
        lamsoil=soil_out["lamsoil"], csoil=soil_out["csoil"],
        rosoil=soil_out["rosoil"], dzs=dzs_b,
        radwater_integr=integr, lsh_water=lsh, dzeta_int=dzeta_int_p,
        ddz05=ddz05_b, dt=dt, fetch=fetch, uwind=uwind, vwind=vwind,
        u=mom_out["u2"][:, 0], v=mom_out["v2"][:, 0], tempair=tempair,
        pressure=pressure, humair=humair, zref=p["zref"],
        roughness=roughness, relhums=sc["relhums"], aM=sc["aM"], bM=sc["bM"],
        l1=zero, longwave=longwave, shortwave=shortwave, netrad=-1.0,
        cloud=-1.0, albedo=albedo, albedo_lw=albedo_lw, emissivity=emis)
    sol = _solver(False)(ctx, w["Tsurf_prev"],
                         jnp.full_like(w["surftyp_prev"], 4),
                         jnp.where(p["first"], 0, w["surftyp_prev"]))
    Tw2 = sol["state"]["Tw2"]                       # (L, M+1)
    Tsoil2_p = _pad(sol["state"]["Tsoil2"])         # (L, ns+1)
    Elatent = sol["state"]["Elatent"]
    eflux0_kinem = sol["state"]["eflux0_kinem"]

    # --- deepest soil column (SOILFORLAKE) -------------------------------
    soil_d = dict(soil_out)
    soil_d["Tsoil2"] = Tsoil2_p
    soil_d["h1"] = w["h1"]
    soil_d["Tw1"] = Tw1_p
    soil_f = _soilforlake(soil_d, p_soil, Tsoil2_p, dt, is_=ncols)

    # --- S_DIFF + new density --------------------------------------------
    lamsal = _pad((w["lamw"] - C.lamw0) / C.cw_m_row0 * C.alsal + C.lamsal0)
    Sal2 = s_diff(Sal1_p, dt, dhw, dhw0, dhwp, dhwe, lamsal, area_int,
                  area_half, ddz_p, w["h1"], dzeta_int_p, p["ddz05"], M)
    Tw2_p = _pad(Tw2)
    row2 = water_dens_ts(Tw2_p, Sal2)

    # --- K_EPSILON --------------------------------------------------------
    ke_in = dict(
        e1=ke_st["e1"], eps1=ke_st["eps1"], k2=ke_st["k2"],
        init_keps=ke_st["init_keps"],
        tw1=Tw1_p, tw2=Tw2_p, sal1=Sal1_p, sal2=Sal2,
        u1=_pad(w["u2"]), v1=_pad(w["v2"]),
        u2=_pad(mom_out["u2"]), v2=_pad(mom_out["v2"]),
        row=_pad(row), row2=row2, lamw=lamw_p, lamsal=lamsal,
        h1=w["h1"], l1=zero, dhw=dhw, dhw0=dhw0,
        vol=bath["vol"], botar=bath["botar"], b0=zero,
        tau_air=mom_out["tau_air"], tau_wav=mom_out["tau_wav"],
        tau_i=mom_out["tau_i"], tau_gr=mom_out["tau_gr"],
        uwind=uwind, vwind=vwind, fetch=fetch, roughness=roughness,
        ddz=_pad(p["ddz"]), ddz05=p["ddz05"], dzeta_int=dzeta_int_p,
        dzeta_05int=_pad(p["dzeta_05int"]), area_int=area_int,
        area_half=area_half)
    ke_out = _k_epsilon(ke_in, dt, dt_keps=dt, nstep_keps=1,
                        params=KEpsilonParams(M=M))

    # --- eddy diffusivity / TURB_SCALES ----------------------------------
    if p["backdiff"] == 1:
        lamw_back = diffmin_hs(row, p["ddz"], w["h1"], p["area_lake"], M,
                               p["backdiff0"])
    else:
        lamw_back = jnp.zeros_like(w["lamw"])
    lamw_new = C.lamw0 + jnp.maximum(ke_out["kt"][:, 1:M + 1], 0.0) + lamw_back
    k_turb = (-0.5 * lamw_new / C.cw_m_row0
              * (Tw2_p[:, 2:] + w["Tw1"][:, 1:]
                 - Tw2_p[:, 1:-1] - w["Tw1"][:, :-1])
              / (p["ddz"][None, :] * w["h1"][:, None]))
    turb_density_flux = ddens_dtemp(w["Tw1"][:, 0], 0.0) * eflux0_kinem
    ts = _turb_scales(
        row, p["ddz"], p["ddz"][:-1] + p["ddz"][1:], p["dzeta_int"],
        p["dzeta_05int"], w["h1"], M, Tw2, Sal2[:, 1:],
        mom_out["u2"], mom_out["v2"], lamw_new, lamsal, k_turb,
        jnp.zeros_like(w["lamw"]), eflux0_kinem, turb_density_flux,
        C.g / C.row0 * turb_density_flux,
        mom_out["tau_air"] - mom_out["tau_wav"], p["kor"], integr,
        bath["area_int"], bath["Lx"], bath["Ly"], bath["vol"],
        p["dzeta_int"][None, :] * w["h1"][:, None], p["ddz05"],
        p["ddens_dtemp0"], p["ddens_dsal0"])

    # --- layer update ----------------------------------------------------
    lay = advance_case1(dict(h1=w["h1"], ls1=zero), dhw)

    # every per-step diagnostic is lake-vector shaped (a scalar would change
    # the state pytree after the first step and force a recompilation)
    vec = lambda x: jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (L,))

    # --- assemble the new state ------------------------------------------
    new = {
        "water": dict(
            Tw1=Tw2, Sal1=Sal2[:, 1:], h1=lay["h1"], lamw=lamw_new,
            E1=ke_out["e1"][:, 1:], eps1=ke_out["eps1"][:, 1:],
            u2=mom_out["u2"], v2=mom_out["v2"], Elatent=Elatent,
            Tsurf_prev=sol["Tsurf3"],
            surftyp_prev=jnp.full_like(w["surftyp_prev"], 4),
            cdmw=sol["state"]["cdmw"]),
        "soil": {k: soil_f[k] for k in SOIL_KEYS},
        "ke": dict(e1=ke_out["e1"], eps1=ke_out["eps1"], k2=ke_out["k2"],
                   init_keps=ke_out["init_keps"]),
        "mom": dict(
            k2_old=mom_out["k2_old"], Ri=mom_out["Ri"],
            lsu_water=mom_out["lsu_water"], lsv_water=mom_out["lsv_water"],
            uv=mom_out["uv"], knum=mom_out["knum"],
            veg_sw=mom_out["veg_sw"], velfrict_ice=mom_out["velfrict_ice"],
            alphax=mom["alphax"], alphay=mom["alphay"]),
        "diag": dict(
            Tsurf3=sol["Tsurf3"], dhw=dhw, dhw0=dhw0,
            eflux0_kinem=eflux0_kinem, lsh=lsh, kt=ke_out["kt"],
            row2=row2, integr=integr, tempair=vec(tempair),
            pressure=vec(pressure), humair=vec(humair), precip=vec(precip),
            longwave=vec(longwave), shortwave=vec(shortwave),
            uwind=vec(uwind), vwind=vec(vwind), albedo=albedo,
            roughness=roughness, tau_air=mom_out["tau_air"],
            tau_wav=mom_out["tau_wav"], tau_gr=mom_out["tau_gr"],
            u_star=mom_out["u_star"], Bal3=sol["Bal3"],
            iter=vec(sol["iter"]),
            H_mixed_layer=ts["H_mixed_layer"], maxN=ts["maxN"],
            i_maxN=ts["i_maxN"].astype(jnp.int32),
            Wedderburn=ts["Wedderburn"], LakeNumber=ts["LakeNumber"],
            RiTherm=ts["RiTherm"], ThermThick=ts["ThermThick"]),
    }
    return new, new["diag"]


_SOLVERS = {}


def _solver(first=None):
    """``T_SOLVER`` closure bound to the ``nstep > 1`` branch.

    ``tsolver.t_solver`` uses the model step only through its start-up branch
    ``Tsurf2 = (nstep > 1) & (surftyp == surftyp_prev) ? Tsurf_prev - 10 : -90``.
    Every step but the first satisfies ``nstep > 1``, and on the first step the
    driver's ``surftyp_prev`` is 0 (``build_step_state``) while ``surftyp`` is 4,
    so the guard is false anyway.  :func:`_step` therefore forces
    ``surftyp_prev = 0`` when ``first`` is set (exactly the ``nstep == 1``
    branch) and the *same* compiled solver serves the entire run.
    """
    if "solver" not in _SOLVERS:
        _SOLVERS["solver"] = make_solver(2)
    return _SOLVERS["solver"]


# ---------------------------------------------------------------------------
# compiled entry points
# ---------------------------------------------------------------------------

_JIT_CACHE = {}


def _signature(state):
    """Shape/dtype layout of a state dict (must not change across steps)."""
    return (jax.tree_util.tree_structure(state),
            [(tuple(np.shape(x)), np.dtype(np.asarray(x).dtype).str)
             for x in jax.tree_util.tree_leaves(state)])


def _check_state(old, new):
    """Raise if a step changed the state pytree (would silently recompile)."""
    so, sn = _signature(old), _signature(new)
    if so == sn:
        return
    bad = [(o, n) for o, n in zip(so[1], sn[1]) if o != n]
    raise RuntimeError("step_case1_jax changed the state layout: "
                       f"{len(so[1])} -> {len(sn[1])} leaves, "
                       f"first differences {bad[:3]}")


def _compiled(params):
    """Return ``(jitted_step, static_params)`` for this params signature.

    The static part (sizes, switches, grid) is closed over; the arrays are
    ``jax.jit`` operands.  ``ddens_dtemp0``/``ddens_dsal0`` are the
    ``SET_DENS_CONSTS(5)`` values ``step.step_case1`` hands to ``TURB_SCALES``.
    """
    stat, _dyn = _split(params)
    stat = dict(stat)
    # SET_DENS_CONSTS(5) as `step.step_case1` passes them to TURB_SCALES.
    # (``ddens_dsal(15, 0)`` really is 777.83, but the NumPy reference passes
    # 0.7778285184375; it only multiplies the salinity difference, which is 0
    # for the fresh-water reference cases, so the value is inert -- keep the
    # reference's literal for bit-compatibility.)
    stat["ddens_dtemp0"] = -0.15063272953663126
    stat["ddens_dsal0"] = 0.7778285184375
    key = (tuple(sorted((k, v) for k, v in stat.items() if k != "grid")),
           id(stat["grid"]))
    fn = _JIT_CACHE.get(key)
    if fn is None:
        def step_fn(state, dyn, nstep):
            p = dict(stat)
            p.update(dyn)
            return _step(state, p, nstep)

        fn = jax.jit(step_fn)
        _JIT_CACHE[key] = fn
    return fn, stat


def step_case1_jax(state, params, rows, nstep=1, first=None):
    """One full case-1 step for a batch of lakes.

    Parameters
    ----------
    state : dict
        Batched state as documented in the module docstring (see
        :func:`from_numpy_state`).
    params : dict
        Product of :func:`make_params` (``rows`` is taken from the argument).
    rows : array_like
        The ``(T, 8)`` forcing table (read at ``nstep*dt``).
    nstep : int or traced scalar
        1-based model step; used for the forcing time and the ``T_SOLVER``
        start-up branch.
    first : bool, optional
        Force the ``nstep <= 1`` ``T_SOLVER`` start-up branch.  Required when
        ``nstep`` is traced (inside ``lax.scan``).

    Returns
    -------
    dict
        The new state (same keys as ``state``); the per-step diagnostics are
        in ``state["diag"]``.
    """
    if first is None:
        if isinstance(nstep, (int, np.integer, float, np.floating)):
            first = float(nstep) <= 1.0
        else:
            raise ValueError("nstep is a traced value: pass first=True/False")
    params = {**params, "rows": rows}
    fn, _stat = _compiled(params)
    _stat, dyn = _split(params)
    dyn = {**dyn, "first": jnp.asarray(bool(first))}
    new, _diag = fn(state, dyn, jnp.asarray(nstep, dtype=jnp.float64))
    _check_state(state, new)
    return new


def run_case1_jax(params, rows, nsteps, state=None, nstep0=1, first=True):
    """Run ``nsteps`` case-1 steps, ``lax.scan``-ing over the state dict.

    The first step is evaluated separately (its ``T_SOLVER`` start-up branch is
    forced through ``surftyp_prev = 0``, see :func:`_solver`); steps
    ``2 .. nsteps`` run in a single :func:`jax.lax.scan`.  Returns
    ``(final_state, diagnostics)``, where ``diagnostics`` holds the per-step
    entries of ``state["diag"]`` stacked along a leading step axis.
    """
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")
    state = state if state is not None else params.get("state0")
    if state is None:
        raise ValueError("run_case1_jax needs a state (argument or "
                         "params['state0'])")
    state_in = state
    params = {**params, "rows": rows}
    fn, _stat = _compiled(params)
    _stat, dyn = _split(params)
    dyn1 = {**dyn, "first": jnp.asarray(bool(first))}
    dyn2 = {**dyn, "first": jnp.asarray(False)}

    state, d0 = fn(state, dyn1, jnp.asarray(float(nstep0)))
    _check_state(state_in, state)
    if nsteps == 1:
        return state, jax.tree_util.tree_map(lambda x: x[None], d0)

    def body(carry, nstep):
        return fn(carry, dyn2, nstep)

    state, rest = jax.lax.scan(
        body, state,
        jnp.arange(nstep0 + 1, nstep0 + nsteps, dtype=jnp.float64))
    _check_state(state_in, state)
    diags = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((a[None], b), axis=0), d0, rest)
    return state, diags
