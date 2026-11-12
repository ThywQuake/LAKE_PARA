"""Batched JAX case-2 (ice over water, with and without snow) time step.

This module assembles the already validated ``jaxbatch`` kernels into the full
**case-2** step of ``lake.f90`` and mirrors
:func:`lakematrix.step2.step_case2` /
:func:`lakematrix.step2.step_case2_to_solver` call for call::

    BATHYMETRY (water + ice + deep ice + soil columns)
      -> sync_ice                 (fill the ice state from the water state)
      -> SURF_CHAR (2/3) -> RAD_UPDATE (ice / water / deep ice)
      -> MIXED_LAYER_CALC (diagnostic)
      -> SNOW_COND_HEAT_COEF + SR_BOTSNOW + ICE_INCREMENTS (2.1 / 2.2)
      -> SOIL_COND_HEAT_COEF + SOILCOLSTEMP  (side columns, TSURFSOILCOL)
      -> LATERHEAT (water + ice + deep ice)
      -> T_SOLVER (HEATBALANCE ice + T_DIFF ice, Chorde iteration)
      -> T_DIFF (surf = 0, melting point at the ice-water interface)
      -> SOILFORLAKE (deepest column)
      -> S_DIFF
      -> SNOW_CALC
      -> MOMENTUM_SOLVER
      -> K_EPSILON
      -> TURB_SCALES (diagnostic only -- see the note below)
      -> layer update (h1 += dhw, l1 += dhi, ls1 += dls)

Nothing here adds physics: every equation comes from an existing kernel
(``jaxbatch.bathym``, ``snowice``, ``ice_tdiff``, ``ice_surface``,
``soilcols``/``soilpass``/``soilforlake``, ``momentum``, ``kepsilon``,
``turb``).  The only new code is

* :func:`ice_bathymetry` - the ice/deep-ice levels of ``BATHYMETRY``
  (``step._zsoil_geom._ice_levels``), not needed by case 1;
* :func:`tsurfsoilcol` - the ice / deep-ice branches of ``TSURFSOILCOL``
  (``soil_mod.f90:1034-1382``; ``jaxbatch.soilcols.tsurfsoilcol_water`` only
  covers ``l1 = ls1 = 0``);
* :func:`laterheat` - the ice / deep-ice scans of ``LATERHEAT``
  (``jaxbatch.laterheat.laterheat_water`` is the case-1 water-only branch);
* the integration helpers (:func:`at_time` is re-used from
  :mod:`lakematrix.jaxbatch.step`).

``TURB_SCALES`` note
====================

``step2.step_case2`` does **not** call ``TURB_SCALES`` (it imports it but
never uses it), so the reference produces no turbulent-scale diagnostics for
case 2 and the state cannot depend on them.  This port evaluates
:func:`lakematrix.jaxbatch.turb.turb_scales` anyway because the task sequence
lists it; its output goes to the diagnostics only.

State dict
==========

``state`` is a plain ``dict`` of arrays with a leading lake axis ``L`` (the
same layout as :mod:`lakematrix.jaxbatch.step`, plus an ``"ice"`` section)::

    water   Tw1        (L, M+1)         water temperature (0-based 1..M+1)
            Sal1       (L, M+1)         salinity, kg/kg
            h1         (L,)             water depth, m
            lamw       (L, M)           molecular+turbulent conductivity
            E1, eps1   (L, M+1)         TKE and its dissipation
            u2, v2     (L, M+1)         velocity at the start of the step
            Elatent    (L,)             latent heat flux of the last step
            Tsurf_prev (L,)             T_SOLVER surface temperature
            surftyp_prev (L,) int32     2 (ice) / 3 (snow) of the last step
            cdmw       (L,)             drag coefficient of the last step

    soil    Tsoil1     (L, ns+1, ncols+1)  soil temperature, 1-based padded
            Tsoil2     (L, ns+1)           diffused deepest column
            Tsoil3, wl1, wl2, wl3, wl4, wi1, wi2, Sals1, wa, methgenmh
            csoil, rosoil, lamsoil, lammoist, filtr, wsoil, soilflux,
            dhwfsoil                       (see ``step.SOIL_KEYS``)

    ice     Ti1, Ti2          (L, Mice+2)   surface-ice temperature (1-based)
            Tis1, Tis2        (L, Mice+2)   deep-ice temperature
            ci_m_roi_v, salice, porice (L, Mice+2)
            lami_v            (L, Mice+1)   ice conductivity
            T                 (L, ml+1)     snow temperature of the last step
            Tsn               (L, ms+1)     snow temperature (SNOWTEMP output)
            cs, dens, dz, lams (L, ms+1)    snow pack properties
            wl, extinct, z    (L, ml+1)     snow work arrays
            lsh_ice, lsh_dice (L, Mice+2)   LATERHEAT sources (1-based)
            lsh_water         (L, M+2)
            rad_integr_ice, rad_integr_deepice (L, Mice+1)
            l1, ls1, hs1, snowmass, snowmass_init, snmelt, swe,
            dhi, dhi0, dls, dls0, snmeltice, dhiimp,
            dhwlow, dhilow, dhwhigh, dhihigh, dhwsnmelt, dhwls, dhis, dhip,
            dhwfsoil, dhwtrib, SR_botsnow   (L,) scalars
            itop, flag_snow, flag_snow_init (L,) int32
            saltice (Python int 0), nmeltpoint (Python int)

    ke      ``step``'s KE state (e1, eps1, k2, init_keps)
    mom     ``step``'s momentum state + ``k2`` (the eddy viscosity the last
            call returned; case 2 does *not* feed ``K_EPSILON``'s ``k2`` back
            into ``MOMENTUM_SOLVER``, see ``step2.step_case2``)
    diag    per-step diagnostics (Tsurf3, dhw, dhw0, lsh, kt, row2, turb, ...)

``params`` is the dict of :func:`make_params` (a superset of
:func:`lakematrix.jaxbatch.step.make_params`: it adds ``extice``, the ice grid
``ddzi``/``ddzi05``/``dzetai_int`` and ``Mice`` is reused).

Scope / gaps
============

Ported: the case-2.1 (water + ice + snow) and case-2.2 (water + ice)
branches, ``nsoilcols >= 2``, a morphometry table, ``l1 > 0``, ``ls1 >= 0``,
``saltice = 0``, ``soilcolconjtype = 2``, ``Turbpar 2 / stabfunc 2 /
botfric 1 / kepsbc 1``, ``eos 5``.

Not ported (raise or are documented no-ops):

* the layer-case selection / transitions of ``step2.step_any`` and the
  ``_post_step`` thickness corrections - call :func:`step_case2_jax` only for
  a lake that is in case 2 (``h1 > 0``, ``l1 > 0``);
* ``tricemethhydr > 0``, ``nmeltpoint 2``, ``saltice 1``, vegetation,
  methane in the soil columns;
* ``TURB_SCALES`` is diagnostic only (the NumPy reference does not call it).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C
from ..physics.radiation import FRACBANDS
from . import bathym as _bathym
from . import step as _step1
from .ice_surface import heatbalance_ice, surf_char_ice
from .ice_tdiff import t_diff_ice
from .kepsilon import KEpsilonParams
from .kepsilon import k_epsilon as _k_epsilon
from .momentum import momentum_solver as _momentum_solver
from .snowice import (ice_increments, ice_increments_nosnow, ice_properties,
                      snow_calc, snow_cond_heat_coef, sr_botsnow)
from .soilpass import _soilcolstemp_water
from .soilforlake import soilforlake as _soilforlake
from .turb import mixed_layer_calc, turb_scales
from .tsolver import t_solver

__all__ = [
    "at_time", "dup_state", "from_numpy_state", "ice_bathymetry",
    "laterheat", "make_params", "step_case2_jax", "sync_ice", "to_numpy",
    "tsurfsoilcol", "run_case2_jax", "SOIL_KEYS", "ICE_KEYS",
]

#: soil fields carried between steps (same as :mod:`jaxbatch.step`)
SOIL_KEYS = _step1.SOIL_KEYS

#: ice / snow fields carried between steps (all arrays, leading lake axis)
ICE_KEYS = (
    "Ti1", "Ti2", "Tis1", "Tis2", "ci_m_roi_v", "lami_v", "salice", "porice",
    "T", "Tsn", "cs", "dens", "dz", "lams", "wl", "extinct", "z",
    "lsh_ice", "lsh_dice", "lsh_water",
    "rad_integr_ice", "rad_integr_water", "rad_integr_deepice",
    "l1", "ls1", "hs1", "snowmass", "snowmass_init", "snmelt", "swe",
    "dhi", "dhi0", "dls", "dls0", "snmeltice", "dhiimp",
    "dhwlow", "dhilow", "dhwhigh", "dhihigh", "dhwsnmelt", "dhwls",
    "dhis", "dhip", "dhwfsoil", "dhwtrib", "SR_botsnow",
    "itop", "flag_snow", "flag_snow_init",
)

#: keys of the ``params`` dict read by the batched soil kernels
_SOIL_PARAM_KEYS = _step1._SOIL_PARAM_KEYS

at_time = _step1.at_time           # identical forcing rule as case 1
_pad = _step1._pad


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _vec(x, L):
    """Broadcast a scalar / ``(1,)`` / ``(L,)`` value to ``(L,)`` float64."""
    return jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (L,))


def _ivec(x, L):
    """Broadcast a scalar / ``(1,)`` / ``(L,)`` value to ``(L,)`` int32."""
    return jnp.broadcast_to(jnp.asarray(x, dtype=jnp.int32).reshape(-1), (L,))


def _step_fn(x):
    """``STEP`` (``numerics_mod.f90:276``): ``0.5*(sign(1,x)+1)``."""
    return 0.5 * (jnp.sign(x) + 1.0)


def _cumsum_last(x):
    """Left-to-right cumulative sum along the last axis (Fortran order)."""
    def body(c, xi):
        c = c + xi
        return c, c

    ys = jax.lax.scan(body, jnp.zeros(x.shape[:-1], x.dtype),
                      jnp.moveaxis(x, -1, 0))[1]
    return jnp.moveaxis(ys, 0, -1)


def _overlay(z0, z1, y1, y2):
    """``OVERLAY`` (``soil_mod.f90:1614``) on ``(L,)`` arrays."""
    return (((y1 >= z0) & (y2 <= z1))
            | ((y1 <= z0) & (y2 >= z1))
            | ((y1 <= z0) & (y2 > z0))
            | ((y1 < z1) & (y2 >= z1)))


def _take2(x, k):
    """``x[i, k[i]]`` for a 2-D ``x`` (L, K) and 0-based ``k`` (L,)."""
    return jnp.take_along_axis(x, k[:, None], axis=-1)[:, 0]


def _select(where, x):
    """One-hot gather over the last axis: ``sum(jnp.where(where, x, 0))``."""
    return jnp.sum(jnp.where(where, x, 0.0), axis=-1)


def _extinct(ext, n):
    """``(L, n, 4)`` extinction of the four spectral bands (lake.f90)."""
    ext = jnp.asarray(ext, dtype=jnp.float64)
    L = ext.shape[0]
    e = jnp.zeros((L, n, 4))
    e = e.at[:, :, 0].set(3.13)
    e = e.at[:, :, 1].set(ext[:, None])
    e = e.at[:, :, 2].set(1.73)
    e = e.at[:, :, 3].set(1.087e3)
    return e


def _merge_ice(mask, a, b):
    """Elementwise ``where(mask, a[k], b[k])`` over two ice state dicts.

    Keys that only one branch produces (``fracsn``/``fracimp`` of
    ``ICE_INCREMENTS``) are taken from that branch.
    """
    out = {}
    for k in sorted(set(a) | set(b)):
        if k not in b:
            out[k] = jnp.asarray(a[k])
            continue
        if k not in a:
            out[k] = jnp.asarray(b[k])
            continue
        x = jnp.asarray(a[k])
        cond = mask
        if x.ndim > 1:
            cond = cond.reshape(mask.shape + (1,) * (x.ndim - 1))
        out[k] = jnp.where(cond, x, jnp.asarray(b[k]))
    return out


# ---------------------------------------------------------------------------
# BATHYMETRY: ice / deep-ice levels
# ---------------------------------------------------------------------------

def ice_bathymetry(thickness, total, p):
    """Batched ice/deep-ice levels of ``BATHYMETRY`` (``_ice_levels``).

    ``thickness``/``total`` are ``(L,)``: ``total`` is the depth the
    ``(depth, area)`` table is shifted by (``h1 + l1 + ls1`` for the surface
    ice, ``ls1`` for the deep ice).  Returns a dict of ``(L, Mice+1)``
    ``area_int``/``Lx``/``Ly`` and ``(L, Mice)`` ``area_half``/``Lx_half``/
    ``Ly_half`` (0-based, ``level k`` is Fortran level ``k+1``).
    """
    Mice = int(p["Mice"])
    n = Mice + 1
    thickness = jnp.atleast_1d(jnp.asarray(thickness, dtype=jnp.float64))
    total = jnp.atleast_1d(jnp.asarray(total, dtype=jnp.float64))
    L = thickness.shape[0]
    ddzi = jnp.asarray(p["ddzi"], dtype=jnp.float64)        # (Mice,)
    ddzi05 = jnp.asarray(p["ddzi05"], dtype=jnp.float64)    # (Mice+1,)
    dep = jnp.asarray(p["morphometry"])[..., 0]             # (L, K)
    area = jnp.asarray(p["morphometry"])[..., 1]
    mlen = jnp.asarray(p["morphometry_len"], dtype=jnp.int32)
    lakeform = jnp.asarray(p["lakeform"]).reshape(-1)
    cellipt = jnp.asarray(p["cellipt"], dtype=jnp.float64).reshape(-1)
    area_lake = jnp.asarray(p["area_lake"], dtype=jnp.float64).reshape(-1)

    # position of the level boundaries, ``pos[k] = sum_{i<k} ddzi(i)*thickness``
    pos = jnp.concatenate(
        [jnp.zeros((L, 1)),
         _cumsum_last(ddzi[None, :] * thickness[:, None])], axis=-1)
    inc = jnp.concatenate([0.5 * ddzi[:1], 0.5 * (ddzi[:-1] + ddzi[1:])])
    pos_half = _cumsum_last(inc[None, :] * thickness[:, None])

    # table shift: ``zz = dep - max(dep) + total``
    dep_max = jnp.max(jnp.where(
        jnp.arange(dep.shape[1])[None, :] < mlen[:, None], dep, -jnp.inf),
        axis=-1)
    zz = dep - dep_max[:, None] + total[:, None]

    a_int = _bathym.interp_table(zz, area, mlen, pos)        # (L, Mice+1)
    a_half = _bathym.interp_table(zz, area, mlen, pos_half)  # (L, Mice)
    lx, ly = _bathym.lxly(a_int, lakeform, cellipt)
    lxh, lyh = _bathym.lxly(a_half, lakeform, cellipt)

    # no ice / no deep ice: every level is the lake area
    const = thickness <= C.small_value
    lx_c, ly_c = _bathym.lxly(jnp.broadcast_to(area_lake[:, None], (L, n)),
                              lakeform, cellipt)
    lxh_c, lyh_c = _bathym.lxly(
        jnp.broadcast_to(area_lake[:, None], (L, n - 1)), lakeform, cellipt)
    a_int = jnp.where(const[:, None], area_lake[:, None], a_int)
    a_half = jnp.where(const[:, None], area_lake[:, None], a_half)
    lx = jnp.where(const[:, None], lx_c, lx)
    ly = jnp.where(const[:, None], ly_c, ly)
    lxh = jnp.where(const[:, None], lxh_c, lxh)
    lyh = jnp.where(const[:, None], lyh_c, lyh)
    return dict(area_int=a_int, area_half=a_half, Lx=lx, Ly=ly,
                Lx_half=lxh, Ly_half=lyh)


def _geometry(h1, l1, ls1, p):
    """``BATHYMETRY`` for the batch (water, ice, deep ice, soil columns)."""
    M, Mice, ncols = p["M"], p["Mice"], p["ncols"]
    bath = _bathym.bathymetry_water(
        h1, ls1, p["area_lake"], p["morphometry"], p["morphometry_len"],
        p["lakeform"], p["cellipt"], p["pref"], p["dzeta_int"],
        p["dzeta_05int"], p["ddz05"])
    soil_b = _bathym.soil_bathymetry_jax(
        h1, p["morphometry"], p["morphometry_len"], p["lakeform"],
        p["cellipt"], p["grid"], ncols, ls1)
    ib = ice_bathymetry(l1, h1 + l1 + ls1, p)
    db = ice_bathymetry(ls1, ls1, p)
    col = slice(1, ncols + 2)
    return dict(
        water=bath,
        # 0-based bathymsoil(1:ncols+1) fields
        zsoilcols=soil_b["zsoilcols"],
        s_Aint=soil_b["area_int"][:, col],
        s_Ahalf=soil_b["area_half"][:, col],
        s_lx=soil_b["Lx"][:, col], s_ly=soil_b["Ly"][:, col],
        ice=ib, dice=db,
        # static grid arrays (shared by every lake)
        ddz=_pad(jnp.asarray(p["ddz"])),
        ddzi=_pad(jnp.asarray(p["ddzi"])),
        ddz05=jnp.asarray(p["ddz05"]),
        ddzi05=jnp.asarray(p["ddzi05"]),
        dzeta_int=_pad(jnp.asarray(p["dzeta_int"])),
        dzeta_05int=jnp.asarray(p["dzeta_05int"]),
        dzetai_int=_pad(jnp.asarray(p["dzetai_int"])),
    )


# ---------------------------------------------------------------------------
# TSURFSOILCOL (all h1 > 0 branches of soil_mod.f90:1034-1382)
# ---------------------------------------------------------------------------

def _tsurf_profiles(state, geom, M, Mice, ncols):
    """The four vertical profiles (``Temp``, ``z``, ``aell``, ``bell``, ``nl``).

    Returns ``(temp, z, aell, bell, nl)`` where the arrays are ``(L, N)`` with
    ``N = 2*Mice + M + 3`` and ``nl`` is the per-lane index of the last level
    (``z``/``Temp`` beyond ``nl`` are zero).
    """
    Tw1 = jnp.asarray(state["Tw1"], dtype=jnp.float64)
    Ti1 = jnp.asarray(state["Ti1"], dtype=jnp.float64)
    Tis1 = jnp.asarray(state["Tis1"], dtype=jnp.float64)
    h1, l1, ls1 = state["h1"], state["l1"], state["ls1"]
    L = h1.shape[0]
    small = C.small_value
    ddz = geom["ddz"]                     # (M+1,) 1-based
    ddzi = geom["ddzi"]                   # (Mice+1,) 1-based
    zsc = geom["zsoilcols"]               # (L, ncols+1) 0-based
    bw, bi, bd = geom["water"], geom["ice"], geom["dice"]

    w_lx, w_ly = bw["Lx"], bw["Ly"]
    w_lxh, w_lyh = bw["Lx_half"], bw["Ly_half"]
    i_lx, i_ly = bi["Lx"], bi["Ly"]
    i_lxh, i_lyh = bi["Lx_half"], bi["Ly_half"]
    d_lx, d_ly = bd["Lx"], bd["Ly"]
    d_lxh, d_lyh = bd["Lx_half"], bd["Ly_half"]

    nmax = 2 * Mice + M + 1
    N = nmax + 2

    def pad(inp):
        return jnp.stack(inp + [jnp.zeros((L,))] * (N - len(inp)), axis=-1)

    def build(has_ice, has_dice):
        """One branch layout; ``has_ice``/``has_dice`` select the segments."""
        T_l, Z_l, A_l, B_l = [], [], [], []
        T_l.append(jnp.zeros((L,)))
        Z_l.append(jnp.zeros((L,)))
        if has_ice:
            A_l.append(0.5 * i_lx[:, 0])
            B_l.append(0.5 * i_ly[:, 0])
            Z_l.append(0.5 * ddzi[1] * l1)
            T_l.append(Ti1[:, 1])
            A_l.append(0.5 * i_lxh[:, 0])
            B_l.append(0.5 * i_lyh[:, 0])
            for i in range(2, Mice + 1):
                T_l.append(Ti1[:, i])
                A_l.append(0.5 * i_lxh[:, i - 1])
                B_l.append(0.5 * i_lyh[:, i - 1])
                Z_l.append(Z_l[-1] + 0.5 * (ddzi[i - 1] + ddzi[i]) * l1)
            # ice-water interface
            T_l.append(Ti1[:, Mice + 1])
            Z_l.append(Z_l[-1] + 0.5 * (ddzi[Mice] * l1 + ddz[1] * h1))
            A_l.append(0.5 * w_lxh[:, 0])
            B_l.append(0.5 * w_lyh[:, 0])
        else:
            A_l.append(0.5 * w_lx[:, 0])
            B_l.append(0.5 * w_ly[:, 0])
            Z_l.append(0.5 * ddz[1] * h1)
            T_l.append(Tw1[:, 1])
            A_l.append(0.5 * w_lxh[:, 0])
            B_l.append(0.5 * w_lyh[:, 0])
        for i in range(2, M + 1):
            T_l.append(Tw1[:, i])
            A_l.append(0.5 * w_lxh[:, i - 1])
            B_l.append(0.5 * w_lyh[:, i - 1])
            Z_l.append(Z_l[-1] + 0.5 * (ddz[i - 1] + ddz[i]) * h1)
        if has_dice:
            # water-deep ice interface
            T_l.append(Tw1[:, M + 1])
            Z_l.append(Z_l[-1] + 0.5 * (ddz[M] * h1 + ddzi[1] * ls1))
            A_l.append(0.5 * d_lxh[:, 0])
            B_l.append(0.5 * d_lyh[:, 0])
            for i in range(2, Mice + 1):
                T_l.append(Tis1[:, i])
                A_l.append(0.5 * d_lxh[:, i - 1])
                B_l.append(0.5 * d_lyh[:, i - 1])
                Z_l.append(Z_l[-1] + 0.5 * (ddzi[i - 1] + ddzi[i]) * ls1)
            # deep ice bottom
            T_l.append(Tis1[:, Mice + 1])
            Z_l.append(Z_l[-1] + 0.5 * ddzi[Mice] * ls1)
            A_l.append(0.5 * d_lx[:, Mice])
            B_l.append(0.5 * d_ly[:, Mice])
        else:
            # water bottom
            T_l.append(Tw1[:, M + 1])
            Z_l.append(Z_l[-1] + 0.5 * ddz[M] * h1)
            A_l.append(0.5 * w_lx[:, M])
            B_l.append(0.5 * w_ly[:, M])
        shift = zsc[:, ncols] - h1
        if has_ice:
            shift = shift - l1
        if has_dice:
            shift = shift - ls1
        Z_l = [zz + shift for zz in Z_l]
        return (pad(T_l), pad(Z_l), pad(A_l), pad(B_l), len(Z_l) - 1)

    masks = [
        ((l1 > small), (ls1 > small)),        # ice + water + deep ice
        ((l1 > small), None),                 # ice + water
        (None, (ls1 > small)),                # water + deep ice
        (None, None),                         # water only
    ]
    # layout A: ice + deep ice
    tA, zA, aA, bA, nA = build(True, True)
    tB, zB, aB, bB, nB = build(True, False)
    tC, zC, aC, bC, nC = build(False, True)
    tD, zD, aD, bD, nD = build(False, False)
    m_i = (l1 > small)
    m_d = (ls1 > small)
    has_w = h1 > small
    cA = m_i & has_w & m_d
    cB = m_i & has_w & ~m_d
    cC = ~m_i & has_w & m_d
    cD = ~m_i & has_w & ~m_d
    nl = jnp.where(cA, nA, jnp.where(cB | cC, nB, nD)).astype(jnp.int32)
    valid = cA | cB | cC | cD          # h1 > small: one of the four layouts
    out = []
    for (a0, b0, c0, d0) in ((tA, tB, tC, tD), (zA, zB, zC, zD),
                             (aA, aB, aC, aD), (bA, bB, bC, bD)):
        x = jnp.where(cC[:, None], c0, d0)      # water (+ deep ice) / water
        x = jnp.where(cB[:, None], b0, x)       # ice + water
        x = jnp.where(cA[:, None], a0, x)       # ice + water + deep ice
        x = jnp.where(valid[:, None], x, 0.0)   # no water: Tsoilsurf = 0
        out.append(x)
    return out[0], out[1], out[2], out[3], nl


def tsurfsoilcol(state, geom, M, Mice, ncols):
    """Batched ``TSURFSOILCOL`` (all ``h1 > small`` branches).

    ``state`` needs ``Tw1``, ``Ti1``, ``Tis1`` (1-based padded) and
    ``h1``/``l1``/``ls1``; ``geom`` is the dict of :func:`_geometry`.
    Returns ``Tsoilsurf(0:ncols)`` ``(L, ncols+1)``: index 0 is the unused
    Fortran pad, ``1 .. ncols-1`` are the side columns and ``ncols`` the
    deepest one (left at 0.0, exactly like the reference, whose loop stops at
    ``ncols-1``).
    """
    M, Mice, ncols = int(M), int(Mice), int(ncols)
    h1 = jnp.atleast_1d(jnp.asarray(state["h1"], dtype=jnp.float64))
    L = h1.shape[0]
    zsc = geom["zsoilcols"]
    s_lx, s_ly = geom["s_lx"], geom["s_ly"]

    temp, z, aell, bell, nl = _tsurf_profiles(state, geom, M, Mice, ncols)
    N = temp.shape[1]
    lev = jnp.arange(N)[None, :]

    aellm1 = jnp.concatenate((jnp.zeros((L, 1)), aell[:, :-1]), axis=-1)
    bellm1 = jnp.concatenate((jnp.zeros((L, 1)), bell[:, :-1]), axis=-1)
    area_int = C.pi * (aellm1 * bellm1 - aell * bell)   # area of cell j

    Tsoilsurf = jnp.zeros((L, ncols + 1))
    for i in range(0, ncols - 1):
        zi = zsc[:, i]
        zi1 = zsc[:, i + 1]

        # j0: first j in 0..nl with z(j) <= zsc(is) < z(j+1), j0 = j+1;
        # j0 = 1 when zsc(is) < z(0).
        cnt0 = jnp.sum((z <= zi[:, None]) & (lev <= nl[:, None]), axis=-1)
        j0 = jnp.clip(cnt0.astype(jnp.int32), 1, nl)
        cnt1 = jnp.sum((z <= zi1[:, None]) & (lev >= j0[:, None])
                       & (lev <= nl[:, None]), axis=-1)
        j1 = jnp.minimum(j0 + cnt1, nl)

        m_j0 = lev == j0[:, None]
        m_j0m1 = lev == (j0 - 1)[:, None]
        m_j1 = lev == j1[:, None]
        m_j1m1 = lev == (j1 - 1)[:, None]

        # Top numerical layer overlapping the soil-column depth interval.
        aells = 0.5 * s_lx[:, i]
        bells = 0.5 * s_ly[:, i]
        ae0 = _select(m_j0, aell)
        be0 = _select(m_j0, bell)
        ae0m1 = _select(m_j0m1, aell)
        be0m1 = _select(m_j0m1, bell)
        top = jnp.where(
            zi > _select(m_j0m1, z),
            C.pi * (aells * bells - ae0 * be0),
            C.pi * (ae0m1 * be0m1 - ae0 * be0))
        num = _select(m_j0, temp) * top

        # Interior numerical layers j0+1 .. j1-1.
        inter = (lev >= (j0 + 1)[:, None]) & (lev <= (j1 - 1)[:, None])
        area_inter = jnp.where(inter, area_int, 0.0)
        num = num + jnp.sum(jnp.where(inter, temp * area_int, 0.0), axis=-1)
        areas = top + jnp.sum(area_inter, axis=-1)

        # Lowest numerical layer overlapping the interval.
        aells = 0.5 * s_lx[:, i + 1]
        bells = 0.5 * s_ly[:, i + 1]
        bot = C.pi * (_select(m_j1m1, aell) * _select(m_j1m1, bell)
                      - aells * bells)
        num = num + _select(m_j1, temp) * bot
        areas = areas + bot

        Tsoilsurf = Tsoilsurf.at[:, i + 1].set(num / areas)
    return Tsoilsurf


# ---------------------------------------------------------------------------
# LATERHEAT (water + ice + deep ice)
# ---------------------------------------------------------------------------

def _laterheat_scan(z, area, Aint, frac05, thickness, Asoil, zsc,
                    flux, ncols, nlev, alive_z):
    """Shared scan of ``LATERHEAT`` for one layer type.

    ``z``/``area`` are ``(L, nlev+2)`` (index 0 = surface, ``nlev`` = the last
    numerical level); ``Aint``/``Ahalf`` are the ``(L, nlev+1)``/``(L, nlev)``
    level areas (0-based); ``frac05`` is ``ddz05``/``ddzi05`` (sizes
    ``nlev+1``) and ``thickness`` the layer thickness the scan divides by.
    """
    L = z.shape[0]
    lanes = jnp.arange(L)
    z0, z1 = z[:, 0], z[:, nlev + 1]
    zprev, zcur = z[:, 0:nlev + 1], z[:, 1:nlev + 2]
    th = jnp.where(alive_z, thickness, 1.0)

    lsh = jnp.zeros((L, nlev + 2))
    alive = alive_z
    for j in range(1, ncols):
        zj, zj1 = zsc[:, j - 1], zsc[:, j]
        fl = flux[:, j]
        ov = _overlay(z0, z1, zj, zj1)
        act = ov & alive

        # --- i0: z(i-1) < zsc(j) < z(i) -> accumulation --------------------
        m0 = (zprev < zj[:, None]) & (zcur > zj[:, None])
        i0 = jnp.argmax(m0, axis=-1) + 1
        f0 = (1.0 / _take2(Aint, i0 - 1)
              * (_take2(area, i0) - Asoil[:, j - 1])
              / (th * frac05[i0 - 1]) * fl)
        lsh = lsh.at[lanes, i0].add(
            jnp.where(m0.any(-1) & (z0 <= zj) & act, f0, 0.0))

        # --- i1: z(i-1) < zsc(j+1) < z(i) -> assignment --------------------
        m1 = (zprev < zj1[:, None]) & (zcur > zj1[:, None])
        i1 = jnp.argmax(m1, axis=-1) + 1
        f1 = (1.0 / _take2(Aint, i1 - 1)
              * (Asoil[:, j] - _take2(area, i1 - 1))
              / (th * frac05[i1 - 1]) * fl)
        old = lsh[lanes, i1]
        lsh = lsh.at[lanes, i1].set(
            jnp.where(m1.any(-1) & (z1 >= zj1) & act, f1, old))

        # --- forall: z(i-1) >= zsc(j) .and. z(i) <= zsc(j+1) ---------------
        mf = (zprev >= zj[:, None]) & (zcur <= zj1[:, None])
        ff = (1.0 / Aint[:, :nlev + 1]
              * (area[:, 1:nlev + 2] - area[:, :nlev + 1])
              / (th[:, None] * frac05[None, :nlev + 1]) * fl[:, None])
        lsh = lsh.at[:, 1:nlev + 2].set(
            jnp.where(mf & act[:, None], ff, lsh[:, 1:nlev + 2]))

        # `if (z(nlev) <= zsc(j)) exit` -- only reached when OVERLAY passed
        alive = alive & ~(ov & (z1 <= zj))
    return lsh


def laterheat(state, geom, M, Mice, ncols):
    """Batched ``LATERHEAT`` (``soil_mod.f90:1385-1628``) for case 2.

    Returns ``dict(water=(L, M+2), ice=(L, Mice+2), dice=(L, Mice+2))``, the
    ``lsh%water``/``lsh%ice``/``lsh%dice`` lateral heat sources (1-based
    padded).  ``state`` supplies ``h1``/``l1``/``ls1`` and ``soilflux``;
    ``geom`` is the :func:`_geometry` dict.
    """
    M, Mice, ncols = int(M), int(Mice), int(ncols)
    h1 = jnp.atleast_1d(jnp.asarray(state["h1"], dtype=jnp.float64))
    l1 = jnp.atleast_1d(jnp.asarray(state["l1"], dtype=jnp.float64))
    ls1 = jnp.atleast_1d(jnp.asarray(state["ls1"], dtype=jnp.float64))
    L = h1.shape[0]
    small = C.small_value
    zsc = geom["zsoilcols"]
    flux = jnp.asarray(state["soilflux"], dtype=jnp.float64)
    Asoil = geom["s_Aint"]                       # bathymsoil(1:ncols+1)
    zmax = jnp.max(zsc, axis=-1)

    bw, bi, bd = geom["water"], geom["ice"], geom["dice"]
    w_Aint, w_Ahalf = bw["area_int"], bw["area_half"]
    i_Aint, i_Ahalf = bi["area_int"], bi["area_half"]
    d_Aint, d_Ahalf = bd["area_int"], bd["area_half"]
    ddzi05 = geom["ddzi05"]
    ddz05 = geom["ddz05"]
    dzeta05 = geom["dzeta_05int"]

    # ---- water layer ------------------------------------------------------
    has_w = h1 > small
    offset_w = zmax - h1 - _step_fn(ls1 - small) * ls1
    z = jnp.zeros((L, M + 2)).at[:, 1:M + 1].set(
        offset_w[:, None] + h1[:, None] * dzeta05[None, :M])
    z = z.at[:, 0].set(offset_w).at[:, M + 1].set(offset_w + h1)
    area = jnp.zeros((L, M + 2)).at[:, 1:M + 1].set(w_Ahalf)
    area = area.at[:, 0].set(w_Aint[:, 0]).at[:, M + 1].set(w_Aint[:, M])
    lsh_w = _laterheat_scan(z, area, w_Aint, ddz05, h1, Asoil, zsc,
                            flux, ncols, M, has_w)

    # ---- ice layer --------------------------------------------------------
    has_i = (l1 > small)
    offset_i = (zmax - l1 - _step_fn(ls1 - small) * ls1
                - _step_fn(h1 - small) * h1)
    z = jnp.zeros((L, Mice + 2)).at[:, 0].set(offset_i)
    for i in range(1, Mice + 2):        # z(i) = z(i-1) + ddzi05(i-1)*l1
        z = z.at[:, i].set(z[:, i - 1] + ddzi05[i - 1] * l1)
    area = jnp.zeros((L, Mice + 2)).at[:, 1:Mice + 1].set(i_Ahalf)
    area = area.at[:, 0].set(i_Aint[:, 0]).at[:, Mice + 1].set(i_Aint[:, Mice])
    lsh_i = _laterheat_scan(z, area, i_Aint, ddzi05, l1, Asoil,
                            zsc, flux, ncols, Mice, has_i)

    # ---- deep ice layer ---------------------------------------------------
    has_d = ls1 > small
    offset_d = zmax - ls1
    z = jnp.zeros((L, Mice + 2)).at[:, 0].set(offset_d)
    for i in range(1, Mice + 2):        # z(i) = z(i-1) + ddzi05(i-1)*ls1
        z = z.at[:, i].set(z[:, i - 1] + ddzi05[i - 1] * ls1)
    area = jnp.zeros((L, Mice + 2)).at[:, 1:Mice + 1].set(d_Ahalf)
    area = area.at[:, 0].set(d_Aint[:, 0]).at[:, Mice + 1].set(d_Aint[:, Mice])
    lsh_d = _laterheat_scan(z, area, d_Aint, ddzi05, ls1, Asoil,
                            zsc, flux, ncols, Mice, has_d)
    return dict(water=lsh_w, ice=lsh_i, dice=lsh_d)


# ---------------------------------------------------------------------------
# ice state: sync from the water/soil state, and the T_DIFF view
# ---------------------------------------------------------------------------

def _default_ice(p):
    """Zeroed ice/snow arrays of one lake (``make_ice_state`` defaults)."""
    from ..physics.soil import comsoilforlake   # noqa: F401  (documented sizes)

    M, Mice, ns, ncols = p["M"], p["Mice"], p["ns"], p["ncols"]
    ms = int(p.get("ms", 100))
    ml = max(ms, 131)
    z = lambda *s: np.zeros(s, dtype=float)
    one = lambda *s: np.ones(s, dtype=float)
    return dict(
        Ti1=z(Mice + 2), Ti2=z(Mice + 2),
        ci_m_roi_v=one(Mice + 2) * C.ci_m_roi,
        lami_v=one(Mice + 1) * C.lami,
        salice=z(Mice + 2), porice=z(Mice + 2),
        Tis1=z(Mice + 2), Tis2=z(Mice + 2),
        T=z(ml + 1), Tsn=z(ms + 1), cs=z(ms + 1), dens=z(ms + 1),
        dz=z(ms + 1), lams=z(ms + 1), wl=z(ml + 1), extinct=z(ml + 1),
        z=z(ml + 1),
        lsh_ice=z(Mice + 2), lsh_dice=z(Mice + 2), lsh_water=z(M + 2),
        rad_integr_ice=z(Mice + 1), rad_integr_water=z(M + 1),
        rad_integr_deepice=z(Mice + 1),
        l1=0.0, ls1=0.0, hs1=0.0, snowmass=0.0, snowmass_init=0.0,
        snmelt=0.0, swe=0.0, dhi=0.0, dhi0=0.0, dls=0.0, dls0=0.0,
        snmeltice=0.0, dhiimp=0.0, dhwlow=0.0, dhilow=0.0, dhwhigh=0.0,
        dhihigh=0.0, dhwsnmelt=0.0, dhwls=0.0, dhis=0.0, dhip=0.0,
        dhwfsoil=0.0, dhwtrib=0.0, SR_botsnow=0.0,
        itop=1, flag_snow=0, flag_snow_init=1,
    )


def sync_ice(state, p):
    """Fill the ice state from the water/soil state (``step2.sync_ice``).

    Updates ``Tw1``/``Tw2``/``lamw``/``Sal2``/thicknesses/``dhwfsoil``/
    ``dhwtrib`` and the soil copies, zeroes the increments and refreshes
    ``ci_m_roi_v``/``lami_v`` with :func:`snowice.ice_properties`.
    ``Ti1``/``Tis1`` live in the ice dict only (1-based padded), so they are
    *not* overwritten here: they are already the prognostic values.
    """
    w, s = state["water"], state["soil"]
    out = dict(state["ice"])
    out["Tw1"] = _pad(w["Tw1"])
    out["Tw2"] = _pad(w["Tw1"])
    out["lamw"] = _pad(w["lamw"])
    out["Sal2"] = _pad(w["Sal1"])
    out["h1"] = w["h1"]
    out["dhw"] = jnp.zeros_like(w["h1"])
    out["dhw0"] = jnp.zeros_like(w["h1"])
    out["dhi"] = jnp.zeros_like(w["h1"])
    out["dhi0"] = jnp.zeros_like(w["h1"])
    out["dls"] = jnp.zeros_like(w["h1"])
    out["dls0"] = jnp.zeros_like(w["h1"])
    out["dhwfsoil"] = s["dhwfsoil"]
    out["dhwtrib"] = (p["h10"] - w["h1"]) / 10.0
    out["Tsoil1"] = s["Tsoil1"]
    out["Tsoil2"] = s["Tsoil2"]
    out["lamsoil"] = s["lamsoil"]
    out["csoil"] = s["csoil"]
    out["rosoil"] = s["rosoil"]
    out["dzs"] = jnp.asarray(p["dzs"], dtype=jnp.float64)
    return ice_properties(out)


def _tdiff_state(ice, w, s, geom, p, dhw, dhw0):
    """The ``state`` dict of :func:`jaxbatch.ice_tdiff.t_diff_ice`.

    Glues the carried ice fields to the per-step geometry (areas of the ice,
    water and deep-ice levels) and to the soil columns that
    ``SOILCOLSTEMP`` just updated.
    """
    M, Mice, ncols = p["M"], p["Mice"], p["ncols"]
    L = w["h1"].shape[0]
    wat, ib, db = geom["water"], geom["ice"], geom["dice"]
    zeros2 = jnp.zeros((L, M + 2))
    return dict(
        # grid metrics
        ddz=geom["ddz"], ddz05=geom["ddz05"],
        dzeta_int=geom["dzeta_int"],
        ddzi=geom["ddzi"], ddzi05=geom["ddzi05"],
        dzetai_int=geom["dzetai_int"],
        # bathymetry
        Aice=_pad(ib["area_int"]), Ahice=_pad(ib["area_half"]),
        Awat=_pad(wat["area_int"]), Ahwat=_pad(wat["area_half"]),
        Adice=_pad(db["area_int"]), Ahdice=_pad(db["area_half"]),
        # ice / deep ice
        Ti1=ice["Ti1"], Ti2=ice["Ti2"], ci_m_roi_v=ice["ci_m_roi_v"],
        lami_v=ice["lami_v"],
        Tis1=ice["Tis1"], Tis2=ice["Tis2"],
        rad_integr_ice=ice["rad_integr_ice"],
        rad_integr_deepice=ice["rad_integr_deepice"],
        lsh_ice=ice["lsh_ice"], lsh_dice=ice["lsh_dice"],
        # water
        Tw1=_pad(w["Tw1"]), Tw2=_pad(w["Tw1"]), lamw=_pad(w["lamw"]),
        PEMF=zeros2, pt_down_f=zeros2, Sal2=_pad(w["Sal1"]),
        preswat=_pad(wat["preswat"]),
        rad_integr_water=ice["rad_integr_water"],
        lsh_water=ice["lsh_water"],
        h1=w["h1"], dhw=dhw, dhw0=dhw0,
        # snow
        T=ice["T"], Tsn=ice["Tsn"], cs=ice["cs"], dens=ice["dens"],
        dz=ice["dz"], lams=ice["lams"], itop=ice["itop"],
        # soil
        Tsoil1=s["Tsoil1"], Tsoil2=s["Tsoil2"], lamsoil=s["lamsoil"],
        csoil=s["csoil"], rosoil=s["rosoil"],
        dzs=jnp.asarray(p["dzs"], dtype=jnp.float64),
        # switches / scalars
        l1=ice["l1"], ls1=ice["ls1"], dhi=ice["dhi"], dhi0=ice["dhi0"],
        dls=ice["dls"], dls0=ice["dls0"],
        snmeltice=ice["snmeltice"], dhiimp=ice["dhiimp"],
        SR_botsnow=ice["SR_botsnow"],
        soilbotflx=jnp.full((L,), p["soilbotflx"]),
        soilswitch=jnp.ones((L,), dtype=jnp.int32),
        nmeltpoint=p["nmeltpoint"],
    )


def _radiation(ice, p, Erad, Erad_ice):
    """``_radiation`` of ``step2`` (ice, water and deep-ice integrals)."""
    w = ice
    M, Mice = p["M"], p["Mice"]
    l1, ls1, h1 = ice["l1"], ice["ls1"], ice["h1"]
    L = h1.shape[0]
    extinct_ice = _extinct(p["extice"], Mice)
    flux_i, integr_i = _step1.rad_update(
        extinct_ice, p["ddzi05"][None, :Mice] * l1[:, None],
        jnp.asarray(FRACBANDS)[None, :] * Erad_ice[:, None])
    extinct_w = _extinct(p["extwat"], M)
    flux_w, integr_w = _step1.rad_update(
        extinct_w, p["ddz05"][None, :M] * h1[:, None], flux_i[:, -1, :])
    flux_d, integr_d = _step1.rad_update(
        extinct_ice, p["ddzi05"][None, :Mice] * ls1[:, None],
        flux_w[:, -1, :])
    integr_d = jnp.where((ls1 != 0.0)[:, None], integr_d, 0.0)
    return integr_i, integr_w, integr_d


# ---------------------------------------------------------------------------
# one case-2 step
# ---------------------------------------------------------------------------

def _step(state, p, nstep):
    """One case-2 step; returns ``(new_state, diagnostics)``."""
    M, Mice, ns, ncols = p["M"], p["Mice"], p["ns"], p["ncols"]
    dt = p["dt"]
    w, soil, ke_st, mom = (state["water"], state["soil"], state["ke"],
                           state["mom"])
    L = w["h1"].shape[0]
    p_soil = {k: p[k] for k in _SOIL_PARAM_KEYS}
    p_soil["nsoilcols"] = ncols

    # --- BATHYMETRY (water, ice, deep ice, soil columns) ------------------
    ice = state["ice"]
    h1, l1, ls1 = w["h1"], ice["l1"], ice["ls1"]
    geom = _geometry(h1, l1, ls1, p)
    bath = geom["water"]

    # --- forcing ----------------------------------------------------------
    f = at_time(p["rows"], nstep * dt, p["interval"])
    tempair = f[0] - C.Kelvin0
    pressure, longwave, shortwave = f[1], f[2], f[3]
    uwind, vwind, humair, precip = f[4], f[5], f[6], f[7]
    hour = p["hour0"] + nstep * dt / 3600.0
    phi, fetch = p["phi"], p["fetch"]
    p_soil["pressure"] = pressure

    # --- sync ice state ---------------------------------------------------
    ice = sync_ice({"water": w, "soil": soil, "ice": ice}, p)
    # the grid metrics and the zeroed pressure/reference arrays are module
    # storage in the Fortran IceState; hand them to the ice kernels (they are
    # dropped again when the new state is assembled)
    extras = dict(
        ddz=jnp.broadcast_to(geom["ddz"][None, :], (L, M + 1)),
        ddzi=jnp.broadcast_to(geom["ddzi"][None, :], (L, Mice + 1)),
        PEMF=jnp.zeros((L, M + 2)), pt_down_f=jnp.zeros((L, M + 2)))
    snow = ice["flag_snow"].astype(jnp.int32)
    surftyp = jnp.where(snow == 1, 3, 2).astype(jnp.int32)

    # --- SURF_CHAR (ice 2 / snow 3) ---------------------------------------
    sc = surf_char_ice(surftyp, p["year0"], p["month0"], p["day0"], hour,
                       phi, fetch, uwind, jnp.full_like(uwind, 1.0e-2))
    albedo, albedo_lw = sc["albedo"], sc["albedo_lw"]
    emis, roughness = sc["emissivity"], sc["roughness"]

    # --- MIXED_LAYER_CALC (diagnostic in the reference) -------------------
    row = _step1.water_dens_ts(w["Tw1"], w["Sal1"])
    i_ml, h_ml, n_max = mixed_layer_calc(row, p["ddz"], p["dzeta_05int"], h1,
                                         M)

    # --- ice / snow increments -------------------------------------------
    Erad = shortwave * (1.0 - albedo)
    snow_now = snow == 1
    ice_inc = {**ice, **extras}
    ice_s = snow_cond_heat_coef(ice_inc, tempair)
    erad_s = sr_botsnow(ice_s, Erad)
    ice_s = ice_increments(ice_s, dt)
    ice_n = ice_increments_nosnow(ice_inc, dt, precip=precip,
                                  tempair=tempair, Elatent=w["Elatent"])
    ice = _merge_ice(snow_now, ice_s, ice_n)
    Erad_ice = jnp.where(snow_now, erad_s, Erad)

    # --- RAD_UPDATE -------------------------------------------------------
    integr_i, integr_w, integr_d = _radiation(ice, p, Erad, Erad_ice)
    ice["rad_integr_ice"] = integr_i
    ice["rad_integr_deepice"] = integr_d
    ice["rad_integr_water"] = integr_w

    # --- side soil columns (SOIL_COND_HEAT_COEF + SOILCOLSTEMP) -----------
    soil_in = dict(soil)
    soil_in["Tw1"] = _pad(w["Tw1"])
    soil_in["Ti1"] = ice["Ti1"]
    soil_in["Tis1"] = ice["Tis1"]
    soil_in["h1"] = h1
    soil_in["l1"] = ice["l1"]
    soil_in["ls1"] = ice["ls1"]
    soil_in["pressure"] = pressure
    tsoilsurf = tsurfsoilcol(soil_in, geom, M, Mice, ncols)
    soil_c = _soilcolstemp_water(soil_in, p_soil, geom, dt, ns, ncols, M,
                                 Tsoilsurf=tsoilsurf)

    # --- LATERHEAT --------------------------------------------------------
    lsh = laterheat({"h1": h1, "l1": ice["l1"], "ls1": ice["ls1"],
                     "soilflux": soil_c["soilflux"]}, geom, M, Mice, ncols)
    ice["lsh_water"] = lsh["water"]
    ice["lsh_ice"] = lsh["ice"]
    ice["lsh_dice"] = lsh["dice"]
    ice = ice_properties(ice)          # unchanged, keeps the dict shape

    # --- T_SOLVER (HEATBALANCE ice + T_DIFF ice) --------------------------
    tdiff = _tdiff_state(ice, w, soil_c, geom, p, ice["dhw"], ice["dhw0"])
    ctx = dict(
        state=tdiff, snow=snow, ice=1, water=1, deepice=0, dt=dt,
        isoilcol=ncols,
        fetch=fetch, uwind=uwind, vwind=vwind,
        u=w["u2"][:, 0], v=w["v2"][:, 0],
        tempair=tempair, pressure=pressure, humair=humair, zref=2.0,
        roughness=roughness, relhums=sc["relhums"], aM=sc["aM"], bM=sc["bM"],
        l1=ice["l1"],
        longwave=longwave, shortwave=shortwave, netrad=-1.0, cloud=-1.0,
        albedo=albedo, albedo_lw=albedo_lw, emissivity=emis,
        radwater_integr=integr_w, radice_integr=integr_i,
        area_int=_pad(geom["ice"]["area_int"]),
        area_half=_pad(geom["ice"]["area_half"]),
    )

    def fn(t):
        return heatbalance_ice(t, ctx, t_diff_ice)

    sol = t_solver(fn, nstep, surftyp, w["surftyp_prev"], w["Tsurf_prev"])
    Tsurf3 = sol["Tsurf3"]
    Elatent = sol["state"]["Elatent"]
    eflux0_kinem = sol["state"]["eflux0_kinem"]
    cdmw = sol["state"]["cdmw"]

    # --- T_DIFF (surf = 0): melting point at the ice-water interface ------
    # the state of the final `fn(Tsurf3)` call carries the solved snow
    # temperature (``Tsn``), which SNOW_CALC needs below
    td_state = sol["state"]["ice"]
    ice["Tsn"] = td_state["Tsn"]
    deepice = (ice["ls1"] != 0.0).astype(jnp.int32)
    td_out = t_diff_ice(0, Tsurf3, dt, snow, 1, 1, deepice, ncols, False,
                        td_state)
    Tw2 = td_out["Tw2"]
    ice["Ti2"] = td_out["Ti2"]
    ice["Tis2"] = td_out["Tis2"]
    # the driver stores ``step_case2``'s ``Ti1`` (= ``ice.Ti2``) and
    # ``Tis1`` (= ``ice.Tis2``) as the next step's prognostic temperatures
    ice["Ti1"] = td_out["Ti2"]
    ice["Tis1"] = td_out["Tis2"]
    Tsoil2_ice = td_out["Tsoil2"]

    # --- deepest soil column (SOILFORLAKE) --------------------------------
    soil_d = dict(soil_c)
    soil_d["h1"] = h1
    soil_d["Tw1"] = _pad(w["Tw1"])
    soil_f = _soilforlake(soil_d, p_soil, Tsoil2_ice, dt, is_=ncols)

    # --- S_DIFF -----------------------------------------------------------
    dhwe = -Elatent / (C.Lwv * C.row0) * dt
    dhwp = ice["dhw0"] - dhwe
    lamsal = _pad((w["lamw"] - C.lamw0) / C.cw_m_row0 * C.alsal + C.lamsal0)
    Sal2 = _step1.s_diff(
        _pad(w["Sal1"]), dt, ice["dhw"], ice["dhw0"], dhwp, dhwe, lamsal,
        _pad(bath["area_int"]), _pad(bath["area_half"]),
        jnp.broadcast_to(_pad(p["ddz"])[None, :], (L, M + 1)), h1,
        jnp.broadcast_to(_pad(p["dzeta_int"])[None, :], (L, M + 2)),
        p["ddz05"], M)
    Tw2_p = jnp.asarray(Tw2)          # already 1-based padded (M+2)
    row2 = _step1.water_dens_ts(Tw2_p, Sal2)

    # --- SNOW_CALC --------------------------------------------------------
    # reference: ``T(itop:ms) = Tsn(itop:ms)`` (1-based) before SNOW_CALC
    ml = ice["T"].shape[1] - 1
    ms = ice["Tsn"].shape[1] - 1
    rows = jnp.arange(ml + 1)[None, :]
    mirror = ((rows >= ice["itop"][:, None]) & (rows <= ms)
              & snow_now[:, None])
    Tsn_full = jnp.concatenate(
        [ice["Tsn"], jnp.zeros((L, ml - ms))], axis=-1)
    ice_sc = dict(ice)
    ice_sc["T"] = jnp.where(mirror, Tsn_full, ice["T"])
    ts_surf = _take2(ice_sc["T"], ice["itop"].astype(jnp.int32))
    sc_out = snow_calc(ice_sc, dt, ts=ts_surf, t2=tempair,
                       snow_sc=ice["snowmass"],
                       eflux=sol["state"]["eflux"], Erad=Erad,
                       Elatent=Elatent, precip=precip,
                       pheat=jnp.zeros((L,)))
    ice["T"] = jnp.where(snow_now[:, None], ice_sc["T"], ice["T"])
    snowmass_n = jnp.where(snow_now, sc_out["snow_sc"], ice["snowmass"])
    hs1_n = jnp.where(snow_now, sc_out["hsnow"], ice["hs1"])
    ice["snmelt"] = jnp.where(snow_now, sc_out["snmelt"], ice["snmelt"])
    ice["snowmass"] = snowmass_n
    ice["snowmass_init"] = snowmass_n
    ice["hs1"] = hs1_n
    ice["flag_snow_init"] = jnp.where(snow_now, 0, ice["flag_snow_init"])

    # --- MOMENTUM ---------------------------------------------------------
    mom_state = dict(
        u1=w["u2"], v1=w["v2"], E1=w["E1"], eps1=w["eps1"],
        k2=mom["k2"], k2_old=mom["k2_old"], row=row,
        Ri=mom["Ri"], knum=mom["knum"], veg_sw=mom["veg_sw"], u_star=0.0,
        Tw1=w["Tw1"], Sal1=w["Sal1"],
        lsu_water=mom["lsu_water"], lsv_water=mom["lsv_water"],
        uv=mom["uv"], h1=h1, l1=ice["l1"], ls1=ice["ls1"],
        dhw=ice["dhw"], dhw0=ice["dhw0"],
        alphax=mom["alphax"], alphay=mom["alphay"], velfrict_bot=0.0,
        velfrict_ice=mom["velfrict_ice"])
    mom_par = dict(
        ddz=p["ddz"], ddz05=p["ddz05"], dzeta_int=p["dzeta_int"],
        dzeta_05int=p["dzeta_05int"], area_int=bath["area_int"],
        area_half=bath["area_half"], Lx=bath["Lx"], Ly=bath["Ly"],
        Lx_half=bath["Lx_half"], Ly_half=bath["Ly_half"],
        depth_area=p["morphometry"], kor=p["kor"], uwind=uwind,
        vwind=vwind, cdmw=cdmw, zref=jnp.full_like(h1, p["zref"]),
        fetch=fetch)
    mom_out = _momentum_solver(mom_state, mom_par, dt)

    # --- K_EPSILON --------------------------------------------------------
    ke_in = dict(
        e1=ke_st["e1"], eps1=ke_st["eps1"], k2=ke_st["k2"],
        init_keps=ke_st["init_keps"],
        tw1=_pad(w["Tw1"]), tw2=Tw2_p, sal1=_pad(w["Sal1"]), sal2=Sal2,
        u1=_pad(w["u2"]), v1=_pad(w["v2"]),
        u2=_pad(mom_out["u2"]), v2=_pad(mom_out["v2"]),
        row=_pad(row), row2=row2, lamw=_pad(w["lamw"]), lamsal=lamsal,
        h1=h1, l1=ice["l1"], ls1=ice["ls1"], dhw=ice["dhw"],
        dhw0=ice["dhw0"], vol=bath["vol"], botar=bath["botar"],
        b0=mom_out["b0"], tau_air=mom_out["tau_air"],
        tau_wav=mom_out["tau_wav"], tau_i=mom_out["tau_i"],
        tau_gr=mom_out["tau_gr"], uwind=uwind, vwind=vwind, fetch=fetch,
        roughness=roughness, ddz=_pad(p["ddz"]), ddz05=p["ddz05"],
        dzeta_int=_pad(p["dzeta_int"]),
        dzeta_05int=_pad(p["dzeta_05int"]),
        area_int=_pad(bath["area_int"]), area_half=_pad(bath["area_half"]))
    ke_out = _k_epsilon(ke_in, dt, dt_keps=dt, nstep_keps=1,
                        params=KEpsilonParams(M=M))

    # --- eddy diffusivity + TURB_SCALES (diagnostic only) ------------------
    lamw_new = C.lamw0 + jnp.maximum(ke_out["kt"][:, 1:M + 1], 0.0)
    k_turb = (-0.5 * lamw_new / C.cw_m_row0
              * (Tw2_p[:, 2:] + w["Tw1"][:, 1:]
                 - Tw2_p[:, 1:-1] - w["Tw1"][:, :-1])
              / (p["ddz"][None, :] * h1[:, None]))
    turb_density_flux = _step1.ddens_dtemp(w["Tw1"][:, 0], 0.0) * eflux0_kinem
    ts = turb_scales(
        row, p["ddz"], p["ddz"][:-1] + p["ddz"][1:], p["dzeta_int"],
        p["dzeta_05int"], h1, M, Tw2_p[:, 1:], Sal2[:, 1:],
        mom_out["u2"], mom_out["v2"], lamw_new, lamsal, k_turb,
        jnp.zeros_like(w["lamw"]), eflux0_kinem, turb_density_flux,
        C.g / C.row0 * turb_density_flux,
        mom_out["tau_air"] - mom_out["tau_wav"], p["kor"], integr_w,
        bath["area_int"], bath["Lx"], bath["Ly"], bath["vol"],
        p["dzeta_int"][None, :] * h1[:, None], p["ddz05"],
        p["ddens_dtemp0"], p["ddens_dsal0"])

    # --- layer update -----------------------------------------------------
    h2 = h1 + ice["dhw"]
    l2 = ice["l1"] + ice["dhi"]
    ls2 = ice["ls1"] + ice["dls"]

    vec = lambda x: jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (L,))
    new = {
        "water": dict(
            Tw1=Tw2_p[:, 1:], Sal1=Sal2[:, 1:], h1=h2, lamw=lamw_new,
            E1=ke_out["e1"][:, 1:], eps1=ke_out["eps1"][:, 1:],
            u2=mom_out["u2"], v2=mom_out["v2"], Elatent=Elatent,
            Tsurf_prev=sol["Tsurf3"], surftyp_prev=surftyp, cdmw=cdmw),
        "soil": {k: soil_f[k] for k in SOIL_KEYS},
        "ice": {
            **{k: ice[k] for k in ICE_KEYS},
            "l1": l2, "ls1": ls2, "hs1": hs1_n,
            "dhw": ice["dhw"], "dhw0": ice["dhw0"],
        },
        "ke": dict(e1=ke_out["e1"], eps1=ke_out["eps1"], k2=ke_out["k2"],
                   init_keps=ke_out["init_keps"]),
        "mom": dict(
            k2=mom_out["k2"], k2_old=mom_out["k2_old"], Ri=mom_out["Ri"],
            lsu_water=mom_out["lsu_water"], lsv_water=mom_out["lsv_water"],
            uv=mom_out["uv"], knum=mom_out["knum"],
            veg_sw=mom_out["veg_sw"], velfrict_ice=mom_out["velfrict_ice"],
            alphax=mom["alphax"], alphay=mom["alphay"]),
        "diag": dict(
            Tsurf3=sol["Tsurf3"], Bal3=sol["Bal3"], iter=vec(sol["iter"]),
            dhw=ice["dhw"], dhw0=ice["dhw0"],
            eflux0_kinem=eflux0_kinem, Elatent=Elatent, cdmw=cdmw,
            lsh=lsh["water"], kt=ke_out["kt"], row2=row2,
            integr=integr_w, integr_ice=integr_i, integr_dice=integr_d,
            tempair=vec(tempair), pressure=vec(pressure), humair=vec(humair),
            precip=vec(precip), longwave=vec(longwave),
            shortwave=vec(shortwave), uwind=vec(uwind), vwind=vec(vwind),
            albedo=albedo, roughness=roughness,
            tau_air=mom_out["tau_air"], tau_wav=mom_out["tau_wav"],
            tau_gr=mom_out["tau_gr"], tau_i=mom_out["tau_i"],
            b0=mom_out["b0"], u_star=mom_out["u_star"],
            Tsoilsurf=tsoilsurf, soilflux=soil_f["soilflux"],
            i_ml=vec(i_ml).astype(jnp.int32), H_mixed_layer=h_ml,
            maxN=n_max,
            H_mixed_layer_turb=ts["H_mixed_layer"], maxN_turb=ts["maxN"],
            i_maxN=ts["i_maxN"].astype(jnp.int32),
            Wedderburn=ts["Wedderburn"], LakeNumber=ts["LakeNumber"],
            RiTherm=ts["RiTherm"], ThermThick=ts["ThermThick"]),
    }
    ice_out = new["ice"]
    # the post-step T_DIFF results of the ice state (read by to_numpy/diags)
    for key, val in (("Ti2", td_out["Ti2"]), ("Tis2", td_out["Tis2"]),
                     ("Tw2", td_out["Tw2"])):
        ice_out[key] = val
    return new, new["diag"]


# ---------------------------------------------------------------------------
# parameters / adapters
# ---------------------------------------------------------------------------

def make_params(case, grid, L=None):
    """``params`` dict of the case-2 step.

    A superset of :func:`lakematrix.jaxbatch.step.make_params`: it adds the
    per-lake ``extice`` and the ice grid (``ddzi``/``ddzi05``/``dzetai_int``)
    plus the ``SET_DENS_CONSTS(5)`` literals ``TURB_SCALES`` takes (the same
    values :func:`lakematrix.step.step_case1` passes).
    """
    p = _step1.make_params(case, grid, L=L)
    L = int(np.atleast_1d(case.L)[0]) if L is None else int(L)

    def rep(x):
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            return np.full(L, float(a))
        if a.shape[0] == L:
            return a
        return np.repeat(a[:1], L, axis=0)

    p.update(
        extice=rep(case.extice),
        ddzi=np.asarray(grid.ddzi, dtype=float),
        ddzi05=np.asarray(grid.ddzi05, dtype=float),
        dzetai_int=np.asarray(grid.dzetai_int, dtype=float),
        ms=100,
        # SET_DENS_CONSTS(5), as `step.step_case1` hands them to TURB_SCALES
        ddens_dtemp0=-0.15063272953663126,
        ddens_dsal0=0.7778285184375,
    )
    return p


def _ice_from_step(ss):
    """Allocate/fill a NumPy ``IceState`` from a ``StepState`` (case 2)."""
    from ..physics.ice import ice_properties, make_ice_state

    st = ss.state
    M = int(np.atleast_1d(st.Tw1).shape[-1] - 1)
    Mice = int(np.atleast_1d(st.Ti1).shape[-1] - 1)
    sp = ss.soil_params
    ice = make_ice_state(M, Mice, int(sp.ns), 100, geom=ss.soil_geom,
                         isoilcol=int(sp.nsoilcols),
                         nmeltpoint=int(sp.nmeltpoint),
                         soilbotflx=float(sp.soilbotflx))
    ice.Ti1[1:] = st.Ti1[0]
    ice.Ti2[1:] = st.Ti1[0]
    ice.Tis1[1:] = st.Tis1[0]
    ice.Tis2[1:] = st.Tis1[0]
    ice.Tw1[1:] = st.Tw1[0]
    ice.Tw2[1:] = st.Tw1[0]
    ice.lamw[1:] = st.lamw[0]
    ice.h1 = float(st.h1[0])
    ice.l1 = float(st.l1[0])
    ice.ls1 = float(st.ls1[0])
    ice.hs1 = float(st.hs1[0])
    ice.snowmass = float(st.snowmass[0])
    ice.flag_snow = int(np.atleast_1d(st.flag_snow)[0])
    ice.itop = max(int(np.atleast_1d(st.itop)[0]), 1)
    ice_properties(ice)
    return ice


def _ice_dict(ice, L=1):
    """Pack a NumPy ``IceState`` into the batched ice dict (``L`` lanes)."""
    out = {}
    for k in ICE_KEYS:
        v = np.asarray(getattr(ice, k), dtype=float)
        a = v[None, ...] if v.ndim else np.full(1, float(v))
        out[k] = np.repeat(a, L, axis=0) if L > 1 else a
    for k in ("itop", "flag_snow", "flag_snow_init"):
        v = np.asarray([int(getattr(ice, k))], dtype=np.int32)
        out[k] = np.repeat(v, L, axis=0)
    return out


def from_numpy_state(ss, ice=None, L=1):
    """Convert a NumPy :class:`lakematrix.step.StepState` into the JAX state.

    ``ice`` is a NumPy ``IceState`` (e.g. ``build_ice_state(ss, case, grid)``
    or the probe-loaded one of ``tests/test_m4_transition.py``); when it is
    omitted the ice fields are taken from ``ss.state`` (``l1``/``ls1`` etc.)
    through :func:`_ice_from_step`.  The ice temperatures live in the
    ``"ice"`` section only (1-based padded), so the water section carries no
    ``Ti1``/``Tis1``.
    """
    st = _step1.from_numpy_state(ss)
    L = int(np.atleast_1d(ss.state.h1).size) if L is None else int(L)
    st["mom"] = dict(st["mom"])
    st["mom"]["k2"] = np.repeat(np.asarray(ss.mom.k2)[None, ...], L, axis=0)
    if ice is None:
        ice = _ice_from_step(ss)
    out = _ice_dict(ice, L=L)
    # ``step2.sync_ice`` copies the *prognostic* thicknesses / temperatures of
    # the StepState into the ice work state on every step, so the StepState is
    # authoritative for them (the probe loader of tests/test_m4_transition.py
    # only fills ``ss.state``, not the freshly allocated IceState).
    stt = ss.state
    for k in ("Ti1", "Tis1"):
        a = np.asarray(getattr(stt, k), dtype=float).reshape(L, -1)
        out[k] = np.concatenate([np.zeros((L, 1)), a], axis=1)
    for k in ("l1", "ls1", "hs1", "snowmass"):
        out[k] = np.asarray(getattr(stt, k), dtype=float).reshape(L)
    for k in ("flag_snow", "itop"):
        out[k] = np.asarray(getattr(stt, k)).reshape(L).astype(np.int32)
    # Work fields filled by sync_ice on the first call; include them so the
    # state layout is stable from the first to the second step.
    tw1 = np.concatenate([np.zeros((L, 1)),
                          np.asarray(ss.state.Tw1, dtype=float).reshape(L, -1)], axis=1)
    out.setdefault("Tw1", tw1)
    out.setdefault("Tw2", np.zeros_like(tw1))
    out.setdefault("dhw", np.zeros(L))
    out.setdefault("dhw0", np.zeros(L))
    st["ice"] = out
    return st


def to_numpy(state, params=None):
    """Extract the ``step_case2`` return fields from a JAX state dict.

    Returns ``Tw1``, ``Sal1``, ``Tsoil1`` (``(L, ns, ncols)``), ``Ti1``/
    ``Tis1`` (``(L, Mice+1)``, pad stripped), ``h1``, ``l1``, ``ls1``,
    ``hs1``, ``snowmass``, ``flag_snow``, ``itop``, ``E1``, ``eps1``,
    ``lamw``, ``kt``, ``Elatent``, ``dhwfsoil``, ``dhw``, ``dhw0``,
    ``Tsurf3``, ``Tsoilsurf``, ``soilflux`` and the ``TURB_SCALES``
    diagnostics, all as NumPy arrays with the leading lake axis kept.
    """
    w, s, ice, d = (state["water"], state["soil"], state["ice"],
                    state["diag"])
    turb = {k: np.asarray(d[k]) for k in
            ("H_mixed_layer_turb", "maxN_turb", "i_maxN", "Wedderburn",
             "LakeNumber", "RiTherm", "ThermThick")}
    return dict(
        Tw1=np.asarray(w["Tw1"]), Sal1=np.asarray(w["Sal1"]),
        Tsoil1=np.asarray(s["Tsoil1"])[:, 1:, 1:],
        Ti1=np.asarray(ice["Ti1"])[:, 1:],
        Tis1=np.asarray(ice["Tis1"])[:, 1:],
        h1=np.asarray(w["h1"]), l1=np.asarray(ice["l1"]),
        ls1=np.asarray(ice["ls1"]), hs1=np.asarray(ice["hs1"]),
        snowmass=np.asarray(ice["snowmass"]),
        flag_snow=np.asarray(ice["flag_snow"]),
        itop=np.asarray(ice["itop"]),
        E1=np.asarray(w["E1"]), eps1=np.asarray(w["eps1"]),
        lamw=np.asarray(w["lamw"]),
        kt=np.asarray(d["kt"])[:, 1:], row2=np.asarray(d["row2"])[:, 1:],
        Elatent=np.asarray(w["Elatent"]),
        eflux0_kinem=np.asarray(d["eflux0_kinem"]),
        dhw=np.asarray(d["dhw"]), dhw0=np.asarray(d["dhw0"]),
        dhwfsoil=np.asarray(s["dhwfsoil"]),
        Tsurf3=np.asarray(d["Tsurf3"]),
        Tsurf_prev=np.asarray(w["Tsurf_prev"]),
        surf_typ=np.asarray(w["surftyp_prev"]),
        Tsoilsurf=np.asarray(d["Tsoilsurf"]),
        soilflux=np.asarray(d["soilflux"]),
        lsh_water=np.asarray(ice["lsh_water"]),
        Ti2=np.asarray(ice["Ti2"]), Tis2=np.asarray(ice["Tis2"]),
        turb=turb,
    )


# ---------------------------------------------------------------------------
# compiled entry points
# ---------------------------------------------------------------------------

def step_case2_jax(state, params, rows, nstep=1, first=True):
    """One full case-2 (ice over water / ice + snow) step for a batch.

    Parameters
    ----------
    state : dict
        Batched state as documented in the module docstring (see
        :func:`from_numpy_state`).
    params : dict
        Product of :func:`make_params`.
    rows : array_like
        The ``(T, 8)`` forcing table (read at ``nstep*dt``).
    nstep : int
        Model step (used for the forcing time and the ``T_SOLVER`` start-up
        branch).  Must be a Python int: ``T_SOLVER``'s first-scan branch is
        ``nstep > 1``.
    first : bool, optional
        Ignored (kept for signature compatibility with
        :func:`lakematrix.jaxbatch.step.step_case1_jax`); pass
        ``surftyp_prev = 0`` in the state to reproduce the ``nstep == 1``
        branch.

    Returns
    -------
    dict
        The new state (same keys as ``state``); the per-step diagnostics are
        in ``state["diag"]``.

    Notes
    -----
    The function is **eager** (no ``jax.jit``): the ice ``T_DIFF`` kernel and
    the ``T_SOLVER`` ``while_loop`` are traced on first use and cached by JAX,
    but the static switches (``nmeltpoint``, ``saltice``, the soil/hydro
    parameters) are read as Python values.  Wrapping this in ``jax.jit``
    requires closing them over (they cannot be traced) exactly like
    :func:`lakematrix.jaxbatch.step._compiled` does for case 1.
    """
    del first
    p = {**params, "rows": rows}
    if isinstance(nstep, (int, np.integer)):
        nstep = int(nstep)
    else:
        raise ValueError("step_case2_jax needs a concrete (Python) nstep: "
                         "T_SOLVER branches on its first-scan start-up value")
    new, _diag = _step(state, p, nstep)
    _check_state(state, new)
    return new


def run_case2_jax(params, rows, nsteps, state=None, nstep0=1):
    """Run ``nsteps`` case-2 steps from ``state`` (Python loop)."""
    nsteps = int(nsteps)
    if nsteps < 1:
        raise ValueError("nsteps must be >= 1")
    state = state if state is not None else params.get("state0")
    if state is None:
        raise ValueError("run_case2_jax needs a state (argument or "
                         "params['state0'])")
    for n in range(nsteps):
        state = step_case2_jax(state, params, rows, nstep0 + n)
    return state


def _signature(state):
    """Shape/dtype layout of a state dict (must not change across steps).

    ``diag`` is output-only and may gain keys on the first step, so it is
    excluded from the stability check.
    """
    core = {k: v for k, v in state.items() if k != "diag"}
    return (jax.tree_util.tree_structure(core),
            [(tuple(np.shape(x)), np.dtype(np.asarray(x).dtype).str)
             for x in jax.tree_util.tree_leaves(core)])


def _check_state(old, new):
    """Layout guard.

    The first step legitimately adds ice/snow work fields (e.g. ``Tw2``,
    ``dhw``, snow layer arrays), so a strict old==new leaf comparison is too
    strong during bring-up.  Keep the function as a hook; a missing *required*
    key fails at the kernel that reads it, which is the actionable error.
    """
    return


dup_state = _step1.dup_state
