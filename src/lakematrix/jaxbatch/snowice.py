"""JAX batched (leading lake axis) ice / snow thermodynamics.

Port of the case-2/3 helper routines of :mod:`lakematrix.physics.ice`:

===========================  ==============================================
NumPy reference              this module
===========================  ==============================================
``ice_properties``           :func:`ice_properties`
``snow_cond_heat_coef``      :func:`snow_cond_heat_coef`
``sr_botsnow``               :func:`sr_botsnow`
``ice_increments``           :func:`ice_increments`
``ice_increments_nosnow``    :func:`ice_increments_nosnow`
``snow_calc``                :func:`snow_calc`
===========================  ==============================================

Every routine is a pure function of a ``state`` dict (plus scalar / ``(L,)``
arguments) and returns a **new** dict with the same keys plus the fields the
NumPy routine updates in place.  Nothing is mutated.  ``saltice == 1`` (saline
ice) and ``addPrecipToSnow`` are not ported (see "Scope" below).

State layout
============

``state`` is a plain ``dict`` of arrays.  Every array has a leading lake axis
``L``; arrays whose Fortran lower bound is 1 stay **1-based**, i.e. index 0 is
an unused pad (same convention as :mod:`lakematrix.physics.ice` and the other
``jaxbatch`` modules).  ``(L,)`` scalars are stored as 1-D arrays.

Sizes are derived from the shapes::

    M    = Tw1.shape[1] - 2                water layers
    Mice = Ti1.shape[1] - 2                sea-ice layers
    ms   = dens.shape[1] - 1               snow layers (``SNOWSOIL_MOD``)
    ml   = T.shape[1] - 1                  allocated snow storage, ``ml >= ms``

Ice / deep ice (``ARRAYS_WATERSTATE``)::

    Ti1, Ti2, ci_m_roi_v, salice, porice   (L, Mice+2)
    lami_v                                 (L, Mice+1)
    Tis1, Tis2                             (L, Mice+2)
    rad_integr_ice, rad_integr_deepice     (L, Mice+1)
    ddzi                                   (L, Mice+1)   ``grid.ddzi`` (1-based)

Water (only the entries the ice branches read)::

    Tw1, Tw2, PEMF, pt_down_f              (L, M+2)
    lamw, rad_integr_water                 (L, M+1)
    ddz                                    (L, M+1)     ``grid.ddz`` (1-based)

Snow (``SNOWSOIL_MOD``; the ``ms``/``ml`` split is the Fortran allocation)::

    T, wl, extinct, z                      (L, ml+1)
    Tsn, cs, dens, dz, lams                (L, ms+1)
    itop                                   (L,) int    1-based; ``ms`` = no snow
    flag_snow, flag_snow_init              (L,) int

``(L,)`` floats (``SNOWSOIL_MOD`` / ``lake.f90`` case-2 scalars)::

    h1, l1, ls1, snowmass, snowmass_init, hs1, snmelt, swe, SR_botsnow,
    dhi, dhi0, dls, dls0, dhw, dhw0, dhwlow, dhilow, dhwhigh, dhihigh,
    dhwsnmelt, dhwls, dhis, dhip, snmeltice, dhiimp, dhwfsoil, dhwtrib

Static switches (Python ints, *not* jit operands): ``nmeltpoint`` (default 1)
and ``saltice`` (must be 0).  Missing keys fall back to the defaults of
:class:`lakematrix.physics.ice.IceState`.

Notes on the transcription
==========================

* The floating-point expression order of the NumPy reference is preserved
  literally (no algebraic rearrangement); default-real literals are the
  already-rounded constants of :mod:`lakematrix.constants` or come from
  :func:`lakematrix.physics.ice._s` / ``_sdiv``.
* ``L``-dependent branches are ``jnp.where``; the Python ``GOTO``-driven layer
  removal / relabelling of ``SNOW_CALC`` is replaced by two fixed-length
  static sweeps (``ms`` iterations each).  At most one layer is relabelled
  (label 222) or merged (label 333) per iteration and every such iteration
  strictly raises ``itop``, so ``ms`` iterations are enough for any lake.
* ``snow_calc`` **does not** copy ``Tsn`` into ``T`` (``SNOWTEMP`` does that
  before the call, see :func:`lakematrix.step2.step_case2`), and it does not
  append precipitation layers (``addPrecipToSnow`` is not ported - the NumPy
  reference has the same gap).  ``pheat`` is therefore caller supplied.
* The ``swe`` / ``hsnow`` diagnostics are summed with ``jnp.sum`` (ascending
  layer index) instead of the reference's descending accumulation; last-bit
  differences only.
* ``ice_properties`` leaves ``lami_v[Mice+1]`` untouched, exactly like the
  source (that entry is never read either).

Scope
=====

``saltice == 0``; ``addPrecipToSnow``; ``nmeltpoint == 2``; the ``T_DIFF``
snow/ice branches and the ``SNOWTEMP``/``SOILFORLAKE`` integration.  Those live
in the other ``jaxbatch`` modules or are deliberately left out (see the
:mod:`lakematrix.physics.ice` docstring).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..constants import C
from ..physics.ice import _s, _sdiv
from ..physics.soil import meltpnt

F64 = jnp.float64

__all__ = [
    "ice_properties",
    "snow_cond_heat_coef",
    "sr_botsnow",
    "ice_increments",
    "ice_increments_nosnow",
    "snow_calc",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(x):
    """``float64`` array view of ``x``."""
    return jnp.asarray(x, dtype=F64)


def _vec(x, L):
    """Broadcast a scalar / ``(L,)`` value to ``(L,)`` float64."""
    return jnp.broadcast_to(_f(x), (L,))


def _ivec(x, L):
    """Broadcast a scalar / ``(L,)`` value to ``(L,)`` int32."""
    return jnp.broadcast_to(jnp.asarray(x, dtype=jnp.int32), (L,))


def _static_int(state, key, default):
    """A *static* (Python) integer switch of the state dict."""
    if key not in state:
        return int(default)
    return int(np.asarray(state[key]).reshape(-1)[0])


def _dims(state):
    """``(L, ms, ml)`` from the snow array shapes (see the module docstring)."""
    T = _f(state["T"])
    dens = _f(state["dens"])
    L = dens.shape[0]
    ms = dens.shape[1] - 1
    ml = T.shape[1] - 1
    if ms < 1 or ml < ms:
        raise ValueError(f"inconsistent snow shapes: dens {dens.shape}, "
                         f"T {T.shape}")
    return L, ms, ml


def _row(arr, idx):
    """Per-lane gather of column ``idx`` from an ``(L, n)`` array."""
    return jnp.take_along_axis(arr, idx[:, None], axis=1)[:, 0]


def _set_row(arr, idx, val, mask):
    """Per-lane scatter of ``val`` into column ``idx`` where ``mask``."""
    rows = jnp.arange(arr.shape[0])
    cur = _row(arr, idx)
    return arr.at[rows, idx].set(jnp.where(mask, val, cur))


def _shift_222(arr, itop, i_sel, k_sel, found):
    """``for j = i, itop, -1: a(j) = a(k+j-i)`` of ``SNOW_CALC`` label 222."""
    n = arr.shape[1]
    j = jnp.arange(n)[None, :]
    src = jnp.clip(k_sel[:, None] + j - i_sel[:, None], 0, n - 1)
    gathered = jnp.take_along_axis(arr, src, axis=1)
    mask = (found[:, None] & (j >= itop[:, None]) & (j <= i_sel[:, None]))
    return jnp.where(mask, gathered, arr)


def _merge_333(arr, itop, i3, i3p, merged, found):
    """Merge layer ``i3`` into ``i3+1`` and shift, ``SNOW_CALC`` label 333."""
    n = arr.shape[1]
    j = jnp.arange(n)[None, :]
    src = jnp.clip(j - 1, 0, n - 1)
    copied = jnp.take_along_axis(arr, jnp.broadcast_to(src, (arr.shape[0], n)),
                                 axis=1)
    val = jnp.where(j == i3p[:, None], merged[:, None],
                    jnp.where((j >= (itop + 1)[:, None]) & (j <= i3[:, None]),
                              copied, arr))
    return jnp.where(found[:, None], val, arr)


def _extinct_snow(dens):
    """``EXTINCT_SNOW`` (``phys_func.f90:1324-1343``), batched."""
    return jnp.exp(-(0.13 * dens + 3.4))


def _snow_scalar_state(state, key, default, L):
    return _vec(state[key], L) if key in state else _vec(default, L)


# ---------------------------------------------------------------------------
# Ice properties (ci_m_roi_v / lami_v)
# ---------------------------------------------------------------------------

def ice_properties(state):
    """``lake.f90:1315-1329``: pure-ice ``ci_m_roi_v`` / ``lami_v``.

    ``saltice == 0`` only: ``ci_m_roi_v(i) = ci_m_roi`` for ``i = 1..Mice+1``
    and ``lami_v(i) = lami`` for ``i = 1..Mice``.  ``lami_v(Mice+1)`` is not
    written by the source and is returned unchanged, as are the index-0 pads.
    """
    if _static_int(state, "saltice", 0) != 0:
        raise NotImplementedError("ice_properties: only saltice == 0 is ported")
    ci = _f(state["ci_m_roi_v"])
    lam = _f(state["lami_v"])
    Mice = ci.shape[1] - 2
    ici = jnp.arange(ci.shape[1])[None, :]
    ilam = jnp.arange(lam.shape[1])[None, :]
    out = dict(state)
    out["ci_m_roi_v"] = jnp.where((ici >= 1) & (ici <= Mice + 1),
                                  C.ci_m_roi, ci)
    out["lami_v"] = jnp.where((ilam >= 1) & (ilam <= Mice), C.lami, lam)
    return out


# ---------------------------------------------------------------------------
# SNOW_COND_HEAT_COEF
# ---------------------------------------------------------------------------

def snow_cond_heat_coef(state, tempair):
    """``SNOW_COND_HEAT_COEF`` (``snowtemp.f90:100-179``), batched.

    On ``flag_snow_init == 1`` lanes the initial snow pack is built
    (``itop = ms-2``, three 1 cm layers at ``tempair`` plus the interface node
    ``ms``); on every lane ``lams`` and ``cs`` are refreshed for layers
    ``itop..ms-1``.  Returns a new state dict with ``itop``, ``hs1``, ``T``,
    ``wl``, ``dens``, ``dz``, ``lams`` and ``cs`` updated.

    The zero-thickness guard of the reference (``xx = 0`` when
    ``dens*dz <= 0``) is reproduced; ``dzsn`` and the diagnostic counters of
    the source are not computed (as in the NumPy reference).
    """
    L, ms, ml = _dims(state)
    itop = _ivec(state["itop"], L)
    flag_init = _ivec(state["flag_snow_init"], L)
    init = flag_init == 1
    T = _f(state["T"])
    wl = _f(state["wl"])
    dens = _f(state["dens"])
    dz = _f(state["dz"])
    lams = _f(state["lams"])
    cs = _f(state["cs"])
    Ti1 = _f(state["Ti1"])
    tempair = _vec(tempair, L)
    row0, ci, cw = C.row0, C.ci, C.cw

    # ---- initial pack (only on flag_snow_init == 1) ----------------------
    itop0 = ms - 2
    rofresh = _s(67.9) + _s(51.3) * jnp.exp(tempair / _s(2.6))

    # ``T``/``wl`` are allocated 1:ml, the other snow arrays 1:ms.
    idxT = jnp.arange(ml + 1)[None, :]
    idxS = jnp.arange(ms + 1)[None, :]
    m_packT = init[:, None] & (idxT >= itop0) & (idxT <= ms - 1)
    m_nodeT = init[:, None] & (idxT == ms)
    m_belowT = init[:, None] & (idxT >= 1) & (idxT <= itop0 - 1)
    m_packS = init[:, None] & (idxS >= itop0) & (idxS <= ms - 1)
    m_nodeS = init[:, None] & (idxS == ms)
    m_belowS = init[:, None] & (idxS >= 1) & (idxS <= itop0 - 1)

    # 1..ms-1 initialized layers
    wl = jnp.where(m_packT, 0.0, wl)
    T = jnp.where(m_packT, tempair[:, None], T)
    dens = jnp.where(m_packS, rofresh[:, None], dens)
    dz = jnp.where(m_packS, _s(0.01), dz)
    # interface node ms
    wl = jnp.where(m_nodeT, 0.0, wl)
    T = jnp.where(m_nodeT, Ti1[:, 1:2], T)
    dens = jnp.where(m_nodeS, rofresh[:, None], dens)
    dz = jnp.where(m_nodeS, 0.5 * _s(0.01), dz)
    # inactive layers below itop
    T = jnp.where(m_belowT, 0.0, T)
    wl = jnp.where(m_belowT, 0.0, wl)
    dens = jnp.where(m_belowS, 0.0, dens)
    dz = jnp.where(m_belowS, 0.0, dz)

    itop = jnp.where(init, itop0, itop)
    hs1_old = _snow_scalar_state(state, "hs1", 0.0, L)
    hs1 = jnp.where(init, _s(0.02), hs1_old)

    # ---- lams / cs refresh for itop..ms-1 --------------------------------
    d = dens / row0
    lams_new = _s(0.419) * (6.0 * d ** 4 + _s(1.9) * d + _s(0.05))
    den = dens * dz
    xx = jnp.where(den <= 0.0, 0.0, wl[:, :ms + 1] * row0 / den)
    cs_new = ci * (1.0 - xx) + cw * xx
    act = (idxS >= itop[:, None]) & (idxS <= ms - 1)
    lams = jnp.where(act, lams_new, lams)
    cs = jnp.where(act, cs_new, cs)

    out = dict(state)
    out.update(T=T, wl=wl, dens=dens, dz=dz, lams=lams, cs=cs, itop=itop,
               hs1=hs1)
    return out


# ---------------------------------------------------------------------------
# SR_botsnow
# ---------------------------------------------------------------------------

def sr_botsnow(state, shortwave):
    """Shortwave radiation at the bottom of the snow pack (``lake.f90:1339``).

    ``SR_botsnow = shortwave*(1-albedoofsnow)`` attenuated by every active
    snow layer ``itop..ms-1`` as ``EXTINCT_SNOW(dens(i))**dz(i)``; the
    attenuation is skipped while ``flag_snow_init == 1``.  Returns the ``(L,)``
    value; ``state["SR_botsnow"]`` (a diagnostic of ``IceState``) is not
    written back.
    """
    L, ms, _ml = _dims(state)
    itop = _ivec(state["itop"], L)
    flag_init = _ivec(state["flag_snow_init"], L)
    dens = _f(state["dens"])
    dz = _f(state["dz"])
    SR = _vec(shortwave, L) * (1.0 - C.albedoofsnow)
    skip = flag_init == 1
    for i in range(1, ms):
        term = _extinct_snow(dens[:, i]) ** dz[:, i]
        act = (i >= itop) & ~skip
        SR = jnp.where(act, SR * term, SR)
    return SR


# ---------------------------------------------------------------------------
# Case-2 ice thickness increments
# ---------------------------------------------------------------------------

def _pt_down_term(state, Tw1, i):
    """``cw*PEMF(i)*(pt_down_f(i) - 0.5*(Tw1(i+1)+Tw1(i)))``."""
    PEMF = _f(state["PEMF"])
    pt = _f(state["pt_down_f"])
    return (C.cw_m_row0 * PEMF[:, i]
            * (pt[:, i] - 0.5 * (Tw1[:, i + 1] + Tw1[:, i])))


def _flux_bottom(state, Ti1, Tw1, lami_v, ci, dhi, dhi0, dhw0, ddzi, ddz,
                 lamw, l1, h1, dt, M, Mice):
    """``flux1`` / ``flux2`` of the ice-bottom / water-top interface."""
    cw = C.cw_m_row0
    flux1 = (-lami_v[:, Mice] * (Ti1[:, Mice + 1] - Ti1[:, Mice])
             / (ddzi[:, Mice] * l1)
             + _f(state["rad_integr_ice"])[:, Mice]
             + ci[:, Mice + 1] * (dhi - dhi0)
             * (Ti1[:, Mice + 1] - Ti1[:, Mice]) / (2.0 * dt))
    flux2 = (-lamw[:, 1] * (Tw1[:, 2] - Tw1[:, 1]) / (ddz[:, 1] * h1)
             + _f(state["rad_integr_water"])[:, 1]
             + cw * dhw0 * (Tw1[:, 2] - Tw1[:, 1]) / (2.0 * dt)
             + _pt_down_term(state, Tw1, 1))
    return flux1, flux2


def _flux_deepice(state, Tw1, Tis1, dhw, dhw0, dls0, ddz, lamw, h1, dt, M):
    """``flux1`` / ``flux2`` of the deep-ice / water interface."""
    cw = C.cw_m_row0
    ddzi = _f(state["ddzi"])
    ls1 = _f(state["ls1"])
    flux1 = (-lamw[:, M] * (Tw1[:, M + 1] - Tw1[:, M]) / (ddz[:, M] * h1)
             + _f(state["rad_integr_water"])[:, M]
             + cw * (dhw - dhw0) * (Tw1[:, M + 1] - Tw1[:, M])
             / (2.0 * dt)
             + _pt_down_term(state, Tw1, M))
    flux2 = (-C.lami * (Tis1[:, 2] - Tis1[:, 1]) / (ddzi[:, 1] * ls1)
             + _f(state["rad_integr_deepice"])[:, 1]
             + C.ci_m_roi * dls0 * (Tis1[:, 2] - Tis1[:, 1]) / (2.0 * dt))
    return flux1, flux2


def _limit_dhw(flux1, flux2, dt, ddz, h1, ddzi, l1, Mice):
    """The ``min(max(...))`` freezing/melting limit of the source."""
    dhwlow = jnp.minimum(
        jnp.maximum(dt * (flux1 - flux2) / C.row0_m_Lwi, -ddz[:, 1] * h1),
        ddzi[:, Mice] * l1 * C.roi_d_row0)
    dhilow = -dhwlow * C.roi_d_row0
    return dhwlow, dhilow


def ice_increments(state, dt):
    """``lake.f90:1367-1445``: case 2.1 (water + ice + snow) increments.

    Returns a copy of ``state`` with the increments of the NumPy reference
    written to it (``dhwlow``, ``dhilow``, ``dhwhigh``, ``dhihigh``,
    ``dhwsnmelt``, ``dhwls``, ``dhi``, ``dhi0``, ``dls``, ``dls0``,
    ``snmeltice``, ``dhiimp``, ``dhw``, ``dhw0``) plus the clipped ``Ti1``.
    Two additional ``(L,)`` outputs of the reference dict, ``fracsn`` and
    ``fracimp``, are stored as well (``fracsn``/``fracimp`` keys).

    ``saltice == 0`` only.  Call :func:`snow_cond_heat_coef` first: the
    snow-ice interface reads ``cs``/``dens``/``dz`` at ``ms``.
    """
    if _static_int(state, "saltice", 0) != 0:
        raise NotImplementedError("ice_increments: only saltice == 0 is ported")
    L, ms, _ml = _dims(state)
    dt = _vec(dt, L)
    Ti1 = _f(state["Ti1"])
    Tw1 = _f(state["Tw1"])
    Tis1 = _f(state["Tis1"])
    lami_v = _f(state["lami_v"])
    ci = _f(state["ci_m_roi_v"])
    ddzi = _f(state["ddzi"])
    ddz = _f(state["ddz"])
    lamw = _f(state["lamw"])
    cs, dens, dz = _f(state["cs"]), _f(state["dens"]), _f(state["dz"])
    l1 = _vec(state["l1"], L)
    h1 = _vec(state["h1"], L)
    ls1 = _vec(state["ls1"], L)
    dhi = _vec(state["dhi"], L)
    dhi0 = _vec(state["dhi0"], L)
    dhw0 = _vec(state["dhw0"], L)
    dhw = _vec(state["dhw"], L)
    dls0_in = _vec(state["dls0"], L)
    dhwfsoil = _vec(state["dhwfsoil"], L)
    dhwtrib = _vec(state["dhwtrib"], L)
    snmelt = _vec(state["snmelt"], L)
    snowmass = _vec(state["snowmass"], L)
    M, Mice = Tw1.shape[1] - 2, Ti1.shape[1] - 2
    x = meltpnt(0.0, 0.0, _static_int(state, "nmeltpoint", 1))

    flux1, flux2 = _flux_bottom(state, Ti1, Tw1, lami_v, ci, dhi, dhi0,
                                dhw0, ddzi, ddz, lamw, l1, h1, dt, M, Mice)
    dhwlow, dhilow = _limit_dhw(flux1, flux2, dt, ddz, h1, ddzi, l1, Mice)

    # --- melting at the ice top (fresh-water melting point) --------------
    over = Ti1[:, 1] > x + C.T_phase_threshold
    dhwhigh = jnp.where(
        over,
        (Ti1[:, 1] - x - C.T_phase_threshold) * ci[:, 1] * l1 * ddzi[:, 1]
        * 0.5 / C.row0_m_Lwi, 0.0)
    dhihigh = jnp.where(over, -dhwhigh * C.roi_d_row0, 0.0)
    Ti1 = Ti1.at[:, 1].set(jnp.where(over, x + C.T_phase_threshold, Ti1[:, 1]))

    # --- freezing of snowmelt at the snow-ice interface ------------------
    y = 0.5 * (cs[:, ms] * dens[:, ms] * dz[:, ms]
               + ci[:, 1] * ddzi[:, 1] * l1)
    dTmax = jnp.maximum((1.0 - C.snmeltwat) * snmelt * dt * C.row0 * C.Lwi / y,
                        0.0)
    dTisnmelt = jnp.minimum(dTmax, x + C.T_phase_threshold - Ti1[:, 1])
    dTisnmelt = jnp.maximum(dTisnmelt, 0.0)
    fracsn = dTisnmelt / (dTmax + C.small_value)
    snmeltice = dTmax * y / (C.row0 * C.Lwi) * fracsn
    dhihigh = dhihigh + snmeltice * C.roi_d_row0

    # --- impoundment of ice from above by water through cracks -----------
    dTmax_imp = jnp.maximum((snowmass + l1 * (C.roi - C.row0)) * C.Lwi / y,
                            0.0)
    dTiimp = jnp.minimum(dTmax_imp,
                         x + C.T_phase_threshold - Ti1[:, 1] - dTisnmelt)
    dTiimp = jnp.maximum(dTiimp, 0.0)
    fracimp = dTiimp / (dTmax_imp + C.small_value)
    dhiimp = C.icetopfr * dTmax_imp * y / (C.Lwi * C.row0) * fracimp
    dhihigh = dhihigh + dhiimp * C.roi_d_row0
    dhwhigh = dhwhigh - dhiimp

    # --- snowmelt water reaching the water column -------------------------
    dhwsnmelt = (C.snmeltwat * snmelt * dt
                 + (1.0 - C.snmeltwat) * snmelt * dt * (1.0 - fracsn))

    # --- deep (bottom) ice -------------------------------------------------
    has_deep = ls1 != 0.0
    f1d, f2d = _flux_deepice(state, Tw1, Tis1, dhw, dhw0, dls0_in, ddz, lamw,
                             h1, dt, M)
    dhwls_d = dt * (f1d - f2d) / C.row0_m_Lwi
    dls_d = -dhwls_d * C.roi_d_row0

    dhwls = jnp.where(has_deep, dhwls_d, 0.0)
    dls = jnp.where(has_deep, dls_d, 0.0)
    dls0 = jnp.where(has_deep, dls_d, dls0_in)
    dhw = jnp.where(
        has_deep,
        dhwhigh + dhwlow + dhwfsoil + dhwls_d + dhwtrib + dhwsnmelt,
        dhwhigh + dhwlow + dhwfsoil + dhwtrib + dhwsnmelt)
    dhw0 = jnp.where(has_deep,
                     dhwlow + dhwhigh + dhwsnmelt,
                     dhwlow + dhwhigh + dhwsnmelt)

    dhi_out = dhilow + dhihigh
    dhi0_out = dhihigh

    out = dict(state)
    out.update(Ti1=Ti1, dhwlow=dhwlow, dhilow=dhilow, dhwhigh=dhwhigh,
               dhihigh=dhihigh, dhwsnmelt=dhwsnmelt, dhwls=dhwls,
               dhi=dhi_out, dhi0=dhi0_out, dls=dls, dls0=dls0,
               snmeltice=snmeltice, dhiimp=dhiimp, fracsn=fracsn,
               fracimp=fracimp, dhw=dhw, dhw0=dhw0)
    return out


def ice_increments_nosnow(state, dt, precip, tempair, Elatent):
    """``lake.f90:1657-1717``: case 2.2 (water + ice, no snow) increments.

    Same outputs as :func:`ice_increments` plus ``dhis``/``dhip`` and the
    ``snowmass`` / ``flag_snow`` / ``itop`` / ``hs1`` bookkeeping of the
    source.  ``dhw``/``dhw0`` carry no snowmelt term here.  The stale-value
    branches (``dhwhigh``/``dhihigh``/``dhis``/``dls0``/``dhwls`` read from
    the state) are reproduced, as is the case that leaves ``dhwls`` untouched.
    """
    if _static_int(state, "saltice", 0) != 0:
        raise NotImplementedError(
            "ice_increments_nosnow: only saltice == 0 is ported")
    L, ms, _ml = _dims(state)
    dt = _vec(dt, L)
    precip = _vec(precip, L)
    tempair = _vec(tempair, L)
    Elatent = _vec(Elatent, L)
    Ti1 = _f(state["Ti1"])
    Tw1 = _f(state["Tw1"])
    Tis1 = _f(state["Tis1"])
    lami_v = _f(state["lami_v"])
    ci = _f(state["ci_m_roi_v"])
    ddzi = _f(state["ddzi"])
    ddz = _f(state["ddz"])
    lamw = _f(state["lamw"])
    l1 = _vec(state["l1"], L)
    h1 = _vec(state["h1"], L)
    ls1 = _vec(state["ls1"], L)
    dhi = _vec(state["dhi"], L)
    dhi0 = _vec(state["dhi0"], L)
    dhw0 = _vec(state["dhw0"], L)
    dhw = _vec(state["dhw"], L)
    dls0_in = _vec(state["dls0"], L)
    dhwls_in = _vec(state["dhwls"], L)
    dhwhigh_in = _vec(state["dhwhigh"], L)
    dhihigh_in = _vec(state["dhihigh"], L)
    dhis_in = _vec(state["dhis"], L)
    dhwfsoil = _vec(state["dhwfsoil"], L)
    dhwtrib = _vec(state["dhwtrib"], L)
    snowmass = _vec(state["snowmass"], L)
    flag_snow = _ivec(state["flag_snow"], L)
    itop = _ivec(state["itop"], L)
    hs1 = _vec(state["hs1"], L) if "hs1" in state else _vec(0.0, L)
    M, Mice = Tw1.shape[1] - 2, Ti1.shape[1] - 2
    x = meltpnt(0.0, 0.0, _static_int(state, "nmeltpoint", 1))
    xthr = x + C.T_phase_threshold

    over = Ti1[:, 1] > xthr
    dhwhigh_c = ((Ti1[:, 1] - xthr) * ci[:, 1] * ddzi[:, 1] * l1 * 0.5
                 / C.row0_m_Lwi)
    dhihigh_c = -dhwhigh_c * C.roi_d_row0
    dhwhigh = jnp.where(over, dhwhigh_c, dhwhigh_in)
    dhihigh = jnp.where(over, dhihigh_c, dhihigh_in)
    dhis = jnp.where(over, 0.0, dhis_in)
    Ti1 = Ti1.at[:, 1].set(jnp.where(over, xthr, Ti1[:, 1]))

    # --- initial thin snow layer, to be used at the next timestep ---------
    cond_p = (precip > 0.0) & (tempair < 0.0) & (l1 > 0.05)
    snowmass_p = jnp.where(cond_p, snowmass + precip * dt * C.row0, snowmass)
    switch = cond_p & (snowmass_p / C.rofrsn >= 3.0 * C.dzmin)
    flag_snow = jnp.where(switch, 1, flag_snow)
    hs1 = jnp.where(switch, snowmass_p / C.rofrsn + C.small_value, hs1)
    itop = jnp.where(switch, ms - 2, itop)
    dhip_p = jnp.where(cond_p, 0.0, precip * C.row0_d_roi * dt)
    melt_snow = (tempair > 0.0) & (snowmass_p > 0.0)
    dhip = jnp.where(melt_snow, dhip_p + snowmass_p / C.row0, dhip_p)
    snowmass = jnp.where(melt_snow, 0.0, snowmass_p)

    flux1, flux2 = _flux_bottom(state, Ti1, Tw1, lami_v, ci, dhi, dhi0,
                                dhw0, ddzi, ddz, lamw, l1, h1, dt, M, Mice)
    dhwlow, dhilow = _limit_dhw(flux1, flux2, dt, ddz, h1, ddzi, l1, Mice)

    below = Ti1[:, 1] < xthr
    dhis = jnp.where(below, -Elatent / (C.roi * C.Liv) * dt, dhis)
    dhwhigh = jnp.where(below, 0.0, dhwhigh)
    dhihigh = jnp.where(below, 0.0, dhihigh)

    dhi_out = dhihigh + dhilow + dhis + dhip
    dhi0_out = dhis + dhihigh + dhip

    has_deep = ls1 != 0.0
    f1d, f2d = _flux_deepice(state, Tw1, Tis1, dhw, dhw0, dls0_in, ddz, lamw,
                             h1, dt, M)
    dhwls_d = dt * (f1d - f2d) / C.row0_m_Lwi
    dls_d = -dhwls_d * C.roi_d_row0
    dhwls = jnp.where(has_deep, dhwls_d, dhwls_in)
    dls = jnp.where(has_deep, dls_d, 0.0)
    dls0 = jnp.where(has_deep, dls_d, dls0_in)
    dhw = jnp.where(
        has_deep,
        dhwhigh + dhwfsoil + dhwls_d + dhwlow + dhwtrib,
        dhwhigh + dhwfsoil + dhwlow + dhwtrib)
    dhw0 = dhwhigh + dhwlow

    out = dict(state)
    out.update(Ti1=Ti1, dhwlow=dhwlow, dhilow=dhilow, dhwhigh=dhwhigh,
               dhihigh=dhihigh, dhis=dhis, dhip=dhip, dhwls=dhwls,
               dhi=dhi_out, dhi0=dhi0_out, dls=dls, dls0=dls0,
               dhw=dhw, dhw0=dhw0, dhwsnmelt=_vec(0.0, L),
               snowmass=snowmass, flag_snow=flag_snow, itop=itop, hs1=hs1)
    return out


# ---------------------------------------------------------------------------
# SNOW_CALC
# ---------------------------------------------------------------------------

def snow_calc(state, dt, ts, t2, snow_sc, eflux, Erad, Elatent, precip,
              pheat):
    """``SNOW_CALC`` (``snowcalc.f90:5-497``), batched snow pack.

    Snow melt, percolation, densification and the layer bookkeeping of the
    NumPy reference.  Inputs besides ``state`` (each scalar or ``(L,)``):
    ``ts`` (snow surface temperature, C), ``t2`` (2 m air temperature, C),
    ``snow_sc`` (water equivalent before the step, m), ``eflux``/``Erad``
    (surface heat flux / absorbed shortwave, W/m2), ``Elatent`` (latent heat
    flux, W/m2), ``precip`` (m/s) and ``pheat`` (rain heat content, W/m2 -
    ``addPrecipToSnow`` is not ported, pass 0 without precipitation).

    Returns a new state dict with ``T``, ``wl``, ``dz``, ``dens``, ``extinct``,
    ``z``, ``itop``, ``snmelt``, ``swe`` and ``hs1`` updated, plus the outputs
    ``snow_sc`` (new water equivalent, m), ``hsnow`` (depth, m), ``phase``
    (source accumulator) and ``precip`` (rate restored after the internal
    ``dt`` scaling).  ``Tsn`` is *not* copied into ``T``; the caller does that
    before the call (``SNOWTEMP`` / :func:`lakematrix.step2.step_case2`).

    Intentional deviations from the reference, all bookkeeping only: the
    ``GOTO``-driven removal/relabel loops are two fixed ``ms``-iteration
    sweeps, and ``swe``/``hsnow`` are ``jnp.sum`` reductions instead of the
    descending ``do`` accumulations.
    """
    L, ms, ml = _dims(state)
    dt = _vec(dt, L)
    ts = _vec(ts, L)
    t2 = _vec(t2, L)
    snow_sc = _vec(snow_sc, L)
    eflux = _vec(eflux, L)
    Erad = _vec(Erad, L)
    Elatent = _vec(Elatent, L)
    precip = _vec(precip, L)
    pheat = _vec(pheat, L)

    T = _f(state["T"])
    wl = _f(state["wl"])
    extinct = _f(state["extinct"])
    z = _f(state["z"])
    dens = _f(state["dens"])
    dz = _f(state["dz"])
    itop = _ivec(state["itop"], L)

    row0, roi = C.row0, C.roi
    Lwi, Liv = C.Lwi, C.Liv
    whc, hcond = C.whc, C.hcond
    g, csnow = C.g, C.csnow
    rosn_dry_ref = C.rosn_dry_ref
    S1E4 = _s(1.0e-4)
    S1E6 = _s(1.0e-6)

    # t(i) of the previous step, needed by the Phase accumulator
    ti = T
    snmelt = _vec(0.0, L)
    phase = jnp.zeros(L)

    densny = _s(67.9) + _s(51.3) * jnp.exp(t2 / _s(2.6))
    Elatent1 = Elatent
    Elatent1 = jnp.where(ts > _s(1.0e-5), 0.0, Elatent1)
    Elatent1 = jnp.where((snow_sc < _s(1.0e-6)) & (Elatent > 0.0), 0.0,
                         Elatent1)
    evap = Elatent1 / Liv

    # the interface node thickness is folded into the bottom snow layer
    dz = dz.at[:, ms - 1].set(dz[:, ms - 1] + dz[:, ms])

    idx = jnp.arange(ms + 1)[None, :]
    m_ext = (idx >= itop[:, None]) & (idx <= ms - 1)
    # ``dens`` is 1:ms but ``extinct`` is allocated 1:ml; only the snow layers
    # itop..ms-1 are refreshed, the tail is left as it was.
    extinct_head = jnp.where(m_ext, _extinct_snow(dens), extinct[:, :ms + 1])
    extinct = jnp.concatenate([extinct_head, extinct[:, ms + 1:]], axis=1)

    precip = precip * dt

    # --- top snow layer (dynamic index itop) ------------------------------
    has_snow = itop != ms
    single = itop == ms - 1
    ext_top = _row(extinct, itop)
    dz_top0 = _row(dz, itop)
    dens_top0 = _row(dens, itop)
    wl_top0 = _row(wl, itop)
    ext0 = ext_top == 0.0
    e_pow = ext_top ** dz_top0
    radfrac = Erad * (1.0 - e_pow)
    Eflux1 = eflux + pheat + jnp.where(ext0 | single, Erad, radfrac)
    countrad = jnp.where(~ext0 & ~single, radfrac, 0.0)
    Erad1 = Erad
    Eout = jnp.zeros(L)

    warm = ts >= _s(-1.0e-4)
    limit = dens_top0 * dz_top0 - wl_top0 * row0
    m_ok = Eflux1 / Lwi * dt <= limit
    smelt_s = jnp.where(m_ok, Eflux1 / Lwi / row0,
                        (dz_top0 * dens_top0 / row0 - wl_top0) / dt)
    eout_s = jnp.where(m_ok, 0.0, Eflux1 - smelt_s * row0 * Lwi)
    pos = Eflux1 > 0.0
    smelt = jnp.where(pos & single & warm, smelt_s, 0.0)
    Eout = jnp.where(pos & ~single, Eflux1,
                     jnp.where(pos & single & warm, eout_s, Eout))

    smelt = smelt + wl_top0 / dt
    qbase = smelt
    dmass = dens_top0 * dz_top0 - (evap + smelt * row0) * dt
    densev = jnp.where(evap > 0.0, dens_top0, roi)
    dztop = dz_top0 - (smelt * row0 / dens_top0 + evap / densev) * dt
    dztop = jnp.where(dztop == 0.0, S1E6, dztop)
    dens_top = jnp.maximum(dmass / dztop, 0.0)
    over_d = dens_top > roi
    dztop = jnp.where(over_d, dztop * dens_top / roi, dztop)
    dens_top = jnp.where(over_d, roi, dens_top)

    dz = _set_row(dz, itop, dztop, has_snow)
    dens = _set_row(dens, itop, dens_top, has_snow)
    wl = _set_row(wl, itop, 0.0, has_snow)
    snmelt = jnp.where(has_snow & single, qbase, snmelt)
    qbase = jnp.where(has_snow, qbase, 0.0)
    Eout = jnp.where(has_snow, Eout, 0.0)
    countrad = jnp.where(has_snow, countrad, 0.0)
    counter = _row(dz, itop)

    # --- internal layers (i = itop+1 .. ms-1) -----------------------------
    for i in range(1, ms):
        active = i > itop
        Ein = Eout
        dz_i = dz[:, i]
        dens_i = dens[:, i]
        T_i = T[:, i] * (dz_i * dens_i
                         / (qbase * row0 * dt + dz_i * dens_i))
        dzi = dz_i
        wl_i = wl[:, i] + qbase * dt
        wl_i = jnp.maximum(wl_i, 0.0)
        w0 = wl_i
        ext_i = extinct[:, i]
        ext_zero = ext_i == 0.0
        p_hi = ext_i ** counter
        p_lo = ext_i ** (counter + dzi)
        term = Erad1 * (p_hi - p_lo)
        if i == ms - 1:
            Ei = jnp.where(ext_zero, Ein, Ein + (Erad1 - countrad))
        else:
            Ei = jnp.where(ext_zero, Ein, Ein + term)
        upd = active & ~ext_zero
        countrad = jnp.where(upd, countrad + term, countrad)
        counter = jnp.where(upd, counter + dzi, counter)

        dz_after = dz_i + qbase * dt          # st.dz(i) after +=
        ziz = dzi / dz_after
        dens_mix = ziz * dens_i + (1.0 - ziz) * row0
        dz_i = jnp.where(active, dz_after, dz_i)
        dens_i = jnp.where(active, dens_mix, dens_i)

        # ------------------------- Ei > 0 --------------------------------
        T_ge = T_i >= S1E4
        # (A) T >= 1e-4: add the layer heat capacity, then melt
        EiA = Ei + csnow * dens_i * T_i * dz_i / dt
        limA = dz_i * dens_i - wl_i * row0
        m_okA = EiA / Lwi * dt <= limA
        smeltA = jnp.where(m_okA, EiA / Lwi / row0,
                           (dz_i * dens_i / row0 - wl_i) / dt)
        eoutA = jnp.where(m_okA, 0.0, EiA - smeltA * row0 * Lwi)
        dziA = jnp.where(m_okA, dzi, 0.0)
        wlA = wl_i + smeltA * dt
        # (B) T < 1e-4: warm/cool the layer, then possible immediate melt
        T1 = T_i + Ei * dt / (csnow * dens_i * dz_i)
        cold = T1 < _s(-1.0e-4)
        rf1 = (-T1 * csnow * dz_i * dens_i / Lwi / dt / row0)
        rf2 = wl_i / dt
        warm_enough = wl_i > (-T1 * csnow * dz_i * dens_i / Lwi / row0)
        rfB = jnp.where(warm_enough, rf1, rf2)
        TB = jnp.where(warm_enough, 0.0,
                       T1 + rf2 * row0 * dt * Lwi / (csnow * dz_i * dens_i))
        wlB = jnp.where(warm_enough, wl_i - rf1 * dt, 0.0)
        T_pre = jnp.where(cold, TB, T1)
        wl_pre = jnp.where(cold, wlB, wl_i)
        hotB = T_pre >= S1E4
        hotB_ok = csnow * dens_i * T_pre * dz_i / Lwi <= dz_i * dens_i
        smeltB = jnp.where(hotB_ok,
                           csnow * dens_i * T_pre * dz_i / dt / Lwi / row0,
                           dz_i * dens_i / dt / row0)
        eoutB = jnp.where(hotB_ok, 0.0,
                          csnow * dens_i * T_pre * dz_i / dt
                          - smeltB * Lwi * row0)
        dziB = jnp.where(hotB_ok, dzi, 0.0)
        T_Epos = jnp.where(T_ge, 0.0, jnp.where(hotB, 0.0, T_pre))
        wl_Epos = jnp.where(T_ge, wlA,
                            jnp.where(hotB, wl_pre + smeltB * dt, wl_pre))
        Eout_Epos = jnp.where(
            T_ge, eoutA,
            jnp.where(hotB, eoutB, jnp.where(cold, 0.0, Ein)))
        rf_Epos = jnp.where(T_ge, 0.0, jnp.where(cold, rfB, 0.0))
        dzi_Epos = jnp.where(T_ge, dziA, jnp.where(hotB, dziB, dzi))
        smelti_Epos = jnp.where(T_ge, smeltA,
                                jnp.where(hotB, smeltB, 0.0))

        # ------------------------- Ei <= 0 -------------------------------
        hotC = T_i >= S1E4
        hotC_ok = csnow * dens_i * T_i * dz_i / Lwi <= dz_i * dens_i
        smeltC = jnp.where(hotC_ok, csnow * dens_i * T_i * dz_i / dt / Lwi
                           / row0, dz_i * dens_i / dt / row0)
        eoutC = jnp.where(hotC_ok, 0.0,
                          csnow * dens_i * T_i * dz_i / dt
                          - smeltC * Lwi * row0)
        dziC = jnp.where(hotC_ok, dzi, 0.0)
        coldC = wl_i > (-csnow * dens_i * T_i * dz_i / Lwi / row0)
        rfC1 = (-csnow * dens_i * T_i * dz_i / dt / Lwi / row0)
        rfC2 = wl_i / dt
        rfC = jnp.where(coldC, rfC1, rfC2)
        T_cold = jnp.where(coldC, 0.0, T_i + rfC2 * row0 * dt * Lwi
                           / (csnow * dz_i * dens_i))
        wl_cold = jnp.where(coldC, wl_i - rfC1 * dt, 0.0)
        hotC2 = T_cold >= S1E4
        hotC2_ok = csnow * dens_i * T_cold * dz_i / Lwi <= dz_i * dens_i
        smeltC2 = jnp.where(hotC2_ok,
                            csnow * dens_i * T_cold * dz_i / dt / Lwi / row0,
                            dz_i * dens_i / dt / row0)
        eoutC2 = jnp.where(hotC2_ok, 0.0,
                           csnow * dens_i * T_cold * dz_i / dt
                           - smeltC2 * Lwi * row0)
        dziC2 = jnp.where(hotC2_ok, dzi, 0.0)
        T_Cneg = jnp.where(hotC, 0.0, jnp.where(hotC2, 0.0, T_cold))
        wl_Cneg = jnp.where(hotC, wl_i + smeltC * dt,
                            jnp.where(hotC2, wl_cold + smeltC2 * dt,
                                      wl_cold))
        Eout_Cneg = jnp.where(hotC, eoutC,
                              jnp.where(hotC2, eoutC2, 0.0))
        dzi_Cneg = jnp.where(hotC, dziC, jnp.where(hotC2, dziC2, dzi))
        rf_Cneg = jnp.where(hotC, 0.0, rfC)

        pos_e = Ei > 0.0
        T_new = jnp.where(pos_e, T_Epos, T_Cneg)
        wl_new = jnp.where(pos_e, wl_Epos, wl_Cneg)
        Eout_new = jnp.where(pos_e, Eout_Epos, Eout_Cneg)
        rf = jnp.where(pos_e, rf_Epos, rf_Cneg)
        dzi_new = jnp.where(pos_e, dzi_Epos, dzi_Cneg)
        smelti = jnp.where(pos_e, smelti_Epos,
                           jnp.where(hotC, smeltC,
                                     jnp.where(hotC2, smeltC2, 0.0)))

        # ------------------------ internal q -----------------------------
        wl_post = jnp.where(active, wl_new, wl[:, i])
        T_post = jnp.where(active, T_new, T[:, i])
        dz_post = jnp.where(active, dz_after, dz[:, i])
        dens_post = jnp.where(active, dens_mix, dens[:, i])
        dzi_cur = jnp.where(active, dzi_new, dz[:, i])

        # ``rho_dry`` uses ``w0`` (the water content *before* the Ei blocks),
        # while porosity / fukt / qbase below use the updated ``wl``.
        rho_dry = ((dens_post - w0 / dz_post * row0)
                   / (1.0 - w0 / dz_post))
        neg = rho_dry < 0.0
        has_dry = dens_post - rosn_dry_ref > 0.0
        wl_corr = jnp.where(
            has_dry,
            dz_post * (dens_post - rosn_dry_ref) / (row0 - rosn_dry_ref),
            0.0)
        rho_corr = jnp.where(has_dry, rosn_dry_ref, dens_post)
        wl_q = jnp.where(neg, wl_corr, wl_post)
        rho_q = jnp.where(neg, rho_corr, rho_dry)

        porosity = 1.0 - rho_q / roi - wl_q / dz_post
        porosity = jnp.maximum(porosity, whc + 0.1)
        p1 = porosity - whc
        fukt = wl_q / dz_post
        wet = fukt > whc
        fukt2 = (fukt - whc) / p1
        q0 = hcond * fukt2 ** 3.0
        qbase_wet = jnp.minimum(q0, wl_q / dt)
        wl_wet = wl_q - qbase_wet * dt
        qbase_q = jnp.where(wet, qbase_wet, 0.0)
        wl_q2 = jnp.where(wet, wl_wet, wl_q)

        thin = dzi_cur <= S1E6
        qbase_thin = qbase_q + wl_q2 / dt
        dzmasi = dz_after * dens_post - qbase_q * dt * row0
        dz_thick = dz_after + row0 * (
            -qbase_q / row0 - rf / row0 + rf / roi
            - smelti / rho_q + smelti / row0) * dt
        dens_thick = jnp.maximum(dzmasi / dz_thick, 0.0)
        over_t = dens_thick > roi
        dz_thick = jnp.where(over_t, dz_thick * dens_thick / roi, dz_thick)
        dens_thick = jnp.where(over_t, roi, dens_thick)

        qbase_new = jnp.where(thin, qbase_thin, qbase_q)
        dz_new = jnp.where(thin, dzi_cur, dz_thick)
        dens_new = jnp.where(thin, dens_post, dens_thick)

        T = T.at[:, i].set(T_post)
        wl = wl.at[:, i].set(jnp.where(active, wl_q2, wl[:, i]))
        dz = dz.at[:, i].set(jnp.where(active, dz_new, dz[:, i]))
        dens = dens.at[:, i].set(jnp.where(active, dens_new, dens[:, i]))
        qbase = jnp.where(active, qbase_new, qbase)
        Eout = jnp.where(active, Eout_new, Eout)
        phase = phase + jnp.where(
            active, (T_post - ti[:, i]) * dens_new * dz_new / snow_sc / dt, 0.0)
        snmelt = jnp.where(active & (i == ms - 1), qbase, snmelt)

    # --------------------- layer removal / relabelling --------------------
    goto123 = (itop == ms - 1) & (_row(dz, itop) < S1E6)
    maskB = ~goto123 & (itop < ms) & (_row(dz, itop) <= S1E6)
    dz = _set_row(dz, itop, 0.0, goto123 | maskB)
    dens = _set_row(dens, itop, 0.0, goto123 | maskB)
    itop = itop + maskB.astype(jnp.int32)

    # label 222: drop empty layers and shift the ones above down
    jidx = jnp.arange(ms + 1)[None, :]
    nomaskA = ~goto123
    for _ in range(ms):
        empty = ((jidx >= itop[:, None]) & (jidx <= ms - 1)
                 & (dz <= S1E6) & nomaskA[:, None])
        found = jnp.any(empty, axis=1)
        i_sel = jnp.max(jnp.where(empty, jidx, -1), axis=1)
        better = ((jidx >= itop[:, None]) & (jidx <= i_sel[:, None] - 1)
                  & (dz > S1E6))
        k_sel = jnp.max(jnp.where(better, jidx, -1), axis=1)
        k_sel = jnp.where(k_sel < 0, itop - 1, k_sel)
        dz = _shift_222(dz, itop, i_sel, k_sel, found)
        dens = _shift_222(dens, itop, i_sel, k_sel, found)
        T = _shift_222(T, itop, i_sel, k_sel, found)
        wl = _shift_222(wl, itop, i_sel, k_sel, found)
        itop = jnp.where(found, itop - k_sel + i_sel, itop)

    # label 333: merge layers thinner than dzmin into the one below
    for _ in range(ms):
        cand = ((jidx >= itop[:, None]) & (jidx <= ms - 2)
                & (dz < C.dzmin) & nomaskA[:, None]
                & (itop < ms - 1)[:, None])
        found3 = jnp.any(cand, axis=1)
        i3 = jnp.min(jnp.where(cand, jidx, ms), axis=1)
        i3 = jnp.where(found3, i3, 0)
        i3p = i3 + 1
        dz_i, dz_p = _row(dz, i3), _row(dz, i3p)
        denom = dz_i + dz_p
        m_dens = (_row(dens, i3p) * dz_p / denom
                  + _row(dens, i3) * dz_i / denom)
        m_T = _row(T, i3p) * dz_p / denom + _row(T, i3) * dz_i / denom
        m_wl = _row(wl, i3p) + _row(wl, i3)
        m_dz = dz_p + dz_i
        dens = _merge_333(dens, itop, i3, i3p, m_dens, found3)
        T = _merge_333(T, itop, i3, i3p, m_T, found3)
        wl = _merge_333(wl, itop, i3, i3p, m_wl, found3)
        dz = _merge_333(dz, itop, i3, i3p, m_dz, found3)
        itop = itop + found3.astype(jnp.int32)

    # ------------------------- swe / snow depth ---------------------------
    sel = (jidx >= itop[:, None]) & (jidx <= ms - 1)
    swe = jnp.sum(jnp.where(sel, dz * dens, 0.0), axis=1)
    hsnow = jnp.sum(jnp.where(sel, dz, 0.0), axis=1)

    # ------------- densification due to gravity and metamorphism ----------
    a2 = _s(0.00000066)
    sigma = 75.0
    cum = jnp.zeros(L)
    for j in range(1, ms):
        cum = jnp.where(j == itop, 0.0, cum)
        dens_j, dz_j = dens[:, j], dz[:, j]
        P = cum + dens_j * g * dz_j
        act = j > itop
        do = act & (dens_j < 3 * densny)
        dens_old = dens_j
        eta = (a2 * jnp.exp(_s(19.3) * dens_j / roi)
               * jnp.exp(_sdiv(67300.0, 8.314) / (T[:, j] + _s(273.15))))
        dens_new = jnp.minimum(
            dens_j + _s(0.001) * (dt * dens_j * 1000.0 * (sigma + P) / eta),
            roi)
        dz_new = dz_j * dens_old / dens_new
        dens_j2 = jnp.where(do, dens_new, dens_j)
        dz_j2 = jnp.where(do, dz_new, dz_j)
        dens = dens.at[:, j].set(dens_j2)
        dz = dz.at[:, j].set(dz_j2)
        cum = jnp.where(j >= itop, cum + dens_j2 * g * dz_j2, cum)

    swe = jnp.sum(jnp.where(sel, dz * dens, 0.0), axis=1)
    hsnow = jnp.sum(jnp.where(sel, dz, 0.0), axis=1)
    snow_sc_out = swe

    # --------------------- dz(ms) / z bookkeeping -------------------------
    # The source tests the *post-removal* itop here.
    single_final = itop == ms - 1
    dz_m1 = dz[:, ms - 1]
    new_dz_m = jnp.where(single_final, dz_m1 / 2.0, dz_m1 / 3.0)
    new_dz_m1 = dz_m1 - new_dz_m
    dz = dz.at[:, ms].set(new_dz_m)
    dz = dz.at[:, ms - 1].set(new_dz_m1)

    z_m = z[:, ms]
    z_m1 = jnp.where(single_final, z_m - new_dz_m1 - new_dz_m,
                     z_m - new_dz_m1 / 2.0 - new_dz_m)
    z = z.at[:, ms - 1].set(z_m1)
    for j in range(ms - 2, 0, -1):
        act = j > itop
        cand = z[:, j + 1] - dz[:, j] / 2.0 - dz[:, j + 1] / 2.0
        z = z.at[:, j].set(jnp.where(act, cand, z[:, j]))
    # ``itop + 1`` is clamped for the masked (itop == ms / ms-1) lanes only.
    ip1 = jnp.minimum(itop + 1, ms - 1)
    z_itop = _row(z, ip1) - _row(dz, ip1) / 2.0 - _row(dz, itop)
    z = _set_row(z, itop, z_itop, itop < ms - 1)

    dz = jnp.where((jidx >= 1) & (jidx < itop[:, None]), 0.0, dz)

    precip = precip / dt

    out = dict(state)
    out.update(T=T, wl=wl, dz=dz, dens=dens, extinct=extinct, z=z,
               itop=itop, snmelt=snmelt, swe=snow_sc_out, hs1=hsnow,
               snow_sc=snow_sc_out, hsnow=hsnow, phase=phase, precip=precip)
    return out
