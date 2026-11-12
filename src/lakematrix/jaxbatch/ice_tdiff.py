"""Batched JAX port of ``T_DIFF`` for the ice/snow branches (cases 2 and 3).

NumPy reference: :func:`lakematrix.physics.ice.t_diff_ice` (Fortran
``t_solver_mod.f90:219-656``) together with ``diff_coef_ice`` /
``diff_coef_snow`` / ``diff_coef_deepice`` and the soil/water coefficient
routines of :mod:`lakematrix.physics.tsolver`.

The public entry point is :func:`t_diff_ice`::

    t_diff_ice(surf, Tsurf, dt, snow, ice, water, deepice, isoilcol, ifflux,
               state) -> dict

``state`` is a plain ``dict`` whose array entries carry a **leading lake
axis** ``L``; the 1-based padding of the NumPy reference is kept, so
``Ti1[..., j]`` is Fortran ``Ti1(j)`` and ``rad_integr_ice[..., j]`` is
Fortran ``RadIce%integr(j)`` (0-based bound, as in the reference).  The
returned dict is a new ``dict`` (the input is not mutated): every input key is
echoed and ``Tw2``/``Ti2``/``Tis2``/``Tsoil2``/``Tsn`` hold the values the
NumPy reference would have written into ``IceState`` - arrays the branch does
not touch keep their input value.

Supported branches
==================

``ice == 1`` for every lane (case 2 and case 3 of ``lake.f90``); the
snow-on-soil / pure-soil / water-only branches of ``T_DIFF`` are *not* ported
(they belong to the other surface types).  Every combination of

* ``surf`` 1 (Dirichlet ``Tsurf``) and 0 (melting point at the ice-water
  interface; ``water == 0`` returns the state unchanged, as the Fortran
  ``RETURN``),
* ``snow`` 0/1,
* ``water`` 0/1,
* ``deepice`` 0/1 (only read on the ``surf == 0`` path),
* ``soilswitch`` 0/1,

is built and selected per lane with ``jnp.where``, so the flags may be
Python ints (fast path: unused systems are skipped) or ``(L,)`` integer
arrays (all systems are built).  ``nmeltpoint`` 0/1 is supported; 2 raises
(Jackett et al. 2004 is not ported, same as :func:`lakematrix.physics.soil.meltpnt`).
``isoilcol`` is a static Python int.

State keys and shapes
=====================

All entries are ``jax``/``numpy`` arrays; ``L`` is the batch size.  Items
marked "lake axis optional" accept a bare grid array with the Fortran
padding, broadcast to every lane (the metric arrays of one grid are shared):

that is ``(n,)`` instead of ``(L, n)``.

Grid metrics (1-based unless noted)
-----------------------------------
=====================  ==============  ==================================
``ddz``                (L, M+1)        ``ARRAYS_GRID%ddz``
``ddz05``              (L, M+1)        Fortran ``ddz05(0:M)``
``dzeta_int``          (L, M+2)        ``DZETA`` integral, 1-based + pad
``ddzi``               (L, Mice+1)
``ddzi05``             (L, Mice+1)     Fortran ``ddzi05(0:Mice)``
``dzetai_int``         (L, Mice+2)
=====================  ==============  ==================================

Bathymetry (the ``Bathym`` lists of ``IceState``, fields ``area_int`` /
``area_half``; index 0 is the unused pad)
--------------------------------------------------------------------
=====================  ==============  ==================================
``Aice`` / ``Adice``   (L, Mice+2)     ``bathymice``/``bathymdice``
``Ahice`` / ``Ahdice`` (L, Mice+1)
``Awat``               (L, M+2)        ``bathymwater``
``Ahwat``              (L, M+1)
=====================  ==============  ==================================

Ice (1-based)
-------------
=====================  ==============  ==================================
``Ti1`` / ``Ti2``      (L, Mice+2)     surface-ice temperature
``ci_m_roi_v``         (L, Mice+2)     ``ci_m_roi_v(1:Mice+1)``
``lami_v``             (L, Mice+1)     ``lami_v(1:Mice)``
``rad_integr_ice``     (L, Mice+1)     ``RadIce%integr(0:Mice)``
``lsh_ice``            (L, Mice+2)     ``lsh%ice``
``dhi`` / ``dhi0``     (L,)            surface-ice increments
``l1``                 (L,)            surface-ice thickness
=====================  ==============  ==================================

Deep ice (1-based)
------------------
=====================  ==============  ==================================
``Tis1`` / ``Tis2``    (L, Mice+2)     deep-ice temperature
``rad_integr_deepice`` (L, Mice+1)     ``RadDeepIce%integr(0:Mice)``
``lsh_dice``           (L, Mice+2)     ``lsh%dice``
``dls`` / ``dls0``     (L,)            deep-ice increments
``ls1``                (L,)            deep-ice thickness
=====================  ==============  ==================================

Water (1-based unless noted)
----------------------------
=====================  ==============  ==================================
``Tw1`` / ``Tw2``      (L, M+2)
``lamw``               (L, M+1)        ``lamw(0:M)``
``PEMF``               (L, M+2)
``pt_down_f``          (L, M+2)
``Sal2``               (L, M+2)        salinity for ``MELTPNT``
``preswat``            (L, M+2)        pressure for ``MELTPNT``
``rad_integr_water``   (L, M+1)        ``RadWater%integr(0:M)``
``lsh_water``          (L, M+2)        ``lsh%water``
``h1``                 (L,)            water thickness
``dhw`` / ``dhw0``     (L,)            water increments
=====================  ==============  ==================================

Snow (``SNOWSOIL_MOD``, 1-based; ``ms`` is the snow-ice node)
-----------------------------------------------------------
=====================  ==============  ==================================
``T``                  (L, >= ms+1)    old snow temperature
``Tsn``                (L, ms+1)       SNOWTEMP temperature (output)
``cs``                 (L, ms+1)       heat capacity
``dens``               (L, ms+1)       density
``dz``                 (L, ms+1)       layer thickness
``lams``               (L, ms+1)       heat conductance
``itop``               (L,) int        top snow node (required only when
                                       ``snow == 1`` is possible)
``SR_botsnow``         (L,)            shortwave at the snow bottom
``snmeltice``          (L,)            frozen snowmelt (``lake.f90`` case 2)
``dhiimp``             (L,)            ice impoundment
=====================  ==============  ==================================

Soil (1-based, ``ic = isoilcol``)
---------------------------------
=====================  ==============  ==================================
``Tsoil1``             (L, ns+1, ncols+1)
``Tsoil2``             (L, ns+1)
``lamsoil``            (L, ns+1)
``csoil``              (L, ns+1)
``rosoil``             (L, ns+1)
``dzs``                (L, ns+1)
``soilbotflx``         (L,)            ``soilbotflx%par``
=====================  ==============  ==================================

Switches
--------
=====================  ==============  ==================================
``soilswitch``         (L,) int/0-d    default 1 (``SOILSWITCH_ON``)
``nmeltpoint``         int (static)    default 1; only 0/1 supported
=====================  ==============  ==================================

Optional keys: ``itop``/``soilswitch`` default as shown; every other key is
only read by a branch that can be selected, but the module docstring above
lists the keys used when ``ice == 1`` (all of them are required for a full
case-2/3 batch).

Sizes ``M``, ``Mice``, ``ns`` and ``ms`` are derived from the array shapes
(``Tw1``, ``Ti1``, ``Tsoil1``, ``lams``); the ``PROGONKA`` work length is
``nn = max(ms + Mice + ns - 1, M + ns)``, i.e. the reference ``nveclen``
without its unused last slot.

Implementation notes
====================

Every branch is solved on a fixed row layout with *identity* rows
(``a = b = d = 0``, ``c = 1``) in all slots it does not use.  An identity row
decouples from its neighbours in the ``PROGONKA`` recurrence, so solving the
padded system is the same computation as solving the active sub-system of the
Fortran (and :func:`lakematrix.physics.ice._progonka`), which keeps the
transcription auditable: only one layout per ``surf``/``snow`` combination is
needed and the per-lane flags select between the boundary rows.

The source's own quirks are kept: the snow-ice no-soil interface mixes
``ci_m_roi`` (pure ice) into a ``ci_m_roi_v`` expression, and the
water + deep ice no-soil interface does **not** scale
``RadDeepIce%integr(Mice)`` by ``xx`` (``t_solver_mod.f90:327`` and
``635-645``).
"""

import numpy as np
import jax.numpy as jnp

from ..constants import C
from ..physics.ice import SOILSWITCH_OFF, SOILSWITCH_ON
from ..physics.soil import _MELTPNT_DTDC
from ..physics.tridiag import progonka_jax

__all__ = ["t_diff_ice", "STATE_KEYS"]

#: keys read from a full case-2/3 ``state``; every input key is echoed
STATE_KEYS = (
    "ddz", "ddz05", "dzeta_int", "ddzi", "ddzi05", "dzetai_int",
    "Aice", "Ahice", "Awat", "Ahwat", "Adice", "Ahdice",
    "Ti1", "Ti2", "ci_m_roi_v", "lami_v", "rad_integr_ice", "lsh_ice",
    "Tis1", "Tis2", "rad_integr_deepice", "lsh_dice",
    "Tw1", "Tw2", "lamw", "PEMF", "pt_down_f", "Sal2", "preswat",
    "rad_integr_water", "lsh_water",
    "T", "Tsn", "cs", "dens", "dz", "lams", "itop",
    "Tsoil1", "Tsoil2", "lamsoil", "csoil", "rosoil", "dzs",
    "h1", "l1", "ls1", "dhi", "dhi0", "dls", "dls0", "dhw", "dhw0",
    "snmeltice", "dhiimp", "soilbotflx", "SR_botsnow", "soilswitch",
)


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------

def _lane(state, key, tail, L):
    """``state[key]`` as ``(L,) + tail`` float64.

    A bare ``len(tail)``-dimensional array with the same trailing shape is
    broadcast over the lake axis.
    """
    if key not in state:
        raise KeyError(f"state is missing the required key {key!r}")
    a = jnp.asarray(state[key], dtype=jnp.float64)
    if a.ndim == len(tail):
        if a.shape != tail:
            raise ValueError(f"state[{key!r}]: expected shape {tail}, "
                             f"got {a.shape}")
        return jnp.broadcast_to(a, (L,) + tail)
    if a.ndim == len(tail) + 1:
        if a.shape != (L,) + tail:
            raise ValueError(f"state[{key!r}]: expected shape {(L,)+tail}, "
                             f"got {a.shape}")
        return a
    raise ValueError(f"state[{key!r}]: expected a {tail} grid array or a "
                     f"{(L,)+tail} lake array, got {a.shape}")


def _lane_atleast(state, key, tail, L):
    """Like :func:`_lane` but a longer trailing axis is sliced to ``tail``.

    The reference ``Bathym`` lists and the ``Rad*%integr`` arrays are padded
    past their last read node (``bathym_levels`` returns ``n_levels + 2``
    entries), so the converter may pass any trailing length ``>= tail``.
    """
    n = len(tail)
    if key not in state:
        raise KeyError(f"state is missing the required key {key!r}")
    a = jnp.asarray(state[key], dtype=jnp.float64)
    if a.ndim not in (n, n + 1):
        raise ValueError(f"state[{key!r}]: expected a {n}-D array, "
                         f"got {a.shape}")
    if a.ndim == n + 1:
        if a.shape[0] != L:
            raise ValueError(f"state[{key!r}]: leading lake axis "
                             f"{a.shape[0]} != L={L}")
        a = a[..., :tail[0]]
        if a.shape[1:] != tail:
            raise ValueError(f"state[{key!r}]: expected trailing {tail}, "
                             f"got {a.shape[1:]}")
        return a
    a = a[..., :tail[0]]
    if a.shape != tail:
        raise ValueError(f"state[{key!r}]: expected trailing {tail}, "
                         f"got {a.shape}")
    return jnp.broadcast_to(a, (L,) + tail)


def _snow_temp(state, L, ms):
    """``T`` with at least ``ms+1`` trailing entries (the reference's ML+1)."""
    if "T" not in state:
        raise KeyError("state is missing the required key 'T'")
    a = jnp.asarray(state["T"], dtype=jnp.float64)
    if a.ndim not in (1, 2):
        raise ValueError(f"state['T']: expected (n,) or (L, n), got {a.shape}")
    if a.shape[-1] < ms + 1:
        raise ValueError(f"state['T']: needs at least {ms+1} entries, "
                         f"got {a.shape[-1]}")
    a = a[..., :ms + 1]
    if a.ndim == 1:
        return jnp.broadcast_to(a, (L, ms + 1))
    if a.shape[0] != L:
        raise ValueError(f"state['T']: leading lake axis {a.shape[0]} != {L}")
    return a


def _lane_int(state, key, L, default=None):
    """``state[key]`` as an ``(L,)`` int32 array (scalar broadcast)."""
    if key not in state:
        if default is None:
            raise KeyError(f"state is missing the required key {key!r}")
        return jnp.broadcast_to(jnp.asarray(default, dtype=jnp.int32), (L,))
    a = jnp.asarray(state[key])
    if a.ndim == 0:
        return jnp.broadcast_to(a.astype(jnp.int32), (L,))
    if a.shape != (L,):
        raise ValueError(f"state[{key!r}]: expected a scalar or ({L},), "
                         f"got {a.shape}")
    return a.astype(jnp.int32)


def _scal(x, L, name):
    """Python/array scalar as an ``(L,)`` float array."""
    a = jnp.asarray(x, dtype=jnp.float64)
    if a.ndim == 0:
        return jnp.broadcast_to(a, (L,))
    if a.shape != (L,):
        raise ValueError(f"{name}: expected a scalar or ({L},), got {a.shape}")
    return a


def _flag(x, L, name):
    """``(L,)`` int32 flag from a Python int, 0-d array or ``(L,)`` array."""
    a = jnp.asarray(x)
    if a.ndim == 0:
        return jnp.broadcast_to(a.astype(jnp.int32), (L,))
    if a.shape != (L,):
        raise ValueError(f"{name}: expected a scalar or ({L},), got {a.shape}")
    return a.astype(jnp.int32)


def _concrete_int(x):
    """The Python value of a concrete scalar flag, else ``None``."""
    if isinstance(x, (bool, int, np.integer)):
        return int(x)
    if isinstance(x, np.ndarray) and x.shape == ():
        return int(x)
    return None


def _meltpnt(sal, npar):
    """Vectorised ``MELTPNT`` for ``npar`` 0/1 (``phys_func.f90:778-798``)."""
    if _concrete_int(npar) == 2:
        raise NotImplementedError(
            "MELTPNT: nmeltpoint=2 (Jackett et al. 2004) is not ported")
    n = jnp.asarray(npar)
    return jnp.where(n == 0, 0.0, 0.0 - sal * _MELTPNT_DTDC)


# ---------------------------------------------------------------------------
# tridiagonal helpers
# ---------------------------------------------------------------------------

def _identity(L, nn):
    """Identity-padded ``(L, nn+1)`` coefficient arrays (a, b, c, d)."""
    z = jnp.zeros((L, nn + 1))
    return z, z, jnp.ones((L, nn + 1)), z


def _put(a, b, c, d, rows, ai, bi, ci, di, mask=None, per_lane=False):
    """Set ``a/b/c/d`` at Fortran ``rows``; masked-out lanes stay identity.

    ``rows`` is either a (possibly scalar) row index shared by every lane, or
    a ``(L,)`` per-lane index vector when ``per_lane`` is true.
    """
    if mask is not None:
        ai = jnp.where(mask, ai, jnp.zeros_like(ai))
        bi = jnp.where(mask, bi, jnp.zeros_like(bi))
        ci = jnp.where(mask, ci, jnp.ones_like(ci))
        di = jnp.where(mask, di, jnp.zeros_like(di))
    if per_lane:
        lane = jnp.arange(a.shape[0])
        return (a.at[lane, rows].set(ai), b.at[lane, rows].set(bi),
                c.at[lane, rows].set(ci), d.at[lane, rows].set(di))
    return (a.at[..., rows].set(ai), b.at[..., rows].set(bi),
            c.at[..., rows].set(ci), d.at[..., rows].set(di))


def _solve(a, b, c, d, L, nn):
    """``PROGONKA`` on rows 1..nn; returns the padded ``(L, nn+1)`` solution."""
    sol = progonka_jax(a[:, 1:nn + 1], b[:, 1:nn + 1],
                       c[:, 1:nn + 1], d[:, 1:nn + 1])
    return jnp.concatenate((jnp.zeros((L, 1)), sol), axis=-1)


# ---------------------------------------------------------------------------
# DIFF_COEF cases (batched over the lake axis)
# ---------------------------------------------------------------------------

def _coef_ice(st, dt, j):
    """Batched ``DIFF_COEF`` ``ice_indic``; ``j`` is the ice node vector."""
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    l1 = st["l1"][:, None]
    lami_v, cim = st["lami_v"], st["ci_m_roi_v"]
    A, Ah = st["Aice"], st["Ahice"]
    ddzi, ddzi05, dzeta = st["ddzi"], st["ddzi05"], st["dzetai_int"]
    dhi, dhi0 = st["dhi"][:, None], st["dhi0"][:, None]
    Ti1, rad, lsh = st["Ti1"], st["rad_integr_ice"], st["lsh_ice"]
    ai = 0.5 * (-lami_v[:, j - 1] * Ah[:, j - 1]
                / (ddzi[:, j - 1] * l1 * A[:, j])
                + cim[:, j] * dhi * dzeta[:, j] * dt_inv05
                - cim[:, j] * dhi0 * dt_inv05)
    bi = 0.5 * (-lami_v[:, j] * Ah[:, j]
                / (ddzi[:, j] * l1 * A[:, j])
                - cim[:, j] * dhi * dzeta[:, j] * dt_inv05
                + cim[:, j] * dhi0 * dt_inv05)
    cinv = ai + bi - dt_inv * ddzi05[:, j - 1] * l1 * cim[:, j]
    dinv = (-Ti1[:, j] * dt_inv * ddzi05[:, j - 1] * l1 * cim[:, j]
            + rad[:, j] - rad[:, j - 1]
            + ai * Ti1[:, j - 1] + bi * Ti1[:, j + 1]
            - (ai + bi) * Ti1[:, j]
            - lsh[:, j] * ddzi05[:, j - 1] * l1)
    return ai, bi, cinv, dinv


def _coef_deepice(st, dt, j):
    """Batched ``DIFF_COEF`` ``deepice_indic`` (pure ice constants)."""
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    lami, cim = C.lami, C.ci_m_roi
    ls1 = st["ls1"][:, None]
    A, Ah = st["Adice"], st["Ahdice"]
    ddzi, ddzi05, dzeta = st["ddzi"], st["ddzi05"], st["dzetai_int"]
    dls, dls0 = st["dls"][:, None], st["dls0"][:, None]
    Tis1, rad, lsh = st["Tis1"], st["rad_integr_deepice"], st["lsh_dice"]
    ai = 0.5 * (-lami * Ah[:, j - 1] / (ddzi[:, j - 1] * ls1 * A[:, j])
                + cim * dls * dzeta[:, j] * dt_inv05
                - cim * dls0 * dt_inv05)
    bi = 0.5 * (-lami * Ah[:, j] / (ddzi[:, j] * ls1 * A[:, j])
                - cim * dls * dzeta[:, j] * dt_inv05
                + cim * dls0 * dt_inv05)
    cinv = ai + bi - dt_inv * ddzi05[:, j - 1] * ls1 * cim
    dinv = (-Tis1[:, j] * dt_inv * ddzi05[:, j - 1] * ls1 * cim
            + rad[:, j] - rad[:, j - 1]
            + ai * Tis1[:, j - 1] + bi * Tis1[:, j + 1]
            - (ai + bi) * Tis1[:, j]
            - lsh[:, j] * ddzi05[:, j - 1] * ls1)
    return ai, bi, cinv, dinv


def _coef_snow(st, dt, j):
    """Batched ``DIFF_COEF`` ``snow_indic`` (``j`` is the snow node)."""
    dt_inv = 1.0 / dt
    lams, dz, cs, dens, T = (st["lams"], st["dz"], st["cs"], st["dens"],
                             st["T"])
    lam2 = 0.5 * (lams[:, j] + lams[:, j + 1])
    lam1 = 0.5 * (lams[:, j - 1] + lams[:, j])
    dzmean = 0.5 * (dz[:, j - 1] + dz[:, j])
    ai = -0.5 * lam1 / jnp.maximum(dz[:, j - 1], 1e-12)
    bi = -0.5 * lam2 / jnp.maximum(dz[:, j], 1e-12)
    cinv = ai + bi - dt_inv * (cs[:, j] * dens[:, j] * dzmean)
    dinv = (-T[:, j] * dt_inv * (cs[:, j] * dens[:, j] * dzmean)
            + ai * T[:, j - 1]
            - (ai + bi) * T[:, j]
            + bi * T[:, j + 1])
    return ai, bi, cinv, dinv


def _coef_soil(st, dt, j):
    """Batched ``DIFF_COEF`` ``soil_indic`` for column ``isoilcol``."""
    dt_inv = 1.0 / dt
    Tsoil1 = st["Tsoil1"][:, :, st["isoilcol"]]
    lamsoil, csoil, rosoil, dzs = (st["lamsoil"], st["csoil"], st["rosoil"],
                                   st["dzs"])
    lam2 = 0.5 * (lamsoil[:, j] + lamsoil[:, j + 1])
    lam1 = 0.5 * (lamsoil[:, j - 1] + lamsoil[:, j])
    dzmean = 0.5 * (dzs[:, j - 1] + dzs[:, j])
    ai = -0.5 * lam1 / dzs[:, j - 1]
    bi = -0.5 * lam2 / dzs[:, j]
    cinv = ai + bi - dt_inv * (csoil[:, j] * rosoil[:, j] * dzmean)
    dinv = (-Tsoil1[:, j] * dt_inv * (csoil[:, j] * rosoil[:, j] * dzmean)
            + ai * Tsoil1[:, j - 1]
            - (ai + bi) * Tsoil1[:, j]
            + bi * Tsoil1[:, j + 1])
    return ai, bi, cinv, dinv


def _coef_water(st, dt, j):
    """Batched ``DIFF_COEF`` ``water_indic`` (Fortran ``j`` 1-based)."""
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    cw = C.cw_m_row0
    Tw1, lamw = st["Tw1"], st["lamw"]
    Awat, Ahwat = st["Awat"], st["Ahwat"]
    ddz, dzeta, ddz05 = st["ddz"], st["dzeta_int"], st["ddz05"]
    h1 = st["h1"][:, None]
    dhw, dhw0 = st["dhw"][:, None], st["dhw0"][:, None]
    PEMF, pt_down = st["PEMF"], st["pt_down_f"]
    rad, lsh = st["rad_integr_water"], st["lsh_water"]
    aa = 0.5 * (-Ahwat[:, j - 1] * lamw[:, j - 1]
                / (ddz[:, j - 1] * h1 * Awat[:, j])
                + cw * dhw * dzeta[:, j] * dt_inv05
                - cw * dhw0 * dt_inv05)
    ai = aa + cw * 0.5 * PEMF[:, j - 1]
    bb = 0.5 * (-Ahwat[:, j] * lamw[:, j] / (ddz[:, j] * h1 * Awat[:, j])
                - cw * dhw * dzeta[:, j] * dt_inv05
                + cw * dhw0 * dt_inv05)
    bi = bb - cw * 0.5 * PEMF[:, j]
    cinv = (aa + bb - dt_inv * ddz05[:, j - 1] * h1 * cw
            + cw * 0.5 * (PEMF[:, j] - PEMF[:, j - 1]))
    dinv = (-Tw1[:, j] * dt_inv * ddz05[:, j - 1] * h1 * cw
            + (Ahwat[:, j] * rad[:, j] - Ahwat[:, j - 1] * rad[:, j - 1])
            / Awat[:, j]
            - cw * PEMF[:, j - 1] * pt_down[:, j - 1]
            + cw * PEMF[:, j] * pt_down[:, j]
            + aa * Tw1[:, j - 1] + bb * Tw1[:, j + 1]
            - (aa + bb) * Tw1[:, j]
            - lsh[:, j] * ddz05[:, j - 1] * h1)
    return ai, bi, cinv, dinv


# ---------------------------------------------------------------------------
# lower boundary rows (per-lane scalars, shape (L,))
# ---------------------------------------------------------------------------

def _put_boundary_ice(st, dt, a, b, c, d, row, melt_sal, snow_layout):
    """Ice lower boundary: water Dirichlet / ice-soil / ice-only.

    The ice-only row differs between the snow layout (``t_solver_mod.f90:
    323-336``) and the no-snow layout (``411-424``): the former mixes the
    pure-ice constant ``ci_m_roi`` into the ``(dhi-dhi0)`` term and multiplies
    ``0.5*dt_inv`` in a different order.  Both are reproduced literally.
    """
    dt_inv = 1.0 / dt
    Mice, l1 = st["Mice"], st["l1"]
    lami_v, cim = st["lami_v"], st["ci_m_roi_v"]
    ddzi = st["ddzi"]
    A, Ah = st["Aice"], st["Ahice"]
    Ti1, rad = st["Ti1"], st["rad_integr_ice"]
    Tsoil1 = st["Tsoil1"][:, :, st["isoilcol"]]
    lamsoil, csoil, rosoil, dzs = (st["lamsoil"], st["csoil"], st["rosoil"],
                                   st["dzs"])
    w = st["water"] == 1
    son = st["soilswitch"] == SOILSWITCH_ON

    # ice-soil interface (t_solver_mod.f90:303-313 / 392-401)
    xx_on = A[:, Mice + 1] / Ah[:, Mice]
    a_on = 0.5 * (-lami_v[:, Mice] / (ddzi[:, Mice] * l1)
                  + xx_on * 0.5 * cim[:, Mice + 1]
                  * (st["dhi"] - st["dhi0"]) * dt_inv)
    b_on = -0.25 * (lamsoil[:, 1] + lamsoil[:, 2]) / dzs[:, 1]
    cc_on = 0.5 * dt_inv * (csoil[:, 1] * rosoil[:, 1] * dzs[:, 1]
                            + xx_on * cim[:, Mice + 1] * ddzi[:, Mice] * l1)
    c_on = a_on + b_on - cc_on
    d_on = (-Ti1[:, Mice + 1] * cc_on
            - rad[:, Mice]
            + a_on * Ti1[:, Mice]
            - (a_on + b_on) * Ti1[:, Mice + 1]
            + b_on * Tsoil1[:, 2])

    # ice only, no water and no soil (t_solver_mod.f90:323-336 / 411-424)
    xx_off = Ah[:, Mice] / A[:, Mice + 1]
    a_off = 0.5 * (-xx_off * lami_v[:, Mice] / (ddzi[:, Mice] * l1)
                   + 0.5 * cim[:, Mice + 1]
                   * (st["dhi"] - st["dhi0"]) * dt_inv)
    if snow_layout:
        c_off = (-xx_off * 0.5 * lami_v[:, Mice] / (ddzi[:, Mice] * l1)
                 - 0.5 * cim[:, Mice + 1] * ddzi[:, Mice] * l1 * dt_inv
                 + 0.25 * C.ci_m_roi * (st["dhi"] - st["dhi0"]) * dt_inv)
        d_off = (-0.5 * Ti1[:, Mice + 1] * cim[:, Mice + 1]
                 * ddzi[:, Mice] * l1 * dt_inv
                 - xx_off * rad[:, Mice]
                 + st["soilbotflx"]
                 + (xx_off * 0.5 * lami_v[:, Mice] / (ddzi[:, Mice] * l1)
                    - 0.25 * cim[:, Mice + 1]
                    * (st["dhi"] - st["dhi0"]) * dt_inv) * Ti1[:, Mice + 1]
                 + a_off * Ti1[:, Mice])
    else:
        c_off = (-xx_off * 0.5 * lami_v[:, Mice] / (ddzi[:, Mice] * l1)
                 + 0.5 * dt_inv * cim[:, Mice + 1]
                 * (-ddzi[:, Mice] * l1
                    + 0.5 * (st["dhi"] - st["dhi0"])))
        d_off = (-Ti1[:, Mice + 1] * 0.5 * dt_inv * cim[:, Mice + 1]
                 * ddzi[:, Mice] * l1
                 - xx_off * rad[:, Mice]
                 + st["soilbotflx"]
                 + (xx_off * 0.5 * lami_v[:, Mice] / (ddzi[:, Mice] * l1)
                    - 0.25 * dt_inv * cim[:, Mice + 1]
                    * (st["dhi"] - st["dhi0"])) * Ti1[:, Mice + 1]
                 + a_off * Ti1[:, Mice])

    mlt = _meltpnt(melt_sal, st["nmeltpoint"])
    av = jnp.where(w, 0.0, jnp.where(son, a_on, a_off))
    bv = jnp.where(w, 0.0, jnp.where(son, b_on, 0.0))
    cv = jnp.where(w, 1.0, jnp.where(son, c_on, c_off))
    dv = jnp.where(w, mlt, jnp.where(son, d_on, d_off))
    return _put(a, b, c, d, row, av, bv, cv, dv)


def _put_boundary_water(st, dt, a, b, c, d, row):
    """Water lower boundary: deep-ice Dirichlet / water-soil / water-only."""
    dt_inv = 1.0 / dt
    M = st["M"]
    cw = C.cw_m_row0
    Tw1, lamw = st["Tw1"], st["lamw"]
    Awat, Ahwat = st["Awat"], st["Ahwat"]
    ddz, h1 = st["ddz"], st["h1"]
    PEMF, pt_down = st["PEMF"], st["pt_down_f"]
    rad = st["rad_integr_water"]
    Tsoil1 = st["Tsoil1"][:, :, st["isoilcol"]]
    lamsoil, csoil, rosoil, dzs = (st["lamsoil"], st["csoil"], st["rosoil"],
                                   st["dzs"])
    d1 = st["deepice"] == 1
    son = st["soilswitch"] == SOILSWITCH_ON

    # water-soil interface (t_solver_mod.f90:446-455 / 573-582)
    xx_on = Awat[:, M + 1] / Ahwat[:, M]
    a_on = (0.5 * (-lamw[:, M] / (ddz[:, M] * h1)
                   + xx_on * 0.5 * dt_inv * cw * (st["dhw"] - st["dhw0"]))
            + xx_on * 0.5 * cw * PEMF[:, M])
    b_on = -0.25 * (lamsoil[:, 1] + lamsoil[:, 2]) / dzs[:, 1]
    cc_on = 0.5 * dt_inv * (csoil[:, 1] * rosoil[:, 1] * dzs[:, 1]
                            + xx_on * cw * ddz[:, M] * h1)
    c_on = a_on + b_on - cc_on - xx_on * cw * PEMF[:, M]
    d_on = (-Tw1[:, M + 1] * cc_on
            - rad[:, M]
            - xx_on * cw * PEMF[:, M] * pt_down[:, M]
            + a_on * Tw1[:, M]
            - (a_on + b_on) * Tw1[:, M + 1]
            + b_on * Tsoil1[:, 2])

    # water only, no soil and no deep ice (t_solver_mod.f90:466-477 / 592-603)
    xx_off = Ahwat[:, M] / Awat[:, M + 1]
    a_off = (0.5 * (-xx_off * lamw[:, M] / (ddz[:, M] * h1)
                    + 0.5 * dt_inv * cw * (st["dhw"] - st["dhw0"]))
             + 0.5 * cw * PEMF[:, M])
    c_off = (a_off - 0.5 * dt_inv * cw * ddz[:, M] * h1
             - cw * PEMF[:, M])
    d_off = (-Tw1[:, M + 1] * 0.5 * dt_inv * cw * ddz[:, M] * h1
             - xx_off * rad[:, M]
             - cw * PEMF[:, M] * pt_down[:, M]
             + st["soilbotflx"]
             + 0.5 * (-xx_off * lamw[:, M] / (ddz[:, M] * h1)
                      + 0.5 * dt_inv * cw * (st["dhw"] - st["dhw0"]))
             * (Tw1[:, M] - Tw1[:, M + 1]))

    mlt = _meltpnt(st["Sal2"][:, M + 1], st["nmeltpoint"])
    av = jnp.where(d1, 0.0, jnp.where(son, a_on, a_off))
    bv = jnp.where(d1, 0.0, jnp.where(son, b_on, 0.0))
    cv = jnp.where(d1, 1.0, jnp.where(son, c_on, c_off))
    dv = jnp.where(d1, mlt, jnp.where(son, d_on, d_off))
    return _put(a, b, c, d, row, av, bv, cv, dv)


def _put_soil_bottom(st, a, b, c, d, row, mask):
    """Soil bottom row (``a = c = 1``), written only for the selected lanes."""
    ns = st["ns"]
    lamsoil, dzs = st["lamsoil"], st["dzs"]
    di = (-st["soilbotflx"] * dzs[:, ns - 1]
          / (lamsoil[:, ns - 1] + lamsoil[:, ns]))
    return _put(a, b, c, d, row, 1.0, 0.0, 1.0, di, mask)


# ---------------------------------------------------------------------------
# systems (identity padded; one layout per surf/snow combination)
# ---------------------------------------------------------------------------

def _system_snow(st, dt, Tsurf, L, nn):
    """``surf == 1`` with snow (rows ``itop .. ms+Mice+ns-1``)."""
    dt_inv = 1.0 / dt
    ms, Mice, ns = st["ms"], st["Mice"], st["ns"]
    itop, l1 = st["itop"], st["l1"]
    a, b, c, d = _identity(L, nn)

    # top Dirichlet node at the per-lane itop (the snow-ice interface below
    # overwrites it when itop == ms, exactly like the reference)
    # snow layers itop+1 .. ms-1 (node == row); the masked-out rows are
    # written as identity, so the per-lane Dirichlet node below is set after
    # this put (its row itop is never a snow-layer row)
    rows = jnp.arange(1, ms)
    ai, bi, ci, di = _coef_snow(st, dt, rows)
    mask = rows[None, :] > itop[:, None]
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di, mask)

    # top Dirichlet node at the per-lane itop (the snow-ice interface below
    # overwrites it when itop == ms, exactly like the reference)
    a, b, c, d = _put(a, b, c, d, itop, 0.0, 0.0, 1.0, Tsurf,
                      per_lane=True)

    # snow-ice interface, row ms (t_solver_mod.f90:280-288)
    lami_v, cim = st["lami_v"], st["ci_m_roi_v"]
    ddzi, rad_ice = st["ddzi"], st["rad_integr_ice"]
    cs, dens, dz = st["cs"], st["dens"], st["dz"]
    Ti1 = st["Ti1"]
    y = (cs[:, ms] * dens[:, ms] * jnp.maximum(dz[:, ms], 1e-12)
         + cim[:, 1] * ddzi[:, 1] * l1)
    a_ms = (-0.25 * (st["lams"][:, ms - 1] + st["lams"][:, ms])
            / jnp.maximum(dz[:, ms - 1], 1e-12))
    b_ms = 0.5 * (-lami_v[:, 1] / (ddzi[:, 1] * l1)
                  + 0.5 * cim[:, 1] * st["dhi0"] * dt_inv)
    c_ms = a_ms + b_ms - 0.5 * dt_inv * y
    d_ms = (-Ti1[:, 1] * 0.5 * dt_inv * y
            - st["SR_botsnow"] + rad_ice[:, 1]
            + a_ms * st["T"][:, ms - 1]
            - (a_ms + b_ms) * Ti1[:, 1]
            + b_ms * Ti1[:, 2]
            - (st["snmeltice"] + st["dhiimp"]) * C.Lwi * C.row0 * dt_inv)
    a, b, c, d = _put(a, b, c, d, ms, a_ms, b_ms, c_ms, d_ms)

    # ice layers, rows ms+1 .. ms+Mice-1, node j = row-ms+1
    rows = jnp.arange(ms + 1, ms + Mice)
    ai, bi, ci, di = _coef_ice(st, dt, rows - ms + 1)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di)

    # boundary row ms+Mice: water Dirichlet / ice-soil / ice-only
    a, b, c, d = _put_boundary_ice(st, dt, a, b, c, d, ms + Mice,
                                   st["Sal2"][:, 1], snow_layout=True)

    # soil tail (water == 0 and soilswitch == 1)
    rows = jnp.arange(ms + Mice + 1, ms + Mice + ns - 1)
    ai, bi, ci, di = _coef_soil(st, dt, rows - ms - Mice + 1)
    mask = st["soil_tail_mask"][:, None] & jnp.ones(
        (1, rows.shape[0]), dtype=bool)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di, mask)
    a, b, c, d = _put_soil_bottom(st, a, b, c, d, ms + Mice + ns - 1,
                                  st["soil_tail_mask"])
    return a, b, c, d


def _system_nosnow(st, dt, Tsurf, L, nn):
    """``surf == 1`` without snow (rows ``1 .. Mice+ns``)."""
    Mice, ns = st["Mice"], st["ns"]
    a, b, c, d = _identity(L, nn)

    a, b, c, d = _put(a, b, c, d, 1, 0.0, 0.0, 1.0, Tsurf)

    rows = jnp.arange(2, Mice + 1)
    ai, bi, ci, di = _coef_ice(st, dt, rows)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di)

    a, b, c, d = _put_boundary_ice(st, dt, a, b, c, d, Mice + 1,
                                   st["Sal2"][:, 1], snow_layout=False)

    rows = jnp.arange(Mice + 2, Mice + ns)
    ai, bi, ci, di = _coef_soil(st, dt, rows - Mice)
    mask = st["soil_tail_mask"][:, None] & jnp.ones(
        (1, rows.shape[0]), dtype=bool)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di, mask)
    a, b, c, d = _put_soil_bottom(st, a, b, c, d, Mice + ns,
                                  st["soil_tail_mask"])
    return a, b, c, d


def _system_water(st, dt, L, nn):
    """``surf == 0`` water column of the ice + water branch (rows 1..M+ns)."""
    M, ns = st["M"], st["ns"]
    a, b, c, d = _identity(L, nn)
    mlt = _meltpnt(st["Sal2"][:, 1], st["nmeltpoint"])
    a, b, c, d = _put(a, b, c, d, 1, 0.0, 0.0, 1.0, mlt)

    rows = jnp.arange(2, M + 1)
    ai, bi, ci, di = _coef_water(st, dt, rows)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di)

    a, b, c, d = _put_boundary_water(st, dt, a, b, c, d, M + 1)

    rows = jnp.arange(M + 2, M + ns)
    ai, bi, ci, di = _coef_soil(st, dt, rows - M)
    mask = st["water_soil_mask"][:, None] & jnp.ones(
        (1, rows.shape[0]), dtype=bool)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di, mask)
    a, b, c, d = _put_soil_bottom(st, a, b, c, d, M + ns,
                                  st["water_soil_mask"])
    return a, b, c, d


def _system_deepice(st, dt, L, nn):
    """``surf == 0`` deep-ice column (rows ``1 .. Mice+ns``)."""
    dt_inv = 1.0 / dt
    Mice, ns = st["Mice"], st["ns"]
    a, b, c, d = _identity(L, nn)
    mlt = _meltpnt(st["Sal2"][:, st["M"] + 1], st["nmeltpoint"])
    a, b, c, d = _put(a, b, c, d, 1, 0.0, 0.0, 1.0, mlt)

    rows = jnp.arange(2, Mice + 1)
    ai, bi, ci, di = _coef_deepice(st, dt, rows)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di)

    # deep-ice lower boundary, row Mice+1
    lami, cim = C.lami, C.ci_m_roi
    ls1 = st["ls1"]
    ddzi = st["ddzi"]
    A, Ah = st["Adice"], st["Ahdice"]
    Tis1, rad = st["Tis1"], st["rad_integr_deepice"]
    Tsoil1 = st["Tsoil1"][:, :, st["isoilcol"]]
    lamsoil, csoil, rosoil, dzs = (st["lamsoil"], st["csoil"], st["rosoil"],
                                   st["dzs"])
    son = st["soilswitch"] == SOILSWITCH_ON
    d1 = st["deepice"] == 1

    # deepice-soil interface (t_solver_mod.f90:536-544 / 616-624)
    xx_on = A[:, Mice + 1] / Ah[:, Mice]
    a_on = 0.5 * (-lami / (ddzi[:, Mice] * ls1)
                  + xx_on * 0.5 * dt_inv * cim * (st["dls"] - st["dls0"]))
    b_on = -0.25 * (lamsoil[:, 1] + lamsoil[:, 2]) / dzs[:, 1]
    cc_on = 0.5 * dt_inv * (csoil[:, 1] * rosoil[:, 1] * dzs[:, 1]
                            + xx_on * cim * ddzi[:, Mice] * ls1)
    c_on = a_on + b_on - cc_on
    d_on = (-Tis1[:, Mice + 1] * cc_on
            - rad[:, Mice]
            + a_on * Tis1[:, Mice]
            - (a_on + b_on) * Tis1[:, Mice + 1]
            + b_on * Tsoil1[:, 2])

    # no soil (t_solver_mod.f90:635-645): RadDeepIce%integr(Mice) is *not*
    # scaled by xx in the source
    xx_off = Ah[:, Mice] / A[:, Mice + 1]
    c_off = (-xx_off * 0.5 * lami / (ddzi[:, Mice] * ls1)
             + 0.5 * dt_inv * cim
             * (-ddzi[:, Mice] * ls1 + 0.5 * (st["dls"] - st["dls0"])))
    a_off = (-xx_off * 0.5 * lami / (ddzi[:, Mice] * ls1)
             + 0.25 * dt_inv * cim * (st["dls"] - st["dls0"]))
    d_off = (-Tis1[:, Mice + 1] * cim * 0.5 * dt_inv * ddzi[:, Mice] * ls1
             - rad[:, Mice]
             + st["soilbotflx"]
             + (-xx_off * 0.5 * lami / (ddzi[:, Mice] * ls1)
                + 0.25 * dt_inv * cim * (st["dls"] - st["dls0"]))
             * (Tis1[:, Mice] - Tis1[:, Mice + 1]))
    av = jnp.where(son, a_on, a_off)
    bv = jnp.where(son, b_on, 0.0)
    cv = jnp.where(son, c_on, c_off)
    dv = jnp.where(son, d_on, d_off)
    a, b, c, d = _put(a, b, c, d, Mice + 1, av, bv, cv, dv)

    rows = jnp.arange(Mice + 2, Mice + ns)
    ai, bi, ci, di = _coef_soil(st, dt, rows - Mice)
    mask = (son & d1)[:, None] & jnp.ones((1, rows.shape[0]), dtype=bool)
    a, b, c, d = _put(a, b, c, d, rows, ai, bi, ci, di, mask)
    a, b, c, d = _put_soil_bottom(st, a, b, c, d, Mice + ns, son & d1)
    return a, b, c, d


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def t_diff_ice(surf, Tsurf, dt, snow, ice, water, deepice, isoilcol, ifflux,
               state):
    """Batched ``T_DIFF`` for the ice/snow branches (cases 2 and 3).

    See the module docstring for the ``state`` layout.  ``surf``/``snow``/
    ``ice``/``water``/``deepice`` may be Python ints or ``(L,)`` integer
    arrays; ``Tsurf`` a scalar or ``(L,)``.  ``isoilcol`` is a static int and
    ``ifflux`` is only used by the (unported) soil-only branch, so it is
    ignored.  Returns a new state dict.
    """
    Tw1 = jnp.asarray(state["Tw1"], dtype=jnp.float64)
    Ti1 = jnp.asarray(state["Ti1"], dtype=jnp.float64)
    Tsoil1 = jnp.asarray(state["Tsoil1"], dtype=jnp.float64)
    lams = jnp.asarray(state["lams"], dtype=jnp.float64)
    if Tw1.ndim != 2 or Ti1.ndim != 2 or Tsoil1.ndim != 3 or lams.ndim != 2:
        raise ValueError(
            "state arrays need a leading lake axis: Tw1 (L, M+2), "
            "Ti1 (L, Mice+2), Tsoil1 (L, ns+1, ncols+1), lams (L, ms+1)")
    L = Tw1.shape[0]
    M = Tw1.shape[-1] - 2
    Mice = Ti1.shape[-1] - 2
    ns = Tsoil1.shape[-2] - 1
    ms = lams.shape[-1] - 1
    if M < 1 or Mice < 1 or ns < 2 or ms < 2:
        raise ValueError(
            f"unsupported sizes: M={M}, Mice={Mice}, ns={ns}, ms={ms}")
    ic = int(isoilcol)
    if not 1 <= ic <= Tsoil1.shape[-1] - 1:
        raise ValueError(f"isoilcol={ic} outside 1..{Tsoil1.shape[-1]-1}")

    snow_l = _flag(snow, L, "snow")
    ice_l = _flag(ice, L, "ice")
    water_l = _flag(water, L, "water")
    deepice_l = _flag(deepice, L, "deepice")
    surf_l = _flag(surf, L, "surf")
    Tsurf_l = _scal(Tsurf, L, "Tsurf")

    c_ice = _concrete_int(ice)
    if c_ice is not None and c_ice == 0:
        raise NotImplementedError(
            "jaxbatch.ice_tdiff only ports the ice branches (ice=1); the "
            "snow-soil / soil-only / water-only branches of T_DIFF are not "
            "ported")

    st = {
        "M": M, "Mice": Mice, "ns": ns, "ms": ms, "isoilcol": ic,
        "nmeltpoint": state.get("nmeltpoint", 1),
        "snow": snow_l, "ice": ice_l, "water": water_l,
        "deepice": deepice_l, "surf": surf_l,
        "soilswitch": _lane_int(state, "soilswitch", L, SOILSWITCH_ON),
        "itop": _lane_int(state, "itop", L, 1),
    }
    for key, tail in (
            ("ddz", (M + 1,)), ("ddz05", (M + 1,)), ("dzeta_int", (M + 2,)),
            ("ddzi", (Mice + 1,)), ("ddzi05", (Mice + 1,)),
            ("dzetai_int", (Mice + 2,)),
            ("Ti1", (Mice + 2,)), ("ci_m_roi_v", (Mice + 2,)),
            ("lami_v", (Mice + 1,)), ("lsh_ice", (Mice + 2,)),
            ("Tis1", (Mice + 2,)), ("lsh_dice", (Mice + 2,)),
            ("Tw1", (M + 2,)), ("lamw", (M + 1,)), ("PEMF", (M + 2,)),
            ("pt_down_f", (M + 2,)), ("Sal2", (M + 2,)),
            ("preswat", (M + 2,)), ("lsh_water", (M + 2,)),
            ("cs", (ms + 1,)), ("dens", (ms + 1,)), ("dz", (ms + 1,)),
            ("lams", (ms + 1,)),
            ("Tsoil1", (ns + 1, Tsoil1.shape[-1])),
            ("lamsoil", (ns + 1,)), ("csoil", (ns + 1,)),
            ("rosoil", (ns + 1,)), ("dzs", (ns + 1,))):
        st[key] = _lane(state, key, tail, L)
    for key, tail in (
            ("Aice", (Mice + 2,)), ("Ahice", (Mice + 1,)),
            ("Awat", (M + 2,)), ("Ahwat", (M + 1,)),
            ("Adice", (Mice + 2,)), ("Ahdice", (Mice + 1,)),
            ("rad_integr_ice", (Mice + 1,)),
            ("rad_integr_deepice", (Mice + 1,)),
            ("rad_integr_water", (M + 1,))):
        st[key] = _lane_atleast(state, key, tail, L)
    st["T"] = _snow_temp(state, L, ms)
    for key in ("h1", "l1", "ls1", "dhi", "dhi0", "dls", "dls0", "dhw",
                "dhw0", "snmeltice", "dhiimp", "soilbotflx", "SR_botsnow"):
        st[key] = _scal(state[key], L, key)

    nn = max(ms + Mice + ns - 1, M + ns)

    s1 = surf_l == 1
    s0 = surf_l == 0
    sn = snow_l == 1
    w = water_l == 1
    dw = water_l == 0
    d1 = deepice_l == 1
    tgt = ice_l == 1
    son = st["soilswitch"] == SOILSWITCH_ON
    soff = st["soilswitch"] == SOILSWITCH_OFF
    st["soil_tail_mask"] = tgt & dw & son
    st["water_soil_mask"] = tgt & w & s0 & ~d1 & son

    # static short-circuit: skip systems that no lane can select
    c_surf = _concrete_int(surf)
    c_snow = _concrete_int(snow)
    use_snow = (c_snow in (None, 1) and c_surf in (None, 1)
                and c_ice in (None, 1))
    use_nosnow = (c_snow in (None, 0) and c_surf in (None, 1)
                  and c_ice in (None, 1))
    use_surf0 = c_surf in (None, 0) and c_ice in (None, 1)

    z = jnp.zeros((L, nn + 1))
    if use_snow:
        TA = _solve(*_system_snow(st, dt, Tsurf_l, L, nn), L, nn)
    else:
        TA = z
    if use_nosnow:
        TB = _solve(*_system_nosnow(st, dt, Tsurf_l, L, nn), L, nn)
    else:
        TB = z
    if use_surf0:
        TC = _solve(*_system_water(st, dt, L, nn), L, nn)
        TD = _solve(*_system_deepice(st, dt, L, nn), L, nn)
    else:
        TC = z
        TD = z

    # --- Ti2 -----------------------------------------------------------
    pad = jnp.zeros((L, 1))
    if use_snow:
        Ti2_A = jnp.concatenate((pad, TA[:, ms:ms + Mice + 1]), axis=-1)
    else:
        Ti2_A = jnp.zeros((L, Mice + 2))
    Ti2_B = TB[:, :Mice + 2]
    Ti2_in = jnp.asarray(state["Ti2"], dtype=jnp.float64)
    pick = jnp.where((s1 & sn & tgt)[:, None], Ti2_A, Ti2_B)
    Ti2 = jnp.where((s1 & tgt)[:, None], pick, Ti2_in)

    # --- Tw2 -----------------------------------------------------------
    Tw2_in = jnp.asarray(state["Tw2"], dtype=jnp.float64)
    Tw2 = jnp.where((s0 & w & tgt)[:, None], TC[:, :M + 2], Tw2_in)

    # --- Tis2 ----------------------------------------------------------
    Tis2_in = jnp.asarray(state["Tis2"], dtype=jnp.float64)
    Tis2 = jnp.where((s0 & w & d1 & tgt)[:, None], TD[:, :Mice + 2], Tis2_in)

    # --- Tsn -----------------------------------------------------------
    Tsn_in = jnp.asarray(state["Tsn"], dtype=jnp.float64)
    rows0 = jnp.arange(ms + 1)
    sel = s1 & sn & tgt
    snow_rows = ((rows0[None, :] >= st["itop"][:, None])
                 & (rows0[None, :] <= ms) & sel[:, None])
    Tsn = jnp.where(snow_rows, TA[:, :ms + 1], Tsn_in)

    # --- Tsoil2 --------------------------------------------------------
    Tsoil2_in = jnp.asarray(state["Tsoil2"], dtype=jnp.float64)
    m_A = s1 & sn & tgt & dw & son
    m_B = s1 & ~sn & tgt & dw & son
    m_C = s0 & w & tgt & ~d1 & son
    m_D = s0 & w & tgt & d1 & son
    m_off = tgt & ((s1 & dw & soff) | (s0 & w & soff))
    TA_soil = jnp.concatenate((pad, TA[:, ms + Mice:ms + Mice + ns]), axis=-1)
    TB_soil = jnp.concatenate((pad, TB[:, Mice + 1:Mice + ns + 1]), axis=-1)
    TC_soil = jnp.concatenate((pad, TC[:, M + 1:M + ns + 1]), axis=-1)
    TD_soil = jnp.concatenate((pad, TD[:, Mice + 1:Mice + ns + 1]), axis=-1)
    TSL = st["Tsoil1"][:, :, ic]
    Tsoil2 = Tsoil2_in
    Tsoil2 = jnp.where(m_off[:, None], TSL, Tsoil2)
    Tsoil2 = jnp.where(m_A[:, None], TA_soil, Tsoil2)
    Tsoil2 = jnp.where(m_B[:, None], TB_soil, Tsoil2)
    Tsoil2 = jnp.where(m_C[:, None], TC_soil, Tsoil2)
    Tsoil2 = jnp.where(m_D[:, None], TD_soil, Tsoil2)

    out = dict(state)
    out["Ti2"] = Ti2
    out["Tw2"] = Tw2
    out["Tis2"] = Tis2
    out["Tsoil2"] = Tsoil2
    out["Tsn"] = Tsn
    return out
