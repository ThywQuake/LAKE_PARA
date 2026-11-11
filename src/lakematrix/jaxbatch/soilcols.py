"""JAX-vectorised case-1 open-water **side** soil columns (``SOILCOLSTEMP``).

Fortran: ``SOILCOLSTEMP`` (``soil_mod.f90:804-1031``); NumPy reference:
:func:`lakematrix.physics.soil.soilcolstemp`, :func:`~.tsurfsoilcol`,
:func:`~.soil_cond_heat_coef`, ``_t_diff_soil``/``_diff_coef_soil`` and
``SOILFORLAKE``.  This is the B3/M5 port of the **case-1 open-water** path, i.e.
for every side column ``is = 1 .. nsoilcols-1``::

    SOIL_COND_HEAT_COEF(is)
    T_DIFF(surf=1, Tsoilsurf(is), dt, 0, 0, 0, 0, is, .false.)   -> Tsoil2
    soilflux(is) = ...
    SOILFORLAKE(nveclen, dt, a, b, c, d, is)

with ``h1 > small_value``, ``l1 = 0``, ``ls1 = 0`` (no ice, no deep ice), which
is the only branch the case-1 driver reaches.  The deepest column ``nsoilcols``
is *not* touched here; the driver does it after ``T_SOLVER`` (exactly like
``lake.f90``).  ``SOILFORLAKE`` itself is the already validated
:mod:`lakematrix.jaxbatch.soilforlake` kernel, and
``SOIL_COND_HEAT_COEF`` is :func:`lakematrix.jaxbatch.soil.soil_cond_heat_coef`.

Interface
=========

Two public kernels, the pass that wires them together, and a geometry packer
(``jitted_tsurfsoilcol_water`` / ``jitted_t_diff_soil`` are ``jax.jit``
aliases of the two kernels).  Every array has a leading lake axis ``L``; the
level/column axes are **1-based padded** where the NumPy reference is 1-based
(index 0 is the unused Fortran pad) and **0-based** where the reference gets
its arrays from ``grid``/``bathym`` (documented per key).

``tsurfsoilcol_water(Tw1, h1, zsoilcols, ddz, w_lx, w_ly, w_lx_half,
w_ly_half, s_lx, s_ly, M, ncols)`` -> ``(L, ncols+1)``
    Water-only branch of ``TSURFSOILCOL`` (``soil_mod.f90:1316-1362``), the
    mean temperature of each side soil column, area-weighted over the water
    layers it overlaps.  Arguments:

    ==============  ==================  =====================================
    argument        shape               meaning
    ==============  ==================  =====================================
    ``Tw1``         ``(L, M+2)``        1-based water profile (``Tw1[1:M+1]``);
                                       ``SoilState.Tw1``
    ``h1``          ``(L,)``            water depth, m (must be ``> small``)
    ``zsoilcols``   ``(L, ncols+1)``    0-based ``geom.zsoilcols[1:]``, the
                                       table depths of the column tops *and*
                                       of the deepest point (``zsoilcols(ncols+1)``)
    ``ddz``         ``(L, M+1)``        1-based ``geom.ddz`` (``grid.ddz`` padded)
    ``w_lx``,       ``(L, M+1)``        0-based water ``bathymwater(1:M+1)%Lx``
    ``w_ly``                           (``Bathym`` list, ``*_half`` for
                                       ``(L, M)`` ``bathymwater(1:M)``)
    ``w_lx_half``,  ``(L, M)``
    ``w_ly_half``
    ``s_lx``,       ``(L, ncols+1)``    0-based ``bathymsoil(1:ncols+1)%Lx``
    ``s_ly``
    ``M``, ``ncols``  ``int``           static sizes (``M`` water layers,
                                       ``ncols = nsoilcols``)
    ==============  ==================  =====================================

    1-D geometry arguments are accepted as well and broadcast over ``L``.
    Index 0 of the result is the Fortran pad; entries ``1 .. ncols-1`` are the
    side columns and entry ``ncols`` is the deepest column (left at 0.0, it is
    not computed by ``TSURFSOILCOL`` in the loop ``is = 1 .. ncols-1``).
    Use :func:`pack_geometry` to build the arrays from one lake's
    :class:`~lakematrix.physics.soil.SoilGeometry`.

``t_diff_soil(Tsoil1_col, lamsoil, csoil, rosoil, dzs, tsurs, dt, soilbotflx,
ns)`` -> ``(L, ns+1)``
    Soil-only branch of ``T_DIFF`` (``t_solver_mod.f90:517-536`` plus
    ``DIFF_COEF`` case ``soil_indic``, ``t_solver_mod.f90:699-717``) solved with
    :func:`lakematrix.physics.tridiag.progonka_jax`.  ``Tsoil1_col`` is
    ``Tsoil1[:, :, is_]`` (``(L, ns+1)``), ``lamsoil``/``csoil``/``rosoil`` are
    ``(L, ns+1)`` for the *same* column, ``dzs`` is the 1-D static soil grid
    ``(ns+1,)``, ``tsurs`` is ``Tsoilsurf(is)`` ``(L,)`` and ``soilbotflx`` the
    ``soilbotflx`` scalar.  Returns ``Tsoil2(1:ns)`` with the usual pad at 0
    (``ifflux = .false.``, the only call ``SOILCOLSTEMP`` makes).

``soilcolstemp_water(state, params, geom, dt)`` -> ``dict``
    The whole heat pass over the side columns (see :func:`soilcolstemp_water`).

``jitted_tsurfsoilcol_water`` / ``jitted_t_diff_soil``
    ``jax.jit`` aliases of the two pure sub-kernels above.

Static sizes (``ns``, ``ncols``, ``M``) are read from ``params``/shapes, so the
Python ``for`` loops over the columns and levels are unrolled at trace time
(``jnp.where``/masks replace the Fortran ``if``/``break`` searches).

``state`` dict (input, not mutated)
-----------------------------------
=============================  ====================  =========================
key                            shape                 meaning
=============================  ====================  =========================
``Tsoil1``, ``wl1``, ``wi1``   ``(L, ns+1, nc+1)``   prognostic soil fields
``Sals1``                      ``(L, ns+1, nc+1)``   soil salinity (read only)
``Tw1``                        ``(L, M+2)``          water profile (read only)
``h1``                         ``(L,)``              water depth (read only)
``soilflux``                   ``(L, nc+1)``         *optional* incoming flux
                                                     (index 0 / nc preserved)
other keys of :class:`~lakematrix.physics.soil.SoilState` are carried through
unchanged (``Tsoil3``, ``wl4``, ``wi2``, ``wa``, ``methgenmh``, ...).
=============================  ====================  =========================

``params`` dict — the union of the :mod:`~lakematrix.jaxbatch.soilforlake` and
:func:`~lakematrix.jaxbatch.soil.soil_cond_heat_coef` parameter dicts:
``ns``, ``nsoilcols``, ``POR``, ``rosdry``, ``dzs``, ``dzss``, ``zsoil``,
``BH``, ``DLMAX``, ``FLWMAX``, ``WLM0``, ``WLM7``, ``pressure``,
``soilbotflx``, ``nmeltpoint``, ``tricemethhydr`` (and optionally
``soilcolconjtype``, default ``2``).

``geom`` dict — the :func:`pack_geometry` keys (``zsoilcols``, ``ddz``,
``w_lx``, ``w_ly``, ``w_lx_half``, ``w_ly_half``, ``s_lx``, ``s_ly``), all with
the leading lake axis (except ``ddz`` which may be shared).

Returns
-------
A **new** dict: a copy of ``state`` with, for columns ``1 .. ncols-1``,

* ``Tsoil1``/``wl1``/``wi1`` — updated by ``SOILFORLAKE``;
* ``Tsoil2``/``Tsoil3``/``wl4``/``wi2``/``wa``/``methgenmh``/``wl2``/``wl3``
  — the work values of the *last* side column (the reference's module storage
  is shared by all columns, so only the last column's values survive);
* ``csoil``/``rosoil``/``lamsoil``/``lammoist``/``filtr``/``wsoil`` — the
  ``SOIL_COND_HEAT_COEF`` coefficients of the last side column.  This is the
  state ``lake.f90`` hands to the deepest-column path, which is why the order
  matters (see the NumPy module docstring);
* ``soilflux`` (``(L, nc+1)``) — entry ``is`` is the heat flux into side column
  ``is`` (W/m2); entries 0 and ``ncols`` keep their incoming value;
* ``dhwfsoil`` (``(L,)``) — ``dhwfsoil`` of the last side column;
* ``Tsoilsurf`` (``(L, nc+1)``) — diagnostic, the ``TSURFSOILCOL`` result.

Notes / deviations
==================

* Only the water-only branch is implemented (``h1 > small_value``,
  ``l1 = 0``, ``ls1 = 0``).  The ice / deep-ice / no-water branches of
  ``TSURFSOILCOL`` are **not** ported; the caller is responsible for the branch
  selection.  ``soilcolconjtype = 1`` (``Tsoilsurf = Tw1(icent)``) raises
  ``NotImplementedError``, like the methane pass of ``SOILCOLSTEMP``
  (``contr(2) = 1``) — neither is part of this port.
* The column / level searches of ``TSURFSOILCOL`` use the same floating-point
  comparisons as the reference (``z[j] <= zsoilcols[i]`` with ``z`` built in
  the same summation order).  This matters: for the reference geometry
  ``zsoilcols(ncols)`` sits *exactly* on the last water interface, so the
  ``j1`` decision is taken by a ``<=`` equality.
* The area sums are evaluated as masked reductions over the level axis.  The
  terms are the same and in the same order as the reference (top, interior
  ascending, bottom); only the reduction tree differs, which shows up at the
  ``1e-16`` relative level.
* ``tsurfsoilcol_water`` does not reproduce the Fortran warning branch
  ``j1 <= j0`` ("water does not flood all soil columns"); the reference raises
  ``RuntimeError`` there for an unset ``j1``.  The reference's own constraint
  is that every side-column boundary lies inside the water column
  (``zsoilcols(2) > z(0)``), which for ``summer60d`` means ``h1 > ~18.06 m``.

Not ported / gaps
=================

* The whole ``SOILCOLSTEMP`` pass is **not** ``jax.jit``-compiled in one go:
  the shared :func:`lakematrix.jaxbatch.soil.soil_cond_heat_coef` computes
  ``cr = float(jnp.float32(0.2)) * 4180.0`` (``jaxbatch/soil.py:34``), which
  raises ``ConcretizationTypeError`` under tracing.  Hoisting that literal to
  a module constant is the one-line fix; shared files are not touched by this
  port.  ``tsurfsoilcol_water`` and ``t_diff_soil`` are jitted on their own as
  :data:`jitted_tsurfsoilcol_water` / :data:`jitted_t_diff_soil`.
* ``l1 > 0`` / ``ls1 > 0`` (ice, deep ice) and the no-water fallback of
  ``TSURFSOILCOL``, ``soilcolconjtype = 1`` and the methane pass
  (``contr(2) = 1``) are out of scope, as documented above.
* ``TSURFSOILCOL``'s deep-column entry ``Tsoilsurf(ncols)`` is left at 0.0
  (the source loop stops at ``ncols-1``); the reference behaves the same.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C
from ..physics.tridiag import progonka_jax
from .soil import soil_cond_heat_coef
from .soilforlake import soilforlake

__all__ = [
    "jitted_t_diff_soil",
    "jitted_tsurfsoilcol_water",
    "pack_geometry",
    "soilcolstemp_water",
    "t_diff_soil",
    "tsurfsoilcol_water",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _batched(x):
    """Append a leading lake axis to a 1-D array; leave higher ranks alone."""
    x = jnp.asarray(x)
    return x[None, ...] if x.ndim == 1 else x


def _select(where, x):
    """One-hot gather over the last axis: ``sum(jnp.where(where, x, 0))``."""
    return jnp.sum(jnp.where(where, x, 0.0), axis=-1)


# ---------------------------------------------------------------------------
# TSURFSOILCOL (water-only branch)
# ---------------------------------------------------------------------------

def tsurfsoilcol_water(Tw1, h1, zsoilcols, ddz, w_lx, w_ly, w_lx_half,
                       w_ly_half, s_lx, s_ly, M, ncols):
    """``TSURFSOILCOL`` water-only branch; see the module docstring.

    Returns ``Tsoilsurf(0:ncols)`` (``(L, ncols+1)``, index 0 unused, entries
    ``1 .. ncols-1`` filled, the deepest entry left at 0.0).
    """
    M, ncols = int(M), int(ncols)
    Tw1 = _batched(Tw1)
    h1 = jnp.atleast_1d(jnp.asarray(h1))
    L = h1.shape[0]
    zsc = _batched(zsoilcols)
    ddz = _batched(ddz)
    w_lx = _batched(w_lx)
    w_ly = _batched(w_ly)
    w_lx_half = _batched(w_lx_half)
    w_ly_half = _batched(w_ly_half)
    s_lx = _batched(s_lx)
    s_ly = _batched(s_ly)

    # --- water-layer grid of the branch (soil_mod.f90:1316-1362) -----------
    # Temp(1:M+1) = Tw1(1:M+1); index 0 is the unused pad.
    temp = jnp.concatenate((jnp.zeros((L, 1)), Tw1[:, 1:M + 2]), axis=-1)

    # z(0) = 0, z(1) = 0.5*d dz(1)*h1, z(j) += 0.5*(ddz(j-1)+ddz(j))*h1,
    # z(M+1) += 0.5*ddz(M)*h1 -- the reference's summation order is kept.
    zl = [jnp.zeros((L,))]
    zl.append(0.5 * ddz[:, 1] * h1)
    for j in range(2, M + 1):
        zl.append(zl[-1] + 0.5 * (ddz[:, j - 1] + ddz[:, j]) * h1)
    zl.append(zl[-1] + 0.5 * ddz[:, M] * h1)
    z = jnp.stack(zl, axis=-1)
    # z(:) = z(:) + zsc(ncols+1) - h1
    z = z + zsc[:, ncols:ncols + 1] - h1[:, None]

    # aell(0) = 0.5*bw(1)%Lx, aell(1:M) = 0.5*bw(1:M)%Lx_half,
    # aell(M+1) = 0.5*bw(M+1)%Lx (same for bell).
    aell = jnp.concatenate((0.5 * w_lx[:, 0:1], 0.5 * w_lx_half,
                            0.5 * w_lx[:, M:M + 1]), axis=-1)
    bell = jnp.concatenate((0.5 * w_ly[:, 0:1], 0.5 * w_ly_half,
                            0.5 * w_ly[:, M:M + 1]), axis=-1)
    aellm1 = jnp.concatenate((jnp.zeros((L, 1)), aell[:, :-1]), axis=-1)
    bellm1 = jnp.concatenate((jnp.zeros((L, 1)), bell[:, :-1]), axis=-1)

    lev = jnp.arange(M + 2)[None, :]            # j = 0 .. M+1
    area_int = C.pi * (aellm1 * bellm1 - aell * bell)   # area of cell j

    Tsoilsurf = jnp.zeros((L, ncols + 1))
    for i in range(0, ncols - 1):
        # Fortran is = i+1: zsc(is), zsc(is+1) and bs(is), bs(is+1).
        zi = zsc[:, i]
        zi1 = zsc[:, i + 1]

        # j0: first j in 0..M with z(j) <= zsc(is) < z(j+1), i.e. j0 = j+1;
        # j0 = 1 when zsc(is) < z(0).  z is monotone, so the count of
        # {j in 0..M: z(j) <= zsc} is max{j}+1 = j0 (and 0 -> the j0 = 1 case).
        cnt0 = jnp.sum((z <= zi[:, None]) & (lev <= M), axis=-1)
        j0 = jnp.clip(cnt0.astype(jnp.int32), 1, M + 1)
        # j1: first j in j0..M with z(j) <= zsc(is+1) < z(j+1), j1 = j+1.
        cnt1 = jnp.sum((z <= zi1[:, None]) & (lev >= j0[:, None])
                       & (lev <= M), axis=-1)
        j1 = jnp.minimum(j0 + cnt1, M + 1)

        m_j0 = lev == j0[:, None]
        m_j0m1 = lev == (j0 - 1)[:, None]
        m_j1 = lev == j1[:, None]
        m_j1m1 = lev == (j1 - 1)[:, None]

        # Top numerical layer overlapping the soil column interval.
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
# T_DIFF (soil-only branch)
# ---------------------------------------------------------------------------

def t_diff_soil(Tsoil1_col, lamsoil, csoil, rosoil, dzs, tsurs, dt,
                soilbotflx, ns):
    """Soil-only ``T_DIFF`` (``DIFF_COEF`` case ``soil_indic``) -> ``(L, ns+1)``.

    ``Tsoil1_col``, ``lamsoil``, ``csoil``, ``rosoil`` are ``(L, ns+1)`` for the
    same soil column; ``dzs`` is the 1-D static soil grid; ``tsurs`` is the
    surface temperature ``(L,)``; ``dt`` in seconds.  The local ``a``..``d``
    work arrays are 1-based (index 0 unused) exactly like the reference, and
    ``ifflux = .false.`` is the only branch ``SOILCOLSTEMP`` uses.
    """
    ns = int(ns)
    Tcol = _batched(Tsoil1_col)
    L = Tcol.shape[0]
    lamsoil = _batched(lamsoil)
    csoil = _batched(csoil)
    rosoil = _batched(rosoil)
    dzs = jnp.asarray(dzs)
    tsurs = jnp.atleast_1d(jnp.asarray(tsurs))

    dt_inv = 1.0 / dt
    a = jnp.zeros((L, ns + 2))
    b = jnp.zeros((L, ns + 2))
    c = jnp.zeros((L, ns + 2))
    d = jnp.zeros((L, ns + 2))

    # Surface boundary: continuity with the water (ifflux = .false.).
    b = b.at[:, 1].set(0.0)
    c = c.at[:, 1].set(1.0)
    d = d.at[:, 1].set(tsurs)

    # DIFF_COEF: i = 2 .. ns-1 with j = i - m0 + n0 = i (m0 = n0 = 2).
    i = jnp.arange(2, ns)
    j = i
    lam2 = 0.5 * (lamsoil[:, j] + lamsoil[:, j + 1])
    lam1 = 0.5 * (lamsoil[:, j - 1] + lamsoil[:, j])
    dzmean = 0.5 * (dzs[j - 1] + dzs[j])
    ai = -0.5 * lam1 / dzs[j - 1]
    bi = -0.5 * lam2 / dzs[j]
    a = a.at[:, i].set(ai)
    b = b.at[:, i].set(bi)
    c = c.at[:, i].set(ai + bi - dt_inv * (csoil[:, j] * rosoil[:, j] * dzmean))
    d = d.at[:, i].set(
        -Tcol[:, j] * dt_inv * (csoil[:, j] * rosoil[:, j] * dzmean)
        + ai * Tcol[:, j - 1] - (ai + bi) * Tcol[:, j]
        + bi * Tcol[:, j + 1])

    # Bottom boundary: zero flux (soilbotflx).
    c = c.at[:, ns].set(0.5)
    a = a.at[:, ns].set(0.5)
    d = d.at[:, ns].set(
        -soilbotflx * dzs[ns - 1] / (lamsoil[:, ns - 1] + lamsoil[:, ns])
        + a[:, ns] * Tcol[:, ns - 1] - c[:, ns] * Tcol[:, ns])

    y = progonka_jax(a[:, 1:ns + 1], b[:, 1:ns + 1], c[:, 1:ns + 1],
                     d[:, 1:ns + 1])
    return jnp.concatenate((jnp.zeros((L, 1)), y), axis=-1)


# ---------------------------------------------------------------------------
# SOILCOLSTEMP (side columns, water-only branch)
# ---------------------------------------------------------------------------

def soilcolstemp_water(state, params, geom, dt):
    """Case-1 ``SOILCOLSTEMP`` heat pass over the side soil columns.

    ``state``/``params``/``geom`` are dicts, see the module docstring.  Returns
    a new state dict; ``state`` and its arrays are not mutated.
    """
    conj = int(params.get("soilcolconjtype", 2))
    if conj != 2:
        raise NotImplementedError(
            f"SOILCOLSTEMP: soilcolconjtype={conj} (Tsoilsurf = Tw1(icent)) is "
            "not ported; only the TSURFSOILCOL branch (2) is implemented")
    nmeltpoint = int(params.get("nmeltpoint", 1))
    tricemethhydr = float(params.get("tricemethhydr", 0.0))
    Tsoil1 = jnp.asarray(state["Tsoil1"])
    ns = Tsoil1.shape[-2] - 1
    ncols = Tsoil1.shape[-1] - 1
    M = jnp.asarray(geom["ddz"]).shape[-1] - 1
    return _soilcolstemp_water(state, params, geom, dt, ns, ncols, M,
                              nmeltpoint, tricemethhydr)


def _soilcolstemp_water(state, params, geom, dt, ns, ncols, M, nmeltpoint,
                        tricemethhydr):
    """Implementation with the static values passed explicitly (jit-safe)."""
    ns, ncols, M = int(ns), int(ncols), int(M)
    nmeltpoint, tricemethhydr = int(nmeltpoint), float(tricemethhydr)
    if nmeltpoint not in (0, 1):
        raise NotImplementedError(
            "MELTPNT: nmeltpoint=2 (Jackett et al. 2004, FP_THETA) is not "
            "ported")
    if tricemethhydr > 0.0:
        raise NotImplementedError(
            "tricemethhydr > 0 (methane hydrate dissociation) is not ported")
    # SOILFORLAKE reads ns/nsoilcols/nmeltpoint/tricemethhydr from its params
    # dict with int()/float(); keep Python scalars there so that call stays
    # traceable wherever the pass is called from a jitted context.
    p = dict(params)
    p.update(ns=ns, nsoilcols=ncols, nmeltpoint=nmeltpoint,
             tricemethhydr=tricemethhydr)

    Tsoil1 = jnp.asarray(state["Tsoil1"])
    wl1 = jnp.asarray(state["wl1"])
    wi1 = jnp.asarray(state["wi1"])
    Tw1 = jnp.asarray(state["Tw1"])
    h1 = jnp.atleast_1d(jnp.asarray(state["h1"]))
    L = Tsoil1.shape[0]

    POR = jnp.asarray(params["POR"])
    rosdry = jnp.asarray(params["rosdry"])
    dzs = jnp.asarray(params["dzs"])
    BH = jnp.asarray(params["BH"])
    DLMAX = jnp.asarray(params["DLMAX"])
    FLWMAX = jnp.asarray(params["FLWMAX"])
    soilbotflx = params["soilbotflx"]

    # Tsoilsurf for the side columns, computed once before the loop.
    zsc = geom["zsoilcols"]
    Tsoilsurf = tsurfsoilcol_water(
        Tw1, h1, zsc, geom["ddz"], geom["w_lx"], geom["w_ly"],
        geom["w_lx_half"], geom["w_ly_half"], geom["s_lx"], geom["s_ly"],
        M, ncols)

    if "soilflux" in state:
        soilflux = jnp.asarray(state["soilflux"])
    else:
        soilflux = jnp.zeros((L, ncols + 1))

    out = dict(state)
    out["soilflux"] = soilflux
    dt_inv = 1.0 / dt
    for is_ in range(1, ncols):            # side columns, as SOILCOLSTEMP
        Tsoil1 = out["Tsoil1"]
        wl1 = out["wl1"]
        wi1 = out["wi1"]

        csoil, rosoil, lamsoil, lammoist, filtr, wsoil = soil_cond_heat_coef(
            wl1, wi1, POR, rosdry, dzs, BH, DLMAX, FLWMAX, ns, is_)

        Tsoil2 = t_diff_soil(Tsoil1[:, :, is_], lamsoil, csoil, rosoil, dzs,
                             Tsoilsurf[:, is_], dt, soilbotflx, ns)

        # Heat flux into the soil, W/m2 (soil_mod.f90:929-932).
        flux_is = (csoil[:, 1] * rosoil[:, 1] * dt_inv * 0.5 * dzs[1]
                   * (Tsoil2[:, 1] - Tsoil1[:, 1, is_])
                   - 0.25 * (lamsoil[:, 1] + lamsoil[:, 2])
                   * (Tsoil2[:, 2] + Tsoil1[:, 2, is_]
                      - Tsoil2[:, 1] - Tsoil1[:, 1, is_]) / dzs[1])
        soilflux = soilflux.at[:, is_].set(flux_is)

        # SOILFORLAKE reads the module-wide csoil/rosoil/lammoist/filtr of this
        # column and Tsoil2 just computed; hand them over in the state dict.
        work = dict(out)
        work.update(csoil=csoil, rosoil=rosoil, lamsoil=lamsoil,
                    lammoist=lammoist, filtr=filtr, wsoil=wsoil,
                    Tsoil2=Tsoil2, soilflux=soilflux)
        out = soilforlake(work, p, Tsoil2, dt, is_=is_)

    out["soilflux"] = soilflux
    out["Tsoilsurf"] = Tsoilsurf
    return out


# Pure sub-kernels of the pass: these are traceable on their own.  The whole
# pass is not, because SOIL_COND_HEAT_COEF (shared module) is eager-only; see
# the "Not ported / gaps" section of the module docstring.
jitted_tsurfsoilcol_water = jax.jit(tsurfsoilcol_water,
                                    static_argnames=("M", "ncols"))
jitted_t_diff_soil = jax.jit(t_diff_soil, static_argnames=("ns",))


# ---------------------------------------------------------------------------
# geometry packer
# ---------------------------------------------------------------------------

def pack_geometry(geom, M=None, ncols=None):
    """Pack one lake's :class:`~lakematrix.physics.soil.SoilGeometry` for the
    batched kernels (leading lake axis ``L = 1``; repeat for a real batch).

    Returns the ``geom`` dict documented in the module docstring.  ``M`` and
    ``ncols`` default to ``geom.M`` / ``len(geom.zsoilcols) - 2``.
    """
    M = int(geom.M if M is None else M)
    ncols = int(len(geom.zsoilcols) - 2 if ncols is None else ncols)

    def _levels(levels, attr, n):
        return np.array([[getattr(levels[k], attr) for k in range(1, n + 1)]])

    return {
        "zsoilcols": np.asarray(geom.zsoilcols, dtype=float)[None, 1:],
        "ddz": np.asarray(geom.ddz, dtype=float)[None, :],
        "w_lx": _levels(geom.bathymwater, "Lx", M + 1),
        "w_ly": _levels(geom.bathymwater, "Ly", M + 1),
        "w_lx_half": _levels(geom.bathymwater, "Lx_half", M),
        "w_ly_half": _levels(geom.bathymwater, "Ly_half", M),
        "s_lx": _levels(geom.bathymsoil, "Lx", ncols + 1),
        "s_ly": _levels(geom.bathymsoil, "Ly", ncols + 1),
    }
