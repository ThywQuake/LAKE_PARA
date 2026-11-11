"""JAX (batched) water bathymetry and soil-column bathymetry for case 1.

Port of the water part of ``BATHYMETRY`` (``bathym_mod.f90:7-382``) and of the
``multsoil`` block that fills ``bathymsoil`` (the soil columns of the case-1
lake), with a leading lake axis ``(L, ...)``.  The NumPy reference is
:func:`lakematrix.physics.bathym.bathymetry` /
:func:`lakematrix.physics.bathym.soil_bathymetry`, together with
``grid.zsoilcols`` and ``grid.interp_profile`` (``LININTERPOL`` with the
first-segment extrapolation below the table).

Everything is written with static shapes and ``jnp.where`` / ``argmax`` so it
can be ``jax.jit``-ed; per-lake branches (lake form, the morphometry table
length, the absence of a table) are masks, not Python ``if``.

Array conventions (``L`` = number of lakes, ``M`` = water layers,
``ncols`` = ``nsoilcols``): every array has a leading ``(L,)`` axis and the
Fortran 1-based lists are kept 1-based (padded) where the reference keeps them:

===========================  ==========================================
``h1``, ``ls1``               (L,)      water / deep-ice thickness
``area_lake``                (L,)      ``area_lake`` of the driver
``morphometry``              (L, K, 2)  ``(depth, area)`` table, rows
                                        ``>= morphometry_len`` are padding
``morphometry_len``          (L,)      number of valid table rows
``lakeform``, ``cellipt``    (L,)      ``lakeform`` (1 ellipse, 2 rectangle)
                                        and the ``cellipt`` aspect ratio
``dzeta_int``                (M+1,)    ``grid.dzeta_int``
``dzeta_05int``              (M,)      ``grid.dzeta_05int``
``ddz05``                    (M+1,)    optional ``grid.ddz05``
===========================  ==========================================

Two liberties are taken on purpose and documented at the relevant places:

* ``bathymetry_water`` decides the "no morphometry" branch **per lake**
  (``morphometry_len == 0`` or ``morphometry[l, 0, 1] < 0``).  The NumPy port
  uses the lake-0 flag for the whole call, which only differs for a batch that
  mixes lakes with and without a table;
* an unsupported ``lakeform`` yields NaN instead of raising, because the choice
  has to happen inside the traced kernel.  Case 1 uses ``lakeform = 2``.

The ``grid`` argument of the soil helpers is static data (a NumPy
:class:`lakematrix.grid.Grid`, which is **not** a JAX pytree).  For tracing,
close over it -- ``jax.jit(lambda h1, morph: zsoilcols_jax(h1, morph, GRID,
5))`` -- instead of passing it as a traced argument.
"""

import jax.numpy as jnp

from ..constants import C

FORM_ELLIPSE = 1
FORM_RECTANGLE = 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _valid_rows(morphometry, morphometry_len):
    """(L, K) boolean mask of the real (non-padding) rows of the table."""
    K = morphometry.shape[1]
    n = jnp.asarray(morphometry_len, dtype=jnp.int32).reshape(-1)
    return jnp.arange(K, dtype=jnp.int32)[None, :] < n[:, None]


def _constant_area(morphometry, morphometry_len):
    """(L,) mask: lake has no ``(depth, area)`` table -> constant area."""
    n = jnp.asarray(morphometry_len, dtype=jnp.int32).reshape(-1)
    if morphometry.shape[1] == 0:
        return jnp.ones(n.shape[0], dtype=bool)
    # NaN padding compares False, so this is the per-lake "no data" flag.
    return (n == 0) | (morphometry[:, 0, 1] < 0.0)


def interp_table(dep, area, morphometry_len, x):
    """Batched ``grid.interp_profile`` on a padded ``(depth, area)`` table.

    ``dep``, ``area``: ``(L, K)`` table columns (rows ``>= morphometry_len``
    are padding, anything -- NaN included -- is fine).  ``x``: ``(L, Q)``.
    Returns ``(L, Q)``.

    Semantics of ``LININTERPOL`` (``init.f90:68-117``, see
    :func:`lakematrix.grid.interp_profile`): linear between the nodes, linear
    extrapolation with the first segment below the first node, and the last
    area held above the last node.  Lakes with ``morphometry_len < 2`` produce
    garbage (they take the constant-area branch in the callers).
    """
    K = dep.shape[1]
    if K < 2:
        raise ValueError("interp_table needs at least two table points")
    valid = _valid_rows(dep, morphometry_len)
    # number of real nodes <= x; k = index of the segment that contains x
    count = jnp.sum(valid[:, None, :] & (dep[:, None, :] <= x[..., None]),
                    axis=-1)
    k = jnp.clip(count - 1, 0, K - 2)
    x0 = jnp.take_along_axis(dep[:, None, :], k[..., None], axis=-1)[..., 0]
    x1 = jnp.take_along_axis(dep[:, None, :], (k + 1)[..., None],
                             axis=-1)[..., 0]
    y0 = jnp.take_along_axis(area[:, None, :], k[..., None], axis=-1)[..., 0]
    y1 = jnp.take_along_axis(area[:, None, :], (k + 1)[..., None],
                             axis=-1)[..., 0]
    # Same arithmetic order as numpy.interp: slope first, then slope*(x-x0)+y0.
    slope = (y1 - y0) / (x1 - x0)
    out = slope * (x - x0) + y0
    last = jnp.maximum(
        jnp.asarray(morphometry_len, dtype=jnp.int32).reshape(-1) - 1, 0)
    dep_last = jnp.take_along_axis(dep, last[:, None], axis=-1)[:, 0]
    area_last = jnp.take_along_axis(area, last[:, None], axis=-1)[:, 0]
    return jnp.where(x >= dep_last[:, None], area_last[:, None], out)


def lxly(area, lakeform, cellipt):
    """Batched ``LXLY``: ``(Lx, Ly)`` for ``area`` of shape ``(L, Q)``.

    ``lakeform = 1``: ellipse ``b = sqrt(area/(pi*cellipt))``,
    ``a = b*cellipt``, ``Lx = 2a``, ``Ly = 2b``; ``lakeform = 2``: rectangle
    ``Ly = sqrt(area/cellipt)``, ``Lx = cellipt*Ly``.  Any other value yields
    NaN (the Fortran
    code stops; a traced kernel cannot).
    """
    area = jnp.asarray(area)
    form = jnp.asarray(lakeform).reshape(-1)[:, None]
    cell = jnp.asarray(cellipt, dtype=jnp.float64).reshape(-1)[:, None]
    b = jnp.sqrt(area / (C.pi * cell))
    a = b * cell
    ly = jnp.sqrt(area / cell)
    lx_rect = cell * ly
    lx = jnp.where(form == FORM_ELLIPSE, 2.0 * a,
                   jnp.where(form == FORM_RECTANGLE, lx_rect, jnp.nan))
    ly_out = jnp.where(form == FORM_ELLIPSE, 2.0 * b,
                       jnp.where(form == FORM_RECTANGLE, ly, jnp.nan))
    return lx, ly_out


def _ddz05_from_zeta(dzeta_int):
    """``grid.ddz05`` rebuilt from ``dzeta_int`` (fallback for ``vol``)."""
    ddz = jnp.diff(jnp.asarray(dzeta_int))
    return jnp.concatenate([0.5 * ddz[:1],
                            0.5 * (ddz[:-1] + ddz[1:]),
                            0.5 * ddz[-1:]])


def _interval_index(lo, hi, x):
    """0-based index of the unique ``j`` with ``lo[j] < x <= hi[j]``, or -1.

    ``lo``/``hi``: ``(L, K)`` cell bounds; ``x``: ``(L, Q)``.  The Fortran
    loop has no ``exit``, but the intervals are disjoint, so the index is
    unique; -1 marks "not found" (the default of the ``bathym`` type).
    """
    mask = (x[..., None] > lo[:, None, :]) & (x[..., None] <= hi[:, None, :])
    return jnp.where(jnp.any(mask, axis=-1), jnp.argmax(mask, axis=-1), -1)


def _gather_level(z, j):
    """``z[i, clip(j[i, q], 0)]`` for a 2-D ``z`` (L, K) and 0-based ``j``."""
    idx = jnp.clip(j, 0)[..., None]
    return jnp.take_along_axis(z[:, None, :], idx, axis=-1)[..., 0]


# ---------------------------------------------------------------------------
# water bathymetry
# ---------------------------------------------------------------------------
def bathymetry_water(h1, ls1, area_lake, morphometry, morphometry_len,
                     lakeform, cellipt, pressure, dzeta_int, dzeta_05int,
                     ddz05=None):
    """Batched water part of ``BATHYMETRY`` (case 1, ``h1 > 0``).

    Parameters as in the module docstring: ``h1``/``ls1``/``area_lake``/
    ``morphometry_len``/``lakeform``/``cellipt`` are ``(L,)``,
    ``morphometry`` is ``(L, K, 2)``, ``dzeta_int`` is ``(M+1,)`` and
    ``dzeta_05int`` is ``(M,)``.  ``pressure`` is a scalar or ``(L,)``.

    Returns a dict of the NumPy ``Bathym`` fields:

    =======================  ==========  ===================================
    ``z_full``               (L, M+1)    ``dzeta_int*h1``
    ``z_half``               (L, M)      ``dzeta_05int*h1``
    ``preswat``              (L, M+1)    ``pressure + row0*g*z_full``
    ``area_int``             (L, M+1)    interpolated ``area_int``
    ``area_half``            (L, M)      interpolated ``area_half``
    ``vol``                  (L,)        ``sum(area_int*h1*ddz05)``
    ``botar``                (L,)        ``area_int(1)`` (Fortran telescopes)
    ``Lx``, ``Ly``           (L, M+1)    ``LXLY(area_int)``
    ``rad_int``              (L, M+1)    ``sqrt(area_int/pi)``
    ``Lx_half``, ``Ly_half``  (L, M)     ``LXLY(area_half)``
    =======================  ==========  ===================================

    ``ddz05`` (``grid.ddz05``, ``(M+1,)``) is optional: when it is given
    ``vol`` uses the grid array directly, otherwise it is rebuilt from
    ``dzeta_int`` (``ddz = diff(dzeta_int)``), which differs from the grid
    value by a few ulp of the cumulative sum.
    """
    h1 = jnp.atleast_1d(jnp.asarray(h1))
    L = h1.shape[0]
    ls1 = jnp.broadcast_to(jnp.asarray(ls1, dtype=h1.dtype), h1.shape)
    area_lake = jnp.broadcast_to(jnp.asarray(area_lake, dtype=h1.dtype),
                                 h1.shape)
    cellipt = jnp.broadcast_to(jnp.asarray(cellipt, dtype=h1.dtype), h1.shape)
    lakeform = jnp.broadcast_to(jnp.asarray(lakeform).reshape(-1), h1.shape)
    zeta = jnp.asarray(dzeta_int)
    zeta05 = jnp.asarray(dzeta_05int)
    M = zeta.shape[0] - 1

    z_full = zeta[None, :] * h1[:, None]
    z_half = zeta05[None, :] * h1[:, None]
    pressure = jnp.asarray(pressure)
    if pressure.ndim == 1:      # (L,) -> one pressure per lake
        pressure = pressure[:, None]
    preswat = pressure + C.row0 * C.g * z_full

    dep = jnp.asarray(morphometry)[..., 0]     # (L, K)
    area = jnp.asarray(morphometry)[..., 1]    # (L, K)
    if dep.shape[1] >= 2:
        n = jnp.asarray(morphometry_len, dtype=jnp.int32).reshape(-1)
        valid = _valid_rows(morphometry, n)
        dmax = jnp.max(jnp.where(valid, dep, -jnp.inf), axis=-1)
        constant = _constant_area(morphometry, n)
        # The water surface relative to the deepest table point; the same
        # shift is used for every level (bathym_mod.f90:200-207).
        work1 = dep - dmax[:, None] + h1[:, None] + ls1[:, None]
        area_int = interp_table(work1, area, n, z_full)
        area_half = interp_table(work1, area, n, z_half)
        area_int = jnp.where(constant[:, None], area_lake[:, None], area_int)
        area_half = jnp.where(constant[:, None], area_lake[:, None], area_half)
    else:  # no table at all: constant cross-section
        area_int = jnp.broadcast_to(area_lake[:, None], (L, M + 1))
        area_half = jnp.broadcast_to(area_lake[:, None], (L, M))

    botar = area_int[:, 0]
    if ddz05 is None:
        weights = _ddz05_from_zeta(zeta)
    else:
        weights = jnp.asarray(ddz05)
    vol = jnp.sum(area_int * h1[:, None] * weights[None, :], axis=1)

    Lx, Ly = lxly(area_int, lakeform, cellipt)
    Lx_half, Ly_half = lxly(area_half, lakeform, cellipt)
    rad_int = jnp.sqrt(area_int / C.pi)
    return {
        "z_full": z_full, "z_half": z_half, "preswat": preswat,
        "area_int": area_int, "area_half": area_half, "vol": vol,
        "botar": botar, "Lx": Lx, "Ly": Ly, "rad_int": rad_int,
        "Lx_half": Lx_half, "Ly_half": Ly_half,
    }


# ---------------------------------------------------------------------------
# soil-column bathymetry (bathymsoil)
# ---------------------------------------------------------------------------
def zsoilcols_jax(h1, morphometry, grid, nsoilcols):
    """Batched ``grid.zsoilcols`` (``INIT_VAR`` lines 200-211).

    ``h1``: ``(L,)``; ``morphometry``: ``(L, K, 2)`` with NaN padding beyond
    the real rows (the padding is ignored via ``nanmax``); ``grid``: the
    static :class:`lakematrix.grid.Grid`.  Returns ``(L, nsoilcols+1)``, the
    lake depth of every soil-column top, the last entry being the maximum
    table depth.

    ``nsoilcols <= 1`` is rejected (the NumPy helper returns ``None``, which
    has no static-shape equivalent); a lake without a table yields NaN.
    """
    ncols = int(nsoilcols)
    if ncols <= 1:
        raise ValueError("zsoilcols_jax needs nsoilcols >= 2")
    h1 = jnp.atleast_1d(jnp.asarray(h1))
    dep = jnp.asarray(morphometry)[..., 0]                    # (L, K)
    dmax = jnp.nanmax(dep, axis=-1)                           # (L,)
    work = (dmax - h1 * 0.5 * float(grid.ddz_last)) / float(ncols - 1)
    zsc = jnp.arange(ncols, dtype=jnp.float64)[None, :] * work[:, None]
    return jnp.concatenate([zsc, dmax[:, None]], axis=1)


def soil_bathymetry_jax(h1, morphometry, morphometry_len, lakeform, cellipt,
                        grid, nsoilcols, ls1=0.0):
    """Batched ``bathymsoil`` of the case-1 ``BATHYMETRY`` (multsoil block).

    ``h1``/``ls1``: ``(L,)`` (``ls1`` is scalar or ``(L,)``; case 1 uses 0);
    ``morphometry``/``morphometry_len``/``lakeform``/``cellipt`` as in
    :func:`bathymetry_water`; ``grid`` and ``nsoilcols`` static.

    The reference is :func:`lakematrix.physics.bathym.soil_bathymetry`, which
    returns a 1-based list of ``soil.Bathym``.  Here every field is an array
    with shape ``(L, ncols+2)``: index 0 is the unused Fortran pad, ``1 ..
    ncols+1`` are the columns (``ncols+1`` is the deepest table point).  The
    ``bathym`` defaults are reproduced for the entries the reference never
    writes (``area_half(ncols+1) = 0``, ``itop = ibot = icent = 1``,
    ``dzSL = dzSLc = 0``, and ``Lx_half = Ly_half = 0``, which the NumPy
    ``Bathym`` leaves at its 0.0 default for soil columns).

    Returns a dict with

    ==============  ==============  =======================================
    ``zsoilcols``   (L, ncols+1)    :func:`zsoilcols_jax` (no pad)
    ``work3``       (L, ncols)      mid-points ``0.5*(zsc[i]+zsc[i+1])``
    ``area_int``    (L, ncols+2)    ``interp_profile(dep, area, zsoilcols)``
    ``area_half``   (L, ncols+2)    ``interp_profile(dep, area, work3)``
    ``Lx``, ``Ly``  (L, ncols+2)    ``LXLY(area_int)``
    ``itop``        (L, ncols+2)    water level of the column top
    ``dzSL``        (L, ncols+2)    height of the top above ``z_full(itop-1)``
    ``ibot``        (L, ncols+2)    water level of the next column top
    ``icent``       (L, ncols+2)    water half level closest to ``work3``
    ``dzSLc``       (L, ncols+2)    ``work3`` height above ``z_half(icent-2)``
    ==============  ==============  =======================================

    Note the off-by-one of the original: ``itop`` of column ``c`` (2-based
    Fortran ``c = 2..ncols``) is found from ``zsoilcols(c)`` while its
    ``dzSL`` is measured from ``work3(c)``; ``ibot`` of column ``c-1`` uses the
    same ``zsoilcols(c)``.  That quirk is reproduced.

    The lake needs a morphometry table with at least two rows
    (:func:`interp_table` requirement); ``ls1`` is only a coordinate shift and
    is zero in case 1.
    """
    nc = int(nsoilcols)
    if nc <= 1:
        raise ValueError("soil_bathymetry_jax needs nsoilcols >= 2")
    M = int(grid.M)
    if M < 2:
        raise ValueError("soil_bathymetry_jax needs M >= 2")
    h1 = jnp.atleast_1d(jnp.asarray(h1))
    L = h1.shape[0]
    ls1 = jnp.broadcast_to(jnp.asarray(ls1, dtype=h1.dtype), h1.shape)
    lakeform = jnp.broadcast_to(jnp.asarray(lakeform).reshape(-1), h1.shape)
    cellipt = jnp.broadcast_to(jnp.asarray(cellipt, dtype=h1.dtype), h1.shape)
    dep = jnp.asarray(morphometry)[..., 0]
    area = jnp.asarray(morphometry)[..., 1]
    n = jnp.asarray(morphometry_len, dtype=jnp.int32).reshape(-1)
    valid = _valid_rows(morphometry, n)
    dmax = jnp.max(jnp.where(valid, dep, -jnp.inf), axis=-1)

    zsc = zsoilcols_jax(h1, morphometry, grid, nc)             # (L, nc+1)
    work3 = 0.5 * (zsc[:, :-1] + zsc[:, 1:])                    # (L, nc)
    area_int0 = interp_table(dep, area, n, zsc)                # (L, nc+1)
    area_half0 = interp_table(dep, area, n, work3)             # (L, nc)
    lx0, ly0 = lxly(area_int0, lakeform, cellipt)              # (L, nc+1)

    area_int = jnp.zeros((L, nc + 2)).at[:, 1:nc + 2].set(area_int0)
    area_half = jnp.zeros((L, nc + 2)).at[:, 1:nc + 1].set(area_half0)
    Lx = jnp.zeros((L, nc + 2)).at[:, 1:nc + 2].set(lx0)
    Ly = jnp.zeros((L, nc + 2)).at[:, 1:nc + 2].set(ly0)

    zeta = jnp.asarray(grid.dzeta_int)
    zeta05 = jnp.asarray(grid.dzeta_05int)
    z_full = zeta[None, :] * h1[:, None]                       # (L, M+1)
    z_half = zeta05[None, :] * h1[:, None]                     # (L, M)
    # Coordinate shift between the table and the model (bathym_mod.f90:146)
    dz = -dmax + h1 + ls1

    ones = jnp.ones((L, nc + 2), dtype=jnp.int32)
    zeros = jnp.zeros((L, nc + 2))

    # --- itop / dzSL: columns 2..ncols from zsoilcols(2..ncols) ------------
    x_top = zsc[:, 1:nc] + dz[:, None]                         # (L, nc-1)
    j_top = _interval_index(z_full[:, :-1], z_full[:, 1:], x_top)
    found = j_top >= 0
    one = jnp.int32(1)
    top = jnp.where(found, j_top + 1, one).astype(jnp.int32)
    itop = ones.at[:, 2:nc + 1].set(top)
    zlev = _gather_level(z_full, jnp.maximum(j_top, 0))
    dzSL = zeros.at[:, 2:nc + 1].set(
        jnp.where(found, work3[:, 1:nc] + dz[:, None] - zlev, 0.0))
    itop = itop.at[:, 1].set(one)                              # b(1)%itop = 1
    dzSL = dzSL.at[:, 1].set(work3[:, 0] + dz - z_full[:, 0])

    # --- ibot: column c-1 from zsoilcols(c), columns 1..ncols-1 -----------
    ibot = ones.at[:, 1:nc].set(top)
    ibot = ibot.at[:, nc].set(jnp.int32(M + 1))

    # --- icent / dzSLc: columns 1..ncols-1 from work3(1..ncols-1) ---------
    x_c = work3[:, :nc - 1] + dz[:, None]                      # (L, nc-1)
    j_c = _interval_index(z_half[:, :-1], z_half[:, 1:], x_c)
    cfound = j_c >= 0
    zc = _gather_level(z_half, jnp.maximum(j_c, 0))
    icent = ones.at[:, 1:nc].set(jnp.where(cfound, j_c + 2, one).astype(
        jnp.int32))
    dzSLc = zeros.at[:, 1:nc].set(
        jnp.where(cfound, work3[:, :nc - 1] + dz[:, None] - zc, 0.0))

    return {
        "zsoilcols": zsc, "work3": work3,
        "area_int": area_int, "area_half": area_half,
        "Lx": Lx, "Ly": Ly,
        "itop": itop, "dzSL": dzSL, "ibot": ibot,
        "icent": icent, "dzSLc": dzSLc,
    }
