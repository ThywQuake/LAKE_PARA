"""JAX-vectorised ``LATERHEAT`` water branch (case 1 open water), batch over lakes.

Faithful port of the water branch of :func:`lakematrix.physics.soil.laterheat`
(``LATERHEAT``, soil_mod.f90:1385-1628), i.e. the ``h1 > small, l1 = ls1 = 0``
case used by the case-1 driver.  A leading lake axis is prepended to every
array and the layer / column searches are done with masks (``jnp.where`` /
``argmax``) instead of Python ``if``/``break`` so the whole batch is traced.

Two quirks of the Fortran source are reproduced on purpose:

* the ``i1`` contribution is an **assignment** (``= +`` in the Fortran), not an
  accumulation, so a later soil column silently overwrites an earlier one at
  the same level;
* the ``forall`` sweep is also an assignment and it runs *after* the ``i0`` /
  ``i1`` updates, so it can overwrite an ``i0`` accumulation or an ``i1``
  assignment at the same level.

The scan order (``j = 1 .. ncols-1``, and the ``if (z(M+1) <= zsc(j)) exit``
break) is preserved exactly by iterating the (static) column index in Python
and carrying an ``alive`` mask.

Column/level searches (soil column ``j``, level ``i``; ``z`` increases with
``i``)::

    i0: first i in 1..M+1 with z(i-1) <  zsc(j)   < z(i)      (accumulate)
    i1: first i in 1..M+1 with z(i-1) <  zsc(j+1) < z(i)      (assign)
    forall: all i in 1..M+1 with z(i-1) >= zsc(j) and z(i) <= zsc(j+1)

Array conventions (``L`` = number of lakes, ``M`` = water layers,
``ncols`` = ``nsoilcols``):

``zsoilcols``    (L, ncols+1)  0-based ``zsoilcols(1:nsoilcols+1)`` of
                               :class:`~lakematrix.physics.soil.SoilGeometry`
                               (``geom.zsoilcols[1:]``), the table depths of the
                               soil-column tops *including* the deepest point.
``area_int``     (L, M+1)      0-based water ``area_int(1:M+1)`` (``Bathym``).
``area_half``    (L, M)        0-based water ``area_half(1:M)`` (``Bathym``).
``ddz05``        (M+1,)        ``grid.ddz05`` (Fortran lower bound 0); its
                               running sum is ``zeta_05int``, which is how the
                               layer depths ``z_half = h1*zeta_05int`` are
                               rebuilt (the signature carries no ``z_half``).
``soilflux``     (L, ncols+1)  1-based ``SoilState.soilflux`` (index 0 unused),
                               i.e. pass ``ss.soil.soilflux`` unchanged.
``soil_area_int`` (L, ncols+1) *optional*: soil-column cross-sections
                               ``bathymsoil(1:ncols+1)%area_int`` from
                               ``SOILBATHYMETRY`` (``geom.bathymsoil``), aligned
                               with ``zsoilcols``.  See the note below.

.. note::

   ``LATERHEAT`` mixes the *water* layer areas with ``bathymsoil(j)%area_int``,
   the lake area at the table depth ``zsoilcols(j)`` — a value that comes from
   the morphometry depth/area table and **cannot** be recovered from the
   discretised water levels.  ``soil_area_int`` is therefore an extra optional
   argument; without it the soil areas are interpolated from the ``(z, area)``
   water profile, which is close but only approximates the table (do not expect
   ``1e-12`` against the NumPy reference in that case).  ``soil_area_int=None``
   reproduces ``interp_profile(table_depth, table_area, zsoilcols)`` exactly
   when it is supplied.
"""

import jax
import jax.numpy as jnp

from ..constants import C


def _unpad(x, n):
    """Accept ``n`` 0-based entries, or ``n+1`` 1-based (padded) entries."""
    x = jnp.asarray(x)
    if x.shape[-1] == n:
        return x
    if x.shape[-1] == n + 1:
        return x[..., 1:]
    raise ValueError(
        f"expected last dimension {n} (0-based) or {n + 1} (1-based padded), "
        f"got shape {x.shape}")


def _cumsum_seq(x):
    """Left-to-right cumulative sum, bit-identical to ``numpy.cumsum``.

    ``jnp.cumsum`` is reassociated by XLA (it already differs from
    ``numpy.cumsum`` in the 6th digit of ``ddz05`` for ``M = 40``), which
    matters here: ``zsoilcols(ncols+1)`` sits *exactly* on the deepest layer
    interface ``z(M) = offset + h1*zeta_05int(M)`` (``zsoilcols`` places the
    last-but-one column boundary half a cell above the lake bottom), so the
    ``z(i-1) < zsoilcols(j+1) < z(i)`` test at that level is decided by the
    last bit.  A ``lax.scan`` keeps the Fortran summation order.
    """
    def body(c, xi):
        c = c + xi
        return c, c

    return jax.lax.scan(body, jnp.float64(0.0), x)[1]


def _take2(x, k):
    """``x[i, k[i]]`` for a 2-D ``x`` (L, K) and 0-based ``k`` (L,)."""
    return jnp.take_along_axis(x, k[:, None], axis=-1)[:, 0]


def _interp_profile(xs, ys, x):
    """Piecewise-linear lookup on a batched ``(z, area)`` profile.

    ``INTERP_PROFILE``/``LININTERPOL`` semantics: linear between the nodes,
    linear extrapolation from the first segment below the profile, and the last
    value above it.  Used only for the approximate ``soil_area_int`` fallback.
    """
    K = xs.shape[-1]
    k = jnp.clip(jnp.sum(xs[:, None, :] <= x[..., None], axis=-1) - 1, 0, K - 2)
    x0 = jnp.take_along_axis(xs[:, None, :], k[..., None], axis=-1)[..., 0]
    x1 = jnp.take_along_axis(xs[:, None, :], (k + 1)[..., None], axis=-1)[..., 0]
    y0 = jnp.take_along_axis(ys[:, None, :], k[..., None], axis=-1)[..., 0]
    y1 = jnp.take_along_axis(ys[:, None, :], (k + 1)[..., None], axis=-1)[..., 0]
    out = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return jnp.where(x >= xs[:, -1:], ys[:, -1:], out)


def laterheat_water(zsoilcols, h1, area_int, area_half, ddz05, soilflux,
                    M, ncols, soil_area_int=None):
    """Batched case-1 ``LATERHEAT`` water branch; returns ``lsh_water`` (L, M+2).

    ``M`` and ``ncols`` are static Python ints.  Index 0 of the result is the
    usual Fortran pad (``lsh_water(0)`` is never written); levels ``1 .. M+1``
    hold the lateral heat source in W/m3 for the water layers.
    """
    h1 = jnp.atleast_1d(jnp.asarray(h1))
    L = h1.shape[0]
    small = C.small_value
    zsc = _unpad(zsoilcols, ncols + 1)          # zsoilcols(1:ncols+1)
    Aint = _unpad(area_int, M + 1)              # bathymwater(1:M+1)%area_int
    Ahalf = _unpad(area_half, M)                # bathymwater(1:M)%area_half
    ddz05 = jnp.asarray(ddz05)
    flux = jnp.asarray(soilflux)                # 1-based: flux[..., j] is j

    # zmax = maxval(zsoilcols) over the lake; offset skips the ice shift
    # (ls1 = 0 in the water-only branch).
    zmax = jnp.max(zsc, axis=-1)
    offset = zmax - h1

    # z(0) = water surface, z(1:M) = numerical layer bounds, z(M+1) = bottom.
    zeta05 = _cumsum_seq(ddz05[:M])             # zeta_05int(1:M), see above
    z = jnp.zeros((L, M + 2)).at[:, 1:M + 1].set(
        offset[:, None] + h1[:, None] * zeta05[None, :])
    z = z.at[:, 0].set(offset)
    z = z.at[:, M + 1].set(offset + h1)

    area = jnp.zeros((L, M + 2)).at[:, 1:M + 1].set(Ahalf[:, :M])
    area = area.at[:, 0].set(Aint[:, 0])
    area = area.at[:, M + 1].set(Aint[:, M])

    if soil_area_int is None:
        Asoil = _interp_profile(z, area, zsc)   # approximate, see module doc
    else:
        Asoil = _unpad(soil_area_int, ncols + 1)

    z0, z1 = z[:, 0], z[:, M + 1]
    zprev, zcur = z[:, 0:M + 1], z[:, 1:M + 2]   # z(i-1), z(i) for i = 1..M+1
    lanes = jnp.arange(L)
    alive = h1 > small                          # Fortran: `if (h1 > small)`
    h1s = jnp.where(alive, h1, 1.0)             # keep 1/h1 finite when dry

    water = jnp.zeros((L, M + 2))
    for j in range(1, ncols):                   # waterscan
        zj, zj1 = zsc[:, j - 1], zsc[:, j]
        fl = flux[:, j]
        # OVERLAY(z(0), z(M+1), zsoilcols(j), zsoilcols(j+1))
        ov = (((zj >= z0) & (zj1 <= z1))
              | ((zj <= z0) & (zj1 >= z1))
              | ((zj <= z0) & (zj1 > z0))
              | ((zj < z1) & (zj1 >= z1)))
        act = ov & alive

        # --- i0: z(i-1) < zsoilcols(j) < z(i)  -> accumulation ------------
        m0 = (zprev < zj[:, None]) & (zcur > zj[:, None])
        i0 = jnp.argmax(m0, axis=-1) + 1        # first hit (z is monotone)
        f0 = (1.0 / _take2(Aint, i0 - 1)
              * (_take2(area, i0) - Asoil[:, j - 1])
              / (h1s * ddz05[i0 - 1]) * fl)
        water = water.at[lanes, i0].add(
            jnp.where(m0.any(-1) & (z0 <= zj) & act, f0, 0.0))

        # --- i1: z(i-1) < zsoilcols(j+1) < z(i)  -> assignment ------------
        m1 = (zprev < zj1[:, None]) & (zcur > zj1[:, None])
        i1 = jnp.argmax(m1, axis=-1) + 1
        f1 = (1.0 / _take2(Aint, i1 - 1)
              * (Asoil[:, j] - _take2(area, i1 - 1))
              / (h1s * ddz05[i1 - 1]) * fl)
        old = water[lanes, i1]
        water = water.at[lanes, i1].set(
            jnp.where(m1.any(-1) & (z1 >= zj1) & act, f1, old))

        # --- forall: z(i-1) >= zsoilcols(j) .and. z(i) <= zsoilcols(j+1) --
        mf = (zprev >= zj[:, None]) & (zcur <= zj1[:, None])
        ff = (1.0 / Aint[:, :M + 1]
              * (area[:, 1:M + 2] - area[:, :M + 1])
              / (h1s[:, None] * ddz05[:M + 1][None, :]) * fl[:, None])
        water = water.at[:, 1:M + 2].set(
            jnp.where(mf & act[:, None], ff, water[:, 1:M + 2]))

        alive = alive & ~(z1 <= zj)             # `if (z(M+1) <= zsc(j)) exit`
    return water


jitted_laterheat_water = jax.jit(laterheat_water,
                                 static_argnames=("M", "ncols"))
