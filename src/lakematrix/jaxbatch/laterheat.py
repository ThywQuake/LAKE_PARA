"""JAX-batched ``LATERHEAT`` (water branch), with a leading lake axis.

Port of the ``h1 > small, l1 = 0, ls1 = 0`` case of
:func:`lakematrix.physics.soil.laterheat` (``LATERHEAT``, soil_mod.f90:1385-
1628): the lateral heat source that the soil columns of a case-1 open-water
lake inject into each water layer.

Everything is batched over lakes (``L``): the layer searches become boolean
masks and ``argmax`` (``z`` is monotone in the level index, so ``argmax`` is
the ``break`` of the Fortran loop), and the column loop is a Python loop over
the static ``ncols`` index carrying an ``alive`` mask for the Fortran
``if (z(M+1) <= zsoilcols(j)) exit``.  No Python ``if`` is applied to a traced
value.

Preserved quirks
================

* ``i1`` is an **assignment** (``lsh%water(i1) = + ...`` in the Fortran), not
  an accumulation: a later soil column overwrites an earlier one that wrote
  the same level.  ``i0`` is a true accumulation.
* The ``forall`` sweep is also an assignment and runs *after* both searches,
  so it overwrites an ``i0`` accumulation or an ``i1`` assignment at the same
  level.
* The Fortran ``exit`` is only evaluated when ``OVERLAY`` was true (the
  ``continue`` skips the rest of the body); reproduced by gating the break
  with ``overlay``.
* The top/bottom guards (``z(0) <= zj`` / ``z(M+1) >= zj1``) are reproduced
  even though monotone ``z`` makes them redundant.

Numerics
========

The layer masks are *exact* float comparisons on ``z``, so an ulp of
difference can select a different level.  Two measures keep the batched kernel
bit-identical to the NumPy reference (and identical under ``jit``):
``_cumsum_seq`` sums ``ddz05`` in the ``grid.dzeta_05int`` order, and an
``optimization_barrier`` stops XLA from contracting ``offset + h1*zeta05``
into an FMA (the reference rounds the product and the sum separately).

Case-1 scope
============

``state.l1 = state.ls1 = 0`` is assumed, so ``step(ls1 - small)*ls1 = 0``
and ``offset = maxval(zsoilcols) - h1``.  The ice/deep-ice scans of
``LATERHEAT`` are not implemented here.

Inputs (``M`` water layers, ``ncols = nsoilcols``; all floats are float64)
==========================================================================

``zsoilcols``    (L, ncols+1)  0-based ``geom.zsoilcols[1:ncols+2]`` (table
                               depth of every soil-column top, deepest point
                               last).  A 1-based padded ``(L, ncols+2)`` array
                               is accepted too (leading dummy dropped).
``h1``           (L,)          water depth.
``area_int``     (L, M+1)      0-based ``bathym.area_int`` (water areas at the
                               integer levels, surface first).
``area_half``    (L, M)        0-based ``bathym.area_half`` (water areas at the
                               layer centres).  A 1-based padded ``(L, M+1)``
                               array is accepted as well (leading dummy
                               dropped, i.e. Fortran ``bathymwater(0:M)``).
``ddz05``        (M+1,)        ``grid.ddz05`` (Fortran lower bound 0).  Its
                               running sum is ``dzeta_05int``, so the layer
                               depths ``z(i) = offset + h1*zeta_05int(i)`` are
                               rebuilt bit-exactly from this array (no
                               ``z_half`` argument is needed).
``soilflux``     (L, ncols+1)  1-based padded ``state.soilflux`` (index 0
                               unused; column ``j`` is ``soilflux[j]``).
``soil_area_int`` (L, ncols+1) *optional*: ``geom.bathymsoil[1:ncols+2]``
                               ``.area_int`` — the lake area at each soil
                               table depth.

Returns ``lsh_water`` (L, M+2); index 0 is the Fortran pad (always 0) and
levels ``1 .. M+1`` are the lateral heat source in W/m3.

The ``soil_area_int`` gap
=========================

``LATERHEAT`` mixes the *water* areas with ``bathymsoil(j)%area_int``, which
comes from the morphometry depth/area table (``SOILBATHYMETRY``) and is
**not** a function of the water bathymetry.  The 8-argument signature above
does not carry it, so:

* pass ``soil_area_int`` for an exact (``<= 1e-12``) reproduction of the
  NumPy reference — this is the intended coupling for the batch driver;
* leaving it ``None`` falls back to interpolating the ``(z, area)`` water
  profile at ``zsoilcols``, which is only an approximation: the water nodes
  are layer centres and lose the kinks of the depth/area table.  On
  ``ref/v1/summer60d`` that fallback is off by up to ``~4e-2`` relative
  (``~36`` W/m3 absolute), so it is **not** 1e-12 accurate.

Warning: here the fallback is deliberately permissive so that the documented
8-argument call still works; do not use it for a validation-grade run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constants import C

__all__ = ["laterheat_water", "jitted_laterheat_water"]


def _unpad(x, n):
    """Return the ``n`` valid entries of a 0-based or 1-based padded array.

    ``x[..., -1]`` must be ``n`` (0-based: all entries valid) or ``n + 1``
    (1-based, Fortran-style: ``x[..., 0]`` is the dummy and is dropped).
    """
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

    ``grid.dzeta_05int`` is built by a sequential loop, so a reassociated sum
    differs from it in the last bits.  That matters here: ``zsoilcols`` can
    sit *exactly* on a layer interface, and the ``z(i-1) < zsoilcols(j+1) <
    z(i)`` test is then decided by that last bit.  Plain ``jnp.cumsum`` -- and
    even a ``lax.scan`` of ``c + x`` once XLA parallelises it -- is
    reassociated; writing the array element by element leaves a real
    loop-carried dependence, so eager and jitted code both reproduce the NumPy
    summation order exactly.
    """
    n = x.shape[0]
    acc0 = jnp.zeros(n + 1, dtype=jnp.result_type(x, 0.0))

    def body(i, acc):
        return acc.at[i + 1].set(acc[i] + x[i])

    return jax.lax.fori_loop(0, n, body, acc0)[1:]


def _take2(x, k):
    """``x[i, k[i]]`` for a 2-D ``x`` (L, K) and 0-based ``k`` (L,)."""
    return jnp.take_along_axis(x, k[:, None], axis=-1)[:, 0]


def _interp_profile(xs, ys, x):
    """``LININTERPOL`` semantics on a batched ``(z, area)`` profile.

    Same rules as :func:`lakematrix.grid.interp_profile`: linear between the
    nodes, linear extrapolation below the first node (first segment), and the
    last value at or above the last node.  Only used by the approximate
    ``soil_area_int=None`` fallback.
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
    """Batched case-1 ``LATERHEAT`` water scan; returns ``lsh_water`` (L, M+2).

    ``M`` and ``ncols`` are static Python ints.  See the module docstring for
    the array conventions and for the ``soil_area_int`` caveat (pass it to
    match :func:`lakematrix.physics.soil.laterheat` exactly).
    """
    h1 = jnp.atleast_1d(jnp.asarray(h1)).reshape(-1)
    L = h1.shape[0]
    small = C.small_value

    zsc = _unpad(zsoilcols, ncols + 1)          # zsoilcols(1:ncols+1)
    Aint = _unpad(area_int, M + 1)              # bathymwater(1:M+1)%area_int
    Ahalf = _unpad(area_half, M)                # bathymwater(1:M)%area_half
    ddz05 = jnp.asarray(ddz05)                  # ddz05(0:M), lower bound 0
    flux = jnp.asarray(soilflux)                # 1-based: flux[..., j] == j

    # --- geometry: z(i) and area(i), i = 0 .. M+1 -------------------------
    # offset = maxval(zsoilcols) - h1  (ls1 = 0 in the case-1 water branch)
    offset = jnp.max(zsc, axis=-1) - h1
    # h1*dzeta_05int is the layer-centre depth; dzeta_05int = cumsum(ddz05).
    zeta05 = _cumsum_seq(ddz05[:M])             # zeta_05int(1:M), see above
    # The barrier keeps XLA from contracting offset + h1*zeta05 into an FMA:
    # the reference rounds the product and the sum separately, and the layer
    # masks below are exact comparisons on z (ties must not flip).
    z_half = jax.lax.optimization_barrier(h1[:, None] * zeta05[None, :])
    depth = offset[:, None] + z_half
    z = jnp.concatenate(
        (offset[:, None], depth, (offset + h1)[:, None]), axis=-1)
    area = jnp.concatenate(
        (Aint[:, :1], Ahalf[:, :M], Aint[:, M:M + 1]), axis=-1)

    if soil_area_int is None:
        # Approximate: the water profile is sampled at the layer centres and
        # has lost the depth/area table's kinks (see the module docstring).
        Asoil = _interp_profile(z, area, zsc)
    else:
        Asoil = _unpad(soil_area_int, ncols + 1)  # bathymsoil(1:ncols+1)

    # --- the scan ---------------------------------------------------------
    # zprev/zcur are z(i-1)/z(i) for i = 1 .. M+1; the column k of the mask
    # array corresponds to level i = k+1 (Fortran 1-based level).
    z0 = z[:, 0]
    z1 = z[:, M + 1]
    zprev = z[:, :M + 1]
    zcur = z[:, 1:]                             # z(i) = z[i+1] in 0-based
    lanes = jnp.arange(L)

    alive = h1 > small                          # Fortran `if (h1 > small)`
    h1s = jnp.where(alive, h1, 1.0)             # keep the arithmetic finite

    water = jnp.zeros((L, M + 2))
    for j in range(1, ncols):                   # waterscan
        zj = zsc[:, j - 1]                      # zsoilcols(j)
        zj1 = zsc[:, j]                         # zsoilcols(j+1)
        fl = flux[:, j]                         # soilflux(j)

        # OVERLAY(z(0), z(M+1), zsoilcols(j), zsoilcols(j+1))
        ov = (((zj >= z0) & (zj1 <= z1))
              | ((zj <= z0) & (zj1 >= z1))
              | ((zj <= z0) & (zj1 > z0))
              | ((zj < z1) & (zj1 >= z1)))
        act = ov & alive

        # i0: first i with z(i-1) < zsoilcols(j) < z(i)  ->  accumulation
        m0 = (zprev < zj[:, None]) & (zcur > zj[:, None])
        k0 = jnp.argmax(m0, axis=-1)            # 0-based column => level k0+1
        f0 = (1.0 / _take2(Aint, k0)
              * (_take2(area, k0 + 1) - Asoil[:, j - 1])
              / (h1s * ddz05[k0]) * fl)
        water = water.at[lanes, k0 + 1].add(
            jnp.where(m0.any(-1) & (z0 <= zj) & act, f0, 0.0))

        # i1: first i with z(i-1) < zsoilcols(j+1) < z(i)  ->  assignment
        m1 = (zprev < zj1[:, None]) & (zcur > zj1[:, None])
        k1 = jnp.argmax(m1, axis=-1)
        f1 = (1.0 / _take2(Aint, k1)
              * (Asoil[:, j] - _take2(area, k1))
              / (h1s * ddz05[k1]) * fl)
        hit1 = m1.any(-1) & (z1 >= zj1) & act
        water = water.at[lanes, k1 + 1].set(
            jnp.where(hit1, f1, water[lanes, k1 + 1]))

        # forall i = 1 .. M+1: z(i-1) >= zsoilcols(j) .and. z(i) <= zsoilcols(j+1)
        mf = (zprev >= zj[:, None]) & (zcur <= zj1[:, None])
        ff = (1.0 / Aint[:, :M + 1]
              * (area[:, 1:M + 2] - area[:, :M + 1])
              / (h1s[:, None] * ddz05[None, :M + 1]) * fl[:, None])
        water = water.at[:, 1:M + 2].set(
            jnp.where(mf & act[:, None], ff, water[:, 1:M + 2]))

        # `if (z(M+1) <= zsoilcols(j)) exit` — reached only if OVERLAY passed
        alive = alive & ~(ov & (z1 <= zj))
    return water


jitted_laterheat_water = jax.jit(laterheat_water,
                                 static_argnames=("M", "ncols"))
