"""Batched JAX port of ``MIXED_LAYER_CALC`` and ``TURB_SCALES`` (case 1).

Mirrors :func:`lakematrix.physics.turb.mixed_layer_calc` and
:func:`lakematrix.physics.turb.turb_scales` with a **leading lake axis**
``(L, ...)`` so that the turbulence diagnostics can be produced for a whole
batch of lakes in one JIT-compiled call.  The NumPy implementation stays the
reference; every formula, constant and operation order is copied from it (the
NumPy functions themselves are ports of ``phys_func.f90:1072-1154`` --
``MIXED_LAYER_CALC`` -- and ``phys_func.f90:914-1072`` -- ``TURB_SCALES``).

Scope -- the configuration this is validated for
-----------------------------------------------
Case 1, open water, ``Turbpar = 2`` (k-epsilon closure, so ``lamw``/``lamsal``
come from the k-epsilon diffusivities), ``eos = 5``, ``lindens = 0``.  This is
exactly the argument set :func:`lakematrix.step.step_case1` builds.  The two
routines themselves are switch-free -- they only see the arguments below -- so
there is nothing to raise ``NotImplementedError`` for; the other Fortran
configurations differ in what the *caller* passes (ice/snow, another
``Turbpar``, another equation of state or ``lindens = 1`` need different
``lamw``/``DDENS_*``) and are simply not covered by the validation.

Array conventions
-----------------
``L``  number of lakes in the batch, ``M`` number of water layers.

Every array may be passed *shared* (shape ``(n,)`` / scalar) or *batched*
(shape ``(L, n)``); a leading axis of length 1 is broadcast to ``L``.  Slices
follow the NumPy reference, i.e. the arrays are **0-based copies of the 1-based
Fortran arrays** (``Fortran a(1:M+1)`` -> ``a[0:M+1]``), except where an array
is documented as keeping a Fortran lower bound of 0 (``ddz05``, ``integr``) --
for those the 0-based index *is* the Fortran index.

========================  =============  =====================================
argument                  shape          meaning (NumPy reference)
========================  =============  =====================================
``row``                    ``(L, M+1)``  density ``row(1:M+1)`` = ``row[0:M+1]``
``ddz``                    ``(L, M)``    layer thickness ``ddz(1:M)``
``ddz2``                   ``(L, M-1)``  **unused** (signature parity)
``dzeta_int``              ``(L, M+1)``  **unused** (signature parity)
``dzeta_05int``            ``(L, M)``    ``dzeta_05int(1:M)``
``h1``                     ``(L,)``      water depth, m
``M``                      int            number of water layers (static)
``tw2``, ``sal2``          ``(L, M+1)``  ``Tw2(1:M+1)``, ``Sal2(1:M+1)``
``u2``, ``v2``             ``(L, M+1)``  ``u2(1:M+1)``, ``v2(1:M+1)``
``lamw``                   ``(L, M)``    ``lamw(1:M)``, W/(m K)
``lamsal``                 ``(L, M)``    ``lamsal(1:M)``; the padded
                                         ``(L, M+1)`` form built by
                                         ``step_case1`` is accepted and
                                         sliced to ``[..., :M]`` as the
                                         NumPy loop does
``k_turb_T_flux``          ``(L, M)``    **unused** (commented-out Fortran)
``T_massflux``             ``(L, M)``    **unused** (commented-out Fortran)
``eflux0_kinem``           ``(L,)``      surface kinematic heat flux, K m/s
``turb_density_flux``      ``(L,)``      **unused** (argument parity)
``Buoyancy0``              ``(L,)``      surface buoyancy flux, m2/s3
``tau``                    ``(L,)``      effective wind stress, N/m2
``kor``                    ``(L,)``      Coriolis parameter, 1/s
``rad_integr``             ``(L, M+1)``  ``RadWater%integr(0:M)`` -- Fortran
                                         bound 0, so ``rad_integr[i]`` is
                                         ``integr(i)``; it is indexed with the
                                         *1-based* ``i_maxN``, as in the
                                         reference
``area_int``               ``(L, M+1)``  ``area_int(1:M+1)``
``Lx``, ``Ly``             ``(L, M+1)``  ``Lx(1:M+1)`` / ``Ly(1:M+1)``; only the
                                         surface value ``[..., 0]`` is used
                                         (``max(Lx(1), Ly(1))``).  ``(L,)``
                                         values with only that entry are
                                         accepted too
``vol``                    ``(L,)``      lake volume, m3
``z_full``                 ``(L, M+1)``  ``Grid.z_full = dzeta_int*h1``, m
``ddz05``                  ``(L, M+1)``  ``Grid.ddz05`` -- Fortran bound 0:M
``DDENS_DTEMP0``,          ``(L,)`` or   ``SET_DDENS_DTEMP``/``SET_DDENS_DSAL``
``DDENS_DSAL0``            float          at ``(Tw1(1), Sal1(1))``, the
                                         ``density.set_dens_consts()`` values
========================  =============  =====================================

Output
------
``mixed_layer_calc`` returns ``(i_ML, H_mixed_layer, maxN)``:

* ``i_ML`` -- ``(L,)`` int32, the **1-based** mixed-layer index, i.e. the same
  number the NumPy/Fortran functions return (``1 .. M``);
* ``H_mixed_layer`` -- ``(L,)`` mixed-layer depth, m;
* ``maxN`` -- ``(L,)`` maximum buoyancy frequency, 1/s.

``turb_scales`` returns the same ``dict`` as the NumPy function (plus the
``i_maxN``/``H_mixed_layer``/``maxN`` it re-exports): the 11 scalars are
``(L,)`` arrays, ``Rp``/``Rpdens`` are ``(L, M)`` arrays.

Reference quirks reproduced on purpose
--------------------------------------
* ``maxN`` is computed from ``argmax(bvf)`` **before** the well-mixed fallback
  ``ml = M``, so ``maxN`` still belongs to the original layer (Fortran order).
* The fallback test of the NumPy port is ``(row[M] - row[0])/row0 < 1e-3``,
  i.e. it uses the bottom density ``row(M+1)``; the Fortran source tests
  ``row(M)`` (the second-deepest level, ``row[M-1]``).  This module follows the
  NumPy port, which is the reference the batch is validated against (they are
  identical for the reference case, see NOTES.md).
* The thermocline search has no ``exit``: if the 14 degC (8 degC) crossing
  occurs more than once, the **last** crossing wins.  That is reproduced with a
  masked ``max`` over the layer index, not with ``argmax`` (which would return
  the first one).
* When a crossing is not found the reference keeps the initialised defaults:
  ``x1 = 0`` for the 14 degC level but ``x2 = h1`` for the 8 degC level, and
  all interpolated ``u``/``v``/``Sal`` values stay ``0``.  So a profile that
  never reaches 8 degC yields ``ThermThick = h1``, not ``0``.

JIT
---
``jitted_mixed_layer_calc`` / ``jitted_turb_scales`` are ``jax.jit`` aliases
with ``M`` as a static argument.  ``make_turb_scales_kernel(M)`` returns a
jitted callable with ``M`` bound (positional signature without ``M``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constants import C
from ..physics.density import (A11, A12, A13, A14, alpha1, alpha2, alpha3,
                               alpha4, alpha5, beta1, beta2, beta3, gamma1)

__all__ = ["mixed_layer_calc", "turb_scales", "jitted_mixed_layer_calc",
           "jitted_turb_scales", "make_turb_scales_kernel"]

#: ``Ttherm0`` / ``Ttherm1`` -- temperatures of the top/bottom of the
#: thermocline (phys_func.f90:952).
TTHERM0 = 14.0
TTHERM1 = 8.0

_G = C.g
_ROW0 = C.row0
_SMALL = C.small_value
_NIU_WAT = C.niu_wat
_G_D_KELVIN0 = C.g / C.Kelvin0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ddens_dtemp(temp, sal):
    """``SET_DDENS_DTEMP`` for ``eos 5`` (``density.ddens_dtemp``), vectorised."""
    sg = sal * 1.0e3
    out = (-(_ROW0 / A12) * (temp - A14) / ((temp + A13) ** 2)
           * ((temp + A13) * ((temp - A14) + 2.0 * (temp + A11))
              - (temp + A11) * (temp - A14)))
    a = (-alpha2 + 2.0 * alpha3 * temp - 3.0 * alpha4 * temp ** 2
         + 4.0 * alpha5 * temp ** 3)
    b = beta2 - 2.0 * beta3 * temp
    return out + a * sg + b * sg ** 1.5


def _ddens_dsal(temp, sal):
    """``SET_DDENS_DSAL`` for ``eos 5`` (``density.ddens_dsal``), vectorised."""
    sg = sal * 1.0e3
    a = (alpha1 - alpha2 * temp + alpha3 * temp ** 2 - alpha4 * temp ** 3
         + alpha5 * temp ** 4)
    b = -beta1 + beta2 * temp - beta3 * temp ** 2
    c = gamma1
    return (a + 1.5 * b * sg ** 0.5 + 2.0 * c * sg) * 1.0e3


def _bound_from_below(x, xmin):
    """Fortran ``BOUND_FROM_BELOW``: ``max(|x|, xmin) * sign(1, x)``.

    ``sign(1, x) = +1`` for ``x >= 0`` (including ``-0.0``), as in
    ``numerics_mod.f90:10-20`` and :func:`physics.turb._bound_from_below`.
    """
    return jnp.where(x >= 0.0, jnp.maximum(x, xmin), -jnp.maximum(-x, xmin))


def _as2(x, n, key, n_ok=None):
    """``(n,)`` / ``(L, n)`` -> ``(L, n)`` (one leading lake axis).

    ``n_ok`` optionally widens the accepted last dimension(s) (``lamsal`` may
    be the padded ``M+1`` array the reference callers pass).
    """
    a = jnp.asarray(x, dtype=jnp.float64)
    if a.ndim == 0:
        raise ValueError(f"{key}: expected an array of length {n}, "
                         f"got a scalar")
    if a.ndim == 1:
        a = a[None, :]
    ok = (n,) if n_ok is None else tuple(n_ok)
    if a.ndim != 2 or a.shape[-1] not in ok:
        allow = " or ".join(f"(L, {m})" for m in ok)
        raise ValueError(f"{key}: expected shape {allow}, got {a.shape}")
    return a


def _fit(a, L, n, key):
    """Broadcast the lake axis of ``a`` (``(1, n)`` or ``(L, n)``) to ``L``."""
    if a.shape[0] == 1 and L != 1:
        return jnp.broadcast_to(a, (L, n))
    if a.shape[0] != L:
        raise ValueError(f"{key}: leading lake dim {a.shape[0]} != L={L}")
    return a


def _scal(x, L, key):
    """Broadcast a scalar / ``(1,)`` / ``(L,)`` value to ``(L,)``."""
    a = jnp.asarray(x, dtype=jnp.float64)
    if a.ndim > 1:
        raise ValueError(f"{key}: expected a scalar or (L,), got {a.shape}")
    if a.ndim == 1:
        if a.shape[0] not in (1, L):
            raise ValueError(f"{key}: expected {L} values, got {a.shape}")
        if a.shape[0] == 1:
            a = a[0]
    return jnp.broadcast_to(a, (L,))


def _surf2(x, L, M, key):
    """``(L, M+1)`` / ``(L, 1)`` / ``(L,)`` / scalar -> ``(L,)`` ``[..., 0]``."""
    a = jnp.asarray(x, dtype=jnp.float64)
    if a.ndim == 0:
        return jnp.broadcast_to(a, (L,))
    if a.ndim == 2:
        if a.shape[-1] not in (1, M + 1):
            raise ValueError(f"{key}: expected shape (L, {M + 1}) or (L, 1), "
                             f"got {a.shape}")
        a = a[..., 0]
        if a.shape[0] == 1 and L != 1:
            return jnp.broadcast_to(a, (L,))
        if a.shape[0] != L:
            raise ValueError(f"{key}: leading lake dim {a.shape[0]} != L={L}")
        return a
    return _scal(a, L, key)


def _ordered_sum(x):
    """Sum over the last axis in index order (bit-identical to the loops).

    ``jnp.sum`` may reassociate (pairwise/tree reduction), which differs from
    the left-to-right ``zm = zm + ...`` accumulation of the reference by a few
    ulp; the explicit Python loop keeps the reference order (``M`` is static).
    """
    acc = jnp.zeros(x.shape[:-1], dtype=x.dtype)
    for i in range(x.shape[-1]):
        acc = acc + x[..., i]
    return acc


def _take(x, idx):
    """``x[..., idx]`` per lake for a traced ``(L,)`` index; ``x`` is ``(L, n)``.

    ``idx`` may hold the ``-1`` "no match" sentinel; it is clamped so the
    gather stays in bounds (the caller masks the result with ``jnp.where``).
    """
    return jnp.take_along_axis(x, jnp.maximum(idx, 0)[:, None], axis=-1)[:, 0]


def _pair(x, idx):
    """``(x[:, idx], x[:, idx+1])`` per lake; ``x`` is ``(L, n)``."""
    return (_take(x[:, :-1], idx), _take(x[:, 1:], idx))


def _last_match(mask):
    """Index of the **last** ``True`` per row, ``-1`` when there is none."""
    n = mask.shape[-1]
    return jnp.max(jnp.where(mask, jnp.arange(n), -1), axis=-1)


# ---------------------------------------------------------------------------
# MIXED_LAYER_CALC
# ---------------------------------------------------------------------------
def mixed_layer_calc(row, ddz, dzeta_05int, h, M):
    """Batched ``MIXED_LAYER_CALC`` (phys_func.f90:1072-1154).

    ``row`` is the density ``row(1:M+1)`` with shape ``(L, M+1)`` (no Fortran
    index-0 pad -- the slice ``row[1:M+1] - row[0:M]`` of the reference is
    ``row[:, 1:] - row[:, :-1]`` here).  ``ddz`` is ``ddz(1:M)``, ``(L, M)``;
    ``dzeta_05int`` is ``dzeta_05int(1:M)``, ``(L, M)``; ``h`` is the water
    depth, ``(L,)`` or a scalar.

    Returns ``(i_ML, H_mixed_layer, maxN)`` -- ``i_ML`` is 1-based (``(L,)``
    int32), ``H_mixed_layer`` and ``maxN`` are ``(L,)`` float64:

    ``bvf(i) = g/row0*(row(i+1) - row(i))/(h*ddz(i))``, ``i = 1..M``;
    ``i_ML  = argmax(bvf) + 1`` (first maximum, as ``maxloc``);
    ``maxN  = sqrt(max(bvf(i_ML), 0))`` -- computed *before* the fallback;
    ``i_ML = M`` if ``(row(M+1) - row(1))/row0 < 1e-3`` (well mixed);
    ``H_mixed_layer = dzeta_05int(i_ML)*h``.

    The ``(row(M+1) - row(1))`` test is the NumPy port's (see the module
    docstring); pass ``row[1:]`` when starting from the 1-based padded
    ``water_dens_ts(np.r_[0, Tw1], np.r_[0, Sal1])`` form.
    """
    row = _as2(row, M + 1, "row")
    L = int(row.shape[0])
    ddz = _fit(_as2(ddz, M, "ddz"), L, M, "ddz")
    dzeta_05int = _fit(_as2(dzeta_05int, M, "dzeta_05int"), L, M, "dzeta_05int")
    h = _scal(h, L, "h")

    bvf = _G / _ROW0 * (row[:, 1:M + 1] - row[:, 0:M]) / (h[:, None] * ddz[:, 0:M])
    ml = jnp.argmax(bvf, axis=-1) + 1                       # (L,) int32, 1-based
    bvf_max = _take(bvf, ml - 1)
    max_n = jnp.sqrt(jnp.maximum(bvf_max, 0.0))
    # `if ((row(M) - row(1))/row0 < 1.E-3) ml = M` -- after maxN.
    ml = jnp.where((row[:, M] - row[:, 0]) / _ROW0 < 1.0e-3, M, ml)
    H_mixed_layer = _take(dzeta_05int, ml - 1) * h
    return ml, H_mixed_layer, max_n


# ---------------------------------------------------------------------------
# TURB_SCALES
# ---------------------------------------------------------------------------
def turb_scales(row, ddz, ddz2, dzeta_int, dzeta_05int, h1, M, tw2, sal2,
                u2, v2, lamw, lamsal, k_turb_T_flux, T_massflux,
                eflux0_kinem, turb_density_flux, Buoyancy0, tau, kor,
                rad_integr, area_int, Lx, Ly, vol, z_full, ddz05,
                DDENS_DTEMP0, DDENS_DSAL0):
    """Batched ``TURB_SCALES`` (phys_func.f90:914-1072).

    Signature and semantics are those of
    :func:`lakematrix.physics.turb.turb_scales`, with a leading lake axis on
    every array (see the module docstring for the shapes).  ``M`` is the
    number of water layers (a Python int -- it must stay static under ``jit``).

    ``ddz2``, ``dzeta_int``, ``k_turb_T_flux``, ``T_massflux`` and
    ``turb_density_flux`` are accepted for signature parity but not read (the
    reference does not read them either: the ``k_turb_T_flux`` based
    mixed-layer depth is commented out in the Fortran).

    Returns a ``dict`` with ``i_maxN`` (1-based, ``(L,)`` int32),
    ``H_mixed_layer``, ``maxN``, ``w_conv_scale``, ``T_conv_scale``,
    ``Wedderburn``, ``LakeNumber``, ``Rossby_rad``, ``ThermThick``,
    ``ReTherm``, ``RiTherm`` as ``(L,)`` arrays and ``Rp``, ``Rpdens`` as
    ``(L, M)`` arrays.
    """
    row = _as2(row, M + 1, "row")
    L = int(row.shape[0])
    ddz = _fit(_as2(ddz, M, "ddz"), L, M, "ddz")
    dzeta_05int = _fit(_as2(dzeta_05int, M, "dzeta_05int"), L, M, "dzeta_05int")
    h1 = _scal(h1, L, "h1")
    tw2 = _fit(_as2(tw2, M + 1, "tw2"), L, M + 1, "tw2")
    sal2 = _fit(_as2(sal2, M + 1, "sal2"), L, M + 1, "sal2")
    u2 = _fit(_as2(u2, M + 1, "u2"), L, M + 1, "u2")
    v2 = _fit(_as2(v2, M + 1, "v2"), L, M + 1, "v2")
    lamw = _fit(_as2(lamw, M, "lamw"), L, M, "lamw")
    lam_a = _as2(lamsal, M, "lamsal", n_ok=(M, M + 1))
    # the reference loops over i = 1:M, so a padded (M+1) array is sliced
    lamsal = _fit(lam_a, L, lam_a.shape[-1], "lamsal")[:, :M]
    eflux0_kinem = _scal(eflux0_kinem, L, "eflux0_kinem")
    Buoyancy0 = _scal(Buoyancy0, L, "Buoyancy0")
    tau = _scal(tau, L, "tau")
    kor = _scal(kor, L, "kor")
    rad_integr = _fit(_as2(rad_integr, M + 1, "rad_integr"), L, M + 1,
                      "rad_integr")
    area_int = _fit(_as2(area_int, M + 1, "area_int"), L, M + 1, "area_int")
    vol = _scal(vol, L, "vol")
    z_full = _fit(_as2(z_full, M + 1, "z_full"), L, M + 1, "z_full")
    ddz05 = _fit(_as2(ddz05, M + 1, "ddz05"), L, M + 1, "ddz05")
    DDENS_DTEMP0 = _scal(DDENS_DTEMP0, L, "DDENS_DTEMP0")
    DDENS_DSAL0 = _scal(DDENS_DSAL0, L, "DDENS_DSAL0")
    # Surface length scale: `max(bathymwater(1)%Lx, bathymwater(1)%Ly)`.
    Lxy = jnp.maximum(_surf2(Lx, L, M, "Lx"), _surf2(Ly, L, M, "Ly"))

    i_maxN, H_mixed_layer, maxN = mixed_layer_calc(row, ddz, dzeta_05int, h1, M)

    # w_conv_scale: `integr` keeps the Fortran bound 0, so it is indexed with
    # the 1-based `i_maxN` -- exactly like the reference.
    integr_ml = _take(rad_integr, i_maxN)
    buoy = (Buoyancy0 - (rad_integr[:, 1] - integr_ml) * _G_D_KELVIN0
            / C.cw_m_row0)
    w_conv_scale = jnp.maximum(buoy * H_mixed_layer, 0.0) ** (1.0 / 3.0)
    T_conv_scale = -eflux0_kinem / (w_conv_scale + _SMALL)

    Wedderburn = jnp.clip(_G * (row[:, M] - row[:, 0]) * H_mixed_layer
                          * H_mixed_layer
                          / ((tau + _SMALL) * Lxy), 1.0e-2, 1.0e2)

    # Depth of the volume centre (zv) and of the centre of mass (zm).
    y = ddz05 * h1[:, None] * area_int
    x = y * z_full
    zm = _ordered_sum(row * x) / _ordered_sum(row * y)
    zv = _ordered_sum(x) / _ordered_sum(y)
    LakeNumber = ((zm - zv) * vol * _ROW0 * _G * 2.0 * H_mixed_layer
                  / (zv * (tau + _SMALL) * area_int[:, 0] * Lxy))
    LakeNumber = jnp.clip(LakeNumber, 1.0e-2, 1.0e2)
    Rossby_rad = (jnp.sqrt(_G * jnp.maximum(row[:, M] - row[:, 0], _SMALL)
                           / _ROW0 * H_mixed_layer)
                  / jnp.maximum(kor, _SMALL))

    # Thermocline thickness: last 14 degC / 8 degC crossing wins (no `exit`).
    i14 = _last_match((tw2[:, :-1] >= TTHERM0) & (tw2[:, 1:] < TTHERM0))
    i8 = _last_match((tw2[:, :-1] >= TTHERM1) & (tw2[:, 1:] < TTHERM1))
    ddzy = ddz * h1[:, None]                       # y = ddz(i)*h1
    tw_lo, tw_hi = _pair(tw2, i14)
    z_lo = _take(z_full[:, :M], i14)
    y14 = _take(ddzy, i14)
    x1_cross = z_lo + (tw_lo - TTHERM0) * y14 / (tw_lo - tw_hi)
    x1_frac = (x1_cross - z_lo) / y14
    tw_lo8, tw_hi8 = _pair(tw2, i8)
    z_lo8 = _take(z_full[:, :M], i8)
    y8 = _take(ddzy, i8)
    x2_cross = z_lo8 + (tw_lo8 - TTHERM1) * y8 / (tw_lo8 - tw_hi8)
    x2_frac = (x2_cross - z_lo8) / y8

    found14 = i14 >= 0
    found8 = i8 >= 0
    u2_lo, u2_hi = _pair(u2, i14)
    v2_lo, v2_hi = _pair(v2, i14)
    sal_lo, sal_hi = _pair(sal2, i14)
    u8_lo, u8_hi = _pair(u2, i8)
    v8_lo, v8_hi = _pair(v2, i8)
    sal8_lo, sal8_hi = _pair(sal2, i8)

    x1 = jnp.where(found14, x1_cross, 0.0)
    u1 = jnp.where(found14, u2_lo + x1_frac * (u2_hi - u2_lo), 0.0)
    v1 = jnp.where(found14, v2_lo + x1_frac * (v2_hi - v2_lo), 0.0)
    Sal1 = jnp.where(found14, sal_lo + x1_frac * (sal_hi - sal_lo), 0.0)
    x2 = jnp.where(found8, x2_cross, h1)           # default x2 = h1
    u2t = jnp.where(found8, u8_lo + x2_frac * (u8_hi - u8_lo), 0.0)
    v2t = jnp.where(found8, v8_lo + x2_frac * (v8_hi - v8_lo), 0.0)
    Sal2t = jnp.where(found8, sal8_lo + x2_frac * (sal8_hi - sal8_lo), 0.0)

    ThermThick = x2 - x1
    ReTherm = jnp.sqrt((u2t - u1) ** 2 + (v2t - v1) ** 2) * ThermThick / _NIU_WAT
    RiTherm = (_G / _ROW0
               * (DDENS_DTEMP0 * (TTHERM1 - TTHERM0)
                  + DDENS_DSAL0 * (Sal2t - Sal1))
               * ThermThick
               / ((u2t - u1) ** 2 + (v2t - v1) ** 2 + _SMALL))

    # Rp: ratio of density increments by salinity and temperature,
    # Rpdens: ratio of the density fluxes.
    Tm = 0.5 * (tw2[:, 1:] + tw2[:, :-1])
    Sm = 0.5 * (sal2[:, 1:] + sal2[:, :-1])
    x1 = _ddens_dtemp(Tm, Sm)
    x2 = _ddens_dsal(Tm, Sm)
    x = _bound_from_below(tw2[:, 1:] - tw2[:, :-1], _SMALL)
    Rp = -x2 / x1 * (sal2[:, 1:] - sal2[:, :-1]) / x
    Rpdens = lamsal / lamw * Rp

    return dict(i_maxN=i_maxN, H_mixed_layer=H_mixed_layer, maxN=maxN,
                w_conv_scale=w_conv_scale, T_conv_scale=T_conv_scale,
                Wedderburn=Wedderburn, LakeNumber=LakeNumber,
                Rossby_rad=Rossby_rad, ThermThick=ThermThick,
                ReTherm=ReTherm, RiTherm=RiTherm, Rp=Rp, Rpdens=Rpdens)


#: ``jax.jit`` aliases; ``M`` (arg 4 of ``mixed_layer_calc``, arg 6 of
#: ``turb_scales``) is static.
jitted_mixed_layer_calc = jax.jit(mixed_layer_calc, static_argnums=(4,))
jitted_turb_scales = jax.jit(turb_scales, static_argnums=(6,))


def make_turb_scales_kernel(M):
    """Return a jitted ``turb_scales`` with ``M`` bound (and dropped from the
    positional signature)."""
    def kernel(row, ddz, ddz2, dzeta_int, dzeta_05int, h1, tw2, sal2, u2, v2,
               lamw, lamsal, k_turb_T_flux, T_massflux, eflux0_kinem,
               turb_density_flux, Buoyancy0, tau, kor, rad_integr, area_int,
               Lx, Ly, vol, z_full, ddz05, DDENS_DTEMP0, DDENS_DSAL0):
        return turb_scales(row, ddz, ddz2, dzeta_int, dzeta_05int, h1, M, tw2,
                           sal2, u2, v2, lamw, lamsal, k_turb_T_flux,
                           T_massflux, eflux0_kinem, turb_density_flux,
                           Buoyancy0, tau, kor, rad_integr, area_int, Lx, Ly,
                           vol, z_full, ddz05, DDENS_DTEMP0, DDENS_DSAL0)

    return jax.jit(kernel)
