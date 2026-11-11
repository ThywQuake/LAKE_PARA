"""JAX-vectorised case-1 T_DIFF (water + soil), batch over lakes.

This is the first B1 kernel of the M5 port.  It mirrors
``physics.tsolver.diff_coef_water`` / ``diff_coef_soil`` / ``t_diff_water``
with a leading lake axis and is JIT-compiled.  The NumPy implementation stays
the reference and is used by the existing driver.
"""

import jax
import jax.numpy as jnp

from ..constants import C
from ..physics.tridiag import progonka_jax


def _pad1(x):
    shape = x.shape[:-1] + (1,)
    return jnp.concatenate((jnp.zeros(shape, dtype=x.dtype), x), axis=-1)


def diff_coef_soil(a, b, c, d, n0, n1, m0, m1, dt, lamsoil, csoil, rosoil,
                   dzs, Tsoil1, isoilcol):
    """Vectorised ``DIFF_COEF`` soil case; leading axis is the lake."""
    dt_inv = 1.0 / dt
    i = jnp.arange(m0, m1 + 1)
    j = i - m0 + n0
    lam2 = 0.5 * (lamsoil[..., j] + lamsoil[..., j + 1])
    lam1 = 0.5 * (lamsoil[..., j - 1] + lamsoil[..., j])
    dzmean = 0.5 * (dzs[..., j - 1] + dzs[..., j])
    ai = -0.5 * lam1 / dzs[..., j - 1]
    bi = -0.5 * lam2 / dzs[..., j]
    a = a.at[..., i].set(ai)
    b = b.at[..., i].set(bi)
    c = c.at[..., i].set(ai + bi - dt_inv * (csoil[..., j] * rosoil[..., j] * dzmean))
    d = d.at[..., i].set(
        -Tsoil1[..., j, isoilcol] * dt_inv * (csoil[..., j] * rosoil[..., j] * dzmean)
        + ai * Tsoil1[..., j - 1, isoilcol]
        - (ai + bi) * Tsoil1[..., j, isoilcol]
        + bi * Tsoil1[..., j + 1, isoilcol])
    return a, b, c, d


def diff_coef_water(a, b, c, d, n0, n1, m0, m1, dt, Tw1, lamw, area_int,
                    area_half, ddz, h1, dhw, dhw0, PEMF, pt_down_f,
                    rad_integr, lsh_water, dzeta_int, ddz05):
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    cw = C.cw_m_row0
    h1 = jnp.atleast_1d(h1)
    dhw = jnp.atleast_1d(dhw)
    dhw0 = jnp.atleast_1d(dhw0)
    i = jnp.arange(m0, m1 + 1)
    j = i - m0 + n0
    aa = 0.5 * (-area_half[..., j - 1] * lamw[..., j - 1]
                / (ddz[..., j - 1] * h1[:, None] * area_int[..., j])
                + cw * dhw[:, None] * dzeta_int[..., j] * dt_inv05
                - cw * dhw0[:, None] * dt_inv05)
    a = a.at[..., i].set(aa + cw * 0.5 * PEMF[..., j - 1])
    bb = 0.5 * (-area_half[..., j] * lamw[..., j]
                / (ddz[..., j] * h1[:, None] * area_int[..., j])
                - cw * dhw[:, None] * dzeta_int[..., j] * dt_inv05
                + cw * dhw0[:, None] * dt_inv05)
    b = b.at[..., i].set(bb - cw * 0.5 * PEMF[..., j])
    c = c.at[..., i].set(aa + bb - dt_inv * ddz05[..., j - 1] * h1[:, None] * cw
                         + cw * 0.5 * (PEMF[..., j] - PEMF[..., j - 1]))
    d = d.at[..., i].set(
        -Tw1[..., j] * dt_inv * ddz05[..., j - 1] * h1[:, None] * cw
        + (area_half[..., j] * rad_integr[..., j]
           - area_half[..., j - 1] * rad_integr[..., j - 1]) / area_int[..., j]
        - cw * PEMF[..., j - 1] * pt_down_f[..., j - 1]
        + cw * PEMF[..., j] * pt_down_f[..., j]
        + aa * Tw1[..., j - 1] + bb * Tw1[..., j + 1]
        - (aa + bb) * Tw1[..., j]
        - lsh_water[..., j] * ddz05[..., j - 1] * h1[:, None])
    return a, b, c, d


def t_diff_water(Tsurf, dt, Tw1, Tsoil1, lamw, area_int, area_half, h1, ddz,
                 dhw, dhw0, PEMF, pt_down_f, lamsoil, csoil, rosoil, dzs,
                 rad_integr, dzeta_int, ddz05, M, ns, isoilcol,
                 soilbotflx=0.0, lsh_water=None):
    """Batched case-1 ``T_DIFF``; all arrays have a leading lake axis (1-based)."""
    Tsurf = jnp.atleast_1d(Tsurf)
    h1 = jnp.atleast_1d(h1)
    dhw = jnp.atleast_1d(dhw)
    dhw0 = jnp.atleast_1d(dhw0)
    L = Tsurf.shape[0]
    nn = M + ns
    a = jnp.zeros((L, nn + 1))
    b = jnp.zeros((L, nn + 1))
    c = jnp.zeros((L, nn + 1))
    d = jnp.zeros((L, nn + 1))
    if lsh_water is None:
        lsh_water = jnp.zeros((L, nn + 1))
    b = b.at[:, 1].set(0.0)
    c = c.at[:, 1].set(1.0)
    d = d.at[:, 1].set(Tsurf)
    a, b, c, d = diff_coef_water(
        a, b, c, d, 2, M, 2, M, dt, Tw1, lamw, area_int, area_half, ddz,
        h1, dhw, dhw0, PEMF, pt_down_f, rad_integr, lsh_water, dzeta_int,
        ddz05)
    cw = C.cw_m_row0
    dt_inv = 1.0 / dt
    xx = area_int[:, M + 1] / area_half[:, M]
    a = a.at[:, M + 1].set(
        0.5 * (-lamw[:, M] / (ddz[:, M] * h1)
               + xx * 0.5 * dt_inv * cw * (dhw - dhw0))
        + xx * 0.5 * cw * PEMF[:, M])
    b = b.at[:, M + 1].set(-0.25 * (lamsoil[:, 1] + lamsoil[:, 2]) / dzs[:, 1])
    cc = 0.5 * dt_inv * (csoil[:, 1] * rosoil[:, 1] * dzs[:, 1]
                         + xx * cw * ddz[:, M] * h1)
    c = c.at[:, M + 1].set(a[:, M + 1] + b[:, M + 1] - cc - xx * cw * PEMF[:, M])
    d = d.at[:, M + 1].set(
        -Tw1[:, M + 1] * cc - rad_integr[:, M]
        - xx * cw * PEMF[:, M] * pt_down_f[:, M]
        + a[:, M + 1] * Tw1[:, M]
        - (a[:, M + 1] + b[:, M + 1]) * Tw1[:, M + 1]
        + b[:, M + 1] * Tsoil1[:, 2, isoilcol])
    a, b, c, d = diff_coef_soil(a, b, c, d, 2, ns - 1, M + 2, M + ns - 1,
                                dt, lamsoil, csoil, rosoil, dzs, Tsoil1,
                                isoilcol)
    c = c.at[:, M + ns].set(1.0)
    a = a.at[:, M + ns].set(1.0)
    d = d.at[:, M + ns].set(-soilbotflx * dzs[:, ns - 1]
                            / (lamsoil[:, ns - 1] + lamsoil[:, ns]))
    sol = progonka_jax(a[:, 1:nn + 1], b[:, 1:nn + 1],
                       c[:, 1:nn + 1], d[:, 1:nn + 1])
    Temp = jnp.concatenate((jnp.zeros((L, 1)), sol), axis=-1)
    return Temp[:, 1:M + 2], Temp[:, M + 1:M + ns + 1]


jitted_t_diff_water = jax.jit(t_diff_water, static_argnames=("M", "ns", "isoilcol"))
