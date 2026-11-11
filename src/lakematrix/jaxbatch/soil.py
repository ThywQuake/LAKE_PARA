"""JAX (batched) soil column helpers used by case 1."""

import jax.numpy as jnp

from ..constants import C
from ..physics import soil as S


def soil_cond_johansen(wl, wi, rosdry, por):
    water_vol = wl / (por * (wl + C.row0 / C.roi * wi + C.row0 / rosdry))
    ice_vol = wi / (por * (wi + C.roi / C.row0 * wl + C.roi / rosdry))
    waterice = water_vol + ice_vol
    kersten = S._CK_CONST * waterice / (1.0 + (S._CK_CONST - 1.0) * waterice)
    water_sat = por * water_vol / waterice
    ice_sat = por * ice_vol / waterice
    lambda_sat = (S._LAMBDA_SOLIDS ** (1.0 - por)
                  * C.lamw0 ** water_sat * C.lami ** ice_sat)
    lambda_dry = S._CK_CONST1 * 10.0 ** (-S._CK_CONST2 * por)
    return (lambda_sat - lambda_dry) * kersten + lambda_dry


def wl_max(por, rosdry, wi):
    return por * C.row0 / (rosdry * (1.0 - por)) - wi * C.row0 / C.roi


def soil_cond_heat_coef(wl1, wi1, POR, rosdry, dzs, BH, DLMAX, FLWMAX,
                        ns, is_):
    """Batched ``SOIL_COND_HEAT_COEF`` for column ``is_``.

    Returns ``(csoil, rosoil, lamsoil, lammoist, filtr, wsoil)`` with shape
    ``(L, ns+1)`` (index 0 is the Fortran pad).  ``is_`` is a Python int.
    """
    L = wl1.shape[0]
    cr = float(jnp.float32(0.2)) * 4180.0
    cimh = C.ci
    wl = wl1[:, :, is_]
    wi = wi1[:, :, is_]
    i = jnp.arange(1, ns + 1)
    wl_i = wl[:, i]
    wi_i = wi[:, i]
    csoil = jnp.zeros((L, ns + 1))
    rosoil = jnp.zeros((L, ns + 1))
    lamsoil = jnp.zeros((L, ns + 1))
    lammoist = jnp.zeros((L, ns + 1))
    filtr = jnp.zeros((L, ns + 1))
    wsoil = jnp.zeros((L, ns + 1))
    csoil = csoil.at[:, i].set(cr + wl_i * C.cw + wi_i * cimh)
    rosoil = rosoil.at[:, i].set(rosdry[None, i] * (1.0 - POR[None, i])
                                 * (1.0 + wi_i + wl_i))
    lamsoil = lamsoil.at[:, i].set(
        soil_cond_johansen(wl_i, wi_i, rosdry[None, i], POR[None, i]))
    wlmax = wl_max(POR[None, i], rosdry[None, i], wi_i)
    arg = jnp.maximum(wl_i / wlmax, S._ARG_MIN)
    lammoist = lammoist.at[:, i].set(
        DLMAX[None, i] * arg ** (BH[None, i] + 2.0))
    # inter-layer percolation / vapour diffusivity
    j = jnp.arange(1, ns)
    wlmax_j = wl_max(0.5 * (POR[None, j] + POR[None, j + 1]),
                     0.5 * (rosdry[None, j] + rosdry[None, j + 1]),
                     0.5 * (wi[:, j] + wi[:, j + 1]))
    wlmax_i = wl_max(POR[None, j + 1], rosdry[None, j + 1], wi[:, j + 1])
    fil = jnp.where(
        wlmax_i < S._S001, 0.0,
        0.5 * (FLWMAX[None, j + 1] + FLWMAX[None, j])
        * (0.5 * (wl[:, j + 1] + wl[:, j]) / wlmax_j)
        ** (BH[None, j + 1] + BH[None, j] + 3.0))
    filtr = filtr.at[:, j].set(fil)
    lam1 = 0.5 * (lammoist[:, j] + lammoist[:, j + 1])
    wsoil = wsoil.at[:, j].set(
        (-lam1 * (wl[:, j + 1] - wl[:, j]) / dzs[None, j] + filtr[:, j])
        * 0.5 * (rosdry[None, j] + rosdry[None, j + 1])
        * (1.0 - 0.5 * (POR[None, j] + POR[None, j + 1])) / C.row0)
    return csoil, rosoil, lamsoil, lammoist, filtr, wsoil
