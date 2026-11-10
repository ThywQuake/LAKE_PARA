"""Shortwave radiation in water, ice and snow (`RAD_UPDATE`).

Fortran: `lake_modules.f90:386-403`.  The Beer-Lambert integration is

    flux(0, b) = flux0(b)
    flux(i, b) = flux(i-1, b) * exp(-extinct(i, b) * thickness(i-1))
    integr(i)  = sum_b flux(i, b)

Fortran arrays are 1-based for `extinct` (1..nlevels) and 0-based for
`flux`/`integr` (0..nlevels).  Here `extinct` is `(..., nlevels, nbands)`,
`thickness` `(..., nlevels)`, `flux0` `(..., nbands)`; the returned
`flux`/`integr` are `(..., nlevels+1, nbands)` and `(..., nlevels+1)`.
"""

import jax.numpy as jnp


NBANDS = 4
EXTWATBANDS = jnp.array([3.13, 0.81, 1.73, 1.087e3])
FRACBANDS = jnp.array([0.046, 0.429, 0.208, 0.317])


def rad_update(extinct, thickness, flux0):
    extinct = jnp.asarray(extinct)
    thickness = jnp.asarray(thickness)
    flux0 = jnp.asarray(flux0)
    nlevels = extinct.shape[-2]
    flux = jnp.zeros(extinct.shape[:-2] + (nlevels + 1, extinct.shape[-1]), dtype=extinct.dtype)
    flux = flux.at[..., 0, :].set(flux0)
    for i in range(nlevels):
        flux = flux.at[..., i + 1, :].set(
            flux[..., i, :] * jnp.exp(-extinct[..., i, :] * thickness[..., i, None]))
    integr = flux.sum(axis=-1)
    return flux, integr
