"""Shortwave radiation in water, ice and snow (`RAD_UPDATE`).

Fortran: `lake_modules.f90:386-403`.  The Beer-Lambert integration is

    flux(0, b) = flux0(b)
    flux(i, b) = flux(i-1, b) * exp(-extinct(i, b) * thickness(i-1))
    integr(i)  = sum_b flux(i, b)

The cumulative product is evaluated with NumPy, not JAX: this routine is a
pure per-lake array operation and calling JAX primitives in a Python loop made
it the dominant cost of a single-lake step (M5 measurement).  A JAX version can
be added for the batched runner without changing this API.
"""

import numpy as np

NBANDS = 4
EXTWATBANDS = np.array([3.13, 0.81, 1.73, 1.087e3])
FRACBANDS = np.array([0.046, 0.429, 0.208, 0.317])


def rad_update(extinct, thickness, flux0):
    extinct = np.asarray(extinct)
    thickness = np.asarray(thickness)
    flux0 = np.asarray(flux0)
    nlevels = extinct.shape[-2]
    att = np.exp(-extinct * thickness[..., None])
    flux = np.empty(extinct.shape[:-2] + (nlevels + 1, extinct.shape[-1]),
                    dtype=np.result_type(att, flux0))
    flux[..., 0, :] = flux0
    flux[..., 1:, :] = np.cumprod(att, axis=-2) * flux0[..., None, :]
    integr = flux.sum(axis=-1)
    return flux, integr
