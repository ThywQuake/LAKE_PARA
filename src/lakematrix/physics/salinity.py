"""Salinity diffusion (`S_DIFF`, water branch) for fresh / low-salinity water.

Fortran: `salinity_mod.f90:14-240` and the `water_salinity_indic` case of
`DIFF_COEF` (`t_solver_mod.f90:828-840`).  v1 is fresh-water: the reference
testlake has `Sal1 = 0`, so the solve is a no-op, but the full water branch is
transcribed for generality.
"""

import numpy as np

from .tridiag import progonka


def diff_coef_salinity(a, b, c, d, n0, n1, m0, m1, dt, Sal1, lamsal,
                       area_int, area_half, ddz, h1, dhw, dhw0,
                       dzeta_int, ddz05):
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        ai = 0.5 * (-area_half[j - 1] * lamsal[j - 1] / (ddz[j - 1] * h1 * area_int[j])
                    + dhw * dzeta_int[j] * dt_inv05 - dhw0 * dt_inv05)
        bi = 0.5 * (-area_half[j] * lamsal[j] / (ddz[j] * h1 * area_int[j])
                    - dhw * dzeta_int[j] * dt_inv05 + dhw0 * dt_inv05)
        a[i] = ai
        b[i] = bi
        c[i] = ai + bi - dt_inv * ddz05[j - 1] * h1
        d[i] = (-Sal1[j] * dt_inv * ddz05[j - 1] * h1
                + ai * Sal1[j - 1] + bi * Sal1[j + 1]
                - (ai + bi) * Sal1[j])


def s_diff(Sal1, dt, dhw, dhw0, dhwp, dhwe, lamsal, area_int, area_half,
           ddz, h1, dzeta_int, ddz05, M, Sflux0=0.0):
    """Water-column salinity after one `S_DIFF` (no ice/soil salinity)."""
    a = np.zeros(M + 2); b = np.zeros(M + 2)
    c = np.zeros(M + 2); d = np.zeros(M + 2)
    Sflux0 = Sflux0 + (dhwp + dhwe) / dt * (0.0 - Sal1[1])
    diff_coef_salinity(a, b, c, d, 2, M, 2, M, dt, Sal1, lamsal, area_int,
                       area_half, ddz, h1, dhw, dhw0, dzeta_int, ddz05)
    x = 0.5 * (-area_half[1] * lamsal[1] / (h1 * ddz[1] * area_int[1])
               + dhw0 / (2.0 * dt))
    c[1] = x - ddz[1] * h1 / (2.0 * dt)
    b[1] = x
    d[1] = (-ddz[1] * h1 * Sal1[1] / (2.0 * dt) - Sflux0
            + x * (Sal1[2] - Sal1[1]))
    x = 0.5 * (-area_half[M] * lamsal[M] / (ddz[M] * h1 * area_int[M + 1])
               + (dhw - dhw0) / (2.0 * dt))
    c[M + 1] = x - ddz[M] * h1 / (2.0 * dt)
    a[M + 1] = x
    d[M + 1] = (-Sal1[M + 1] * ddz[M] * h1 / (2.0 * dt)
                - x * (Sal1[M + 1] - Sal1[M]))
    Sol = progonka(a[1:M + 2], b[1:M + 2], c[1:M + 2], d[1:M + 2])
    Sol = np.maximum(Sol, 0.0)
    out = np.zeros(M + 2)
    out[1:] = Sol
    return out
