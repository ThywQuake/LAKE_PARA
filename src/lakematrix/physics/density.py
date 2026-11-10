"""Water density (`DENSITY_W` and the `WATER_DENSITY` module).

Fortran: `phys_func.f90:160-201` (`DENSITY_W`) and the McCutcheon et al.
(1993) formula `WATER_DENS_TS` (phys_func.f90:283-322).  testlake uses
`eos 5`, `lindens 0`, so v1 implements that path; the other equations of state
raise a clear error instead of silently returning a different density.
"""

import numpy as np

from ..constants import C

# WATER_DENSITY module coefficients, phys_func.f90:8-40.
A11 = 288.9414
A12 = 508929.2
A13 = 68.12963
A14 = 3.9863
alpha1 = 8.2449e-1
alpha2 = 4.0899e-3
alpha3 = 7.6438e-5
alpha4 = 8.2467e-7
alpha5 = 5.3675e-9
beta1 = 5.724e-3
beta2 = 1.0227e-4
beta3 = 1.6546e-6
gamma1 = 4.8314e-4

EOS_HOSTETLER = 1
EOS_TEOS = 2
EOS_KIVU = 3
EOS_UNESCO = 4
EOS_MCCATCHEON = 5


def water_dens_ts(temp, sal, row0=C.row0):
    """McCutcheon et al. (1993) density, kg/m^3; temp in C, sal in kg/kg."""
    temp = np.asarray(temp, dtype=float)
    sal = np.asarray(sal, dtype=float)
    sg = sal * 1.0e3
    dens = row0 * (1.0 - (temp + A11) * (temp - A14) ** 2 / (A12 * (temp + A13)))
    a = alpha1 - alpha2 * temp + alpha3 * temp ** 2 - alpha4 * temp ** 3 + alpha5 * temp ** 4
    b = -beta1 + beta2 * temp - beta3 * temp ** 2
    c = gamma1
    return dens + a * sg + b * sg ** 1.5 + c * sg ** 2


def density_w(temp, sal, preswat, eos=EOS_MCCATCHEON, lindens=0):
    """Port of `DENSITY_W` for the supported configuration (eos 5, lindens 0)."""
    if lindens != 0:
        raise NotImplementedError("linear density (lindens=1) is not ported yet")
    if eos != EOS_MCCATCHEON:
        raise NotImplementedError(f"eos={eos} is not ported; v1 supports eos 5")
    return water_dens_ts(temp, sal)
