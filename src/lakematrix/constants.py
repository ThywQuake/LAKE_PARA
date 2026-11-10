"""Physical and numeric constants of LAKE 3.2 (thermodynamic core).

Values are copied from `lakebatch/core/source/model/lake_modules.f90`
(`DEFINE_PHYS_CONSTANTS`, `DEFINE_PHYS_PARAMETERS`, `NUMERIC_PARAMS`).

**Float precision matters here.**  Most assignments in `DEFINE_PHYS_CONSTANTS`
use default-real (single precision) literals such as `g = 9.814` or
`Kelvin0 = 273.15`, which Fortran first rounds to `real(4)` and then stores in
the `real(8)` variable.  The Python value must therefore be
`float(np.float32(literal))`, not the exact decimal.  This was found by the
probe: `Kelvin0` differs from 273.15 by 6.1e-6 K, `g` by 1.3e-7 m/s2, and the
surface coefficients `aMagw`/`bMagw`/`emissivity` are float32 values.  Only
literals written with a `d` exponent (`d-5`, `d+23`, ...) or integer-valued
reals are exact in float64.
"""

from dataclasses import dataclass

import numpy as np


def _s(x):
    """Value of a default-real Fortran literal assigned to a real(8) variable."""
    return float(np.float32(x))


@dataclass(frozen=True)
class Constants:
    # PHYS_CONSTANTS, lake_modules.f90:195-309
    g: float = _s(9.814)
    omega: float = 7.29e-5          # 7.29d-5, double
    sigma: float = 5.67e-8          # 5.67d-8, double
    roa0: float = _s(1.273)
    row0: float = 1000.0
    roi: float = 917.0
    rofrsn: float = 100.0
    rosn_dry_ref: float = 100.0
    cw: float = 3990.0
    ci: float = 2150.0
    cp: float = 1005.0
    csnow: float = 2100.0
    lamair: float = _s(0.024)
    lami: float = _s(2.2)
    lamw0: float = _s(0.561)
    # lamw0/(cw*row0)*1.E-2: the variables are double, the 1.E-2 literal is single
    lamsal0: float = _s(0.561) / (3990.0 * 1000.0) * _s(1.0e-2)
    niu_atm: float = 1.8e-5         # 1.8d-5, double
    niu_wat: float = 1.307e-6       # 1.307d-6, double
    surf_tension_wat: float = 72.0e-3  # 72.d-3, double
    Lwi: float = 333500.0
    Lwv: float = 2501000.0
    Liv: float = 2834600.0
    kappa: float = _s(0.4)
    Rd: float = _s(287.05)
    Rwv: float = _s(461.91)
    R_univ: float = _s(8.31441)
    Avogadro_number: float = 6.022141e23   # 6.022141d+23, double
    Planck_const: float = 6.62607e-34      # 6.62607d-34, double
    clight: float = 3.0e8                  # 3.d+8, double
    lambda_PAR0: float = 550.0e-9          # 550.d-9, double
    short2PAR: float = _s(0.48)
    solar_constant: float = _s(1367.0)
    aMagw: float = _s(7.6326)
    bMagw: float = _s(241.9)
    aMagi: float = _s(9.5)
    bMagi: float = _s(265.5)
    albedoofwater: float = _s(0.06)
    albedoofice: float = _s(0.40)
    albedoofsnow: float = _s(0.85)
    albedoofsoil: float = _s(0.3)
    albedoofwater_lw: float = _s(0.02)
    albedoofice_lw: float = _s(0.03)
    albedoofsnow_lw: float = _s(0.01)
    albedoofsoil_lw: float = 0.0
    emissivityofwater: float = _s(0.98)
    emissivityofice: float = _s(0.97)
    emissivityofsnow: float = _s(0.99)
    emissivityofsoil: float = _s(0.9)
    roughness0: float = 1.0e-4             # 1.d-4, double
    charnock_const: float = _s(0.0144)
    Kelvin0: float = _s(273.15)
    pref: float = 1.0e5                    # 1.d+5, double
    esatsurf0: float = _s(610.7)
    sabs: float = 0.0
    sabs0: float = 0.0
    alsal: float = 1.0
    almeth: float = 1.0
    aloxyg: float = 1.0
    alcarbdi: float = 1.0
    salice0: float = 0.0
    poricebot: float = _s(0.05)
    asalice: float = float(np.float32(1.0) / (np.float32(86400.0) * np.float32(365.0)))
    snmeltwat: float = 0.0
    icetopfr: float = 1.0
    z0_bot: float = 1.3e-4                 # 1.3d-4, double
    clin_bot: float = _s(4.0e-3)
    cquad_bot: float = _s(8.0e-2)
    Racr: float = _s(657.511)
    k_PA: float = _s(1.7)

    # PHYS_PARAMETERS, lake_modules.f90:110-128
    whc: float = _s(0.04)
    hcond: float = _s(0.01)
    sncr: float = _s(0.01)

    # NUMERIC_PARAMS, lake_modules.f90:68-90
    dznorm: float = _s(0.05)
    dzmin: float = _s(0.02)
    UpperLayer: float = _s(0.02)
    pi: float = float(np.pi)
    small_value: float = 1.0e-20           # 1.d-20, double
    min_ice_thick: float = _s(0.05)
    min_water_thick: float = _s(0.05)
    T_phase_threshold: float = 1.0e-5      # 1.d-5, double
    minwind: float = _s(1.0e-2)
    hour_sec: float = 3600.0
    day_sec: float = 86400.0

    # Derived constants, DEFINE_PHYS_CONSTANTS
    @property
    def ci_m_roi(self):
        return self.ci * self.roi

    @property
    def cw_m_row0(self):
        return self.cw * self.row0

    @property
    def row0_m_Lwi(self):
        return self.row0 * self.Lwi

    @property
    def roi_m_Lwi(self):
        return self.roi * self.Lwi

    @property
    def row0_d_roi(self):
        return self.row0 / self.roi

    @property
    def roi_d_row0(self):
        return self.roi / self.row0

    @property
    def roa0_d_row0(self):
        return self.roa0 / self.row0

    @property
    def roa0_d_roi(self):
        return self.roa0 / self.roi

    @property
    def row0_d_roa0(self):
        return self.row0 / self.roa0

    @property
    def Rd_d_cp(self):
        return self.Rd / self.cp

    @property
    def Rd_d_Rwv(self):
        return self.Rd / self.Rwv


C = Constants()
