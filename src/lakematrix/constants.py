"""Physical and numeric constants of LAKE 3.2 (thermodynamic core).

Values are copied from `lakebatch/core/source/model/lake_modules.f90`
(`DEFINE_PHYS_CONSTANTS`, `DEFINE_PHYS_PARAMETERS`, `NUMERIC_PARAMS`).  Keep
this file as the single place where a Fortran constant is repeated; when a
routine needs a new one, add it here with the Fortran name.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Constants:
    # PHYS_CONSTANTS, lake_modules.f90:195-309
    g: float = 9.814
    omega: float = 7.29e-5
    sigma: float = 5.67e-8
    roa0: float = 1.273
    row0: float = 1000.0
    roi: float = 917.0
    rofrsn: float = 100.0
    rosn_dry_ref: float = 100.0
    cw: float = 3990.0
    ci: float = 2150.0
    cp: float = 1005.0
    csnow: float = 2100.0
    lamair: float = 0.024
    lami: float = 2.2
    lamw0: float = 0.561
    lamsal0: float = 0.561 / (3990.0 * 1000.0) * 1.0e-2
    niu_atm: float = 1.8e-5
    niu_wat: float = 1.307e-6
    surf_tension_wat: float = 72.0e-3
    Lwi: float = 333500.0
    Lwv: float = 2501000.0
    Liv: float = 2834600.0
    kappa: float = 0.4
    Rd: float = 287.05
    Rwv: float = 461.91
    R_univ: float = 8.31441
    Avogadro_number: float = 6.022141e23
    Planck_const: float = 6.62607e-34
    clight: float = 3.0e8
    lambda_PAR0: float = 550.0e-9
    short2PAR: float = 0.48
    solar_constant: float = 1367.0
    aMagw: float = 7.6326
    bMagw: float = 241.9
    aMagi: float = 9.5
    bMagi: float = 265.5
    albedoofwater: float = 0.06
    albedoofice: float = 0.40
    albedoofsnow: float = 0.85
    albedoofsoil: float = 0.3
    albedoofwater_lw: float = 0.02
    albedoofice_lw: float = 0.03
    albedoofsnow_lw: float = 0.01
    albedoofsoil_lw: float = 0.0
    emissivityofwater: float = 0.98
    emissivityofice: float = 0.97
    emissivityofsnow: float = 0.99
    emissivityofsoil: float = 0.9
    roughness0: float = 1.0e-4
    charnock_const: float = 0.0144
    Kelvin0: float = 273.15
    pref: float = 1.0e5
    esatsurf0: float = 610.7
    sabs: float = 0.0
    sabs0: float = 0.0
    alsal: float = 1.0
    almeth: float = 1.0
    aloxyg: float = 1.0
    alcarbdi: float = 1.0
    salice0: float = 0.0
    poricebot: float = 0.05
    asalice: float = 1.0 / (86400.0 * 365.0)
    snmeltwat: float = 0.0
    icetopfr: float = 1.0
    z0_bot: float = 1.3e-4
    clin_bot: float = 4.0e-3
    cquad_bot: float = 8.0e-2
    Racr: float = 657.511
    k_PA: float = 1.7

    # PHYS_PARAMETERS, lake_modules.f90:110-128
    whc: float = 0.04
    hcond: float = 0.01
    sncr: float = 0.01

    # NUMERIC_PARAMS, lake_modules.f90:68-90
    dznorm: float = 0.05
    dzmin: float = 0.02
    UpperLayer: float = 0.02
    pi: float = float(np.pi)
    small_value: float = 1.0e-20
    min_ice_thick: float = 0.05
    min_water_thick: float = 0.05
    T_phase_threshold: float = 1.0e-5
    minwind: float = 1.0e-2
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
