"""LAKE 3.2 soil thermodynamics (``SOIL_LAKE_MOD``) and its ``PHYS_FUNC`` helpers.

Fortran source: ``lakebatch/core/source/model/soil_mod.f90`` (commit ``6657ebc``)
and ``phys_func.f90``.  This module is a line-by-line transcription of

    ====================  ==========================================
    Fortran               Python
    ====================  ==========================================
    COMSOILFORLAKE        ``soil_type_parameters`` / ``comsoilforlake``
    SOILFORLAKE           ``soilforlake``
    SOIL_COND_HEAT_COEF   ``soil_cond_heat_coef``
    SOILCOLSTEMP          ``soilcolstemp``
    TSURFSOILCOL          ``tsurfsoilcol``
    LATERHEAT             ``laterheat``
    OVERLAY               ``overlay``
    ====================  ==========================================

plus the ``PHYS_FUNC`` helpers they call: ``WL_MAX``, ``UNFRWAT``, ``WI_MAX``,
``SOIL_COND_JOHANSEN``, ``MELTPNT``/``MELTINGPOINT`` (and ``TEMPMHYDRDISS``,
which is only reached when methane hydrates are enabled).  ``W_SEDIM`` is *not*
transcribed: it belongs to the sedimentation module (``sedim 0`` here) and
``soil_mod`` never calls it.

Conventions
===========

**Indexing.**  Every array whose Fortran lower bound is 1 is stored *1-based*:
``a[0]`` is an unused pad and ``a[i]`` is Fortran ``a(i)``.  This keeps the
transcription auditable (``zsoil[i]``, ``wl1[i, is]``, ``bathymwater[i]`` read
like the source).  Arrays whose Fortran lower bound is 0 (``ddz05(0:M)``,
``ddzi05(0:Mice)``) stay 0-based, matching ``grid.py``.  Column indices ``is``
run over ``1 .. nsoilcols``; the side columns are ``1 .. nsoilcols-1`` and the
deepest column is ``nsoilcols``.

Because the Fortran arrays are module storage, a few branches leave entries
*unwritten* (e.g. ``b(1)``/``c(1)`` in ``SOILFORLAKE`` when there is no water,
ice or snow at all, or ``wl3(1)`` when neither of its two branches applies).
Those arrays live in :class:`SoilState`, so the same persistence is
reproduced; do not recreate ``SoilState`` between steps.

**Floating point.**  ``ireals`` is ``real(8)`` but most literals in the source
are default real (``real(4)``).  A default-real literal assigned to a
``real(8)`` variable is first rounded to ``float32``; therefore ``0.01`` becomes
``float(np.float32(0.01)) = 0.009999999776482582`` while ``1.``, ``0.5`` and
``10.5`` are exact.  The same holds for default-real *expressions*: ``3.50/100.``
is evaluated in ``float32`` before being widened.  Literals written with a ``d``
exponent (``1.d-2``, ``1.d+3``) or without a fractional part are double.  Every
such choice is commented where it occurs (``_s`` = single-precision literal,
``_sdiv`` = single-precision quotient, ``_S001`` = ``float32(0.01)``).
Constants already rounded in :mod:`lakematrix.constants` are reused from there
(``C.g``, ``C.lamw0``, ``C.UpperLayer``, ...).

**Target configuration** (``ref/v1/*/setup/testlake_setup.dat``):
``soiltype 5``, ``nsoilcols 5``, ``ns 10``, ``soil_depth 10.``,
``tricemethhydr 0.``, ``salsoil 0``, ``saltice 0``, ``sedim 0``,
``nmeltpoint 1``, ``soilbotflx 0.``, ``soilcolconjtype 2``.  With that setup
``Sals1 == 0`` (``INIT_VAR``), so the melting point is 0 C; ``saltice``/``sedim``
act in the ice/sediment modules, not here.

**Unreachable branches** (transcribed anyway, marked ``UNREACHABLE`` in the
code):

* ``tricemethhydr > 0`` — methane-hydrate latent heat in ``SOILFORLAKE``,
  ``cpmethhydr`` heat capacity in ``SOIL_COND_HEAT_COEF``, ``densmh`` density in
  ``WL_MAX``/``WI_MAX`` and ``TEMPMHYDRDISS``.
* ``bcswitch == 2`` in ``SOILCOLSTEMP`` — ``bcswitch`` is a compile-time
  ``parameter`` equal to 1; the branch also needs ``LOGFLUX``.
* ``soilcolconjtype == 1`` in ``SOILCOLSTEMP`` — the testlake setup uses 2.
* ``contr(2) == 1`` (methane in the soil columns) — calls ``METHANE`` from
  ``methane_mod.f90``, outside this module; raises ``NotImplementedError``
  instead of pretending.
* ``ifflux == .true.`` in the soil-only ``T_DIFF`` branch — ``SOILCOLSTEMP``
  always passes ``.false.``.
* ``nmeltpoint == 2`` (Jackett et al. 2004, ``FP_THETA``) — not ported.

**Call order matters.**  ``csoil``/``rosoil``/``lamsoil``/``lammoist``/``filtr``/
``wsoil`` are one-dimensional arrays *shared by all soil columns* (filled for one
column at a time).  ``lake.f90`` first calls ``SOIL_COND_HEAT_COEF(nsoilcols)``
(line 1055), then ``SOILCOLSTEMP`` overwrites them for ``is = 1 .. nsoilcols-1``
(line 927) and finally ``SOILFORLAKE`` for ``is = nsoilcols`` runs (line 1073) on
the coefficients left over from column ``nsoilcols-1``.  ``SOILFORLAKE`` also
expects ``Tsoil2`` to hold the *diffused* temperature of the same column
(computed by ``T_DIFF``, either inside ``SOILCOLSTEMP`` or by ``T_SOLVER`` with
``isoilcol = nsoilcols``).  Reproduce that order in the driver;
:func:`soilcolstemp` does it for the side columns.

**Batching.**  The functions are per-lake (a single ``lb_point_ctx`` point):
shapes are ``(ns, nsoilcols)`` and ``(M+1)`` without a leading lake axis.  A
batch wrapper loops over lakes.

**Validation** (M3 probe, step 1 of the reference case, ``dt = 20 s``): starting
from ``INITVAR`` and the probe bathymetry, :func:`soilcolstemp` reproduces the
Fortran ``LAKE_exit`` side-column temperatures **bit for bit** (max abs diff
0.0 over all 4 columns x 10 layers), and :func:`soil_cond_heat_coef` matches the
``T_DIFF`` probe ``csoil``/``rosoil``/``lamsoil`` exactly; ``zsoil``/``dzs``
match ``INITVAR``/``TD_in`` exactly.  The moisture/phase-change branches
(``wl3``/``wl4``/``wi2``/``dhwfsoil``) have **no probe output yet** — they are
transcribed but only checked for conservation, ``PROGONKA`` residual (1e-16) and
long-run stability (400 steps of freeze/melt without NaN, ``wl1`` inside
``[0, WL_MAX]``, ``wl4`` inside the pore capacity).

**Not ported / TODO.**  The methane part of ``SOILCOLSTEMP`` (``METHANE``
call), ``FP_THETA`` and the ``bcswitch == 2`` surface-flux boundary condition.
``SOILFORLAKE``'s diagnostic ``print`` block for ``Tsoil3 < -10`` is reproduced
behind ``debug=True``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..constants import C
from .tridiag import progonka

__all__ = [
    "Bathym",
    "LatHeatSource",
    "SoilGeometry",
    "SoilParams",
    "SoilState",
    "SoilTypeParameters",
    "bathym_levels",
    "comsoilforlake",
    "laterheat",
    "make_soil_state",
    "meltpnt",
    "meltingpoint",
    "overlay",
    "pad1",
    "soil_cond_heat_coef",
    "soil_cond_johansen",
    "soil_type_parameters",
    "soilcolstemp",
    "soilforlake",
    "step",
    "tempmhydrdiss",
    "tsurfsoilcol",
    "unfrwat",
    "wi_max",
    "wl_max",
]


# ---------------------------------------------------------------------------
# Precision helpers (see the module docstring)
# ---------------------------------------------------------------------------

def _s(x: float) -> float:
    """Value of a default-real (``real(4)``) Fortran literal stored in real(8)."""
    return float(np.float32(x))


def _sdiv(x: float, y: float) -> float:
    """A default-real Fortran quotient, e.g. ``3.50/100.``, rounded to float32."""
    return float(np.float32(x) / np.float32(y))


def _fdiv(x, y):
    """IEEE division with Fortran semantics: inf/nan instead of an exception."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float64(x) / np.float64(y)


def pad1(x):
    """Return ``x`` with a leading zero, for 0-based -> 1-based conversion.

    ``grid.py``/``bathym.py`` return 0-based arrays; this module indexes arrays
    the Fortran way, so pad them once (``Tw1 = pad1(probe_Tw1)``).  For
    ``ddz05``/``ddzi05`` (Fortran lower bound 0) do **not** pad.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,))
    out[..., 1:] = x
    return out


# ``0.01`` is a default-real literal everywhere it appears in the source.
_S001 = _s(0.01)
# ``1.d-2`` (double) used by ``ARG = max(ARG,1.d-2)``.
_ARG_MIN = 1.0e-2


# ---------------------------------------------------------------------------
# Constants that only live in METH_OXYG_CONSTANTS (lake_modules.f90:1560-1640,
# 1900-1980).  They matter only for the methane-hydrate branches.
# ---------------------------------------------------------------------------

_MOLMASS_CH4 = 16.0               # molmass_ch4 = 16.
_MOLMASS_H2O = 18.0               # molmass_h2o = 18.
_NCH4_D_NH2O = _sdiv(1.0, 5.75)   # nch4_d_nh2o = 1./5.75  (default real)
_ALPHAMH = _NCH4_D_NH2O * _MOLMASS_CH4 / _MOLMASS_H2O   # alphamh
_CPMETHHYDR = 2160.0              # cpmethhydr = 2160., J/(kg*K)
_DENSMH = 900.0                   # densmh = 9.d+2, kg/m3
_MOLMASS_METHHYDR = ((_MOLMASS_H2O + _MOLMASS_CH4 * _NCH4_D_NH2O)
                     / (1.0 + _NCH4_D_NH2O))
_METHHYDRDISS = 5.53e4 * 1.0e3 / _MOLMASS_METHHYDR   # J/kg

# MELTPNT constants (phys_func.f90:778-798).  dtdc is used by npar = 1; pnorm
# and Pa2dbar belong to the npar = 2 (Jackett) branch, which is not ported.
_MELTPNT_DTDC = _s(66.7)     # real(4) parameter dtdc = 66.7
_MELTPNT_PNORM = 101325.0    # parameter pnorm = 101325   (npar = 2)
_PA2DBAR = 1.0e-4            # parameter Pa2dbar = 1.d-4   (npar = 2)

# TEMPMHYDRDISS constants (phys_func.f90:1528-1545)
_TEMPMH_C1 = _s(3.79e-3)     # parameter c1 = 3.79E-3
_TEMPMH_C2 = _s(2.83e-4)     # parameter c2 = 2.83E-4


# ---------------------------------------------------------------------------
# COMSOILFORLAKE soil-type table (soil_mod.f90:100-182)
# ---------------------------------------------------------------------------

NUM_SOIL = 11  # NUMERIC_PARAMS::Num_Soil


@dataclass(frozen=True)
class SoilTypeParameters:
    """The 11-row soil parameter table of ``COMSOILFORLAKE``.

    Each array is 1-based (``arr[soiltype]``), so ``arr[5]`` is the "silt loam"
    row used by the reference cases.  Every entry is the float32 rounding of the
    default-real literal in the source; ``psi_max``, ``gamma_max`` and
    ``lambda_max`` are additionally single-precision quotients.
    """

    b: np.ndarray           # parameter b, dimensionless
    psi_max: np.ndarray     # saturated water potential, m
    porosity: np.ndarray    # porosity, m3/m3
    gamma_max: np.ndarray   # saturated hydraulic conductivity, m/s
    lambda_max: np.ndarray  # saturated water diffusivity
    W_0: np.ndarray         # max unfrozen water at 0 C, kg/kg
    W_m: np.ndarray         # max unfrozen water at T << 0 C, kg/kg


def soil_type_parameters() -> SoilTypeParameters:
    """Return the ``COMSOILFORLAKE`` table (soil_mod.f90:56-182)."""
    b = np.zeros(NUM_SOIL + 1)
    psi = np.zeros(NUM_SOIL + 1)
    por = np.zeros(NUM_SOIL + 1)
    gam = np.zeros(NUM_SOIL + 1)
    lam = np.zeros(NUM_SOIL + 1)
    w0 = np.zeros(NUM_SOIL + 1)
    wm = np.zeros(NUM_SOIL + 1)

    b[1:] = [_s(v) for v in (4.05, 4.38, 4.90, 5.39, 5.30, 7.12, 8.52, 7.75,
                             10.4, 10.4, 11.4)]
    psi[1:] = [_sdiv(v, 100.0) for v in (3.50, 1.78, 7.18, 14.6, 56.6, 8.63,
                                         36.1, 14.6, 6.16, 17.4, 18.6)]
    por[1:] = [_s(v) for v in (0.395, 0.410, 0.435, 0.451, 0.485, 0.420, 0.476,
                               0.477, 0.426, 0.492, 0.482)]
    gam[1:] = [_sdiv(v, 100.0) for v in (0.01760, 0.01560, 0.00340, 0.00069,
                                         0.00072, 0.00063, 0.00024, 0.00017,
                                         0.00021, 0.00010, 0.00013)]
    lam[1:] = [_sdiv(v, 10000.0) for v in (0.29800, 0.13900, 0.13700, 0.06190,
                                           0.20400, 0.03070, 0.03720, 0.01240,
                                           0.00641, 0.00755, 0.00926)]
    w0[1:] = [_s(v) for v in (0.02, 0.05, 0.08, 0.20, 0.18, 0.13, 0.27, 0.24,
                              0.23, 0.30, 0.40)]
    wm[1:] = [_s(v) for v in (0.01, 0.02, 0.03, 0.08, 0.07, 0.06, 0.12, 0.11,
                              0.10, 0.15, 0.20)]
    return SoilTypeParameters(b=b, psi_max=psi, porosity=por, gamma_max=gam,
                              lambda_max=lam, W_0=w0, W_m=wm)


# ---------------------------------------------------------------------------
# Small numerical helpers: PROGONKA, STEP, OVERLAY
# ---------------------------------------------------------------------------

def _progonka_1based(a, b, c, f, n):
    """``PROGONKA(numerics_mod.f90:243-273)`` over the window ``1..n``.

    ``a``, ``b``, ``c``, ``f`` are 1-based; index 0 is ignored.  Uses the
    already validated port in :mod:`lakematrix.physics.tridiag`.
    """
    y = progonka(a[1:n + 1], b[1:n + 1], c[1:n + 1], f[1:n + 1])
    out = np.zeros(n + 1)
    out[1:] = y
    return out


def step(x) -> float:
    """Heaviside ``STEP`` (numerics_mod.f90:276-286): ``0.5*(sign(1,x)+1)``."""
    # gfortran implements SIGN with a bit test (copysign), so -0.0 gives -1.
    return 0.5 * (math.copysign(1.0, float(x)) + 1.0)


def overlay(x1, x2, y1, y2) -> bool:
    """``OVERLAY`` (soil_mod.f90:1614-1627): do the two intervals overlap?"""
    return ((y1 >= x1 and y2 <= x2)
            or (y1 <= x1 and y2 >= x2)
            or (y1 <= x1 and y2 > x1)
            or (y1 < x2 and y2 >= x2))


# ---------------------------------------------------------------------------
# PHYS_FUNC helpers (phys_func.f90)
# ---------------------------------------------------------------------------

def wl_max(por, rosdry, wi, mh):
    """``WL_MAX`` (phys_func.f90:1364-1381): max liquid water, kg/kg dry soil."""
    work = _DENSMH if mh > 0.0 else C.roi  # UNREACHABLE for mh = 0
    return por * C.row0 / (rosdry * (1.0 - por)) - wi * C.row0 / work


def unfrwat(t, i, params: "SoilParams"):
    """``UNFRWAT`` (phys_func.f90:1346-1361): unfrozen water at ``t`` (C)."""
    # unfrwat = (WLM0(i)-WLM7(i))*exp(T/3.) + WLM7(i)   (3. is exact)
    return (params.WLM0[i] - params.WLM7[i]) * math.exp(t / 3.0) + params.WLM7[i]


def wi_max(por, rosdry, mh):
    """``WI_MAX`` (phys_func.f90:1384-1400): max ice content, kg/kg dry soil."""
    work = _DENSMH if mh > 0.0 else C.roi  # UNREACHABLE for mh = 0
    return por * work / (rosdry * (1.0 - por))


# Cote and Konrad (2005) / Johansen (1975) constants of SOIL_COND_JOHANSEN.
_CK_CONST = 1.9            # CK_consts(3): silty and clay soils
_CK_CONST1 = 0.75          # CK_consts1(2): mineral soils, W/(m*K)
_CK_CONST2 = 1.2           # CK_consts2(2): mineral soils, n/d
_LAMBDA_QUARTZ = 7.7       # lambda_quartz
_LAMBDA_OTHMIN = 3.0       # lambda_othmin (quartz_ratio = 0.1 <= 0.2)
_LAMBDA_SOLIDS = (_LAMBDA_QUARTZ ** 0.1
                  * _LAMBDA_OTHMIN ** (1.0 - 0.1))  # lambda_solids


def soil_cond_johansen(wl, wi, rosdry, por):
    """``SOIL_COND_JOHANSEN`` (phys_func.f90:1403-1497), W/(m*K).

    ``wl``, ``wi``: liquid water and ice content w.r.t. dry soil mass (kg/kg);
    ``rosdry``: dry soil density (kg/m3); ``por``: porosity (m3/m3).
    """
    wl = np.float64(wl)
    wi = np.float64(wi)
    por = np.float64(por)
    rosdry = np.float64(rosdry)

    # Conversion from mass ratios to volume ratios.
    water_vol_ratio = _fdiv(wl, por * (wl + C.row0 / C.roi * wi + C.row0 / rosdry))
    ice_vol_ratio = _fdiv(wi, por * (wi + C.roi / C.row0 * wl + C.roi / rosdry))
    waterice_vol_ratio = water_vol_ratio + ice_vol_ratio

    # Kersten number.
    kersten = _fdiv(_CK_CONST * waterice_vol_ratio,
                    1.0 + (_CK_CONST - 1.0) * waterice_vol_ratio)
    water_sat_ratio = _fdiv(por * water_vol_ratio, waterice_vol_ratio)
    ice_sat_ratio = _fdiv(por * ice_vol_ratio, waterice_vol_ratio)

    # Johansen (1975), extended for the ice content.
    lambda_sat = (np.float64(_LAMBDA_SOLIDS) ** (1.0 - por)
                  * np.float64(C.lamw0) ** water_sat_ratio
                  * np.float64(C.lami) ** ice_sat_ratio)
    # Cote and Konrad (2005) dry-soil conductivity.
    lambda_dry = _CK_CONST1 * 10.0 ** (-_CK_CONST2 * por)
    return (lambda_sat - lambda_dry) * kersten + lambda_dry


def meltpnt(c, p, npar):
    """``MELTPNT`` (phys_func.f90:778-798): freezing point, C.

    ``npar == 0``: 0 C; ``npar == 1``: ``-C*66.7`` (``c`` in kg/kg); ``npar == 2``:
    Jackett et al. (2004), which needs ``FP_THETA`` and is not ported.
    """
    if npar == 0:
        return 0.0
    if npar == 1:
        return 0.0 - c * _MELTPNT_DTDC
    raise NotImplementedError(
        "MELTPNT: nmeltpoint=2 (Jackett et al. 2004, FP_THETA) is not ported")


def tempmhydrdiss(p):
    """``TEMPMHYDRDISS`` (phys_func.f90:1528-1545), C.  UNREACHABLE for mh = 0."""
    # 1./(c1 - c2*log10(p*1.E-6)) - Kelvin0
    return 1.0 / (_TEMPMH_C1 - _TEMPMH_C2 * math.log10(p * _s(1.0e-6))) - C.Kelvin0


def meltingpoint(s, p, methhydr, nmeltpoint):
    """``MELTINGPOINT`` (phys_func.f90:1548-1564): C."""
    if methhydr > 0.0:
        return tempmhydrdiss(p)  # UNREACHABLE for tricemethhydr = 0
    return meltpnt(s, p, nmeltpoint)


# ---------------------------------------------------------------------------
# Dataclasses: parameters, geometry, state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SoilParams:
    """Static soil parameters, the result of ``COMSOILFORLAKE``.

    All ``(ns)`` arrays are 1-based (index 0 unused).  ``BH``/``PSIMAX``/``POR``/
    ``FLWMAX``/``DLMAX``/``WLM0``/``WLM7`` keep the Fortran names; ``POR`` is the
    array the source also spells ``por`` (Fortran is case-insensitive).
    """

    ns: int
    nsoilcols: int
    soiltype: int
    soil_depth: float
    upper_layer: float
    zsoil: np.ndarray       # (ns+1) cell interfaces, m
    dzs: np.ndarray         # (ns+1) layer thicknesses (1..ns-1)
    dzss: np.ndarray        # (ns+1) half-cell widths (1..ns)
    BH: np.ndarray          # b
    PSIMAX: np.ndarray      # -psi_max
    POR: np.ndarray         # porosity (== Fortran ``por``)
    FLWMAX: np.ndarray      # saturated hydraulic conductivity
    DLMAX: np.ndarray       # saturated water diffusivity
    WLM0: np.ndarray        # max unfrozen water at 0 C
    WLM7: np.ndarray        # max unfrozen water at T << 0 C
    rosdry: np.ndarray      # dry soil density, kg/m3
    tricemethhydr: float = 0.0
    nmeltpoint: int = 1
    soilbotflx: float = 0.0
    soilcolconjtype: int = 2


@dataclass
class Bathym:
    """Scalar subset of the Fortran ``bathym`` type (lake_modules.f90:656-661).

    Used for the per-level lists ``bathymwater(1:M+1)``, ``bathymice(1:Mice+1)``,
    ``bathymdice(1:Mice+1)`` and ``bathymsoil(1:nsoilcols+1)`` (1-based lists:
    element 0 is a dummy).
    """

    area_int: float = 0.0
    area_half: float = 0.0
    Lx: float = 0.0
    Ly: float = 0.0
    Lx_half: float = 0.0
    Ly_half: float = 0.0
    dzSL: float = 0.0
    dzSLc: float = 0.0
    itop: int = 1
    ibot: int = 1
    icent: int = 1


def bathym_levels(n_levels, *, area_int=None, area_half=None, Lx=None, Ly=None,
                  Lx_half=None, Ly_half=None):
    """Pack the vectorised arrays of :mod:`lakematrix.physics.bathym` into a
    1-based list of :class:`Bathym`.

    ``n_levels`` is ``M+1`` (water / deep ice) or ``Mice+1`` (ice).  The
    ``*_half`` arrays have ``n_levels-1`` entries, as in the Fortran type.
    """
    out = [Bathym() for _ in range(n_levels + 2)]
    for k in range(1, n_levels + 1):
        b = out[k]
        if area_int is not None:
            b.area_int = float(area_int[k - 1])
        if Lx is not None:
            b.Lx = float(Lx[k - 1])
        if Ly is not None:
            b.Ly = float(Ly[k - 1])
        if k < n_levels:
            if area_half is not None:
                b.area_half = float(area_half[k - 1])
            if Lx_half is not None:
                b.Lx_half = float(Lx_half[k - 1])
            if Ly_half is not None:
                b.Ly_half = float(Ly_half[k - 1])
    return out


@dataclass
class SoilGeometry:
    """Grid spacing and bathymetry read by ``SOILCOLSTEMP``/``LATERHEAT``.

    ``zsoilcols`` is the per-lake ``gsp%zsoilcols(:, ix, iy)`` (1-based,
    ``nsoilcols+1`` entries).  ``ddz``/``ddzi``/``z_half`` are 1-based;
    ``ddz05``/``ddzi05`` keep the Fortran 0-based bound (``ddz05(0:M)``), like
    ``grid.py``.  The bathymetry lists are rebuilt every step by ``BATHYMETRY``.
    """

    M: int
    Mice: int
    zsoilcols: np.ndarray
    ddz: np.ndarray
    ddzi: np.ndarray
    ddz05: np.ndarray
    ddzi05: np.ndarray
    z_half: np.ndarray
    bathymwater: List[Bathym]
    bathymice: List[Bathym]
    bathymdice: List[Bathym]
    bathymsoil: List[Bathym]


@dataclass
class LatHeatSource:
    """``lsh_type`` (lake_modules.f90:711-713): lateral heat from soil columns."""

    water: np.ndarray   # (M+1), 1-based
    ice: np.ndarray     # (Mice+1), 1-based
    dice: np.ndarray    # (Mice+1), 1-based


@dataclass
class SoilState:
    """Soil state, work arrays and the coefficients shared by all columns.

    Prognostic: ``Tsoil1``, ``wl1``, ``wi1``, ``Sals1`` (``(ns, nsoilcols)``,
    1-based in both dimensions).  ``Tsoil3``/``wl4``/``wi2`` are the phase-change
    results of ``SOILFORLAKE``; ``Tsoil2``/``wl2``/``wl3``/``wa``/``methgenmh``
    are 1-D (``ns``) exactly as in ``ARRAYS_SOIL``.

    ``a``/``b``/``c``/``d`` are the caller's ``nveclen`` PROGONKA workspaces
    (``lake.f90`` passes the same arrays to every ``SOILFORLAKE`` call, so an
    unwritten first row persists between calls).  ``h1``/``l1``/``hs1``/``ls1``
    are copies of the layer thicknesses, ``pressure`` is ``ATMOS%pressure`` and
    ``Tw1``/``Ti1``/``Tis1`` are the water/ice temperatures needed by
    ``TSURFSOILCOL`` (read-only here).
    """

    Tsoil1: np.ndarray
    Tsoil3: np.ndarray
    Tsoil2: np.ndarray
    wl1: np.ndarray
    wl4: np.ndarray
    wl2: np.ndarray
    wl3: np.ndarray
    wi1: np.ndarray
    wi2: np.ndarray
    Sals1: np.ndarray
    wa: np.ndarray
    methgenmh: np.ndarray
    csoil: np.ndarray
    rosoil: np.ndarray
    lamsoil: np.ndarray
    lammoist: np.ndarray
    filtr: np.ndarray
    wsoil: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    soilflux: np.ndarray
    h1: float = 0.0
    l1: float = 0.0
    hs1: float = 0.0
    ls1: float = 0.0
    pressure: float = C.pref
    dhwfsoil: float = 0.0
    isoilcol: int = 1
    Tw1: Optional[np.ndarray] = None
    Ti1: Optional[np.ndarray] = None
    Tis1: Optional[np.ndarray] = None
    geom: Optional[SoilGeometry] = None


# ---------------------------------------------------------------------------
# COMSOILFORLAKE
# ---------------------------------------------------------------------------

def comsoilforlake(*, soiltype, ns, nsoilcols, soil_depth,
                   upper_layer=C.UpperLayer, tricemethhydr=0.0, nmeltpoint=1,
                   soilbotflx=0.0, soilcolconjtype=2) -> SoilParams:
    """``COMSOILFORLAKE`` (soil_mod.f90:8-212): soil grid + soil-type arrays.

    Builds ``zsoil``/``dzs``/``dzss``, fills the per-level soil type arrays from
    row ``soiltype`` of :func:`soil_type_parameters` and sets
    ``rosdry(:) = 1.2E+3``.
    """
    soiltype = int(soiltype)
    if soiltype <= 0:
        # Fortran: write(*,*) 'LAKE: Soil type is not given: STOP'; STOP
        raise ValueError("LAKE: Soil type is not given: STOP")
    if soiltype > NUM_SOIL:
        raise ValueError(f"soiltype={soiltype} exceeds the {NUM_SOIL}-row table")

    table = soil_type_parameters()

    # zsoil(1) = 0., then the power-law / linear grid (soil_mod.f90:71-82).
    zsoil = np.zeros(ns + 1)
    if upper_layer > 0.0:
        # pow = log(UpperLayer/depth%par)/log(1.d0/float(ns-1))
        pow_ = math.log(upper_layer / soil_depth) / math.log(1.0 / float(ns - 1))
        for i in range(2, ns + 1):
            # (1.*(i-1)/(ns-1)) is a default-real expression: float32 quotient.
            base = float(np.float32(np.float32(1.0) * np.float32(i - 1))
                         / np.float32(ns - 1))
            zsoil[i] = base ** pow_ * soil_depth
    else:
        for i in range(2, ns + 1):
            base = float(np.float32(np.float32(1.0) * np.float32(i - 1))
                         / np.float32(ns - 1))
            zsoil[i] = base * soil_depth

    dzs = np.zeros(ns + 1)
    for i in range(1, ns):
        dzs[i] = zsoil[i + 1] - zsoil[i]

    dzss = np.zeros(ns + 1)
    for i in range(1, ns + 1):
        if i == 1:
            dzss[i] = 0.5 * dzs[i]           # 0.5d0*dzs(i)
        elif i == ns:
            dzss[i] = 0.5 * dzs[i - 1]
        else:
            dzss[i] = 0.5 * (dzs[i - 1] + dzs[i])

    # Per-level copies for the selected soil type (soil_mod.f90:184-193).
    st = soiltype
    BH = np.full(ns + 1, table.b[st])
    PSIMAX = np.full(ns + 1, -table.psi_max[st])
    POR = np.full(ns + 1, table.porosity[st])
    FLWMAX = np.full(ns + 1, table.gamma_max[st])
    DLMAX = np.full(ns + 1, table.lambda_max[st])
    WLM0 = np.full(ns + 1, table.W_0[st])
    WLM7 = np.full(ns + 1, table.W_m[st])
    WLM0[0] = WLM7[0] = 0.0  # keep the pad finite

    # rosdry(:) = 1.2E+3  (default-real literal; 1200 is exact in float32)
    rosdry = np.full(ns + 1, _s(1.2e3))

    return SoilParams(
        ns=ns, nsoilcols=nsoilcols, soiltype=st, soil_depth=float(soil_depth),
        upper_layer=float(upper_layer), zsoil=zsoil, dzs=dzs, dzss=dzss,
        BH=BH, PSIMAX=PSIMAX, POR=POR, FLWMAX=FLWMAX, DLMAX=DLMAX,
        WLM0=WLM0, WLM7=WLM7, rosdry=rosdry,
        tricemethhydr=float(tricemethhydr), nmeltpoint=int(nmeltpoint),
        soilbotflx=float(soilbotflx), soilcolconjtype=int(soilcolconjtype))


def make_soil_state(params: SoilParams, geom: Optional[SoilGeometry] = None, *,
                    h1=0.0, l1=0.0, hs1=0.0, ls1=0.0,
                    pressure=None, Tw1=None, Ti1=None, Tis1=None) -> SoilState:
    """Allocate a zeroed :class:`SoilState` (1-based shapes, index 0 unused)."""
    ns = params.ns
    nc = params.nsoilcols
    z2 = lambda a, b: np.zeros((a, b))
    z1 = lambda a: np.zeros(a)
    return SoilState(
        Tsoil1=z2(ns + 1, nc + 1), Tsoil3=z2(ns + 1, nc + 1), Tsoil2=z1(ns + 1),
        wl1=z2(ns + 1, nc + 1), wl4=z2(ns + 1, nc + 1), wl2=z1(ns + 1),
        wl3=z1(ns + 1), wi1=z2(ns + 1, nc + 1), wi2=z2(ns + 1, nc + 1),
        Sals1=z2(ns + 1, nc + 1), wa=z1(ns + 1), methgenmh=z1(ns + 1),
        csoil=z1(ns + 1), rosoil=z1(ns + 1), lamsoil=z1(ns + 1),
        lammoist=z1(ns + 1), filtr=z1(ns + 1), wsoil=z1(ns + 1),
        a=z1(ns + 2), b=z1(ns + 2), c=z1(ns + 2), d=z1(ns + 2),
        soilflux=z1(nc + 1), h1=float(h1), l1=float(l1), hs1=float(hs1),
        ls1=float(ls1), pressure=C.pref if pressure is None else float(pressure),
        dhwfsoil=0.0, isoilcol=nc, Tw1=Tw1, Ti1=Ti1, Tis1=Tis1, geom=geom)


# ---------------------------------------------------------------------------
# SOIL_COND_HEAT_COEF
# ---------------------------------------------------------------------------

def soil_cond_heat_coef(state: SoilState, params: SoilParams, is_=None):
    """``SOIL_COND_HEAT_COEF(is)`` (soil_mod.f90:712-801), in place.

    Fills ``csoil``/``rosoil``/``lamsoil``/``lammoist``/``filtr``/``wsoil`` for
    soil column ``is`` (default ``state.isoilcol``).  ``soil_mod.f90:761`` also
    computes ``wlmax_i = POR(i)*ROW0/(rosdry(i)*(1-por(i)))`` before it is
    overwritten at line 775; that dead store is omitted.
    """
    is_ = state.isoilcol if is_ is None else int(is_)
    ns = params.ns
    mh = params.tricemethhydr
    por = params.POR
    rosdry = params.rosdry
    dzs = params.dzs

    # cr = 0.2*4180.d0: 0.2 is default real, 4180.d0 double.
    cr = _s(0.2) * 4180.0
    # cimh => cpmethhydr if tricemethhydr > 0, else ci.
    cimh = _CPMETHHYDR if mh > 0.0 else C.ci  # UNREACHABLE for mh = 0

    wl1 = state.wl1
    wi1 = state.wi1

    for i in range(1, ns + 1):
        bb1 = params.BH[i] + 2.0  # BB1 = BH(i) + 2.d0
        state.csoil[i] = cr + wl1[i, is_] * C.cw + wi1[i, is_] * cimh
        state.rosoil[i] = (rosdry[i] * (1.0 - por[i])
                           * (1.0 + wi1[i, is_] + wl1[i, is_]))
        state.lamsoil[i] = soil_cond_johansen(wl1[i, is_], wi1[i, is_],
                                              rosdry[i], por[i])
        wlmax_i = wl_max(por[i], rosdry[i], wi1[i, is_], mh)
        arg = _fdiv(wl1[i, is_], wlmax_i)
        arg = max(arg, _ARG_MIN)  # max(ARG, 1.d-2)
        state.lammoist[i] = params.DLMAX[i] * arg ** bb1

    for i in range(1, ns):
        wlmax_i = wl_max(por[i + 1], rosdry[i + 1], wi1[i + 1, is_], mh)
        if wlmax_i < _S001:
            # NB: the source condition also had the (now removed) test
            # wl1(i+1,is)/wlmax_i > 0.98.
            state.filtr[i] = 0.0
        else:
            wlmax_i = wl_max(0.5 * (por[i] + por[i + 1]),
                             0.5 * (rosdry[i] + rosdry[i + 1]),
                             0.5 * (wi1[i, is_] + wi1[i + 1, is_]), mh)
            state.filtr[i] = (0.5 * (params.FLWMAX[i + 1] + params.FLWMAX[i])
                              * (0.5 * (wl1[i + 1, is_] + wl1[i, is_])
                                 / wlmax_i) ** (params.BH[i + 1] + params.BH[i] + 3))

    for i in range(1, ns):
        lammoist1 = 0.5 * (state.lammoist[i] + state.lammoist[i + 1])
        state.wsoil[i] = ((-lammoist1 * (wl1[i + 1, is_] - wl1[i, is_]) / dzs[i]
                           + state.filtr[i])
                          * 0.5 * (rosdry[i] + rosdry[i + 1])
                          * (1.0 - 0.5 * (por[i] + por[i + 1])) / C.row0)
    return state


# ---------------------------------------------------------------------------
# DIFF_COEF (soil case) and the soil-only branch of T_DIFF
# ---------------------------------------------------------------------------

def _diff_coef_soil(a, b, c, d, n0, n1, m0, m1, dt, state, params, Tsoil1_col):
    """``DIFF_COEF`` case ``soil_indic`` (t_solver_mod.f90:699-717).

    ``n0``/``n1``/``m0``/``m1`` follow the Fortran convention; ``Tsoil1_col`` is
    ``Tsoil1(:, isoilcol)``.  ``a``..``d`` are 1-based and modified in place.
    """
    dt_inv = 1.0 / dt
    lamsoil = state.lamsoil
    csoil = state.csoil
    rosoil = state.rosoil
    dzs = params.dzs
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        lam2 = 0.5 * (lamsoil[j] + lamsoil[j + 1])
        lam1 = 0.5 * (lamsoil[j - 1] + lamsoil[j])
        dzmean = 0.5 * (dzs[j - 1] + dzs[j])
        a[i] = -0.5 * lam1 / dzs[j - 1]
        b[i] = -0.5 * lam2 / dzs[j]
        c[i] = a[i] + b[i] - dt_inv * (csoil[j] * rosoil[j] * dzmean)
        d[i] = (-Tsoil1_col[j] * dt_inv * (csoil[j] * rosoil[j] * dzmean)
                + a[i] * Tsoil1_col[j - 1]
                - (a[i] + b[i]) * Tsoil1_col[j]
                + b[i] * Tsoil1_col[j + 1])


def _t_diff_soil(state: SoilState, params: SoilParams, tsurs, isoilcol, dt,
                 ifflux=False):
    """Soil-only branch of ``T_DIFF`` (t_solver_mod.f90:517-536).

    ``surf == 1`` and ``snow == ice == water == deepice == 0``, which is the
    only way ``SOILCOLSTEMP`` calls ``T_DIFF`` (``ifflux = .false.``).  Returns
    the new ``Tsoil2(1:ns)``; ``a``..``d`` are local, exactly as in ``T_DIFF``
    (they are deallocated on exit there, so nothing persists).
    """
    ns = params.ns
    a = np.zeros(ns + 2)
    b = np.zeros(ns + 2)
    c = np.zeros(ns + 2)
    d = np.zeros(ns + 2)
    dt_inv = 1.0 / dt
    dzs = params.dzs
    Tcol = state.Tsoil1[:, isoilcol]

    if ifflux:
        # UNREACHABLE from SOILCOLSTEMP (ifflux is always .false. there).
        cc = state.rosoil[1] * state.csoil[1] * 0.5 * dzs[1] * dt_inv
        xx = 0.25 * (state.lamsoil[1] + state.lamsoil[2]) / dzs[1]
        b[1] = xx
        c[1] = cc + xx
        d[1] = tsurs + xx * Tcol[2] + (cc - xx) * Tcol[1]
    else:
        b[1] = 0.0
        c[1] = 1.0
        d[1] = tsurs

    _diff_coef_soil(a, b, c, d, 2, ns - 1, 2, ns - 1, dt, state, params, Tcol)

    c[ns] = 0.5
    a[ns] = 0.5
    d[ns] = (-params.soilbotflx * dzs[ns - 1]
             / (state.lamsoil[ns - 1] + state.lamsoil[ns])
             + a[ns] * Tcol[ns - 1] - c[ns] * Tcol[ns])

    y = _progonka_1based(a, b, c, d, ns)
    return y


# ---------------------------------------------------------------------------
# SOILFORLAKE
# ---------------------------------------------------------------------------

# data psiit/6,3,7,2,5,4,8,1,0,0/ (real(8), exact)
_PSIIT = (0.0, 6.0, 3.0, 7.0, 2.0, 5.0, 4.0, 8.0, 1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# SOILFORLAKE
# ---------------------------------------------------------------------------

# data psiit/6,3,7,2,5,4,8,1,0,0/ (real(8), exact)
_PSIIT = (0.0, 6.0, 3.0, 7.0, 2.0, 5.0, 4.0, 8.0, 1.0, 0.0, 0.0)

def soilforlake(state: SoilState, params: SoilParams, dt, is_=None,
                debug=False):
    """``SOILFORLAKE(nn,dt,a,b,c,d,is)`` (soil_mod.f90:215-709), in place.

    Liquid water (``wl``) and ice (``wi``) evolution for soil column ``is``
    (default ``state.isoilcol``): moisture diffusion (``PROGONKA``), gravitational
    infiltration, the excess-water redistribution, the phase-change iteration,
    the porosity adjustment and the ``wa`` diagnostic.  ``Tsoil2`` must already
    hold the diffused temperature of the same column (see the module docstring);
    the split is

      1. heat diffusion  -> done by the caller's ``T_DIFF`` (writes ``Tsoil2``),
      2. moisture steps  -> this routine (writes ``wl2``/``wl3``/``wl4``),
      3. phase processes -> this routine (writes ``Tsoil3``/``wi2``).

    Updates ``Tsoil1``, ``wl1``, ``wi1``, ``Tsoil3``, ``wl4``, ``wi2``,
    ``methgenmh``, ``wa`` and ``dhwfsoil``.  ``Sals1`` is only read (with
    ``salsoil 0`` it is zero and the melting point is 0 C).  Returns ``state``.
    """
    is_ = state.isoilcol if is_ is None else int(is_)
    ns = params.ns
    mh = params.tricemethhydr
    row0 = C.row0
    por = params.POR
    rosdry = params.rosdry
    dzs = params.dzs
    dzss = params.dzss

    Tsoil1 = state.Tsoil1
    Tsoil3 = state.Tsoil3
    Tsoil2 = state.Tsoil2
    wl1 = state.wl1
    wl4 = state.wl4
    wl2 = state.wl2
    wl3 = state.wl3
    wi1 = state.wi1
    wi2 = state.wi2
    Sals1 = state.Sals1
    csoil = state.csoil
    rosoil = state.rosoil
    lammoist = state.lammoist
    filtr = state.filtr
    a, b, c, d = state.a, state.b, state.c, state.d

    h1, l1, hs1, ls1 = state.h1, state.l1, state.hs1, state.ls1

    # hicemelt => methhydrdiss if tricemethhydr > 0, else Lwi.
    hicemelt = _METHHYDRDISS if mh > 0.0 else C.Lwi  # UNREACHABLE for mh = 0

    # Iteration weights (soil_mod.f90:299-303).  ma/mi are default real; 10.5
    # and 4.5 are exactly representable.
    ma, mi = 10.5, 4.5
    alp = np.zeros(9)  # 1-based, 1..8
    for i in range(1, 9):
        alp[i] = 2.0 * (mi + ma - (ma - mi)
                        * math.cos((2.0 * _PSIIT[i] - 1.0) / 16.0 * C.pi)) ** (-1.0)

    pressoil = np.zeros(ns + 1)
    for i in range(1, ns + 1):
        pressoil[i] = state.pressure + row0 * C.g * (h1 + params.zsoil[i])

    # --- Step 1 of the moisture splitting: implicit diffusion ---------------
    # (Wflow, Wfhigh and the dhwfsoil accumulator initialisations are dead
    # stores in the source except for dhwfsoil1..4.)
    dhwfsoil1 = 0.0
    dhwfsoil2 = 0.0
    dhwfsoil3 = 0.0
    dhwfsoil4 = 0.0

    if (((l1 != 0.0 and h1 == 0.0) or ls1 != 0.0
         or (hs1 != 0.0 and l1 == 0.0 and h1 == 0.0))):
        # No water: zero-flux top boundary.
        c[1] = -1.0 - dt * (lammoist[1] + lammoist[2]) / (dzs[1] * dzs[1])
        b[1] = -dt * (lammoist[1] + lammoist[2]) / (dzs[1] * dzs[1])
        d[1] = -wl1[1, is_]

    if h1 != 0.0 and ls1 == 0.0:
        # Open water: saturated top layer (Dirichlet on WL_MAX).
        c[1] = 1.0
        b[1] = 0.0
        d[1] = wl_max(por[1], rosdry[1], wi1[1, is_], mh)

    lammoist1 = 0.5 * (lammoist[ns - 1] + lammoist[ns])
    c[ns] = -1.0 - 2.0 * dt * lammoist1 / (dzs[ns - 1] * dzs[ns - 1])
    a[ns] = -2.0 * dt * lammoist1 / (dzs[ns - 1] * dzs[ns - 1])
    d[ns] = -wl1[ns, is_]

    for i in range(2, ns):
        lammoist2 = 0.5 * (lammoist[i] + lammoist[i + 1])
        lammoist1 = 0.5 * (lammoist[i - 1] + lammoist[i])
        dzmean = 0.5 * (dzs[i - 1] + dzs[i])
        a[i] = -dt / dzmean * lammoist1 / dzs[i - 1]
        b[i] = -dt / dzmean * lammoist2 / dzs[i]
        c[i] = -dt / dzmean * (lammoist1 / dzs[i - 1]
                               + lammoist2 / dzs[i]) - 1.0
        d[i] = -wl1[i, is_]

    wl2[:] = _progonka_1based(a, b, c, d, ns)

    if h1 != 0.0 and ls1 == 0.0:
        lammoist1 = 0.5 * (lammoist[1] + lammoist[2])
        dhwfsoil1 = -(wl2[1] * (dzs[1] / (2.0 * dt) + lammoist1 / dzs[1])
                      - wl2[2] * lammoist1 / dzs[1]
                      - wl1[1, is_] * dzs[1] / (2.0 * dt)) * \
            rosdry[1] * (1.0 - por[1]) / row0 * dt

    # --- Step 2: gravitational infiltration --------------------------------
    for i in range(2, ns):
        wl3[i] = wl2[i] + dt * (filtr[i - 1] - filtr[i]) / dzss[i]

    if ((h1 == 0.0 and l1 != 0.0) or ls1 != 0.0):
        wl3[1] = wl2[1] + dt * (-filtr[1] + 0.0) / dzss[1]
        # NB: if none of the two branches applies (h1 == l1 == ls1 == 0 with
        # hs1 /= 0), wl3(1) keeps its value from the previous call, as in
        # Fortran (module storage).

    if h1 != 0.0 and ls1 == 0.0:
        wl3[1] = wl_max(por[1], rosdry[1], wi1[1, is_], mh)
        dhwfsoil2 = -filtr[1] * rosdry[1] * (1.0 - por[1]) / row0 * dt

    filtr_low = 0.0
    wl3[ns] = wl2[ns] + dt * (-filtr_low + filtr[ns - 1]) / dzss[ns]

    # Redistribute water that exceeds WL_MAX (soil_mod.f90:439-487).
    for i in range(1, ns + 1):
        wlmax_i = wl_max(por[i], rosdry[i], wi1[i, is_], mh)
        if wl3[i] > wlmax_i:
            surplus = wl3[i] - wlmax_i
            for j in range(1, ns + 1):  # cy1
                wlmax_j = wl_max(por[j], rosdry[j], wi1[j, is_], mh)
                if wlmax_j - wl3[j] > 0.0:
                    den_j = rosdry[j] * (1.0 - por[j]) * dzss[j]
                    den_i = rosdry[i] * (1.0 - por[i]) * dzss[i]
                    if wlmax_j - wl3[j] > surplus * den_i / den_j:
                        wl3[j] = wl3[j] + surplus * den_i / den_j
                        surplus = 0.0
                        break
                    else:
                        surplus = surplus - (wlmax_j - wl3[j]) * den_j / den_i
                        wl3[j] = wlmax_j
            dhwfsoil3 = (dhwfsoil3
                         + surplus * rosdry[i] * (1.0 - por[i]) * dzss[i] / row0)
            wl3[i] = wlmax_i

        if wl3[i] < 0.0:
            # Deficit: take water from the layer below, else from above;
            # label 1 (wl3(i) = 0) is reached in all cases.
            found = False
            if i != 1:
                for j in range(i - 1, 0, -1):
                    need = -wl3[i] * rosdry[i] * (1.0 - por[i]) * dzss[i] \
                        / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                    if wl3[j] > need:
                        wl3[j] = wl3[j] + wl3[i] * rosdry[i] * (1.0 - por[i]) \
                            * dzss[i] / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                        found = True
                        break
            if not found and i != ns:
                for j in range(i + 1, ns + 1):
                    need = -wl3[i] * rosdry[i] * (1.0 - por[i]) * dzss[i] \
                        / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                    if wl3[j] > need:
                        wl3[j] = wl3[j] + wl3[i] * rosdry[i] * (1.0 - por[i]) \
                            * dzss[i] / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                        found = True
                        break
            # Fortran does not book dhwfsoil here (the line is commented out).
            wl3[i] = 0.0

    # --- Step 3: phase processes (soil_mod.f90:490-570) --------------------
    mhsep = 1.0 + mh * _ALPHAMH  # UNREACHABLE for mh = 0 (mhsep = 1)

    iter_ = 0
    iter3_ = 0
    while True:
        for i in range(1, ns + 1):
            # work = MELTINGPOINT(Sals1(i,is)/wl3(i),pressoil(i),mh)
            work = meltingpoint(_fdiv(Sals1[i, is_], wl3[i]), pressoil[i], mh,
                                params.nmeltpoint)
            if wi1[i, is_] > 0.0 and Tsoil2[i] > work + _S001:
                # Melting of soil ice.
                potphenergy = ((Tsoil2[i] - (work + _S001))
                               * csoil[i] * rosoil[i] * dzss[i])
                if potphenergy >= wi1[i, is_] * rosdry[i] * dzss[i] \
                        * (1.0 - por[i]) * hicemelt:
                    wl4[i, is_] = wl3[i] + _fdiv(wi1[i, is_], mhsep)
                    wi2[i, is_] = 0.0
                    Tsoil3[i, is_] = (work + _S001
                                      + (potphenergy
                                         - (wi1[i, is_] * rosdry[i] * dzss[i]
                                            * (1.0 - por[i]) * hicemelt))
                                      / (csoil[i] * rosoil[i] * dzss[i]))
                else:
                    wl4[i, is_] = (wl3[i] + potphenergy
                                   / (rosdry[i] * dzss[i] * (1.0 - por[i]) * hicemelt)
                                   / mhsep)
                    wi2[i, is_] = (wi1[i, is_] - potphenergy
                                   / (rosdry[i] * dzss[i] * (1.0 - por[i]) * hicemelt))
                    Tsoil3[i, is_] = work + _S001
            else:
                if wl3[i] >= 0.0 and Tsoil2[i] < work - _S001:
                    # Freezing of soil water.
                    potphenergy = (-(Tsoil2[i] - work + _S001)
                                   * csoil[i] * rosoil[i] * dzss[i])
                    watavailtofreeze = ((wl3[i] - unfrwat(Tsoil2[i], i, params))
                                        * rosdry[i] * dzss[i] * (1.0 - por[i])
                                        * mhsep * hicemelt)
                    if watavailtofreeze < 0.0:
                        Tsoil3[i, is_] = (Tsoil2[i] + watavailtofreeze
                                          / (csoil[i] * rosoil[i] * dzss[i]))
                        wi2[i, is_] = (wi1[i, is_]
                                       + (wl3[i] - unfrwat(Tsoil2[i], i, params))
                                       * mhsep)
                        wl4[i, is_] = unfrwat(Tsoil2[i], i, params)
                        continue  # cycle c2
                    if potphenergy >= watavailtofreeze:
                        Tsoil3[i, is_] = (work - _S001
                                          - (potphenergy - watavailtofreeze)
                                          / (csoil[i] * rosoil[i] * dzss[i]))
                        wi2[i, is_] = (wi1[i, is_]
                                       + (wl3[i] - unfrwat(Tsoil2[i], i, params))
                                       * mhsep)
                        wl4[i, is_] = unfrwat(Tsoil2[i], i, params)
                    else:
                        wl4[i, is_] = (wl3[i] - potphenergy
                                       / (rosdry[i] * dzss[i] * (1.0 - por[i])
                                          * hicemelt) / mhsep)
                        wi2[i, is_] = (wi1[i, is_] + potphenergy
                                       / (rosdry[i] * dzss[i] * (1.0 - por[i])
                                          * hicemelt))
                        Tsoil3[i, is_] = work - _S001
                else:
                    wl4[i, is_] = wl3[i]
                    wi2[i, is_] = wi1[i, is_]
                    Tsoil3[i, is_] = Tsoil2[i]

        # Consistency check and under-relaxation (iter = 0, alp(1) first).
        max1 = 0.0
        for i in range(1, ns + 1):
            if (Tsoil3[i, is_]
                    < meltingpoint(_fdiv(Sals1[i, is_], wl4[i, is_]), pressoil[i],
                                   mh, params.nmeltpoint) - _S001
                    and abs(wl4[i, is_] - unfrwat(Tsoil3[i, is_], i, params)) > max1):
                max1 = abs(wl4[i, is_] - unfrwat(Tsoil3[i, is_], i, params))

        if max1 > _S001 and iter3_ < 20:
            w = alp[iter_ + 1]
            wl3[1:ns + 1] = wl3[1:ns + 1] + w * (wl4[1:ns + 1, is_] - wl3[1:ns + 1])
            wi1[1:ns + 1, is_] = (wi1[1:ns + 1, is_]
                                  + w * (wi2[1:ns + 1, is_] - wi1[1:ns + 1, is_]))
            Tsoil2[1:ns + 1] = (Tsoil2[1:ns + 1]
                                + w * (Tsoil3[1:ns + 1, is_] - Tsoil2[1:ns + 1]))
            iter_ += 1
            iter3_ += 1
            if iter_ == 8:
                iter_ = 0
            continue  # goto 20
        iter_ = 0
        iter3_ = 0
        break

    # Negative ice content: return it to liquid water.
    for i in range(1, ns + 1):
        if wi2[i, is_] < 0.0:
            wl4[i, is_] = wl4[i, is_] + wi2[i, is_] / mhsep
            wi2[i, is_] = 0.0

    # Methane generation due to methane hydrate dissociation (0 for mh = 0).
    for i in range(1, ns + 1):
        state.methgenmh[i] = (-mh * (wi2[i, is_] - wi1[i, is_]) * 1.0e3
                              / (dt * mhsep * _MOLMASS_H2O) * _NCH4_D_NH2O
                              * rosdry[i] * (1.0 - por[i]))

    # Water above the pore volume: push it to other layers / out of the column.
    for i in range(1, ns + 1):
        cap_i = (por[i] * row0 / (rosdry[i] * (1.0 - por[i]))
                 - wi2[i, is_] * C.row0_d_roi)
        if wl4[i, is_] > cap_i:
            surplus = wl4[i, is_] - cap_i
            for j in range(1, ns + 1):  # c3
                cap_j = (por[j] * row0 / (rosdry[j] * (1.0 - por[j]))
                         - wi2[j, is_] * C.row0_d_roi)
                if cap_j - wl4[j, is_] > 0.0:
                    den_j = rosdry[j] * (1.0 - por[j]) * dzss[j]
                    den_i = rosdry[i] * (1.0 - por[i]) * dzss[i]
                    if cap_j - wl4[j, is_] > surplus * den_i / den_j:
                        wl4[j, is_] = wl4[j, is_] + surplus * den_i / den_j
                        surplus = 0.0
                        break
                    else:
                        surplus = surplus - (cap_j - wl4[j, is_]) * den_j / den_i
                        wl4[j, is_] = cap_j
            dhwfsoil4 = (dhwfsoil4
                         + surplus * rosdry[i] * (1.0 - por[i]) * dzss[i] / row0)
            wl4[i, is_] = cap_i

        if wl4[i, is_] < 0.0:
            found = False
            if i != 1:
                for j in range(i - 1, 0, -1):
                    need = -wl4[i, is_] * rosdry[i] * (1.0 - por[i]) * dzss[i] \
                        / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                    if wl4[j, is_] > need:
                        wl4[j, is_] = wl4[j, is_] + wl4[i, is_] * rosdry[i] \
                            * (1.0 - por[i]) * dzss[i] \
                            / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                        found = True
                        break
            if not found and i != ns:
                for j in range(i + 1, ns + 1):
                    need = -wl4[i, is_] * rosdry[i] * (1.0 - por[i]) * dzss[i] \
                        / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                    if wl4[j, is_] > need:
                        wl4[j, is_] = wl4[j, is_] + wl4[i, is_] * rosdry[i] \
                            * (1.0 - por[i]) * dzss[i] \
                            / (rosdry[j] * (1.0 - por[j]) * dzss[j])
                        found = True
                        break
            wl4[i, is_] = 0.0  # label 2

    state.dhwfsoil = dhwfsoil1 + dhwfsoil2 + dhwfsoil3 + dhwfsoil4

    # Diagnostic print of the original model (kept behind a flag).
    if debug and np.any(Tsoil3[1:ns + 1, is_] < -10.0):
        tag = f'(column {is_})'
        print('SOILFORLAKE: Tsoil11', tag, '=', Tsoil1[1:ns + 1, is_])
        print('SOILFORLAKE: Tsoil12', tag, '=', Tsoil2[1:ns + 1])
        print('SOILFORLAKE: Tsoil13', tag, '=', Tsoil3[1:ns + 1, is_])
        print('SOILFORLAKE: wl1', tag, '=', wl1[1:ns + 1, is_])
        print('SOILFORLAKE: wl2', tag, '=', wl2[1:ns + 1])
        print('SOILFORLAKE: wl3', tag, '=', wl3[1:ns + 1])
        print('SOILFORLAKE: wl4', tag, '=', wl4[1:ns + 1, is_])
        print('SOILFORLAKE: wi1', tag, '=', wi1[1:ns + 1, is_])
        print('SOILFORLAKE: wi2', tag, '=', wi2[1:ns + 1, is_])

    # Split the phase-change output back into the prognostic arrays.
    Tsoil1[1:ns + 1, is_] = Tsoil3[1:ns + 1, is_]
    wl1[1:ns + 1, is_] = wl4[1:ns + 1, is_]
    wi1[1:ns + 1, is_] = wi2[1:ns + 1, is_]

    # Diagnostic air content, kg/kg dry soil (soil_mod.f90:699-703).
    for i in range(1, ns + 1):
        state.wa[i] = (por[i] * C.roa0 / ((1.0 - por[i]) * rosdry[i])
                       - C.roa0_d_row0 * wl1[i, is_]
                       - C.roa0_d_roi * wi1[i, is_])
        state.wa[i] = max(state.wa[i], 0.0)
    return state


# ---------------------------------------------------------------------------
# TSURFSOILCOL
# ---------------------------------------------------------------------------

def tsurfsoilcol(state: SoilState, params: SoilParams):
    """``TSURFSOILCOL`` (soil_mod.f90:1034-1382): mean temperature of each side
    soil column, area-averaged over the water/ice/deep-ice layers it overlaps.

    Returns ``Tsoilsurf(1:nsoilcols)`` (1-based; element 0 unused).  Only the
    four branches with ``h1 > small_value`` exist in the source; if the lake has
    no water the Fortran leaves ``Tsoilsurf`` undefined (it is ``intent(out)``),
    so the returned values stay 0 there (never reached by the reference cases,
    which call this only when ``SOILCOLSTEMP`` runs on a water column).
    """
    geom = state.geom
    if geom is None:
        raise ValueError("tsurfsoilcol needs state.geom")
    if state.Tw1 is None or state.Ti1 is None or state.Tis1 is None:
        raise ValueError("tsurfsoilcol needs state.Tw1/Ti1/Tis1")
    M, Mice, ncols = geom.M, geom.Mice, params.nsoilcols
    h1, l1, ls1 = state.h1, state.l1, state.ls1
    zsc = geom.zsoilcols
    ddz, ddzi = geom.ddz, geom.ddzi
    Tw1, Ti1, Tis1 = state.Tw1, state.Ti1, state.Tis1
    bw = geom.bathymwater
    bi = geom.bathymice
    bd = geom.bathymdice
    bs = geom.bathymsoil
    small = C.small_value

    Tsoilsurf = np.zeros(ncols + 1)

    # --- the four layer cases of the source -------------------------------
    if l1 > small and h1 > small and ls1 > small:      # ice + water + deep ice
        nl = 2 * Mice + M
        Temp = np.zeros(nl + 2)
        z = np.zeros(nl + 2)
        aell = np.zeros(nl + 2)
        bell = np.zeros(nl + 2)
        # Ice top
        z[0] = 0.0
        z[1] = 0.5 * ddzi[1] * l1
        aell[0] = 0.5 * bi[1].Lx
        aell[1] = 0.5 * bi[1].Lx_half
        bell[0] = 0.5 * bi[1].Ly
        bell[1] = 0.5 * bi[1].Ly_half
        Temp[1] = Ti1[1]
        j = 2
        # Ice interior
        for i in range(2, Mice + 1):
            Temp[j] = Ti1[i]
            aell[j] = 0.5 * bi[i].Lx_half
            bell[j] = 0.5 * bi[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddzi[i - 1] + ddzi[i]) * l1
            j += 1
        # Ice-water interface
        Temp[j] = Ti1[Mice + 1]
        z[j] = z[j - 1] + 0.5 * (ddzi[Mice] * l1 + ddz[1] * h1)
        aell[j] = 0.5 * bw[1].Lx_half
        bell[j] = 0.5 * bw[1].Ly_half
        j += 1
        # Water interior
        for i in range(2, M + 1):
            Temp[j] = Tw1[i]
            aell[j] = 0.5 * bw[i].Lx_half
            bell[j] = 0.5 * bw[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddz[i - 1] + ddz[i]) * h1
            j += 1
        # Water-deep ice interface
        Temp[j] = Tw1[M + 1]
        z[j] = z[j - 1] + 0.5 * (ddz[M] * h1 + ddzi[1] * ls1)
        aell[j] = 0.5 * bd[1].Lx_half
        bell[j] = 0.5 * bd[1].Ly_half
        j += 1
        # Deep ice interior
        for i in range(2, Mice + 1):
            Temp[j] = Tis1[i]
            aell[j] = 0.5 * bd[i].Lx_half
            bell[j] = 0.5 * bd[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddzi[i - 1] + ddzi[i]) * ls1
            j += 1
        # Deep ice bottom
        Temp[j] = Tis1[Mice + 1]
        z[j] = z[j - 1] + 0.5 * ddzi[Mice] * ls1
        aell[j] = 0.5 * bd[Mice + 1].Lx
        bell[j] = 0.5 * bd[Mice + 1].Ly
        z[:] = z[:] + zsc[ncols + 1] - h1 - l1 - ls1

    elif l1 > small and h1 > small and ls1 < small:    # ice + water
        nl = Mice + M
        Temp = np.zeros(nl + 2)
        z = np.zeros(nl + 2)
        aell = np.zeros(nl + 2)
        bell = np.zeros(nl + 2)
        z[0] = 0.0
        z[1] = 0.5 * ddzi[1] * l1
        aell[0] = 0.5 * bi[1].Lx
        aell[1] = 0.5 * bi[1].Lx_half
        bell[0] = 0.5 * bi[1].Ly
        bell[1] = 0.5 * bi[1].Ly_half
        Temp[1] = Ti1[1]
        j = 2
        for i in range(2, Mice + 1):
            Temp[j] = Ti1[i]
            aell[j] = 0.5 * bi[i].Lx_half
            bell[j] = 0.5 * bi[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddzi[i - 1] + ddzi[i]) * l1
            j += 1
        # Ice-water interface
        Temp[j] = Ti1[Mice + 1]
        z[j] = z[j - 1] + 0.5 * (ddzi[Mice] * l1 + ddz[1] * h1)
        aell[j] = 0.5 * bw[1].Lx_half
        bell[j] = 0.5 * bw[1].Ly_half
        j += 1
        # Water interior
        for i in range(2, M + 1):
            Temp[j] = Tw1[i]
            aell[j] = 0.5 * bw[i].Lx_half
            bell[j] = 0.5 * bw[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddz[i - 1] + ddz[i]) * h1
            j += 1
        # Water bottom
        Temp[j] = Tw1[M + 1]
        z[j] = z[j - 1] + 0.5 * ddz[M] * h1
        aell[j] = 0.5 * bw[M + 1].Lx
        bell[j] = 0.5 * bw[M + 1].Ly
        z[:] = z[:] + zsc[ncols + 1] - h1 - l1

    elif l1 < small and h1 > small and ls1 > small:    # water + deep ice
        nl = Mice + M
        Temp = np.zeros(nl + 2)
        z = np.zeros(nl + 2)
        aell = np.zeros(nl + 2)
        bell = np.zeros(nl + 2)
        # Water top
        z[0] = 0.0
        z[1] = 0.5 * ddz[1] * h1
        aell[0] = 0.5 * bw[1].Lx
        aell[1] = 0.5 * bw[1].Lx_half
        bell[0] = 0.5 * bw[1].Ly
        bell[1] = 0.5 * bw[1].Ly_half
        Temp[1] = Tw1[1]
        j = 2
        for i in range(2, M + 1):
            Temp[j] = Tw1[i]
            aell[j] = 0.5 * bw[i].Lx_half
            bell[j] = 0.5 * bw[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddz[i - 1] + ddz[i]) * h1
            j += 1
        # Water-deep ice interface
        Temp[j] = Tw1[M + 1]
        z[j] = z[j - 1] + 0.5 * (ddz[M] * h1 + ddzi[1] * ls1)
        aell[j] = 0.5 * bd[1].Lx_half
        bell[j] = 0.5 * bd[1].Ly_half
        j += 1
        for i in range(2, Mice + 1):
            Temp[j] = Tis1[i]
            aell[j] = 0.5 * bd[i].Lx_half
            bell[j] = 0.5 * bd[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddzi[i - 1] + ddzi[i]) * ls1
            j += 1
        # Deep ice bottom
        Temp[j] = Tis1[Mice + 1]
        z[j] = z[j - 1] + 0.5 * ddzi[Mice] * ls1
        aell[j] = 0.5 * bd[Mice + 1].Lx
        bell[j] = 0.5 * bd[Mice + 1].Ly
        z[:] = z[:] + zsc[ncols + 1] - h1 - ls1

    elif l1 < small and h1 > small and ls1 < small:    # water only
        nl = M
        Temp = np.zeros(nl + 2)
        z = np.zeros(nl + 2)
        aell = np.zeros(nl + 2)
        bell = np.zeros(nl + 2)
        z[0] = 0.0
        z[1] = 0.5 * ddz[1] * h1
        aell[0] = 0.5 * bw[1].Lx
        aell[1] = 0.5 * bw[1].Lx_half
        bell[0] = 0.5 * bw[1].Ly
        bell[1] = 0.5 * bw[1].Ly_half
        Temp[1] = Tw1[1]
        j = 2
        for i in range(2, M + 1):
            Temp[j] = Tw1[i]
            aell[j] = 0.5 * bw[i].Lx_half
            bell[j] = 0.5 * bw[i].Ly_half
            z[j] = z[j - 1] + 0.5 * (ddz[i - 1] + ddz[i]) * h1
            j += 1
        # Water bottom
        Temp[j] = Tw1[M + 1]
        z[j] = z[j - 1] + 0.5 * ddz[M] * h1
        aell[j] = 0.5 * bw[M + 1].Lx
        bell[j] = 0.5 * bw[M + 1].Ly
        z[:] = z[:] + zsc[ncols + 1] - h1
    else:
        # No water in the lake: the Fortran leaves Tsoilsurf undefined.
        warnings.warn("TSURFSOILCOL: h1 <= small_value; Tsoilsurf is undefined "
                      "in the Fortran source and is returned as zeros")
        return Tsoilsurf

    # --- area-weighted temperature of each side soil column ---------------
    for i in range(1, ncols):  # except the lowest soil column
        # Grid cells falling into the depth interval of this soil column.
        if zsc[i] < z[0]:
            j0 = 1
        else:
            j0 = None
            for j in range(0, nl + 1):
                if z[j] <= zsc[i] and z[j + 1] > zsc[i]:
                    j0 = j + 1
                    break
            if j0 is None:
                # Fortran reads an unset j0 here; not reachable for the
                # reference geometry (zsc lies inside the water column).
                raise RuntimeError(
                    "TSURFSOILCOL: no grid cell found for the soil column top "
                    f"(i={i}, zsoilcols={zsc[i]:.3f}, z0={z[0]:.3f}, "
                    f"z1={z[nl]:.3f}); the Fortran would use an unset j0")

        j1 = None
        for j in range(j0, nl + 1):
            if z[j] <= zsc[i + 1] and z[j + 1] > zsc[i + 1]:
                j1 = j + 1
                break
        if j1 is None:
            # Same comment as for j0: z(nl+1) is never assigned in the source.
            raise RuntimeError(
                "TSURFSOILCOL: no grid cell found for the soil column bottom "
                f"(i={i}, zsoilcols={zsc[i + 1]:.3f}, z0={z[0]:.3f}, "
                f"z1={z[nl]:.3f}); the Fortran would use an unset j1")

        if j1 <= j0:
            warnings.warn('Warning: water does not flood all soil columns')

        areas = 0.0
        Tsoilsurf[i] = 0.0

        # Top numerical layer overlapping the soil column depth interval.
        if zsc[i] > z[j0 - 1]:
            aells = 0.5 * bs[i].Lx
            bells = 0.5 * bs[i].Ly
            area = C.pi * (aells * bells - aell[j0] * bell[j0])
            Tsoilsurf[i] = Tsoilsurf[i] + Temp[j0] * area
            areas = areas + area
        else:
            area = C.pi * (aell[j0 - 1] * bell[j0 - 1] - aell[j0] * bell[j0])
            Tsoilsurf[i] = Tsoilsurf[i] + Temp[j0] * area
            areas = areas + area

        # Interior numerical layers.
        if j0 + 1 <= j1 - 1:
            for j in range(j0 + 1, j1):
                area = C.pi * (aell[j - 1] * bell[j - 1] - aell[j] * bell[j])
                Tsoilsurf[i] = Tsoilsurf[i] + Temp[j] * area
                areas = areas + area

        # Lowest numerical layer overlapping the depth interval.
        aells = 0.5 * bs[i + 1].Lx
        bells = 0.5 * bs[i + 1].Ly
        area = C.pi * (aell[j1 - 1] * bell[j1 - 1] - aells * bells)
        Tsoilsurf[i] = Tsoilsurf[i] + Temp[j1] * area
        areas = areas + area

        Tsoilsurf[i] = _fdiv(Tsoilsurf[i], areas)

    return Tsoilsurf


# ---------------------------------------------------------------------------
# LATERHEAT
# ---------------------------------------------------------------------------

def laterheat(state: SoilState, params: SoilParams, soilflux, *,
              onlywater=False, lsh: Optional[LatHeatSource] = None):
    """``LATERHEAT`` (soil_mod.f90:1385-1628): lateral heat source from the
    soil columns for the water, ice and deep-ice layers.

    ``soilflux(j)`` is the heat flux into soil column ``j`` (1-based).  Returns a
    :class:`LatHeatSource` (also updated in place when ``lsh`` is given).

    Two quirks of the source are preserved deliberately:

    * the ``forall`` over interior layers also uses ``gsp%zsoilcols(j+1,iy,iy)``
      instead of ``(j+1,ix,iy)`` (a typo in the Fortran; identical for a single
      lake, ``nx = ny = 1``) — marked ``FIXME`` below;
    * the ``i1`` contribution is an assignment (``= +``), not an accumulation.
    """
    geom = state.geom
    if geom is None:
        raise ValueError("laterheat needs state.geom")
    M, Mice, ncols = geom.M, geom.Mice, params.nsoilcols
    zsc = geom.zsoilcols
    h1, l1, ls1 = state.h1, state.l1, state.ls1
    small = C.small_value
    if lsh is None:
        lsh = LatHeatSource(water=np.zeros(M + 2), ice=np.zeros(Mice + 2),
                            dice=np.zeros(Mice + 2))

    # Fortran: maxval(gsp%zsoilcols) over the whole (nsoilcols+1, nx, ny) array.
    # Per lake this is the deepest point of its own zsoilcols profile.
    zmax = float(np.max(zsc[1:ncols + 2]))

    # --- Water layer exists -------------------------------------------------
    if h1 > small:
        offset = zmax - h1 - step(ls1 - small) * ls1
        z = np.zeros(M + 2)
        area = np.zeros(M + 2)
        z[0] = offset
        area[0] = geom.bathymwater[1].area_int
        for i in range(1, M + 1):
            z[i] = offset + geom.z_half[i]
            area[i] = geom.bathymwater[i].area_half
        z[M + 1] = offset + h1
        area[M + 1] = geom.bathymwater[M + 1].area_int

        lsh.water[:] = 0.0
        for j in range(1, ncols):  # waterscan
            zj = zsc[j]
            zj1 = zsc[j + 1]
            if not overlay(z[0], z[M + 1], zj, zj1):
                continue

            i0 = None
            if z[0] <= zj:
                for i in range(1, M + 2):
                    if z[i - 1] < zj and z[i] > zj:
                        i0 = i
                        break
            if i0 is not None:
                lsh.water[i0] = lsh.water[i0] + (
                    1.0 / geom.bathymwater[i0].area_int
                    * (area[i0] - geom.bathymsoil[j].area_int)
                    / (h1 * geom.ddz05[i0 - 1]) * soilflux[j])

            i1 = None
            if z[M + 1] >= zj1:
                for i in range(1, M + 2):
                    if z[i - 1] < zj1 and z[i] > zj1:
                        i1 = i
                        break
            if i1 is not None:
                lsh.water[i1] = (
                    1.0 / geom.bathymwater[i1].area_int
                    * (geom.bathymsoil[j + 1].area_int - area[i1 - 1])
                    / (h1 * geom.ddz05[i1 - 1]) * soilflux[j])

            # forall (i = 1:M+1, z(i-1) >= zsc(j) .and. z(i) <= zsc(j+1))
            # FIXME: the Fortran upper bound is zsoilcols(j+1,iy,iy); for a
            # single lake (nx = ny = 1) this is the same value.
            for i in range(1, M + 2):
                if z[i - 1] >= zj and z[i] <= zj1:
                    lsh.water[i] = (
                        1.0 / geom.bathymwater[i].area_int
                        * (area[i] - area[i - 1])
                        / (h1 * geom.ddz05[i - 1]) * soilflux[j])

            if z[M + 1] <= zj:
                break

    # --- Ice layer exists ---------------------------------------------------
    if l1 > small and not onlywater:
        offset = (zmax - l1 - step(ls1 - small) * ls1
                  - step(h1 - small) * h1)
        z = np.zeros(Mice + 2)
        area = np.zeros(Mice + 2)
        z[0] = offset
        area[0] = geom.bathymice[1].area_int
        for i in range(1, Mice + 2):
            z[i] = z[i - 1] + geom.ddzi05[i - 1] * l1
        for i in range(1, Mice + 1):
            area[i] = geom.bathymice[i].area_half
        area[Mice + 1] = geom.bathymice[Mice + 1].area_int

        lsh.ice[:] = 0.0
        for j in range(1, ncols):  # icescan
            zj = zsc[j]
            zj1 = zsc[j + 1]
            if not overlay(z[0], z[Mice + 1], zj, zj1):
                continue

            i0 = None
            if z[0] <= zj:
                for i in range(1, Mice + 2):
                    if z[i - 1] < zj and z[i] > zj:
                        i0 = i
                        break
            if i0 is not None:
                lsh.ice[i0] = lsh.ice[i0] + (
                    1.0 / geom.bathymice[i0].area_int
                    * (area[i0] - geom.bathymsoil[j].area_int)
                    / (l1 * geom.ddzi05[i0 - 1]) * soilflux[j])

            i1 = None
            if z[Mice + 1] >= zj1:
                for i in range(1, Mice + 2):
                    if z[i - 1] < zj1 and z[i] > zj1:
                        i1 = i
                        break
            if i1 is not None:
                lsh.ice[i1] = (
                    1.0 / geom.bathymice[i1].area_int
                    * (geom.bathymsoil[j + 1].area_int - area[i1 - 1])
                    / (l1 * geom.ddzi05[i1 - 1]) * soilflux[j])

            # FIXME: same zsoilcols(j+1,iy,iy) typo as in the water scan.
            for i in range(1, Mice + 2):
                if z[i - 1] >= zj and z[i] <= zj1:
                    lsh.ice[i] = (
                        1.0 / geom.bathymice[i].area_int
                        * (area[i] - area[i - 1])
                        / (l1 * geom.ddzi05[i - 1]) * soilflux[j])

            if z[Mice + 1] <= zj:
                break

    # --- Deep ice layer exists ---------------------------------------------
    if ls1 > small and not onlywater:
        offset = zmax - ls1
        z = np.zeros(Mice + 2)
        area = np.zeros(Mice + 2)
        z[0] = offset
        area[0] = geom.bathymdice[1].area_int
        for i in range(1, Mice + 2):
            z[i] = z[i - 1] + geom.ddzi05[i - 1] * ls1
        for i in range(1, Mice + 1):
            area[i] = geom.bathymdice[i].area_half
        area[Mice + 1] = geom.bathymdice[Mice + 1].area_int

        lsh.dice[:] = 0.0
        for j in range(1, ncols):  # dicescan
            zj = zsc[j]
            zj1 = zsc[j + 1]
            if not overlay(z[0], z[Mice + 1], zj, zj1):
                continue

            i0 = None
            if z[0] <= zj:
                for i in range(1, Mice + 2):
                    if z[i - 1] < zj and z[i] > zj:
                        i0 = i
                        break
            if i0 is not None:
                lsh.dice[i0] = lsh.dice[i0] + (
                    1.0 / geom.bathymdice[i0].area_int
                    * (area[i0] - geom.bathymsoil[j].area_int)
                    / (ls1 * geom.ddzi05[i0 - 1]) * soilflux[j])

            i1 = None
            if z[Mice + 1] >= zj1:
                for i in range(1, Mice + 2):
                    if z[i - 1] < zj1 and z[i] > zj1:
                        i1 = i
                        break
            if i1 is not None:
                lsh.dice[i1] = (
                    1.0 / geom.bathymdice[i1].area_int
                    * (geom.bathymsoil[j + 1].area_int - area[i1 - 1])
                    / (ls1 * geom.ddzi05[i1 - 1]) * soilflux[j])

            # FIXME: same zsoilcols(j+1,iy,iy) typo as in the water scan.
            for i in range(1, Mice + 2):
                if z[i - 1] >= zj and z[i] <= zj1:
                    lsh.dice[i] = (
                        1.0 / geom.bathymdice[i].area_int
                        * (area[i] - area[i - 1])
                        / (ls1 * geom.ddzi05[i - 1]) * soilflux[j])

            if z[Mice + 1] <= zj:
                break

    return lsh


# ---------------------------------------------------------------------------
# SOILCOLSTEMP
# ---------------------------------------------------------------------------

def soilcolstemp(state: SoilState, params: SoilParams, dt, contr=(1, 0)):
    """``SOILCOLSTEMP`` (soil_mod.f90:804-1031), heat part.

    ``bcswitch = 1`` (compile-time constant): continuity of flux and
    temperature across the soil-water interface, ``soilcolconjtype = 2`` (the
    testlake setup) so the surface temperature of the side columns comes from
    :func:`tsurfsoilcol`.  For every side column ``is = 1 .. nsoilcols-1`` the
    sequence is the Fortran one::

        SOIL_COND_HEAT_COEF(is)
        T_DIFF(surf=1, Tsoilsurf(is), dt, 0, 0, 0, 0, is, .false.)
        soilflux(is) = ...
        SOILFORLAKE(nveclen, dt, a, b, c, d, is)

    ``contr=(1, 0)`` is the heat pass; ``contr=(0, 1)`` (methane in the soil
    columns) is **not ported** — it calls ``METHANE`` from ``methane_mod.f90``
    and raises ``NotImplementedError``.  The deepest column ``nsoilcols`` is
    *not* touched here (``lake.f90`` handles it after ``T_SOLVER``).
    """
    geom = state.geom
    if geom is None:
        raise ValueError("soilcolstemp needs state.geom")
    if state.Tw1 is None:
        raise ValueError("soilcolstemp needs state.Tw1 (water temperatures)")
    ncols = params.nsoilcols
    dt_inv = 1.0 / dt

    if contr[0] == 1:
        # --- heat equation in the soil columns ----------------------------
        # Tsoilsurf for the side columns, computed once before the loop.
        if params.soilcolconjtype == 1:
            Tsoilsurf = np.zeros(ncols + 1)
            for is_ in range(1, ncols):
                i = geom.bathymsoil[is_].icent
                Tsoilsurf[is_] = state.Tw1[i]
        elif params.soilcolconjtype == 2:
            Tsoilsurf = tsurfsoilcol(state, params)
        else:
            raise ValueError(f"unknown soilcolconjtype {params.soilcolconjtype}")

        for is_ in range(1, ncols):
            state.isoilcol = is_
            soil_cond_heat_coef(state, params, is_)
            state.Tsoil2[:] = _t_diff_soil(state, params, Tsoilsurf[is_], is_,
                                          dt, ifflux=False)
            # Heat flux into the soil, W/m2 (soil_mod.f90:929-932).
            state.soilflux[is_] = (
                state.csoil[1] * state.rosoil[1] * dt_inv * 0.5 * params.dzs[1]
                * (state.Tsoil2[1] - state.Tsoil1[1, is_])
                - 0.25 * (state.lamsoil[1] + state.lamsoil[2])
                * (state.Tsoil2[2] + state.Tsoil1[2, is_]
                   - state.Tsoil2[1] - state.Tsoil1[1, is_]) / params.dzs[1])
            soilforlake(state, params, dt, is_)

    if contr[1] == 1:
        # UNREACHABLE / OUT OF SCOPE: methane in the soil columns.  The Fortran
        # branch calls METHANE (methane_mod.f90) for every side column and
        # fills fdiffbot; that module is not part of this transcription.
        raise NotImplementedError(
            "SOILCOLSTEMP: contr(2)=1 (methane in the soil columns) is not "
            "ported; it requires METHANE from methane_mod.f90")

    return state


def water_freeze_melt(temperature, grid_size, melting_point, switch):
    """``WATER_FREEZE_MELT`` (phys_func.f90:859-910).

    ``switch=+1`` checks water -> ice (heat deficit exceeds the latent heat of
    ``min_ice_thick``, or the water is more than 0.2 K below freezing);
    ``switch=-1`` checks ice -> water.
    """
    if switch == 1:
        d = melting_point - C.T_phase_threshold - temperature
        return (d * C.cw_m_row0 * grid_size > C.min_ice_thick * C.roi_m_Lwi
                or d > 0.2)
    if switch == -1:
        d = temperature - C.T_phase_threshold - melting_point
        return (d * C.ci_m_roi * grid_size > C.min_water_thick * C.row0_m_Lwi
                or d > 0.2)
    raise ValueError(f"WATER_FREEZE_MELT: bad switch {switch!r}")
