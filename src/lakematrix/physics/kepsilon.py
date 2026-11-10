"""k-epsilon turbulence closure: `K_EPSILON` and every procedure it calls.

Fortran source: ``lakebatch/core/source/model/turb_mod.f90``.

Ported procedures (Fortran line ranges in brackets):

=========================================  =========================
``SUBROUTINE K_EPSILON``                   [10-1158]
``SUBROUTINE CHECK_MIN_VALUE``             [1161-1178]
``FUNCTION CE_CANUTO``                     [1181-1224]
``FUNCTION CEt_CANUTO``                    [1228-1269]
``FUNCTION SMOMENT_GALPERIN``              [1272-1317]
``FUNCTION SHEAT_GALPERIN``                [1320-1364]
``SUBROUTINE VSMOP_LAKE``                  [1367-1441]
``FUNCTION INTERPOLATE_TO_INTEGER_POINT``  [1444-1461]
``FUNCTION TKE_FLUX_BUOY``                 [1464-1494]
``FUNCTION DISSIPATION_FLUX_BUOY``         [1497-1535]
``FUNCTION TKE_FLUX_SHEAR``                [1538-1579]
``FUNCTION DISSIPATION_FLUX_SHEAR``        [1582-1668]
``FUNCTION TKE_WALL_SHEAR``                [1673-1690]
``FUNCTION DISSIPATION_WALL_SHEAR``        [1696-1719]
=========================================  =========================

Procedures used by ``K_EPSILON`` that live in other Fortran modules are also
reproduced here because they are tiny and are part of the closure's physics:

* ``H13`` (significant wave height), ``phys_func.f90:1985-2004``;
* ``SEICHE_ENERGY``, ``seiches_param_mod.f90:13-56``;
* ``IND_STAB_FACT_DB``, ``numerics_mod.f90:183-207`` (only the ``N=1``,
  ``ind_bound=.false.`` call used here is exercised; the general form is kept).

Two helpers are re-used from the rest of :mod:`lakematrix` instead of being
copied: ``PROGONKA`` (:func:`lakematrix.physics.tridiag.progonka`) and the
density derivatives ``SET_DDENS_DTEMP``/``SET_DDENS_DSAL``
(:func:`lakematrix.physics.density.ddens_dtemp` / ``ddens_dsal``, which
implement the ``eos 5``/``lindens 0`` path used by testlake).

Conventions (same as :mod:`lakematrix.physics.tsolver`)
------------------------------------------------------
* NumPy, **not** jax.numpy: this is the v1 reference implementation.  Scalar
  inputs therefore yield ``np.float64``/Python floats.
* Arrays are **1-based** exactly like Fortran: element 0 is unused.  Arrays
  that Fortran declares ``1:M+1`` have length ``M+2``; arrays declared
  ``1:M`` have length ``M+1``.
* Loop order, iteration counts, convergence tests and every ``max``/``min``
  are transcribed verbatim.  No clipping/smoothing/renumbering was added.
* **Float kinds are preserved.**  A literal with a ``d`` exponent (``1.d-20``,
  ``0.75d0``) is a ``real(8)`` literal and is written as a plain Python float.
  A literal without one (``0.01``, ``1.E-5``, ``2./3.``, ``0.2``) is a
  *default real* (single precision) literal in Fortran; it is first rounded to
  ``real(4)`` and only then stored in a ``real(8)`` variable, so it is written
  here as ``_s(...)`` = ``float(np.float32(...))``.  Every nontrivial rounding
  is flagged with a comment.

Notable precision detail: there are **two** Karman constants in the Fortran.
``TURB_CONST::kar = 0.4d0`` is an exact double, while
``PHYS_CONSTANTS::kappa = 0.4`` is a default-real literal rounded to float32
(``0.4000000059604645``).  ``K_EPSILON`` uses ``kar`` everywhere except in
``DISSIPATION_WALL_SHEAR``, which uses ``kappa``; both are kept distinct.

Branches marked "untested default-off" are transcribed but are not exercised
by the testlake configuration (``Turbpar 2``, ``kepsbc 1``, ``stabfunc 2``,
``kwe 100``, ``nstep_keps 1``, ``M 40``, ``massflux 0``, ``waveenh 0``,
``relwind 1``, ``zserout -999``, no ice).  ``massflux``, ``waveenh`` and
``relwind`` are *not referenced at all* in ``turb_mod.f90``; they only act
upstream on ``tau_air``/``tau_wav``/``roughness`` which enter as arguments.

Usage
-----
::

    from lakematrix.physics import kepsilon

    st = kepsilon.KEpsilonState(e1=..., eps1=..., tw1=..., ddz=..., ...)
    st.init_arrays(M)                    # allocate the remaining arrays
    params = kepsilon.KEpsilonParams(M=M)      # testlake switches
    for _ in range(nstep_keps):          # lake.f90 calls it nstep_keps times
        st = kepsilon.k_epsilon(st, dt, dt_keps, params=params)

The input state is not modified (it is deep-copied), so the returned object
must be kept.

Known deviations / TODOs (all outside the default path unless stated)
----------------------------------------------------------------------
* ``turb_out == 1`` writes ``results/debug/err_keps_iter.dat`` in the Fortran;
  file I/O is not ported.
* ``DISSIPATION_FLUX_BUOY``: the Fortran declares no ``small_value`` (no
  ``implicit none``), so it reads an undefined implicitly-typed real.  We use
  ``NUMERIC_PARAMS::small_value = 1.d-20``; marked ``TODO`` at the function.
* ``AG(1:5)`` and ``DTDZH`` (``contrgrad`` branch) are never assigned in the
  Fortran -- their assignments are commented out.  They are state fields
  defaulting to 0 and the branch is marked ``TODO: unclear``.
* ``TKE_FLUX_SHEAR`` / ``DISSIPATION_FLUX_SHEAR`` have no ``case`` for
  ``nbc = 4/5``; the Fortran result is undefined there (reachable only for
  fractional ice with ``kepsbc 4/5``).  A ``NotImplementedError`` is raised.
* ``FTKES``/``FTKEB`` are SAVE locals that are *never* set when ``kepsbc 5``
  is active, so the ``TKE_turb_trans(1)``/``(M)`` diagnostics use stale values
  there (the Fortran has the same defect).  State fields default to 0.
* The ``zserout`` search loop reads ``z_half`` out of bounds if ``zserout`` is
  above ``z_half(1)`` or below ``z_half(M)``; an ``IndexError`` is raised
  instead of the Fortran's undefined read.
* Density derivatives are only ported for ``eos 5`` / ``lindens 0`` (testlake);
  other combinations raise ``NotImplementedError``.
* ``Ri``, ``k_turb_T_flux``, ``uv``, ``nstep``, ``time`` and ``ls`` are
  imported by the Fortran routine but never used in it, so they are not state
  fields here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import C
from .density import ddens_dsal, ddens_dtemp
from .tridiag import progonka

__all__ = [
    "KEpsilonState",
    "KEpsilonParams",
    "k_epsilon",
    "check_min_value",
    "vsmop_lake",
    "interpolate_to_integer_point",
    "tke_flux_buoy",
    "dissipation_flux_buoy",
    "tke_flux_shear",
    "dissipation_flux_shear",
    "tke_wall_shear",
    "dissipation_wall_shear",
    "ce_canuto",
    "cet_canuto",
    "smoment_galperin",
    "sheat_galperin",
    "h13",
    "seiche_energy",
    "ind_stab_fact_db",
    "sigmaeps",
    "step",
]


def _s(x) -> float:
    """Value of a default-real Fortran literal assigned to a real(8) variable."""
    return float(np.float32(x))


# ---------------------------------------------------------------------------
# TURB_CONST            (lake_modules.f90:801-939)
# ---------------------------------------------------------------------------
# `kar` is `0.4d0`: an exact double, unlike PHYS_CONSTANTS::kappa below.
kar = 0.4                      # 0.4d0

CE0 = 0.09                     # 0.09d0
CEt0 = 0.072                   # 0.072d0
ceps1 = 1.44                   # 1.44d0
ceps2 = 1.92                   # 1.92d0
ceps3_unstable = 1.14          # 1.14d0
ceps3_stable = -0.4            # -0.4d0
sigmaE = 1.0                   # 1.d0

kar_tilde = 0.2                # 2.d-1, Burchard (2002), p. 80

CL = CE0 ** 0.75               # CE0**(0.75d0), set in TURB_CONST_DERIVED_SET

# alpham, TURB_CONST_DERIVED_SET; also recomputed locally in TKE_FLUX_SHEAR
alpham = (1.5 * CE0 ** 0.5 * sigmaE * kar_tilde ** (-2)) ** 0.5

lam_E0 = CE0 / sigmaE          # lam_E0 = CE0/sigmaE (module parameter)
# (other TURB_CONST constants -- lam_T, lam_gen, lamTM, C_wstar, CDN10m,
#  z_CDN, the KC*/Cs*/dtdz0 convection block -- are not used by K_EPSILON and
#  are omitted; add them when porting CONVECTION/MOMENTUM.)

# Coefficients of the Canuto et al. (2001) stability functions
CEcoef01 = 0.127               # 0.127d0
CEcoef02 = 0.1526e-1           # 0.1526d-1
CEcoef03 = 0.16e-3             # 0.16d-3
CEtcoef01 = 0.1190             # 0.1190d0
CEtcoef02 = 0.4294e-2          # 0.4294d-2
CEtcoef03 = 0.66e-3            # 0.66d-3
CEcoef11 = 1.0                 # 1.d0
CEcoef12 = 0.2                 # 0.2d0
CEcoef13 = 0.315e-1            # 0.315d-1
CEcoef14 = 0.58e-2             # 0.58d-2
CEcoef15 = 0.4e-2              # 0.4d-2
CEcoef16 = 0.4e-4              # 0.4d-4

# Mellor and Yamada (1982) constants
A1 = 0.92                      # 0.92d0
B1 = 1.66e1                    # 1.66d+1
C1 = 0.8e-1                    # 0.8d-1
A2 = 0.74                      # 0.74d0
B2 = 1.01e1                    # 1.01d+1

# CL_K_KL_MODEL = 2.d0**(1.5d0)/B1, TURB_CONST_DERIVED_SET
CL_K_KL_MODEL = 2.0 ** 1.5 / B1

Co = 1.9                       # 1.9d0, Satyanarayana et al. (1999)

L0 = 0.1                       # 0.1d0

CON0 = 0.05                    # 0.05d0
CON1 = 0.11                    # 0.11d0
CON2 = 1.56                    # 1.56d0
CONUV = 1.0                    # 1.d0

grav_wave_Gill_const = _s(0.7)  # 0.7 -- *default real*: float32(0.7) != 0.7d0

#Minimal turbulent viscosity/diffusivity coefficient, m**2/s
min_diff = 1.0e-10             # 1.d-10
Pr_stable = 1.0                # 1. ; turbulent Prandtl number at very stable
min_visc = min_diff * Pr_stable

# ---------------------------------------------------------------------------
# SEICHES_PARAM         (seiches_param_mod.f90:7-10)
# ---------------------------------------------------------------------------
# All three are default-real literals in the Fortran.
C_Deff = _s(2.0e-3)            # 2.e-3
alpha_seiche = _s(2.0e-3)      # 2.e-3
cnorm = 1.0                    # 1.

# ---------------------------------------------------------------------------
# Local constants of K_EPSILON that the Fortran sets up in its `firstcall`
# block and then treats as SAVE (turb_mod.f90:282-311, 335-351).
# ---------------------------------------------------------------------------
#: `.true.` : no iterations, semi-implicit scheme.  This is the value the
#: Fortran hard-codes; :class:`KEpsilonParams` carries the effective switch.
SEMI_IMPLICIT_SCHEME = True
AL_IT = _s(0.2)                # al_it   = 0.2      (default real!)
ACC = 1.0e-20                  # ACC     = 1.d-20
ACC2 = 1.0e-20                 # ACC2    = 1.d-20
ACCK = 1.0e-20                 # ACCk    = 1.d-20
Z0 = 1.0e-2                    # z0      = 1.d-2
EPS_MIN0 = 1.0e-12             # eps_min0 = 1.d-12
# E_min_lamw0 = sqrt(eps_min0*1.E-5*lamw0*(cw_m_row0*CEt0)**(-1))
# NOTE the `1.E-5` literal is default real -> float32 rounding.
E_MIN_LAMW0 = math.sqrt(EPS_MIN0 * _s(1.0e-5) * C.lamw0
                        * (C.cw_m_row0 * CEt0) ** (-1))
D_E_IT_MAX = 1.0e-11           # dE_it_max  = 1.d-11
DEPS_IT_MAX = 1.0e-15          # deps_it_max = 1.d-15
VDAMP = 1.0e-3                 # vdamp = 1.d-3

#: `pi = 4.*atan(1.e0_ireals)` is evaluated in *single* precision
#: (turb_mod.f90:276) before being stored in the real(8) local `pi`, which the
#: routine never uses.  Kept to document the rounding; not used below.
PI_LOCAL = float(np.float32(4.0) * np.arctan(np.float32(1.0)))

MONTH_SEC = 30.0 * 24.0 * 60.0 * 60.0   # 30.*24.*60.*60. ; exact = 2592000.
DAY_SEC = float(24 * 60 * 60)           # 24*60.*60.        ; exact = 86400.
HOUR_SEC = 60.0 * 60.0                  # 60.*60.           ; exact = 3600.

#: `real(8), parameter :: maxiter = 35` (turb_mod.f90:211)
MAXITER = 35

#: `DRIVING_PARAMS::missing_value = -999.` (driving_params_mod.f90:30)
MISSING_VALUE = _s(-999.0)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def step(x):
    """Heavyside step, `NUMERICS::STEP` (numerics_mod.f90:280-288).

    ``STEP = 0.5*(sign(1,x) + 1)``.  Fortran ``SIGN(1,x)`` is ``+1`` for
    ``x >= 0`` (including ``+0.0``), unlike :func:`numpy.sign`.
    """
    return 0.5 * ((1.0 if x >= 0.0 else -1.0) + 1.0)


def _fsign1(x):
    """Fortran ``sign(1._ireals, x)``: +1 for x >= 0 (including 0), else -1."""
    return 1.0 if x >= 0.0 else -1.0


def check_min_value(E, N, threshold):
    """`CHECK_MIN_VALUE` (turb_mod.f90:1161-1178).

    ``E(i) = max(E(i), threshold)`` for ``i = 1..N``; in place, 1-based.
    """
    for i in range(1, N + 1):
        E[i] = max(E[i], threshold)


def interpolate_to_integer_point(fim05, fip05, ddzim1, ddzi):
    """`INTERPOLATE_TO_INTEGER_POINT` (turb_mod.f90:1444-1461).

    Linear interpolation from two half-integer points to the integer point
    between them::

        f(i) = f(i-1/2) + (f(i+1/2) - f(i-1/2))*ddz(i-1)/(ddz(i-1)+ddz(i))
    """
    return fim05 + (fip05 - fim05) * ddzim1 / (ddzim1 + ddzi)


def ind_stab_fact_db(a, b, c, N, M, ind_bound=False):
    """`IND_STAB_FACT_DB` (numerics_mod.f90:183-207) -> ``ind_stab``.

    Sufficient condition for the tridiagonal matrix to be diagonal-dominant;
    ``a``, ``b``, ``c``, ``d`` are 1-based.  ``K_EPSILON`` calls it as
    ``IND_STAB_FACT_DB(nveclen,a,b,c,1,M,indstab,ind_bound)`` with
    ``ind_bound = .false.``, i.e. only the interior loop is active.
    """
    if ind_bound:
        if abs(b[N]) >= abs(c[N]) or abs(a[M]) >= abs(c[M]):
            return False
    for i in range(N + 1, M):
        if abs(a[i]) + abs(b[i]) >= abs(c[i]):
            return False
    return True


# ---------------------------------------------------------------------------
# Stability functions
# ---------------------------------------------------------------------------
def ce_canuto(lambda_M, lambda_N):
    """`CE_CANUTO` (turb_mod.f90:1181-1224): stability function for momentum."""
    upper_limit = 0.46             # 0.46d0
    small_number = 1.0e-20         # 1.d-20
    lambda_N_low_limit = -5.9      # -5.9d0

    lambda_N1 = max(lambda_N_low_limit, lambda_N)
    val = ((CEcoef01 + CEcoef02 * lambda_N1 - CEcoef03 * lambda_M)
           / (CEcoef11 + CEcoef12 * lambda_N1 + CEcoef13 * lambda_M
              + CEcoef14 * lambda_N1 * lambda_N1
              + CEcoef15 * lambda_N1 * lambda_M
              - CEcoef16 * lambda_M * lambda_M))
    val = max(small_number, val)
    val = min(upper_limit, val)
    return val


def cet_canuto(lambda_M, lambda_N):
    """`CEt_CANUTO` (turb_mod.f90:1228-1269): stability function for scalars."""
    upper_limit = 0.61             # 0.61d0
    small_number = 1.0e-20         # 1.d-20
    lambda_N_low_limit = -5.9      # -5.9d0

    lambda_N1 = max(lambda_N, lambda_N_low_limit)
    val = ((CEtcoef01 + CEtcoef02 * lambda_N1 + CEtcoef03 * lambda_M)
           / (CEcoef11 + CEcoef12 * lambda_N1 + CEcoef13 * lambda_M
              + CEcoef14 * lambda_N1 ** 2
              + CEcoef15 * lambda_N1 * lambda_M
              - CEcoef16 * lambda_M ** 2))
    val = max(small_number, val)
    val = min(upper_limit, val)
    return val


def smoment_galperin(lambda_N):
    """`SMOMENT_GALPERIN` (turb_mod.f90:1272-1317).

    Equilibrium stability function for momentum, Galperin et al. (1988).
    """
    small_number = _s(1.0e-2)      # 1.d-2 -> default real!
    GH_MAX = (A2 * (12.0 * A1 + B1 + 3.0 * B2)) ** (-1)   # Galperin et al., 1988
    GH_MIN = -0.28                 # -0.28d0

    GH = -lambda_N                 # opposite sign to lambda_N (Mellor & Yamada)
    GH = CL_K_KL_MODEL ** 2 * 0.5 * GH
    GH = min(GH, GH_MAX)
    GH = max(GH, GH_MIN)

    val = 1.0 - 3.0 * C1 - 6.0 * A1 / B1 - 3.0 * A2 * GH * (
        (B2 - 3.0 * A2) * (1.0 - 6.0 * A1 / B1) - 3.0 * C1 * (B2 + 6.0 * A1))
    val = val / ((1.0 - 3.0 * A2 * GH * (6.0 * A1 + B2))
                 * (1.0 - 9.0 * A1 * A2 * GH))
    val = val * A1
    val = max(small_number, val)
    return val


def sheat_galperin(lambda_N):
    """`SHEAT_GALPERIN` (turb_mod.f90:1320-1364).

    Equilibrium stability function for heat, Galperin et al. (1988).
    """
    small_number = _s(1.0e-2)      # 1.d-2 -> default real!
    GH_MAX = (A2 * (12.0 * A1 + B1 + 3.0 * B2)) ** (-1)   # Galperin et al., 1988
    GH_MIN = -0.28                 # -0.28d0

    GH = -lambda_N
    GH = CL_K_KL_MODEL ** 2 * 0.5 * GH
    GH = min(GH, GH_MAX)
    GH = max(GH, GH_MIN)

    val = 1.0 - 6.0 * A1 / B1
    val = val / (1.0 - 3.0 * A2 * GH * (6.0 * A1 + B2))
    val = val * A2
    val = max(small_number, val)
    return val


# ---------------------------------------------------------------------------
# Boundary-condition fluxes
# ---------------------------------------------------------------------------
def tke_flux_buoy(Buoyancy_flux, roughness_length, H_entrainment):
    """`TKE_FLUX_BUOY` (turb_mod.f90:1464-1494).

    Free-convection TKE flux at the water surface.  The Fortran computes the
    coefficient ``const = 2.d0/3.d0*CE0*kar*kar/(sigmaE*CL*CL)`` on the first
    call but the actual expression is commented out and the function returns
    exactly zero (an unfinished parameterisation).  Nothing depends on
    ``const``, so it is not evaluated here.
    """
    # const = 2.d0/3.d0*CE0*kar*kar/(sigmaE*CL*CL)   # computed, never used
    # TKE_FLUX_BUOY = - const*Buoyancy_flux*roughness_length* &
    #                 (1.d0-2.d0*roughness_length/H_entrainment)   # commented out
    return 0.0


def dissipation_flux_buoy(Buoyancy_flux, roughness_length, H_entrainment,
                          dBdz0, TKE0, sigmaeps_=None):
    """`DISSIPATION_FLUX_BUOY` (turb_mod.f90:1497-1535).

    ``DISSIPATION_FLUX_BUOY = -CE0*TKE0**2*dBdz0/((Buoyancy_flux+small_value)*sigmaeps)``

    TODO(ambiguity): the Fortran function has **no** ``implicit none`` and no
    declaration of ``small_value``; with implicit typing it becomes an
    undefined local ``real(4)``.  ``small_value = 1.d-20`` from
    ``NUMERIC_PARAMS`` is the evident intent and is used here.  (Only reached
    for ``kepsbc 4``, a default-off branch.)
    """
    if sigmaeps_ is None:
        sigmaeps_ = sigmaeps(1)
    small_value = 1.0e-20          # NUMERIC_PARAMS::small_value (see TODO above)
    # const = CE0*(kar/CL)**(4.d0/3.d0)/sigmaeps   # computed, never used
    return (-CE0 * TKE0 * TKE0 * dBdz0
            / ((Buoyancy_flux + small_value) * sigmaeps_))


def tke_flux_shear(momentum_flux, kwe, maxTKEinput, nbc):
    """`TKE_FLUX_SHEAR` (turb_mod.f90:1538-1579).

    TKE flux at the surface for a logarithmic shear flow (Burchard and
    Petersen, 1999) or for breaking waves (Craig and Banner, 1994).
    """
    alpham_ = (1.5 * CE0 ** 0.5 * sigmaE * kar_tilde ** (-2)) ** 0.5
    const0 = (1.5 * sigmaE) ** 0.5 * CE0 ** 0.25
    # 2./3. is evaluated in single precision before the promotion to real(8)
    const = _s(2.0 / 3.0) * CE0 ** (-0.5) * kar * const0 * alpham_ / sigmaE

    if nbc == 1:
        # for logarithmic profile (shear only)
        return 0.0
    elif nbc == 2:
        # for breaking waves case, Craig and Banner (1994)
        # (min(...) with maxTKEinput is commented out in the Fortran)
        return kwe * (momentum_flux / C.row0) ** 1.5
    elif nbc == 3:
        # Generalized b.c. for sheared flow with wave breaking (Burchard, 2002)
        return min(const * kwe * (momentum_flux / C.row0) ** 1.5, maxTKEinput)
    # The Fortran `select case (nbc)` has no default: for nbc = 4 or 5 the
    # function result is left undefined.  K_EPSILON only reaches that for the
    # fractional-ice branch with kepsbc 4/5 (default-off), so this is raised
    # instead of returning garbage.  TODO: undefined in the Fortran.
    raise NotImplementedError(
        f"TKE_FLUX_SHEAR has no case for nbc={nbc}; the Fortran result is undefined")


def dissipation_flux_shear(momentum_flux, kturb, Esurf_in, z0_in,
                           kwe, fetch, maxTKEinput, nbc, signwaveheight,
                           sigmaeps_=None):
    """`DISSIPATION_FLUX_SHEAR` (turb_mod.f90:1582-1668).

    TKE dissipation flux at the surface for a logarithmic shear flow
    (Burchard, 2000/2001/2002).  ``fetch``, ``maxTKEinput`` and
    ``signwaveheight`` are dummy arguments that the Fortran never uses.
    """
    if sigmaeps_ is None:
        sigmaeps_ = sigmaeps(1)
    small_number = 1.0e-20         # 1.d-20
    const = 1.0 / sigmaeps_
    const1 = CE0 ** 0.75 / (sigmaeps_ * kar)
    const2 = const1 / kar
    const3 = 1.5 * sigmaE * CE0 ** 0.75 / CE0   # CE0 (not CE) in the denominator
    a = 1.0                        # switch for shear (0 or 1)
    z0s = 1.0e-2                   # 1.d-2 ; signwaveheight*0.85 is commented out

    if nbc == 1:
        # From logarithmic profile (sheared flow without wave breaking)
        Esurf = Esurf_in
        return const1 * kturb * Esurf ** 1.5 / (z0_in * z0_in + small_number)
    elif nbc == 2:
        # Dissipation flux in flow with breaking waves (Burchard, 2001)
        const4 = (1.5 * sigmaE) ** 0.5 * CE0 ** 0.25 * kwe
        Esurf = momentum_flux / C.row0 / (CE0 ** 0.5) * const4 ** _s(2.0 / 3.0)
        xx = kwe * (momentum_flux / C.row0) ** 1.5
        return (const2 * (momentum_flux / C.row0) ** 0.5
                * kar * z0s * const4 ** _s(1.0 / 3.0)
                * (const3 * xx + kar * Esurf ** 1.5)
                / (z0s * z0s + small_number))
    elif nbc == 3:
        # Generalized b.c. for sheared flow with wave breaking (Burchard, 2002)
        const4 = (1.5 * sigmaE) ** 0.5 * CE0 ** 0.25 * kwe
        const5 = (sigmaeps_ ** (-1) * (a + const4) ** _s(1.0 / 3.0)
                  * ((a + const4) / (z0s + small_number) + alpham * const4))
        return const5 * (momentum_flux / C.row0) ** 2
    # See the note in tke_flux_shear: nbc = 4/5 leaves the Fortran result
    # undefined (reached only for fractional ice with kepsbc 4/5).
    raise NotImplementedError(
        f"DISSIPATION_FLUX_SHEAR has no case for nbc={nbc}; the Fortran "
        "result is undefined")


def tke_wall_shear(momentum_flux):
    """`TKE_WALL_SHEAR` (turb_mod.f90:1673-1690)."""
    return (momentum_flux / C.row0) / math.sqrt(CE0)


def dissipation_wall_shear(momentum_flux, buoyancy_flux, dist):
    """`DISSIPATION_WALL_SHEAR` (turb_mod.f90:1696-1719).

    Uses ``PHYS_CONSTANTS::kappa`` (float32-rounded 0.4), not ``kar``.
    """
    return (momentum_flux / C.row0) ** 1.5 / (C.kappa * dist) + buoyancy_flux


def h13(momentum_flux, fetch):
    """`H13` (phys_func.f90:1985-2004): significant wave height, Hasselmann."""
    const = _s(0.051)              # real(8), parameter :: const = 0.051
    return const * math.sqrt(momentum_flux * fetch / (C.g * C.roa0))


def seiche_energy(area_int_1, vol, tau, wind, dt, Eseiches):
    """`SEICHE_ENERGY` (seiches_param_mod.f90:13-56).

    Integral seiche energy at the next time step (Goudsmit et al., 2002),
    integrated with the implicit Patankar (1980) scheme.

    Returns ``(Eseiches, Eseiches_diss)``.  The Fortran's ``bathymwater(1)%``
    prefix means the *surface* area ``area_int(1)``; the ``ls`` argument of the
    Fortran routine is never used.
    """
    Eseiches_old = Eseiches
    gamma_ = C_Deff * area_int_1 * vol ** (-1.5) * C.row0 ** (-0.5)
    # Implicit Patankar scheme (1980)
    beta = dt * math.sqrt(Eseiches_old) * gamma_
    Eseiches_new = ((Eseiches_old + dt * alpha_seiche * area_int_1 * tau * wind)
                    / (1.0 + beta))
    Eseiches_diss = gamma_ * Eseiches_new * math.sqrt(Eseiches_old)
    return Eseiches_new, Eseiches_diss


def vsmop_lake(f, fs, nmin, nmax, k0, k1, gama, perdam):
    """`VSMOP_LAKE` (turb_mod.f90:1367-1441).

    4th-order vertical smoothing; 2nd order for the grid points adjacent to
    the boundary, no smoothing at the boundary points themselves.  If
    ``perdam`` is true only the perturbation ``f - fs`` is smoothed.

    ``nmin``/``nmax`` only size the internal ``work`` array in the Fortran;
    they are kept for signature fidelity.  ``f`` is modified in place.
    """
    um6g = 1.0 - 6.0 * gama
    gasc = 8.0 * gama
    gasc05 = 0.5 * gasc
    umgasc = 1.0 - gasc

    if perdam:
        for i in range(k0, k1 + 1):
            f[i] = f[i] - fs[i]

    work = np.zeros(len(f))
    for i in range(k0 + 2, k1 - 1):
        work[i] = (um6g * f[i]
                   - gama * (f[i + 2] + f[i - 2] - 4.0 * (f[i + 1] + f[i - 1])))

    # do i = k0+1, k1-1, k1-k0-2 : the stride is a computed, exact Fortran
    # stride (with k0=1, k1=M it hits i = 2 and i = M-1, i.e. the two points
    # adjacent to the boundary, which get the 2nd-order filter).
    stride = k1 - k0 - 2
    if stride == 0:
        # Fortran: a zero stride is illegal/undefined
        raise ValueError("VSMOP_LAKE: zero DO stride (k1-k0-2 == 0)")
    i = k0 + 1
    while i <= k1 - 1:
        work[i] = umgasc * f[i] + gasc05 * (f[i - 1] + f[i + 1])
        i += stride

    for i in range(k0 + 1, k1):
        f[i] = work[i]

    if perdam:
        for i in range(k0, k1 + 1):
            f[i] = f[i] + fs[i]


# ---------------------------------------------------------------------------
# TURB_CONST derived quantities that depend on the switch `kepsbc`
# ---------------------------------------------------------------------------
def sigmaeps(kepsbc):
    """``sigmaeps`` from `TURB_CONST_DERIVED_SET` (lake_modules.f90:917-930).

    ``kepsbc 1``: law of the wall in the k-eps model (Burchard, 2002);
    ``kepsbc 2``: analytic solution for unstratified shear-free flow with wave
    breaking (Burchard, 2002); everything else: 1.3.
    """
    if kepsbc == 1:
        return kar * kar / (CE0 ** 0.5 * (ceps2 - ceps1))
    elif kepsbc == 2:
        # 4./3. is a default-real expression -> float32
        return (_s(4.0 / 3.0) * alpham + 1.0) * (alpham + 1.0) * kar_tilde * kar_tilde \
            / (ceps2 * CE0 ** 0.5)
    return 1.3                     # 1.3d0


def lam_eps0(kepsbc):
    """``lam_eps0 = CE0/sigmaeps`` (``TURB_CONST`` module variable)."""
    return CE0 / sigmaeps(kepsbc)


# ---------------------------------------------------------------------------
# State and configuration
# ---------------------------------------------------------------------------
#: All ``1:M+1`` arrays of ``KEpsilonState`` (Fortran bound ``1:M+1``).
_ARR_M1 = (
    "e1", "e2", "eps1", "eps2", "e_it1", "e_it2", "eps_it1", "eps_it2",
    "e12", "eps12", "k2", "k2t", "k5", "k2_mid", "k3_mid", "k5_mid",
    "l", "tf", "wu", "wv", "gamt", "gamu", "gamv", "ce", "cet",
    "lam_e", "lam_eps", "re", "c1aup", "knum", "veg_sw", "row", "row2",
    "kt",
)

#: All ``1:M`` arrays of ``KEpsilonState`` (Fortran bound ``1:M``).
_ARR_M = (
    "s", "gen", "f", "tke_turb_trans", "gen_seiches",
    "c_eps1", "c_eps2", "c_eps3", "z_half",
)


@dataclass
class KEpsilonState:
    """Everything `K_EPSILON` reads or writes outside its dummy arguments.

    The fields mirror

    * the per-lake module arrays of ``ARRAYS_TURB``/``ARRAYS``/
      ``ARRAYS_WATERSTATE``/``ARRAYS_GRID``/``ARRAYS_BATHYM`` that the routine
      touches, and
    * the SAVE locals that survive from one call to the next (the ones listed
      in ``lb_ctx_k_epsilon.inc``: ``firstcall``, ``init_keps``, ``Buoyancy1``,
      ``dBdz0``, ``FTKES``/``FTKEB``, ...).

    Arrays are 1-based (element 0 unused).  ``init_arrays(M)`` allocates every
    missing array with zeros; call it once before the first ``k_epsilon``.

    The routine is called ``nstep_keps`` times per model time step and
    ``E1``/``eps1`` *and* ``E2``/``eps2`` all live in this object, exactly as
    they do in the Fortran module.
    """

    # -- prognostic TKE and dissipation (1:M+1) -----------------------------
    e1: Optional[np.ndarray] = None          # E1, TKE, m**2/s**2
    eps1: Optional[np.ndarray] = None        # eps1, m**2/s**3
    e2: Optional[np.ndarray] = None
    eps2: Optional[np.ndarray] = None

    # -- work arrays of the iteration (1:M+1) -------------------------------
    e_it1: Optional[np.ndarray] = None
    e_it2: Optional[np.ndarray] = None
    eps_it1: Optional[np.ndarray] = None
    eps_it2: Optional[np.ndarray] = None
    e12: Optional[np.ndarray] = None
    eps12: Optional[np.ndarray] = None

    # -- diffusivities and length scales (1:M+1 unless noted) ---------------
    k2: Optional[np.ndarray] = None          # eddy viscosity (before *CE)
    k2t: Optional[np.ndarray] = None         # eddy diffusivity (before *CEt)
    k5: Optional[np.ndarray] = None
    k2_mid: Optional[np.ndarray] = None
    k3_mid: Optional[np.ndarray] = None
    k5_mid: Optional[np.ndarray] = None
    l: Optional[np.ndarray] = None           # turbulence length scale
    tf: Optional[np.ndarray] = None          # 1/(TKE time scale)
    kt: Optional[np.ndarray] = None          # eddy diffusivity for heat, 1:M+1

    # -- TKE equation terms (1:M) -------------------------------------------
    s: Optional[np.ndarray] = None           # buoyancy production
    gen: Optional[np.ndarray] = None         # shear production
    f: Optional[np.ndarray] = None           # Gen + S (output only)
    tke_turb_trans: Optional[np.ndarray] = None
    gen_seiches: Optional[np.ndarray] = None

    # -- non-gradient transport (1:M+1) -------------------------------------
    wu: Optional[np.ndarray] = None
    wv: Optional[np.ndarray] = None
    gamt: Optional[np.ndarray] = None
    gamu: Optional[np.ndarray] = None
    gamv: Optional[np.ndarray] = None

    # -- stability functions (1:M+1 declared, 1:M used) ----------------------
    ce: Optional[np.ndarray] = None
    cet: Optional[np.ndarray] = None
    lam_e: Optional[np.ndarray] = None
    lam_eps: Optional[np.ndarray] = None
    re: Optional[np.ndarray] = None
    c1aup: Optional[np.ndarray] = None

    # -- coefficients of the dissipation equation (1:M) ---------------------
    c_eps1: Optional[np.ndarray] = None
    c_eps2: Optional[np.ndarray] = None
    c_eps3: Optional[np.ndarray] = None

    # -- misc module arrays -------------------------------------------------
    knum: Optional[np.ndarray] = None        # set to 0 every call, never read
    veg_sw: Optional[np.ndarray] = None      # set in the firstcall block only
    row: Optional[np.ndarray] = None         # density at level 1, 1:M+1
    row2: Optional[np.ndarray] = None        # only used in a commented line
    z_half: Optional[np.ndarray] = None      # 1:M, for the zserout budget

    # -- water column state read by the routine -----------------------------
    u1: Optional[np.ndarray] = None          # 1:M+1
    v1: Optional[np.ndarray] = None
    u2: Optional[np.ndarray] = None
    v2: Optional[np.ndarray] = None
    um: Optional[np.ndarray] = None          # overwritten here, 1:M+1
    vm: Optional[np.ndarray] = None
    tw1: Optional[np.ndarray] = None         # 1:M+1
    tw2: Optional[np.ndarray] = None
    sal1: Optional[np.ndarray] = None
    sal2: Optional[np.ndarray] = None
    lamw: Optional[np.ndarray] = None        # 1:M (unused: only via cw_m_row0)
    lamsal: Optional[np.ndarray] = None

    # -- grid / bathymetry (1-based) ---------------------------------------
    ddz: Optional[np.ndarray] = None         # 1:M
    ddz2: Optional[np.ndarray] = None        # 1:M-1
    ddz05: Optional[np.ndarray] = None       # 0:M
    ddz052: Optional[np.ndarray] = None      # 2:M-1
    ddz054: Optional[np.ndarray] = None      # 2:M-1
    dzeta_int: Optional[np.ndarray] = None   # 1:M+1
    dzeta_05int: Optional[np.ndarray] = None  # 1:M
    area_int: Optional[np.ndarray] = None    # 1:M+1
    area_half: Optional[np.ndarray] = None   # 1:M

    # -- scalar layer/bathymetry state -------------------------------------
    h1: float = 0.0                          # water depth, m
    l1: float = 0.0                          # ice thickness, m
    dhw: float = 0.0                         # water depth change, m
    dhw0: float = 0.0
    vol: float = 0.0                         # lake volume, m**3
    botar: float = 0.0                       # bottom area, m**2

    # -- forcing (arguments of the Fortran subroutine) ---------------------
    b0: float = 0.0                          # ice concentration
    tau_air: float = 0.0                     # wind stress, N/m**2
    tau_wav: float = 0.0                     # wave stress, N/m**2
    tau_i: float = 0.0                       # ice-water stress, N/m**2
    tau_gr: float = 0.0                      # bottom stress, N/m**2
    uwind: float = 0.0                       # wind components, m/s
    vwind: float = 0.0
    fetch: float = 0.0                       # m
    h_veg: float = 0.0                       # vegetation height, m (STOP if > 0)
    a_veg: float = 0.0                       # dummy argument, never used
    c_veg: float = 0.0                       # dummy argument, never used
    roughness: float = 0.0                   # dummy argument, never used
    # (`roughness0` from PHYS_CONSTANTS is used everywhere instead)

    # -- turbulence diagnostics (module scalars) ---------------------------
    Buoyancy0: float = 0.0                   # surface buoyancy flux
    H_entrainment: float = 0.0
    Ri_bulk: float = 0.0
    signwaveheight: float = 0.0
    Eseiches: float = 0.0                    # in/out of SEICHE_ENERGY
    Eseiches_diss: float = 0.0
    TKE_balance: float = 0.0
    eps_balance: float = 0.0
    S_integr_positive: float = 0.0
    S_integr_negative: float = 0.0
    Gen_integr: float = 0.0
    eps_integr: float = 0.0
    E_integr: float = 0.0
    TKE_turb_trans_integr: float = 0.0
    Seps_integr_positive: float = 0.0
    Seps_integr_negative: float = 0.0
    Geneps_integr: float = 0.0
    epseps_integr: float = 0.0
    TF_integr: float = 0.0
    tke_budget_terms: Optional[np.ndarray] = None   # 1:5

    # -- SAVE locals that survive between calls ----------------------------
    firstcall: bool = True                   # DATA firstcall /.true./
    init_keps: bool = False                  # per-lake "initialised" flag
    Buoyancy1: float = 0.0                   # used by kepsbc 4 only
    dBdz0: float = 0.0                       # used by kepsbc 4 only
    FTKES: float = 0.0                       # surface TKE flux, 1:M+1 eq.
    FTKEB: float = 0.0                       # bottom TKE flux
    FS: float = 0.0                          # surface dissipation flux
    FB: float = 0.0                          # bottom dissipation flux
    TKES: float = 0.0                        # Dirichlet TKE at the surface
    TKEB: float = 0.0
    DISS_TKES: float = 0.0
    DISS_TKEB: float = 0.0
    iter: int = 0                            # iteration counters (iterat path)
    iter_t: int = 0
    nl: int = 0
    i_entrain: int = 0
    i_entrain_old: int = 0
    E_min: float = 0.0
    eps_min: float = 0.0
    E_min_Racr: float = 0.0
    dE_it: float = 0.0
    deps_it: float = 0.0
    #: SAVE local `AG(1:5)`, used only by the `contrgrad` branch.  The Fortran
    #: never assigns it (all five assignments are commented out), so the values
    #: read there are undefined -> zeros + TODO (see `k_epsilon`).
    AG: Optional[np.ndarray] = None
    #: SAVE local `DTDZH`, used only by the `contrgrad` branch.  Its assignment
    #: is commented out in the Fortran (turb_mod.f90:433) -> undefined.
    DTDZH: float = 0.0

    def init_arrays(self, M):
        """Allocate every missing array (1-based: index 0 stays unused)."""
        for name in _ARR_M1:
            cur = getattr(self, name)
            if cur is None:
                setattr(self, name, np.zeros(M + 2))
            elif len(cur) < M + 2:
                raise ValueError(f"KEpsilonState.{name}: length {len(cur)} < M+2={M + 2}")
        for name in _ARR_M:
            cur = getattr(self, name)
            if cur is None:
                setattr(self, name, np.zeros(M + 1))
            elif len(cur) < M + 1:
                raise ValueError(f"KEpsilonState.{name}: length {len(cur)} < M+1={M + 1}")
        if self.tke_budget_terms is None:
            self.tke_budget_terms = np.zeros(6)   # TKE_budget_terms(1:5)
        if self.AG is None:
            self.AG = np.zeros(6)                 # AG(1:5)
        return self


@dataclass
class KEpsilonParams:
    """Switches read by `K_EPSILON` (DRIVING_PARAMS + the routine's SAVE setup).

    Defaults are the testlake configuration
    (``lakebatch/scratch/out10.work/testlake/setup/lake_setup.dat``).
    """

    M: int = 40                    # number of water layers (DRIVING_PARAMS::M)
    Turbpar: int = 2               # K_EPSILON is only called for Turbpar == 2
    kepsbc: int = 1                # boundary-condition switch for E and eps
    stabfunc: int = 2              # 1 const, 2 Canuto, 3 Galperin
    kwe: float = 100.0             # wave-breaking TKE input coefficient
    zserout: float = MISSING_VALUE  # depth of the TKE-budget output (-999: off)
    missing_value: float = MISSING_VALUE  # DRIVING_PARAMS::missing_value
    eos: int = 5                   # equation of state (SET_DDENS_*)
    lindens: int = 0               # linear density switch
    turb_out: int = 0              # 1: write iteration errors to a file

    # SAVE locals hard-coded in the `firstcall` block of K_EPSILON
    semi_implicit_scheme: bool = SEMI_IMPLICIT_SCHEME  # .true.: no iterations
    smooth: bool = False               # smooth E and eps at the end
    perdam: bool = False               # smooth only the perturbation
    contrgrad: bool = False            # counter-gradient transport
    keps_coef: int = 1                 # 1 standard, 2 Aupoix et al. (1989)
    grav_wave_Gill: int = 0            # internal-wave shear production
    seiches_Goudsmit: int = 0          # seiche energy -> TKE
    maxiter: int = MAXITER             # 35

    def __post_init__(self):
        if self.kepsbc not in (1, 2, 3, 4, 5):
            # The Fortran `select`/`if` chains would leave the tridiagonal
            # coefficients from the previous iteration in place.
            raise ValueError(f"KEpsilonParams: unsupported kepsbc={self.kepsbc}")


# ---------------------------------------------------------------------------
# The k-epsilon solver
# ---------------------------------------------------------------------------
def k_epsilon(state, dt, dt_keps=None, nstep_keps=None, params=None):
    """Advance the k-epsilon closure by one (sub-)time step: `K_EPSILON`.

    Parameters
    ----------
    state : KEpsilonState
        Full model state for one lake; see :class:`KEpsilonState`.
    dt : float
        The **model** time step, s.  The Fortran routine only sees it through
        the water-depth bookkeeping ``dhw_keps = dhw/real(nstep_keps)``.
    dt_keps : float, optional
        The k-epsilon sub-step, s -- i.e. the value that ``lake.f90`` passes as
        the Fortran dummy argument ``dt`` (``dt_keps = dt/nstep_keps``, called
        ``nstep_keps`` times per model step).  Defaults to ``dt``.
    nstep_keps : int, optional
        Number of k-epsilon sub-steps per model step.  Defaults to
        ``round(dt/dt_keps)`` (1 when ``dt_keps`` is not given).
    params : KEpsilonParams, optional
        Switch settings; defaults to the testlake configuration.

    Returns
    -------
    KEpsilonState
        A new state (the argument is not modified) with ``e1``/``eps1``
        advanced to the new time level, ``k2`` (and ``kt``, ``k2t``, ``tf``,
        ``l``, ``s``, ``gen``, ...) filled for the rest of the model.

    Notes
    -----
    * ``dt`` *inside* the equations is ``dt_keps`` -- that is the Fortran
      dummy argument named ``dt``.
    * The first call for a lake performs **two** passes: an initialization
      pass with ``time_deriv = 0`` (``init_keps`` false) followed by a normal
      pass with ``time_deriv = 1``.  Both are reproduced by the outer
      ``keps_mode`` loop.
    * With the default ``semi_implicit_scheme = .true.`` the inner iteration
      (``cycle_keps``) runs exactly once.
    """
    import copy as _copy

    if params is None:
        params = KEpsilonParams()

    if params.Turbpar != 2:
        # lake.f90:1160 calls K_EPSILON only under `if (Turbpar%par == 2)`.
        raise ValueError("K_EPSILON is only called for Turbpar == 2")

    M = params.M
    st = _copy.deepcopy(state)
    st.init_arrays(M)

    dt_keps = float(dt) if dt_keps is None else float(dt_keps)
    if nstep_keps is None:
        nstep_keps = 1 if dt is None else int(round(float(dt) / dt_keps))
    nstep_keps = int(nstep_keps)
    if nstep_keps < 1:
        raise ValueError("nstep_keps must be >= 1")
    if dt_keps <= 0.0:
        raise ValueError("dt_keps must be > 0")

    kepsbc = params.kepsbc
    stabfunc = params.stabfunc
    sig_eps = sigmaeps(kepsbc)
    lam_eps0_ = lam_eps0(kepsbc)
    # AL = g/row0, ALFT = 1.d0 (turb_mod.f90:278-279; Fortran is case-blind,
    # the SAVE locals are declared as `al` and `alft` at line 188)
    AL = C.g / C.row0
    ALFT = 1.0
    al_it = AL_IT

    # ----- firstcall: one-time initialization (turb_mod.f90:254-335) ------
    firstcall_entry = st.firstcall
    if firstcall_entry:
        if params.turb_out == 1:
            # Fortran opens results/debug/err_keps_iter.dat here; file I/O is
            # not ported (TODO).  nunit_ is only used by that output.
            pass
        # `if (h_veg>0) STOP` -- vegetation friction is not operational.
        if st.h_veg > 0.0:
            raise RuntimeError("Vegetation friction parameterization is not "
                              "operational currently: STOP (turb_mod.f90:324)")
        st.veg_sw = np.zeros(M + 2)
        for i in range(1, M + 2):
            # `if (h1-dzeta_int(i)*h1 < h_veg) veg_sw(i) = 1.`
            if st.h1 - st.dzeta_int[i] * st.h1 < st.h_veg:
                st.veg_sw[i] = 1.0
        st.knum[:] = 0.0        # knum = 0.   (never read inside K_EPSILON)
        # The Fortran also sets, in this block and never again:
        #   month_sec/day_sec/hour_sec, pi = 4.*atan(1.e0_ireals), AL = g/row0,
        #   ALFT = 1.d0, semi_implicit_scheme, al_it, ACC, ACC2, ACCk, z0,
        #   eps_min0, E_min_lamw0, dE_it_max, deps_it_max, smooth, perdam,
        #   vdamp, contrgrad, keps_coef.
        # None of them is read outside this routine, so they are module-level
        # constants / KEpsilonParams fields here (see the top of this module).

    # ----- E_min / eps_min (turb_mod.f90:340-351) -------------------------
    if st.l1 > 0.0:
        # The minimal values for E and epsilon are determined from two
        # conditions: (1) the corresponding turbulent diffusion coefficient is
        # much less than the molecular one; (2) free convection occurs in a
        # quiescent state under Ra > Racr.
        # 1.E-5 is a default-real literal
        st.eps_min = (_s(1.0e-5) * (C.lamw0 / C.cw_m_row0) ** 2 * C.Racr
                      * C.niu_wat / (np.max(st.ddz[1:M + 1]) * st.h1) ** 4)
        st.E_min_Racr = (st.eps_min * (np.max(st.ddz[1:M + 1]) * st.h1) ** 2
                         * (CEt0 * C.Racr * C.niu_wat * C.lamw0
                            / C.cw_m_row0) ** (-0.5))
        # E_min = max(E_min_lamw0, E_min_Racr)   ! commented out in the Fortran
        st.E_min = st.E_min_Racr
    else:
        st.eps_min = EPS_MIN0
        st.E_min = E_MIN_LAMW0

    dt05 = 0.5 * dt_keps            # 0.5d0*dt

    # ----- seiches (turb_mod.f90:375-387) --------------------------------
    if params.seiches_Goudsmit == 1:
        # transposed from SEICHE_ENERGY; testlake has seiches_Goudsmit = 0
        st.Eseiches, st.Eseiches_diss = seiche_energy(
            st.area_int[1], st.vol, st.tau_air - st.tau_wav,
            math.sqrt(st.uwind * st.uwind + st.vwind * st.vwind),
            dt_keps, st.Eseiches)
        for i in range(1, M + 1):
            bvf2 = max(AL * (st.row[i + 1] - st.row[i]) / (st.h1 * st.ddz[i]),
                       0.0)
            st.gen_seiches[i] = ((1.0 - 10.0 * math.sqrt(C_Deff)) * bvf2
                                 / (C.row0 * cnorm * st.botar)
                                 * (st.area_int[i] - st.area_int[i + 1])
                                 / st.area_half[i] * st.Eseiches_diss)
    else:
        # --- untested default-off branch (seiches_Goudsmit = 0) ---
        st.gen_seiches[1:M + 1] = 0.0

    # ----- mixed-layer depth from the TKE profile -------------------------
    # i_entrain = minloc(E1(1:M),1)
    st.i_entrain = int(np.argmin(st.e1[1:M + 1])) + 1
    st.H_entrainment = st.dzeta_05int[st.i_entrain] * st.h1

    # Bulk Richardson number for the mixed layer
    st.Ri_bulk = (C.g / C.row0 * (st.row[st.i_entrain] - st.row[1])
                  * (st.dzeta_int[st.i_entrain] * st.h1)
                  / ((st.u1[st.i_entrain] - st.u1[1]) ** 2
                     + (st.v1[st.i_entrain] - st.v1[1]) ** 2
                     + C.small_value))

    # Significant wave height
    if kepsbc == 2 or kepsbc == 3:
        # --- untested default-off branch (kepsbc = 1) ---
        st.signwaveheight = h13(st.tau_air, st.fetch)

    dhw_keps = st.dhw / float(nstep_keps)
    dhw0_keps = st.dhw0 / float(nstep_keps)

    # This time interpolation ensures correspondence between the mean energy
    # viscous dissipation in the Crank-Nicolson scheme for the momentum
    # equation and the TKE shear production when the semi-implicit scheme is
    # used (semi_implicit_scheme = .true.)
    for i in range(1, M + 2):
        st.um[i] = 0.5 * (st.u2[i] + st.u1[i])
        st.vm[i] = 0.5 * (st.v2[i] + st.v1[i])

    # The TKE injection from waves breaking is limited by the wind kinetic
    # energy input to the wave development
    maxTKEinput = (st.tau_wav / C.row0
                   * math.sqrt(st.uwind * st.uwind + st.vwind * st.vwind))

    # DTDZH=(Tw1(2)-Tw1(1))/(h1*ddz(1))   ! commented out in the Fortran, so the
    # SAVE local DTDZH keeps its (undefined) value; only `contrgrad` uses it.
    DTDZH = st.DTDZH

    next_tstep_keps = False
    trace = []
    trace_coef = []

    # Scratch arrays of the Fortran routine (allocated once per call):
    #   work(1:M+1,1:2)             - buoyancy/shear production, then interpolants
    #   a(1:nveclen) ... d(1:nveclen) - tridiagonal coefficients (nveclen >= M+1)
    #   xx(1:2), yy(1:2)            - reused for several purposes, exactly as in
    #                                 the Fortran
    work = np.zeros((M + 2, 3))   # 1-based: work(1:M+1, 1:2)
    a = np.zeros(M + 2)
    b = np.zeros(M + 2)
    c = np.zeros(M + 2)
    d = np.zeros(M + 2)
    xx = np.zeros(3)
    yy = np.zeros(3)

    # =====================================================================
    # keps_mode: repeat with time_deriv = 0 then (after init) time_deriv = 1
    # =====================================================================
    while not next_tstep_keps:

        if not st.init_keps:
            # Initialization mode: time derivatives are neglected
            time_deriv = 0
        else:
            # Evolution mode: full equations are solved
            time_deriv = 1

        for i in range(1, M + 2):
            st.e_it1[i] = st.e1[i]
            st.eps_it1[i] = st.eps1[i]
            st.k2_mid[i] = 0.0
            st.k2[i] = 0.0
            st.k2t[i] = 0.0

        st.iter_t = 0
        cycle_keps = True
        # if (semi_implicit_scheme) then iterat = .false. else iterat = .true.
        iterat = not params.semi_implicit_scheme

        # Density derivatives with respect to temperature and salinity at the
        # layer interfaces, at the *old* time level.
        if params.eos != 5 or params.lindens != 0:
            raise NotImplementedError(
                "SET_DDENS_DTEMP/SET_DDENS_DSAL are ported for eos 5, "
                f"lindens 0 only (got eos={params.eos}, lindens={params.lindens})")
        rhotemp = np.zeros(M + 1)
        rhosal = np.zeros(M + 1)
        for i in range(1, M + 1):
            tm = 0.5 * (st.tw1[i] + st.tw1[i + 1])
            sm = 0.5 * (st.sal1[i] + st.sal1[i + 1])
            rhotemp[i] = ddens_dtemp(tm, sm)
            rhosal[i] = ddens_dsal(tm, sm)

        # =================================================================
        # iterat_keps: iterations on the nonlinear finite-difference system
        # =================================================================
        while cycle_keps:

            for i in range(1, M + 2):
                st.e12[i] = 0.5 * (st.e1[i] + st.e_it1[i])
                st.eps12[i] = 0.5 * (st.eps1[i] + st.eps_it1[i])

            # -------- stability functions (turb_mod.f90:489-516) ----------
            for i in range(1, M + 1):
                if stabfunc == 1:
                    # --- untested default-off branch (stabfunc = 2) ---
                    st.lam_eps[i] = lam_eps0_
                    st.lam_e[i] = lam_E0
                    st.ce[i] = CE0
                    st.cet[i] = CEt0
                else:
                    work[i, 1] = (st.e12[i] * st.e12[i]
                                  / ((st.eps12[i] + ACC2) * (st.eps12[i] + ACC2))
                                  * (rhotemp[i]
                                     * (st.tw1[i + 1] - st.tw1[i])
                                     + C.alsal * rhosal[i]
                                     * (st.sal1[i + 1] - st.sal1[i]))
                                  / (st.h1 * st.ddz[i]) * AL)  # ~ buoyancy production
                    work[i, 2] = (st.e12[i] * st.e12[i]
                                  / ((st.eps12[i] + ACC2) * (st.eps12[i] + ACC2))
                                  * ((st.u1[i + 1] - st.u1[i]) * (st.u1[i + 1] - st.u1[i])
                                     + (st.v1[i + 1] - st.v1[i]) * (st.v1[i + 1] - st.v1[i]))
                                  / (st.h1 * st.h1 * st.ddz[i] * st.ddz[i]))  # ~ shear production
                    if stabfunc == 2:
                        st.ce[i] = ce_canuto(work[i, 2], work[i, 1])
                        st.cet[i] = cet_canuto(work[i, 2], work[i, 1])
                    elif stabfunc == 3:
                        # --- untested default-off branch (stabfunc = 2) ---
                        st.ce[i] = (math.sqrt(2.0) * CL_K_KL_MODEL
                                    * smoment_galperin(work[i, 1]))
                        st.cet[i] = (math.sqrt(2.0) * CL_K_KL_MODEL
                                     * sheat_galperin(work[i, 1]))
                    st.lam_eps[i] = st.ce[i] / sig_eps
                    st.lam_e[i] = st.ce[i] / sigmaE

            # -------- eddy diffusivities between the half levels ----------
            for i in range(2, M + 1):
                work[i, 1] = interpolate_to_integer_point(
                    st.e12[i - 1], st.e12[i], st.ddz[i - 1], st.ddz[i])
                work[i, 2] = interpolate_to_integer_point(
                    st.eps12[i - 1], st.eps12[i], st.ddz[i - 1], st.ddz[i])
                xx[1] = interpolate_to_integer_point(
                    st.lam_eps[i - 1], st.lam_eps[i], st.ddz[i - 1], st.ddz[i])
                xx[2] = interpolate_to_integer_point(
                    st.lam_e[i - 1], st.lam_e[i], st.ddz[i - 1], st.ddz[i])
                if st.iter_t == 0:
                    st.k2_mid[i] = work[i, 1] * work[i, 1] / (work[i, 2] + ACCK)
                else:
                    st.k2_mid[i] = (st.k2_mid[i] * al_it
                                    + work[i, 1] * work[i, 1]
                                    / (work[i, 2] + ACCK) * (1.0 - al_it))
                st.k5_mid[i] = C.niu_wat + max(st.k2_mid[i] * xx[1],
                                               min_visc / sig_eps)
                st.k3_mid[i] = C.niu_wat + max(st.k2_mid[i] * xx[2],
                                               min_visc / sigmaE)

            for i in range(1, M + 1):
                if st.iter_t == 0:
                    st.k2[i] = max(st.e12[i] * st.e12[i] / (st.eps12[i] + ACCK),
                                   min_visc / st.ce[i])
                    st.k2t[i] = max(st.e12[i] * st.e12[i] / (st.eps12[i] + ACCK),
                                    min_diff / st.cet[i])
                else:
                    st.k2[i] = (st.k2[i] * al_it
                                + max(st.e12[i] * st.e12[i] / (st.eps12[i] + ACCK),
                                      min_visc / st.ce[i]) * (1.0 - al_it))
                    st.k2t[i] = (st.k2t[i] * al_it
                                 + max(st.e12[i] * st.e12[i] / (st.eps12[i] + ACCK),
                                       min_diff / st.cet[i]) * (1.0 - al_it))
                st.k5[i] = C.niu_wat + st.k2[i] * st.lam_eps[i]
                st.l[i] = st.e12[i] ** 1.5 / (st.eps12[i] + ACC)
                st.tf[i] = math.sqrt(st.e12[i]) / (st.l[i] + ACC)

            for i in range(2, M):
                st.wu[i] = ((st.e_it1[i + 1] - st.e_it1[i - 1])
                            * (st.u2[i] - st.u2[i - 1])
                            / (st.h1 * st.h1 * st.ddz052[i] * st.ddz[i - 1]))
                st.wv[i] = ((st.e_it1[i + 1] - st.e_it1[i - 1])
                            * (st.v2[i] - st.v2[i - 1])
                            / (st.h1 * st.h1 * st.ddz052[i] * st.ddz[i - 1]))

            # -------- counter-gradient transport (turb_mod.f90:556-589) ---
            if params.contrgrad:
                # --- untested default-off branch (contrgrad = .false.) ---
                # NOTE: AG(1:5) and DTDZH are never assigned in the Fortran
                # (their assignments are commented out at lines 305-309 and
                # 433), so this branch reads undefined SAVE locals.  They are
                # taken from the state (zeros by default) -> TODO: unclear.
                AG = st.AG
                for i in range(2, M + 1):
                    GRADT = (st.tw1[i + 1] - st.tw1[i - 1]) / (st.h1 * st.ddz2[i - 1])
                    GRADU = (st.u2[i] - st.u2[i - 1]) / (st.h1 * st.ddz[i - 1])
                    GRADV = (st.v2[i] - st.v2[i - 1]) / (st.h1 * st.ddz[i - 1])
                    GAMUN = -CONUV * (st.wu[i + 1] - st.wu[i - 1]) / (st.h1 * st.ddz2[i - 1])
                    GAMVN = -CONUV * (st.wv[i + 1] - st.wv[i - 1]) / (st.h1 * st.ddz2[i - 1])
                    COGRTN = DTDZH * _s(0.1)   # `DTDZH*0.1`, 0.1 is default real
                    TFR = st.e_it1[i] / (st.l[i] * st.l[i] + ACC) + ACC
                    COGRUN = GAMUN / TFR
                    COGRVN = GAMVN / TFR
                    # NB: Fortran is case-blind, so `gradT` and `GRADT` are the
                    # same variable.
                    if GRADT < 0.0:
                        st.gamt[i] = -AG[3] * COGRTN
                        st.gamu[i] = -AG[1] * COGRUN
                        st.gamv[i] = -AG[2] * COGRVN
                    else:
                        st.gamt[i] = (-AG[3] * (AL * GRADT ** 2
                                                + CON0 * TFR * COGRTN)
                                      / (AL * GRADT + CON0 * TFR))
                        st.gamu[i] = (-AG[1] * (AL * (CON2 * GRADT + (CON2 - 1.0) * st.gamt[i]) * GRADU
                                                + CON1 * TFR * COGRUN)
                                      / (AL * GRADT + CON1 * TFR))
                        st.gamv[i] = (-AG[2] * (AL * (CON2 * GRADT + (CON2 - 1.0) * st.gamt[i]) * GRADV
                                                + CON1 * TFR * COGRVN)
                                      / (AL * GRADT + CON1 * TFR))
                st.gamt[1] = 0.0
                st.gamu[1] = 0.0
                st.gamv[1] = 0.0
            else:
                for i in range(1, M + 2):
                    st.gamt[i] = 0.0
                    st.gamu[i] = 0.0
                    st.gamv[i] = 0.0

            # -------- shear and buoyancy production (turb_mod.f90:591-606) -
            for i in range(1, M + 1):
                st.gen[i] = (((st.um[i + 1] - st.um[i])
                              * (st.um[i + 1] - st.um[i] + st.h1 * st.ddz[i] * st.gamu[i]))
                             + ((st.vm[i + 1] - st.vm[i])
                                * (st.vm[i + 1] - st.vm[i] + st.h1 * st.ddz[i] * st.gamv[i]))) \
                    / (st.h1 ** 2 * st.ddz[i] ** 2) * st.ce[i]
                # S(i) = - ( 0.5d0*(row(i+1) + row2(i+1) - row(i) - row2(i)) / &
                #          (h1*ddz(i)) + GAMT(i))*ALFT*AL*CEt(i)   ! old form
                st.s[i] = (-(0.5 * (rhotemp[i]
                                    * (st.tw1[i + 1] + st.tw2[i + 1]
                                       - st.tw1[i] - st.tw2[i])
                                    + C.alsal * rhosal[i]
                                    * (st.sal1[i + 1] + st.sal2[i + 1]
                                       - st.sal1[i] - st.sal2[i]))
                            / (st.h1 * st.ddz[i]) + st.gamt[i]) * ALFT * AL * st.cet[i])
                # S(i) = max(S(i),0._ireals)   ! commented out in the Fortran
                st.f[i] = st.gen[i] + st.s[i]
                # Adding shear production of turbulence by gravitational waves
                # in stable stratification (grav_wave_Gill = 0 by default)
                st.gen[i] = (st.gen[i] - params.grav_wave_Gill
                             * grav_wave_Gill_const * step(-st.s[i]) * st.s[i]
                             * st.ce[i] / st.cet[i])

            if st.iter_t == 0:
                st.Buoyancy1 = st.s[1] * st.k2t[1]
                st.dBdz0 = (st.Buoyancy1 - st.Buoyancy0) / (0.5 * st.h1 * st.ddz[1])

            # =============================================================
            # TKE equation
            # =============================================================
            if kepsbc <= 4:
                # Neumann boundary conditions for TKE
                # --- boundary condition at the top ---
                if st.l1 == 0.0:
                    # Open water
                    if kepsbc < 4:
                        st.FTKES = tke_flux_shear(st.tau_air, params.kwe,
                                                  maxTKEinput, kepsbc)
                    elif kepsbc == 4:
                        # --- untested default-off branch (kepsbc = 1) ---
                        st.FTKES = tke_flux_buoy(st.Buoyancy0, C.roughness0,
                                                 st.H_entrainment)
                elif 0.0 < st.l1 <= L0:
                    # --- untested default-off branch (ice) ---
                    # Fractional ice
                    st.FTKES = (st.b0 * tke_flux_shear(st.tau_i, 0.0, maxTKEinput, 1)
                                + (1.0 - st.b0) * tke_flux_shear(
                                    st.tau_air, params.kwe, maxTKEinput, kepsbc))
                elif st.l1 > L0:
                    # --- untested default-off branch (ice) ---
                    # Complete ice cover
                    st.FTKES = tke_flux_shear(st.tau_i, 0.0, maxTKEinput, 1)

                xx[1] = 0.5 * (st.area_int[2] / st.area_half[1] * st.k3_mid[2]
                               * dt_keps / (st.h1 ** 2 * st.ddz[1] * st.ddz05[1])
                               + time_deriv * st.dzeta_05int[1] * dhw_keps
                               / (st.h1 * st.ddz05[1])
                               - time_deriv * dhw0_keps / (st.h1 * st.ddz05[1]))
                yy[1] = dt_keps / (st.h1 * st.ddz[1])
                b[1] = -xx[1]
                c[1] = (-(xx[1] + time_deriv * 1.0)
                        - (abs(st.s[1]) - st.s[1]) / (st.tf[1] + ACC) * dt05
                        - st.tf[1] * dt_keps)
                d[1] = (-time_deriv * st.e1[1] - xx[1] * (st.e1[2] - st.e1[1])
                        - st.area_int[1] / st.area_half[1] * yy[1] * st.FTKES
                        - st.k2t[1] * (st.s[1] + abs(st.s[1])) * dt05
                        - dt_keps * (st.k2[1] * st.gen[1] + st.gen_seiches[1]))

                # --- boundary condition at the bottom ---
                st.FTKEB = -tke_flux_shear(st.tau_gr, 0.0, maxTKEinput, 1)
                xx[2] = 0.5 * (st.area_int[M] / st.area_half[M] * st.k3_mid[M]
                               * dt_keps / (st.h1 ** 2 * st.ddz[M] * st.ddz05[M - 1])
                               - time_deriv * st.dzeta_05int[M] * dhw_keps
                               / (st.h1 * st.ddz05[M - 1])
                               + time_deriv * dhw0_keps / (st.h1 * st.ddz05[M - 1]))
                yy[2] = dt_keps / (st.h1 * st.ddz[M])
                a[M] = -xx[2]
                c[M] = (-(xx[2] + time_deriv * 1.0)
                        - (abs(st.s[M]) - st.s[M]) / (st.tf[M] + ACC) * dt05
                        - st.tf[M] * dt_keps)
                d[M] = (-time_deriv * st.e1[M] - xx[2] * (st.e1[M - 1] - st.e1[M])
                        + st.area_int[M + 1] / st.area_half[M] * yy[2] * st.FTKEB
                        - st.k2t[M] * (st.s[M] + abs(st.s[M])) * dt05
                        - dt_keps * (st.k2[M] * st.gen[M] + st.gen_seiches[M]))
            elif kepsbc == 5:
                # --- untested default-off branch (kepsbc = 1) ---
                # Dirichlet boundary condition for TKE
                # Boundary condition at the top
                if st.l1 == 0.0:
                    # Open water
                    st.TKES = tke_wall_shear(st.tau_air)
                elif 0.0 < st.l1 <= L0:
                    # Fractional ice
                    st.TKES = (st.b0 * tke_wall_shear(st.tau_i)
                               + (1.0 - st.b0) * tke_wall_shear(st.tau_air))
                elif st.l1 > L0:
                    # Complete ice cover
                    st.TKES = tke_wall_shear(st.tau_i)
                b[1] = 0.0
                c[1] = 1.0
                d[1] = st.TKES
                # Boundary condition at the bottom
                st.TKEB = tke_wall_shear(st.tau_gr)
                a[M] = 0.0
                c[M] = 1.0
                d[M] = st.TKEB

            # Coefficients at the internal points of the mesh
            for i in range(2, M):
                a[i] = (time_deriv * st.dzeta_05int[i] * dhw_keps / (st.ddz054[i] * st.h1)
                        - time_deriv * dhw0_keps / (st.ddz054[i] * st.h1)
                        - st.area_int[i] / st.area_half[i] * st.k3_mid[i]
                        * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i - 1]))
                b[i] = (-time_deriv * st.dzeta_05int[i] * dhw_keps / (st.ddz054[i] * st.h1)
                        + time_deriv * dhw0_keps / (st.ddz054[i] * st.h1)
                        - st.area_int[i + 1] / st.area_half[i] * st.k3_mid[i + 1]
                        * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i]))
                c[i] = (a[i] + b[i] - time_deriv * 1.0
                        - (abs(st.s[i]) - st.s[i]) / (st.tf[i] + ACC) * dt05
                        - st.tf[i] * dt_keps)
                d[i] = (-time_deriv * st.e1[i]
                        - st.k2t[i] * (st.s[i] + abs(st.s[i])) * dt05
                        - dt_keps * (st.k2[i] * st.gen[i] + st.gen_seiches[i]))
                d[i] = d[i] - dt_keps * (st.h1 ** 2 * st.ddz[i] * st.ddz2[i]) ** (-1) \
                    * (st.area_int[i + 1] / st.area_half[i] * st.k3_mid[i + 1]
                       * (st.e1[i + 1] - st.e1[i]))
                d[i] = d[i] + dt_keps * (st.h1 ** 2 * st.ddz[i] * st.ddz2[i - 1]) ** (-1) \
                    * (st.area_int[i] / st.area_half[i] * st.k3_mid[i]
                       * (st.e1[i] - st.e1[i - 1]))
                d[i] = d[i] - time_deriv * st.dzeta_05int[i] * dhw_keps \
                    * (st.ddz054[i] * st.h1) ** (-1) * (st.e1[i + 1] - st.e1[i - 1])
                d[i] = d[i] + time_deriv * dhw0_keps \
                    * (st.ddz054[i] * st.h1) ** (-1) * (st.e1[i + 1] - st.e1[i - 1])

            ind_bound = False
            indstab = ind_stab_fact_db(a, b, c, 1, M, ind_bound)
            if not indstab:
                for i in range(2, M):
                    a[i] = -st.k3_mid[i] * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i - 1])
                    b[i] = -st.k3_mid[i + 1] * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i])

            trace_coef.append((np.array(a, copy=True), np.array(b, copy=True),
                               np.array(c, copy=True), np.array(d, copy=True)))
            y = progonka(a[1:M + 1], b[1:M + 1], c[1:M + 1], d[1:M + 1],
                         k=0, n=M - 1)
            st.e_it2[1:M + 1] = y
            st.e_it2[M + 1] = 0.0

            check_min_value(st.e_it2, M, st.E_min)

            st.dE_it = float(np.max(np.abs(st.e_it1[1:M + 2] - st.e_it2[1:M + 2])))

            # C1 is given following Satyanarayana et al., 1999; Aupoix et al., 1989
            if params.keps_coef == 1:
                for i in range(1, M + 1):
                    yy[1] = 0.5 * (1.0 + _fsign1(st.s[i]))
                    zz = 0.5 * (1.0 - _fsign1(st.s[i]))
                    ceps3 = ceps3_stable * zz + ceps3_unstable * yy[1]
                    # ceps3 = 1.14d0 ; ceps3 < -0.4 should be used for stable
                    # stratification (Burchard, 2002); other values: 0<ceps3<0.29
                    # stable / 1.44 unstable (Rodi, 1987), ceps3 >= 1
                    # (Kochergin and Sklyar, 1992)
                    st.c_eps1[i] = ceps1
                    st.c_eps2[i] = ceps2
                    st.c_eps3[i] = ceps3
            elif params.keps_coef == 2:
                # --- untested default-off branch (keps_coef = 1) ---
                for i in range(1, M + 1):
                    st.re[i] = ACC + (2.0 * st.e_it1[i] / 3.0) ** 2 \
                        / (st.eps_it1[i] * C.niu_wat + ACC)
                    st.c1aup[i] = Co / (1.0 + 0.69 * (2.0 - Co) / math.sqrt(st.re[i]))
                    st.c_eps1[i] = st.c1aup[i]
                    st.c_eps2[i] = st.c1aup[i]
                    st.c_eps3[i] = st.c1aup[i]

            # =============================================================
            # dissipation-rate equation
            # =============================================================
            # dist_surf is the distance from the surface where the b.c. for
            # epsilon are formulated (dist_surf = 0 is commented out)
            if kepsbc <= 4:
                # Neumann boundary conditions for TKE dissipation
                # --- boundary condition at the top ---
                if st.l1 == 0.0:
                    if kepsbc < 4:
                        # Extrapolation of the turbulent diffusivity to the
                        # boundary level, assuming a logarithmic profile
                        ksurf1 = max(C.niu_wat, st.ce[1] * st.k2[1]
                                     - kar * math.sqrt(st.tau_air / C.row0)
                                     * 0.5 * st.ddz[1] * st.h1)
                        Esurf1 = (st.e_it1[1] + 0.5 * st.FTKES * sigmaE
                                  * st.h1 * st.ddz[1] / ksurf1)
                        st.FS = dissipation_flux_shear(
                            st.tau_air, ksurf1, Esurf1, Z0, params.kwe,
                            st.fetch, maxTKEinput, kepsbc, st.signwaveheight,
                            sigmaeps_=sig_eps)
                    elif kepsbc == 4:
                        # --- untested default-off branch (kepsbc = 1) ---
                        Esurf1 = st.e_it1[1]
                        st.FS = dissipation_flux_buoy(
                            st.Buoyancy0, C.roughness0, st.H_entrainment,
                            st.dBdz0, Esurf1, sigmaeps_=sig_eps)
                elif 0.0 < st.l1 <= L0:
                    # --- untested default-off branch (ice) ---
                    ksurf1 = max(C.niu_wat, st.ce[1] * st.k2[1]
                                 - kar * math.sqrt(st.tau_air / C.row0)
                                 * 0.5 * st.ddz[1] * st.h1)
                    ksurf2 = max(C.niu_wat, st.ce[1] * st.k2[1]
                                 - kar * math.sqrt(st.tau_i / C.row0)
                                 * 0.5 * st.ddz[1] * st.h1)
                    Esurf1 = (st.e_it1[1] + 0.5 * st.FTKES * sigmaE
                              * st.h1 * st.ddz[1] / ksurf1)
                    Esurf2 = (st.e_it1[1] + 0.5 * st.FTKES * sigmaE
                              * st.h1 * st.ddz[1] / ksurf2)
                    st.FS = (st.b0 * dissipation_flux_shear(
                                 st.tau_i, ksurf2, Esurf2, Z0, 0.0, st.fetch,
                                 maxTKEinput, 1, st.signwaveheight,
                                 sigmaeps_=sig_eps)
                             + (1.0 - st.b0) * dissipation_flux_shear(
                                 st.tau_air, ksurf1, Esurf1, Z0, params.kwe,
                                 st.fetch, maxTKEinput, kepsbc,
                                 st.signwaveheight, sigmaeps_=sig_eps))
                elif st.l1 > L0:
                    # --- untested default-off branch (ice) ---
                    ksurf2 = max(C.niu_wat, st.ce[1] * st.k2[1]
                                 - kar * math.sqrt(st.tau_i / C.row0)
                                 * 0.5 * st.ddz[1] * st.h1)
                    Esurf2 = (st.e_it1[1] + 0.5 * st.FTKES * sigmaE
                              * st.h1 * st.ddz[1] / ksurf2)
                    st.FS = dissipation_flux_shear(
                        st.tau_i, ksurf2, Esurf2, Z0, 0.0, st.fetch,
                        maxTKEinput, 1, st.signwaveheight, sigmaeps_=sig_eps)

                xx[1] = 0.5 * (st.area_int[2] / st.area_half[1] * st.k5_mid[2]
                               * dt_keps / (st.h1 ** 2 * st.ddz[1] * st.ddz05[1])
                               + time_deriv * st.dzeta_05int[1] * dhw_keps
                               / (st.h1 * st.ddz05[1])
                               - time_deriv * dhw0_keps / (st.h1 * st.ddz05[1]))
                yy[1] = dt_keps / (st.h1 * st.ddz[1])
                b[1] = -xx[1]
                c[1] = (-(xx[1] + time_deriv * 1.0)
                        - st.c_eps2[1] * st.tf[1] * dt_keps
                        + st.c_eps3[1] * (-abs(st.s[1]) + st.s[1])
                        / (st.tf[1] + ACC) * dt05)
                d[1] = (-time_deriv * st.eps1[1] - xx[1] * (st.eps1[2] - st.eps1[1])
                        - st.area_int[1] / st.area_half[1] * yy[1] * st.FS
                        - st.c_eps3[1] * (abs(st.s[1]) + st.s[1])
                        * st.tf[1] * st.k2t[1] * dt05
                        - st.c_eps1[1] * st.tf[1] * dt_keps
                        * (st.k2[1] * st.gen[1] + st.gen_seiches[1]))

                # --- boundary condition at the bottom ---
                # Extrapolation of the turbulent diffusivity to the boundary
                # level, assuming a logarithmic profile
                ksurf2b = max(C.niu_wat, st.ce[M] * st.k2[M]
                              - kar * math.sqrt(st.tau_gr / C.row0)
                              * 0.5 * st.ddz[M] * st.h1)
                Esurf2b = (st.e_it1[M] - 0.5 * st.FTKEB * sigmaE
                           * st.h1 * st.ddz[M] / ksurf2b)
                st.FB = -dissipation_flux_shear(
                    st.tau_gr, ksurf2b, Esurf2b, Z0, 0.0, st.fetch,
                    maxTKEinput, 1, st.signwaveheight, sigmaeps_=sig_eps)
                xx[2] = 0.5 * (st.area_int[M] / st.area_half[M] * st.k5_mid[M]
                               * dt_keps / (st.h1 ** 2 * st.ddz[M] * st.ddz05[M - 1])
                               - time_deriv * st.dzeta_05int[M] * dhw_keps
                               / (st.h1 * st.ddz05[M - 1])
                               + time_deriv * dhw0_keps / (st.h1 * st.ddz05[M - 1]))
                yy[2] = dt_keps / (st.h1 * st.ddz[M])
                a[M] = -xx[2]
                c[M] = (-(xx[2] + time_deriv * 1.0)
                        - st.c_eps2[M] * st.tf[M] * dt_keps
                        + st.c_eps3[M] * (-abs(st.s[M]) + st.s[M])
                        / (st.tf[M] + ACC) * dt05)
                d[M] = (-time_deriv * st.eps1[M] - xx[2] * (st.eps1[M - 1] - st.eps1[M])
                        + st.area_int[M + 1] / st.area_half[M] * yy[2] * st.FB
                        - st.c_eps3[M] * (abs(st.s[M]) + st.s[M])
                        * st.tf[M] * st.k2t[M] * dt05
                        - st.c_eps1[M] * st.tf[M] * dt_keps
                        * (st.k2[M] * st.gen[M] + st.gen_seiches[M]))
            elif kepsbc == 5:
                # --- untested default-off branch (kepsbc = 1) ---
                # Dirichlet boundary conditions for TKE dissipation
                dist_surf = 0.5 * st.ddz[1] * st.h1
                if st.l1 == 0.0:
                    # Open water
                    st.DISS_TKES = dissipation_wall_shear(
                        st.tau_air, st.Buoyancy0, dist_surf)
                elif 0.0 < st.l1 <= L0:
                    # Fractional ice
                    st.DISS_TKES = (st.b0 * dissipation_wall_shear(
                        st.tau_i, st.Buoyancy0, dist_surf)
                        + (1.0 - st.b0) * dissipation_wall_shear(
                            st.tau_air, st.Buoyancy0, dist_surf))
                elif st.l1 > L0:
                    # Complete ice cover -- the Fortran uses tau_air here (a bug,
                    # kept verbatim: turb_mod.f90:865)
                    st.DISS_TKES = dissipation_wall_shear(
                        st.tau_air, st.Buoyancy0, dist_surf)
                b[1] = 0.0
                c[1] = 1.0
                d[1] = st.DISS_TKES
                # Boundary condition at the bottom
                dist_surf = 0.5 * st.ddz[M] * st.h1
                st.DISS_TKEB = dissipation_wall_shear(st.tau_gr, 0.0, dist_surf)
                a[M] = 0.0
                c[M] = 1.0
                d[M] = st.DISS_TKEB

            for i in range(2, M):
                a[i] = (time_deriv * st.dzeta_05int[i] * dhw_keps / (st.ddz054[i] * st.h1)
                        - time_deriv * dhw0_keps / (st.ddz054[i] * st.h1)
                        - st.area_int[i] / st.area_half[i] * st.k5_mid[i]
                        * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i - 1]))
                b[i] = (-time_deriv * st.dzeta_05int[i] * dhw_keps / (st.ddz054[i] * st.h1)
                        + time_deriv * dhw0_keps / (st.ddz054[i] * st.h1)
                        - st.area_int[i + 1] / st.area_half[i] * st.k5_mid[i + 1]
                        * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i]))
                c[i] = (a[i] + b[i] - time_deriv * 1.0
                        - st.c_eps2[i] * st.tf[i] * dt_keps
                        + st.c_eps3[i] * (-abs(st.s[i]) + st.s[i])
                        / (st.tf[i] + ACC) * dt05)
                d[i] = (-time_deriv * st.eps1[i]
                        - st.c_eps3[i] * (abs(st.s[i]) + st.s[i])
                        * st.tf[i] * st.k2t[i] * dt05
                        - st.c_eps1[i] * st.tf[i] * dt_keps
                        * (st.k2[i] * st.gen[i] + st.gen_seiches[i]))
                d[i] = d[i] - dt_keps * (st.h1 ** 2 * st.ddz[i] * st.ddz2[i]) ** (-1) \
                    * (st.area_int[i + 1] / st.area_half[i] * st.k5_mid[i + 1]
                       * (st.eps1[i + 1] - st.eps1[i]))
                d[i] = d[i] + dt_keps * (st.h1 ** 2 * st.ddz[i] * st.ddz2[i - 1]) ** (-1) \
                    * (st.area_int[i] / st.area_half[i] * st.k5_mid[i]
                       * (st.eps1[i] - st.eps1[i - 1]))
                d[i] = d[i] - time_deriv * st.dzeta_05int[i] * dhw_keps \
                    * (st.ddz054[i] * st.h1) ** (-1) * (st.eps1[i + 1] - st.eps1[i - 1])
                d[i] = d[i] + time_deriv * dhw0_keps \
                    * (st.ddz054[i] * st.h1) ** (-1) * (st.eps1[i + 1] - st.eps1[i - 1])

            ind_bound = False
            indstab = ind_stab_fact_db(a, b, c, 1, M, ind_bound)
            if not indstab:
                for i in range(2, M):
                    a[i] = -st.k5_mid[i] * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i - 1])
                    b[i] = -st.k5_mid[i + 1] * dt_keps / (st.h1 ** 2 * st.ddz[i] * st.ddz2[i])

            y = progonka(a[1:M + 1], b[1:M + 1], c[1:M + 1], d[1:M + 1],
                         k=0, n=M - 1)
            st.eps_it2[1:M + 1] = y
            st.eps_it2[M + 1] = 0.0

            check_min_value(st.eps_it2, M, st.eps_min)

            st.deps_it = float(np.max(np.abs(st.eps_it1[1:M + 2]
                                             - st.eps_it2[1:M + 2])))

            # -------- convergence / update (turb_mod.f90:917-956) --------
            if iterat:
                # --- untested default-off branch (semi_implicit_scheme =
                # .true. => iterat = .false.) ---
                if ((st.dE_it > D_E_IT_MAX or st.deps_it > DEPS_IT_MAX)
                        and st.iter_t < params.maxiter):
                    for i in range(1, M + 2):
                        st.e_it1[i] = st.e_it1[i] * al_it + st.e_it2[i] * (1.0 - al_it)
                        st.eps_it1[i] = st.eps_it1[i] * al_it + st.eps_it2[i] * (1.0 - al_it)
                    st.iter += 1
                    st.iter_t += 1
                    if st.iter == 8:
                        st.iter = 0
                else:
                    if st.dE_it < _s(1.0e-6) and st.deps_it < _s(1.0e-7):
                        for i in range(1, M + 2):
                            st.e2[i] = st.e_it2[i]
                            st.eps2[i] = st.eps_it2[i]
                        st.iter = 0
                        st.iter_t = 0
                        st.nl = 2
                        cycle_keps = False
                    else:
                        for i in range(1, M + 2):
                            st.eps_it1[i] = st.eps1[i]
                            st.e_it1[i] = st.e1[i]
                        iterat = False
                        st.iter_t = 0
                        st.nl = 1
            else:
                for i in range(1, M + 2):
                    st.e2[i] = st.e_it2[i]
                    st.eps2[i] = st.eps_it2[i]
                # E2   = E_it1   * al_it + E_it2   * (1-al_it)     ! commented out
                # eps2 = eps_it1 * al_it + eps_it2 * (1-al_it)
                st.iter = 0
                st.iter_t = 0
                cycle_keps = False
            # if (dE_it<1.E-10.or.deps_it<1.E-13) nl=3   ! commented out

        # ------------------ end of iterat_keps loop ----------------------
        trace.append(np.array(st.e_it2, copy=True))

        if time_deriv == 0:
            # K-epsilon solver is in initialization mode: obtain initial
            # profiles of E1 and eps1 (time derivatives neglected).
            for i in range(1, M + 2):
                st.e1[i] = st.e2[i]
                st.eps1[i] = st.eps2[i]
            st.init_keps = True
        else:
            # Evolution mode: E2 and eps2 are at the next time step.
            next_tstep_keps = True

    # ---------------------- end of keps_mode loop ------------------------

    # Smoothing of the k and epsilon profiles
    if params.smooth:
        # --- untested default-off branch (smooth = .false.) ---
        nmax = M + 1                       # ubound(E2,1)
        vsmop_lake(st.e2, st.e2, 1, nmax, 1, M, VDAMP, params.perdam)
        check_min_value(st.e2, M, st.E_min)
        vsmop_lake(st.eps2, st.eps2, 1, nmax, 1, M, VDAMP, params.perdam)
        check_min_value(st.eps2, M, st.eps_min)

    for i in range(1, M + 2):
        st.e12[i] = 0.5 * (st.e1[i] + st.e2[i])
        st.eps12[i] = 0.5 * (st.eps1[i] + st.eps2[i])

    if stabfunc != 1:
        for i in range(1, M + 1):
            work[i, 1] = (st.e2[i] * st.e2[i]
                          / ((st.eps2[i] + ACC2) * (st.eps2[i] + ACC2))
                          * (rhotemp[i] * (st.tw2[i + 1] - st.tw2[i])
                             + C.alsal * rhosal[i]
                             * (st.sal2[i + 1] - st.sal2[i]))
                          / (st.h1 * st.ddz[i]) * AL)     # ~ buoyancy production
            work[i, 2] = (st.e2[i] * st.e2[i]
                          / ((st.eps2[i] + ACC2) * (st.eps2[i] + ACC2))
                          * ((st.u2[i + 1] - st.u2[i]) * (st.u2[i + 1] - st.u2[i])
                             + (st.v2[i + 1] - st.v2[i]) * (st.v2[i + 1] - st.v2[i]))
                          / (st.h1 * st.h1 * st.ddz[i] * st.ddz[i]))  # ~ shear production
            if stabfunc == 2:
                st.cet[i] = cet_canuto(work[i, 2], work[i, 1])
            elif stabfunc == 3:
                # --- untested default-off branch (stabfunc = 2) ---
                st.cet[i] = math.sqrt(2.0) * CL_K_KL_MODEL * sheat_galperin(work[i, 1])

    for i in range(1, M + 1):
        st.kt[i] = max(st.cet[i] * st.e2[i] ** 2 / (st.eps2[i] + ACCK),
                       min_diff) * C.cw_m_row0

    # ---------------- diagnostic calculation (turb_mod.f90:1017-1036) ----
    # Turbulent transport of TKE
    for i in range(2, M):
        st.tke_turb_trans[i] = (
            1.0 / (st.area_half[i] * st.h1 * st.h1 * st.ddz[i])
            * (st.area_int[i + 1] * st.k3_mid[i + 1]
               * 0.5 * (st.e2[i + 1] + st.e1[i + 1] - st.e2[i] - st.e1[i])
               / st.ddz05[i]
               - st.area_int[i] * st.k3_mid[i]
               * 0.5 * (st.e2[i] + st.e1[i] - st.e2[i - 1] - st.e1[i - 1])
               / st.ddz05[i - 1]))

    st.tke_turb_trans[1] = (
        1.0 / (st.area_half[1] * st.h1 * st.h1 * st.ddz[1])
        * (st.area_int[2] * st.k3_mid[2]
           * 0.5 * (st.e2[2] + st.e1[2] - st.e2[1] - st.e1[1]) / st.ddz05[1]
           + st.area_int[1] * st.FTKES * st.h1))
    st.tke_turb_trans[M] = (
        1.0 / (st.area_half[M] * st.h1 * st.h1 * st.ddz[M])
        * (-st.area_int[M + 1] * st.FTKEB * st.h1
           - st.area_int[M] * st.k3_mid[M]
           * 0.5 * (st.e2[M] + st.e1[M] - st.e2[M - 1] - st.e1[M - 1])
           / st.ddz05[M - 1]))

    for i in range(1, M + 1):
        st.s[i] = (0.5 * (st.s[i] + abs(st.s[i])) * st.k2t[i]
                   + 0.5 * (-abs(st.s[i]) + st.s[i]) / (st.tf[i] + ACC) * st.e2[i])
        st.gen[i] = st.gen[i] * st.k2[i]

    # Components of the TKE budget for output
    if params.zserout != params.missing_value:
        # --- untested default-off branch (zserout = missing_value) ---
        i = M
        while True:
            if st.z_half[i] - params.zserout < 0.0:
                break
            i -= 1
            if i < 1:
                # z_half(1:M), so `i-1` would read z_half(0): out of bounds
                raise IndexError(
                    "zserout is above the shallowest level (z_half(1)): the "
                    "Fortran loop runs out of the array")
        if i >= M:
            # z_half(M) < zserout: the Fortran then reads z_half(M+1)
            raise IndexError(
                "zserout is below the deepest half level (z_half(M)): the "
                "Fortran interpolates using z_half(M+1), which is out of bounds")
        zz = (params.zserout - st.z_half[i]) / (st.z_half[i + 1] - st.z_half[i])
        st.tke_budget_terms[1] = (zz * (st.e2[i + 1] - st.e1[i + 1]) / dt_keps
                                  + (1.0 - zz) * (st.e2[i] - st.e1[i]) / dt_keps)
        st.tke_budget_terms[2] = (zz * st.tke_turb_trans[i + 1]
                                  + (1.0 - zz) * st.tke_turb_trans[i])
        st.tke_budget_terms[3] = zz * st.gen[i + 1] + (1.0 - zz) * st.gen[i]
        st.tke_budget_terms[4] = zz * st.s[i + 1] + (1.0 - zz) * st.s[i]
        st.tke_budget_terms[5] = (zz * st.e2[i + 1] * st.tf[i + 1]
                                  + (1.0 - zz) * st.e2[i] * st.tf[i])

    # ------------------- integral budgets (turb_mod.f90:1055-1111) ------
    st.S_integr_positive = 0.0
    st.S_integr_negative = 0.0
    st.Gen_integr = 0.0
    st.TKE_turb_trans_integr = 0.0
    st.Seps_integr_positive = 0.0
    st.Seps_integr_negative = 0.0
    st.Geneps_integr = 0.0
    st.epseps_integr = 0.0
    st.eps_integr = 0.0
    st.E_integr = 0.0
    st.TF_integr = 0.0

    for i in range(1, st.i_entrain + 1):
        xx[1] = st.ddz[i] * st.h1
        yy[1] = 0.5 * (1.0 + _fsign1(st.s[i]))
        zz = 0.5 * (1.0 - _fsign1(st.s[i]))

        st.S_integr_positive = st.S_integr_positive + yy[1] * st.s[i] * xx[1]
        st.Seps_integr_positive = (st.Seps_integr_positive
                                   + yy[1] * st.s[i] * st.c_eps3[i]
                                   * st.eps2[i] / st.e2[i] * xx[1])

        st.S_integr_negative = st.S_integr_negative + zz * st.s[i] * xx[1]
        st.Seps_integr_negative = (st.Seps_integr_negative
                                   + zz * st.s[i] * st.c_eps3[i]
                                   * st.eps2[i] / st.e2[i] * xx[1])

        st.Gen_integr = st.Gen_integr + st.gen[i] * xx[1]
        st.TKE_turb_trans_integr = (st.TKE_turb_trans_integr
                                    + st.tke_turb_trans[i] * xx[1])
        st.Geneps_integr = (st.Geneps_integr + st.gen[i] * st.c_eps1[i]
                            * st.eps2[i] / st.e2[i] * xx[1])
        st.epseps_integr = (st.epseps_integr + st.eps2[i] * st.c_eps2[i]
                            * st.eps2[i] / st.e2[i] * xx[1])
        st.eps_integr = st.eps_integr + st.tf[i] * st.e2[i] * xx[1]
        st.E_integr = st.E_integr + st.e2[i] * xx[1]
        st.TF_integr = st.TF_integr + st.tf[i] * xx[1]

    st.TKE_balance = (st.Gen_integr + st.S_integr_positive
                      + st.S_integr_negative - st.eps_integr)
    st.eps_balance = (st.Geneps_integr + st.Seps_integr_positive
                      + st.Seps_integr_negative - st.epseps_integr)

    # Averaging
    st.S_integr_positive = st.S_integr_positive / st.H_entrainment
    st.S_integr_negative = st.S_integr_negative / st.H_entrainment
    st.Gen_integr = st.Gen_integr / st.H_entrainment
    st.TKE_turb_trans_integr = st.TKE_turb_trans_integr / st.H_entrainment
    st.Seps_integr_positive = st.Seps_integr_positive / st.H_entrainment
    st.Seps_integr_negative = st.Seps_integr_negative / st.H_entrainment
    st.Geneps_integr = st.Geneps_integr / st.H_entrainment
    st.epseps_integr = st.epseps_integr / st.H_entrainment
    st.eps_integr = st.eps_integr / st.H_entrainment
    st.E_integr = st.E_integr / st.H_entrainment
    st.TF_integr = st.TF_integr / st.H_entrainment
    st.TKE_balance = st.TKE_balance / st.H_entrainment
    st.eps_balance = st.eps_balance / st.H_entrainment

    # Debug output (turb_mod.f90:1113-1123)
    if firstcall_entry:
        st.i_entrain_old = st.i_entrain
    else:
        # if (i_entrain /= i_entrain_old) then ... i_entrain_old = i_entrain
        pass

    # Output of the iteration errors (turb_mod.f90:1132-1137): the Fortran
    # writes to results/debug/err_keps_iter.dat, file I/O is not ported (TODO).
    if params.turb_out == 1 and not params.semi_implicit_scheme:
        if st.dE_it > 1.0e-7 or st.deps_it > 1.0e-9:
            pass  # write (nunit_, *) nstep, nl, dE_it, deps_it

    # ------------------------------ epilogue ----------------------------
    for i in range(1, M + 2):
        st.e1[i] = st.e2[i]
        st.eps1[i] = st.eps2[i]

    # Recalculation for output
    for i in range(1, M + 1):
        st.k2[i] = st.k2[i] * st.ce[i]

    st.trace = trace
    st.trace_coef = trace_coef
    st.firstcall = False            # turb_mod.f90:1156

    return st
