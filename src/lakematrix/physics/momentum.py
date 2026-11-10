"""Lake momentum equations (Fortran ``MOMENTUM_SOLVER``).

Ported from ``lakebatch/core/source/model/momentum_mod.f90``:

* ``MOMENTUM_SOLVER`` lines 15-1611 (main body 15-1420, internal helpers
  1421-1608),
* ``DHDXDHDY``      lines 1614-1664,
* ``DHDXDHDY2``     lines 1667-1741,
* ``DHDXDHDY3``     lines 1744-1811,
* ``DHDXDHDYML``    lines 1814-1874.

The routines called by the solver that live in other Fortran modules are
re-implemented here only when they are small and/or needed by the target
configuration: ``RICHARDSON_NUMBER``, ``DOMWAVESPEED``, ``CE_CANUTO``,
``SMOMENT_GALPERIN``, ``KRON`` and LAPACK ``SGESV``.  ``MATRIXPROGONKA`` is
**not** re-implemented: the block solve uses
:func:`lakematrix.physics.blocktridiag.matrixprogonka`.  The density
derivatives ``SET_DDENS_DTEMP``/``SET_DDENS_DSAL`` are taken from
:mod:`lakematrix.physics.density` (eos 5 only, as there).

Target configuration
--------------------
``dyn_pgrad=0``, ``botfric=1``, ``horvisc=0``, ``cuette=0``, ``Turbpar=2``,
``seiches`` defaults (``impl_seiches=.true.``, ``lam_force=0.4``),
``lakinterac=1``.  Branches that the target configuration cannot reach are
transcribed and marked ``UNREACHABLE``.  Two paths are deliberately left as
``TODO`` and raise ``NotImplementedError``: ``dyn_pgrad=4`` (needs
``DENSLAYERS``/``RELAYER`` and the Crank-Nicolson seiche scheme) and
``Turbpar=5`` (needs ``EDDY_DIFF_MOST_RICH``); neither is reachable here.

Indexing
--------
Public arrays are **0-based**, matching :class:`lakematrix.grid.Grid`:

===============  ==========================  ===============================
public array     shape                       Fortran
===============  ==========================  ===============================
``ddz``          ``(M,)``                    ``ddz(1:M)``
``ddz05``        ``(M+1,)``                  ``ddz05(0:M)``  (index 0 kept)
``dzeta_int``    ``(M+1,)``                  ``dzeta_int(1:M+1)``
``dzeta_05int``  ``(M,)``                    ``dzeta_05int(1:M)``
``area_int``     ``(M+1,)``                  ``bathymwater(i)%area_int``
``area_half``    ``(M,)``                    ``bathymwater(i)%area_half``
``Lx``, ``Ly``   ``(M+1,)``                  ``Lx(1:M+1)``, ``Ly(1:M+1)``
``Lx_half`` ...  ``(M,)``                    ``bathymwater(i)%Lx_half``
===============  ==========================  ===============================

Internally every 1-based array is padded with an unused element at index 0,
and loop counters keep their Fortran values (``for k in range(2, M + 1)``),
so the source can be compared line by line.  The one asymmetry is
``ddz05``: internally ``_ddz05[i]`` is Fortran ``ddz05(i-1)`` (this is what
the padding of a 0-based array gives).  Everywhere else ``_x[i]`` is Fortran
``x(i)``; therefore a source expression ``ddz05(i-1)`` is written
``_ddz05[i]``, exactly like the surrounding ``x(i)`` references.

Precision
---------
Fortran default-real literals (``0.01``, ``7.7``, ``1.2``, ``1./6.``, ...)
are rounded to ``real(4)`` before being used in ``real(8)`` expressions, so
they appear as ``_s(0.01)`` = ``float(np.float32(0.01))``.  Exact literals
(``0.5``, ``2.``, ``10.``, ``180.``) are written as doubles.  ``d``-exponent
literals (``4.d-1``, ``1.d-20``, ``0.46d0``) are doubles.  The ``SQCOEFS`` /
``CUCOEFS`` systems are assembled in ``real(8)`` but *stored* in ``real(4)``
after every assignment and solved by LAPACK ``SGESV`` in single precision;
that is reproduced with ``np.float32`` stores and a float32 solve.

State that must persist between calls
-------------------------------------
``k2_old`` (allocated zero on the first call), ``firstcall``, ``veg_sw``,
``lsu_water``/``lsv_water``, the seiche ``hx*`` fields, ``itherm``,
``u_star``, ``velfrict_*`` and the ATMOS wind-stress components
``taux``/``tauy`` (the latter are set only in the open-water and thin-ice
branches, exactly like the Fortran module variables).
``momentum_solver`` does not mutate the input state; it returns a new
:class:`MomentumState`.

Notes, deviations and ambiguities
---------------------------------
* ``b0`` is ``None`` when ``l1 == 0`` and ``tau_i`` is ``None`` when
  ``l1 <= 0``: the Fortran declares them ``intent(out)`` and leaves them
  undefined in those cases.  ``velfrict_ice`` keeps its previous value when
  there is no ice (module variable in the Fortran).
* The lateral-friction block writes ``lsv%water(i)`` where the Fortran loop
  variable is left at ``M+1`` (line 405), then zeroes index ``M+1``.  Thus
  ``lsv_water(1)`` is never assigned.  This bug is preserved: entry 1 keeps
  its previous state value (zero in a fresh state).
* The Fortran allocates ``Force_x``/``Force_y`` locally without
  initialisation.  They are explicitly zero-filled here; this only matters
  for the (unreachable) ``cuette=1`` path, which reads them in the interior
  rows without setting them.
* ``Turbpar=1`` evaluates ``dzeta_05int(M+1)`` although that array is
  dimensioned ``1:M``.  A zero sentinel is appended so the transcription
  runs; the Fortran value is undefined.  The branch is unreachable for the
  target configuration.
* ``dyn_pgrad=4`` (DENSLAYERS/RELAYER, Crank-Nicolson seiches) and
  ``Turbpar=5`` (EDDY_DIFF_MOST_RICH) are marked ``TODO`` and raise
  ``NotImplementedError`` instead of inventing physics.
* ``SET_DDENS_DTEMP``/``SET_DDENS_DSAL`` support ``eos=5`` (and
  ``lindens=1``); other equations of state raise ``NotImplementedError``
  with a pointer to :mod:`lakematrix.physics.density`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from ..constants import C
from .blocktridiag import matrixprogonka
from .density import ddens_dsal, ddens_dtemp, set_dens_consts

__all__ = [
    "MomentumParams",
    "MomentumState",
    "MomentumResult",
    "momentum_solver",
    "aversqfunc",
    "avercufunc",
    "sqcoefs",
    "cucoefs",
    "dhdxdhdy",
    "dhdxdhdy2",
    "dhdxdhdy3",
    "dhdxdhdy_ml",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _s(x: float) -> float:
    """Default-real Fortran literal stored in a ``real(8)`` variable."""
    return float(np.float32(x))


_PI = C.pi                     # NUMERIC_PARAMS pi (real(8))
_G = C.g                       # PHYS_CONSTANTS g = 9.814 (default real)
_ROW0 = C.row0
_ROA0 = C.roa0
_LAMW0 = C.lamw0               # 0.561 (default real)
_CW_M_ROW0 = C.cw_m_row0
_KAPPA = C.kappa               # PHYS_CONSTANTS kappa = 0.4 (default real)
_NIU_WAT = C.niu_wat           # 1.307d-6
_Z0_BOT = C.z0_bot             # 1.3d-4
_ROUGHNESS0 = C.roughness0     # 1.d-4
_CLIN_BOT = C.clin_bot         # 4.e-3 (default real)
_CQUAD_BOT = C.cquad_bot       # 8.e-2 (default real)
_SMALL = C.small_value         # 1.d-20
_MINWIND = C.minwind           # 1.e-2 (default real)

# TURB_CONST / NUMERIC parameters used by the turbulence closures.
# All of these are declared with a ``d`` exponent in lake_modules.f90 except
# kappa/kar (see the note on ``_KAR``).
_CE0 = 0.09
_CE_COEF01 = 0.127
_CE_COEF02 = 0.1526e-1
_CE_COEF03 = 0.16e-3
_CE_COEF11 = 1.0
_CE_COEF12 = 0.2
_CE_COEF13 = 0.315e-1
_CE_COEF14 = 0.58e-2
_CE_COEF15 = 0.4e-2
_CE_COEF16 = 0.4e-4
_A1 = 0.92
_B1 = 1.66e1
_C1 = 0.8e-1
_A2 = 0.74
_B2 = 1.01e1
_CL_K_KL_MODEL = 2.0 ** 1.5 / _B1
_KAR = 0.4                     # TURB_CONST kar = 0.4d0 -> double 0.4
_CS = 1.5                      # 1.5d0
_CS1 = 100.0                   # 100.d0
_L0 = 0.1                      # 0.1d0  (thin-ice threshold, m)
_MIN_VISC = 1.0e-10 * 1.0      # min_diff * Pr_stable
_IMPL_SEICHES = True           # logical, parameter :: impl_seiches = .true.

_THIRD = _s(1.0 / 3.0)         # real(4) 1./3. used in SQCOEFS/CUCOEFS
_TWO_THIRDS = _s(2.0 / 3.0)    # real(4) 2./3. in DOMWAVESPEED
_SIXTH = _s(1.0 / 6.0)         # real(4) 1./6. in the Chezy coefficients
_ICE_UNDEF = 0.0               # see tau_i/b0 note in momentum_solver


# ---------------------------------------------------------------------------
# Parameter / state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MomentumParams:
    """Static data of one lake column (Fortran arguments + DRIVING_PARAMS).

    Shapes follow the 0-based convention of :class:`lakematrix.grid.Grid`
    (see the module docstring).
    """

    # --- grid and bathymetry (required, 0-based as documented above)
    M: int
    ddz: np.ndarray                 # (M,)
    ddz05: np.ndarray               # (M+1,)  ddz05[0] is Fortran ddz05(0)
    dzeta_int: np.ndarray           # (M+1,)
    dzeta_05int: np.ndarray         # (M,)
    area_int: np.ndarray            # (M+1,)
    area_half: np.ndarray           # (M,)
    Lx: np.ndarray                  # (M+1,)
    Ly: np.ndarray                  # (M+1,)
    Lx_half: np.ndarray             # (M,)
    Ly_half: np.ndarray             # (M,)
    depth_area: np.ndarray          # (ndatamax, 2); Fortran depth_area(1,2)
    kor: float                      # Coriolis parameter, 1/s

    # --- atmospheric forcing (ATMOS module inputs)
    uwind: float
    vwind: float
    cdmw: float                     # momentum exchange coefficient
    zref: float
    fetch: float

    # --- switches (DRIVING_PARAMS derived types, only %par is used)
    # Defaults are the target configuration of the batch rewrite.
    dyn_pgrad: int = 0
    botfric: int = 1
    horvisc: float = 0.0
    cuette: int = 0
    Turbpar: int = 2
    stabfunc: int = 2
    relwind: int = 1
    momflxpart: int = 0
    momflux0: float = 0.0
    PBLpar: int = 3
    pgrad: float = 0.0
    nManning: float = 3.5e-2
    eos: int = 5
    lindens: int = 0
    # SET_DENS_CONSTS(eos) result, only used when lindens == 1.
    ddens_dtemp0: float | None = None
    ddens_dsal0: float | None = None

    # --- tributaries (only the dyn_pgrad=4/5 paths read them)
    tribheat: int = 0
    N_tribin: int = 0
    N_tribout: int = 0
    iefflloc: int = 0
    itribloc: np.ndarray | None = None
    disch_tribin: np.ndarray | None = None    # (N_tribin, M+1)
    disch_tribout: np.ndarray | None = None   # (?, M+1)

    # --- driver bookkeeping (not used by MOMENTUM_SOLVER; kept for parity)
    lakinterac: int = 1
    year: int = 0
    month: int = 0
    day: int = 0
    hour: float = 0.0

    # --- vegetation friction (h_veg > 0 is a STOP in LAKE 3.2)
    h_veg: float = 0.0
    a_veg: float = 0.0
    c_veg: float = 0.0

    # --- DRIVING_PARAMS missing_value = -999. (default real -> exact)
    missing_value: float = _s(-999.0)


@dataclass
class MomentumState:
    """Prognostic and module-level state touched by ``MOMENTUM_SOLVER``.

    All arrays are 0-based.  ``k2_old`` is the Fortran ``k2_old(1:M)`` and
    must be carried across calls; start it at zeros (Fortran allocates it
    zero-initialised only once).  ``firstcall`` reproduces the ``SAVE``d
    ``firstcall`` flag.
    """

    # --- required water-column fields (M+1 or M)
    u1: np.ndarray                  # (M+1,) old x-velocity
    v1: np.ndarray                  # (M+1,) old y-velocity
    E1: np.ndarray                  # (M+1,) TKE
    eps1: np.ndarray                # (M+1,) TKE dissipation
    k2: np.ndarray                  # (M+1,) eddy viscosity
    k2_old: np.ndarray              # (M,)   previous-step eddy viscosity
    row: np.ndarray                 # (M+1,) density
    Ri: np.ndarray                  # (M,)   Richardson number
    knum: np.ndarray                # (M+1,) wave-breaking TKE input
    veg_sw: np.ndarray              # (M+1,) vegetation switch
    u_star: float                   # friction velocity
    Tw1: np.ndarray                 # (M+1,) water temperature
    Sal1: np.ndarray                # (M+1,) salinity
    lsu_water: np.ndarray           # (M+1,) lateral friction u, lsu%water
    lsv_water: np.ndarray           # (M+1,) lateral friction v, lsv%water
    uv: np.ndarray                  # (M+1,) |u|, recomputed each call
    h1: float                       # lake depth, m
    l1: float                       # ice thickness, m (0 = open water)
    ls1: float                      # snow thickness on ice, m
    dhw: float                      # water-depth change over the step, m
    dhw0: float                     # previous-step water-depth change, m
    alphax: float                   # x surface slope, degrees (in/out)
    alphay: float                   # y surface slope, degrees (in/out)
    velfrict_bot: float             # bottom friction velocity
    velfrict_ice: float             # ice friction velocity
    # ATMOS tauxw/tauyw: set only in the open-water/ice branches, so the
    # Couette branch keeps the previous value (Fortran module variables).
    taux: float = 0.0
    tauy: float = 0.0

    # --- seiche/pressure-gradient state (dyn_pgrad >= 1); persisted
    hx1: float = 0.0
    hx2: float = 0.0
    hy1: float = 0.0
    hy2: float = 0.0
    hx1t: float = 0.0
    hx2t: float = 0.0
    hy1t: float = 0.0
    hy2t: float = 0.0
    hx1ml: np.ndarray | None = None   # (M+1,)
    hx2ml: np.ndarray | None = None
    hy1ml: np.ndarray | None = None
    hy2ml: np.ndarray | None = None
    gradpx_tend: np.ndarray | None = None   # (M+1,)
    gradpy_tend: np.ndarray | None = None   # (M+1,)

    # --- multi-layer seiche machinery (dyn_pgrad = 3/4)
    itherm: np.ndarray | None = None    # (M+2,)  Fortran 1-based indices
    i_maxN: int = 0
    nstep: int = 0
    aki: np.ndarray | None = None       # (nlevs, nlevs)
    bki: np.ndarray | None = None
    rowc: np.ndarray | None = None       # (M+1,)
    LxLyCDL: np.ndarray | None = None    # (M+1, 2)

    firstcall: bool = True


@dataclass
class MomentumResult:
    """Outputs of :func:`momentum_solver` (Fortran intent(out)/module data)."""

    # --- intent(out) arguments of MOMENTUM_SOLVER
    u2: np.ndarray                  # (M+1,) new x-velocity
    v2: np.ndarray                  # (M+1,) new y-velocity
    b0: float | None                # undef. unless 0 < l1 <= L0
    tau_air: float
    tau_gr: float
    tau_i: float | None             # undef. unless l1 > 0 (see notes)
    tau_wav: float

    # --- updated state (carry k2_old/firstcall/hx*/lsu/lsv to the next call)
    state: MomentumState

    # --- module-level diagnostics of ATMOS/ARRAYS_TURB
    taux: float
    tauy: float
    tau_sbl: float
    u_star: float
    velfrict_bot: float
    velfrict_ice: float
    wind10: float
    windwork: float
    discharge: np.ndarray           # (2,)
    u: float
    v: float
    urel: float
    vrel: float
    urels: float
    vrels: float
    wr: float
    C10: float
    ufr: float
    coef1: float
    coef2: float
    Cz_gr: float
    Cz_i: float
    rp: float
    xbot: float
    horvisc_on: int
    Force_x: np.ndarray             # (M+1,)
    Force_y: np.ndarray             # (M+1,)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _pad1(x, n: int, name: str) -> np.ndarray:
    """Copy a 0-based ``(n,)`` array into a 1-based ``(n+1,)`` array."""
    a = np.asarray(x, dtype=float)
    if a.shape != (n,):
        raise ValueError(f"{name} has shape {a.shape}, expected ({n},)")
    out = np.zeros(n + 1)
    out[1:] = a
    return out


def _strip1(a: np.ndarray) -> np.ndarray:
    """Drop the unused index 0 from an internal 1-based array."""
    return np.asarray(a, dtype=float)[1:].copy()


def _copy_state(state: MomentumState) -> MomentumState:
    """Deep copy the arrays of ``state`` so the input stays untouched."""
    kw = {}
    for f in fields(state):
        v = getattr(state, f.name)
        kw[f.name] = v.copy() if isinstance(v, np.ndarray) else v
    return MomentumState(**kw)


def _normalise_state(st: MomentumState, M: int) -> None:
    """Fill optional persistent arrays with the values LAKE starts from."""
    if st.itherm is None:
        st.itherm = np.full(M + 2, -1, dtype=int)
    if st.hx1ml is None:
        st.hx1ml = np.zeros(M + 1)
    if st.hx2ml is None:
        st.hx2ml = np.zeros(M + 1)
    if st.hy1ml is None:
        st.hy1ml = np.zeros(M + 1)
    if st.hy2ml is None:
        st.hy2ml = np.zeros(M + 1)
    if st.gradpx_tend is None:
        st.gradpx_tend = np.zeros(M + 1)
    if st.gradpy_tend is None:
        st.gradpy_tend = np.zeros(M + 1)
    if st.rowc is None:
        st.rowc = np.zeros(M + 1)
    if st.LxLyCDL is None:
        st.LxLyCDL = np.full((M + 1, 2), _s(-999.0))


def _fsign(a: float, b: float) -> float:
    """Fortran ``SIGN(a,b)`` (b == 0 counts as positive)."""
    return abs(a) if b >= 0.0 else -abs(a)


def _kron(i: int, j: int) -> float:
    """Fortran ``KRON`` (numerics_mod.f90:175)."""
    return 1.0 if i == j else 0.0


def _sgesv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """LAPACK ``SGESV`` (single precision) for the small SQCOEFS systems."""
    try:
        x = np.linalg.solve(np.asarray(a, dtype=np.float32),
                            np.asarray(b, dtype=np.float32))
    except np.linalg.LinAlgError as exc:  # Fortran returns ierr != 0
        raise RuntimeError("SGESV: singular matrix in SQCOEFS/CUCOEFS") from exc
    return np.asarray(x, dtype=np.float32)


def _botfric_xx(par: int, u: float, v: float) -> float:
    """Lateral drag coefficient switch (momentum_mod.f90:369-376)."""
    if par == 0:
        return 0.0                      # UNREACHABLE (botfric=1)
    if par == 1:
        return _CLIN_BOT                # target (botfric=1)
    if par == 2:
        return _CQUAD_BOT * math.sqrt(u * u + v * v)  # UNREACHABLE
    raise ValueError(f"botfric={par}: Fortran SELECT CASE has no default")


def _set_ddens_dtemp(params: MomentumParams, temp: float, sal: float) -> float:
    """``SET_DDENS_DTEMP`` (phys_func.f90:89-121)."""
    if params.lindens != 0:
        if params.ddens_dtemp0 is not None:
            return params.ddens_dtemp0
        if params.eos != 5:
            raise NotImplementedError(
                "TODO: DDENS_DTEMP0 for eos != 5 is not ported")
        return set_dens_consts()[0]
    if params.eos == 5:                      # eos_McCatcheon
        return float(ddens_dtemp(temp, sal))
    # eos 1/3/4 (Hostetler/Kivu/UNESCO) live in phys_func.f90 and are not
    # ported; eos 2 is a STOP in the Fortran.
    raise NotImplementedError(
        f"TODO: SET_DDENS_DTEMP for eos={params.eos} (only eos=5 is ported; "
        "see physics/density.py)")


def _set_ddens_dsal(params: MomentumParams, temp: float, sal: float) -> float:
    """``SET_DDENS_DSAL`` (phys_func.f90:124-156)."""
    if params.lindens != 0:
        if params.ddens_dsal0 is not None:
            return params.ddens_dsal0
        if params.eos != 5:
            raise NotImplementedError(
                "TODO: DDENS_DSAL0 for eos != 5 is not ported")
        return set_dens_consts()[1]
    if params.eos == 5:
        return float(ddens_dsal(temp, sal))
    raise NotImplementedError(
        f"TODO: SET_DDENS_DSAL for eos={params.eos} (only eos=5 is ported; "
        "see physics/density.py)")


def _richardson_number(M, u, v, ddz, h, row):
    """``RICHARDSON_NUMBER`` (phys_func.f90:1157-1175), 1-based internal."""
    Ri = np.zeros(M + 1)
    for i in range(1, M + 1):
        dudz2dvdz2 = max(((u[i + 1] - u[i]) ** 2 + (v[i + 1] - v[i]) ** 2)
                         / (ddz[i] * h), _SMALL)
        Ri[i] = _G / _ROW0 * (row[i + 1] - row[i]) / dudz2dvdz2
    return Ri


def _dom_wave_speed(ufr: float, fetch: float) -> float:
    """``DOMWAVESPEED`` (phys_func.f90:1955-1982).

    ``C1 = 0.051`` and ``C2 = 0.96`` are default-real parameters, hence
    float32-rounded; ``(C1/C2)**(2./3.)`` uses the real(4) exponent.
    """
    c1 = _s(0.051)
    c2 = _s(0.96)
    c3 = (c1 / c2) ** _TWO_THIRDS
    return c3 * (ufr * _G * fetch) ** _SIXTH


def _ce_canuto(lambda_m: float, lambda_n: float) -> float:
    """``CE_CANUTO`` (turb_mod.f90:1181-1224), Canuto et al. (2001)."""
    lambda_n1 = max(-5.9, lambda_n)
    ce = ((_CE_COEF01 + _CE_COEF02 * lambda_n1 - _CE_COEF03 * lambda_m)
          / (_CE_COEF11 + _CE_COEF12 * lambda_n1 + _CE_COEF13 * lambda_m
             + _CE_COEF14 * lambda_n1 * lambda_n1
             + _CE_COEF15 * lambda_n1 * lambda_m
             - _CE_COEF16 * lambda_m * lambda_m))
    return min(0.46, max(1.0e-20, ce))


def _smoment_galperin(lambda_n: float) -> float:
    """``SMOMENT_GALPERIN`` (turb_mod.f90:1272-1317)."""
    gh_max = (_A2 * (12.0 * _A1 + _B1 + 3.0 * _B2)) ** (-1)
    gh_min = -0.28
    gh = -lambda_n
    gh = _CL_K_KL_MODEL ** 2 * 0.5 * gh
    gh = min(gh, gh_max)
    gh = max(gh, gh_min)
    num = (1.0 - 3.0 * _C1 - 6.0 * _A1 / _B1
           - 3.0 * _A2 * gh * ((_B2 - 3.0 * _A2) * (1.0 - 6.0 * _A1 / _B1)
                               - 3.0 * _C1 * (_B2 + 6.0 * _A1)))
    den = ((1.0 - 3.0 * _A2 * gh * (6.0 * _A1 + _B2))
           * (1.0 - 9.0 * _A1 * _A2 * gh))
    val = num / den * _A1
    return max(1.0e-2, val)


# ---------------------------------------------------------------------------
# Internal helper procedures of MOMENTUM_SOLVER (momentum_mod.f90:1420-1608)
# ---------------------------------------------------------------------------

def aversqfunc(aa: float, bb: float, cc: float, x1: float, x2: float) -> float:
    """``AVERSQFUNC`` (momentum_mod.f90:1422-1427)."""
    return (aa / 3.0 * (x2 ** 3 - x1 ** 3)
            + 0.5 * bb * (x2 ** 2 - x1 ** 2)
            + cc * (x2 - x1)) / (x2 - x1)


def avercufunc(aa: float, bb: float, cc: float, dd: float,
               x1: float, x2: float) -> float:
    """``AVERCUFUNC`` (momentum_mod.f90:1430-1436). Unused in LAKE 3.2."""
    return (aa / 4.0 * (x2 ** 4 - x1 ** 4)
            + bb / 3.0 * (x2 ** 3 - x1 ** 3)
            + 0.5 * cc * (x2 ** 2 - x1 ** 2)
            + dd * (x2 - x1)) / (x2 - x1)


def sqcoefs(nl, m_, m1_, m2_, x1, x2, itherm, area_int, ddz05, h1):
    """``SQCOEFS`` (momentum_mod.f90:1439-1505).

    Parameters ``itherm``, ``area_int`` and ``ddz05`` are 1-based internal
    arrays; ``ddz05[i]`` is Fortran ``ddz05(i-1)``.  Returns ``(aa, bb, cc)``.
    """
    am = np.zeros((3, 3), dtype=np.float32)
    rhs = np.zeros(3, dtype=np.float32)
    xxx = 0.0
    vol1 = 0.0
    for ii in range(itherm[nl] + 1, itherm[nl + 1] + 1):
        dx = ddz05[ii] * h1
        am[0, 0] = np.float32(float(am[0, 0]) + _THIRD * area_int[ii]
                              * ((xxx + dx) ** 3 - xxx ** 3))
        am[0, 1] = np.float32(float(am[0, 1]) + 0.5 * area_int[ii]
                              * ((xxx + dx) ** 2 - xxx ** 2))
        am[0, 2] = np.float32(float(am[0, 2]) + area_int[ii] * dx)
        xxx = xxx + dx
        vol1 = vol1 + area_int[ii] * dx
    am[1, 0] = np.float32(x1 ** 2)
    am[1, 1] = np.float32(x1)
    am[1, 2] = np.float32(1.0)
    am[2, 0] = np.float32(x2 ** 2)
    am[2, 1] = np.float32(x2)
    am[2, 2] = np.float32(1.0)
    rhs[0] = np.float32(m_ * vol1)
    rhs[1] = np.float32(m1_)
    rhs[2] = np.float32(m2_)
    sol = _sgesv(am, rhs)
    return float(sol[0]), float(sol[1]), float(sol[2])


def cucoefs(nsp, nl, m_, m1_, m2_, x1, x2, u1, v1,
            itherm, area_int, ddz05, h1):
    """``CUCOEFS`` (momentum_mod.f90:1509-1608).

    ``itherm``, ``area_int``, ``ddz05``, ``u1``, ``v1`` are internal 1-based
    arrays.  Unused in LAKE 3.2 (the calls are commented out).
    """
    if nsp == 1:
        workk = u1
    elif nsp == 2:
        workk = v1
    else:
        raise ValueError(f"CUCOEFS: nsp={nsp} (Fortran has no default)")
    am = np.zeros((4, 4), dtype=np.float32)
    rhs = np.zeros(4, dtype=np.float32)
    x11 = 0.0
    vol1 = 0.0
    for ii in range(itherm[nl] + 1, itherm[nl + 1] + 1):
        dx = ddz05[ii] * h1
        am[0, 0] = np.float32(float(am[0, 0]) + 0.25 * area_int[ii]
                              * ((x11 + dx) ** 4 - x11 ** 4))
        am[0, 1] = np.float32(float(am[0, 1]) + _THIRD * area_int[ii]
                              * ((x11 + dx) ** 3 - x11 ** 3))
        am[0, 2] = np.float32(float(am[0, 2]) + 0.5 * area_int[ii]
                              * ((x11 + dx) ** 2 - x11 ** 2))
        am[0, 3] = np.float32(float(am[0, 3]) + area_int[ii] * dx)
        x11 = x11 + dx
        vol1 = vol1 + area_int[ii] * dx
    x11 = 0.0
    vol2 = 0.0
    for ii in range(itherm[nl] + 1, itherm[nl + 1] + 1):
        dx = ddz05[ii] * h1
        am[1, 0] = np.float32(float(am[1, 0]) + 0.25 * area_int[ii]
                              * ((x11 + dx) ** 4 - x11 ** 4) * workk[ii])
        am[1, 1] = np.float32(float(am[1, 1]) + _THIRD * area_int[ii]
                              * ((x11 + dx) ** 3 - x11 ** 3) * workk[ii])
        am[1, 2] = np.float32(float(am[1, 2]) + 0.5 * area_int[ii]
                              * ((x11 + dx) ** 2 - x11 ** 2) * workk[ii])
        am[1, 3] = np.float32(float(am[1, 3]) + area_int[ii] * dx * workk[ii])
        x11 = x11 + dx
        vol2 = vol2 + area_int[ii] * dx * workk[ii]
    am[2, 0] = np.float32(x1 ** 3)
    am[2, 1] = np.float32(x1 ** 2)
    am[2, 2] = np.float32(x1)
    am[2, 3] = np.float32(1.0)
    am[3, 0] = np.float32(x2 ** 3)
    am[3, 1] = np.float32(x2 ** 2)
    am[3, 2] = np.float32(x2)
    am[3, 3] = np.float32(1.0)
    rhs[0] = np.float32(m_ * vol1)
    rhs[1] = np.float32(m_ * vol2)
    rhs[2] = np.float32(m1_)
    rhs[3] = np.float32(m2_)
    sol = _sgesv(am, rhs)
    return (float(sol[0]), float(sol[1]), float(sol[2]), float(sol[3]))


# ---------------------------------------------------------------------------
# Pressure-gradient helpers (momentum_mod.f90:1614-1874)
# ---------------------------------------------------------------------------

def dhdxdhdy(M, u, v, area_int, Lx, Ly, ddz05, h1, dt,
             hx1, hx2, hy1, hy2):
    """``DHDXDHDY`` (momentum_mod.f90:1614-1664).

    ``u``, ``v``, ``area_int``, ``Lx``, ``Ly`` are 1-based; ``ddz05[i]`` is
    Fortran ``ddz05(i-1)``.  Returns the updated ``(hx1, hx2, hy1, hy2)``.
    """
    rint = 0.0
    for i in range(1, M + 2):
        rint = rint + h1 * ddz05[i] * Lx[i] * v[i]
    rint = _PI * dt * rint / area_int[1]
    hy2 = hy2 + rint
    hy1 = hy1 - rint

    rint = 0.0
    for i in range(1, M + 2):
        rint = rint + h1 * ddz05[i] * Ly[i] * u[i]
    rint = _PI * dt * rint / area_int[1]
    hx2 = hx2 + rint
    hx1 = hx1 - rint
    return hx1, hx2, hy1, hy2


def dhdxdhdy2(itherm, M, u, v, area_int, area_half, Lx, Ly, ddz05, h1, dt,
              hx1, hx2, hy1, hy2, hx1t, hx2t, hy1t, hy2t):
    """``DHDXDHDY2`` (momentum_mod.f90:1667-1741).

    ``itherm`` is the scalar numerical level (Fortran dummy argument).
    Returns ``(hx1, hx2, hy1, hy2, hx1t, hx2t, hy1t, hy2t)``.
    """
    # Surface layer deviations
    rint = 0.0
    for i in range(1, itherm):
        rint = rint + h1 * ddz05[i] * Lx[i] * v[i]
    rint = _PI * dt * rint / area_int[1]
    hy2 = hy2 + rint
    hy1 = hy1 - rint

    rint = 0.0
    for i in range(1, itherm):
        rint = rint + h1 * ddz05[i] * Ly[i] * u[i]
    rint = _PI * dt * rint / area_int[1]
    hx2 = hx2 + rint
    hx1 = hx1 - rint

    # Thermocline surface deviations
    rint = 0.0
    for i in range(itherm, M + 2):
        rint = rint + h1 * ddz05[i] * Lx[i] * v[i]
    rint = _PI * dt * rint / area_half[itherm - 1]
    hy2t = hy2t + rint
    hy1t = hy1t - rint

    rint = 0.0
    for i in range(itherm, M + 2):
        rint = rint + h1 * ddz05[i] * Ly[i] * u[i]
    rint = _PI * dt * rint / area_half[itherm - 1]
    hx2t = hx2t + rint
    hx1t = hx1t - rint
    return hx1, hx2, hy1, hy2, hx1t, hx2t, hy1t, hy2t


def dhdxdhdy3(itherm, M, u, v, area_int, area_half, Lx, Ly, ddz05, h1, dt,
              hx1ml, hx2ml, hy1ml, hy2ml):
    """``DHDXDHDY3`` (momentum_mod.f90:1744-1811).

    Only used by the explicit seiche scheme, which is dead code because
    ``impl_seiches`` is a ``parameter`` equal to ``.true.`` in LAKE 3.2.
    Returns the four updated arrays.
    """
    for i in range(1, M + 2):
        if itherm[i + 1] > 0:
            areatop = area_int[1] if i == 1 else area_half[itherm[i]]
            rint = 0.0
            for j in range(itherm[i] + 1, itherm[i + 1] + 1):
                rint = rint + h1 * ddz05[j] * Lx[j] * v[j]
            rint = _PI * dt * rint / areatop
            hy2ml[i] = hy2ml[i] + rint
            hy1ml[i] = hy1ml[i] - rint

            rint = 0.0
            for j in range(itherm[i] + 1, itherm[i + 1] + 1):
                rint = rint + h1 * ddz05[j] * Ly[j] * u[j]
            rint = _PI * dt * rint / areatop
            hx2ml[i] = hx2ml[i] + rint
            hx1ml[i] = hx1ml[i] - rint
        else:
            break
    return hx1ml, hx2ml, hy1ml, hy2ml


def dhdxdhdy_ml(M, u, v, area_int, area_half, Lx, Ly, ddz05, h1, dt,
                hx1ml, hx2ml, hy1ml, hy2ml):
    """``DHDXDHDYML`` (momentum_mod.f90:1814-1874). Returns updated arrays."""
    rint = h1 * ddz05[1] * Ly[1] * u[1]
    rint = _PI * dt * rint / area_int[1]
    hx2ml[1] = hx2ml[1] + rint
    hx1ml[1] = hx1ml[1] - rint
    for i in range(2, M + 2):
        rint = h1 * ddz05[i] * Ly[i] * u[i]
        rint = _PI * dt * rint / area_half[i - 1]
        hx2ml[i] = hx2ml[i] + rint
        hx1ml[i] = hx1ml[i] - rint

    rint = h1 * ddz05[1] * Lx[1] * v[1]
    rint = _PI * dt * rint / area_int[1]
    hy2ml[1] = hy2ml[1] + rint
    hy1ml[1] = hy1ml[1] - rint
    for i in range(2, M + 2):
        rint = h1 * ddz05[i] * Lx[i] * v[i]
        rint = _PI * dt * rint / area_half[i - 1]
        hy2ml[i] = hy2ml[i] + rint
        hy1ml[i] = hy1ml[i] - rint
    return hx1ml, hx2ml, hy1ml, hy2ml


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def momentum_solver(state: MomentumState, params: MomentumParams,
                    dt: float) -> MomentumResult:
    """One time step of the LAKE 3.2 momentum equations.

    Fortran: ``MOMENTUM_SOLVER`` (momentum_mod.f90:15-1611).  Arrays are
    0-based at the boundary, 1-based internally (see module docstring).
    """
    M = int(params.M)
    if M < 1:
        raise ValueError("M must be at least 1")
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    st = _copy_state(state)
    _normalise_state(st, M)

    # ---- 1-based copies of the inputs ----------------------------------
    u1 = _pad1(st.u1, M + 1, "u1")
    v1 = _pad1(st.v1, M + 1, "v1")
    E1 = _pad1(st.E1, M + 1, "E1")
    eps1 = _pad1(st.eps1, M + 1, "eps1")
    k2 = _pad1(st.k2, M + 1, "k2")
    k2_old = _pad1(st.k2_old, M, "k2_old")
    row = _pad1(st.row, M + 1, "row")
    Ri = _pad1(st.Ri, M, "Ri")
    knum = _pad1(st.knum, M + 1, "knum")
    veg_sw = _pad1(st.veg_sw, M + 1, "veg_sw")
    Tw1 = _pad1(st.Tw1, M + 1, "Tw1")
    Sal1 = _pad1(st.Sal1, M + 1, "Sal1")
    lsu = _pad1(st.lsu_water, M + 1, "lsu_water")
    lsv = _pad1(st.lsv_water, M + 1, "lsv_water")
    uv = _pad1(st.uv, M + 1, "uv")
    hx1ml = _pad1(st.hx1ml, M + 1, "hx1ml")
    hx2ml = _pad1(st.hx2ml, M + 1, "hx2ml")
    hy1ml = _pad1(st.hy1ml, M + 1, "hy1ml")
    hy2ml = _pad1(st.hy2ml, M + 1, "hy2ml")
    gradpx_tend = _pad1(st.gradpx_tend, M + 1, "gradpx_tend")
    gradpy_tend = _pad1(st.gradpy_tend, M + 1, "gradpy_tend")
    rowc = _pad1(st.rowc, M + 1, "rowc")
    itherm = np.zeros(M + 3, dtype=int)
    itherm_in = np.asarray(st.itherm, dtype=int)
    if itherm_in.shape != (M + 2,):
        raise ValueError(f"itherm has shape {itherm_in.shape}, expected ({M + 2},)")
    itherm[1:] = itherm_in

    ddz = _pad1(params.ddz, M, "ddz")
    ddz05 = _pad1(params.ddz05, M + 1, "ddz05")   # ddz05[i] = Fortran ddz05(i-1)
    dzeta_int = _pad1(params.dzeta_int, M + 1, "dzeta_int")
    dzeta_05int = _pad1(params.dzeta_05int, M, "dzeta_05int")
    # Fortran CASE 1 (line 433) reads dzeta_05int(M+1) although the array is
    # dimensioned 1:M, i.e. it reads out of bounds.  Keep a sentinel slot so
    # that this (unreachable) branch still runs; 0.0 is a documented choice,
    # not the (undefined) Fortran value.
    dzeta_05int = np.concatenate([dzeta_05int, np.zeros(1)])
    area_int = _pad1(params.area_int, M + 1, "area_int")
    area_half = _pad1(params.area_half, M, "area_half")
    Lx = _pad1(params.Lx, M + 1, "Lx")
    Ly = _pad1(params.Ly, M + 1, "Ly")
    Lx_half = _pad1(params.Lx_half, M, "Lx_half")
    Ly_half = _pad1(params.Ly_half, M, "Ly_half")
    LxLyCDL = np.full((M + 2, 3), float(params.missing_value))
    LxLyCDL[1:M + 2, 1:3] = np.asarray(st.LxLyCDL, dtype=float)

    h1 = float(st.h1)
    l1 = float(st.l1)
    ls1 = float(st.ls1)
    dhw = float(st.dhw)
    dhw0 = float(st.dhw0)
    alphax = float(st.alphax)
    alphay = float(st.alphay)

    # Switch for the sloping-bottom drag law (line 249): botfric=1 -> 0.
    xbot = float(params.botfric - 1)

    Force_x = np.zeros(M + 2)
    Force_y = np.zeros(M + 2)

    # Accumulators/coefficient arrays for the 2x2 block tridiagonal system:
    # Fortran am_, bm_, cm_ are (1:nveclen,1:2,1:2), dm_/ym_ (1:nveclen,1:2).
    am_ = np.zeros((M + 2, 2, 2))
    bm_ = np.zeros((M + 2, 2, 2))
    cm_ = np.zeros((M + 2, 2, 2))
    dm_ = np.zeros((M + 2, 2))
    xx3 = np.zeros(3)

    # ---- firstcall block (lines 208-244) --------------------------------
    AL = _G / _ROW0                       # line 215
    ACC2 = 1.0e-20                        # 1.d-20
    ACCk = 1.0e-20                        # 1.d-20
    lam_force = 4.0e-1                    # 4.d-1 (assigned, never used)
    # month_sec/day_sec/hour_sec are assigned in the Fortran (lines 210-212)
    # but never used by this routine.

    if st.firstcall:
        knum[1:] = 0.0                    # line 220: knum = 0.
        if params.h_veg > 0.0:            # lines 226-230: STOP
            raise NotImplementedError(
                "vegetation friction (h_veg > 0) is not operational in LAKE "
                "3.2: the Fortran prints and STOPs")
        veg_sw[1:] = 0.0
        for i in range(1, M + 2):
            if h1 - dzeta_int[i] * h1 < params.h_veg:
                veg_sw[i] = 1.0
    horvisc_ON = 1 if params.dyn_pgrad == 4 else 0

    # ---- wind and surface stress (lines 260-315) ------------------------
    u_mod = u1[1]
    v_mod = v1[1]
    if params.relwind == 1:
        urel = params.uwind - u_mod
        vrel = params.vwind - v_mod
    elif params.relwind == 0:
        # UNREACHABLE for the target configuration (relwind=1).
        urel = params.uwind
        vrel = params.vwind
    else:
        raise ValueError(f"relwind={params.relwind}: incorrect meaning (STOP)")

    urels = _fsign(_MINWIND, urel) if abs(urel) < _MINWIND else urel
    vrels = _fsign(_MINWIND, vrel) if abs(vrel) < _MINWIND else vrel

    kor2 = 0.5 * params.kor * dt          # 0.5d0*kor*dt
    wr = math.sqrt(urels ** 2 + vrels ** 2)
    tau_air = _ROA0 * params.cdmw * wr
    ufr = math.sqrt(tau_air / _ROW0)

    # Wind speed at 10 m, Wuest and Lorke (2003); line 294.
    wind10 = wr * math.log(10.0 / _ROUGHNESS0) / math.log(
        params.zref / _ROUGHNESS0)
    C10 = params.cdmw * wr / (wind10 * wind10)

    # Partition of the momentum flux between waves and currents (line 311).
    tau_wav = (params.momflxpart * _ROA0 * C10
               * (wind10 - _dom_wave_speed(math.sqrt(tau_air / _ROA0),
                                           params.fetch) / _s(1.2)) ** 2)
    tau_sbl = max(0.0, tau_air - tau_wav)
    u_star = math.sqrt(tau_sbl / _ROW0)

    # ---- Chezy coefficients and bottom/ice friction (lines 320-363) -----
    if ls1 > 0.0:
        rp = _s(0.01)                     # roughness on ice bottom
    else:
        rp = _s(0.1)                      # roughness on ground bottom
    if l1 > 0.0:
        Cz_gr = _s(7.7) * math.sqrt(_G) * (0.5 * h1 / rp) ** _SIXTH
    else:
        Cz_gr = _s(7.7) * math.sqrt(_G) * (h1 / rp) ** _SIXTH
    if l1 > 0.0:
        rp = _s(0.01)
        Cz_i = _s(7.7) * math.sqrt(_G) * (0.5 * h1 / rp) ** _SIXTH
    else:
        Cz_i = 0.0                        # undefined in Fortran; unused

    tau_grx = 0.0                         # assigned, never used
    tau_gry = 0.0

    velx_log = 0.5 * (u1[M + 1] + u1[M])
    vely_log = 0.5 * (v1[M + 1] + v1[M])
    coef2 = (math.sqrt(velx_log ** 2 + vely_log ** 2) * _KAPPA ** 2
             / math.log(h1 * 0.5 * ddz[M] / _Z0_BOT) ** 2)
    velfrict_bot = (math.sqrt(velx_log ** 2 + vely_log ** 2) * _KAPPA
                    / math.log(h1 * 0.5 * ddz[M] / _Z0_BOT))
    tau_gr = _ROW0 * velfrict_bot ** 2

    coef1 = 0.0                           # only defined when l1 > 0
    tau_i = None
    if l1 > 0.0:
        velx_log = 0.5 * (u1[1] + u1[2])
        vely_log = 0.5 * (v1[1] + v1[2])
        coef1 = (-math.sqrt(velx_log ** 2 + vely_log ** 2) * _KAPPA ** 2
                 / math.log(h1 * 0.5 * ddz[1] / _Z0_BOT) ** 2)
        velfrict_ice = (math.sqrt(velx_log ** 2 + vely_log ** 2) * _KAPPA
                        / math.log(h1 * 0.5 * ddz[1] / _Z0_BOT))
        tau_i = _ROW0 * velfrict_ice ** 2
    else:
        velfrict_ice = float(st.velfrict_ice)

    # ---- lateral boundary friction (lines 365-409) ----------------------
    depth_area = np.asarray(params.depth_area, dtype=float)
    if depth_area.ndim != 2 or depth_area.shape[1] != 2:
        raise ValueError("depth_area must have shape (ndatamax, 2)")
    if depth_area[0, 1] >= 0.0:
        for i in range(2, M + 1):
            xx = _botfric_xx(params.botfric, u1[i], v1[i])
            lsu[i] = (-1.0 / area_int[i]
                      * (area_half[i] - area_half[i - 1])
                      / (ddz05[i] * h1) * xx)
            lsv[i] = (-1.0 / area_int[i]
                      * (area_half[i] - area_half[i - 1])
                      / (ddz05[i] * h1) * xx)
        xx = _botfric_xx(params.botfric, u1[1], v1[1])
        lsu[1] = (-2.0 / area_int[1]
                  * (area_half[1] - area_int[1]) / (ddz[1] * h1) * xx)
        # BUG preserved from the Fortran (line 405): the source writes
        # ``lsv%water(i)`` with the loop variable left at M+1 (not 1); index
        # M+1 is then zeroed on line 408, so lsv%water(1) is never assigned
        # in this branch.  With lsu/lsv carried as state, entry 1 keeps its
        # previous value (initially zero here).
        i_after = M + 1 if M >= 2 else 2
        lsv[i_after] = (-2.0 / area_int[1]
                        * (area_half[1] - area_int[1]) / (ddz[1] * h1) * xx)
        lsu[M + 1] = 0.0
        lsv[M + 1] = 0.0

    # ---- eddy viscosity (lines 412-506) ---------------------------------
    for i in range(1, M + 2):
        uv[i] = math.sqrt(u1[i] ** 2 + v1[i] ** 2)
    Ri = _richardson_number(M, u1, v1, ddz, h1, row)

    tp_name = ""                          # diagnostic only, unused
    if params.Turbpar == 1:
        tp_name = "SL_Empir"
        k2[1] = ((_LAMW0 * 10.0
                  + (wind10 * 2.0 / params.zref / 20.0)
                  * (_LAMW0 * 1000.0 - _LAMW0 * 10.0)) / _CW_M_ROW0)
        ext_lamw = math.log(k2[1] * _CW_M_ROW0 / (_LAMW0 * 10.0)) / h1
        for i in range(2, M + 2):
            # dzeta_05int[i] with i=M+1 is the sentinel described at the top.
            k2[i] = k2[1] * math.exp(-ext_lamw * dzeta_05int[i] * h1) + _NIU_WAT
        k2[1] = k2[1] + _NIU_WAT

    elif params.Turbpar == 2:
        for i in range(1, M + 2):
            if E1[i] <= 0.0:
                E1[i] = _s(10.0 ** -16)     # 10.**(-16) in real(4)
            if eps1[i] <= 0.0:
                eps1[i] = _s(10.0 ** -18)   # 10.**(-18) in real(4)
        CE = np.zeros(M + 1)
        for i in range(1, M + 1):
            if params.stabfunc == 1:
                CE[i] = _CE0
            else:
                lambda_N = ((E1[i] * E1[i]
                             / ((eps1[i] + ACC2) * (eps1[i] + ACC2)))
                            * (_set_ddens_dtemp(params, Tw1[i], Sal1[i])
                               * (Tw1[i + 1] - Tw1[i])
                               + _set_ddens_dsal(params, Tw1[i], Sal1[i])
                               * (Sal1[i + 1] - Sal1[i]))
                            / (h1 * ddz[i]) * AL)
                lambda_M = ((E1[i] * E1[i]
                             / ((eps1[i] + ACC2) * (eps1[i] + ACC2)))
                            * ((u1[i + 1] - u1[i]) * (u1[i + 1] - u1[i])
                               + (v1[i + 1] - v1[i]) * (v1[i + 1] - v1[i]))
                            / (h1 * h1 * ddz[i] * ddz[i]))
                if params.stabfunc == 2:
                    CE[i] = _ce_canuto(lambda_M, lambda_N)
                elif params.stabfunc == 3:
                    CE[i] = (math.sqrt(2.0) * _CL_K_KL_MODEL
                             * _smoment_galperin(lambda_N))
                else:
                    # Fortran leaves CE(i) undefined for other stabfunc.
                    raise ValueError(
                        f"stabfunc={params.stabfunc}: must be 1, 2 or 3")
        for i in range(1, M + 1):
            k2[i] = (max(CE[i] * E1[i] ** 2 / (eps1[i] + ACCk), _MIN_VISC)
                     + _NIU_WAT + knum[i])

    elif params.Turbpar == 3:
        tp_name = "NICK"
        for i in range(1, M + 1):
            zup = h1 - dzeta_05int[i] * h1
            lm = h1 * (_s(0.14) - _s(0.08) * (1.0 - zup / h1) ** 2
                       - _s(0.06) * (1.0 - zup / h1) ** 4)
            k2[i] = (lm ** 2 * abs((uv[i + 1] - uv[i]) / (ddz[i] * h1))
                     * math.exp(-_CS * Ri[i]) + _NIU_WAT)

    elif params.Turbpar == 4:
        tp_name = "PARAB"
        for i in range(1, M + 1):
            zup = h1 - dzeta_05int[i] * h1
            k2[i] = (_KAR * ufr * zup * (1.0 - zup / h1)
                     * math.exp(-_CS * Ri[i]) + _NIU_WAT)

    elif params.Turbpar == 5:
        # UNREACHABLE for the target configuration (Turbpar=2).
        # TODO(momentum): port EDDY_DIFF_MOST_RICH / MOST_FUNCTIONS_RICH
        # (turb_mod.f90:1857-1899 and below) before enabling Turbpar=5.
        raise NotImplementedError(
            "TODO: Turbpar=5 needs EDDY_DIFF_MOST_RICH, which is not ported")

    elif params.Turbpar == 6:
        # Fortran prints and STOPs (lines 494-496).
        raise RuntimeError("Turbpar = 6 is not an operational setting: STOP")

    elif params.Turbpar == 7:
        tp_name = "RNG"
        for i in range(1, M + 1):
            zup = h1 - dzeta_05int[i] * h1
            k2[i] = (_NIU_WAT
                     * (1.0 + max(3.0 * _KAR * (zup / _NIU_WAT * ufr) ** 3
                                  * (1.0 - zup / h1) ** 3 - _CS1, 0.0))
                     ** _THIRD
                     * math.exp(-_CS * Ri[i]) + _NIU_WAT)

    else:
        raise ValueError(f"Turbpar={params.Turbpar}: no such parameterization")

    # ---- coefficients of the block system (lines 518-1398) --------------
    u2 = u1.copy()
    v2 = v1.copy()
    if params.Turbpar == 5:               # UNREACHABLE, raised above
        raise NotImplementedError("TODO: Turbpar=5")
    else:
        k2_old[1:M + 1] = k2[1:M + 1]

    um = u1.copy()
    vm = v1.copy()

    b0 = None
    taux = float(st.taux)
    tauy = float(st.tauy)

    # ===== top boundary (lines 545-1272) ================================
    if params.cuette in (0, 11):
        k2[1] = area_half[1] / area_int[1] * k2[1]
        k2_old[1] = area_half[1] / area_int[1] * k2_old[1]

        xx_old = 0.5 * (2.0 - dhw0 * h1 * ddz[1] / (k2[1] * dt))
        xx = 0.5 * (2.0 * k2_old[1] / k2[1]
                    - dhw0 * h1 * ddz[1] / (k2[1] * dt))
        yy = (ddz[1] * h1) ** 2 / (k2[1] * dt)

        # ---- open water: l1 == 0 (lines 562-1169) ----------------------
        if l1 == 0.0:
            if params.cuette == 11 or params.PBLpar == 0:
                # UNREACHABLE for the target configuration (cuette=0,
                # PBLpar=3): constant momentum flux at the top.
                taux = params.momflux0
                tauy = 0.0
            else:
                taux = urels / wr * tau_sbl
                tauy = vrels / wr * tau_sbl

            if params.dyn_pgrad == 1:
                # UNREACHABLE for the target configuration (dyn_pgrad=0).
                hx1, hx2, hy1, hy2 = dhdxdhdy(
                    M, u1, v1, area_int, Lx, Ly, ddz05, h1, dt,
                    st.hx1, st.hx2, st.hy1, st.hy2)
                st.hx1, st.hx2, st.hy1, st.hy2 = hx1, hx2, hy1, hy2
                fx = -_G * _PI * 0.5 * (hx2 - hx1) / Lx[1]
                Force_x[1:] = fx
                Force_y[1:] = -_G * _PI * 0.5 * (hy2 - hy1) / Ly[1]

            elif params.dyn_pgrad == 2 and (1 < st.i_maxN < M + 1):
                # UNREACHABLE for the target configuration (dyn_pgrad=0).
                i_maxN = st.i_maxN
                (hx1, hx2, hy1, hy2,
                 hx1t, hx2t, hy1t, hy2t) = dhdxdhdy2(
                    i_maxN, M, u1, v1, area_int, area_half, Lx, Ly, ddz05,
                    h1, dt, st.hx1, st.hx2, st.hy1, st.hy2,
                    st.hx1t, st.hx2t, st.hy1t, st.hy2t)
                st.hx1, st.hx2, st.hy1, st.hy2 = hx1, hx2, hy1, hy2
                st.hx1t, st.hx2t, st.hy1t, st.hy2t = hx1t, hx2t, hy1t, hy2t

                fx = -_G * _PI * 0.5 * (
                    (hx2 - hx1) / Lx[1]
                    + (hx2t - hx1t) / Lx_half[i_maxN - 1])
                fy = -_G * _PI * 0.5 * (
                    (hy2 - hy1) / Ly[1]
                    + (hy2t - hy1t) / Ly_half[i_maxN - 1])
                Force_x[1:i_maxN] = fx
                Force_y[1:i_maxN] = fy

                # Variable grid spacing to be included (Fortran comment).
                row1 = float(np.sum(row[1:i_maxN])) / float(i_maxN - 1)
                row2 = float(np.sum(row[i_maxN:M + 2])) / float(M + 2 - i_maxN)
                fx = -_G * _PI * 0.5 * (
                    row1 / (row2 * Lx[1]) * (hx2 - hx1)
                    + (hx2t - hx1t) / Lx_half[i_maxN - 1])
                fy = -_G * _PI * 0.5 * (
                    row1 / (row2 * Ly[1]) * (hy2 - hy1)
                    + (hy2t - hy1t) / Ly_half[i_maxN - 1])
                Force_x[i_maxN:] = fx
                Force_y[i_maxN:] = fy

            elif params.dyn_pgrad == 3:
                # UNREACHABLE for the target configuration (dyn_pgrad=0).
                hx1ml, hx2ml, hy1ml, hy2ml = dhdxdhdy_ml(
                    M, u1, v1, area_int, area_half, Lx, Ly, ddz05, h1, dt,
                    hx1ml, hx2ml, hy1ml, hy2ml)
                # x-pressure gradient
                xx1 = row[1] * (hx2ml[1] - hx1ml[1]) / Lx[1]
                yy1 = 0.0
                for i in range(2, M + 2):
                    yy1 = yy1 + (hx2ml[i] - hx1ml[i]) / Lx_half[i - 1]
                for i in range(1, M + 2):
                    Force_x[i] = -0.5 / row[i] * _PI * _G * (xx1 + yy1 * row[i])
                    if i != M + 1:
                        xx1 = xx1 + row[i + 1] * (hx2ml[i + 1] - hx1ml[i + 1]) / Lx_half[i]
                        yy1 = yy1 - (hx2ml[i + 1] - hx1ml[i + 1]) / Lx_half[i]
                # y-pressure gradient
                xx1 = row[1] * (hy2ml[1] - hy1ml[1]) / Ly[1]
                yy1 = 0.0
                for i in range(2, M + 2):
                    yy1 = yy1 + (hy2ml[i] - hy1ml[i]) / Ly_half[i - 1]
                for i in range(1, M + 2):
                    Force_y[i] = -0.5 / row[i] * _PI * _G * (xx1 + yy1 * row[i])
                    if i != M + 1:
                        xx1 = xx1 + row[i + 1] * (hy2ml[i + 1] - hy1ml[i + 1]) / Ly_half[i]
                        yy1 = yy1 - (hy2ml[i + 1] - hy1ml[i + 1]) / Ly_half[i]
                if 1 < st.i_maxN < M + 1:
                    st.hx1 = float(np.sum(hx1ml[1:st.i_maxN]))
                    st.hx2 = float(np.sum(hx2ml[1:st.i_maxN]))
                    st.hy1 = float(np.sum(hy1ml[1:st.i_maxN]))
                    st.hy2 = float(np.sum(hy2ml[1:st.i_maxN]))
                    st.hx1t = float(np.sum(hx1ml[st.i_maxN:M + 2]))
                    st.hx2t = float(np.sum(hx2ml[st.i_maxN:M + 2]))
                    st.hy1t = float(np.sum(hy1ml[st.i_maxN:M + 2]))
                    st.hy2t = float(np.sum(hy2ml[st.i_maxN:M + 2]))
                else:
                    st.hx1 = float(np.sum(hx1ml[1:M + 2]))
                    st.hx2 = float(np.sum(hx2ml[1:M + 2]))
                    st.hy1 = float(np.sum(hy1ml[1:M + 2]))
                    st.hy2 = float(np.sum(hy2ml[1:M + 2]))
                    st.hx1t = 0.0
                    st.hx2t = 0.0
                    st.hy1t = 0.0
                    st.hy2t = 0.0

            elif params.dyn_pgrad == 4:
                # UNREACHABLE for the target configuration (dyn_pgrad=0).
                # TODO(momentum): port lines 655-1126 of momentum_mod.f90:
                #   * DENSLAYERS (lines 1878-2170) and RELAYER (2175-2310);
                #   * the nlevs==1 implicit scheme, the (dead) explicit
                #     scheme and the Crank-Nicolson scheme with SGESV,
                #     SQCOEFS and tributary corrections.
                raise NotImplementedError(
                    "TODO: dyn_pgrad=4 (multi-layer seiches) needs "
                    "DENSLAYERS/RELAYER, which are not ported")

            else:
                # dyn_pgrad 0 (target), 5, 6 and any other value
                Force_x[1:] = 0.0
                Force_y[1:] = 0.0

            # Horizontal pressure gradient for the river case (line 1131).
            if params.dyn_pgrad == 5:
                # UNREACHABLE for the target configuration.
                if params.disch_tribin is None:
                    raise ValueError("dyn_pgrad=5 needs disch_tribin")
                scross = h1 * Ly[1]
                hydr_radius = scross / (Ly[1] + 2.0 * h1)
                disch = float(np.sum(params.disch_tribin))
                speedav = disch / scross
                alphax = 180.0 / _PI * (
                    speedav ** 2 * params.nManning ** 2
                    / (hydr_radius ** 4.0 / 3.0)
                    - taux / (_ROW0 * h1 * _G))

            if params.dyn_pgrad == 6:
                # UNREACHABLE for the target configuration.
                alphax = 180.0 / _PI * params.pgrad / (_ROW0 * _G)

            for i in (1, 2):
                xx3[i] = (horvisc_ON * 0.5 * _PI * _PI * params.horvisc * dt
                          / LxLyCDL[1, i] ** 2)
                cm_[1, i - 1, i - 1] = (
                    xx + yy * (1.0 + xx3[i])
                    + 0.5 * yy * dt * params.a_veg * params.c_veg
                    * math.sqrt(um[1] ** 2 + vm[1] ** 2) * veg_sw[1])
                bm_[1, i - 1, i - 1] = xx
            # Friction at the lateral boundaries
            cm_[1, 0, 0] = (cm_[1, 0, 0]
                            + yy * dt * lsu[1] * (0.5 * (1.0 - xbot) + xbot))
            cm_[1, 1, 1] = (cm_[1, 1, 1]
                            + yy * dt * lsv[1] * (0.5 * (1.0 - xbot) + xbot))

            bm_[1, 0, 1] = 0.0
            bm_[1, 1, 0] = 0.0
            cm_[1, 0, 1] = -params.kor * (h1 * ddz[1]) ** 2 / (2.0 * k2[1])
            cm_[1, 1, 0] = params.kor * (h1 * ddz[1]) ** 2 / (2.0 * k2[1])

            dm_[1, 0] = (
                (2.0 * taux * ddz[1] * h1 / (_ROW0 * k2[1])
                 + yy * (1.0 - xx3[1]) * um[1])
                + (um[2] - um[1]) * xx_old
                + params.kor * (h1 * ddz[1]) ** 2 * vm[1] / (2.0 * k2[1])
                + yy * dt * _G * math.tan(_PI * alphax / 180.0)
                + yy * dt * Force_x[1]
                - 0.5 * (1.0 - xbot) * lsu[1] * um[1] * yy * dt)
            dm_[1, 1] = (
                (2.0 * tauy * ddz[1] * h1 / (_ROW0 * k2[1])
                 + yy * (1.0 - xx3[2]) * vm[1])
                + (vm[2] - vm[1]) * xx_old
                - params.kor * (h1 * ddz[1]) ** 2 * um[1] / (2.0 * k2[1])
                + yy * dt * _G * math.tan(_PI * alphay / 180.0)
                + yy * dt * Force_y[1]
                - 0.5 * (1.0 - xbot) * lsv[1] * vm[1] * yy * dt)

        # ---- thin ice: 0 < l1 <= L0 (lines 1176-1214) -----------------
        if 0.0 < l1 <= _L0:
            b0 = l1 / _L0
            taux = urels / wr * tau_sbl
            tauy = vrels / wr * tau_sbl

            Force_x[1:] = 0.0
            Force_y[1:] = 0.0

            for i in (1, 2):
                cm_[1, i - 1, i - 1] = (
                    xx + yy - 0.5 * 2.0 * b0 * coef1 * ddz[1] * h1 / k2[1]
                    + 0.5 * yy * dt * params.a_veg * params.c_veg
                    * math.sqrt(um[1] ** 2 + vm[1] ** 2) * veg_sw[1])
                bm_[1, i - 1, i - 1] = (
                    xx + 0.5 * 2.0 * b0 * coef1 * ddz[1] * h1 / k2[1])
            cm_[1, 0, 0] = (cm_[1, 0, 0]
                            + yy * dt * lsu[1] * (0.5 * (1.0 - xbot) + xbot))
            cm_[1, 1, 1] = (cm_[1, 1, 1]
                            + yy * dt * lsv[1] * (0.5 * (1.0 - xbot) + xbot))

            bm_[1, 0, 1] = 0.0
            bm_[1, 1, 0] = 0.0
            cm_[1, 0, 1] = -params.kor * (h1 * ddz[1]) ** 2 / (2.0 * k2[1])
            cm_[1, 1, 0] = params.kor * (h1 * ddz[1]) ** 2 / (2.0 * k2[1])

            dm_[1, 0] = yy * um[1]
            dm_[1, 0] = (
                dm_[1, 0] + (um[2] - um[1]) * xx_old
                + 2.0 * taux * ddz[1] * h1 / (_ROW0 * k2[1]) * (1.0 - b0)
                + params.kor * (h1 * ddz[1]) ** 2 * vm[1] / (2.0 * k2[1])
                + yy * dt * _G * math.tan(_PI * alphax / 180.0)
                + yy * dt * Force_x[1]
                - 0.5 * (1.0 - xbot) * lsu[1] * um[1] * yy * dt)
            dm_[1, 1] = yy * vm[1]
            dm_[1, 1] = (
                dm_[1, 1] + (vm[2] - vm[1]) * xx_old
                + 2.0 * tauy * ddz[1] * h1 / (_ROW0 * k2[1]) * (1.0 - b0)
                - params.kor * (h1 * ddz[1]) ** 2 * um[1] / (2.0 * k2[1])
                + yy * dt * _G * math.tan(_PI * alphay / 180.0)
                + yy * dt * Force_y[1]
                - 0.5 * (1.0 - xbot) * lsv[1] * vm[1] * yy * dt)

        # ---- thick ice: l1 > L0 (lines 1219-1255) ---------------------
        if l1 > _L0:
            Force_x[1:] = 0.0
            Force_y[1:] = 0.0

            for i in (1, 2):
                cm_[1, i - 1, i - 1] = (
                    xx + yy - 0.5 * 2.0 * coef1 * ddz[1] * h1 / k2[1]
                    + 0.5 * yy * dt * params.a_veg * params.c_veg
                    * math.sqrt(um[1] ** 2 + vm[1] ** 2) * veg_sw[1])
                bm_[1, i - 1, i - 1] = (
                    xx + 0.5 * 2.0 * coef1 * ddz[1] * h1 / k2[1])
            cm_[1, 0, 0] = (cm_[1, 0, 0]
                            + yy * dt * lsu[1] * (0.5 * (1.0 - xbot) + xbot))
            cm_[1, 1, 1] = (cm_[1, 1, 1]
                            + yy * dt * lsv[1] * (0.5 * (1.0 - xbot) + xbot))

            bm_[1, 0, 1] = 0.0
            bm_[1, 1, 0] = 0.0
            cm_[1, 0, 1] = -params.kor * (h1 * ddz[1]) ** 2 / (2.0 * k2[1])
            cm_[1, 1, 0] = params.kor * (h1 * ddz[1]) ** 2 / (2.0 * k2[1])

            dm_[1, 0] = yy * um[1]
            dm_[1, 0] = (
                dm_[1, 0] + (um[2] - um[1]) * xx_old
                + params.kor * (h1 * ddz[1]) ** 2 * vm[1] / (2.0 * k2[1])
                + yy * dt * _G * math.tan(_PI * alphax / 180.0)
                + yy * dt * Force_x[1]
                - 0.5 * (1.0 - xbot) * lsu[1] * um[1] * dt * yy)
            dm_[1, 1] = yy * vm[1]
            dm_[1, 1] = (
                dm_[1, 1] + (vm[2] - vm[1]) * xx_old
                - params.kor * (h1 * ddz[1]) ** 2 * um[1] / (2.0 * k2[1])
                + yy * dt * _G * math.tan(_PI * alphay / 180.0)
                + yy * dt * Force_y[1]
                - 0.5 * (1.0 - xbot) * lsv[1] * vm[1] * dt * yy)

        k2[1] = area_int[1] / area_half[1] * k2[1]
        k2_old[1] = area_int[1] / area_half[1] * k2_old[1]

    elif params.cuette == 1:
        # UNREACHABLE for the target configuration (cuette=0).
        # B.c.s for Couette flow: top boundary.
        cm_[1, :, :] = 0.0
        bm_[1, :, :] = 0.0
        cm_[1, 0, 0] = 1.0
        cm_[1, 1, 1] = 1.0
        dm_[1, 0] = u1[1]
        dm_[1, 1] = v1[1]
    else:
        raise ValueError(f"cuette={params.cuette}: must be 0, 1 or 11")

    # ===== bottom boundary (lines 1274-1321) ============================
    if params.cuette in (0, 11):
        k2[M] = area_half[M] / area_int[M + 1] * k2[M]
        k2_old[M] = area_half[M] / area_int[M + 1] * k2_old[M]

        xx_old = 0.5 * (2.0 - ddz[M] * h1 * (dhw - dhw0) / (k2[M] * dt))
        xx = 0.5 * (2.0 * k2_old[M] / k2[M]
                    - ddz[M] * h1 * (dhw - dhw0) / (k2[M] * dt))
        yy = (ddz[M] * h1) ** 2 / (k2[M] * dt)

        for i in (1, 2):
            xx3[i] = (horvisc_ON * 0.5 * _PI * _PI * params.horvisc * dt
                      / LxLyCDL[M + 1, i] ** 2)
            cm_[M + 1, i - 1, i - 1] = (
                xx + yy * (1.0 + xx3[i])
                + 0.5 * 2.0 * coef2 * ddz[M] * h1 / k2[M]
                + 0.5 * yy * dt * params.a_veg * params.c_veg
                * math.sqrt(um[M + 1] ** 2 + vm[M + 1] ** 2) * veg_sw[M + 1])
            am_[M + 1, i - 1, i - 1] = (
                xx - 0.5 * 2.0 * coef2 * ddz[M] * h1 / k2[M])

        am_[M + 1, 0, 1] = 0.0
        am_[M + 1, 1, 0] = 0.0
        cm_[M + 1, 0, 1] = -params.kor * (h1 * ddz[M]) ** 2 / (2.0 * k2[M])
        cm_[M + 1, 1, 0] = params.kor * (h1 * ddz[M]) ** 2 / (2.0 * k2[M])

        dm_[M + 1, 0] = yy * (1.0 - xx3[1]) * um[M + 1]
        dm_[M + 1, 0] = (
            dm_[M + 1, 0] - (um[M + 1] - um[M]) * xx_old
            + params.kor * (h1 * ddz[M]) ** 2 * vm[M + 1] / (2.0 * k2[M])
            + yy * dt * _G * math.tan(_PI * alphax / 180.0)
            + yy * dt * Force_x[M + 1])
        dm_[M + 1, 1] = yy * (1.0 - xx3[2]) * vm[M + 1]
        dm_[M + 1, 1] = (
            dm_[M + 1, 1] - (vm[M + 1] - vm[M]) * xx_old
            - params.kor * (h1 * ddz[M]) ** 2 * um[M + 1] / (2.0 * k2[M])
            + yy * dt * _G * math.tan(_PI * alphay / 180.0)
            + yy * dt * Force_y[M + 1])

        k2[M] = area_int[M + 1] / area_half[M] * k2[M]
        k2_old[M] = area_int[M + 1] / area_half[M] * k2_old[M]

    elif params.cuette == 1:
        # UNREACHABLE for the target configuration (cuette=0).
        cm_[M + 1, :, :] = 0.0
        am_[M + 1, :, :] = 0.0
        cm_[M + 1, 0, 0] = 1.0
        cm_[M + 1, 1, 1] = 1.0
        dm_[M + 1, 0] = u1[M + 1]
        dm_[M + 1, 1] = v1[M + 1]

    # ===== interior points (lines 1328-1381) ============================
    for k in range(2, M + 1):
        for i in (1, 2):
            for j in (1, 2):
                am_[k, i - 1, j - 1] = _kron(i, j) * (
                    dzeta_int[k] * dhw / (2.0 * (ddz[k - 1] + ddz[k]) * h1)
                    - dhw0 / (2.0 * (ddz[k - 1] + ddz[k]) * h1)
                    - area_half[k - 1] / area_int[k] * k2_old[k - 1] * dt
                    / (h1 ** 2 * (ddz[k] + ddz[k - 1]) * ddz[k - 1]))
                bm_[k, i - 1, j - 1] = -_kron(i, j) * (
                    dzeta_int[k] * dhw / (2.0 * (ddz[k - 1] + ddz[k]) * h1)
                    + dhw0 / (2.0 * (ddz[k - 1] + ddz[k]) * h1)
                    + area_half[k] / area_int[k] * k2_old[k] * dt
                    / (h1 ** 2 * (ddz[k] + ddz[k - 1]) * ddz[k]))

        xx = (-(area_half[k] / area_int[k] * k2_old[k] / ddz[k]
                + area_half[k - 1] / area_int[k] * k2_old[k - 1] / ddz[k - 1])
              * dt / (h1 ** 2 * (ddz[k] + ddz[k - 1]))
              - 0.5 * dt * params.c_veg * params.a_veg
              * math.sqrt(um[k] ** 2 + vm[k] ** 2) * veg_sw[k])

        xx3[1] = (horvisc_ON * 0.5 * _PI * _PI * params.horvisc * dt
                  / LxLyCDL[k, 1] ** 2)
        xx3[2] = (horvisc_ON * 0.5 * _PI * _PI * params.horvisc * dt
                  / LxLyCDL[k, 2] ** 2)
        cm_[k, 0, 0] = xx - 1.0 * (1.0 + xx3[1])
        cm_[k, 1, 1] = xx - 1.0 * (1.0 + xx3[2])
        cm_[k, 0, 0] = (cm_[k, 0, 0]
                        - dt * 0.5 * lsu[k] * (0.5 * (1.0 - xbot) + xbot))
        cm_[k, 1, 1] = (cm_[k, 1, 1]
                        - dt * 0.5 * lsv[k] * (0.5 * (1.0 - xbot) + xbot))

        cm_[k, 0, 1] = kor2
        cm_[k, 1, 0] = -cm_[k, 0, 1]

        dm_[k, 0] = (
            -um[k] * (1.0 - xx3[1])
            - kor2 * vm[k]
            - dt * _G * math.tan(_PI * alphax / 180.0)
            - Force_x[k] * dt
            + 0.5 * (1.0 - xbot) * lsu[k] * um[k] * dt
            - dt * (h1 ** 2 * (ddz[k] + ddz[k - 1]) * ddz[k]) ** (-1)
            * (area_half[k] / area_int[k] * k2[k] * (um[k + 1] - um[k]))
            + dt * (h1 ** 2 * (ddz[k] + ddz[k - 1]) * ddz[k - 1]) ** (-1)
            * (area_half[k - 1] / area_int[k] * k2[k - 1] * (um[k] - um[k - 1]))
            - dzeta_int[k] * dhw
            * (2.0 * (ddz[k] + ddz[k - 1]) * h1) ** (-1)
            * (um[k + 1] - um[k - 1])
            + dhw0 * (2.0 * (ddz[k] + ddz[k - 1]) * h1) ** (-1)
            * (um[k + 1] - um[k - 1]))

        dm_[k, 1] = (
            -vm[k] * (1.0 - xx3[2])
            + kor2 * um[k]
            - dt * _G * math.tan(_PI * alphay / 180.0)
            - Force_y[k] * dt
            + 0.5 * (1.0 - xbot) * lsv[k] * vm[k] * dt
            - dt * (h1 ** 2 * (ddz[k] + ddz[k - 1]) * ddz[k]) ** (-1)
            * (area_half[k] / area_int[k] * k2[k] * (vm[k + 1] - vm[k]))
            + dt * (h1 ** 2 * (ddz[k] + ddz[k - 1]) * ddz[k - 1]) ** (-1)
            * (area_half[k - 1] / area_int[k] * k2[k - 1] * (vm[k] - vm[k - 1]))
            - dzeta_int[k] * dhw
            * (2.0 * (ddz[k] + ddz[k - 1]) * h1) ** (-1)
            * (vm[k + 1] - vm[k - 1])
            + dhw0 * (2.0 * (ddz[k] + ddz[k - 1]) * h1) ** (-1)
            * (vm[k + 1] - vm[k - 1]))

    # ===== solve and outputs (lines 1393-1418) ==========================
    # MATRIXPROGONKA: -am_(k) y(k-1) + cm_(k) y(k) - bm_(k) y(k+1) = dm_(k)
    ym = matrixprogonka(am_[1:M + 2], bm_[1:M + 2], cm_[1:M + 2],
                        dm_[1:M + 2])
    u2[1:] = ym[:, 0]
    v2[1:] = ym[:, 1]

    windwork = taux * u2[1] + tauy * v2[1]
    discharge = np.zeros(2)
    for i in range(1, M + 2):
        discharge[0] = (discharge[0]
                        + h1 * ddz05[i] * u1[i] * Ly[i])
        discharge[1] = (discharge[1]
                        + h1 * ddz05[i] * v1[i] * Lx[i])

    # ---- write the updated state back ----------------------------------
    st.u1 = _strip1(u1)
    st.v1 = _strip1(v1)
    st.E1 = _strip1(E1)
    st.eps1 = _strip1(eps1)
    st.k2 = _strip1(k2)
    st.k2_old = _strip1(k2_old)
    st.row = _strip1(row)
    st.Ri = _strip1(Ri)
    st.knum = _strip1(knum)
    st.veg_sw = _strip1(veg_sw)
    st.u_star = u_star
    st.Tw1 = _strip1(Tw1)
    st.Sal1 = _strip1(Sal1)
    st.lsu_water = _strip1(lsu)
    st.lsv_water = _strip1(lsv)
    st.uv = _strip1(uv)
    st.h1 = h1
    st.l1 = l1
    st.ls1 = ls1
    st.dhw = dhw
    st.dhw0 = dhw0
    st.alphax = alphax
    st.alphay = alphay
    st.velfrict_bot = velfrict_bot
    st.velfrict_ice = velfrict_ice
    st.taux = taux
    st.tauy = tauy
    st.hx1ml = _strip1(hx1ml)
    st.hx2ml = _strip1(hx2ml)
    st.hy1ml = _strip1(hy1ml)
    st.hy2ml = _strip1(hy2ml)
    st.gradpx_tend = _strip1(gradpx_tend)
    st.gradpy_tend = _strip1(gradpy_tend)
    st.rowc = _strip1(rowc)
    st.itherm = np.asarray(itherm[1:], dtype=int)
    st.LxLyCDL = LxLyCDL[1:M + 2, 1:3].copy()
    st.firstcall = False

    return MomentumResult(
        u2=_strip1(u2),
        v2=_strip1(v2),
        b0=b0,
        tau_air=tau_air,
        tau_gr=tau_gr,
        tau_i=tau_i,
        tau_wav=tau_wav,
        state=st,
        taux=taux,
        tauy=tauy,
        tau_sbl=tau_sbl,
        u_star=u_star,
        velfrict_bot=velfrict_bot,
        velfrict_ice=velfrict_ice,
        wind10=wind10,
        windwork=windwork,
        discharge=discharge,
        u=u_mod,
        v=v_mod,
        urel=urel,
        vrel=vrel,
        urels=urels,
        vrels=vrels,
        wr=wr,
        C10=C10,
        ufr=ufr,
        coef1=coef1,
        coef2=coef2,
        Cz_gr=Cz_gr,
        Cz_i=Cz_i,
        rp=rp,
        xbot=xbot,
        horvisc_on=horvisc_ON,
        Force_x=_strip1(Force_x),
        Force_y=_strip1(Force_y),
    )
