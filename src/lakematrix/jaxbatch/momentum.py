"""JAX-vectorised ``MOMENTUM_SOLVER``, batch over lakes (case-1 testlake).

Batched port of :func:`lakematrix.physics.momentum.momentum_solver`
(Fortran ``MOMENTUM_SOLVER``, ``momentum_mod.f90:15-1611``) with a leading
lake axis ``(L, ...)``.  The NumPy implementation stays the reference; this
kernel reproduces it statement by statement, with the ``if`` branches of the
scalar code replaced by ``jnp.where`` and the 2x2 block-tridiagonal solve done
by :func:`lakematrix.physics.blocktridiag.matrixprogonka_jax` (the same
recurrence the NumPy reference uses).

Supported configuration
-----------------------
The target configuration of the batch rewrite (``lake_case 1`` / testlake):

``Turbpar=2`` (Canuto closure), ``l1=0`` (open water), ``ls1=0``,
``botfric=1``, ``dyn_pgrad=0`` (no seiche/pressure-gradient force),
``cuette=0``, ``relwind=1``, ``PBLpar=3``, ``eos=5``/``lindens=0``,
no vegetation (``h_veg=0``), no tributaries, flat bottom (``alphax=alphay=0``
in the test state), no horizontal viscosity (``horvisc_ON = (dyn_pgrad == 4)
= 0``, so ``horvisc``/``LxLyCDL`` are ignored exactly as in NumPy).
``momflxpart=0`` by default, but the (cheap) wave/current momentum-flux
partition is transcribed, so a non-zero value also works.

Branches that cannot be reached in that configuration are **not** ported and
raise ``NotImplementedError``:

* ``Turbpar`` 1/3/4/5/6/7 (only 2 is ported),
* ``dyn_pgrad`` 1/2/3/4/5/6 (only 0, i.e. zero pressure gradient, is ported),
* ``cuette=1`` (Couette test case), ``stabfunc`` other than 2/3,
* vegetation friction ``h_veg > 0`` (a ``STOP`` in LAKE 3.2 as well),
* ``eos`` other than 5 / ``lindens != 0`` (the density derivatives).

Ice (``l1 > 0``) is *transcribed* (thin ice ``0 < l1 <= L0`` and thick ice
``l1 > L0`` top-boundary rows are evaluated and selected with ``jnp.where``),
so the velocity solution is faithful for an ice-covered column too.  The
NumPy reference returns ``b0 = None`` and ``tau_i = None`` where Fortran
leaves them undefined; here ``b0`` is ``0`` outside ``0 < l1 <= L0`` and
``tau_i`` is ``0`` for ``l1 <= 0`` (documented deviation, no physics change).

Interface
---------
``momentum_solver(state, params, dt) -> dict``

``state`` (mapping, every array has a leading lake axis ``L``; **0-based**,
following :class:`lakematrix.grid.Grid` and
:class:`lakematrix.physics.momentum.MomentumState`)::

    u1, v1        (L, M+1)   old x/y velocity                    [read]
    E1, eps1      (L, M+1)   TKE, dissipation (floored if <= 0)   [read]
    k2            (L, M+1)   eddy viscosity (overwritten 1..M)    [read]
    k2_old        (L, M)     viscosity of the previous call       [read]
    row           (L, M+1)   density (Ri only)                    [read]
    knum          (L, M+1)   wave-breaking TKE input              [read]
    veg_sw        (L, M+1)   vegetation switch                    [read]
    Tw1, Sal1     (L, M+1)   temperature, salinity (Canuto)       [read]
    lsu_water     (L, M+1)   lateral friction u (carried over)    [read]
    lsv_water     (L, M+1)   lateral friction v (carried over)    [read]
    h1            (L,)       water depth, m                       [read]
    dhw, dhw0     (L,)       water-depth change, this/last step   [read]
    l1, ls1       (L,)       ice and snow thickness, m            [read]
    alphax, alphay (L,)      surface slopes, degrees              [read]
    velfrict_ice  (L,)       previous ice friction velocity       [read]
                            (only used when ``l1 <= 0``; optional,
                            defaults to 0)

Not read (recomputed on every call, kept for interface parity):
``uv``, ``Ri``, ``u_star``, ``taux``, ``tauy``, ``velfrict_bot``.  The
``firstcall`` block of the Fortran (zeroing ``knum``/``veg_sw``) is the
caller's job: the batch driver obtains them from the ``K_EPSILON`` state, as
``step.build_step_state`` already does.

``params`` (mapping, leading lake axis; required keys)::

    ddz           (L, M)      Fortran ddz(1:M)
    ddz05         (L, M+1)    Fortran ddz05(0:M)  (index 0 kept, as in NumPy)
    dzeta_int     (L, M+1)
    dzeta_05int   (L, M)      accepted, unused for Turbpar=2
    area_int      (L, M+1)
    area_half     (L, M)
    Lx, Ly        (L, M+1)
    Lx_half       (L, M)      accepted, unused (only horvisc/LxLyCDL uses them)
    Ly_half       (L, M)      accepted, unused
    depth_area    (L, ndata, 2) or (L, 2)
                              only ``[..., 0, 1] >= 0`` is used, exactly like
                              ``if depth_area(1,2) >= 0`` in the NumPy port
    kor           (L,)        Coriolis parameter, 1/s
    uwind, vwind  (L,)        wind at ``zref``, m/s
    cdmw          (L,)        momentum exchange coefficient
    zref          (L,)        wind reference height, m
    fetch         (L,)        fetch, m

Not required: ``M`` (derived from ``area_half.shape[-1]``), ``missing_value``
and the seiche/coupling arrays (``LxLyCDL``, ``hx*``, ``itherm``, ...).

Switches are keyword-only Python arguments (static under ``jax.jit``); the
defaults are the target configuration:
``Turbpar=2, stabfunc=2, botfric=1, relwind=1, dyn_pgrad=0, cuette=0,
PBLpar=3, momflxpart=0.0, a_veg=0.0, c_veg=0.0, horvisc=0.0, eos=5,
lindens=0, h_veg=0.0``.

Returned dict
-------------
``u2``, ``v2``        (L, M+1)   new velocity (feed back as ``u1``/``v1``)
``k2``                (L, M+1)   eddy viscosity (feed back as ``k2``)
``k2_old``            (L, M)     viscosity of this call (feedback as above)
``E1``, ``eps1``      (L, M+1)   floored TKE/dissipation
``lsu_water``, ``lsv_water`` (L, M+1) updated lateral friction
``uv``                (L, M+1)   |u| of the old velocity
``Ri``                (L, M)     Richardson number
``knum``, ``veg_sw``  (L, M+1)   passed through
``taux``, ``tauy``, ``tau_air``, ``tau_gr``, ``tau_i``, ``tau_wav``,
``tau_sbl``, ``u_star``, ``velfrict_bot``, ``velfrict_ice``, ``wind10``,
``C10``, ``ufr``, ``coef1``, ``coef2``, ``Cz_gr``, ``Cz_i``, ``rp``, ``b0``,
``u``, ``v``, ``urel``, ``vrel``, ``urels``, ``vrels``, ``wr``, ``windwork``

                 all ``(L,)`` (``xbot`` is a float, ``horvisc_on`` an int)
``discharge``         (L, 2)
``Force_x``, ``Force_y`` (L, M+1)  (identically zero for ``dyn_pgrad=0``)

Notes
-----
* The Fortran/NumPy order of operations is reproduced statement by statement
  (including the ``area_half/area_int`` scalings of ``k2`` at the top and
  bottom rows and their exact inverse), and every vectorised expression keeps
  the operand order of the scalar statement, so the float64 coefficients and
  the solution agree with the NumPy reference to the last bit (eager).
* The lateral-friction bug of the Fortran at line 405 is preserved:
  ``lsv_water(1)`` is never assigned in that block, so entry 0 of
  ``lsv_water`` keeps its previous value (zero in a fresh state), while
  ``lsu_water(1)`` is assigned.  Entry ``M+1`` is zeroed.
* ``k2[M+1]`` (Fortran ``k2(M+1)``) is not updated by the turbulence block
  (the loop stops at ``M``) and is returned unchanged, as in NumPy.
* ``row``/``Ri`` are only needed by ``Turbpar`` 3/4/7, but ``Ri`` is returned
  because the NumPy state carries it.
"""

from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp

from ..constants import C
from ..physics.blocktridiag import matrixprogonka_jax
from ..physics.density import (A11, A12, A13, A14, alpha1, alpha2, alpha3,
                               alpha4, alpha5, beta1, beta2, beta3, gamma1)

__all__ = ["momentum_solver", "jitted_momentum_solver"]


# ---------------------------------------------------------------------------
# Constants (copied from physics/momentum.py so both ports agree exactly)
# ---------------------------------------------------------------------------

def _s(x: float) -> float:
    """Default-real Fortran literal stored in a ``real(8)`` variable."""
    return float(np.float32(x))


_PI = C.pi
_G = C.g
_ROW0 = C.row0
_ROA0 = C.roa0
_KAPPA = C.kappa
_NIU_WAT = C.niu_wat
_Z0_BOT = C.z0_bot
_ROUGHNESS0 = C.roughness0
_CLIN_BOT = C.clin_bot
_CQUAD_BOT = C.cquad_bot
_SMALL = C.small_value
_MINWIND = C.minwind

_SQRT_G = math.sqrt(_G)
_LOG_10_R0 = math.log(10.0 / _ROUGHNESS0)

# TURB_CONST / numeric parameters of the closures (see physics/momentum.py).
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

_TWO_THIRDS = _s(2.0 / 3.0)
_SIXTH = _s(1.0 / 6.0)
_L0 = 0.1                      # thin-ice threshold, m
_MIN_VISC = 1.0e-10

_STATIC_ARGS = ("Turbpar", "stabfunc", "botfric", "relwind", "dyn_pgrad",
                "cuette", "PBLpar", "momflxpart", "a_veg", "c_veg",
                "horvisc", "eos", "lindens", "h_veg")


# ---------------------------------------------------------------------------
# Small helpers (JAX versions of the NumPy module-level helpers)
# ---------------------------------------------------------------------------

def _pad1(x):
    """0-based ``(..., n)`` -> 1-based ``(..., n+1)`` (leading zero)."""
    shape = x.shape[:-1] + (1,)
    return jnp.concatenate((jnp.zeros(shape, dtype=x.dtype), x), axis=-1)


def _diagblocks(v):
    """``(..., n)`` diagonal values -> ``(..., n, 2, 2)`` diagonal blocks."""
    z = jnp.zeros_like(v)
    return jnp.stack((jnp.stack((v, z), axis=-1),
                      jnp.stack((z, v), axis=-1)), axis=-2)


def _ddens_dtemp(temp, sal):
    """``SET_DDENS_DTEMP`` for ``eos=5`` (cf. ``physics/density.py``)."""
    sg = sal * 1.0e3
    out = (-(_ROW0 / A12) * (temp - A14) / ((temp + A13) ** 2)
           * ((temp + A13) * ((temp - A14) + 2.0 * (temp + A11))
              - (temp + A11) * (temp - A14)))
    a = (-alpha2 + 2.0 * alpha3 * temp - 3.0 * alpha4 * temp ** 2
         + 4.0 * alpha5 * temp ** 3)
    b = beta2 - 2.0 * beta3 * temp
    return out + a * sg + b * sg ** 1.5


def _ddens_dsal(temp, sal):
    """``SET_DDENS_DSAL`` for ``eos=5`` (cf. ``physics/density.py``)."""
    sg = sal * 1.0e3
    a = (alpha1 - alpha2 * temp + alpha3 * temp ** 2 - alpha4 * temp ** 3
         + alpha5 * temp ** 4)
    b = -beta1 + beta2 * temp - beta3 * temp ** 2
    c = gamma1
    return (a + 1.5 * b * sg ** 0.5 + 2.0 * c * sg) * 1.0e3


def _ce_canuto(lambda_m, lambda_n):
    """``CE_CANUTO`` (Canuto et al. 2001), clipped to [1e-20, 0.46]."""
    lambda_n1 = jnp.maximum(-5.9, lambda_n)
    ce = ((_CE_COEF01 + _CE_COEF02 * lambda_n1 - _CE_COEF03 * lambda_m)
          / (_CE_COEF11 + _CE_COEF12 * lambda_n1 + _CE_COEF13 * lambda_m
             + _CE_COEF14 * lambda_n1 * lambda_n1
             + _CE_COEF15 * lambda_n1 * lambda_m
             - _CE_COEF16 * lambda_m * lambda_m))
    return jnp.minimum(0.46, jnp.maximum(1.0e-20, ce))


def _smoment_galperin(lambda_n):
    """``SMOMENT_GALPERIN`` (only used by ``stabfunc=3``)."""
    gh_max = (_A2 * (12.0 * _A1 + _B1 + 3.0 * _B2)) ** (-1)
    gh_min = -0.28
    gh = -lambda_n
    gh = _CL_K_KL_MODEL ** 2 * 0.5 * gh
    gh = jnp.minimum(gh, gh_max)
    gh = jnp.maximum(gh, gh_min)
    num = (1.0 - 3.0 * _C1 - 6.0 * _A1 / _B1
           - 3.0 * _A2 * gh * ((_B2 - 3.0 * _A2) * (1.0 - 6.0 * _A1 / _B1)
                               - 3.0 * _C1 * (_B2 + 6.0 * _A1)))
    den = ((1.0 - 3.0 * _A2 * gh * (6.0 * _A1 + _B2))
           * (1.0 - 9.0 * _A1 * _A2 * gh))
    return jnp.maximum(1.0e-2, num / den * _A1)


def _dom_wave_speed(ufr, fetch):
    """``DOMWAVESPEED`` (momentum-flux partition of ``momflxpart``)."""
    c3 = (_s(0.051) / _s(0.96)) ** _TWO_THIRDS
    return c3 * (ufr * _G * fetch) ** _SIXTH


def _botfric_xx(par, u, v):
    """Lateral drag coefficient switch ``botfric`` (momentum_mod.f90:369)."""
    xx = jnp.where(par == 0, 0.0, _CLIN_BOT)          # par 0 / 1
    xx = jnp.where(par == 2, _CQUAD_BOT * jnp.sqrt(u * u + v * v), xx)
    return xx


def _first_layer(depth_area):
    """``depth_area(1,2)`` of the Fortran: first row, second column."""
    depth_area = jnp.asarray(depth_area)
    if depth_area.ndim == 3:                         # (L, ndata, 2)
        return depth_area[..., 0, 1]
    if depth_area.ndim == 2:                         # (L, 2)
        return depth_area[..., 1]
    raise ValueError("depth_area must have shape (L, ndata, 2) or (L, 2)")


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def momentum_solver(state, params, dt, *, Turbpar=2, stabfunc=2, botfric=1,
                    relwind=1, dyn_pgrad=0, cuette=0, PBLpar=3,
                    momflxpart=0.0, a_veg=0.0, c_veg=0.0, horvisc=0.0,
                    eos=5, lindens=0, h_veg=0.0):
    """Batched one-step LAKE 3.2 momentum solve; see the module docstring.

    All ``state``/``params`` arrays carry a leading lake axis.  Returns a dict
    with the new velocities (``u2``/``v2``), the updated turbulence fields and
    the diagnostics of :class:`lakematrix.physics.momentum.MomentumResult`.
    """
    # ---- configuration checks (Python level: static under jax.jit) ------
    if Turbpar != 2:
        raise NotImplementedError(
            f"jaxbatch momentum: only Turbpar=2 is ported (got {Turbpar})")
    if dyn_pgrad != 0:
        raise NotImplementedError(
            "jaxbatch momentum: dyn_pgrad != 0 (seiche / pressure-gradient "
            "force, DENSLAYERS/RELAYER) is not ported")
    if cuette not in (0, 11):
        raise NotImplementedError(
            f"jaxbatch momentum: cuette={cuette} is not ported (0/11 only)")
    if relwind not in (0, 1):
        raise ValueError(f"relwind={relwind}: incorrect meaning (STOP)")
    if stabfunc not in (2, 3):
        raise NotImplementedError(
            f"jaxbatch momentum: stabfunc={stabfunc} is not ported "
            "(2 = Canuto, 3 = Galperin)")
    if lindens != 0 or eos != 5:
        raise NotImplementedError(
            "jaxbatch momentum: only eos=5 / lindens=0 (McCutcheon) is ported")
    if h_veg > 0.0:
        raise NotImplementedError(
            "vegetation friction (h_veg > 0) is not operational in LAKE 3.2")
    # NOTE: `horvisc` is NOT checked: the NumPy reference sets
    # horvisc_ON = (dyn_pgrad == 4), so with dyn_pgrad=0 the horizontal
    # viscosity terms are exactly zero whatever the value of horvisc.

    f64 = jnp.float64
    sarr = lambda k: jnp.asarray(state[k], dtype=f64)
    parr = lambda k: jnp.asarray(params[k], dtype=f64)
    scal = lambda k: jnp.atleast_1d(jnp.asarray(state[k], dtype=f64))
    # static switch for the sloping-bottom drag law (Fortran line 249)
    xbot = float(botfric - 1)
    horvisc_on = 0
    # horizontal-viscosity terms are zero in case 1 (horvisc_ON = 0)
    xx3_1 = 0.0
    xx3_2 = 0.0

    # ---- 0-based inputs -> 1-based internal arrays (as in the NumPy port)
    u1 = _pad1(sarr("u1"))
    v1 = _pad1(sarr("v1"))
    E1 = _pad1(sarr("E1"))
    eps1 = _pad1(sarr("eps1"))
    k2 = _pad1(sarr("k2"))
    k2_old = _pad1(sarr("k2_old"))
    row = _pad1(sarr("row"))
    knum = _pad1(sarr("knum"))
    veg_sw = _pad1(sarr("veg_sw"))
    Tw1 = _pad1(sarr("Tw1"))
    Sal1 = _pad1(sarr("Sal1"))
    lsu = _pad1(sarr("lsu_water"))
    lsv = _pad1(sarr("lsv_water"))

    ddz = _pad1(parr("ddz"))                  # (L, M+1), 1-based 1..M
    ddz05 = _pad1(parr("ddz05"))              # (L, M+2), ddz05[i]=Fortran(i-1)
    dzeta_int = _pad1(parr("dzeta_int"))      # (L, M+2)
    area_int = _pad1(parr("area_int"))        # (L, M+2)
    area_half = _pad1(parr("area_half"))      # (L, M+1)
    Lx = _pad1(parr("Lx"))                    # (L, M+2)
    Ly = _pad1(parr("Ly"))                    # (L, M+2)

    M = int(area_half.shape[-1]) - 1
    if M < 1:
        raise ValueError("M must be at least 1")
    lead = u1.shape[:-1]

    h1 = scal("h1")
    dhw = scal("dhw")
    dhw0 = scal("dhw0")
    l1 = scal("l1")
    ls1 = scal("ls1")
    alphax = scal("alphax")
    alphay = scal("alphay")
    velfrict_ice_prev = (scal("velfrict_ice") if "velfrict_ice" in state
                         else jnp.zeros_like(h1))
    h1c = h1[..., None]
    dhwc = dhw[..., None]
    dhw0c = dhw0[..., None]

    kor = jnp.atleast_1d(parr("kor"))
    uwind = jnp.atleast_1d(parr("uwind"))
    vwind = jnp.atleast_1d(parr("vwind"))
    cdmw = jnp.atleast_1d(parr("cdmw"))
    zref = jnp.atleast_1d(parr("zref"))
    fetch = jnp.atleast_1d(parr("fetch"))

    # coefficient of the sloping-bottom drag law, used in the row coefficients
    xb_avg = 0.5 * (1.0 - xbot) + xbot

    # =====================================================================
    # lines 208-244: constants of the routine
    # =====================================================================
    AL = _G / _ROW0
    ACC2 = 1.0e-20
    ACCk = 1.0e-20

    # =====================================================================
    # lines 260-315: wind and surface stress (relwind=1)
    # =====================================================================
    u_mod = u1[..., 1]
    v_mod = v1[..., 1]
    if relwind == 1:
        urel = uwind - u_mod
        vrel = vwind - v_mod
    else:
        urel = jnp.broadcast_to(uwind, u_mod.shape)
        vrel = jnp.broadcast_to(vwind, v_mod.shape)

    # Fortran SIGN(minwind, x): x == 0 counts as positive
    urels = jnp.where(jnp.abs(urel) < _MINWIND,
                      jnp.where(urel < 0.0, -_MINWIND, _MINWIND), urel)
    vrels = jnp.where(jnp.abs(vrel) < _MINWIND,
                      jnp.where(vrel < 0.0, -_MINWIND, _MINWIND), vrel)

    kor2 = (0.5 * kor * dt)[..., None]
    wr = jnp.sqrt(urels ** 2 + vrels ** 2)
    tau_air = _ROA0 * cdmw * wr
    ufr = jnp.sqrt(tau_air / _ROW0)
    wind10 = wr * _LOG_10_R0 / jnp.log(zref / _ROUGHNESS0)
    C10 = cdmw * wr / (wind10 * wind10)
    tau_wav = (momflxpart * _ROA0 * C10
               * (wind10 - _dom_wave_speed(jnp.sqrt(tau_air / _ROA0),
                                           fetch) / _s(1.2)) ** 2)
    tau_sbl = jnp.maximum(0.0, tau_air - tau_wav)
    u_star = jnp.sqrt(tau_sbl / _ROW0)

    # =====================================================================
    # lines 320-363: Chezy coefficients, bottom and ice friction
    # =====================================================================
    rp = jnp.where(ls1 > 0.0, _s(0.01), _s(0.1))
    Cz_gr = jnp.where(l1 > 0.0,
                      _s(7.7) * _SQRT_G * (0.5 * h1 / rp) ** _SIXTH,
                      _s(7.7) * _SQRT_G * (h1 / rp) ** _SIXTH)
    Cz_i = jnp.where(l1 > 0.0,
                     _s(7.7) * _SQRT_G * (0.5 * h1 / _s(0.01)) ** _SIXTH,
                     jnp.zeros_like(h1))

    ddz_M = ddz[..., M]
    velx_log = 0.5 * (u1[..., M + 1] + u1[..., M])
    vely_log = 0.5 * (v1[..., M + 1] + v1[..., M])
    vel_bot = jnp.sqrt(velx_log ** 2 + vely_log ** 2)
    log_bot = jnp.log(h1 * 0.5 * ddz_M / _Z0_BOT)
    coef2 = vel_bot * _KAPPA ** 2 / log_bot ** 2
    velfrict_bot = vel_bot * _KAPPA / log_bot
    tau_gr = _ROW0 * velfrict_bot ** 2

    ddz_1 = ddz[..., 1]
    velx_log1 = 0.5 * (u1[..., 1] + u1[..., 2])
    vely_log1 = 0.5 * (v1[..., 1] + v1[..., 2])
    vel_ice = jnp.sqrt(velx_log1 ** 2 + vely_log1 ** 2)
    log_ice = jnp.log(h1 * 0.5 * ddz_1 / _Z0_BOT)
    coef1 = jnp.where(l1 > 0.0, -vel_ice * _KAPPA ** 2 / log_ice ** 2,
                      jnp.zeros_like(h1))
    velfrict_ice = jnp.where(l1 > 0.0, vel_ice * _KAPPA / log_ice,
                             velfrict_ice_prev)
    tau_i = jnp.where(l1 > 0.0, _ROW0 * velfrict_ice ** 2,
                      jnp.zeros_like(h1))

    # =====================================================================
    # lines 365-409: lateral boundary friction (lsu%water / lsv%water)
    # =====================================================================
    lateral = _first_layer(params["depth_area"]) >= 0.0     # (L,)
    # interior levels i = 2..M (1-based)
    xx_int = _botfric_xx(botfric, u1[..., 2:M + 1], v1[..., 2:M + 1])
    lat_int = (-1.0 / area_int[..., 2:M + 1]
               * (area_half[..., 2:M + 1] - area_half[..., 1:M])
               / (ddz05[..., 2:M + 1] * h1c) * xx_int)
    xx_one = _botfric_xx(botfric, u1[..., 1], v1[..., 1])
    lat_one = (-2.0 / area_int[..., 1]
               * (area_half[..., 1] - area_int[..., 1]) / (ddz[..., 1] * h1)
               * xx_one)
    rows = jnp.zeros(lead + (M + 2,), dtype=f64)
    rows = rows.at[..., 1].set(lat_one)
    rows = rows.at[..., 2:M + 1].set(lat_int)
    # lsv%water(1) is never assigned in the Fortran (line 405 bug): keep the
    # previous state value; lsv%water(M+1) is zeroed on line 408.
    rows_v = rows.at[..., 1].set(lsv[..., 1]).at[..., M + 1].set(0.0)
    rows = rows.at[..., M + 1].set(0.0)
    lsu = jnp.where(lateral[..., None], rows, lsu)
    lsv = jnp.where(lateral[..., None], rows_v, lsv)

    # =====================================================================
    # lines 412-420: |u| and the Richardson number
    # =====================================================================
    uv = jnp.sqrt(u1 ** 2 + v1 ** 2)
    dudz = ((u1[..., 2:M + 2] - u1[..., 1:M + 1]) ** 2
            + (v1[..., 2:M + 2] - v1[..., 1:M + 1]) ** 2) \
        / (ddz[..., 1:M + 1] * h1c)
    dudz = jnp.maximum(dudz, _SMALL)
    Ri = jnp.zeros(lead + (M + 1,), dtype=f64)
    Ri = Ri.at[..., 1:M + 1].set(
        _G / _ROW0 * (row[..., 2:M + 2] - row[..., 1:M + 1]) / dudz)

    # =====================================================================
    # lines 422-506: eddy viscosity, Turbpar=2 with stabfunc 2/3
    # =====================================================================
    E1 = jnp.where(E1 <= 0.0, _s(10.0 ** -16), E1)
    eps1 = jnp.where(eps1 <= 0.0, _s(10.0 ** -18), eps1)

    ee = E1[..., 1:M + 1]
    ep = eps1[..., 1:M + 1]
    fac = ee * ee / ((ep + ACC2) * (ep + ACC2))
    lambda_N = (fac
                * (_ddens_dtemp(Tw1[..., 1:M + 1], Sal1[..., 1:M + 1])
                   * (Tw1[..., 2:M + 2] - Tw1[..., 1:M + 1])
                   + _ddens_dsal(Tw1[..., 1:M + 1], Sal1[..., 1:M + 1])
                   * (Sal1[..., 2:M + 2] - Sal1[..., 1:M + 1]))
                / (h1c * ddz[..., 1:M + 1]) * AL)
    lambda_M = (fac
                * ((u1[..., 2:M + 2] - u1[..., 1:M + 1]) ** 2
                   + (v1[..., 2:M + 2] - v1[..., 1:M + 1]) ** 2)
                / (h1c * h1c * ddz[..., 1:M + 1] * ddz[..., 1:M + 1]))
    if stabfunc == 2:
        CE = _ce_canuto(lambda_M, lambda_N)
    else:
        CE = math.sqrt(2.0) * _CL_K_KL_MODEL * _smoment_galperin(lambda_N)
    k2 = k2.at[..., 1:M + 1].set(
        jnp.maximum(CE * ee ** 2 / (ep + ACCk), _MIN_VISC)
        + _NIU_WAT + knum[..., 1:M + 1])

    # lines 508-516: k2_old = k2 (levels 1..M; k2_old has no level M+1)
    k2_old = k2_old.at[..., 1:M + 1].set(k2[..., 1:M + 1])

    Force_x = jnp.zeros(lead + (M + 2,), dtype=f64)
    Force_y = jnp.zeros(lead + (M + 2,), dtype=f64)
    gx = dt * _G * jnp.tan(_PI * alphax / 180.0)
    gy = dt * _G * jnp.tan(_PI * alphay / 180.0)
    gxc = gx[..., None]
    gyc = gy[..., None]

    # =====================================================================
    # lines 545-1272: top boundary row (1-based index 1)
    # =====================================================================
    if cuette in (0, 11):
        k2 = k2.at[..., 1].set(area_half[..., 1] / area_int[..., 1]
                               * k2[..., 1])
        k2_old = k2_old.at[..., 1].set(area_half[..., 1] / area_int[..., 1]
                                       * k2_old[..., 1])

        k2_1 = k2[..., 1]
        xx_old_t = 0.5 * (2.0 - dhw0 * h1 * ddz_1 / (k2_1 * dt))
        xx_t = 0.5 * (2.0 * k2_old[..., 1] / k2_1
                      - dhw0 * h1 * ddz_1 / (k2_1 * dt))
        yy_t = (ddz_1 * h1) ** 2 / (k2_1 * dt)

        # open water (l1 == 0, cuette != 11, PBLpar != 0)
        taux_t = urels / wr * tau_sbl
        tauy_t = vrels / wr * tau_sbl

        k2ddz_t2 = (h1 * ddz_1) ** 2
        veg_t = (0.5 * yy_t * dt * a_veg * c_veg
                 * jnp.sqrt(u1[..., 1] ** 2 + v1[..., 1] ** 2)
                 * veg_sw[..., 1])
        cor_t = kor * k2ddz_t2 / (2.0 * k2_1)     # cm(1:2) off-diagonal

        ow_cm = xx_t + yy_t * (1.0 + xx3_1) + veg_t
        ow_cm00 = ow_cm + yy_t * dt * lsu[..., 1] * xb_avg
        ow_cm11 = ow_cm + yy_t * dt * lsv[..., 1] * xb_avg
        ow_bm = xx_t
        ow_dm0 = ((2.0 * taux_t * ddz_1 * h1 / (_ROW0 * k2_1)
                   + yy_t * (1.0 - xx3_1) * u1[..., 1])
                  + (u1[..., 2] - u1[..., 1]) * xx_old_t
                  + cor_t * v1[..., 1]
                  + yy_t * gx + yy_t * dt * Force_x[..., 1]
                  - 0.5 * (1.0 - xbot) * lsu[..., 1] * u1[..., 1] * yy_t * dt)
        ow_dm1 = ((2.0 * tauy_t * ddz_1 * h1 / (_ROW0 * k2_1)
                   + yy_t * (1.0 - xx3_2) * v1[..., 1])
                  + (v1[..., 2] - v1[..., 1]) * xx_old_t
                  - cor_t * u1[..., 1]
                  + yy_t * gy + yy_t * dt * Force_y[..., 1]
                  - 0.5 * (1.0 - xbot) * lsv[..., 1] * v1[..., 1] * yy_t * dt)

        # thin ice (0 < l1 <= L0)
        b0 = jnp.where((l1 > 0.0) & (l1 <= _L0), l1 / _L0,
                       jnp.zeros_like(h1))
        fric_b0 = 0.5 * 2.0 * b0 * coef1 * ddz_1 * h1 / k2_1
        ti_cm = xx_t + yy_t - fric_b0 + veg_t
        ti_cm00 = ti_cm + yy_t * dt * lsu[..., 1] * xb_avg
        ti_cm11 = ti_cm + yy_t * dt * lsv[..., 1] * xb_avg
        ti_bm = xx_t + fric_b0
        ti_dm0 = (yy_t * u1[..., 1]
                  + (u1[..., 2] - u1[..., 1]) * xx_old_t
                  + 2.0 * taux_t * ddz_1 * h1 / (_ROW0 * k2_1) * (1.0 - b0)
                  + cor_t * v1[..., 1]
                  + yy_t * gx + yy_t * dt * Force_x[..., 1]
                  - 0.5 * (1.0 - xbot) * lsu[..., 1] * u1[..., 1] * yy_t * dt)
        ti_dm1 = (yy_t * v1[..., 1]
                  + (v1[..., 2] - v1[..., 1]) * xx_old_t
                  + 2.0 * tauy_t * ddz_1 * h1 / (_ROW0 * k2_1) * (1.0 - b0)
                  - cor_t * u1[..., 1]
                  + yy_t * gy + yy_t * dt * Force_y[..., 1]
                  - 0.5 * (1.0 - xbot) * lsv[..., 1] * v1[..., 1] * yy_t * dt)

        # thick ice (l1 > L0)
        fric_1 = 0.5 * 2.0 * coef1 * ddz_1 * h1 / k2_1
        th_cm = xx_t + yy_t - fric_1 + veg_t
        th_cm00 = th_cm + yy_t * dt * lsu[..., 1] * xb_avg
        th_cm11 = th_cm + yy_t * dt * lsv[..., 1] * xb_avg
        th_bm = xx_t + fric_1
        th_dm0 = (yy_t * u1[..., 1]
                  + (u1[..., 2] - u1[..., 1]) * xx_old_t
                  + cor_t * v1[..., 1]
                  + yy_t * gx + yy_t * dt * Force_x[..., 1]
                  - 0.5 * (1.0 - xbot) * lsu[..., 1] * u1[..., 1] * yy_t * dt)
        th_dm1 = (yy_t * v1[..., 1]
                  + (v1[..., 2] - v1[..., 1]) * xx_old_t
                  - cor_t * u1[..., 1]
                  + yy_t * gy + yy_t * dt * Force_y[..., 1]
                  - 0.5 * (1.0 - xbot) * lsv[..., 1] * v1[..., 1] * yy_t * dt)

        m_open = l1 == 0.0
        m_thin = (l1 > 0.0) & (l1 <= _L0)

        def sel(open_, thin_, thick_):
            return jnp.where(m_open, open_, jnp.where(m_thin, thin_, thick_))

        cm_t00 = sel(ow_cm00, ti_cm00, th_cm00)
        cm_t11 = sel(ow_cm11, ti_cm11, th_cm11)
        bm_t = sel(ow_bm, ti_bm, th_bm)
        dm_t0 = sel(ow_dm0, ti_dm0, th_dm0)
        dm_t1 = sel(ow_dm1, ti_dm1, th_dm1)
        cm_t01 = -cor_t
        cm_t10 = cor_t

        # lines 1270-1271: undo the surface scaling for the interior rows
        k2 = k2.at[..., 1].set(area_int[..., 1] / area_half[..., 1]
                               * k2[..., 1])
        k2_old = k2_old.at[..., 1].set(area_int[..., 1] / area_half[..., 1]
                                       * k2_old[..., 1])

    else:                                   # cuette == 1 (unreachable)
        raise NotImplementedError("jaxbatch momentum: cuette=1 is not ported")

    # =====================================================================
    # lines 1274-1321: bottom boundary row (1-based index M+1)
    # =====================================================================
    k2 = k2.at[..., M].set(area_half[..., M] / area_int[..., M + 1]
                           * k2[..., M])
    k2_old = k2_old.at[..., M].set(area_half[..., M] / area_int[..., M + 1]
                                   * k2_old[..., M])
    k2_M = k2[..., M]
    xx_old_b = 0.5 * (2.0 - ddz_M * h1 * (dhw - dhw0) / (k2_M * dt))
    xx_b = 0.5 * (2.0 * k2_old[..., M] / k2_M
                  - ddz_M * h1 * (dhw - dhw0) / (k2_M * dt))
    yy_b = (ddz_M * h1) ** 2 / (k2_M * dt)

    k2ddz_b2 = (h1 * ddz_M) ** 2
    fric_b = 0.5 * 2.0 * coef2 * ddz_M * h1 / k2_M
    veg_b = (0.5 * yy_b * dt * a_veg * c_veg
             * jnp.sqrt(u1[..., M + 1] ** 2 + v1[..., M + 1] ** 2)
             * veg_sw[..., M + 1])
    cm_b00 = xx_b + yy_b * (1.0 + xx3_1) + fric_b + veg_b
    cm_b11 = xx_b + yy_b * (1.0 + xx3_2) + fric_b + veg_b
    am_b = xx_b - fric_b
    cor_b = kor * k2ddz_b2 / (2.0 * k2_M)
    cm_b01 = -cor_b
    cm_b10 = cor_b
    dm_b0 = (yy_b * (1.0 - xx3_1) * u1[..., M + 1]
             - (u1[..., M + 1] - u1[..., M]) * xx_old_b
             + cor_b * v1[..., M + 1]
             + yy_b * gx + yy_b * dt * Force_x[..., M + 1])
    dm_b1 = (yy_b * (1.0 - xx3_2) * v1[..., M + 1]
             - (v1[..., M + 1] - v1[..., M]) * xx_old_b
             - cor_b * u1[..., M + 1]
             + yy_b * gy + yy_b * dt * Force_y[..., M + 1])

    k2 = k2.at[..., M].set(area_int[..., M + 1] / area_half[..., M]
                           * k2[..., M])
    k2_old = k2_old.at[..., M].set(area_int[..., M + 1] / area_half[..., M]
                                   * k2_old[..., M])

    # =====================================================================
    # lines 1328-1381: interior rows k = 2..M
    # =====================================================================
    kk = slice(2, M + 1)
    km = slice(1, M)
    kn = slice(3, M + 2)

    ddz_k = ddz[..., kk]
    ddz_km = ddz[..., km]
    den_h = ddz_k + ddz_km

    am_int = (dzeta_int[..., kk] * dhwc / (2.0 * den_h * h1c)
              - dhw0c / (2.0 * den_h * h1c)
              - area_half[..., km] / area_int[..., kk] * k2_old[..., km] * dt
              / (h1c ** 2 * den_h * ddz_km))
    bm_int = -(dzeta_int[..., kk] * dhwc / (2.0 * den_h * h1c)
               + dhw0c / (2.0 * den_h * h1c)
               + area_half[..., kk] / area_int[..., kk] * k2_old[..., kk] * dt
               / (h1c ** 2 * den_h * ddz_k))

    xx_int = (-(area_half[..., kk] / area_int[..., kk] * k2_old[..., kk]
                / ddz_k
                + area_half[..., km] / area_int[..., kk] * k2_old[..., km]
                / ddz_km)
              * dt / (h1c ** 2 * den_h)
              - 0.5 * dt * c_veg * a_veg
              * jnp.sqrt(u1[..., kk] ** 2 + v1[..., kk] ** 2)
              * veg_sw[..., kk])

    cm_00 = xx_int - 1.0 * (1.0 + xx3_1)
    cm_11 = xx_int - 1.0 * (1.0 + xx3_2)
    cm_00 = cm_00 - dt * 0.5 * lsu[..., kk] * xb_avg
    cm_11 = cm_11 - dt * 0.5 * lsv[..., kk] * xb_avg

    den_k = h1c ** 2 * den_h
    dm_0 = (-u1[..., kk] * (1.0 - xx3_1)
            - kor2 * v1[..., kk]
            - gxc
            - Force_x[..., kk] * dt
            + 0.5 * (1.0 - xbot) * lsu[..., kk] * u1[..., kk] * dt
            - dt * (den_k * ddz_k) ** (-1)
            * (area_half[..., kk] / area_int[..., kk] * k2[..., kk]
               * (u1[..., kn] - u1[..., kk]))
            + dt * (den_k * ddz_km) ** (-1)
            * (area_half[..., km] / area_int[..., kk] * k2[..., km]
               * (u1[..., kk] - u1[..., km]))
            - dzeta_int[..., kk] * dhwc
            * (2.0 * den_h * h1c) ** (-1)
            * (u1[..., kn] - u1[..., km])
            + dhw0c * (2.0 * den_h * h1c) ** (-1)
            * (u1[..., kn] - u1[..., km]))
    dm_1 = (-v1[..., kk] * (1.0 - xx3_2)
            + kor2 * u1[..., kk]
            - gyc
            - Force_y[..., kk] * dt
            + 0.5 * (1.0 - xbot) * lsv[..., kk] * v1[..., kk] * dt
            - dt * (den_k * ddz_k) ** (-1)
            * (area_half[..., kk] / area_int[..., kk] * k2[..., kk]
               * (v1[..., kn] - v1[..., kk]))
            + dt * (den_k * ddz_km) ** (-1)
            * (area_half[..., km] / area_int[..., kk] * k2[..., km]
               * (v1[..., kk] - v1[..., km]))
            - dzeta_int[..., kk] * dhwc
            * (2.0 * den_h * h1c) ** (-1)
            * (v1[..., kn] - v1[..., km])
            + dhw0c * (2.0 * den_h * h1c) ** (-1)
            * (v1[..., kn] - v1[..., km]))

    # =====================================================================
    # lines 1393-1418: assemble, solve, write the outputs
    # =====================================================================
    am = jnp.zeros(lead + (M + 2, 2, 2), dtype=f64)
    bm = jnp.zeros(lead + (M + 2, 2, 2), dtype=f64)
    cm = jnp.zeros(lead + (M + 2, 2, 2), dtype=f64)
    dm = jnp.zeros(lead + (M + 2, 2), dtype=f64)

    # interior rows
    am = am.at[..., kk, :, :].set(_diagblocks(am_int))
    bm = bm.at[..., kk, :, :].set(_diagblocks(bm_int))
    cm = cm.at[..., kk, 0, 0].set(cm_00)
    cm = cm.at[..., kk, 1, 1].set(cm_11)
    cm = cm.at[..., kk, 0, 1].set(kor2)
    cm = cm.at[..., kk, 1, 0].set(-kor2)
    dm = dm.at[..., kk, 0].set(dm_0)
    dm = dm.at[..., kk, 1].set(dm_1)
    # top row (1)
    bm = bm.at[..., 1, 0, 0].set(bm_t)
    bm = bm.at[..., 1, 1, 1].set(bm_t)
    cm = cm.at[..., 1, 0, 0].set(cm_t00)
    cm = cm.at[..., 1, 1, 1].set(cm_t11)
    cm = cm.at[..., 1, 0, 1].set(cm_t01)
    cm = cm.at[..., 1, 1, 0].set(cm_t10)
    dm = dm.at[..., 1, 0].set(dm_t0)
    dm = dm.at[..., 1, 1].set(dm_t1)
    # bottom row (M+1)
    am = am.at[..., M + 1, 0, 0].set(am_b)
    am = am.at[..., M + 1, 1, 1].set(am_b)
    cm = cm.at[..., M + 1, 0, 0].set(cm_b00)
    cm = cm.at[..., M + 1, 1, 1].set(cm_b11)
    cm = cm.at[..., M + 1, 0, 1].set(cm_b01)
    cm = cm.at[..., M + 1, 1, 0].set(cm_b10)
    dm = dm.at[..., M + 1, 0].set(dm_b0)
    dm = dm.at[..., M + 1, 1].set(dm_b1)

    # MATRIXPROGONKA on rows 1..M+1:
    #   -am(k) y(k-1) + cm(k) y(k) - bm(k) y(k+1) = dm(k)
    ym = matrixprogonka_jax(am[..., 1:M + 2, :, :], bm[..., 1:M + 2, :, :],
                            cm[..., 1:M + 2, :, :], dm[..., 1:M + 2, :])
    u2 = ym[..., 0]
    v2 = ym[..., 1]

    windwork = taux_t * u2[..., 0] + tauy_t * v2[..., 0]
    discharge_x = jnp.sum(h1c * ddz05[..., 1:M + 2] * u1[..., 1:M + 2]
                          * Ly[..., 1:M + 2], axis=-1)
    discharge_y = jnp.sum(h1c * ddz05[..., 1:M + 2] * v1[..., 1:M + 2]
                          * Lx[..., 1:M + 2], axis=-1)
    discharge = jnp.stack((discharge_x, discharge_y), axis=-1)

    return dict(
        u2=u2, v2=v2,
        k2=k2[..., 1:], k2_old=k2_old[..., 1:],
        E1=E1[..., 1:], eps1=eps1[..., 1:],
        lsu_water=lsu[..., 1:], lsv_water=lsv[..., 1:],
        uv=uv[..., 1:], Ri=Ri[..., 1:],
        knum=knum[..., 1:], veg_sw=veg_sw[..., 1:],
        taux=taux_t, tauy=tauy_t, tau_air=tau_air, tau_gr=tau_gr,
        tau_i=tau_i, tau_wav=tau_wav, tau_sbl=tau_sbl, u_star=u_star,
        velfrict_bot=velfrict_bot, velfrict_ice=velfrict_ice,
        wind10=wind10, C10=C10, ufr=ufr, coef1=coef1, coef2=coef2,
        Cz_gr=Cz_gr, Cz_i=Cz_i, rp=rp, b0=b0, xbot=xbot,
        u=u_mod, v=v_mod, urel=urel, vrel=vrel, urels=urels, vrels=vrels,
        wr=wr, windwork=windwork, discharge=discharge,
        Force_x=Force_x[..., 1:], Force_y=Force_y[..., 1:],
        horvisc_on=horvisc_on,
    )


jitted_momentum_solver = jax.jit(momentum_solver, static_argnames=_STATIC_ARGS)
