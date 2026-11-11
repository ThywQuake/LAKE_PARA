"""Batched JAX port of ``K_EPSILON`` (case 1 / testlake path).

Mirrors :func:`lakematrix.physics.kepsilon.k_epsilon` with a **leading lake
axis** ``(L, ...)`` so that the k-epsilon closure can run for a whole batch of
lakes in one JIT-compiled call.  The NumPy implementation stays the reference;
every physics constant is imported from it (never re-typed).

Scope -- the only configuration implemented; everything else raises
``NotImplementedError``
--------------------------------------------------------------------------
``Turbpar = 2``, ``kepsbc = 1`` (law-of-the-wall Neumann boundary condition),
``stabfunc = 2`` (Canuto et al. 2001), ``semi_implicit_scheme = True``
(exactly one pass of the inner ``iterat_keps`` loop), ``l1 = 0`` (open water,
no ice/snow), ``h_veg = 0`` (no vegetation), ``contrgrad = False``,
``smooth = False``, ``grav_wave_Gill = 0``, ``seiches_Goudsmit = 0``,
``keps_coef = 1``, ``zserout = missing_value``, ``eos = 5``, ``lindens = 0``.

As in the reference, the two ``keps_mode`` passes (``time_deriv = 0`` then
``time_deriv = 1``) are run for every lake whose ``init_keps`` is false; a lake
that is already initialised gets the single evolution pass.  The per-lake flags
are combined with ``jnp.where``, so both cases can be mixed in one batch.

Input: ``state`` dict
---------------------
A value may be a Python scalar, a 1-D array (the same state for every lake) or
a batched ``(L, ...)`` array.  Arrays are **1-based** exactly like the
Fortran/NumPy state: index 0 is unused (``ddz05`` keeps the Fortran bound
0:M).  ``M`` comes from ``params.M``; ``L`` is the leading length of ``e1``.
Missing optional keys default to zero.

===============================  ===================  ===================================
key                              shape                meaning
===============================  ===================  ===================================
``e1``, ``eps1``                  ``(L, M+2)``        TKE (m2/s2) / dissipation (m2/s3)
``k2``, ``k2t``, ``k5``            ``(L, M+2)``        diffusivities (inputs ignored)
``k2_mid``, ``k3_mid``, ``k5_mid`` ``(L, M+2)``        half-level diffusivities (ignored)
``l``, ``tf``, ``kt``              ``(L, M+2)``        length scale, 1/time scale, heat diff.
``gamt``, ``gamu``, ``gamv``       ``(L, M+2)``        counter-gradient terms (zeroed here)
``row``, ``row2``                  ``(L, M+2)``        density (``row`` feeds ``Ri_bulk``)
``u1``, ``v1``, ``u2``, ``v2``     ``(L, M+2)``        velocities, old/new level, m/s
``um``, ``vm``                     ``(L, M+2)``        time-interpolated velocity (overwritten)
``tw1``, ``tw2``                   ``(L, M+2)``        water temperature, degC
``sal1``, ``sal2``                 ``(L, M+2)``        salinity, kg/kg
``knum``, ``veg_sw``               ``(L, M+2)``        module arrays (ignored, echoed)
``lamw``, ``lamsal``               ``(L, M+1)``        molecular conductivities (ignored)
``z_half``                         ``(L, M+1)``        level depths (only read by ``zserout``)
``ddz``                            ``(L, M+1)``        layer thickness 1:M
``ddz2``                           ``(L, M+2)``        ``ddz(i)+ddz(i+1)``; derived if absent
``ddz054``                         ``(L, M+2)``        ``2*(ddz(i-2)/2+ddz(i-1)+ddz(i)/2)``
``ddz05``                          ``(L, M+1)``        ``Grid.ddz05`` (Fortran bound 0:M)
``dzeta_int``                      ``(L, M+2)``        cumulative depth fraction 1:M+1
``dzeta_05int``                    ``(L, M+1)``        half-level depth fraction 1:M
``area_int``                       ``(L, M+2)``        layer areas, m2
``area_half``                      ``(L, M+1)``        half-level areas, m2
``h1``                             ``(L,)``            water depth, m
``dhw``, ``dhw0``                  ``(L,)``            water-depth increments, m
``vol``, ``botar``                 ``(L,)``            volume / bottom area (unused)
``b0``                             ``(L,)``            ice concentration (unused)
``tau_air``, ``tau_wav``           ``(L,)``            wind / wave stress, N/m2
``tau_i``, ``tau_gr``              ``(L,)``            ice-water / bottom stress, N/m2
``uwind``, ``vwind``               ``(L,)``            wind components, m/s
``fetch``, ``roughness``           ``(L,)``            fetch (m) / roughness length
``l1``                             ``(L,)`` or float   ice thickness; must be 0
``h_veg``                          ``(L,)`` or float   vegetation height; must be 0
``init_keps``                      ``(L,)`` or bool    ``KEpsilonState.init_keps``
===============================  ===================  ===================================

Required are the keys the equations read: ``e1``, ``eps1``, ``tw1``, ``tw2``,
``sal1``, ``sal2``, ``u1``..``v2``, ``row``, ``h1``, ``tau_air``, ``tau_gr``,
``dhw``, ``dhw0`` and the geometry ``ddz``, ``ddz05``, ``dzeta_int``,
``dzeta_05int``, ``area_int``, ``area_half``.  Everything else in the table is
optional; ``tau_wav``/``uwind``/``vwind`` only feed the ``maxTKEinput``
diagnostic for ``kepsbc = 1``.

Output
------
``dict`` with every input key (echoed where untouched) plus the advanced
prognostics ``e1``, ``eps1``, ``e2``, ``eps2`` and the arrays/scalars the
reference state carries.  ``k2`` is the *output* eddy viscosity (already
multiplied by ``ce``) and ``kt`` is the heat diffusivity in m2/s, exactly as
returned by the NumPy routine.  The 1:M arrays ``c_eps1``..``c_eps3``, ``s``,
``gen``, ``f`` and ``tke_turb_trans`` keep the reference bound, i.e. length
``M+1`` (index 0 unused); the 1:M+1 arrays are ``(L, M+2)``.

Deviations from the NumPy reference (none of them affect case 1)
----------------------------------------------------------------
* ``wu``/``wv`` (read only by the ``contrgrad`` branch) are not computed.
* ``Buoyancy1``/``dBdz0`` (read only by ``kepsbc = 4``) are not computed.
* ``firstcall`` bookkeeping of ``veg_sw``/``knum`` is left to the caller; the
  Fortran ``STOP`` for ``h_veg > 0`` becomes a scope error.
* ``st.k3_mid(1)``/``st.k5_mid(1)`` are never assigned by the reference (they
  carry over between calls); here they are 0, as on the first call of a run.
* ``dt``/``dt_keps``/``nstep_keps`` are Python scalars (static arguments of
  :func:`make_kernel`).  ``nstep_keps`` only scales ``dhw_keps`` here, because
  in ``lake.f90`` it is the *caller* that invokes ``K_EPSILON`` that many
  times.
* The integral budgets are reduced with a masked ``jnp.sum`` instead of the
  Fortran sequential loop, so their last bits can differ from NumPy
  (<= 1e-13 absolute over the checked reference cases).
* ``trace``/``trace_coef`` (Python lists), the ``nl``/``iter`` counters and
  the ``results/debug`` file I/O are not reproduced.
* ``E_min``/``eps_min`` are the ``l1 == 0`` branch only (``E_MIN_LAMW0`` /
  ``EPS_MIN0``); the ``l1 > 0`` Racr formulas are not ported.
"""

from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp

from ..constants import C
from ..physics import density as _dens
from ..physics.kepsilon import (
    ACC,
    ACC2,
    ACCK,
    CE0,
    CEcoef01,
    CEcoef02,
    CEcoef03,
    CEcoef11,
    CEcoef12,
    CEcoef13,
    CEcoef14,
    CEcoef15,
    CEcoef16,
    CEtcoef01,
    CEtcoef02,
    CEtcoef03,
    E_MIN_LAMW0,
    EPS_MIN0,
    KEpsilonParams,
    Z0,
    ceps1,
    ceps2,
    ceps3_stable,
    ceps3_unstable,
    kar,
    min_diff,
    min_visc,
    sigmaE,
    sigmaeps,
)
from ..physics.tridiag import progonka_jax

__all__ = ["k_epsilon", "make_kernel"]

#: SAVE local ``ALFT = 1.d0`` (turb_mod.f90:279)
ALFT = 1.0
#: SAVE local ``AL = g/row0`` (turb_mod.f90:278)
AL = C.g / C.row0
NIU_WAT = C.niu_wat
#: ``DISSIPATION_FLUX_SHEAR``: ``small_number = 1.d-20``
SMALL_NUMBER = 1.0e-20


# ---------------------------------------------------------------------------
# helpers, vectorised one-for-one from the NumPy reference
# ---------------------------------------------------------------------------
def _ce_canuto(lambda_m, lambda_n):
    """``CE_CANUTO`` (turb_mod.f90:1181-1224), vectorised."""
    upper_limit = 0.46
    small_number = 1.0e-20
    lambda_n1 = jnp.maximum(-5.9, lambda_n)
    val = ((CEcoef01 + CEcoef02 * lambda_n1 - CEcoef03 * lambda_m)
           / (CEcoef11 + CEcoef12 * lambda_n1 + CEcoef13 * lambda_m
              + CEcoef14 * lambda_n1 * lambda_n1
              + CEcoef15 * lambda_n1 * lambda_m
              - CEcoef16 * lambda_m * lambda_m))
    val = jnp.maximum(small_number, val)
    return jnp.minimum(upper_limit, val)


def _cet_canuto(lambda_m, lambda_n):
    """``CEt_CANUTO`` (turb_mod.f90:1228-1269), vectorised."""
    upper_limit = 0.61
    small_number = 1.0e-20
    lambda_n1 = jnp.maximum(-5.9, lambda_n)
    val = ((CEtcoef01 + CEtcoef02 * lambda_n1 + CEtcoef03 * lambda_m)
           / (CEcoef11 + CEcoef12 * lambda_n1 + CEcoef13 * lambda_m
              + CEcoef14 * lambda_n1 ** 2
              + CEcoef15 * lambda_n1 * lambda_m
              - CEcoef16 * lambda_m ** 2))
    val = jnp.maximum(small_number, val)
    return jnp.minimum(upper_limit, val)


def _ddens_dtemp(temp, sal):
    """``DDENS_DTEMP`` for eos 5 (phys_func.f90:325-360), vectorised."""
    sg = sal * 1.0e3
    out = (-(C.row0 / _dens.A12) * (temp - _dens.A14) / ((temp + _dens.A13) ** 2)
           * ((temp + _dens.A13) * ((temp - _dens.A14) + 2.0 * (temp + _dens.A11))
              - (temp + _dens.A11) * (temp - _dens.A14)))
    a = (-_dens.alpha2 + 2.0 * _dens.alpha3 * temp
         - 3.0 * _dens.alpha4 * temp ** 2 + 4.0 * _dens.alpha5 * temp ** 3)
    b = _dens.beta2 - 2.0 * _dens.beta3 * temp
    return out + a * sg + b * sg ** 1.5


def _ddens_dsal(temp, sal):
    """``DDENS_DSAL`` for eos 5 (phys_func.f90:363-388), vectorised."""
    sg = sal * 1.0e3
    a = (_dens.alpha1 - _dens.alpha2 * temp + _dens.alpha3 * temp ** 2
         - _dens.alpha4 * temp ** 3 + _dens.alpha5 * temp ** 4)
    b = -_dens.beta1 + _dens.beta2 * temp - _dens.beta3 * temp ** 2
    c = _dens.gamma1
    return (a + 1.5 * b * sg ** 0.5 + 2.0 * c * sg) * 1.0e3


def _interp(fim05, fip05, ddzim1, ddzi):
    """``INTERPOLATE_TO_INTEGER_POINT`` (turb_mod.f90:1444-1461), vectorised."""
    return fim05 + (fip05 - fim05) * ddzim1 / (ddzim1 + ddzi)


def _fsign1(x):
    """Fortran ``sign(1._ireals, x)``: +1 for ``x >= 0`` (including 0)."""
    return jnp.where(x >= 0.0, 1.0, -1.0)


def _concrete_all(x, value):
    """``bool(all(x == value))`` for concrete values, ``None`` under ``jit``.

    Used for the ``l1 == 0`` / ``h_veg == 0`` scope checks, which must not
    break tracing inside :func:`make_kernel`.
    """
    try:
        return bool(jnp.all(jnp.asarray(x) == value))
    except Exception:                      # TracerBoolConversionError
        return None


def _sh(x, k):
    """``x[:, i+k]`` on the 1-based padded index ``i``, zero outside the array.

    The padded arrays have length ``M+2`` and index ``j`` *is* Fortran index
    ``j`` for ``j = 0..M+1``; index 0 is unused and ``M+1`` is the extra
    Fortran bound (zero in the reference).
    """
    n = x.shape[-1]
    idx = jnp.arange(n) + k
    ok = (idx >= 0) & (idx < n)
    return jnp.where(ok[None, :], x[:, jnp.clip(idx, 0, n - 1)], 0.0)


# ---------------------------------------------------------------------------
# parameter handling / scope checks
# ---------------------------------------------------------------------------
def _as_params(params):
    """Return a :class:`KEpsilonParams` from ``None``/mapping/dataclass/object."""
    if params is None:
        return KEpsilonParams()
    if isinstance(params, KEpsilonParams):
        return params
    if isinstance(params, Mapping):
        return KEpsilonParams(**dict(params))
    fields = KEpsilonParams.__dataclass_fields__
    return KEpsilonParams(**{k: getattr(params, k) for k in fields
                             if hasattr(params, k)})


def _check_scope(p):
    """Reject every switch outside the ported case-1 path."""
    if int(p.Turbpar) != 2:
        raise NotImplementedError(
            f"K_EPSILON is only called for Turbpar == 2 (got {p.Turbpar})")
    if int(p.kepsbc) != 1:
        raise NotImplementedError(
            f"jaxbatch.k_epsilon implements kepsbc = 1 only (got {p.kepsbc})")
    if int(p.stabfunc) != 2:
        raise NotImplementedError(
            "jaxbatch.k_epsilon implements stabfunc = 2 (Canuto) only "
            f"(got {p.stabfunc})")
    if not bool(p.semi_implicit_scheme):
        raise NotImplementedError(
            "jaxbatch.k_epsilon implements semi_implicit_scheme = True only")
    for name in ("smooth", "perdam", "contrgrad"):
        if bool(getattr(p, name)):
            raise NotImplementedError(
                f"jaxbatch.k_epsilon does not implement {name} = True")
    if int(p.keps_coef) != 1:
        raise NotImplementedError(
            "jaxbatch.k_epsilon implements keps_coef = 1 only")
    if int(p.grav_wave_Gill) != 0:
        raise NotImplementedError(
            "jaxbatch.k_epsilon implements grav_wave_Gill = 0 only")
    if int(p.seiches_Goudsmit) != 0:
        raise NotImplementedError(
            "jaxbatch.k_epsilon implements seiches_Goudsmit = 0 only")
    if float(p.zserout) != float(p.missing_value):
        raise NotImplementedError(
            "jaxbatch.k_epsilon does not implement the zserout TKE budget")
    if int(p.eos) != 5 or int(p.lindens) != 0:
        raise NotImplementedError(
            "SET_DDENS_DTEMP/SET_DDENS_DSAL are ported for eos 5, "
            f"lindens 0 only (got eos={p.eos}, lindens={p.lindens})")


# ---------------------------------------------------------------------------
# state preparation
# ---------------------------------------------------------------------------
#: ``1:M+1`` arrays of ``KEpsilonState`` (M+1 values); padded to M+2 so that
#: the padded index ``j`` *is* the Fortran index ``j`` (index 0 unused).
_ARR_M2 = (
    "e1", "eps1", "e2", "eps2", "k2", "k2t", "k5", "k2_mid", "k3_mid",
    "k5_mid", "l", "tf", "kt", "gamt", "gamu", "gamv", "row", "row2",
    "knum", "veg_sw", "u1", "v1", "u2", "v2", "um", "vm",
    "tw1", "tw2", "sal1", "sal2", "dzeta_int", "area_int",
)
#: ``1:M`` arrays (M values, index 0 unused); padded with one trailing zero.
#: ``lamsal`` is unused here and is accepted with either length.
_ARR_M1 = ("ddz", "dzeta_05int", "area_half", "lamw", "lamsal", "ddz2",
           "ddz054")
#: per-lake scalars
_SCAL = (
    "h1", "dhw", "dhw0", "vol", "botar", "b0", "tau_air", "tau_wav", "tau_i",
    "tau_gr", "uwind", "vwind", "fetch", "roughness",
)
#: keys that the equations read -> must be supplied
_REQUIRED = ("e1", "eps1", "tw1", "tw2", "sal1", "sal2", "u1", "v1", "u2",
             "v2", "row", "h1", "tau_air", "tau_gr", "dhw", "dhw0")
#: geometry keys that must be supplied
_GEOM = ("ddz", "dzeta_int", "dzeta_05int", "area_int", "area_half", "ddz05")


def _arr(x, L, n, key, short_ok=False):
    """Broadcast ``x`` to ``(L, n)``; ``short_ok`` pads a 1:M array by one."""
    a = jnp.asarray(x, dtype=jnp.float64)
    if a.ndim == 0:
        raise ValueError(f"state[{key!r}]: expected an array, got a scalar")
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[-1] == n:
        pass
    elif short_ok and a.shape[-1] == n - 1:
        a = jnp.concatenate([a, jnp.zeros(a.shape[:-1] + (1,))], axis=-1)
    else:
        raise ValueError(f"state[{key!r}]: expected last dim {n}"
                         f"{' or ' + str(n - 1) if short_ok else ''}, "
                         f"got {a.shape}")
    if a.shape[0] == 1:
        return jnp.broadcast_to(a, (L, n))
    if a.shape[0] != L:
        raise ValueError(f"state[{key!r}]: leading dim {a.shape[0]} != {L}")
    return a


def _scal(x, L, key):
    """Broadcast a scalar / ``(1,)`` / ``(L,)`` value to ``(L,)``."""
    a = jnp.asarray(x, dtype=jnp.float64)
    if a.ndim > 1:
        raise ValueError(f"state[{key!r}]: expected a scalar, got {a.shape}")
    if a.ndim == 1:
        if a.shape[0] not in (1, L):
            raise ValueError(f"state[{key!r}]: expected {L} values, "
                             f"got {a.shape}")
        if a.shape[0] == 1:
            a = a[0]
    return jnp.broadcast_to(a, (L,))


def _prepare(state, M):
    """Return ``(arrays, L)``: every known key padded/broadcast for the batch."""
    missing = [k for k in _REQUIRED + _GEOM if k not in state]
    if missing:
        raise KeyError(f"state is missing required keys: {missing}")
    e1 = jnp.asarray(state["e1"], dtype=jnp.float64)
    L = int(e1.shape[0]) if e1.ndim >= 2 else 1
    n2 = M + 2
    out = {}
    for k in _ARR_M2:
        out[k] = (_arr(state[k], L, n2, k) if k in state
                  else jnp.zeros((L, n2)))     # work arrays / unused inputs
    for k in _ARR_M1:
        if k in state:
            out[k] = _arr(state[k], L, n2, k, short_ok=True)
    for k in _SCAL:
        out[k] = _scal(state[k], L, k) if k in state else jnp.zeros((L,))
    # `contrgrad = .false.`: the reference sets GAMT/GAMU/GAMV to 0 on every
    # pass, so the incoming values are irrelevant.
    out["gamt"] = out["gamu"] = out["gamv"] = jnp.zeros((L, n2))
    # `ddz05` is `Grid.ddz05`: Fortran bound 0:M (M+1 values), padded by one.
    out["ddz05"] = _arr(state["ddz05"], L, n2, "ddz05", short_ok=True)
    if "z_half" in state:                  # only read by the zserout budget
        out["z_half"] = jnp.asarray(state["z_half"], dtype=jnp.float64)
    d = out["ddz"]                             # padded: d[:, i] = ddz(i)
    if "ddz2" not in state:
        # `ddz2(1:M-1) = ddz(1:M-1) + ddz(2:M)` (step.build_step_state)
        out["ddz2"] = jnp.concatenate(
            [jnp.zeros((L, 1)), d[:, 1:M] + d[:, 2:M + 1],
             jnp.zeros((L, 2))], axis=-1)
    if "ddz054" not in state:
        # `ddz054(k) = 2*(0.5*ddz(k-1) + ddz(k) + 0.5*ddz(k+1))`, k = 2:M-1
        # (the `ke.ddz054` construction of step.build_step_state)
        out["ddz054"] = jnp.concatenate(
            [jnp.zeros((L, 2)),
             2.0 * (0.5 * d[:, 1:M - 1] + d[:, 2:M] + 0.5 * d[:, 3:M + 1]),
             jnp.zeros((L, 2))], axis=-1)
    return out, L


# ---------------------------------------------------------------------------
# one `keps_mode` pass
# ---------------------------------------------------------------------------
def _pass(s, td, M, dt, dhw_keps, dhw0_keps, sig_eps):
    """One ``keps_mode`` pass of ``K_EPSILON``.

    ``td`` is ``time_deriv`` (``(L,)`` or a Python float): ``0`` reproduces the
    initialization pass (time derivatives neglected), ``1`` the evolution
    pass.  With ``semi_implicit_scheme = True`` the inner ``iterat_keps`` loop
    runs exactly once, so ``iter_t == 0`` in every ``if`` and ``e_it1 == e1``.
    """
    L = s["e1"].shape[0]
    n = M + 2
    i1 = slice(1, M + 1)          # Fortran 1:M
    i2 = slice(2, M)              # Fortran 2:M-1
    jl = slice(1, M)              # i-1 for i = 2:M
    jm = slice(2, M + 1)          # i   for i = 2:M (half-level interpolation)
    jh = slice(2, M + 2)          # i+1 for i = 1:M (differencing stencil)
    z = jnp.zeros((L, n))
    zL = jnp.zeros((L,))
    td = jnp.broadcast_to(jnp.asarray(td, dtype=jnp.float64), (L,))
    tdc = td[:, None]
    dt05 = 0.5 * dt

    e1, eps1 = s["e1"], s["eps1"]
    e_it1, eps_it1 = e1, eps1                 # reset at the start of the pass
    e12, eps12 = e1, eps1                     # 0.5*(e1 + e_it1)

    ddz, ddz2, ddz054 = s["ddz"], s["ddz2"], s["ddz054"]
    ddz05 = s["ddz05"]
    dzeta_05int = s["dzeta_05int"]
    area_int, area_half = s["area_int"], s["area_half"]
    h1 = s["h1"]
    h1sq = h1 * h1
    tw1, tw2 = s["tw1"], s["tw2"]
    sal1, sal2 = s["sal1"], s["sal2"]
    u1, v1, u2, v2 = s["u1"], s["v1"], s["u2"], s["v2"]
    um, vm = s["um"], s["vm"]
    rhotemp, rhosal = s["rhotemp"], s["rhosal"]
    gamt, gamu, gamv = s["gamt"], s["gamu"], s["gamv"]

    # -------- stability functions (turb_mod.f90:489-516) ------------------
    ce, cet, lam_e, lam_eps, work1, work2 = z, z, z, z, z, z
    eps_d = eps12[:, i1] + ACC2
    ratio = e12[:, i1] * e12[:, i1] / (eps_d * eps_d)
    work1 = work1.at[:, i1].set(
        ratio * (rhotemp[:, i1] * (tw1[:, jh] - tw1[:, i1])
                 + C.alsal * rhosal[:, i1] * (sal1[:, jh] - sal1[:, i1]))
        / (h1[:, None] * ddz[:, i1]) * AL)
    du1 = u1[:, jh] - u1[:, i1]
    dv1 = v1[:, jh] - v1[:, i1]
    work2 = work2.at[:, i1].set(
        ratio * (du1 * du1 + dv1 * dv1)
        / ((h1sq[:, None] * ddz[:, i1]) * ddz[:, i1]))
    ce = ce.at[:, i1].set(_ce_canuto(work2[:, i1], work1[:, i1]))
    cet = cet.at[:, i1].set(_cet_canuto(work2[:, i1], work1[:, i1]))
    lam_eps = lam_eps.at[:, i1].set(ce[:, i1] / sig_eps)
    lam_e = lam_e.at[:, i1].set(ce[:, i1] / sigmaE)

    # -------- eddy diffusivities between the half levels ------------------
    e_mid = _interp(e12[:, jl], e12[:, jm], ddz[:, jl], ddz[:, jm])
    eps_mid = _interp(eps12[:, jl], eps12[:, jm], ddz[:, jl], ddz[:, jm])
    lam_eps_mid = _interp(lam_eps[:, jl], lam_eps[:, jm],
                          ddz[:, jl], ddz[:, jm])
    lam_e_mid = _interp(lam_e[:, jl], lam_e[:, jm], ddz[:, jl], ddz[:, jm])
    k2_mid = z.at[:, jm].set(e_mid * e_mid / (eps_mid + ACCK))
    k5_mid = z.at[:, jm].set(
        NIU_WAT + jnp.maximum(k2_mid[:, jm] * lam_eps_mid, min_visc / sig_eps))
    k3_mid = z.at[:, jm].set(
        NIU_WAT + jnp.maximum(k2_mid[:, jm] * lam_e_mid, min_visc / sigmaE))

    # -------- diffusivities and length scales at the levels --------------
    k2, k2t, k5, ell, tf = z, z, z, z, z
    q = (e12[:, i1] * e12[:, i1]) / (eps12[:, i1] + ACCK)
    k2 = k2.at[:, i1].set(jnp.maximum(q, min_visc / ce[:, i1]))
    k2t = k2t.at[:, i1].set(jnp.maximum(q, min_diff / cet[:, i1]))
    k5 = k5.at[:, i1].set(NIU_WAT + k2[:, i1] * lam_eps[:, i1])
    ell = ell.at[:, i1].set(e12[:, i1] ** 1.5 / (eps12[:, i1] + ACC))
    tf = tf.at[:, i1].set(jnp.sqrt(e12[:, i1]) / (ell[:, i1] + ACC))

    # -------- shear and buoyancy production (turb_mod.f90:591-606) ------
    du = um[:, jh] - um[:, i1]
    dv = vm[:, jh] - vm[:, i1]
    gen = z.at[:, i1].set(
        ((du * (du + h1[:, None] * ddz[:, i1] * gamu[:, i1]))
         + (dv * (dv + h1[:, None] * ddz[:, i1] * gamv[:, i1])))
        / (h1sq[:, None] * ddz[:, i1] ** 2) * ce[:, i1])
    buoy = z.at[:, i1].set(
        (-(0.5 * (rhotemp[:, i1]
                  * (tw1[:, jh] + tw2[:, jh] - tw1[:, i1] - tw2[:, i1])
                  + C.alsal * rhosal[:, i1]
                  * (sal1[:, jh] + sal2[:, jh] - sal1[:, i1] - sal2[:, i1]))
           / (h1[:, None] * ddz[:, i1]) + gamt[:, i1]) * ALFT * AL * cet[:, i1]))
    f_out = gen + buoy

    # =====================================================================
    # TKE equation: Neumann b.c. (`kepsbc <= 4`), open water (`l1 == 0`)
    # =====================================================================
    FTKES = zL                      # TKE_FLUX_SHEAR(..., nbc=1) == 0
    FTKEB = -zL                     # -TKE_FLUX_SHEAR(tau_gr, ..., nbc=1)
    a, b, c, d = z, z, z, z
    # -- boundary condition at the top
    xx1 = 0.5 * (area_int[:, 2] / area_half[:, 1] * k3_mid[:, 2] * dt
                 / (h1sq * ddz[:, 1] * ddz05[:, 1])
                 + td * dzeta_05int[:, 1] * dhw_keps / (h1 * ddz05[:, 1])
                 - td * dhw0_keps / (h1 * ddz05[:, 1]))
    yy1 = dt / (h1 * ddz[:, 1])
    b = b.at[:, 1].set(-xx1)
    c = c.at[:, 1].set(-(xx1 + td)
                       - (jnp.abs(buoy[:, 1]) - buoy[:, 1])
                       / (tf[:, 1] + ACC) * dt05
                       - tf[:, 1] * dt)
    d = d.at[:, 1].set(
        -td * e1[:, 1] - xx1 * (e1[:, 2] - e1[:, 1])
        - area_int[:, 1] / area_half[:, 1] * yy1 * FTKES
        - k2t[:, 1] * (buoy[:, 1] + jnp.abs(buoy[:, 1])) * dt05
        - dt * (k2[:, 1] * gen[:, 1]))
    # -- boundary condition at the bottom
    xx2 = 0.5 * (area_int[:, M] / area_half[:, M] * k3_mid[:, M] * dt
                 / (h1sq * ddz[:, M] * ddz05[:, M - 1])
                 - td * dzeta_05int[:, M] * dhw_keps / (h1 * ddz05[:, M - 1])
                 + td * dhw0_keps / (h1 * ddz05[:, M - 1]))
    yy2 = dt / (h1 * ddz[:, M])
    a = a.at[:, M].set(-xx2)
    c = c.at[:, M].set(-(xx2 + td)
                       - (jnp.abs(buoy[:, M]) - buoy[:, M])
                       / (tf[:, M] + ACC) * dt05
                       - tf[:, M] * dt)
    d = d.at[:, M].set(
        -td * e1[:, M] - xx2 * (e1[:, M - 1] - e1[:, M])
        + area_int[:, M + 1] / area_half[:, M] * yy2 * FTKEB
        - k2t[:, M] * (buoy[:, M] + jnp.abs(buoy[:, M])) * dt05
        - dt * (k2[:, M] * gen[:, M]))
    # -- interior points (computed on the whole array, written at 2:M-1)
    e1p, e1m = _sh(e1, 1), _sh(e1, -1)
    ddz2m = _sh(ddz2, -1)
    h1c = h1[:, None]
    dhw_t = tdc * dzeta_05int * dhw_keps[:, None] * (ddz054 * h1c) ** (-1)
    dhw0_t = tdc * dhw0_keps[:, None] * (ddz054 * h1c) ** (-1)
    a_int = (dhw_t - dhw0_t
             - area_int / area_half * k3_mid * dt
             / (h1sq[:, None] * ddz * ddz2m))
    b_int = (-dhw_t + dhw0_t
             - _sh(area_int, 1) / area_half * _sh(k3_mid, 1) * dt
             / (h1sq[:, None] * ddz * ddz2))
    a = a.at[:, i2].set(a_int[:, i2])
    b = b.at[:, i2].set(b_int[:, i2])
    c = c.at[:, i2].set(
        a[:, i2] + b[:, i2] - td[:, None]
        - (jnp.abs(buoy[:, i2]) - buoy[:, i2]) / (tf[:, i2] + ACC) * dt05
        - tf[:, i2] * dt)
    d_int = (-tdc * e1
             - k2t * (buoy + jnp.abs(buoy)) * dt05
             - dt * (k2 * gen))
    d_int = d_int - dt * (h1sq[:, None] * ddz * ddz2) ** (-1) \
        * (_sh(area_int, 1) / area_half * _sh(k3_mid, 1) * (e1p - e1))
    d_int = d_int + dt * (h1sq[:, None] * ddz * ddz2m) ** (-1) \
        * (area_int / area_half * k3_mid * (e1 - e1m))
    d_int = d_int - dhw_t * (e1p - e1m) + dhw0_t * (e1p - e1m)
    d = d.at[:, i2].set(d_int[:, i2])

    # `IND_STAB_FACT_DB(..., ind_bound=.false.)`: fall back to the pure
    # diffusion coefficients when the matrix is not diagonally dominant.
    indstab = jnp.all(
        (jnp.abs(a[:, i2]) + jnp.abs(b[:, i2])) < jnp.abs(c[:, i2]), axis=-1)
    a = jnp.where(indstab[:, None], a,
                  a.at[:, i2].set((-k3_mid * dt
                                   / (h1sq[:, None] * ddz * ddz2m))[:, i2]))
    b = jnp.where(indstab[:, None], b,
                  b.at[:, i2].set((-_sh(k3_mid, 1) * dt
                                   / (h1sq[:, None] * ddz * ddz2))[:, i2]))

    y = progonka_jax(a[:, i1], b[:, i1], c[:, i1], d[:, i1], k=0, n=M - 1)
    e_it2 = z.at[:, i1].set(y)
    e_it2 = e_it2.at[:, M + 1].set(0.0)
    # CHECK_MIN_VALUE(E_it2, M, E_min); l1 == 0 -> E_min = E_MIN_LAMW0
    e_it2 = e_it2.at[:, i1].set(jnp.maximum(e_it2[:, i1], E_MIN_LAMW0))
    dE_it = jnp.max(jnp.abs(e_it1[:, 1:M + 2] - e_it2[:, 1:M + 2]), axis=-1)

    # -------- coefficients of the dissipation equation (Satyanarayana) ---
    yy1s = 0.5 * (1.0 + _fsign1(buoy))
    zzs = 0.5 * (1.0 - _fsign1(buoy))
    c_eps3 = z.at[:, i1].set(
        (ceps3_stable * zzs + ceps3_unstable * yy1s)[:, i1])
    c_eps1 = z.at[:, i1].set(ceps1)
    c_eps2 = z.at[:, i1].set(ceps2)

    # =====================================================================
    # dissipation-rate equation
    # =====================================================================
    ksurf1 = jnp.maximum(NIU_WAT, ce[:, 1] * k2[:, 1]
                         - kar * jnp.sqrt(s["tau_air"] / C.row0)
                         * 0.5 * ddz[:, 1] * h1)
    Esurf1 = e_it1[:, 1] + 0.5 * FTKES * sigmaE * h1 * ddz[:, 1] / ksurf1
    const1 = CE0 ** 0.75 / (sig_eps * kar)
    FS = const1 * ksurf1 * Esurf1 ** 1.5 / (Z0 * Z0 + SMALL_NUMBER)

    ksurf2b = jnp.maximum(NIU_WAT, ce[:, M] * k2[:, M]
                          - kar * jnp.sqrt(s["tau_gr"] / C.row0)
                          * 0.5 * ddz[:, M] * h1)
    Esurf2b = e_it1[:, M] - 0.5 * FTKEB * sigmaE * h1 * ddz[:, M] / ksurf2b
    FB = -(const1 * ksurf2b * Esurf2b ** 1.5 / (Z0 * Z0 + SMALL_NUMBER))

    a, b, c, d = z, z, z, z
    xx1 = 0.5 * (area_int[:, 2] / area_half[:, 1] * k5_mid[:, 2] * dt
                 / (h1sq * ddz[:, 1] * ddz05[:, 1])
                 + td * dzeta_05int[:, 1] * dhw_keps / (h1 * ddz05[:, 1])
                 - td * dhw0_keps / (h1 * ddz05[:, 1]))
    yy1 = dt / (h1 * ddz[:, 1])
    b = b.at[:, 1].set(-xx1)
    c = c.at[:, 1].set(-(xx1 + td)
                       - c_eps2[:, 1] * tf[:, 1] * dt
                       + c_eps3[:, 1] * (-jnp.abs(buoy[:, 1]) + buoy[:, 1])
                       / (tf[:, 1] + ACC) * dt05)
    d = d.at[:, 1].set(
        -td * eps1[:, 1] - xx1 * (eps1[:, 2] - eps1[:, 1])
        - area_int[:, 1] / area_half[:, 1] * yy1 * FS
        - c_eps3[:, 1] * (jnp.abs(buoy[:, 1]) + buoy[:, 1])
        * tf[:, 1] * k2t[:, 1] * dt05
        - c_eps1[:, 1] * tf[:, 1] * dt * (k2[:, 1] * gen[:, 1]))
    xx2 = 0.5 * (area_int[:, M] / area_half[:, M] * k5_mid[:, M] * dt
                 / (h1sq * ddz[:, M] * ddz05[:, M - 1])
                 - td * dzeta_05int[:, M] * dhw_keps / (h1 * ddz05[:, M - 1])
                 + td * dhw0_keps / (h1 * ddz05[:, M - 1]))
    yy2 = dt / (h1 * ddz[:, M])
    a = a.at[:, M].set(-xx2)
    c = c.at[:, M].set(-(xx2 + td)
                       - c_eps2[:, M] * tf[:, M] * dt
                       + c_eps3[:, M] * (-jnp.abs(buoy[:, M]) + buoy[:, M])
                       / (tf[:, M] + ACC) * dt05)
    d = d.at[:, M].set(
        -td * eps1[:, M] - xx2 * (eps1[:, M - 1] - eps1[:, M])
        + area_int[:, M + 1] / area_half[:, M] * yy2 * FB
        - c_eps3[:, M] * (jnp.abs(buoy[:, M]) + buoy[:, M])
        * tf[:, M] * k2t[:, M] * dt05
        - c_eps1[:, M] * tf[:, M] * dt * (k2[:, M] * gen[:, M]))
    epsp, epsm = _sh(eps1, 1), _sh(eps1, -1)
    a_int = (dhw_t - dhw0_t
             - area_int / area_half * k5_mid * dt
             / (h1sq[:, None] * ddz * ddz2m))
    b_int = (-dhw_t + dhw0_t
             - _sh(area_int, 1) / area_half * _sh(k5_mid, 1) * dt
             / (h1sq[:, None] * ddz * ddz2))
    a = a.at[:, i2].set(a_int[:, i2])
    b = b.at[:, i2].set(b_int[:, i2])
    c = c.at[:, i2].set(
        a[:, i2] + b[:, i2] - td[:, None]
        - c_eps2[:, i2] * tf[:, i2] * dt
        + c_eps3[:, i2] * (-jnp.abs(buoy[:, i2]) + buoy[:, i2])
        / (tf[:, i2] + ACC) * dt05)
    d_int = (-tdc * eps1
             - c_eps3 * (jnp.abs(buoy) + buoy) * tf * k2t * dt05
             - c_eps1 * tf * dt * (k2 * gen))
    d_int = d_int - dt * (h1sq[:, None] * ddz * ddz2) ** (-1) \
        * (_sh(area_int, 1) / area_half * _sh(k5_mid, 1) * (epsp - eps1))
    d_int = d_int + dt * (h1sq[:, None] * ddz * ddz2m) ** (-1) \
        * (area_int / area_half * k5_mid * (eps1 - epsm))
    d_int = d_int - dhw_t * (epsp - epsm) + dhw0_t * (epsp - epsm)
    d = d.at[:, i2].set(d_int[:, i2])

    indstab = jnp.all(
        (jnp.abs(a[:, i2]) + jnp.abs(b[:, i2])) < jnp.abs(c[:, i2]), axis=-1)
    a = jnp.where(indstab[:, None], a,
                  a.at[:, i2].set((-k5_mid * dt
                                   / (h1sq[:, None] * ddz * ddz2m))[:, i2]))
    b = jnp.where(indstab[:, None], b,
                  b.at[:, i2].set((-_sh(k5_mid, 1) * dt
                                   / (h1sq[:, None] * ddz * ddz2))[:, i2]))

    y = progonka_jax(a[:, i1], b[:, i1], c[:, i1], d[:, i1], k=0, n=M - 1)
    eps_it2 = z.at[:, i1].set(y)
    eps_it2 = eps_it2.at[:, M + 1].set(0.0)
    # CHECK_MIN_VALUE(eps_it2, M, eps_min); l1 == 0 -> eps_min = EPS_MIN0
    eps_it2 = eps_it2.at[:, i1].set(jnp.maximum(eps_it2[:, i1], EPS_MIN0))
    deps_it = jnp.max(jnp.abs(eps_it1[:, 1:M + 2] - eps_it2[:, 1:M + 2]),
                      axis=-1)

    # `iterat = .false.` (semi_implicit_scheme): E2 = E_it2, eps2 = eps_it2
    return {
        "e_it1": e_it1, "eps_it1": eps_it1, "e_it2": e_it2,
        "eps_it2": eps_it2, "e2": e_it2, "eps2": eps_it2,
        "eps12": eps12, "ce": ce, "cet": cet, "lam_e": lam_e,
        "lam_eps": lam_eps, "ceps": (c_eps1, c_eps2, c_eps3),
        "k2": k2, "k2t": k2t, "k5": k5, "k2_mid": k2_mid, "k3_mid": k3_mid,
        "k5_mid": k5_mid, "l": ell, "tf": tf, "f": f_out,
        "s": buoy, "gen": gen, "gamt": gamt, "gamu": gamu, "gamv": gamv,
        "FTKES": FTKES, "FTKEB": FTKEB, "FS": FS, "FB": FB,
        "dE_it": dE_it, "deps_it": deps_it,
    }


# ---------------------------------------------------------------------------
# the public kernel
# ---------------------------------------------------------------------------
def k_epsilon(state, dt, dt_keps=None, nstep_keps=None, params=None):
    """Advance the batched k-epsilon closure by one sub-time step (``K_EPSILON``).

    Parameters
    ----------
    state : dict
        Batched state for ``L`` lakes (leading lake axis), see the module
        docstring.  Not modified.
    dt : float
        The model time step, s (used for ``nstep_keps`` and
        ``dhw_keps = dhw/nstep_keps``).
    dt_keps : float, optional
        The k-epsilon sub-step, s -- the value ``lake.f90`` passes as the
        Fortran dummy argument ``dt``.  Defaults to ``dt``.
    nstep_keps : int, optional
        Number of k-epsilon sub-steps per model step.  Defaults to
        ``round(dt/dt_keps)``.
    params : KEpsilonParams | mapping | None
        Switch settings mirroring
        :class:`lakematrix.physics.kepsilon.KEpsilonParams`.

    Returns
    -------
    dict
        ``state`` with ``e1``/``eps1`` advanced, plus ``e2``, ``eps2``, the
        diffusivities (``k2`` already multiplied by ``ce``, ``k2t``, ``k5``,
        ``kt``), the stability functions and the TKE/eps budget diagnostics.
    """
    p = _as_params(params)
    _check_scope(p)
    if _concrete_all(state.get("l1", 0.0), 0.0) is False:
        raise NotImplementedError(
            "jaxbatch.k_epsilon implements the open-water case l1 == 0 only")
    if _concrete_all(jnp.asarray(state.get("h_veg", 0.0)) <= 0.0, True) is False:
        raise NotImplementedError(
            "vegetation friction is not operational (Fortran STOP); h_veg > 0")

    M = int(p.M)
    dt = float(dt)
    dt_keps = dt if dt_keps is None else float(dt_keps)
    if nstep_keps is None:
        nstep_keps = int(round(dt / dt_keps))
    nstep_keps = int(nstep_keps)
    if nstep_keps < 1:
        raise ValueError("nstep_keps must be >= 1")
    if dt_keps <= 0.0:
        raise ValueError("dt_keps must be > 0")

    sig_eps = sigmaeps(int(p.kepsbc))
    s, L = _prepare(state, M)

    # ----- mixed-layer depth / bulk Richardson number (before the passes) --
    i_entrain = jnp.argmin(s["e1"][:, 1:M + 1], axis=-1) + 1          # (L,)
    H_entrainment = (jnp.take_along_axis(s["dzeta_05int"],
                                         i_entrain[:, None], axis=-1)[:, 0]
                     * s["h1"])
    u_i = jnp.take_along_axis(s["u1"], i_entrain[:, None], axis=-1)[:, 0]
    v_i = jnp.take_along_axis(s["v1"], i_entrain[:, None], axis=-1)[:, 0]
    row_i = jnp.take_along_axis(s["row"], i_entrain[:, None], axis=-1)[:, 0]
    dz_i = jnp.take_along_axis(s["dzeta_int"], i_entrain[:, None], axis=-1)[:, 0]
    Ri_bulk = (AL * (row_i - s["row"][:, 1]) * (dz_i * s["h1"])
               / ((u_i - s["u1"][:, 1]) ** 2 + (v_i - s["v1"][:, 1]) ** 2
                  + C.small_value))

    # ----- time-interpolated velocity (Crank-Nicolson shear production) ---
    s["um"] = 0.5 * (s["u2"] + s["u1"])
    s["vm"] = 0.5 * (s["v2"] + s["v1"])

    # ----- density derivatives at the interfaces, old time level ---------
    tm = 0.5 * (s["tw1"][:, 1:M + 1] + s["tw1"][:, 2:M + 2])
    sm = 0.5 * (s["sal1"][:, 1:M + 1] + s["sal1"][:, 2:M + 2])
    rhotemp = jnp.zeros((L, M + 2)).at[:, 1:M + 1].set(_ddens_dtemp(tm, sm))
    rhosal = jnp.zeros((L, M + 2)).at[:, 1:M + 1].set(_ddens_dsal(tm, sm))
    s["rhotemp"] = rhotemp
    s["rhosal"] = rhosal

    # ----- TKE injection from wave breaking (diagnostic only for kepsbc=1) -
    maxTKEinput = (s["tau_wav"] / C.row0
                   * jnp.sqrt(s["uwind"] * s["uwind"]
                              + s["vwind"] * s["vwind"]))

    dhw_keps = s["dhw"] / float(nstep_keps)
    dhw0_keps = s["dhw0"] / float(nstep_keps)
    args = (M, dt_keps, dhw_keps, dhw0_keps, sig_eps)

    # ----- keps_mode loop: time_deriv = 0 then 1 unless already initialised
    ik = state.get("init_keps", False)
    if isinstance(ik, bool):
        all_init, ik_arr = ik, None
    else:
        all_init, ik_arr = False, _scal(ik, L, "init_keps")
    if not all_init:
        init = _pass(s, 0.0, *args)
        if ik_arr is None:
            e1_mid, eps1_mid = init["e2"], init["eps2"]
        else:
            cond = (ik_arr > 0.5)[:, None]
            e1_mid = jnp.where(cond, s["e1"], init["e2"])
            eps1_mid = jnp.where(cond, s["eps1"], init["eps2"])
        s = dict(s)
        s["e1"] = e1_mid
        s["eps1"] = eps1_mid
    out = _pass(s, 1.0, *args)

    # ---------------- diagnostics (turb_mod.f90:957-1111) ----------------
    e1, e2 = s["e1"], out["e2"]
    eps2 = out["eps2"]
    e12 = 0.5 * (e1 + e2)
    eps12 = 0.5 * (s["eps1"] + eps2)
    i1 = slice(1, M + 1)
    jh = slice(2, M + 2)          # i+1 for i = 1:M
    i2 = slice(2, M)
    h1, ddz = s["h1"], s["ddz"]
    h1sq = h1 * h1
    # stability function for heat at the new time level (stabfunc = 2)
    eps_d = eps2[:, i1] + ACC2
    ratio = e2[:, i1] * e2[:, i1] / (eps_d * eps_d)
    w1 = (ratio * (rhotemp[:, i1] * (s["tw2"][:, jh] - s["tw2"][:, i1])
                   + C.alsal * rhosal[:, i1]
                   * (s["sal2"][:, jh] - s["sal2"][:, i1]))
          / (h1[:, None] * ddz[:, i1]) * AL)
    du2 = s["u2"][:, jh] - s["u2"][:, i1]
    dv2 = s["v2"][:, jh] - s["v2"][:, i1]
    w2 = (ratio * (du2 * du2 + dv2 * dv2)
          / ((h1sq[:, None] * ddz[:, i1]) * ddz[:, i1]))
    cet = jnp.zeros((L, M + 2)).at[:, i1].set(_cet_canuto(w2, w1))
    kt = jnp.zeros((L, M + 2)).at[:, i1].set(
        jnp.maximum(cet[:, i1] * e2[:, i1] ** 2 / (eps2[:, i1] + ACCK),
                    min_diff) * C.cw_m_row0)

    # ----- turbulent transport of TKE (density: diagnostics only) --------
    ddz05p = s["ddz05"]           # already padded: ddz05p[:, i] = ddz05(i)
    k3_mid = out["k3_mid"]
    ttt = jnp.zeros((L, M + 2))
    e2p, e2m = _sh(e2, 1), _sh(e2, -1)
    e1p, e1m = _sh(e1, 1), _sh(e1, -1)
    d05m = _sh(ddz05p, -1)
    ttt_full = (1.0 / (s["area_half"] * h1[:, None] * h1[:, None] * ddz)
                * (_sh(s["area_int"], 1) * _sh(k3_mid, 1)
                   * 0.5 * (e2p + e1p - e2 - e1) / ddz05p
                   - s["area_int"] * k3_mid
                   * 0.5 * (e2 + e1 - e2m - e1m) / d05m))
    ttt = ttt.at[:, i2].set(ttt_full[:, i2])
    ttt = ttt.at[:, 1].set(
        1.0 / (s["area_half"][:, 1] * h1 * h1 * ddz[:, 1])
        * (s["area_int"][:, 2] * k3_mid[:, 2]
           * 0.5 * (e2[:, 2] + e1[:, 2] - e2[:, 1] - e1[:, 1]) / ddz05p[:, 1]
           + s["area_int"][:, 1] * out["FTKES"] * h1))
    ttt = ttt.at[:, M].set(
        1.0 / (s["area_half"][:, M] * h1 * h1 * ddz[:, M])
        * (-s["area_int"][:, M + 1] * out["FTKEB"] * h1
           - s["area_int"][:, M] * k3_mid[:, M]
           * 0.5 * (e2[:, M] + e1[:, M] - e2[:, M - 1] - e1[:, M - 1])
           / ddz05p[:, M - 1]))

    # ----- TKE / eps production-dissipation terms of the budgets ---------
    buoy, gen, tf, k2, k2t = out["s"], out["gen"], out["tf"], out["k2"], out["k2t"]
    S = (0.5 * (buoy + jnp.abs(buoy)) * k2t
         + 0.5 * (-jnp.abs(buoy) + buoy) / (tf + ACC) * e2)
    G = gen * k2

    # ----- integral budgets (turb_mod.f90:1055-1111) ---------------------
    idx = jnp.arange(M + 2)[None, :]
    mask = (idx >= 1) & (idx <= i_entrain[:, None])
    xx = ddz * h1[:, None]
    ystep = 0.5 * (1.0 + _fsign1(S))
    zstep = 0.5 * (1.0 - _fsign1(S))
    c_eps1, c_eps2, c_eps3 = out["ceps"]
    ratio_eps = eps2 / e2
    zero = jnp.zeros((L, M + 2))
    S_pos = jnp.sum(jnp.where(mask, ystep * S * xx, zero), axis=-1)
    S_neg = jnp.sum(jnp.where(mask, zstep * S * xx, zero), axis=-1)
    Seps_pos = jnp.sum(jnp.where(
        mask, ystep * S * c_eps3 * ratio_eps * xx, zero), axis=-1)
    Seps_neg = jnp.sum(jnp.where(
        mask, zstep * S * c_eps3 * ratio_eps * xx, zero), axis=-1)
    Gen_int = jnp.sum(jnp.where(mask, G * xx, zero), axis=-1)
    TKE_tt = jnp.sum(jnp.where(mask, ttt * xx, zero), axis=-1)
    Geneps = jnp.sum(jnp.where(mask, G * c_eps1 * ratio_eps * xx, zero), axis=-1)
    epseps = jnp.sum(jnp.where(mask, eps2 * c_eps2 * ratio_eps * xx, zero),
                     axis=-1)
    eps_int = jnp.sum(jnp.where(mask, tf * e2 * xx, zero), axis=-1)
    E_int = jnp.sum(jnp.where(mask, e2 * xx, zero), axis=-1)
    TF_int = jnp.sum(jnp.where(mask, tf * xx, zero), axis=-1)
    TKE_balance = (Gen_int + S_pos + S_neg - eps_int) / H_entrainment
    eps_balance = (Geneps + Seps_pos + Seps_neg - epseps) / H_entrainment

    # ------------------------------ epilogue ----------------------------
    k2_out = jnp.zeros((L, M + 2)).at[:, i1].set(
        out["k2"][:, i1] * out["ce"][:, i1])

    res = dict(state)
    res.update({
        "e1": e2, "eps1": eps2, "e2": e2, "eps2": eps2,
        "e12": e12, "eps12": eps12, "e_it1": out["e_it1"],
        "eps_it1": out["eps_it1"], "e_it2": out["e_it2"],
        "eps_it2": out["eps_it2"],
        "k2": k2_out, "k2t": k2t, "k5": out["k5"],
        "k2_mid": out["k2_mid"], "k3_mid": k3_mid, "k5_mid": out["k5_mid"],
        "l": out["l"], "tf": tf, "kt": kt,
        "ce": out["ce"], "cet": cet, "lam_e": out["lam_e"],
        "lam_eps": out["lam_eps"],
        # 1:M arrays: return the Fortran bound, i.e. drop the padding index
        "c_eps1": c_eps1[:, :M + 1], "c_eps2": c_eps2[:, :M + 1],
        "c_eps3": c_eps3[:, :M + 1],
        "s": S[:, :M + 1], "gen": G[:, :M + 1], "f": out["f"][:, :M + 1],
        "tke_turb_trans": ttt[:, :M + 1],
        "gamt": out["gamt"], "gamu": out["gamu"], "gamv": out["gamv"],
        "um": s["um"], "vm": s["vm"], "row": s["row"], "row2": s["row2"],
        "i_entrain": i_entrain, "H_entrainment": H_entrainment,
        "Ri_bulk": Ri_bulk, "signwaveheight": jnp.zeros((L,)),
        "FTKES": out["FTKES"], "FTKEB": out["FTKEB"], "FS": out["FS"],
        "FB": out["FB"], "dE_it": out["dE_it"], "deps_it": out["deps_it"],
        "maxTKEinput": maxTKEinput,
        "S_integr_positive": S_pos / H_entrainment,
        "S_integr_negative": S_neg / H_entrainment,
        "Gen_integr": Gen_int / H_entrainment,
        "TKE_turb_trans_integr": TKE_tt / H_entrainment,
        "Seps_integr_positive": Seps_pos / H_entrainment,
        "Seps_integr_negative": Seps_neg / H_entrainment,
        "Geneps_integr": Geneps / H_entrainment,
        "epseps_integr": epseps / H_entrainment,
        "eps_integr": eps_int / H_entrainment,
        "E_integr": E_int / H_entrainment,
        "TF_integr": TF_int / H_entrainment,
        "TKE_balance": TKE_balance, "eps_balance": eps_balance,
        "init_keps": jnp.ones((L,), dtype=bool),
    })
    return res


def make_kernel(params=None):
    """Return a JIT-compiled ``f(state, dt, dt_keps, nstep_keps)`` closure.

    ``dt``/``dt_keps``/``nstep_keps`` are static arguments (they are Python
    scalars in :func:`k_epsilon`).
    """
    p = _as_params(params)

    @jax.jit(static_argnums=(1, 2, 3))
    def kernel(state, dt, dt_keps, nstep_keps):
        return k_epsilon(state, dt, dt_keps, nstep_keps, p)

    return kernel
