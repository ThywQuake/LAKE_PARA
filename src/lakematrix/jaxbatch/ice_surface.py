"""JAX (batched) ice / snow surface: ``SURF_CHAR``, ``HEATFLUXSURF``,
``RADBALANCE`` and the ``HEATBALANCE`` composition for case 2 (ice) and case 3
(snow).

This is the ice counterpart of :mod:`lakematrix.jaxbatch.surface` (which only
covers the case-1 open-water surface) and of the NumPy
:func:`lakematrix.physics.surface.heatbalance_ice`.

Fortran layout
==============

* ``SURF_CHAR`` ice/snow branches - ``t_solver_mod.f90:940-1034``
* ``HEATFLUXSURF`` ice/snow branches - ``heatbalsurf_mod.f90:113-150``
* ``RADBALANCE`` ice/snow branches - ``heatbalsurf_mod.f90:388-508``
* ``HEATBALANCE`` (``skin%par == 0``) - ``heatbalsurf_mod.f90:8-72``
* ``SENSLATMOM_FLUXES``/``SfcFlx`` - ``SfcFlx_mod.F90:221-612``.  The ice
  branch is ported **here** (:func:`momsenlat_ice`,
  :func:`senslatmom_fluxes_ice`) because
  :mod:`lakematrix.jaxbatch.sfcflx` only covers the open-water branch
  (``h_ice < h_Ice_min_flk``); over ice the saturation curve, the roughness
  lengths, the friction-velocity iteration and the latent heat of sublimation
  all differ, worth ~50 W/m2 in ``Elatent`` at ``l1 = 0.5 m``.

Everything gets a leading lake axis ``L``; the surface-type branch
(``surftyp = 2`` ice vs ``3`` snow) is a *per-lake* flag, so both
``HEATFLUXSURF`` branches are evaluated and combined with ``jnp.where`` - the
whole function is traceable, ``jit``-able and ``vmap``-able, and a batch may
mix bare-ice and snow-covered lakes.

Array conventions
=================

Arrays that the Fortran keeps 1-based stay padded at index 0 (``Ti1``,
``Ti2``, ``lami_v``, ``ci_m_roi_v``, ``ddzi``, ``area_int``, ``area_half``,
``cs``, ``dens``, ``dz``, ``lams``, ``Tsn``, ``T``); ``radice_integr`` keeps
its 0-based bound, i.e. it is ``RADICE%integr(0:Mice)``.
:func:`heatfluxsurf_ice` only reads indices 1 and 2 (surface ice layer and the
one below it), :func:`heatfluxsurf_snow` only reads ``itop``/``itop+1``.

``ctx`` keys (all values are leading-lake batched ``(L, ...)`` unless noted)
==========================================================================

Flags / switches (``ARRAYS``, ``DRIVING_PARAMS``)
    ``snow`` (L,) int ``0/1`` - ``flag_snow``; selects ice vs snow physics.
    ``water``, ``ice``, ``deepice`` (L,) int ``0/1`` - the ``T_DIFF`` flags,
        passed straight through to ``t_diff_fn`` (``ice``/``deepice``
        default to ``1``/``0``).
    ``isoilcol`` - ``nsoilcols`` (static int or (L,) int) for ``T_DIFF``.
    ``dt`` - time step ``[s]`` (static float or (L,)).
    ``state`` - the ice/snow state **dict** handed to ``t_diff_fn`` (input
        copy; the updated dict comes back from ``t_diff_fn``).

Atmosphere / surface exchange (``ATMOS`` + ``SURF_CHAR``)
    ``fetch``, ``uwind``, ``vwind``, ``u``, ``v``, ``tempair``, ``pressure``,
        ``humair``, ``zref``.
    ``roughness``, ``relhums``, ``aM``, ``bM`` - accepted for parity with the
        NumPy :class:`~lakematrix.physics.surface.IceSurfaceContext` (and
        returned by :func:`surf_char_ice`), but **not** read: the
        ``PBLpar = 3`` path of ``SENSLATMOM_FLUXES`` passes no roughness to
        ``SfcFlx_momsenlat`` (it computes the Andreas (2002) ice values itself)
        and uses ``SfcFlx_satwvpres`` rather than ``aM``/``bM``/``relhums``.
        The NumPy reference is equally inert there.
    ``longwave``, ``shortwave``, ``netrad``, ``cloud`` (``netrad``/``cloud``
        are not read - ``ifrad = 1`` as in the reference - but are accepted by
        the NumPy dataclass, so they are documented here too).
    ``albedo``, ``albedo_lw``, ``emissivity`` - from :func:`surf_char_ice`.

Radiation
    ``radice_integr`` (L, Mice+1) - 0-based ``RADICE%integr(0:Mice)``; only
        ``[..., 0]`` and ``[..., 1]`` are used.  Optional: ``None`` means no
        shortwave absorption in the ice (``shortbal = 0``), which is exact for
        a snow-only batch.

Ice / snow arrays read by ``HEATFLUXSURF``
    ``Ti1``, ``Ti2`` (L, Mice+2) - surface-ice temperature *before* / *after*
        ``T_DIFF``, ``T``, ``Tsn`` (L, ms+1) - snow temperature before / after.
    ``lami_v``, ``ci_m_roi_v`` (L, Mice+1), ``l1``, ``dhi0`` (L,),
        ``ddzi`` (L, Mice+1) or (Mice+1,), ``area_int``, ``area_half``
        (L, Mice+1) - ``BATHYMICE`` areas, ``area_int[..., 1]`` =
        ``bathymice(1)%area_int``.
    ``cs``, ``dens``, ``dz``, ``lams`` (L, ms+1), ``itop`` (L,) int - snow
        pack; ``1 <= itop <= ms - 1`` (``lams[itop+1]`` must exist, as in the
        NumPy/ Fortran reference where ``snow_cond_heat_coef`` sets
        ``itop = ms-2``).

Every name in the "Ice / snow arrays" block, plus the four temperatures, is
looked up in the dict returned by ``t_diff_fn`` first, then in ``ctx``, then in
``ctx["state"]`` (see :func:`_pick`).  A ``t_diff_fn`` therefore only has to
return what it changed (typically ``Tsn``/``T`` and ``Ti2``); unchanged
temperatures and material properties are taken from the input state / ``ctx``.

``t_diff_fn`` contract
======================

``t_diff_fn(surf, Tsurf, dt, snow, ice, water, deepice, isoilcol, ifflux,
state) -> dict`` - the batched ``T_DIFF`` kernel (B5 work, not part of this
module), i.e. the JAX port of :func:`lakematrix.physics.ice.t_diff_ice`.  It
must be **pure** (no in-place mutation of ``state``): every call receives the
same ``state`` and returns a fresh dict, which :func:`heatbalance_ice` passes
back as ``out["ice"]``.  ``heatbalance_ice`` calls it exactly once, with
``surf = 1``, ``ifflux = False`` and the flags from ``ctx``.

Returns ``(HB, out)`` where ``out`` mirrors the NumPy
:func:`~lakematrix.physics.surface.heatbalance_ice` state dict plus the
``T_DIFF`` output::

    out["ice"]           dict returned by t_diff_fn (carry it into the next
                         call, e.g. out["ice"] of the previous iteration)
    out["eflux"], out["dHdt"], out["convect_flux"], out["hflux"],
    out["Elatent"], out["Radbal"], out["Radbal_surf"], out["eflux0"],
    out["eflux0_kinem"], out["cdmw"].

Scope
=====

Case 2/3 only (``l1 > 0`` or a snow pack).  Full integration into the batched
step driver (case selection, ``T_SOLVER`` and the ``T_DIFF`` ice kernel) is
B5+, i.e. it depends on the batched ice ``T_DIFF`` that this module
deliberately takes as a callable argument.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C
from ..physics import sfcflx as SF
from .sfcflx import rhoair, spechum
from .surface import heatbalance

#: ``surftyp`` indicators of ``HEATBALANCE`` (``heatbalsurf_mod.f90``).
ICE = 2
SNOW = 3

# ``_f32(1.0e-5)`` / ``_f32(1.0e-3)`` of the NumPy ``surf_char``: default-real
# Fortran literals rounded through float32 before the assignment to real(8).
_ICE_ROUGHNESS = float(np.float32(1.0e-5))
_SNOW_ROUGHNESS = float(np.float32(1.0e-3))
# ``_f32(0.7)``: the shared ``relhums`` of the ice and snow branches.
_RELHUMS = float(np.float32(0.7))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _snow_mask(surftyp):
    """``True`` where ``surftyp == SNOW``; validates a static ``surftyp``."""
    if isinstance(surftyp, (int, np.integer)):
        if int(surftyp) not in (ICE, SNOW):
            raise ValueError(
                f"surf_char_ice/radbalance_ice handle surftyp 2 (ice) and "
                f"3 (snow), got {int(surftyp)}")
        return int(surftyp) == SNOW
    return jnp.asarray(surftyp) == SNOW


def _pick(st, ctx, name):
    """Look ``name`` up in the ``t_diff_fn`` result, then ``ctx``, then state."""
    for src in (st, ctx, ctx.get("state", {})):
        if name in src:
            return src[name]
    raise KeyError(
        f"heatbalance_ice: missing {name!r}; it must be returned by "
        f"t_diff_fn, or provided in ctx, or in ctx['state'] "
        f"(known keys: t_diff_fn={sorted(st)}, ctx={sorted(ctx)}, "
        f"state={sorted(ctx.get('state', {}))})")


def _gather_last(x, idx):
    """``x[..., idx]`` for an arbitrary leading batch / a scalar-or-batched idx."""
    x = jnp.asarray(x)
    idx = jnp.asarray(idx, dtype=jnp.int32)
    batch = jnp.broadcast_shapes(x.shape[:-1], idx.shape)
    x = jnp.broadcast_to(x, batch + x.shape[-1:])
    idx = jnp.broadcast_to(idx, batch)
    return jnp.take_along_axis(x, idx[..., None], axis=-1)[..., 0]


def _check_itop(itop, lams):
    """Static ``1 <= itop <= ms - 1`` guard when both bounds are known.

    ``lams`` is the 0-based ``(..., ms+1)`` array of the NumPy/Fortran
    ``SNOWSOIL_MOD`` bounds, so ``lams[itop + 1]`` needs ``itop + 1 <= ms``.
    A batched ``itop`` cannot be validated outside a trace; the constraint is
    then the caller's (``snow_cond_heat_coef`` sets ``itop = ms - 2``).
    """
    if isinstance(itop, (int, np.integer)) and jnp.ndim(lams) >= 1:
        k = int(jnp.shape(lams)[-1])
        if not 1 <= int(itop) <= k - 2:
            raise ValueError(
                f"heatfluxsurf_snow requires 1 <= itop <= {k - 2} for the "
                f"0-based snow arrays of length {k}, got itop={int(itop)}")


# ---------------------------------------------------------------------------
# SfcFlx, ice branch
# ---------------------------------------------------------------------------
def satwvpres_ice(T):
    """``SfcFlx_satwvpres`` ice branch (``h_ice >= h_Ice_min_flk``).

    :func:`lakematrix.jaxbatch.sfcflx.momsenlat` only ports the *water*
    saturation curve, which is why the ice surface carries its own copy of the
    exchange code: over ice ``b2i_vap``/``b4i_vap`` apply instead of
    ``b2w_vap``/``b4w_vap`` (``SfcFlx_mod.F90:855-936``).
    """
    return SF.b1_vap * jnp.exp(SF.b2i_vap * (T - SF.b3_vap)
                              / (T - SF.b4i_vap))


def roughness_ice(fetch, U_a, u_star):
    """``SfcFlx_roughness`` ice branch (``SfcFlx_mod.F90:700-844``).

    ``fetch`` is unused over ice (the Charnock parameter is bypassed and
    formally set to its minimum); the scalar roughness lengths follow the
    Andreas (2002) fits of ``Re_s = u_star*z0u/nu``, split into a
    molecular-sublayer and a rough regime at ``Re_z0s_ice_t``.
    """
    c_z0u_fetch = SF.c_z0u_rough
    u_star_thresh = SF.c_z0u_smooth * SF.tpsf_nu_u_a / SF.z0u_ice_rough
    z0u = jnp.maximum(SF.z0u_ice_rough,
                      SF.c_z0u_smooth * SF.tpsf_nu_u_a / u_star)
    Re_s = jnp.maximum(u_star * z0u / SF.tpsf_nu_u_a, SF.c_accur_sf)
    lnRe = jnp.log(Re_s)
    mol = Re_s <= SF.Re_z0s_ice_t
    z0t_mol = jnp.minimum(SF.c_z0t_ice_b0t + SF.c_z0t_ice_b1t * lnRe,
                          SF.c_z0t_ice_b0s)
    z0q_mol = jnp.minimum(SF.c_z0q_ice_b0t + SF.c_z0q_ice_b1t * lnRe,
                          SF.c_z0q_ice_b0s)
    z0t_rgh = (SF.c_z0t_ice_b0r + SF.c_z0t_ice_b1r * lnRe
               + SF.c_z0t_ice_b2r * lnRe ** 2)
    z0q_rgh = (SF.c_z0q_ice_b0r + SF.c_z0q_ice_b1r * lnRe
               + SF.c_z0q_ice_b2r * lnRe ** 2)
    z0t = z0u * jnp.exp(jnp.where(mol, z0t_mol, z0t_rgh))
    z0q = z0u * jnp.exp(jnp.where(mol, z0q_mol, z0q_rgh))
    return c_z0u_fetch, u_star_thresh, z0u, z0t, z0q


def momsenlat_ice(height_u, height_tq, fetch, U_a, T_a, q_a, T_s, P_a,
                  h_ice):
    """Ice-branch ``SfcFlx_momsenlat``; returns ``(Qmom, Qsen, Qlat, Qwatvap)``.

    Same structure as :func:`lakematrix.jaxbatch.sfcflx.momsenlat` (the water
    branch, which uses fixed, masked Newton iterations instead of ``while``),
    with the four ice differences of ``SfcFlx_mod.F90:221-612``:

    * the saturation vapour pressure is taken over ice,
    * ``SfcFlx_roughness`` returns the Andreas (2002) ice values,
    * over rough ice the friction velocity is *not* iterated but taken as
      ``kappa*U_a/(log(h/z0u) + psi_u)``,
    * the latent heat of fusion ``tpl_L_f`` is added to ``tpsf_L_evap`` (only
      where ``h_ice >= h_Ice_min_flk``, as in the source).
    """
    wvpres_s = satwvpres_ice(T_s)
    q_s = spechum(wvpres_s, P_a)
    rho_a = rhoair(T_s, q_s, P_a)

    Q_mom_mol = -SF.tpsf_nu_u_a * U_a / height_u
    Q_sen_mol = -SF.tpsf_kappa_t_a * (T_a - T_s) / height_tq
    Q_lat_mol = -SF.tpsf_kappa_q_a * (q_a - q_s) / height_tq

    par = ((T_s - T_a) / T_s * jnp.sqrt(SF.tpsf_kappa_t_a)
           + (q_s - q_a) * SF.tpsf_alpha_q * jnp.sqrt(SF.tpsf_kappa_q_a))
    conv = par > 0.0
    par_c = (par * SF.tpl_grav / SF.tpsf_nu_u_a) ** SF.num_1o3_sf
    Q_sen_con = SF.c_free_conv * jnp.sqrt(SF.tpsf_kappa_t_a) * par_c * (T_s - T_a)
    Q_lat_con = SF.c_free_conv * jnp.sqrt(SF.tpsf_kappa_q_a) * par_c * (q_s - q_a)
    Q_sen_con = jnp.where(conv, Q_sen_con, 0.0)
    Q_lat_con = jnp.where(conv, Q_lat_con, 0.0)

    R_z = height_tq / height_u
    Ri_cr = SF.c_MO_t_stab / SF.c_MO_u_stab ** 2 * R_z
    Ri = (SF.tpl_grav * ((T_a - T_s) / T_s + SF.tpsf_alpha_q * (q_a - q_s))
          / jnp.maximum(U_a, SF.u_wind_min_sf) ** 2)
    Ri = Ri * height_u / SF.Pr_neutral
    low_wind = (U_a < SF.u_wind_min_sf) | (Ri > Ri_cr - SF.c_small_sf)

    # z/L: stable closed form, convective Newton iteration (masked).
    ZoL_stable = jnp.sqrt(1.0 - 4.0 * (SF.c_MO_u_stab - R_z * SF.c_MO_t_stab) * Ri)
    ZoL_stable = (ZoL_stable - 1.0 + 2.0 * SF.c_MO_u_stab * Ri)
    ZoL_stable = ZoL_stable / 2.0 / SF.c_MO_u_stab ** 2 / (Ri_cr - Ri)

    def conv_body(carry):
        us, it = carry
        Fun = (us ** 2 * (SF.c_MO_u_conv * us - 1.0)
               + Ri ** 2 * (1.0 - R_z * SF.c_MO_t_conv * us))
        Fun_prime = (3.0 * SF.c_MO_u_conv * us ** 2 - 2.0 * us
                     - R_z * SF.c_MO_t_conv * Ri ** 2)
        ZoL = us - Fun / Fun_prime
        Delta = jnp.abs(ZoL - us) / jnp.maximum(SF.c_accur_sf, jnp.abs(ZoL + us))
        done = Delta <= SF.c_accur_sf
        return (jnp.where(done, us, ZoL), it + 1)

    us0 = Ri * jnp.maximum(1.0, jnp.sqrt(R_z * SF.c_MO_t_conv / SF.c_MO_u_conv))
    us = us0
    for _ in range(SF.n_iter_max):
        us, _ = conv_body((us, 0))
    Fun = (us ** 2 * (SF.c_MO_u_conv * us - 1.0)
           + Ri ** 2 * (1.0 - R_z * SF.c_MO_t_conv * us))
    Fun_prime = (3.0 * SF.c_MO_u_conv * us ** 2 - 2.0 * us
                 - R_z * SF.c_MO_t_conv * Ri ** 2)
    ZoL_conv = us - Fun / Fun_prime
    ZoL = jnp.where(Ri >= 0.0, ZoL_stable, ZoL_conv)

    # Threshold wind speed, from the roughness at the minimum friction velocity.
    _, u_star_thresh, z0u_sf, z0t_sf, z0q_sf = roughness_ice(
        fetch, U_a, SF.u_star_min_sf)
    _, u_star_thresh, z0u_sf, z0t_sf, z0q_sf = roughness_ice(
        fetch, U_a, u_star_thresh)

    psi_u_stab = SF.c_MO_u_stab * ZoL * (1.0 - jnp.minimum(z0u_sf / height_u, 1.0))
    psi_t = (1.0 - SF.c_MO_u_conv * ZoL) ** SF.c_MO_u_exp
    psi_q = (1.0 - SF.c_MO_u_conv * ZoL
             * jnp.minimum(z0u_sf / height_u, 1.0)) ** SF.c_MO_u_exp
    psi_u_conv = (2.0 * (jnp.arctan(psi_t) - jnp.arctan(psi_q))
                  + 2.0 * jnp.log((1.0 + psi_q) / (1.0 + psi_t))
                  + jnp.log((1.0 + psi_q * psi_q) / (1.0 + psi_t * psi_t)))
    psi_u = jnp.where(ZoL > 0.0, psi_u_stab, psi_u_conv)
    U_a_thresh = u_star_thresh / SF.c_Karman * (jnp.log(height_u / z0u_sf) + psi_u)

    # Friction-velocity iteration; over ice the rough-flow branch is explicit.
    smooth = U_a <= U_a_thresh
    u_star = jnp.maximum(u_star_thresh, SF.u_star_min_sf)
    for _ in range(SF.n_iter_max):
        _, _, z0u_sf, z0t_sf, z0q_sf = roughness_ice(
            fetch, U_a, jnp.where(smooth, jnp.minimum(u_star_thresh, u_star),
                                  jnp.maximum(u_star_thresh, u_star)))
        psi_u_stab = SF.c_MO_u_stab * ZoL * (1.0 - jnp.minimum(z0u_sf / height_u, 1.0))
        psi_t_c = (1.0 - SF.c_MO_u_conv * ZoL) ** SF.c_MO_u_exp
        psi_q_c = (1.0 - SF.c_MO_u_conv * ZoL
                   * jnp.minimum(z0u_sf / height_u, 1.0)) ** SF.c_MO_u_exp
        psi_u_conv = (2.0 * (jnp.arctan(psi_t_c) - jnp.arctan(psi_q_c))
                      + 2.0 * jnp.log((1.0 + psi_q_c) / (1.0 + psi_t_c))
                      + jnp.log((1.0 + psi_q_c * psi_q_c)
                                / (1.0 + psi_t_c * psi_t_c)))
        psi_u = jnp.where(ZoL >= 0.0, psi_u_stab, psi_u_conv)
        psi_t_stab = (SF.c_MO_t_stab * R_z * ZoL
                      * (1.0 - jnp.minimum(z0t_sf / height_tq, 1.0)))
        psi_q_stab = (SF.c_MO_q_stab * R_z * ZoL
                      * (1.0 - jnp.minimum(z0q_sf / height_tq, 1.0)))
        psi_t_u = (1.0 - SF.c_MO_t_conv * R_z * ZoL) ** SF.c_MO_t_exp
        psi_t_s = (1.0 - SF.c_MO_t_conv * R_z * ZoL
                   * jnp.minimum(z0t_sf / height_tq, 1.0)) ** SF.c_MO_t_exp
        psi_q_u = (1.0 - SF.c_MO_q_conv * R_z * ZoL) ** SF.c_MO_q_exp
        psi_q_s = (1.0 - SF.c_MO_q_conv * R_z * ZoL
                   * jnp.minimum(z0q_sf / height_tq, 1.0)) ** SF.c_MO_q_exp
        psi_t_conv = 2.0 * jnp.log((1.0 + psi_t_s) / (1.0 + psi_t_u))
        psi_q_conv = 2.0 * jnp.log((1.0 + psi_q_s) / (1.0 + psi_q_u))
        psi_t_f = jnp.where(ZoL >= 0.0, psi_t_stab, psi_t_conv)
        psi_q_f = jnp.where(ZoL >= 0.0, psi_q_stab, psi_q_conv)
        Fun_stab = jnp.log(height_u / z0u_sf) + psi_u
        Fun_smooth_stab = (Fun_stab + 1.0 + SF.c_MO_u_stab * ZoL
                           * jnp.minimum(z0u_sf / height_u, 1.0)) / SF.c_Karman
        Fun_rough_stab = (Fun_stab - 2.0 - 2.0 * SF.c_MO_u_stab * ZoL
                          * jnp.minimum(z0u_sf / height_u, 1.0)) / SF.c_Karman
        Fun_conv = jnp.log(height_u / z0u_sf) + psi_u
        Fun_smooth_conv = (Fun_conv + 1.0 / psi_q_c) / SF.c_Karman
        Fun_rough_conv = (Fun_conv - 2.0 / psi_q_c) / SF.c_Karman
        Fp = jnp.where(
            smooth,
            jnp.where(ZoL >= 0.0, Fun_smooth_stab, Fun_smooth_conv),
            jnp.where(ZoL >= 0.0, Fun_rough_stab, Fun_rough_conv))
        Fun = jnp.where(ZoL >= 0.0, Fun_stab, Fun_conv) * u_star / SF.c_Karman - U_a
        u_new_smooth = u_star - Fun / Fp
        u_new_rough = (SF.c_Karman * U_a
                       / jnp.maximum(SF.c_small_sf,
                                     jnp.log(height_u / z0u_sf) + psi_u))
        u_new = jnp.where(smooth, u_new_smooth, u_new_rough)
        Delta = jnp.abs((u_new - u_star) / (u_new + u_star))
        u_star = jnp.where(Delta > SF.c_accur_sf, u_new, u_star)

    Q_mom_tur = jnp.where(low_wind, 0.0, -u_star * u_star)
    Q_sen_tur = jnp.where(low_wind, 0.0,
                          -(T_a - T_s) * u_star * SF.c_Karman / SF.Pr_neutral
                          / jnp.maximum(SF.c_small_sf,
                                        jnp.log(height_tq / z0t_sf) + psi_t_f))
    Q_lat_tur = jnp.where(low_wind, 0.0,
                          -(q_a - q_s) * u_star * SF.c_Karman / SF.Sc_neutral
                          / jnp.maximum(SF.c_small_sf,
                                        jnp.log(height_tq / z0q_sf) + psi_q_f))
    Q_mom = jnp.minimum(jnp.minimum(Q_mom_tur, Q_mom_mol), 0.0)
    Q_sen = jnp.where(
        conv, jnp.where(jnp.abs(Q_sen_tur) >= jnp.abs(Q_sen_con),
                        Q_sen_tur, Q_sen_con), Q_sen_tur)
    Q_sen = jnp.where(jnp.abs(Q_sen) < jnp.abs(Q_sen_mol), Q_sen_mol, Q_sen)
    Q_lat = jnp.where(
        conv, jnp.where(jnp.abs(Q_lat_tur) >= jnp.abs(Q_lat_con),
                        Q_lat_tur, Q_lat_con), Q_lat_tur)
    Q_lat = jnp.where(jnp.abs(Q_lat) < jnp.abs(Q_lat_mol), Q_lat_mol, Q_lat)

    Q_mom = Q_mom * rho_a
    Q_sen = Q_sen * rho_a * SF.tpsf_c_a_p
    Q_wat = Q_lat * rho_a
    # Latent heat of sublimation over ice/snow (source: `if (h_ice >= ...)`).
    Lv = SF.tpsf_L_evap + jnp.where(h_ice >= SF.h_Ice_min_flk, SF.tpl_L_f, 0.0)
    Q_lat = Q_wat * Lv
    return Q_mom, Q_sen, Q_lat, Q_wat


def senslatmom_fluxes_ice(Tsurf, fetch, uwind, vwind, u, v, tempair,
                          pressure, humair, zref, h_ice, relhums=_RELHUMS,
                          aM=C.aMagi, bM=C.bMagi, minwind=C.minwind):
    """Ice/snow ``SENSLATMOM_FLUXES`` (``PBLpar = 3``, ``relwind = 1``).

    The batched counterpart of the NumPy
    :func:`lakematrix.physics.surface.senslatmom_fluxes`, with the ice branch
    of ``SfcFlx`` (see :func:`momsenlat_ice`).  It returns the same dict as
    :func:`lakematrix.jaxbatch.surface.senslatmom_fluxes`;
    ``relhums``/``aM``/``bM`` are part of the signature for parity but do not
    enter ``SfcFlx`` (``humsurf``/``TET`` are diagnostics only), exactly as in
    the NumPy version.

    ``h_ice`` is the ice thickness ``l1`` [m].
    """
    urel, vrel = uwind - u, vwind - v
    wr = jnp.maximum(jnp.sqrt(urel ** 2 + vrel ** 2), minwind)
    qmom, qsen, qlat, qwat = momsenlat_ice(zref, zref, fetch, wr,
                                           tempair + C.Kelvin0, humair,
                                           Tsurf + C.Kelvin0, pressure, h_ice)
    tau = -qmom
    return dict(hflux=qsen, Elatent=qlat, tau=tau,
                cdmw=tau / (C.roa0 * wr), xlew=qlat, hw=qsen,
                velfrict=jnp.sqrt(jnp.maximum(tau, 0.0) /
                                  (pressure / (C.Rd * (tempair + C.Kelvin0)))),
                wr=wr)


# ---------------------------------------------------------------------------
# SURF_CHAR
# ---------------------------------------------------------------------------
def surf_char_ice(surftyp, year, month, day, hour, phi, fetch, wind,
                  velfrict_prev):
    """``SURF_CHAR`` ice (2) / snow (3) branch; returns the ``SURF_CHAR`` dict.

    The ice/snow branches of the source are constant tables: they ignore
    ``year``/``month``/``day``/``hour``/``phi``/``fetch``/``wind`` (kept in the
    signature for parity with :func:`lakematrix.physics.surface.surf_char`) and
    do **not** clamp ``velfrict_prev`` to 0.01 the way the water branch does.

    ``surftyp`` may be a static int (validated, ``ValueError`` otherwise) or a
    per-lake array; the latter is resolved with ``jnp.where`` so one batch can
    mix ice and snow lakes.

    Returns ``dict(albedo, albedo_lw, roughness, aM, bM, emissivity, relhums,
    velfrict_prev)``; the floats are the float32-rounded values of
    ``DEFINE_PHYS_CONSTANTS`` (``C.albedoofice`` is already rounded).
    """
    snow = _snow_mask(surftyp)
    albedo = jnp.where(snow, C.albedoofsnow, C.albedoofice)
    albedo_lw = jnp.where(snow, C.albedoofsnow_lw, C.albedoofice_lw)
    roughness = jnp.where(snow, _SNOW_ROUGHNESS, _ICE_ROUGHNESS)
    emissivity = jnp.where(snow, C.emissivityofsnow, C.emissivityofice)
    return dict(albedo=albedo, albedo_lw=albedo_lw, roughness=roughness,
                aM=C.aMagi, bM=C.bMagi, emissivity=emissivity,
                relhums=_RELHUMS, velfrict_prev=jnp.asarray(velfrict_prev))


# ---------------------------------------------------------------------------
# HEATFLUXSURF
# ---------------------------------------------------------------------------
def heatfluxsurf_ice(Ti1, Ti2, lami_v, ci_m_roi_v, l1, dhi0, ddzi, area_int,
                     area_half, dt):
    """``HEATFLUXSURF`` ice branch; returns ``(eflux, dHdt, convect_flux)``.

    ``Ti1`` is the surface-ice temperature before ``T_DIFF``, ``Ti2`` after it;
    ``area_int``/``area_half`` are the ``BATHYMICE`` areas with
    ``area_int[..., 1] = bathymice(1)%area_int`` (the *top ice layer* areas -
    the source uses ``bathymice(1)``, not ``(2)``).  ``convect_flux`` is
    ``0.0`` in the source.
    """
    dt_inv = 1.0 / dt
    eflux = (-0.5 * lami_v[..., 1] / area_int[..., 1] * area_half[..., 1]
             * (Ti2[..., 2] + Ti1[..., 2] - Ti2[..., 1] - Ti1[..., 1])
             / (ddzi[..., 1] * l1))
    dHdt = 0.5 * dt_inv * ci_m_roi_v[..., 1] * (
        ddzi[..., 1] * l1 * (Ti2[..., 1] - Ti1[..., 1])
        + 0.5 * dhi0 * (Ti2[..., 2] + Ti1[..., 2] - Ti2[..., 1] - Ti1[..., 1]))
    return eflux, dHdt, 0.0


def heatfluxsurf_snow(Tsn, T, cs, dens, dz, lams, itop, dt):
    """``HEATFLUXSURF`` snow branch; returns ``(eflux, dHdt, convect_flux)``.

    ``Tsn`` is the snow surface temperature after ``T_DIFF``, ``T`` the
    previous-step snow temperature (``SNOWSOIL_MOD`` keeps them as separate
    arrays); ``itop`` is the 1-based index of the top snow layer of the
    current pack (a Python int or a per-lake ``(L,)`` array) and must satisfy
    ``1 <= itop <= ms - 1``, i.e. ``lams[itop + 1]`` must be in bounds as in
    the reference (where ``snow_cond_heat_coef`` sets ``itop = ms-2``).
    ``convect_flux`` is ``0.0`` in the source.
    """
    _check_itop(itop, lams)
    dt_inv = 1.0 / dt
    itop = jnp.asarray(itop, dtype=jnp.int32)
    itop1 = itop + 1
    eflux = (-(_gather_last(lams, itop) + _gather_last(lams, itop1)) * 0.25
             * (_gather_last(Tsn, itop1) + _gather_last(T, itop1)
                - _gather_last(Tsn, itop) - _gather_last(T, itop))
             / _gather_last(dz, itop))
    dHdt = (0.5 * dt_inv * _gather_last(cs, itop) * _gather_last(dens, itop)
            * _gather_last(dz, itop)
            * (_gather_last(Tsn, itop) - _gather_last(T, itop)))
    return eflux, dHdt, 0.0


# ---------------------------------------------------------------------------
# RADBALANCE
# ---------------------------------------------------------------------------
def radbalance_ice(Tsurf, radice_integr, surftyp, *, longwave, shortwave,
                   albedo, albedo_lw, emissivity):
    """``RADBALANCE`` ice/snow branch; returns the ``radbalance`` 5-tuple.

    ``ifrad = 1`` and ``surftyp`` of 2 (ice) or 3 (snow), either static or
    per-lake.  For ice the absorbed shortwave is the ice radiation integral
    (``radice_integr[..., 0] - radice_integr[..., 1]``); for snow the source
    sets ``shortbal = 0`` - snow shortwave is handled by the ice below it.

    ``Radbal_surf`` is ``None`` for both surface types (only the water branch
    with ``skin == 0`` fills it), so the caller must fall back to ``Radbal``
    exactly like :func:`lakematrix.jaxbatch.surface.heatbalance` does.
    """
    snow = _snow_mask(surftyp)
    surfrad = emissivity * C.sigma * (Tsurf + C.Kelvin0) ** 4
    if radice_integr is None:
        shortbal_ice = 0.0
    else:
        shortbal_ice = radice_integr[..., 0] - radice_integr[..., 1]
    radbal_ice = shortbal_ice + longwave * (1.0 - albedo_lw) - surfrad
    radbal_snow = longwave * (1.0 - albedo_lw) - surfrad
    radbal = jnp.where(snow, radbal_snow, radbal_ice)
    shortbal = jnp.where(snow, 0.0, shortbal_ice)
    swup = shortwave * (1.0 - albedo)
    return radbal, None, shortbal, surfrad, swup


# ---------------------------------------------------------------------------
# HEATBALANCE
# ---------------------------------------------------------------------------
def heatbalance_ice(Tsurf, ctx, t_diff_fn):
    """``HEATBALANCE`` for an ice/snow surface; returns ``(HB, out)``.

    ``Tsurf`` is the surface temperature [degC] the solver is testing (scalar
    or ``(L,)``).  ``ctx`` and ``t_diff_fn`` are described in the module
    docstring; see :class:`lakematrix.physics.surface.IceSurfaceContext` for
    the NumPy equivalent of ``ctx``.

    The sequence is exactly the NumPy reference
    :func:`lakematrix.physics.surface.heatbalance_ice`::

        t_diff_fn(1, Tsurf, ...)            -> new ice / snow temperatures
        HEATFLUXSURF   (snow or ice branch) -> eflux, dHdt, convect_flux
        SENSLATMOM_FLUXES (ice SfcFlx)      -> hflux, Elatent, cdmw, velfrict
        RADBALANCE     (surftyp 2/3)        -> Radbal, eflux0
        HEATBALANCE    (skin == 0)          -> HB

    ``h_ice`` of ``SfcFlx`` is ``l1`` (the ice thickness), as in the NumPy
    reference; the ``hflux``/``Elatent`` path is
    :func:`senslatmom_fluxes_ice`, i.e. the *ice* branch of ``SfcFlx`` (see
    :func:`momsenlat_ice`).

    Both ``HEATFLUXSURF`` branches are evaluated and selected with
    ``jnp.where(ctx["snow"] == 1, ...)``; a lake without a snow pack therefore
    masks the snow branch and vice versa.
    """
    state = ctx["state"]
    st = t_diff_fn(1, Tsurf, ctx["dt"], ctx["snow"], ctx.get("ice", 1),
                   ctx.get("water", 1), ctx.get("deepice", 0),
                   ctx["isoilcol"], False, state)

    snow = jnp.asarray(ctx["snow"]) == 1
    itop = _pick(st, ctx, "itop")

    ef_ice, dh_ice, cv_ice = heatfluxsurf_ice(
        _pick(st, ctx, "Ti1"), _pick(st, ctx, "Ti2"),
        _pick(st, ctx, "lami_v"), _pick(st, ctx, "ci_m_roi_v"),
        _pick(st, ctx, "l1"), _pick(st, ctx, "dhi0"),
        _pick(st, ctx, "ddzi"), _pick(st, ctx, "area_int"),
        _pick(st, ctx, "area_half"), ctx["dt"])
    ef_snow, dh_snow, cv_snow = heatfluxsurf_snow(
        _pick(st, ctx, "Tsn"), _pick(st, ctx, "T"),
        _pick(st, ctx, "cs"), _pick(st, ctx, "dens"),
        _pick(st, ctx, "dz"), _pick(st, ctx, "lams"), itop, ctx["dt"])
    eflux = jnp.where(snow, ef_snow, ef_ice)
    dHdt = jnp.where(snow, dh_snow, dh_ice)
    convect_flux = jnp.where(snow, cv_snow, cv_ice)

    slm = senslatmom_fluxes_ice(
        Tsurf, ctx["fetch"], ctx["uwind"], ctx["vwind"], ctx["u"], ctx["v"],
        ctx["tempair"], ctx["pressure"], ctx["humair"], ctx["zref"],
        _pick(st, ctx, "l1"), relhums=ctx.get("relhums", _RELHUMS),
        aM=ctx.get("aM", C.aMagi), bM=ctx.get("bM", C.bMagi))

    rb, rbs, sb, sr, swup = radbalance_ice(
        Tsurf, ctx.get("radice_integr"), jnp.where(snow, SNOW, ICE),
        longwave=ctx["longwave"], shortwave=ctx["shortwave"],
        albedo=ctx["albedo"], albedo_lw=ctx["albedo_lw"],
        emissivity=ctx["emissivity"])
    HB, eflux0, eflux0_kinem = heatbalance(
        dHdt, rb, rbs, eflux, convect_flux, slm["hflux"], slm["Elatent"])
    return HB, dict(ice=st, eflux=eflux, dHdt=dHdt, convect_flux=convect_flux,
                    hflux=slm["hflux"], Elatent=slm["Elatent"], Radbal=rb,
                    Radbal_surf=rbs, eflux0=eflux0,
                    eflux0_kinem=eflux0_kinem, cdmw=slm["cdmw"],
                    velfrict=slm["velfrict"])


jitted_heatbalance_ice = jax.jit(heatbalance_ice,
                                 static_argnames=("t_diff_fn",))
