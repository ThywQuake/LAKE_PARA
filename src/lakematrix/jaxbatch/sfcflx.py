"""JAX (batched) port of the open-water SfcFlx path.

Only the water branch (``h_ice < h_Ice_min_flk``) used by case 1 is ported;
the ice/snow branches stay in the NumPy implementation.  Branches on
per-lake scalars are written with ``jnp.where`` and the two Newton iterations
use fixed, masked updates so the function is traceable and vmappable.
"""

import jax.numpy as jnp

from ..physics import sfcflx as S


def roughness(fetch, U_a, u_star):
    """``SfcFlx_roughness`` water branch (batched scalars)."""
    c_z0u_fetch = jnp.maximum(U_a, S.u_wind_min_sf) ** 2 / S.tpl_grav / fetch
    c_z0u_fetch = S.c_z0u_rough + S.c_z0u_ftch_f * c_z0u_fetch ** S.c_z0u_ftch_ex
    c_z0u_fetch = jnp.minimum(c_z0u_fetch, S.c_z0u_rough_L)
    u_star_thresh = ((S.c_z0u_smooth / c_z0u_fetch * S.tpl_grav
                      * S.tpsf_nu_u_a) ** S.num_1o3_sf)
    Re_s = u_star ** 3 / S.tpsf_nu_u_a / S.tpl_grav
    Re_s_thresh = S.c_z0u_smooth / c_z0u_fetch
    z0u_smooth = S.c_z0u_smooth * S.tpsf_nu_u_a / u_star
    z0u_rough = c_z0u_fetch * u_star * u_star / S.tpl_grav
    z0u = jnp.where(Re_s <= Re_s_thresh, z0u_smooth, z0u_rough)
    z0q = c_z0u_fetch * jnp.maximum(Re_s, Re_s_thresh)
    z0t = S.c_z0t_rough_1 * z0q ** S.c_z0t_rough_3 - S.c_z0t_rough_2
    z0q = S.c_z0q_rough_1 * z0q ** S.c_z0q_rough_3 - S.c_z0q_rough_2
    z0t = z0u * jnp.exp(-S.c_Karman / S.Pr_neutral * z0t)
    z0q = z0u * jnp.exp(-S.c_Karman / S.Sc_neutral * z0q)
    return c_z0u_fetch, u_star_thresh, z0u, z0t, z0q


def satwvpres(T):
    return S.b1_vap * jnp.exp(S.b2w_vap * (T - S.b3_vap) / (T - S.b4w_vap))


def spechum(wvpres, P):
    return S.tpsf_Rd_o_Rv * wvpres / (P - (1.0 - S.tpsf_Rd_o_Rv) * wvpres)


def rhoair(T, q, P):
    return P / S.tpsf_R_dryair / T / (1.0 + (1.0 / S.tpsf_Rd_o_Rv - 1.0) * q)


def momsenlat(height_u, height_tq, fetch, U_a, T_a, q_a, T_s, P_a):
    """Water ``SfcFlx_momsenlat``; returns (Qmom, Qsen, Qlat, Qwatvap)."""
    wvpres_s = satwvpres(T_s)
    q_s = spechum(wvpres_s, P_a)
    rho_a = rhoair(T_s, q_s, P_a)

    Q_mom_mol = -S.tpsf_nu_u_a * U_a / height_u
    Q_sen_mol = -S.tpsf_kappa_t_a * (T_a - T_s) / height_tq
    Q_lat_mol = -S.tpsf_kappa_q_a * (q_a - q_s) / height_tq

    par = ((T_s - T_a) / T_s * jnp.sqrt(S.tpsf_kappa_t_a)
           + (q_s - q_a) * S.tpsf_alpha_q * jnp.sqrt(S.tpsf_kappa_q_a))
    conv = par > 0.0
    par_c = (par * S.tpl_grav / S.tpsf_nu_u_a) ** S.num_1o3_sf
    Q_sen_con = S.c_free_conv * jnp.sqrt(S.tpsf_kappa_t_a) * par_c * (T_s - T_a)
    Q_lat_con = S.c_free_conv * jnp.sqrt(S.tpsf_kappa_q_a) * par_c * (q_s - q_a)
    Q_sen_con = jnp.where(conv, Q_sen_con, 0.0)
    Q_lat_con = jnp.where(conv, Q_lat_con, 0.0)

    R_z = height_tq / height_u
    Ri_cr = S.c_MO_t_stab / S.c_MO_u_stab ** 2 * R_z
    Ri = (S.tpl_grav * ((T_a - T_s) / T_s + S.tpsf_alpha_q * (q_a - q_s))
          / jnp.maximum(U_a, S.u_wind_min_sf) ** 2)
    Ri = Ri * height_u / S.Pr_neutral
    low_wind = (U_a < S.u_wind_min_sf) | (Ri > Ri_cr - S.c_small_sf)

    # z/L: stable closed form, convective Newton iteration (masked).
    ZoL_stable = jnp.sqrt(1.0 - 4.0 * (S.c_MO_u_stab - R_z * S.c_MO_t_stab) * Ri)
    ZoL_stable = (ZoL_stable - 1.0 + 2.0 * S.c_MO_u_stab * Ri)
    ZoL_stable = ZoL_stable / 2.0 / S.c_MO_u_stab ** 2 / (Ri_cr - Ri)

    def conv_body(carry):
        us, it = carry
        Fun = (us ** 2 * (S.c_MO_u_conv * us - 1.0)
               + Ri ** 2 * (1.0 - R_z * S.c_MO_t_conv * us))
        Fun_prime = (3.0 * S.c_MO_u_conv * us ** 2 - 2.0 * us
                     - R_z * S.c_MO_t_conv * Ri ** 2)
        ZoL = us - Fun / Fun_prime
        Delta = jnp.abs(ZoL - us) / jnp.maximum(S.c_accur_sf, jnp.abs(ZoL + us))
        done = Delta <= S.c_accur_sf
        return (jnp.where(done, us, ZoL), it + 1)

    us0 = Ri * jnp.maximum(1.0, jnp.sqrt(R_z * S.c_MO_t_conv / S.c_MO_u_conv))
    us = us0
    for _ in range(S.n_iter_max):
        us, _ = conv_body((us, 0))
    # reconstruct ZoL from the final iteration
    Fun = (us ** 2 * (S.c_MO_u_conv * us - 1.0)
           + Ri ** 2 * (1.0 - R_z * S.c_MO_t_conv * us))
    Fun_prime = (3.0 * S.c_MO_u_conv * us ** 2 - 2.0 * us
                 - R_z * S.c_MO_t_conv * Ri ** 2)
    ZoL_conv = us - Fun / Fun_prime
    ZoL = jnp.where(Ri >= 0.0, ZoL_stable, ZoL_conv)

    # First roughness call at the minimum friction velocity.
    _, u_star_thresh, z0u_sf, z0t_sf, z0q_sf = roughness(fetch, U_a, S.u_star_min_sf)
    _, u_star_thresh, z0u_sf, z0t_sf, z0q_sf = roughness(fetch, U_a, u_star_thresh)

    psi_u_stab = S.c_MO_u_stab * ZoL * (1.0 - jnp.minimum(z0u_sf / height_u, 1.0))
    psi_t = (1.0 - S.c_MO_u_conv * ZoL) ** S.c_MO_u_exp
    psi_q = (1.0 - S.c_MO_u_conv * ZoL * jnp.minimum(z0u_sf / height_u, 1.0)) ** S.c_MO_u_exp
    psi_u_conv = (2.0 * (jnp.arctan(psi_t) - jnp.arctan(psi_q))
                  + 2.0 * jnp.log((1.0 + psi_q) / (1.0 + psi_t))
                  + jnp.log((1.0 + psi_q * psi_q) / (1.0 + psi_t * psi_t)))
    psi_u = jnp.where(ZoL > 0.0, psi_u_stab, psi_u_conv)
    U_a_thresh = u_star_thresh / S.c_Karman * (jnp.log(height_u / z0u_sf) + psi_u)

    # Friction-velocity Newton iteration, smooth/rough selected with where.
    smooth = U_a <= U_a_thresh
    u_star = jnp.maximum(u_star_thresh, S.u_star_min_sf)
    for _ in range(S.n_iter_max):
        _, _, z0u_sf, z0t_sf, z0q_sf = roughness(
            fetch, U_a, jnp.where(smooth, jnp.minimum(u_star_thresh, u_star),
                                  jnp.maximum(u_star_thresh, u_star)))
        psi_u_stab = S.c_MO_u_stab * ZoL * (1.0 - jnp.minimum(z0u_sf / height_u, 1.0))
        psi_t_c = (1.0 - S.c_MO_u_conv * ZoL) ** S.c_MO_u_exp
        psi_q_c = (1.0 - S.c_MO_u_conv * ZoL * jnp.minimum(z0u_sf / height_u, 1.0)) ** S.c_MO_u_exp
        psi_u_conv = (2.0 * (jnp.arctan(psi_t_c) - jnp.arctan(psi_q_c))
                      + 2.0 * jnp.log((1.0 + psi_q_c) / (1.0 + psi_t_c))
                      + jnp.log((1.0 + psi_q_c * psi_q_c) / (1.0 + psi_t_c * psi_t_c)))
        psi_u = jnp.where(ZoL >= 0.0, psi_u_stab, psi_u_conv)
        psi_t_stab = S.c_MO_t_stab * R_z * ZoL * (1.0 - jnp.minimum(z0t_sf / height_tq, 1.0))
        psi_q_stab = S.c_MO_q_stab * R_z * ZoL * (1.0 - jnp.minimum(z0q_sf / height_tq, 1.0))
        psi_t_u = (1.0 - S.c_MO_t_conv * R_z * ZoL) ** S.c_MO_t_exp
        psi_t_s = (1.0 - S.c_MO_t_conv * R_z * ZoL
                   * jnp.minimum(z0t_sf / height_tq, 1.0)) ** S.c_MO_t_exp
        psi_q_u = (1.0 - S.c_MO_q_conv * R_z * ZoL) ** S.c_MO_q_exp
        psi_q_s = (1.0 - S.c_MO_q_conv * R_z * ZoL
                   * jnp.minimum(z0q_sf / height_tq, 1.0)) ** S.c_MO_q_exp
        psi_t_conv = 2.0 * jnp.log((1.0 + psi_t_s) / (1.0 + psi_t_u))
        psi_q_conv = 2.0 * jnp.log((1.0 + psi_q_s) / (1.0 + psi_q_u))
        psi_t_f = jnp.where(ZoL >= 0.0, psi_t_stab, psi_t_conv)
        psi_q_f = jnp.where(ZoL >= 0.0, psi_q_stab, psi_q_conv)
        Fun_stab = jnp.log(height_u / z0u_sf) + psi_u
        Fun_smooth_stab = (Fun_stab + 1.0 + S.c_MO_u_stab * ZoL
                           * jnp.minimum(z0u_sf / height_u, 1.0)) / S.c_Karman
        Fun_rough_stab = (Fun_stab - 2.0 - 2.0 * S.c_MO_u_stab * ZoL
                          * jnp.minimum(z0u_sf / height_u, 1.0)) / S.c_Karman
        Fun_conv = jnp.log(height_u / z0u_sf) + psi_u
        Fun_smooth_conv = (Fun_conv + 1.0 / psi_q_c) / S.c_Karman
        Fun_rough_conv = (Fun_conv - 2.0 / psi_q_c) / S.c_Karman
        Fp = jnp.where(
            smooth,
            jnp.where(ZoL >= 0.0, Fun_smooth_stab, Fun_smooth_conv),
            jnp.where(ZoL >= 0.0, Fun_rough_stab, Fun_rough_conv))
        Fun = jnp.where(ZoL >= 0.0, Fun_stab, Fun_conv) * u_star / S.c_Karman - U_a
        u_new = u_star - Fun / Fp
        Delta = jnp.abs((u_new - u_star) / (u_new + u_star))
        u_star = jnp.where(Delta > S.c_accur_sf, u_new, u_star)
    Q_mom_tur = jnp.where(low_wind, 0.0, -u_star * u_star)
    Q_sen_tur = jnp.where(low_wind, 0.0,
                          -(T_a - T_s) * u_star * S.c_Karman / S.Pr_neutral
                          / jnp.maximum(S.c_small_sf,
                                        jnp.log(height_tq / z0t_sf) + psi_t_f))
    Q_lat_tur = jnp.where(low_wind, 0.0,
                          -(q_a - q_s) * u_star * S.c_Karman / S.Sc_neutral
                          / jnp.maximum(S.c_small_sf,
                                        jnp.log(height_tq / z0q_sf) + psi_q_f))
    Q_mom = jnp.minimum(jnp.minimum(Q_mom_tur, Q_mom_mol), 0.0)
    if True:
        Q_sen = jnp.where(
            conv, jnp.where(jnp.abs(Q_sen_tur) >= jnp.abs(Q_sen_con),
                            Q_sen_tur, Q_sen_con), Q_sen_tur)
        Q_sen = jnp.where(jnp.abs(Q_sen) < jnp.abs(Q_sen_mol), Q_sen_mol, Q_sen)
        Q_lat = jnp.where(
            conv, jnp.where(jnp.abs(Q_lat_tur) >= jnp.abs(Q_lat_con),
                            Q_lat_tur, Q_lat_con), Q_lat_tur)
        Q_lat = jnp.where(jnp.abs(Q_lat) < jnp.abs(Q_lat_mol), Q_lat_mol, Q_lat)
    Q_mom = Q_mom * rho_a
    Q_sen = Q_sen * rho_a * S.tpsf_c_a_p
    Q_wat = Q_lat * rho_a
    Q_lat = Q_wat * S.tpsf_L_evap
    return Q_mom, Q_sen, Q_lat, Q_wat
