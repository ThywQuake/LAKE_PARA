"""Surface-layer momentum and heat fluxes over lakes (Fortran `MODULE SfcFlx`).

Faithful NumPy port of `lakebatch/core/source/shared/SfcFlx_mod.F90`
(LAKE 3.2, SfcFlx 1.00, D. Mironov, DWD).  The procedures inlined from the
`SfcFlx_*.incf` files are reproduced formula by formula, branch by branch,
including both Newton iterations and their `DO WHILE` convergence criteria:

    SfcFlx_momsenlat   SfcFlx_mod.F90:221-612   (SfcFlx_momsenlat.incf)
    SfcFlx_rhoair      SfcFlx_mod.F90:623-687   (SfcFlx_rhoair.incf)
    SfcFlx_roughness   SfcFlx_mod.F90:700-844   (SfcFlx_roughness.incf)
    SfcFlx_satwvpres   SfcFlx_mod.F90:855-936   (SfcFlx_satwvpres.incf)
    SfcFlx_spechum     SfcFlx_mod.F90:949-1012  (SfcFlx_spechum.incf)
    SfcFlx_lwradwsfc   SfcFlx_mod.F90:1019-1081 (SfcFlx_lwradwsfc.incf)

Notes on the port
-----------------
* `ireals` is `SELECTED_REAL_KIND(12, 200)`, i.e. IEEE float64; `iintegers`
  is the default integer.  No literal in the ported procedures is written
  with `_4`, `dble` or `real(4)`, so every constant here is float64
  (`np.float64`) and no float32 constant is introduced.
* The FLake parameters `tpl_grav`, `tpl_L_f` and `h_Ice_min_flk` are copied
  from `flake_parameters_mod.F90` under their Fortran names instead of being
  taken from `lakematrix.constants`: the LAKE-core constant `g = 9.814`
  differs from FLake's `tpl_grav = 9.81`, and using it would silently change
  the fluxes.  (`tpl_T_f`, `tpl_rho_w_r` and `tpl_c_w` are USEd by the
  Fortran module but not referenced by the procedures ported here.)
* The Fortran module keeps the last results in the module variables
  `u_star_a_sf`, `Q_mom_a_sf`, `Q_sens_a_sf`, `Q_lat_a_sf`, `Q_watvap_a_sf`.
  Those side effects are dropped: every function below is pure.
* `SfcFlx_roughness` is a SUBROUTINE with five INTENT(OUT) arguments, so it
  returns the tuple `(c_z0u_fetch, u_star_thresh, z0u, z0t, z0q)` in that
  order.
* All fluxes are positive when directed upwards, as in the Fortran code.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# FLake parameters (flake_parameters_mod.F90:91,103,113), as USEd by SfcFlx.
# ---------------------------------------------------------------------------
tpl_grav = np.float64(9.81)        # Acceleration due to gravity [m s^-2]
tpl_L_f = np.float64(3.3e05)       # Latent heat of fusion [J kg^-1]
h_Ice_min_flk = np.float64(1.0e-09)  # Minimum ice thickness [m]

# ---------------------------------------------------------------------------
# Dimensionless constants in the Monin-Obukhov similarity relations and in the
# expressions for the roughness lengths, SfcFlx_mod.F90:86-127.
# ---------------------------------------------------------------------------
c_Karman = np.float64(0.40)          # The von Karman constant
Pr_neutral = np.float64(1.0)         # Turbulent Prandtl number at neutral static stability
Sc_neutral = np.float64(1.0)         # Turbulent Schmidt number at neutral static stability
c_MO_u_stab = np.float64(5.0)        # Constant of the MO theory (wind, stable stratification)
c_MO_t_stab = np.float64(5.0)        # Constant of the MO theory (temperature, stable stratification)
c_MO_q_stab = np.float64(5.0)        # Constant of the MO theory (humidity, stable stratification)
c_MO_u_conv = np.float64(15.0)       # Constant of the MO theory (wind, convection)
c_MO_t_conv = np.float64(15.0)       # Constant of the MO theory (temperature, convection)
c_MO_q_conv = np.float64(15.0)       # Constant of the MO theory (humidity, convection)
c_MO_u_exp = np.float64(0.25)        # Constant of the MO theory (wind, exponent)
c_MO_t_exp = np.float64(0.5)         # Constant of the MO theory (temperature, exponent)
c_MO_q_exp = np.float64(0.5)         # Constant of the MO theory (humidity, exponent)
z0u_ice_rough = np.float64(1.0e-03)  # Aerodynamic roughness of the ice surface [m] (rough flow)
c_z0u_smooth = np.float64(0.1)       # Constant in the expression for z0u (smooth flow)
c_z0u_rough = np.float64(1.23e-02)   # The Charnock constant in the expression for z0u (rough flow)
c_z0u_rough_L = np.float64(1.00e-01)  # An increased Charnock constant (used as the upper limit)
c_z0u_ftch_f = np.float64(0.70)      # Factor in the expression for fetch-dependent Charnock parameter
c_z0u_ftch_ex = np.float64(0.3333333)  # Exponent in the expression for fetch-dependent Charnock parameter
c_z0t_rough_1 = np.float64(4.0)      # Constant in the expression for z0t (factor)
c_z0t_rough_2 = np.float64(3.2)      # Constant in the expression for z0t (factor)
c_z0t_rough_3 = np.float64(0.5)      # Constant in the expression for z0t (exponent)
c_z0q_rough_1 = np.float64(4.0)      # Constant in the expression for z0q (factor)
c_z0q_rough_2 = np.float64(4.2)      # Constant in the expression for z0q (factor)
c_z0q_rough_3 = np.float64(0.5)      # Constant in the expression for z0q (exponent)
c_z0t_ice_b0s = np.float64(1.250)    # Constant in the expression for z0t over ice
c_z0t_ice_b0t = np.float64(0.149)    # Constant in the expression for z0t over ice
c_z0t_ice_b1t = np.float64(-0.550)   # Constant in the expression for z0t over ice
c_z0t_ice_b0r = np.float64(0.317)    # Constant in the expression for z0t over ice
c_z0t_ice_b1r = np.float64(-0.565)   # Constant in the expression for z0t over ice
c_z0t_ice_b2r = np.float64(-0.183)   # Constant in the expression for z0t over ice
c_z0q_ice_b0s = np.float64(1.610)    # Constant in the expression for z0q over ice
c_z0q_ice_b0t = np.float64(0.351)    # Constant in the expression for z0q over ice
c_z0q_ice_b1t = np.float64(-0.628)   # Constant in the expression for z0q over ice
c_z0q_ice_b0r = np.float64(0.396)    # Constant in the expression for z0q over ice
c_z0q_ice_b1r = np.float64(-0.512)   # Constant in the expression for z0q over ice
c_z0q_ice_b2r = np.float64(-0.180)   # Constant in the expression for z0q over ice
Re_z0s_ice_t = np.float64(2.5)       # Threshold surface Reynolds number for z0t,z0q over ice (Andreas 2002)
Re_z0u_thresh = np.float64(0.1)      # Threshold roughness Reynolds number [not used]

# ---------------------------------------------------------------------------
# Dimensionless constants, SfcFlx_mod.F90:130-135.
# ---------------------------------------------------------------------------
c_free_conv = np.float64(0.14)       # Constant in the expressions for fluxes in free convection
c_lwrad_emis = np.float64(0.99)      # Surface emissivity with respect to the long-wave radiation

# ---------------------------------------------------------------------------
# Thermodynamic parameters and derived parameters, SfcFlx_mod.F90:138-155.
# ---------------------------------------------------------------------------
tpsf_C_StefBoltz = np.float64(5.67e-08)   # The Stefan-Boltzmann constant [W m^-2 K^-4]
tpsf_R_dryair = np.float64(2.8705e+02)    # Gas constant for dry air [J kg^-1 K^-1]
tpsf_R_watvap = np.float64(4.6151e+02)    # Gas constant for water vapour [J kg^-1 K^-1]
tpsf_c_a_p = np.float64(1.005e+03)        # Specific heat of air at constant pressure [J kg^-1 K^-1]
tpsf_L_evap = np.float64(2.501e+06)       # Specific heat of evaporation [J kg^-1]
tpsf_nu_u_a = np.float64(1.50e-05)        # Kinematic molecular viscosity of air [m^2 s^-1]
tpsf_kappa_t_a = np.float64(2.20e-05)     # Molecular temperature conductivity of air [m^2 s^-1]
tpsf_kappa_q_a = np.float64(2.40e-05)     # Molecular diffusivity of air for water vapour [m^2 s^-1]

tpsf_Rd_o_Rv = tpsf_R_dryair / tpsf_R_watvap          # Ratio of gas constants (Rd/Rv)
tpsf_alpha_q = (np.float64(1.0) - tpsf_Rd_o_Rv) / tpsf_Rd_o_Rv  # Dimensionless ratio

P_a_ref = np.float64(1.0e+05)        # Reference pressure [N m^-2]

# ---------------------------------------------------------------------------
# Security constants, SfcFlx_mod.F90:183-187.
# ---------------------------------------------------------------------------
u_wind_min_sf = np.float64(1.0e-02)  # Minimum wind speed [m s^-1]
u_star_min_sf = np.float64(1.0e-04)  # Minimum value of friction velocity [m s^-1]
c_accur_sf = np.float64(1.0e-07)     # A small number (accuracy)
c_small_sf = np.float64(1.0e-04)     # A small number (used to compute fluxes)

# ---------------------------------------------------------------------------
# Useful constants, SfcFlx_mod.F90:190-191.
# ---------------------------------------------------------------------------
num_1o3_sf = np.float64(1.0) / np.float64(3.0)  # 1/3

# ---------------------------------------------------------------------------
# Local PARAMETERs of SfcFlx_satwvpres, SfcFlx_mod.F90:912-918.
# ---------------------------------------------------------------------------
b1_vap = np.float64(610.78)       # Coefficient [N m^-2]
b3_vap = np.float64(273.16)       # Triple point [K]
b2w_vap = np.float64(17.2693882)  # Coefficient (water)
b2i_vap = np.float64(21.8745584)  # Coefficient (ice)
b4w_vap = np.float64(35.86)       # Coefficient (temperature) [K]
b4i_vap = np.float64(7.66)        # Coefficient (temperature) [K]

# ---------------------------------------------------------------------------
# Local PARAMETER of SfcFlx_momsenlat, SfcFlx_mod.F90:301-302.
# ---------------------------------------------------------------------------
n_iter_max = 24                   # Maximum number of iterations


def SfcFlx_satwvpres(T, h_ice):
    """Saturation water vapour pressure over water or ice [N m^-2].

    Fortran: `SfcFlx_satwvpres`, SfcFlx_mod.F90:855-936.
    Water surface when ``h_ice < h_Ice_min_flk``, ice surface otherwise.
    """
    if h_ice < h_Ice_min_flk:  # Water surface
        return b1_vap * math.exp(b2w_vap * (T - b3_vap) / (T - b4w_vap))
    # Ice surface
    return b1_vap * math.exp(b2i_vap * (T - b3_vap) / (T - b4i_vap))


def SfcFlx_spechum(wvpres, P):
    """Specific humidity as function of water vapour pressure and air pressure.

    Fortran: `SfcFlx_spechum`, SfcFlx_mod.F90:949-1012.
    """
    return tpsf_Rd_o_Rv * wvpres / (P - (np.float64(1.0) - tpsf_Rd_o_Rv) * wvpres)


def SfcFlx_rhoair(T, q, P):
    """Air density [kg m^-3] from temperature [K], specific humidity and pressure.

    Fortran: `SfcFlx_rhoair`, SfcFlx_mod.F90:623-687.
    """
    return P / tpsf_R_dryair / T / (np.float64(1.0) + (np.float64(1.0) / tpsf_Rd_o_Rv - np.float64(1.0)) * q)


def SfcFlx_lwradwsfc(T):
    """Surface long-wave radiation flux [W m^-2] as function of temperature [K].

    Fortran: `SfcFlx_lwradwsfc`, SfcFlx_mod.F90:1019-1081.
    """
    return c_lwrad_emis * tpsf_C_StefBoltz * T ** 4


def SfcFlx_roughness(fetch, U_a, u_star, h_ice):
    """Roughness lengths over a water or ice/snow surface.

    Fortran: `SfcFlx_roughness`, SfcFlx_mod.F90:700-844.  Returns the five
    INTENT(OUT) arguments in order:
    ``(c_z0u_fetch, u_star_thresh, z0u, z0t, z0q)`` -- fetch-dependent
    Charnock parameter, threshold friction velocity [m s^-1], and the
    roughness lengths for wind velocity, potential temperature and specific
    humidity [m].
    """
    if h_ice < h_Ice_min_flk:  # Water surface
        # The Charnock parameter as dependent on dimensionless fetch
        # (the inverse dimensionless fetch is formed first, as in the Fortran)
        c_z0u_fetch = max(U_a, u_wind_min_sf) ** 2 / tpl_grav / fetch
        c_z0u_fetch = c_z0u_rough + c_z0u_ftch_f * c_z0u_fetch ** c_z0u_ftch_ex
        c_z0u_fetch = min(c_z0u_fetch, c_z0u_rough_L)  # Limit Charnock parameter

        # Threshold value of friction velocity
        u_star_thresh = (c_z0u_smooth / c_z0u_fetch * tpl_grav * tpsf_nu_u_a
                         ) ** num_1o3_sf

        # Surface Reynolds number and its threshold value
        Re_s = u_star ** 3 / tpsf_nu_u_a / tpl_grav
        Re_s_thresh = c_z0u_smooth / c_z0u_fetch

        # Aerodynamic roughness
        if Re_s <= Re_s_thresh:
            z0u = c_z0u_smooth * tpsf_nu_u_a / u_star   # Smooth flow
        else:
            z0u = c_z0u_fetch * u_star * u_star / tpl_grav  # Rough flow

        # Roughness for scalars
        z0q = c_z0u_fetch * max(Re_s, Re_s_thresh)
        z0t = c_z0t_rough_1 * z0q ** c_z0t_rough_3 - c_z0t_rough_2
        z0q = c_z0q_rough_1 * z0q ** c_z0q_rough_3 - c_z0q_rough_2
        z0t = z0u * np.exp(-c_Karman / Pr_neutral * z0t)
        z0q = z0u * np.exp(-c_Karman / Sc_neutral * z0q)

    else:  # Ice surface
        # The Charnock parameter is not used over ice, formally set
        # "c_z0u_fetch" to its minimum value
        c_z0u_fetch = c_z0u_rough

        # Threshold value of friction velocity
        u_star_thresh = c_z0u_smooth * tpsf_nu_u_a / z0u_ice_rough

        # Aerodynamic roughness
        z0u = max(z0u_ice_rough, c_z0u_smooth * tpsf_nu_u_a / u_star)

        # Roughness Reynolds number
        Re_s = max(u_star * z0u / tpsf_nu_u_a, c_accur_sf)

        # Roughness for scalars
        if Re_s <= Re_z0s_ice_t:
            z0t = c_z0t_ice_b0t + c_z0t_ice_b1t * math.log(Re_s)
            z0t = min(z0t, c_z0t_ice_b0s)
            z0q = c_z0q_ice_b0t + c_z0q_ice_b1t * math.log(Re_s)
            z0q = min(z0q, c_z0q_ice_b0s)
        else:
            z0t = (c_z0t_ice_b0r + c_z0t_ice_b1r * math.log(Re_s)
                   + c_z0t_ice_b2r * math.log(Re_s) ** 2)
            z0q = (c_z0q_ice_b0r + c_z0q_ice_b1r * math.log(Re_s)
                   + c_z0q_ice_b2r * math.log(Re_s) ** 2)
        z0t = z0u * np.exp(z0t)
        z0q = z0u * np.exp(z0q)

    return c_z0u_fetch, u_star_thresh, z0u, z0t, z0q


def SfcFlx_momsenlat(height_u, height_tq, fetch, U_a, T_a, q_a, T_s, P_a, h_ice):
    """Fluxes of momentum and of sensible and latent heat at the surface.

    Fortran: `SfcFlx_momsenlat`, SfcFlx_mod.F90:221-612.  Returns
    ``(Q_momentum, Q_sensible, Q_latent, Q_watvap)`` in N m^-2, W m^-2,
    W m^-2 and kg m^-2 s^-1; all fluxes are positive when directed upwards.

    ``height_u`` is the wind measurement height [m], ``height_tq`` the
    temperature/humidity height [m], ``fetch`` the wind fetch [m], ``U_a``
    the wind speed [m s^-1], ``T_a`` the air temperature [K], ``q_a`` the
    air specific humidity [-], ``T_s`` the surface temperature (water, ice or
    snow) [K], ``P_a`` the surface air pressure [N m^-2] and ``h_ice`` the
    ice thickness [m].
    """
    # ------------------------------------------------------------------
    # Saturation specific humidity and air density at T=T_s (lines 364-366)
    # ------------------------------------------------------------------
    wvpres_s = SfcFlx_satwvpres(T_s, h_ice)   # Saturation w.v. pressure at T=T_s
    q_s = SfcFlx_spechum(wvpres_s, P_a)       # Saturation specific humidity at T=T_s
    rho_a = SfcFlx_rhoair(T_s, q_s, P_a)      # Air density at T_s and q_s

    # ------------------------------------------------------------------
    # Molecular fluxes (lines 372-375); they are in kinematic units
    # ------------------------------------------------------------------
    Q_mom_mol = -tpsf_nu_u_a * U_a / height_u
    Q_sen_mol = -tpsf_kappa_t_a * (T_a - T_s) / height_tq
    Q_lat_mol = -tpsf_kappa_q_a * (q_a - q_s) / height_tq

    # ------------------------------------------------------------------
    # Fluxes in free convection (lines 381-394)
    # ------------------------------------------------------------------
    par_conv_visc = ((T_s - T_a) / T_s * math.sqrt(tpsf_kappa_t_a)
                     + (q_s - q_a) * tpsf_alpha_q * math.sqrt(tpsf_kappa_q_a))
    if par_conv_visc > 0.0:  # Viscous convection takes place
        l_conv_visc = True
        par_conv_visc = (par_conv_visc * tpl_grav / tpsf_nu_u_a) ** num_1o3_sf
        Q_sen_con = c_free_conv * math.sqrt(tpsf_kappa_t_a) * par_conv_visc
        Q_sen_con = Q_sen_con * (T_s - T_a)
        Q_lat_con = c_free_conv * math.sqrt(tpsf_kappa_q_a) * par_conv_visc
        Q_lat_con = Q_lat_con * (q_s - q_a)
    else:                    # No viscous convection, set fluxes to zero
        l_conv_visc = False
        Q_sen_con = np.float64(0.0)
        Q_lat_con = np.float64(0.0)
    # Momentum flux in free (viscous or CBL-scale) convection is zero
    Q_mom_con = np.float64(0.0)

    # ------------------------------------------------------------------
    # Turbulent fluxes (lines 400-412)
    # ------------------------------------------------------------------
    R_z = height_tq / height_u           # Ratio of "height_tq" to "height_u"
    Ri_cr = c_MO_t_stab / c_MO_u_stab ** 2 * R_z  # Critical Ri
    Ri = (tpl_grav * ((T_a - T_s) / T_s + tpsf_alpha_q * (q_a - q_s))
          / max(U_a, u_wind_min_sf) ** 2)
    Ri = Ri * height_u / Pr_neutral       # Gradient Richardson number

    if U_a < u_wind_min_sf or Ri > Ri_cr - c_small_sf:
        # Low wind or Ri>Ri_cr: set turbulent fluxes to zero
        u_star_st = np.float64(0.0)
        Q_mom_tur = np.float64(0.0)
        Q_sen_tur = np.float64(0.0)
        Q_lat_tur = np.float64(0.0)
    else:
        # --------------------------------------------------------------
        # Compute z/L, where z=height_u (lines 415-437)
        # --------------------------------------------------------------
        if Ri >= 0.0:  # Stable stratification
            ZoL = math.sqrt(1.0 - 4.0 * (c_MO_u_stab - R_z * c_MO_t_stab) * Ri)
            ZoL = ZoL - 1.0 + 2.0 * c_MO_u_stab * Ri
            ZoL = ZoL / 2.0 / c_MO_u_stab / c_MO_u_stab / (Ri_cr - Ri)
        else:          # Convection
            n_iter = 0
            Delta = np.float64(1.0)  # Initial error (large compared to accuracy)
            # Initial guess for ZoL
            u_star_previter = Ri * max(
                np.float64(1.0),
                math.sqrt(R_z * c_MO_t_conv / c_MO_u_conv))
            while Delta > c_accur_sf and n_iter < n_iter_max:
                Fun = (u_star_previter ** 2 * (c_MO_u_conv * u_star_previter - 1.0)
                       + Ri ** 2 * (1.0 - R_z * c_MO_t_conv * u_star_previter))
                Fun_prime = (3.0 * c_MO_u_conv * u_star_previter ** 2
                             - 2.0 * u_star_previter
                             - R_z * c_MO_t_conv * Ri ** 2)
                ZoL = u_star_previter - Fun / Fun_prime
                Delta = (abs(ZoL - u_star_previter)
                         / max(c_accur_sf, abs(ZoL + u_star_previter)))
                u_star_previter = ZoL
                n_iter = n_iter + 1

        # --------------------------------------------------------------
        # Fetch-dependent Charnock parameter, use "u_star_min_sf" (line 440)
        # --------------------------------------------------------------
        (c_z0u_fetch, u_star_thresh,
         z0u_sf, z0t_sf, z0q_sf) = SfcFlx_roughness(fetch, U_a, u_star_min_sf, h_ice)

        # Threshold value of wind speed (lines 443-454)
        u_star_st = u_star_thresh
        (c_z0u_fetch, u_star_thresh,
         z0u_sf, z0t_sf, z0q_sf) = SfcFlx_roughness(fetch, U_a, u_star_st, h_ice)
        if ZoL > 0.0:  # MO function in stable stratification
            psi_u = c_MO_u_stab * ZoL * (1.0 - min(z0u_sf / height_u, 1.0))
        else:          # MO function in convection
            psi_t = (1.0 - c_MO_u_conv * ZoL) ** c_MO_u_exp
            psi_q = ((1.0 - c_MO_u_conv * ZoL
                      * min(z0u_sf / height_u, 1.0)) ** c_MO_u_exp)
            psi_u = (2.0 * (math.atan(psi_t) - math.atan(psi_q))
                     + 2.0 * math.log((1.0 + psi_q) / (1.0 + psi_t))
                     + math.log((1.0 + psi_q * psi_q) / (1.0 + psi_t * psi_t)))
        U_a_thresh = u_star_thresh / c_Karman * (math.log(height_u / z0u_sf) + psi_u)

        # --------------------------------------------------------------
        # Compute friction velocity (lines 457-513)
        # --------------------------------------------------------------
        n_iter = 0
        Delta = np.float64(1.0)     # Initial error (large compared to accuracy)
        u_star_previter = u_star_thresh  # Initial guess for friction velocity
        if U_a <= U_a_thresh:       # Smooth surface
            while Delta > c_accur_sf and n_iter < n_iter_max:
                (c_z0u_fetch, u_star_thresh,
                 z0u_sf, z0t_sf, z0q_sf) = SfcFlx_roughness(
                    fetch, U_a, min(u_star_thresh, u_star_previter), h_ice)
                if ZoL >= 0.0:  # Stable stratification
                    psi_u = (c_MO_u_stab * ZoL
                             * (1.0 - min(z0u_sf / height_u, 1.0)))
                    Fun = math.log(height_u / z0u_sf) + psi_u
                    Fun_prime = ((Fun + 1.0
                                  + c_MO_u_stab * ZoL
                                  * min(z0u_sf / height_u, 1.0)) / c_Karman)
                    Fun = Fun * u_star_previter / c_Karman - U_a
                else:           # Convection
                    psi_t = (1.0 - c_MO_u_conv * ZoL) ** c_MO_u_exp
                    psi_q = ((1.0 - c_MO_u_conv * ZoL
                              * min(z0u_sf / height_u, 1.0)) ** c_MO_u_exp)
                    psi_u = (2.0 * (math.atan(psi_t) - math.atan(psi_q))
                             + 2.0 * math.log((1.0 + psi_q) / (1.0 + psi_t))
                             + math.log((1.0 + psi_q * psi_q)
                                        / (1.0 + psi_t * psi_t)))
                    Fun = math.log(height_u / z0u_sf) + psi_u
                    Fun_prime = (Fun + 1.0 / psi_q) / c_Karman
                    Fun = Fun * u_star_previter / c_Karman - U_a
                u_star_st = u_star_previter - Fun / Fun_prime
                Delta = abs((u_star_st - u_star_previter)
                            / (u_star_st + u_star_previter))
                u_star_previter = u_star_st
                n_iter = n_iter + 1
        else:                       # Rough surface
            while Delta > c_accur_sf and n_iter < n_iter_max:
                (c_z0u_fetch, u_star_thresh,
                 z0u_sf, z0t_sf, z0q_sf) = SfcFlx_roughness(
                    fetch, U_a, max(u_star_thresh, u_star_previter), h_ice)
                if ZoL >= 0.0:  # Stable stratification
                    psi_u = (c_MO_u_stab * ZoL
                             * (1.0 - min(z0u_sf / height_u, 1.0)))
                    Fun = math.log(height_u / z0u_sf) + psi_u
                    Fun_prime = ((Fun - 2.0
                                  - 2.0 * c_MO_u_stab * ZoL
                                  * min(z0u_sf / height_u, 1.0)) / c_Karman)
                    Fun = Fun * u_star_previter / c_Karman - U_a
                else:           # Convection
                    psi_t = (1.0 - c_MO_u_conv * ZoL) ** c_MO_u_exp
                    psi_q = ((1.0 - c_MO_u_conv * ZoL
                              * min(z0u_sf / height_u, 1.0)) ** c_MO_u_exp)
                    psi_u = (2.0 * (math.atan(psi_t) - math.atan(psi_q))
                             + 2.0 * math.log((1.0 + psi_q) / (1.0 + psi_t))
                             + math.log((1.0 + psi_q * psi_q)
                                        / (1.0 + psi_t * psi_t)))
                    Fun = math.log(height_u / z0u_sf) + psi_u
                    Fun_prime = (Fun - 2.0 / psi_q) / c_Karman
                    Fun = Fun * u_star_previter / c_Karman - U_a
                if h_ice >= h_Ice_min_flk:
                    # No iteration is required for rough flow over ice
                    u_star_st = (c_Karman * U_a
                                 / max(c_small_sf,
                                       math.log(height_u / z0u_sf) + psi_u))
                    u_star_previter = u_star_st
                else:           # Iterate in case of open water
                    u_star_st = u_star_previter - Fun / Fun_prime
                Delta = abs((u_star_st - u_star_previter)
                            / (u_star_st + u_star_previter))
                u_star_previter = u_star_st
                n_iter = n_iter + 1

        # --------------------------------------------------------------
        # Momentum flux (line 527)
        # --------------------------------------------------------------
        Q_mom_tur = -u_star_st * u_star_st

        # --------------------------------------------------------------
        # Temperature and specific humidity fluxes (lines 530-551)
        # --------------------------------------------------------------
        (c_z0u_fetch, u_star_thresh,
         z0u_sf, z0t_sf, z0q_sf) = SfcFlx_roughness(fetch, U_a, u_star_st, h_ice)
        if ZoL >= 0.0:  # Stable stratification
            psi_t = (c_MO_t_stab * R_z * ZoL
                     * (1.0 - min(z0t_sf / height_tq, 1.0)))
            psi_q = (c_MO_q_stab * R_z * ZoL
                     * (1.0 - min(z0q_sf / height_tq, 1.0)))
        else:           # Convection
            psi_u = (1.0 - c_MO_t_conv * R_z * ZoL) ** c_MO_t_exp
            psi_t = ((1.0 - c_MO_t_conv * R_z * ZoL
                      * min(z0t_sf / height_tq, 1.0)) ** c_MO_t_exp)
            psi_t = 2.0 * math.log((1.0 + psi_t) / (1.0 + psi_u))
            psi_u = (1.0 - c_MO_q_conv * R_z * ZoL) ** c_MO_q_exp
            psi_q = ((1.0 - c_MO_q_conv * R_z * ZoL
                      * min(z0q_sf / height_tq, 1.0)) ** c_MO_q_exp)
            psi_q = 2.0 * math.log((1.0 + psi_q) / (1.0 + psi_u))
        Q_sen_tur = (-(T_a - T_s) * u_star_st * c_Karman / Pr_neutral
                     / max(c_small_sf, math.log(height_tq / z0t_sf) + psi_t))
        Q_lat_tur = (-(q_a - q_s) * u_star_st * c_Karman / Sc_neutral
                     / max(c_small_sf, math.log(height_tq / z0q_sf) + psi_q))

    # ------------------------------------------------------------------
    # Decide between turbulent, molecular, and convective fluxes (559-588)
    # ------------------------------------------------------------------
    Q_momentum = min(Q_mom_tur, Q_mom_mol, Q_mom_con)  # Momentum flux is negative
    if l_conv_visc:
        # Convection, take fluxes that are maximal in magnitude
        if abs(Q_sen_tur) >= abs(Q_sen_con):
            Q_sensible = Q_sen_tur
        else:
            Q_sensible = Q_sen_con
        if abs(Q_sensible) < abs(Q_sen_mol):
            Q_sensible = Q_sen_mol
        if abs(Q_lat_tur) >= abs(Q_lat_con):
            Q_latent = Q_lat_tur
        else:
            Q_latent = Q_lat_con
        if abs(Q_latent) < abs(Q_lat_mol):
            Q_latent = Q_lat_mol
    else:
        # Stable or neutral stratification, choose fluxes maximal in magnitude
        if abs(Q_sen_tur) >= abs(Q_sen_mol):
            Q_sensible = Q_sen_tur
        else:
            Q_sensible = Q_sen_mol
        if abs(Q_lat_tur) >= abs(Q_lat_mol):
            Q_latent = Q_lat_tur
        else:
            Q_latent = Q_lat_mol

    # ------------------------------------------------------------------
    # Set output (fluxes are no longer in kinematic units, lines 594-599)
    # ------------------------------------------------------------------
    Q_momentum = Q_momentum * rho_a
    Q_sensible = Q_sensible * rho_a * tpsf_c_a_p
    Q_watvap = Q_latent * rho_a
    Q_latent = tpsf_L_evap
    if h_ice >= h_Ice_min_flk:
        Q_latent = Q_latent + tpl_L_f  # Add latent heat of fusion over ice
    Q_latent = Q_watvap * Q_latent

    return Q_momentum, Q_sensible, Q_latent, Q_watvap
