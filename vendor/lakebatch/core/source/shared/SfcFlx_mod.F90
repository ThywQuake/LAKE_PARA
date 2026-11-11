! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

MODULE SfcFlx

!------------------------------------------------------------------------------
!
! Description:
!
!  The main program unit of 
!  the atmospheric surface-layer parameterization scheme "SfcFlx".
!  "SfcFlx" is used to compute fluxes 
!  of momentum and of sensible and latent heat over lakes.
!  The surface-layer scheme developed by Mironov (1991) was used as the starting point.
!  It was modified and further developed to incorporate new results as to 
!  the roughness lenghts for scalar quantities,
!  heat and mass transfer in free convection,
!  and the effect of limited fetch on the momentum transfer.
!  Apart from the momentum flux and sensible and latent heat fluxes,
!  the long-wave radiation flux from the water surface and
!  the long-wave radiation flux from the atmosphere can also be computed.
!  The atmospheric long-wave radiation flux is computed with simple empirical formulae,
!  where the atmospheric emissivity is taken to be dependent on 
!  the water vapour pressure and cloud fraction.
!
!  A description of SfcFlx is available from the author.
!  Dmitrii Mironov 
!  German Weather Service, Kaiserleistr. 29/35, D-63067 Offenbach am Main, Germany. 
!  dmitrii.mironov@dwd.de 
!
!  Lines embraced with "!_tmp" contain temporary parts of the code.
!  Lines embraced/marked with "!_dev" may be replaced
!  as improved parameterizations are developed and tested.
!  Lines embraced/marked with "!_dm" are DM's comments
!  that may be helpful to a user.
!  Lines embraced/marked with "!_dbg" are used
!  for debugging purposes only.
!
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

USE data_parameters  , ONLY :   &
  ireals                      , & ! KIND-type parameter for real variables
  iintegers                       ! KIND-type parameter for "normal" integer variables

USE flake_parameters , ONLY :   &
  tpl_grav                    , & ! Acceleration due to gravity [m s^{-2}]
  tpl_T_f                     , & ! Fresh water freezing point [K]
  tpl_rho_w_r                 , & ! Maximum density of fresh water [kg m^{-3}]
  tpl_c_w                     , & ! Specific heat of water [J kg^{-1} K^{-1}]
  tpl_L_f                     , & ! Latent heat of fusion [J kg^{-1}]
  h_Ice_min_flk                   ! Minimum ice thickness [m]

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations

!  Dimensionless constants in the Monin-Obukhov surface-layer 
!  similarity relations and in the expressions for the roughness lengths.
REAL (KIND = ireals), PARAMETER ::   &
  c_Karman      = 0.40_ireals      , & ! The von Karman constant 
  Pr_neutral    = 1.0_ireals       , & ! Turbulent Prandtl number at neutral static stability
  Sc_neutral    = 1.0_ireals       , & ! Turbulent Schmidt number at neutral static stability
  c_MO_u_stab   = 5.0_ireals       , & ! Constant of the MO theory (wind, stable stratification)
  c_MO_t_stab   = 5.0_ireals       , & ! Constant of the MO theory (temperature, stable stratification)
  c_MO_q_stab   = 5.0_ireals       , & ! Constant of the MO theory (humidity, stable stratification)
  c_MO_u_conv   = 15.0_ireals      , & ! Constant of the MO theory (wind, convection)
  c_MO_t_conv   = 15.0_ireals      , & ! Constant of the MO theory (temperature, convection)
  c_MO_q_conv   = 15.0_ireals      , & ! Constant of the MO theory (humidity, convection)
  c_MO_u_exp    = 0.25_ireals      , & ! Constant of the MO theory (wind, exponent)
  c_MO_t_exp    = 0.5_ireals       , & ! Constant of the MO theory (temperature, exponent)
  c_MO_q_exp    = 0.5_ireals       , & ! Constant of the MO theory (humidity, exponent)
  z0u_ice_rough = 1.0E-03_ireals   , & ! Aerodynamic roughness of the ice surface [m] (rough flow)
  c_z0u_smooth  = 0.1_ireals       , & ! Constant in the expression for z0u (smooth flow) 
  c_z0u_rough   = 1.23E-02_ireals  , & ! The Charnock constant in the expression for z0u (rough flow)
  c_z0u_rough_L = 1.00E-01_ireals  , & ! An increased Charnock constant (used as the upper limit)
  c_z0u_ftch_f  = 0.70_ireals      , & ! Factor in the expression for fetch-dependent Charnock parameter
  c_z0u_ftch_ex = 0.3333333_ireals , & ! Exponent in the expression for fetch-dependent Charnock parameter
  c_z0t_rough_1 = 4.0_ireals       , & ! Constant in the expression for z0t (factor) 
  c_z0t_rough_2 = 3.2_ireals       , & ! Constant in the expression for z0t (factor)
  c_z0t_rough_3 = 0.5_ireals       , & ! Constant in the expression for z0t (exponent) 
  c_z0q_rough_1 = 4.0_ireals       , & ! Constant in the expression for z0q (factor)
  c_z0q_rough_2 = 4.2_ireals       , & ! Constant in the expression for z0q (factor)
  c_z0q_rough_3 = 0.5_ireals       , & ! Constant in the expression for z0q (exponent)
  c_z0t_ice_b0s = 1.250_ireals     , & ! Constant in the expression for z0t over ice
  c_z0t_ice_b0t = 0.149_ireals     , & ! Constant in the expression for z0t over ice
  c_z0t_ice_b1t = -0.550_ireals    , & ! Constant in the expression for z0t over ice
  c_z0t_ice_b0r = 0.317_ireals     , & ! Constant in the expression for z0t over ice
  c_z0t_ice_b1r = -0.565_ireals    , & ! Constant in the expression for z0t over ice
  c_z0t_ice_b2r = -0.183_ireals    , & ! Constant in the expression for z0t over ice
  c_z0q_ice_b0s = 1.610_ireals     , & ! Constant in the expression for z0q over ice
  c_z0q_ice_b0t = 0.351_ireals     , & ! Constant in the expression for z0q over ice
  c_z0q_ice_b1t = -0.628_ireals    , & ! Constant in the expression for z0q over ice
  c_z0q_ice_b0r = 0.396_ireals     , & ! Constant in the expression for z0q over ice
  c_z0q_ice_b1r = -0.512_ireals    , & ! Constant in the expression for z0q over ice
  c_z0q_ice_b2r = -0.180_ireals    , & ! Constant in the expression for z0q over ice
  Re_z0s_ice_t  = 2.5_ireals       , & ! Threshold value of the surface Reynolds number 
                                       ! used to compute z0t and z0q over ice (Andreas 2002)
  Re_z0u_thresh = 0.1_ireals           ! Threshold value of the roughness Reynolds number 
                                       ! [value from Zilitinkevich, Grachev, and Fairall (200),
                                       ! currently not used] 

!  Dimensionless constants 
REAL (KIND = ireals), PARAMETER ::   &
  c_free_conv   = 0.14_ireals          ! Constant in the expressions for fluxes in free convection

!  Dimensionless constants 
REAL (KIND = ireals), PARAMETER ::   &
  c_lwrad_emis  = 0.99_ireals          ! Surface emissivity with respect to the long-wave radiation

!  Thermodynamic parameters
REAL (KIND = ireals), PARAMETER ::        &
  tpsf_C_StefBoltz = 5.67E-08_ireals    , & ! The Stefan-Boltzmann constant [W m^{-2} K^{-4}]
  tpsf_R_dryair    = 2.8705E+02_ireals  , & ! Gas constant for dry air [J kg^{-1} K^{-1}]
  tpsf_R_watvap    = 4.6151E+02_ireals  , & ! Gas constant for water vapour [J kg^{-1} K^{-1}]
  tpsf_c_a_p       = 1.005E+03_ireals   , & ! Specific heat of air at constant pressure [J kg^{-1} K^{-1}]
  tpsf_L_evap      = 2.501E+06_ireals   , & ! Specific heat of evaporation [J kg^{-1}]
  tpsf_nu_u_a      = 1.50E-05_ireals    , & ! Kinematic molecular viscosity of air [m^{2} s^{-1}]
  tpsf_kappa_t_a   = 2.20E-05_ireals    , & ! Molecular temperature conductivity of air [m^{2} s^{-1}]
  tpsf_kappa_q_a   = 2.40E-05_ireals        ! Molecular diffusivity of air for water vapour [m^{2} s^{-1}]

!  Derived thermodynamic parameters
REAL (KIND = ireals), PARAMETER ::                        &
  tpsf_Rd_o_Rv  = tpsf_R_dryair/tpsf_R_watvap           , & ! Ratio of gas constants (Rd/Rv)
  tpsf_alpha_q  = (1._ireals-tpsf_Rd_o_Rv)/tpsf_Rd_o_Rv     ! Diemsnionless ratio 

!  Thermodynamic parameters
REAL (KIND = ireals), PARAMETER ::     &
  P_a_ref             = 1.0E+05_ireals   ! Reference pressure [N m^{-2} = kg m^{-1} s^{-2}]


!  The variables declared below
!  are accessible to all program units of the MODULE "SfcFlx"
!  and to the driving routines that use "SfcFlx".
!  These are basically the quantities computed by SfcFlx.
!  Apart from these quantities, there a few local scalars 
!  used by SfcFlx routines mainly for security reasons.
!  All variables declared below have a suffix "sf".

!  SfcFlx variables of type REAL

!  Roughness lengths
REAL (KIND = ireals) ::    &
  z0u_sf                 , & ! Roughness length with respect to wind velocity [m]
  z0t_sf                 , & ! Roughness length with respect to potential temperature [m]
  z0q_sf                     ! Roughness length with respect to specific humidity [m]

!  Fluxes in the surface air layer
REAL (KIND = ireals) ::    &
  u_star_a_sf            , & ! Friction velocity [m s^{-1}]
  Q_mom_a_sf             , & ! Momentum flux [N m^{-2}]
  Q_sens_a_sf            , & ! Sensible heat flux [W m^{-2}]
  Q_lat_a_sf             , & ! Laten heat flux [W m^{-2}]
  Q_watvap_a_sf              ! Flux of water vapout [kg m^{-2} s^{-1}]

!  Security constants
REAL (KIND = ireals), PARAMETER ::   &
  u_wind_min_sf  = 1.0E-02_ireals  , & ! Minimum wind speed [m s^{-1}]
  u_star_min_sf  = 1.0E-04_ireals  , & ! Minimum value of friction velocity [m s^{-1}]
  c_accur_sf     = 1.0E-07_ireals  , & ! A small number (accuracy)
  c_small_sf     = 1.0E-04_ireals      ! A small number (used to compute fluxes)

!  Useful constants
REAL (KIND = ireals), PARAMETER ::     &
  num_1o3_sf = 1._ireals/3._ireals       ! 1/3

!==============================================================================
! Procedures 
!==============================================================================

CONTAINS

!==============================================================================
!  The codes of the SfcFlx procedures are stored in separate "*.incf" files
!  and are included below.
!------------------------------------------------------------------------------

!==============================================================================
!include 'SfcFlx_lwradatm.incf'

!==============================================================================

!==============================================================================
!include 'SfcFlx_lwradwsfc.incf'

!==============================================================================

!==============================================================================
!include 'SfcFlx_momsenlat.incf'

! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

SUBROUTINE SfcFlx_momsenlat ( height_u, height_tq, fetch,                &
                              U_a, T_a, q_a, T_s, P_a, h_ice,            &
                              Q_momentum, Q_sensible, Q_latent, Q_watvap ) 

!------------------------------------------------------------------------------
!
! Description:
!
!  The SfcFlx routine 
!  where fluxes of momentum and of sensible and latent heat 
!  at the air-water or air-ice (air-snow) interface are computed. 
!
!  Lines embraced with "!_tmp" contain temporary parts of the code.
!  Lines embraced/marked with "!_dev" may be replaced
!  as improved parameterizations are developed and tested.
!  Lines embraced/marked with "!_dm" are DM's comments
!  that may be helpful to a user.
!  Lines embraced/marked with "!_dbg" are used 
!  for debugging purposes only.
!
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

!_dm Parameters are USEd in module "SfcFlx".
!_nu USE data_parameters , ONLY : &
!_nu     ireals,                  & ! KIND-type parameter for real variables
!_nu     iintegers                  ! KIND-type parameter for "normal" integer variables

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations

!  Input (procedure arguments)

REAL (KIND = ireals), INTENT(IN) ::   &
  height_u                          , & ! Height where wind is measured [m]
  height_tq                         , & ! Height where temperature and humidity are measured [m]
  fetch                             , & ! Typical wind fetch [m]
  U_a                               , & ! Wind speed [m s^{-1}]
  T_a                               , & ! Air temperature [K]
  q_a                               , & ! Air specific humidity [-]
  T_s                               , & ! Surface temperature (water, ice or snow) [K]
  P_a                               , & ! Surface air pressure [N m^{-2} = kg m^{-1} s^{-2}]
  h_ice                                 ! Ice thickness [m]

!  Output (procedure arguments)

REAL (KIND = ireals), INTENT(OUT) ::   &
  Q_momentum                         , & ! Momentum flux [N m^{-2}]  
  Q_sensible                         , & ! Sensible heat flux [W m^{-2}]  
  Q_latent                           , & ! Laten heat flux [W m^{-2}]
  Q_watvap                               ! Flux of water vapout [kg m^{-2} s^{-1}]


!  Local parameters of type INTEGER
INTEGER (KIND = iintegers), PARAMETER ::  &
  n_iter_max     = 24                       ! Maximum number of iterations 

!  Local variables of type LOGICAL
LOGICAL ::          &
  l_conv_visc     , & ! Switch, TRUE = viscous free convection, the Nu=C Ra^(1/3) law is used
  l_conv_cbl          ! Switch, TRUE = CBL scale convective structures define surface fluxes 

!  Local variables of type INTEGER
INTEGER (KIND = iintegers) ::   &
  i                           , & ! Loop index
  n_iter                          ! Number of iterations performed 

!  Local variables of type REAL
REAL (KIND = ireals) ::    &
  rho_a                  , & ! Air density [kg m^{-3}]  
  wvpres_s               , & ! Saturation water vapour pressure at T=T_s [N m^{-2}]
  q_s                        ! Saturation specific humidity at T=T_s [-]

!  Local variables of type REAL
REAL (KIND = ireals) ::    &
  Q_mom_tur              , & ! Turbulent momentum flux [N m^{-2}]
  Q_sen_tur              , & ! Turbulent sensible heat flux [W m^{-2}]  
  Q_lat_tur              , & ! Turbulent laten heat flux [W m^{-2}]
  Q_mom_mol              , & ! Molecular momentum flux [N m^{-2}]
  Q_sen_mol              , & ! Molecular sensible heat flux [W m^{-2}]  
  Q_lat_mol              , & ! Molecular laten heat flux [W m^{-2}]
  Q_mom_con              , & ! Momentum flux in free convection [N m^{-2}]
  Q_sen_con              , & ! Sensible heat flux in free convection [W m^{-2}]  
  Q_lat_con                  ! Laten heat flux in free convection [W m^{-2}]

!  Local variables of type REAL
REAL (KIND = ireals) ::    &
  par_conv_visc          , & ! Viscous convection stability parameter
  par_conv_cbl           , & ! CBL convection stability parameter
  c_z0u_fetch            , & ! Fetch-dependent Charnock parameter
  U_a_thresh             , & ! Threshld value of the wind speed [m s^{-1}] 
  u_star_thresh          , & ! Threshld value of friction velocity [m s^{-1}]
  u_star_previter        , & ! Friction velocity from previous iteration [m s^{-1}]
  u_star_n               , & ! Friction velocity at neutral stratification [m s^{-1}]
  u_star_st              , & ! Friction velocity with due regard for stratification [m s^{-1}]
  ZoL                    , & ! The z/L ratio, z=height_u
  Ri                     , & ! Gradient Richardson number 
  Ri_cr                  , & ! Critical value of Ri 
  R_z                    , & ! Ratio of "height_tq" to "height_u"
  Fun                    , & ! A function of generic variable "x"
  Fun_prime              , & ! Derivative of "Fun" with respect to "x"
  Delta                  , & ! Relative error 
  psi_u                  , & ! The MO stability function for wind profile
  psi_t                  , & ! The MO stability function for temperature profile
  psi_q                      ! The MO stability function for specific humidity profile


!==============================================================================
!  Start calculations
!------------------------------------------------------------------------------

!_dm All fluxes are positive when directed upwards.

!------------------------------------------------------------------------------
!  Compute saturation specific humidity and the air density at T=T_s
!------------------------------------------------------------------------------

wvpres_s = SfcFlx_satwvpres(T_s, h_ice)  ! Saturation water vapour pressure at T=T_s
q_s = SfcFlx_spechum (wvpres_s, P_a)     ! Saturation specific humidity at T=T_s
rho_a = SfcFlx_rhoair(T_s, q_s, P_a)     ! Air density at T_s and q_s (surface values)

!------------------------------------------------------------------------------
!  Compute molecular fluxes of momentum and of sensible and latent heat
!------------------------------------------------------------------------------

!_dm The fluxes are in kinematic units
Q_mom_mol = -tpsf_nu_u_a*U_a/height_u 
Q_sen_mol = -tpsf_kappa_t_a*(T_a-T_s)/height_tq    
Q_lat_mol = -tpsf_kappa_q_a*(q_a-q_s)/height_tq  

!------------------------------------------------------------------------------
!  Compute fluxes in free convection
!------------------------------------------------------------------------------

par_conv_visc = (T_s-T_a)/T_s*SQRT(tpsf_kappa_t_a) + (q_s-q_a)*tpsf_alpha_q*SQRT(tpsf_kappa_q_a)
IF(par_conv_visc.GT.0._ireals) THEN   ! Viscous convection takes place
  l_conv_visc = .TRUE.
  par_conv_visc = (par_conv_visc*tpl_grav/tpsf_nu_u_a)**num_1o3_sf
  Q_sen_con = c_free_conv*SQRT(tpsf_kappa_t_a)*par_conv_visc  
  Q_sen_con = Q_sen_con*(T_s-T_a)
  Q_lat_con = c_free_conv*SQRT(tpsf_kappa_q_a)*par_conv_visc
  Q_lat_con = Q_lat_con*(q_s-q_a)
ELSE                                  ! No viscous convection, set fluxes to zero
  l_conv_visc = .FALSE.
  Q_sen_con = 0._ireals 
  Q_lat_con = 0._ireals
END IF
Q_mom_con = 0._ireals                 ! Momentum flux in free (viscous or CBL-scale) convection is zero  

!------------------------------------------------------------------------------
!  Compute turbulent fluxes
!------------------------------------------------------------------------------

R_z   = height_tq/height_u                        ! Ratio of "height_tq" to "height_u"
Ri_cr = c_MO_t_stab/c_MO_u_stab**2_iintegers*R_z  ! Critical Ri
Ri    = tpl_grav*((T_a-T_s)/T_s+tpsf_alpha_q*(q_a-q_s))/MAX(U_a,u_wind_min_sf)**2_iintegers
Ri    = Ri*height_u/Pr_neutral                    ! Gradient Richardson number

Turb_Fluxes: IF(U_a.LT.u_wind_min_sf.OR.Ri.GT.Ri_cr-c_small_sf) THEN  ! Low wind or Ri>Ri_cr 

u_star_st = 0._ireals                       ! Set turbulent fluxes to zero 
Q_mom_tur = 0._ireals                       
Q_sen_tur = 0._ireals   
Q_lat_tur = 0._ireals  

ELSE Turb_Fluxes                            ! Compute turbulent fluxes using MO similarity

! Compute z/L, where z=height_u
IF(Ri.GE.0._ireals) THEN   ! Stable stratification
  ZoL = SQRT(1._ireals-4._ireals*(c_MO_u_stab-R_z*c_MO_t_stab)*Ri)
  ZoL = ZoL - 1._ireals + 2._ireals*c_MO_u_stab*Ri
  ZoL = ZoL/2._ireals/c_MO_u_stab/c_MO_u_stab/(Ri_cr-Ri)
ELSE                       ! Convection
  n_iter = 0_iintegers
  Delta = 1._ireals                ! Set initial error to a large value (as compared to the accuracy)
  u_star_previter = Ri*MAX(1._ireals, SQRT(R_z*c_MO_t_conv/c_MO_u_conv)) ! Initial guess for ZoL
  DO WHILE (Delta.GT.c_accur_sf.AND.n_iter.LT.n_iter_max) 
    Fun = u_star_previter**2_iintegers*(c_MO_u_conv*u_star_previter-1._ireals)  &
        + Ri**2_iintegers*(1._ireals-R_z*c_MO_t_conv*u_star_previter)
    Fun_prime = 3._ireals*c_MO_u_conv*u_star_previter**2_iintegers              &
              - 2._ireals*u_star_previter - R_z*c_MO_t_conv*Ri**2_iintegers
    ZoL = u_star_previter - Fun/Fun_prime
    Delta = ABS(ZoL-u_star_previter)/MAX(c_accur_sf, ABS(ZoL+u_star_previter))
    u_star_previter = ZoL
    n_iter = n_iter + 1_iintegers
  END DO 
!_dbg
!  IF(n_iter.GE.n_iter_max-1_iintegers)  & 
!    WRITE(*,*) 'ZoL: Max No. iters. exceeded (n_iter = ', n_iter, ')!'
!_dbg
END IF

!  Compute fetch-dependent Charnock parameter, use "u_star_min_sf"
CALL SfcFlx_roughness (fetch, U_a, u_star_min_sf, h_ice, c_z0u_fetch, u_star_thresh, z0u_sf, z0t_sf, z0q_sf)

!  Threshold value of wind speed 
u_star_st = u_star_thresh
CALL SfcFlx_roughness (fetch, U_a, u_star_st, h_ice, c_z0u_fetch, u_star_thresh, z0u_sf, z0t_sf, z0q_sf)
IF(ZoL.GT.0._ireals) THEN   ! MO function in stable stratification 
  psi_u = c_MO_u_stab*ZoL*(1._ireals-MIN(z0u_sf/height_u, 1._ireals))
ELSE                        ! MO function in convection
  psi_t = (1._ireals-c_MO_u_conv*ZoL)**c_MO_u_exp
  psi_q = (1._ireals-c_MO_u_conv*ZoL*MIN(z0u_sf/height_u, 1._ireals))**c_MO_u_exp
  psi_u = 2._ireals*(ATAN(psi_t)-ATAN(psi_q))                  &
        + 2._ireals*LOG((1._ireals+psi_q)/(1._ireals+psi_t))   &
        + LOG((1._ireals+psi_q*psi_q)/(1._ireals+psi_t*psi_t))   
END IF 
U_a_thresh = u_star_thresh/c_Karman*(LOG(height_u/z0u_sf)+psi_u)

!  Compute friction velocity 
n_iter = 0_iintegers
Delta = 1._ireals                ! Set initial error to a large value (as compared to the accuracy)
u_star_previter = u_star_thresh  ! Initial guess for friction velocity  
IF(U_a.LE.U_a_thresh) THEN  ! Smooth surface
  DO WHILE (Delta.GT.c_accur_sf.AND.n_iter.LT.n_iter_max) 
    CALL SfcFlx_roughness (fetch, U_a, MIN(u_star_thresh, u_star_previter), h_ice,   &
                           c_z0u_fetch, u_star_thresh, z0u_sf, z0t_sf, z0q_sf)
    IF(ZoL.GE.0._ireals) THEN  ! Stable stratification
      psi_u = c_MO_u_stab*ZoL*(1._ireals-MIN(z0u_sf/height_u, 1._ireals))
      Fun = LOG(height_u/z0u_sf) + psi_u
      Fun_prime = (Fun + 1._ireals + c_MO_u_stab*ZoL*MIN(z0u_sf/height_u, 1._ireals))/c_Karman
      Fun = Fun*u_star_previter/c_Karman - U_a
    ELSE                       ! Convection 
      psi_t = (1._ireals-c_MO_u_conv*ZoL)**c_MO_u_exp
      psi_q = (1._ireals-c_MO_u_conv*ZoL*MIN(z0u_sf/height_u, 1._ireals))**c_MO_u_exp
      psi_u = 2._ireals*(ATAN(psi_t)-ATAN(psi_q))                  &
            + 2._ireals*LOG((1._ireals+psi_q)/(1._ireals+psi_t))   &
            + LOG((1._ireals+psi_q*psi_q)/(1._ireals+psi_t*psi_t))   
      Fun = LOG(height_u/z0u_sf) + psi_u
      Fun_prime = (Fun + 1._ireals/psi_q)/c_Karman
      Fun = Fun*u_star_previter/c_Karman - U_a
    END IF
    u_star_st = u_star_previter - Fun/Fun_prime
    Delta = ABS((u_star_st-u_star_previter)/(u_star_st+u_star_previter))
    u_star_previter = u_star_st
    n_iter = n_iter + 1_iintegers
  END DO 
ELSE                        ! Rough surface
  DO WHILE (Delta.GT.c_accur_sf.AND.n_iter.LT.n_iter_max) 
    CALL SfcFlx_roughness (fetch, U_a, MAX(u_star_thresh, u_star_previter), h_ice,   &
                           c_z0u_fetch, u_star_thresh, z0u_sf, z0t_sf, z0q_sf)
    IF(ZoL.GE.0._ireals) THEN  ! Stable stratification
      psi_u = c_MO_u_stab*ZoL*(1._ireals-MIN(z0u_sf/height_u, 1._ireals))
      Fun = LOG(height_u/z0u_sf) + psi_u
      Fun_prime = (Fun - 2._ireals - 2._ireals*c_MO_u_stab*ZoL*MIN(z0u_sf/height_u, 1._ireals))/c_Karman
      Fun = Fun*u_star_previter/c_Karman - U_a
    ELSE                       ! Convection 
      psi_t = (1._ireals-c_MO_u_conv*ZoL)**c_MO_u_exp
      psi_q = (1._ireals-c_MO_u_conv*ZoL*MIN(z0u_sf/height_u, 1._ireals))**c_MO_u_exp
      psi_u = 2._ireals*(ATAN(psi_t)-ATAN(psi_q))                  &
            + 2._ireals*LOG((1._ireals+psi_q)/(1._ireals+psi_t))   &
            + LOG((1._ireals+psi_q*psi_q)/(1._ireals+psi_t*psi_t))   
      Fun = LOG(height_u/z0u_sf) + psi_u
      Fun_prime = (Fun - 2._ireals/psi_q)/c_Karman
      Fun = Fun*u_star_previter/c_Karman - U_a
    END IF
    IF(h_ice.GE.h_Ice_min_flk) THEN   ! No iteration is required for rough flow over ice
      u_star_st = c_Karman*U_a/MAX(c_small_sf, LOG(height_u/z0u_sf)+psi_u)
      u_star_previter = u_star_st
    ELSE                              ! Iterate in case of open water
      u_star_st = u_star_previter - Fun/Fun_prime
    END IF
    Delta = ABS((u_star_st-u_star_previter)/(u_star_st+u_star_previter))
    u_star_previter = u_star_st
    n_iter = n_iter + 1_iintegers
  END DO 
END IF

!_dbg
!  WRITE(*,*) 'MO stab. func. psi_u = ', psi_u, '   n_iter = ', n_iter
!  WRITE(*,*) '   Wind speed = ', U_a, '  u_* = ', u_star_st
!  WRITE(*,*) '   Fun = ', Fun
!_dbg

!_dbg
!  IF(n_iter.GE.n_iter_max-1_iintegers)  & 
!    WRITE(*,*) 'u_*: Max No. iters. exceeded (n_iter = ', n_iter, ')!'
!_dbg

!  Momentum flux
Q_mom_tur = -u_star_st*u_star_st

!  Temperature and specific humidity fluxes
CALL SfcFlx_roughness (fetch, U_a, u_star_st, h_ice, c_z0u_fetch, u_star_thresh, z0u_sf, z0t_sf, z0q_sf)
IF(ZoL.GE.0._ireals) THEN   ! Stable stratification 
  psi_t = c_MO_t_stab*R_z*ZoL*(1._ireals-MIN(z0t_sf/height_tq, 1._ireals))
  psi_q = c_MO_q_stab*R_z*ZoL*(1._ireals-MIN(z0q_sf/height_tq, 1._ireals))
!_dbg
!  WRITE(*,*) 'STAB: psi_t = ', psi_t, '   psi_q = ', psi_q
!_dbg
ELSE                        ! Convection 
  psi_u = (1._ireals-c_MO_t_conv*R_z*ZoL)**c_MO_t_exp
  psi_t = (1._ireals-c_MO_t_conv*R_z*ZoL*MIN(z0t_sf/height_tq, 1._ireals))**c_MO_t_exp
  psi_t = 2._ireals*LOG((1._ireals+psi_t)/(1._ireals+psi_u))
  psi_u = (1._ireals-c_MO_q_conv*R_z*ZoL)**c_MO_q_exp
  psi_q = (1._ireals-c_MO_q_conv*R_z*ZoL*MIN(z0q_sf/height_tq, 1._ireals))**c_MO_q_exp
  psi_q = 2._ireals*LOG((1._ireals+psi_q)/(1._ireals+psi_u))
!_dbg
!  WRITE(*,*) 'CONV: psi_t = ', psi_t, '   psi_q = ', psi_q
!_dbg
END IF 
Q_sen_tur = -(T_a-T_s)*u_star_st*c_Karman/Pr_neutral  &
          / MAX(c_small_sf, LOG(height_tq/z0t_sf)+psi_t)
Q_lat_tur = -(q_a-q_s)*u_star_st*c_Karman/Sc_neutral  &
          / MAX(c_small_sf, LOG(height_tq/z0q_sf)+psi_q)

END IF Turb_Fluxes

!------------------------------------------------------------------------------
!  Decide between turbulent, molecular, and convective fluxes
!------------------------------------------------------------------------------

Q_momentum = MIN(Q_mom_tur, Q_mom_mol, Q_mom_con)  ! Momentum flux is negative          
IF(l_conv_visc) THEN    ! Convection, take fluxes that are maximal in magnitude 
  IF(ABS(Q_sen_tur).GE.ABS(Q_sen_con)) THEN
    Q_sensible = Q_sen_tur
  ELSE
    Q_sensible = Q_sen_con
  END IF
  IF(ABS(Q_sensible).LT.ABS(Q_sen_mol)) THEN
    Q_sensible = Q_sen_mol
  END IF
  IF(ABS(Q_lat_tur).GE.ABS(Q_lat_con)) THEN
    Q_latent = Q_lat_tur
  ELSE
    Q_latent = Q_lat_con
  END IF
  IF(ABS(Q_latent).LT.ABS(Q_lat_mol)) THEN
    Q_latent = Q_lat_mol
  END IF
ELSE                    ! Stable or neutral stratification, chose fluxes that are maximal in magnitude 
  IF(ABS(Q_sen_tur).GE.ABS(Q_sen_mol)) THEN 
    Q_sensible = Q_sen_tur
  ELSE 
    Q_sensible = Q_sen_mol    
  END IF
  IF(ABS(Q_lat_tur).GE.ABS(Q_lat_mol)) THEN 
    Q_latent = Q_lat_tur
  ELSE 
    Q_latent = Q_lat_mol  
  END IF
END IF

!------------------------------------------------------------------------------
!  Set output (notice that fluxes are no longer in kinematic units)
!------------------------------------------------------------------------------

Q_momentum = Q_momentum*rho_a 
Q_sensible = Q_sensible*rho_a*tpsf_c_a_p
Q_watvap   = Q_latent*rho_a
Q_latent = tpsf_L_evap
IF(h_ice.GE.h_Ice_min_flk) Q_latent = Q_latent + tpl_L_f   ! Add latent heat of fusion over ice
Q_latent = Q_watvap*Q_latent

! Set "*_sf" variables to make fluxes accessible to driving routines that use "SfcFlx"
u_star_a_sf     = u_star_st 
Q_mom_a_sf      = Q_momentum  
Q_sens_a_sf     = Q_sensible 
Q_lat_a_sf      = Q_latent
Q_watvap_a_sf   = Q_watvap

!------------------------------------------------------------------------------
!  End calculations
!==============================================================================

END SUBROUTINE SfcFlx_momsenlat

!==============================================================================

!==============================================================================
!include 'SfcFlx_rhoair.incf'

! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

REAL (KIND = ireals) FUNCTION SfcFlx_rhoair (T, q, P)

!------------------------------------------------------------------------------
!
! Description:
!
!  Computes the air density as function 
!  of temperature, specific humidity and pressure.
!  
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

!_dm Parameters are USEd in module "SfcFlx".
!_nu USE data_parameters , ONLY : &
!_nu     ireals,                  & ! KIND-type parameter for real variables
!_nu     iintegers                  ! KIND-type parameter for "normal" integer variables

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations
 
!  Input (function argument) 
REAL (KIND = ireals), INTENT(IN) ::   &
  T                                 , & ! Temperature [K]
  q                                 , & ! Specific humidity 
  P                                     ! Pressure [N m^{-2} = kg m^{-1} s^{-2}]

!==============================================================================
!  Start calculations
!------------------------------------------------------------------------------

! Air density [kg m^{-3}] 

SfcFlx_rhoair = P/tpsf_R_dryair/T/(1._ireals+(1._ireals/tpsf_Rd_o_Rv-1._ireals)*q)

!------------------------------------------------------------------------------
!  End calculations
!==============================================================================

END FUNCTION SfcFlx_rhoair



!==============================================================================

!==============================================================================
!include 'SfcFlx_roughness.incf'

! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

SUBROUTINE SfcFlx_roughness (fetch, U_a, u_star, h_ice,   & 
                             c_z0u_fetch, u_star_thresh, z0u, z0t, z0q)

!------------------------------------------------------------------------------
!
! Description:
!
!  Computes the water-surface or the ice-surface roughness lengths
!  with respect to wind velocity, potential temperature and specific humidity.
!
!  The water-surface roughness lengths with respect to wind velocity is computed
!  from the Charnock formula when the surface is aerodynamically rough.
!  A simple empirical formulation is used to account for the dependence 
!  of the Charnock parameter on the wind fetch. 
!  When the flow is aerodynamically smooth, the roughness length with respect to 
!  wind velocity is proportional to the depth of the viscous sub-layer.
!  The water-surface roughness lengths for scalars are computed using the power-law 
!  formulations in terms of the roughness Reynolds number (Zilitinkevich et al. 2001).
!  The ice-surface aerodynamic roughness is taken to be constant.
!  The ice-surface roughness lengths for scalars 
!  are computed through the power-law formulations 
!  in terms of the roughness Reynolds number (Andreas 2002).
!
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

!_dm Parameters are USEd in module "SfcFlx".
!_nu USE data_parameters , ONLY :   &
!_nu   ireals                     , & ! KIND-type parameter for real variables
!_nu   iintegers                      ! KIND-type parameter for "normal" integer variables

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations

!  Input (procedure arguments)
REAL (KIND = ireals), INTENT(IN) ::   &
  fetch                             , & ! Typical wind fetch [m]
  U_a                               , & ! Wind speed [m s^{-1}]
  u_star                            , & ! Friction velocity in the surface air layer [m s^{-1}]
  h_ice                                 ! Ice thickness [m]

!  Output (procedure arguments)
REAL (KIND = ireals), INTENT(OUT) ::   &
  c_z0u_fetch                        , & ! Fetch-dependent Charnock parameter
  u_star_thresh                      , & ! Threshold value of friction velocity [m s^{-1}]
  z0u                                , & ! Roughness length with respect to wind velocity [m]
  z0t                                , & ! Roughness length with respect to potential temperature [m]
  z0q                                    ! Roughness length with respect to specific humidity [m]

!  Local variables of type REAL
REAL (KIND = ireals) ::    &
  Re_s                   , & ! Surface Reynolds number 
  Re_s_thresh                ! Threshold value of Re_s

!==============================================================================
!  Start calculations
!------------------------------------------------------------------------------

Water_or_Ice: IF(h_ice.LT.h_Ice_min_flk) THEN  ! Water surface  

! The Charnock parameter as dependent on dimensionless fetch
  c_z0u_fetch = MAX(U_a, u_wind_min_sf)**2_iintegers/tpl_grav/fetch  ! Inverse dimensionless fetch
  c_z0u_fetch = c_z0u_rough + c_z0u_ftch_f*c_z0u_fetch**c_z0u_ftch_ex
  c_z0u_fetch = MIN(c_z0u_fetch, c_z0u_rough_L)                      ! Limit Charnock parameter

! Threshold value of friction velocity
  u_star_thresh = (c_z0u_smooth/c_z0u_fetch*tpl_grav*tpsf_nu_u_a)**num_1o3_sf

! Surface Reynolds number and its threshold value
  Re_s = u_star**3_iintegers/tpsf_nu_u_a/tpl_grav
  Re_s_thresh = c_z0u_smooth/c_z0u_fetch

! Aerodynamic roughness
  IF(Re_s.LE.Re_s_thresh) THEN                 
    z0u = c_z0u_smooth*tpsf_nu_u_a/u_star     ! Smooth flow
  ELSE
    z0u = c_z0u_fetch*u_star*u_star/tpl_grav  ! Rough flow
  END IF 
! Roughness for scalars  
  z0q = c_z0u_fetch*MAX(Re_s, Re_s_thresh)
  z0t = c_z0t_rough_1*z0q**c_z0t_rough_3 - c_z0t_rough_2
  z0q = c_z0q_rough_1*z0q**c_z0q_rough_3 - c_z0q_rough_2
  z0t = z0u*EXP(-c_Karman/Pr_neutral*z0t)
  z0q = z0u*EXP(-c_Karman/Sc_neutral*z0q) 

ELSE Water_or_Ice                              ! Ice surface

! The Charnock parameter is not used over ice, formally set "c_z0u_fetch" to its minimum value
  c_z0u_fetch = c_z0u_rough

! Threshold value of friction velocity
  u_star_thresh = c_z0u_smooth*tpsf_nu_u_a/z0u_ice_rough

! Aerodynamic roughness
  z0u = MAX(z0u_ice_rough, c_z0u_smooth*tpsf_nu_u_a/u_star)

! Roughness Reynolds number 
  Re_s = MAX(u_star*z0u/tpsf_nu_u_a, c_accur_sf)

! Roughness for scalars  
  IF(Re_s.LE.Re_z0s_ice_t) THEN 
    z0t = c_z0t_ice_b0t + c_z0t_ice_b1t*LOG(Re_s)
    z0t = MIN(z0t, c_z0t_ice_b0s)
    z0q = c_z0q_ice_b0t + c_z0q_ice_b1t*LOG(Re_s)
    z0q = MIN(z0q, c_z0q_ice_b0s)
  ELSE 
    z0t = c_z0t_ice_b0r + c_z0t_ice_b1r*LOG(Re_s) + c_z0t_ice_b2r*LOG(Re_s)**2_iintegers
    z0q = c_z0q_ice_b0r + c_z0q_ice_b1r*LOG(Re_s) + c_z0q_ice_b2r*LOG(Re_s)**2_iintegers
  END IF
  z0t = z0u*EXP(z0t)
  z0q = z0u*EXP(z0q)

END IF Water_or_Ice

!------------------------------------------------------------------------------
!  End calculations
!==============================================================================

END SUBROUTINE SfcFlx_roughness

!==============================================================================

!==============================================================================
!include 'SfcFlx_satwvpres.incf'

! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

REAL (KIND = ireals) FUNCTION SfcFlx_satwvpres (T, h_ice)

!------------------------------------------------------------------------------
!
! Description:
!
!  Computes saturation water vapour pressure 
!  over the water surface or over the ice surface
!  as function of temperature. 
!  
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

!_dm Parameters are USEd in module "SfcFlx".
!_nu USE data_parameters  , ONLY : &
!_nu     ireals,                   & ! KIND-type parameter for real variables
!_nu     iintegers                   ! KIND-type parameter for "normal" integer variables

!_dm The variable is USEd in module "SfcFlx".
!_nu USE flake_parameters , ONLY : &
!_nu   h_Ice_min_flk                 ! Minimum ice thickness [m]

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations
 
!  Input (function argument) 
REAL (KIND = ireals), INTENT(IN) ::   &
  T                                 , & ! Temperature [K]
  h_ice                                 ! Ice thickness [m]

!  Local parameters
REAL (KIND = ireals), PARAMETER ::   &
   b1_vap   = 610.78_ireals        , & ! Coefficient [N m^{-2} = kg m^{-1} s^{-2}]
   b3_vap   = 273.16_ireals        , & ! Triple point [K]
   b2w_vap  = 17.2693882_ireals    , & ! Coefficient (water)
   b2i_vap  = 21.8745584_ireals    , & ! Coefficient (ice) 
   b4w_vap  = 35.86_ireals         , & ! Coefficient (temperature) [K]
   b4i_vap  = 7.66_ireals              ! Coefficient (temperature) [K]

!==============================================================================
!  Start calculations
!------------------------------------------------------------------------------

! Saturation water vapour pressure [N m^{-2} = kg m^{-1} s^{-2}]

IF(h_ice.LT.h_Ice_min_flk) THEN  ! Water surface
  SfcFlx_satwvpres = b1_vap*EXP(b2w_vap*(T-b3_vap)/(T-b4w_vap))
ELSE                             ! Ice surface
  SfcFlx_satwvpres = b1_vap*EXP(b2i_vap*(T-b3_vap)/(T-b4i_vap))
END IF 

!------------------------------------------------------------------------------
!  End calculations
!==============================================================================

END FUNCTION SfcFlx_satwvpres



!==============================================================================

!==============================================================================
!include 'SfcFlx_spechum.incf'

! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

REAL (KIND = ireals) FUNCTION SfcFlx_spechum (wvpres, P)

!------------------------------------------------------------------------------
!
! Description:
!
!  Computes specific humidity as function 
!  of water vapour pressure and air pressure. 
!  
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

!_dm Parameters are USEd in module "SfcFlx".
!_nu USE data_parameters , ONLY : &
!_nu     ireals,                  & ! KIND-type parameter for real variables
!_nu     iintegers                  ! KIND-type parameter for "normal" integer variables

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations
 
!  Input (function argument) 
REAL (KIND = ireals), INTENT(IN) ::   &
  wvpres                            , & ! Water vapour pressure [N m^{-2} = kg m^{-1} s^{-2}]
  P                                     ! Air pressure [N m^{-2} = kg m^{-1} s^{-2}]

!==============================================================================
!  Start calculations
!------------------------------------------------------------------------------

! Specific humidity 

SfcFlx_spechum = tpsf_Rd_o_Rv*wvpres/(P-(1._ireals-tpsf_Rd_o_Rv)*wvpres)

!------------------------------------------------------------------------------
!  End calculations
!==============================================================================

END FUNCTION SfcFlx_spechum


! File %M% from Library %Q%
! Version %I% from %G% extracted: %H%
!------------------------------------------------------------------------------

REAL (KIND = ireals) FUNCTION SfcFlx_lwradwsfc (T)

!------------------------------------------------------------------------------
!
! Description:
!
!  Computes the surface long-wave radiation flux
!  as function of temperature. 
!  
!
! Current Code Owner: DWD, Dmitrii Mironov
!  Phone:  +49-69-8062 2705
!  Fax:    +49-69-8062 3721
!  E-mail: dmitrii.mironov@dwd.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2005/11/17 Dmitrii Mironov 
!  Initial release
! !VERSION!  !DATE!     <Your name>
!  <Modification comments>
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
!
! Declarations:
!
! Modules used:

!_dm Parameters are USEd in module "SfcFlx".
!_nu USE data_parameters , ONLY : &
!_nu     ireals,                  & ! KIND-type parameter for real variables
!_nu     iintegers                  ! KIND-type parameter for "normal" integer variables

!==============================================================================

IMPLICIT NONE

!==============================================================================
!
! Declarations
 
!  Input (function argument) 
REAL (KIND = ireals), INTENT(IN) ::   &
  T                                     ! Temperature [K]

!==============================================================================
!  Start calculations
!------------------------------------------------------------------------------

! Long-wave radiation flux [W m^{-2}]

SfcFlx_lwradwsfc = c_lwrad_emis*tpsf_C_StefBoltz*T**4_iintegers

!------------------------------------------------------------------------------
!  End calculations
!==============================================================================

END FUNCTION SfcFlx_lwradwsfc


!==============================================================================

!==============================================================================
!include 'SfcFlx_wvpreswetbulb.incf'

!==============================================================================

END MODULE SfcFlx