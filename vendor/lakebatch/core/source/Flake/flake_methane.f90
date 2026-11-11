MODULE METHANE_SIMPLE

use DATA_PARAMETERS, only : ireals, iintegers

implicit none

real(kind=ireals), parameter :: cw    = 3990.          !> Specific heat of water,  J/(kg*K)
real(kind=ireals), parameter :: row0 = 1.E+3           !> Reference water density, kg/m**3
real(kind=ireals), parameter :: niu_wat = 1.307d-6     !> Molecular viscosity of water at T = 10 C, m**2/s
real(kind=ireals), parameter :: g = 9.80665            !> Reference acceleration due to gravity, m/s**2
real(kind=ireals), parameter :: Kelvin0 = 273.16       !> Temperature of melting point, Kelvin
real(kind=ireals), parameter :: pres_ref = 1013.5*100. !> Atmospheric reference pressure, Pa
real(kind=ireals), parameter :: po2_p = 0.19           !> The ratio of atmospheric oxygen pressure to total atmospheric pressure, n/d

real(kind=ireals), parameter :: por     = 0.4          !> Soil porosity, n/d
real(kind=ireals), parameter :: rhosoil = 1.2E+3*(1.-por) + row0*por      !> Bulk soil density, kg/m**3
real(kind=ireals), parameter :: gammas =  0.4          !> Critical degree of soil saturation for methane bubbling, n/d
real(kind=ireals), parameter :: tortuosity_coef = 0.66 !> The tortuosity of soil pores, n/d

real(kind=ireals), parameter :: diff_hypo0 = 0.561/(cw*row0)*1.E-2 !> Diffusion coefficient in hypolimnion, set to molecular diffusivity, m**2/s
real(kind=ireals), parameter :: alpha0 = 3. !> The rate of decay of methane production with depth 
                                            !! in bottom sediments, m**(-1)

real(kind=ireals), parameter :: q10 = 2.        !> Temperature dependency parameter of methane production, n/d
real(kind=ireals), parameter :: Prod0 = 4.E-8   !> A constant, controlling the magnitude of methane production
                                                !! in sediments, mol/(m**3*s)

real(kind=ireals), parameter :: Henry_const0_ch4 = 1.3d-5   !> Henry constant for methane at the reference temperature, mol/(m**3*Pa)
real(kind=ireals), parameter :: Henry_temp_dep_ch4 = 1.7d+3 !> The temperature dependence of Henry constant for methane, K
real(kind=ireals), parameter :: Henry_temp_ref = 298.15     !> The reference temperature for Henry constants, K
real(kind=ireals), parameter :: Henry_const0_o2 = 1.3d-5    !> Henry constant for oxygen at the reference temperature, mol/(m**3*Pa)
real(kind=ireals), parameter :: Henry_const0_n2 = 6.2d-6    !> Henry constant for nitrogen at the reference temperature, mol/(m**3*Pa)
real(kind=ireals), parameter :: Henry_temp_dep_o2 = 1.7d+3  !> The temperature dependence of Henry constant for oxygen, K
real(kind=ireals), parameter :: Henry_temp_dep_n2 = 1.3d+3  !> The temperature dependence of Henry constant for nitrogen, K

real(kind=ireals), parameter :: n2_atm0 = 4.96d-1  !> Concentration of nitrogen in water, in equilibrium with
                                                   !> atmospheric partial nitrogen pressure (80046 Pa), 
                                                   !> according to Henry law with reference constant, mol/m**3
real(kind=ireals), parameter :: n2_exp_decay = 5.  !> The decay rate in exponential law for nitrogen conecntration in soil, m**(-1)
                                                   !> 50. in (Bazhin, 2001)


real(kind=ireals), parameter :: molm3tomgl_o2 = 32.
real(kind=ireals), parameter :: molm3tomgl_ch4 = 16.
real(kind=ireals), parameter :: k_ch4 = 0.6 / molm3tomgl_ch4 !>(Liikanen et al. 2002; Lofton et al. 2013)
                                                             !!0.44 ! Michaelis constant for methane in methane oxidation,
                                                             !! after Arah&Stephen (1998), Nedwell&Watson (1995) 
real(kind=ireals), parameter :: k_o2 = 0.672 / molm3tomgl_o2 !> (Lidstrom and Somers, 1984)
                                                             !!0.33  ! Michaelis constant for oxygen in methane oxidation, 
                                                             !! after Arah&Stephen (1998), Nedwell&Watson (1995)
real(kind=ireals), parameter :: Vmaxw = 1.d-1/86400. !> Reaction potential in oxygen-saturated Michaelis-Menten kinetics, 
                                                     !! after (Liikanen et al., 2002)


contains
!> Function HENRY_CONST calculates the Henry constant of a substance for a given temperature
FUNCTION HENRY_CONST(henry_const0, temp_dep, temp_ref, temp) !, radius)

implicit none

real(kind=ireals) :: HENRY_CONST

! Input variables
real(kind=ireals), intent(in) :: henry_const0 !> Henry constant at the reference temperature, mol/(m**3*Pa)
real(kind=ireals), intent(in) :: temp_dep     !> Temperature dependence (enthalpy solution devided by universal gas constant), K
real(kind=ireals), intent(in) :: temp_ref     !> Reference temperature, K
real(kind=ireals), intent(in) :: temp         !> Temperature, K
! real(kind=ireals), intent(in) :: radius ! Curvature radius, m, positive for drops, negative for bubbles and zero for flat surface


HENRY_CONST = henry_const0*exp(temp_dep*(1./temp-1./temp_ref))
! Effect of bubble surface curvatue on saturation pressure
!  if (radius /= 0.d0) HENRY_CONST = HENRY_CONST* &
!  & exp(-2.*surf_tension_wat*mol_vol_water/(radius*R_univ*temp))

END FUNCTION HENRY_CONST


FUNCTION SCHMIDT_NUMBER_METHANE(tempC)

implicit none

real(kind=ireals) :: SCHMIDT_NUMBER_METHANE

! Input variables      
real(kind=ireals), intent(in) :: tempC

! Local variables
real(kind=ireals), parameter :: const1 = 1.898d+3
real(kind=ireals), parameter :: const2 = -1.101d+2
real(kind=ireals), parameter :: const3 = 2.834
real(kind=ireals), parameter :: const4 = -2.791d-2

SCHMIDT_NUMBER_METHANE = &
& const1 + const2*tempC + const3*tempC**2 + const4*tempC**3

END FUNCTION SCHMIDT_NUMBER_METHANE


!> Function PISTON_VELOCITY_HEISKANEN calculates gas exchange constant (piston velocity) for methane after Heiskanen et al., 2014
FUNCTION PISTON_VELOCITY_HEISKANEN(wstar,U,temp) result(k)

implicit none

!> Input variables
real(kind=ireals), intent(in) :: wstar !> Convective Deardorff velocity scale in the mixed layer, m/s
real(kind=ireals), intent(in) :: U     !> Wind speed, m/s
real(kind=ireals), intent(in) :: temp  !> Water temperature, Kelvin

!> Local variables
real(kind=ireals), parameter :: C1 = 1.5E-4, C2 = 7.E-2
real(kind=ireals) :: k

k = sqrt((C1*U)**2 + (C2*wstar)**2)*SCHMIDT_NUMBER_METHANE(temp-Kelvin0)**(-0.5)

END FUNCTION PISTON_VELOCITY_HEISKANEN


!> Function calculates the thickness of the bottom viscous sublayer
FUNCTION THICKNESS_VISC_SUBLAYER(ustar,wstar) result(hvisc)

implicit none

!Input variables
real(kind=ireals), intent(in) :: wstar !> Convective Deardorff velocity scale in the mixed layer, m/s
real(kind=ireals), intent(in) :: ustar !> Friction velocity scale in the mixed layer, m/s

!Local variables
real(kind=ireals), parameter :: C  = 8.5 !> 5 -- 11.6 (Wengrove and Foster, GRL 2014, and references therein)
real(kind=ireals), parameter :: C1 = 1.  !> @todo: to be checked in literature
real(kind=ireals) :: hvisc

hvisc = C*niu_wat/sqrt(ustar*ustar + C1 * wstar*wstar)

END FUNCTION THICKNESS_VISC_SUBLAYER

!> Function DIFF_WATER_METHANE calculates the molecular diffusivity 
!! of methane dissolved in liquid water, m**2/s
!! (Broecker and Peng, 1974)
FUNCTION DIFF_WATER_METHANE(temp_C)

implicit none

real(kind=ireals) :: DIFF_WATER_METHANE

! Input variables
real(kind=ireals), intent(in) :: temp_C ! temperature, degrees Celsius

! Local variables
real(kind=ireals), parameter :: const1 = 9.798d-10
real(kind=ireals), parameter :: const2 = 2.986d-11
real(kind=ireals), parameter :: const3 = 4.381d-13

DIFF_WATER_METHANE = const1 + const2*temp_C + const3*temp_C*temp_C
 
END FUNCTION DIFF_WATER_METHANE


SUBROUTINE METHANE_MAIN(H,h_ML,h_s,wstar,ustar,U,pres,T_ML,Tb,C_ML,F_difsurf,F_bubble)

implicit none

!> Input variables
real(kind=ireals), intent(in) :: H     !> Lake depth, m
real(kind=ireals), intent(in) :: h_ML  !> Mixed-layer depth, m
real(kind=ireals), intent(in) :: h_s   !> The thickness of bottom sediments layer, m
real(kind=ireals), intent(in) :: wstar !> Convective Deardorff velocity scale in the mixed layer, m/s
real(kind=ireals), intent(in) :: ustar !> Friction velocity scale in the mixed layer, m/s
real(kind=ireals), intent(in) :: U     !> Wind speed, m/s
real(kind=ireals), intent(in) :: pres  !> Atmospheric pressure, Pa
real(kind=ireals), intent(in) :: T_ML  !> Mixed-layer temperature, Kelvin
real(kind=ireals), intent(in) :: Tb    !> Bottom temperature, Kelvin

!> Output variables
real(kind=ireals), intent(out) :: C_ML      !> Methane concentration in the mixed layer, mol/m**3
real(kind=ireals), intent(out) :: F_difsurf !> Surface diffusive methane flux to the atmosphere, mol/(m**3*s)
real(kind=ireals), intent(out) :: F_bubble  !> Bubble methane flux to the atmosphere, mol/(m**3*s)

!Local variables
integer(kind=iintegers), parameter :: model = 2 !> Switch between model formulations

real(kind=ireals) :: k, k_, sin1, cos1, h_ML_eff, dh
real(kind=ireals) :: gas_exch_const        !> Piston velocity for methane at the water surface, m/s
real(kind=ireals) :: C_b                   !> Methane concentration at the lake bottom, mol/m**3
real(kind=ireals) :: C_O2_ML               !> Oxygen concentration in the mixed layer, mol/m**3
real(kind=ireals) :: C_O2_hypo             !> Oxygen concentration in the hypolimnion, mol/m**3
real(kind=ireals) :: C_CH4_Henri           !> Henry constant for methane, mol/(m**3 * Pa)
real(kind=ireals) :: C_N2_Henry            !> Henry constant for nitrogen, mol/(m**3 * Pa)
real(kind=ireals) :: F_difbot              !> Bottom diffusive flux of methane, mol/(m**2*s)
real(kind=ireals) :: Prod                  !> Methane production in bottom sediments, mol/(m**2*s)
real(kind=ireals) :: CH4ppb                !> Atmospheric methane concentration, ppb
real(kind=ireals) :: CH4pres               !> Atmospheric methane pressure, Pa
real(kind=ireals) :: meth_atmo             !> Methane concentration in water, equilibrated with atmospheric concentration, mol/m**3
real(kind=ireals) :: diff_hypo             !> Diffusion coefficient in hypolimnion, m**2/s
real(kind=ireals) :: meth_oxid_const_hypo  !> Methane oxidation rate constant in hypolimnion
real(kind=ireals) :: meth_oxid_const_ML    !> Methane oxidation rate constant in the mixed layer
real(kind=ireals) :: diffsoil              !> Bulk soil diffusivity for methane, n/d

! Limiting hypolimnion thickness from below by the bottom viscous sublayer depth
dh = max (H - h_ML, THICKNESS_VISC_SUBLAYER(ustar,wstar))
h_ML_eff = H - dh

! Calculation of methane concentration in the mixed layer, equilibrated with atmospheric concentration
CH4ppb = 1.8E+3
CH4pres = CH4ppb*1.E-9*pres
C_CH4_Henri = HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, T_ML)
meth_atmo = CH4pres*C_CH4_Henri

! Calculation of oxygen concentration in the mixed layer, equilibrated with atmospheric concentration
C_O2_ML = po2_p*pres*HENRY_CONST(Henry_const0_o2, Henry_temp_dep_o2, Henry_temp_ref, T_ML)

! Methane gas exchange constant, m/s
gas_exch_const = PISTON_VELOCITY_HEISKANEN(wstar,U,T_ML)

diff_hypo = diff_hypo0


! Methane oxidation constant from Michaelis-Menthen formulation, 
! in the mixed-layer, assuming low methane concentrations
meth_oxid_const_ML = Vmaxw/k_ch4*C_O2_ML/(k_o2 + C_O2_ML)
! Methane oxidation constant from Michaelis-Menthen formulation, 
! in the hypolimnion, formally assumung low methane concentrations (strong assumption)
! and mean oxygen concentration in hypolimnion to be a fixed
! fraction of that in the mixed layer
C_O2_hypo = 0.5*C_O2_ML
meth_oxid_const_hypo = Vmaxw/k_ch4*C_O2_hypo/(k_o2 + C_O2_hypo)

k = sqrt(meth_oxid_const_hypo/diff_hypo)
k_ = diff_hypo*k
sin1 = sinh(k*(H - h_ML_eff))
cos1 = cosh(k*(H - h_ML_eff))

! Total methane production in bottom sediments
Prod = Prod0 * q10 ** (0.1*(Tb - Kelvin0)) / alpha0 * &
       & (1. - exp(-alpha0*h_s))

C_CH4_Henri = HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, Tb)
if     (model == 1) then
  ! Calculation of the botton methane concentration
  C_b = ((row0 * g * H + pres) * C_CH4_Henri) !*0.001
  
  C_ML = (k_ * C_b + gas_exch_const * meth_atmo * sin1) / &
       & ( (gas_exch_const + meth_oxid_const_ML * h_ML_eff) * sin1  + k_* cos1 )
  
  F_difbot = k_ * (C_b * cos1 - C_ML) / sin1
elseif (model == 2) then
  diffsoil = tortuosity_coef * DIFF_WATER_METHANE( Tb )  
  C_N2_Henry = HENRY_CONST(Henry_const0_n2, Henry_temp_dep_n2, Henry_temp_ref, Tb)
  F_difbot = diffsoil * por * C_CH4_Henri * gammas * (rhosoil * g + n2_atm0 * n2_exp_decay / C_N2_Henry)
  F_difbot = min(Prod, F_difbot)
  C_ML = (F_difbot + gas_exch_const * meth_atmo * cos1) / &
       & ( (gas_exch_const + meth_oxid_const_ML * h_ML_eff) * cos1  + k_* sin1)  
endif


!print*, F_difbot, Prod
!read*


F_bubble = Prod - F_difbot
F_difsurf = gas_exch_const * (C_ML - meth_atmo)

END SUBROUTINE METHANE_MAIN


END MODULE METHANE_SIMPLE
