 MODULE WATER_DENSITY

 use LAKE_DATATYPES, only : ireals, iintegers
 use PHYS_CONSTANTS, only : &
 & row0
 use NUMERIC_PARAMS, only : small_value


 implicit none

!The coefficients of UNESCO formula
!for water density dependence on temperature

 real(kind=ireals), parameter:: a0 = 800.969d-7
 real(kind=ireals), parameter:: a1 = 588.194d-7
 real(kind=ireals), parameter:: a2 = 811.465d-8
 real(kind=ireals), parameter:: a3 = 476.600d-10

!The coefficients of formula for
!water density dependence on temperature and salinity
!(McCutcheon et al.,Water Quality and Maidment. 
! Handbook on hydrology, 1993)
 
 real(kind=ireals), parameter:: A11 = 288.9414d0
 real(kind=ireals), parameter:: A12 = 508929.2d0
 real(kind=ireals), parameter:: A13 = 68.12963d0
 real(kind=ireals), parameter:: A14 = 3.9863d0
 real(kind=ireals), parameter:: alpha1 = 8.2449d-1
 real(kind=ireals), parameter:: alpha2 = 4.0899d-3
 real(kind=ireals), parameter:: alpha3 = 7.6438d-5
 real(kind=ireals), parameter:: alpha4 = 8.2467d-7
 real(kind=ireals), parameter:: alpha5 = 5.3675d-9
 real(kind=ireals), parameter:: beta1 = 5.724d-3
 real(kind=ireals), parameter:: beta2 = 1.0227d-4
 real(kind=ireals), parameter:: beta3 = 1.6546d-6
 real(kind=ireals), parameter:: gamma1 = 4.8314d-4

!The coefficients for "Kivu" EOS (Schmid et al., 2003)
 real(kind=ireals), parameter :: at1 = 999.843, at2 = 65.4891d-3, at3 = - 8.56272d-3, at4 = 0.059385d-3
 real(kind=ireals), parameter :: beta = 0.75d-3 ! kg/g
 real(kind=ireals), parameter :: beta_co2 = 0.284d-3 ! coefficient for co2, kg/g
 real(kind=ireals), parameter :: beta_ch4 = -1.25d-3 ! coefficient for ch4, kg/g
 real(kind=ireals), parameter :: co2_ref = 0.d0, ch4_ref = 0.d0 ! reference concentrations, kg/g

 real(kind=ireals), save ::  DDENS_DTEMP0, DDENS_DSAL0, DENS_REF

 integer(kind=iintegers), parameter :: eos_Hostetler = 1
 integer(kind=iintegers), parameter :: eos_TEOS = 2
 integer(kind=iintegers), parameter :: eos_Kivu = 3
 integer(kind=iintegers), parameter :: eos_UNESCO = 4
 integer(kind=iintegers), parameter :: eos_McCatcheon = 5

 !Temperature and salinity reference values
 real(kind=ireals), parameter :: temp_ref = 15. ! deg. C
 real(kind=ireals), parameter :: sal_ref = 0. !kg/kg


 contains

 SUBROUTINE SET_DENS_CONSTS(eos)

 implicit none 

 integer(kind=iintegers), intent(in) :: eos
 

 select case (eos)
 case(eos_Hostetler)
   call DDENS_HOSTETLER(temp_ref,DDENS_DTEMP0,DDENS_DSAL0)
   DENS_REF = DENS_HOSTETLER(temp_ref)
 case(eos_TEOS)
   print*, 'Not operational eos: STOP '
   STOP
 case(eos_Kivu)
   call DDENS_KIVU(temp_ref,sal_ref,DDENS_DTEMP0,DDENS_DSAL0)
   DENS_REF = WATER_DENS_TS_KIVU(temp_ref,sal_ref)
 case(eos_UNESCO)
   call DDENS_UNESCO(temp_ref,DDENS_DTEMP0,DDENS_DSAL0)
   DENS_REF = WATER_DENS_T_UNESCO(temp_ref)
 case(eos_McCatcheon)
   DDENS_DTEMP0 = DDENS_DTEMP(temp_ref,sal_ref) 
   DDENS_DSAL0  = DDENS_DSAL (temp_ref,sal_ref)
   DENS_REF = WATER_DENS_TS(temp_ref,sal_ref)
 end select

 END SUBROUTINE SET_DENS_CONSTS


 FUNCTION SET_DDENS_DTEMP(eos,lindens,Temp,Sal)

 implicit none 

 ! Input/output variables
 integer(kind=iintegers), intent(in) :: eos !> Equation of state identifier
 integer(kind=iintegers), intent(in) :: lindens !> Switch for linear density function
 real(kind=ireals), intent(in) :: Temp, Sal

 real(kind=ireals) :: SET_DDENS_DTEMP

 ! Local variables
 real(kind=ireals) :: x

 if (lindens == 0) then
   select case (eos)
   case(eos_Hostetler)
     call DDENS_HOSTETLER(Temp,SET_DDENS_DTEMP,x)
   case(eos_TEOS)
     print*, 'Not operational eos: STOP '
     STOP
   case(eos_Kivu)
     call DDENS_KIVU(Temp,Sal,SET_DDENS_DTEMP,x)
   case(eos_UNESCO)
     call DDENS_UNESCO(Temp,SET_DDENS_DTEMP,x)
   case(eos_McCatcheon)
     SET_DDENS_DTEMP = DDENS_DTEMP(Temp,Sal) 
   end select
 else
   SET_DDENS_DTEMP = DDENS_DTEMP0
 endif

 END FUNCTION SET_DDENS_DTEMP


 FUNCTION SET_DDENS_DSAL(eos,lindens,Temp,Sal)

 implicit none 

 ! Input/output variables
 integer(kind=iintegers), intent(in) :: eos !> Equation of state identifier
 integer(kind=iintegers), intent(in) :: lindens !> Switch for linear density function
 real(kind=ireals), intent(in) :: Temp, Sal

 real(kind=ireals) :: SET_DDENS_DSAL

 ! Local variables
 real(kind=ireals) :: x

 if (lindens == 0) then
   select case (eos)
   case(eos_Hostetler)
     call DDENS_HOSTETLER(Temp,x,SET_DDENS_DSAL)
   case(eos_TEOS)
     print*, 'Not operational eos: STOP '
     STOP
   case(eos_Kivu)
     call DDENS_KIVU(Temp,Sal,x,SET_DDENS_DSAL)
   case(eos_UNESCO)
     call DDENS_UNESCO(Temp,x,SET_DDENS_DSAL)
   case(eos_McCatcheon)
     SET_DDENS_DSAL = DDENS_DSAL(Temp,Sal) 
   end select
 else
   SET_DDENS_DSAL = DDENS_DSAL0
 endif

 END FUNCTION SET_DDENS_DSAL


 !>Subroutine DENSITY_W calculates density profile
 SUBROUTINE DENSITY_W(M,eos,lindens,Tw,Sal,preswat,row)

 integer(kind=iintegers), intent(in) :: eos, M, lindens

 real(kind=ireals), intent(in) :: Tw(1:M+1), Sal(1:M+1), preswat(1:M+1)

 real(kind=ireals), intent(out) :: row(1:M+1)

 integer(kind=iintegers) :: i

 !Calculation of water density profile at the current timestep
 if (lindens == 0) then
   select case (eos)
   case(eos_Hostetler)
     do i = 1, M+1
       row(i) = DENS_HOSTETLER(Tw(i))
     enddo
   case(eos_TEOS)
     do i = 1, M+1
       row(i) = GSW_RHO(Sal(i),Tw(i),preswat(i))
     enddo
   case(eos_Kivu)
     do i = 1, M+1
       row(i) = WATER_DENS_TS_KIVU(Tw(i),Sal(i)) !0.d0
     enddo
   case(eos_UNESCO)
     do i = 1, M+1
       row(i) = WATER_DENS_T_UNESCO(Tw(i)) !0.d0
     enddo
   case(eos_McCatcheon)
     do i = 1, M+1
       row(i) = WATER_DENS_TS(Tw(i),Sal(i)) ! McCutcheon et al., 1993
     enddo
   end select
 else
   do i = 1, M+1
     row(i) = DENS_REF + DDENS_DTEMP0*(Tw(i) - temp_ref) + &
     &                   DDENS_DSAL0 *(Sal(i) - sal_ref) 
   enddo
 endif

 END SUBROUTINE DENSITY_W


 real(kind=ireals) FUNCTION WATER_DENS_T_UNESCO(Temp)

!The function WATER_DENS_T_UNESCO
!returns the density of water, kg/m**3, 
!as a function of temperature 
!according to UNESCO formula
 
 implicit none

 real(kind=ireals), intent(in):: Temp
 
 WATER_DENS_T_UNESCO = &
&  row0*(1+a0+a1*Temp-a2*Temp**2+a3*Temp**3)
 END FUNCTION WATER_DENS_T_UNESCO


 SUBROUTINE DDENS_UNESCO(Temp,ddens_dtemp,ddens_dsal)
 implicit none

 real(kind=ireals), intent(in):: Temp
 real(kind=ireals), intent(out):: ddens_dtemp, ddens_dsal
 
 ddens_dtemp = &
 & row0*(a1 - 2.*a2*Temp + 3.*a3*Temp**2)
 ddens_dsal = 0.

 END SUBROUTINE DDENS_UNESCO


 FUNCTION WATER_DENS_TS_KIVU(Temp,Sal)

! The function WATER_DENS_TS_KIVU
! resturns the density of water, kg/m**3,
! as a function of temperature and salinity
! adjusted for Lake Kivu (Rwanda & DRC) according to
! (Schmid et al. 2003)

! Input variables:
 
! Temp --- water temperature, deg C
! Sal  --- water salinity,    kg/kg
  
  implicit none

  real(kind=ireals), intent(in):: Temp
  real(kind=ireals), intent(in):: Sal

  real(kind=ireals) :: WATER_DENS_TS_KIVU

  real(kind=ireals) :: Sal_g_kg,A,B,C

! Converting salinity units from kg/kg to g/kg
  Sal_g_kg = Sal*1.d+3

  WATER_DENS_TS_KIVU = (at1 + at2*Temp + at3*Temp**2 + at4*Temp**3)* &
  & (1. + beta*Sal_g_kg + beta_co2*co2_ref + beta_ch4*ch4_ref)

  END FUNCTION WATER_DENS_TS_KIVU


  SUBROUTINE DDENS_KIVU(Temp,Sal,ddens_dtemp,ddens_dsal)
  implicit none

  real(kind=ireals), intent(in) :: Temp, Sal
  real(kind=ireals), intent(out) :: ddens_dtemp, ddens_dsal

  real(kind=ireals) :: Sal_g_kg

! Converting salinity units from kg/kg to g/kg
  Sal_g_kg = Sal*1.d+3

  ddens_dtemp = (at2 + 2.*at3*Temp + 3.*at4*Temp**2)* &
  & (1. + beta*Sal_g_kg + beta_co2*co2_ref + beta_ch4*ch4_ref)
  ddens_dsal = (at1 + at2*Temp + at3*Temp**2 + at4*Temp**3)* &
  & beta*1.d+3

  END SUBROUTINE DDENS_KIVU


  real(kind=ireals) FUNCTION WATER_DENS_TS(Temp,Sal)

! The function WATER_DENS_TS
! resturns the density of water, kg/m**3,
! as a function of temperature and salinity
! according to
! (McCutcheon et al.,Water Quality and Maidment. 
!  Handbook on Hydrology, 1993)

! Input variables:
 
! Temp --- water temperature, deg C
! Sal  --- water salinity,    kg/kg
  
  implicit none

  real(kind=ireals), intent(in):: Temp
  real(kind=ireals), intent(in):: Sal

  real(kind=ireals) Sal_g_kg,A,B,C

! Converting salinity units from kg/kg to g/kg
  Sal_g_kg = Sal*1.d+3

! The first term, dependent on temperature      
  WATER_DENS_TS = row0 * &
 & ( 1.-(Temp+A11)*(Temp-A14)**2/(A12*(Temp+A13) ) )

  A =  alpha1         - alpha2*Temp    + alpha3*Temp**2 &
 &    -alpha4*Temp**3 + alpha5*Temp**4

  B = -beta1          + beta2*Temp     - beta3*Temp**2

  C = gamma1

! Adding the second term, dependent on temperature and salinity
  WATER_DENS_TS = WATER_DENS_TS + &
 & A*Sal_g_kg + B*Sal_g_kg**1.5 + C*Sal_g_kg**2.

  END FUNCTION WATER_DENS_TS


  real(kind=ireals) FUNCTION DDENS_DTEMP(Temp,Sal)

! The function DDENS_DTEMP
! returns the derivative of water density
! on temperature, kg/(m**3*C)
  
! Input variables:
 
! Temp --- water temperature, deg C
! Sal  --- water salinity,    kg/kg

  implicit none

  real(kind=ireals), intent(in):: Temp
  real(kind=ireals), intent(in):: Sal      

  real(kind=ireals) Sal_g_kg,A,B,C

  DDENS_DTEMP = &
  & -(row0/A12)*(Temp-A14)/( (Temp+A13)**2 ) * &
  & ( (Temp+A13)*( (Temp-A14)+2.*(Temp+A11) ) - &
  &  (Temp+A11)*(Temp-A14) )

! Converting salinity units from kg/kg to g/kg
  Sal_g_kg = Sal*1.d+3

  A = - alpha2 + 2.*alpha3*Temp - 3.*alpha4*Temp**2 + 4.*alpha5*Temp**3

  B = + beta2  - 2.*beta3*Temp

  DDENS_DTEMP = DDENS_DTEMP + A*Sal_g_kg + B*Sal_g_kg**1.5

  END FUNCTION DDENS_DTEMP
  

  real(kind=ireals) FUNCTION DDENS_DSAL(Temp,Sal)

! The function DDENS_DSAL
! returns the derivative of water density
! on salinity, , kg/(m**3*kg/kg)
  
! Input variables:
 
! Temp --- water temperature, deg C
! Sal  --- water salinity,    kg/kg

  implicit none

  real(kind=ireals), intent(in):: Temp
  real(kind=ireals), intent(in):: Sal

  real(kind=ireals) Sal_g_kg,A,B,C

  A =   alpha1         - alpha2*Temp    + alpha3*Temp**2 &
  &   - alpha4*Temp**3 + alpha5*Temp**4

  B = - beta1          + beta2*Temp     - beta3*Temp**2

  C = gamma1

! Converting salinity units from kg/kg to g/kg
  Sal_g_kg = Sal*1.d+3

  DDENS_DSAL = &
 & A + 1.5*B*Sal_g_kg**0.5 + 2.*C*Sal_g_kg

  DDENS_DSAL = DDENS_DSAL*1.d+3

  END FUNCTION DDENS_DSAL
  
  
  FUNCTION DENS_HOSTETLER(temp)
  
! Function DENS_HOSTETLER calculates the density of water
! dependent on temperature only, following Hostetler and Bartlein, 1990

  implicit none

  real(kind=ireals) :: DENS_HOSTETLER
  
  real(kind=ireals), intent(in) :: temp
  
! Parameters      
  real(kind=ireals), parameter :: row0 = 1.d+3
  real(kind=ireals), parameter :: temp0 = 3.85
  real(kind=ireals), parameter :: const = 1.9549d-5
  real(kind=ireals), parameter :: const_power = 1.68
  
  DENS_HOSTETLER = row0*(1.-const*abs(temp-temp0)**const_power)
  
  END FUNCTION DENS_HOSTETLER


  SUBROUTINE DDENS_HOSTETLER(Temp,ddens_dtemp,ddens_dsal)
  implicit none

! Parameters      
  real(kind=ireals), parameter :: row0 = 1.d+3
  real(kind=ireals), parameter :: temp0 = 3.85
  real(kind=ireals), parameter :: const = 1.9549d-5
  real(kind=ireals), parameter :: const_power = 1.68

  real(kind=ireals), intent(in):: Temp
  real(kind=ireals), intent(out):: ddens_dtemp, ddens_dsal
  
  ddens_dtemp = row0*( - const_power*const*abs(temp-temp0)**(const_power-1)* &
  & sign(1._ireals,temp-temp0))
  ddens_dsal = 0.

  END SUBROUTINE DDENS_HOSTETLER


!--------------------------------------------------------------------------

!--------------------------------------------------------------------------
! density and enthalpy, based on the 48-term expression for density from TEOS-2010
!--------------------------------------------------------------------------

!==========================================================================
FUNCTION GSW_RHO(sa_,ct_,p_) 
!==========================================================================

!  Calculates in-situ density from Absolute Salinity and Conservative 
!  Temperature, using the computationally-efficient 48-term expression for
!  density in terms of SA, CT and p (McDougall et al., 2011).
!
! sa     : Absolute Salinity                               [kg/kg]
! ct     : Conservative Temperature                        [deg C]
! p      : sea pressure                                    [Pa]
! 
! gsw_rho  : in-situ density (48 term equation)

implicit none

integer, parameter :: r14 = selected_real_kind(14,30)

real (r14), parameter :: v01 =  9.998420897506056d2, v02 =  2.839940833161907d0
real (r14), parameter :: v03 = -3.147759265588511d-2, v04 =  1.181805545074306d-3
real (r14), parameter :: v05 = -6.698001071123802d0, v06 = -2.986498947203215d-2
real (r14), parameter :: v07 =  2.327859407479162d-4, v08 = -3.988822378968490d-2
real (r14), parameter :: v09 =  5.095422573880500d-4, v10 = -1.426984671633621d-5
real (r14), parameter :: v11 =  1.645039373682922d-7, v12 = -2.233269627352527d-2
real (r14), parameter :: v13 = -3.436090079851880d-4, v14 =  3.726050720345733d-6
real (r14), parameter :: v15 = -1.806789763745328d-4, v16 =  6.876837219536232d-7
real (r14), parameter :: v17 = -3.087032500374211d-7, v18 = -1.988366587925593d-8
real (r14), parameter :: v19 = -1.061519070296458d-11, v20 =  1.550932729220080d-10
real (r14), parameter :: v21 =  1.0d0, v22 =  2.775927747785646d-3, v23 = -2.349607444135925d-5
real (r14), parameter :: v24 =  1.119513357486743d-6, v25 =  6.743689325042773d-10
real (r14), parameter :: v26 = -7.521448093615448d-3, v27 = -2.764306979894411d-5
real (r14), parameter :: v28 =  1.262937315098546d-7, v29 =  9.527875081696435d-10
real (r14), parameter :: v30 = -1.811147201949891d-11, v31 = -3.303308871386421d-5
real (r14), parameter :: v32 =  3.801564588876298d-7, v33 = -7.672876869259043d-9
real (r14), parameter :: v34 = -4.634182341116144d-11, v35 =  2.681097235569143d-12
real (r14), parameter :: v36 =  5.419326551148740d-6, v37 = -2.742185394906099d-5
real (r14), parameter :: v38 = -3.212746477974189d-7, v39 =  3.191413910561627d-9
real (r14), parameter :: v40 = -1.931012931541776d-12, v41 = -1.105097577149576d-7
real (r14), parameter :: v42 =  6.211426728363857d-10, v43 = -1.119011592875110d-10
real (r14), parameter :: v44 = -1.941660213148725d-11, v45 = -1.864826425365600d-14
real (r14), parameter :: v46 =  1.119522344879478d-14, v47 = -1.200507748551599d-15
real (r14), parameter :: v48 =  6.057902487546866d-17 

real (kind=ireals), intent(in) :: sa_, ct_, p_
real (r14) :: sa, ct, p, sqrtsa, v_hat_denominator, &
& v_hat_numerator, gsw_rho

sa = sa_*1.d+3 ! Converting kg/kg to g/kg
p = p_*1.d-4   ! Converting pressure from Pa to dbar
ct = ct_

sqrtsa = sqrt(sa)

v_hat_denominator = v01 + ct*(v02 + ct*(v03 + v04*ct))  &
         + sa*(v05 + ct*(v06 + v07*ct) &
         + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct)))) &
         + p*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct) &
         + p*(v17 + ct*(v18 + v19*ct) + v20*sa))

v_hat_numerator = v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct))) &
       + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa &
       + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct)))))  &
       + p*(v37 + ct*(v38 + ct*(v39 + v40*ct))  &
       + sa*(v41 + v42*ct) &
       + p*(v43 + ct*(v44 + v45*ct + v46*sa) &
       + p*(v47 + v48*ct)))

GSW_RHO = v_hat_denominator/v_hat_numerator

return
END FUNCTION GSW_RHO


  END MODULE WATER_DENSITY


  MODULE PHYS_FUNC

  use LAKE_DATATYPES, only : ireals, iintegers
  use NUMERIC_PARAMS, only : small_value
  use NUMERICS, only : BOUND_FROM_BELOW

  contains
  real(kind=ireals) FUNCTION TURB_DENS_FLUX(tempflux,salflux,Temp,Sal)

! Function TURB_DENS_FLUX            _____
! returns the turbulent density flux w'ro' in water
! at given temperature, 
!          salinity,         ____
!          temperature flux  w'T'
!                            ____
!      and salinity    flux  w's'

! Input variables:
! tempflux --- kinematic heat flux, m*C/s
! salflux  --- salinity       flux, m*(kg/kg)/s
! Temp     --- temperature        , C
! Sal      --- salinity           , kg/kg

  use water_density, only : &
 & ddens_dtemp, &
 & ddens_dsal
  
  implicit none
  
  real(kind=ireals), intent(in):: tempflux
  real(kind=ireals), intent(in):: salflux
  real(kind=ireals), intent(in):: Temp
  real(kind=ireals), intent(in):: Sal

  TURB_DENS_FLUX = &
 & ddens_dtemp(Temp,Sal)*tempflux + &
 & ddens_dsal (Temp,Sal)*salflux

  END FUNCTION TURB_DENS_FLUX

 
!  real(kind=ireals) FUNCTION dirdif()
!  use atmos, only:
! & shortwave
! implicit none
!  real(kind=ireals) cloud
!  real(kind=ireals) dirdif0,b,S0,sinh0

!  SAVE

!  b=1./3.
!  S0=1367.
!  cloud = 0.
 
!  dirdif0 = (shortwave-b*S0*sinh0())/(b*(S0*sinh0()-shortwave))
!  dirdif = dirdif0*(1.-cloud)
!  dirdif = max(dirdif,0.d0)
  
!  END FUNCTION dirdif


  FUNCTION SINH0(year,month,day,hour,phi)

! SINH0 is sine of solar angle ( = cosine of zenith angle)   
  use TIME_LAKE_MOD, only : JULIAN_DAY  
  implicit none
  
  real(kind=ireals) :: sinh0
  
  integer(kind=iintegers), intent(in) :: year
  integer(kind=iintegers), intent(in) :: month
  integer(kind=iintegers), intent(in) :: day

  real(kind=ireals)   , intent(in) :: hour
  real(kind=ireals)   , intent(in) :: phi

  real(kind=ireals) delta
  real(kind=ireals) theta
  real(kind=ireals) pi
   real(kind=ireals) phi_rad

  integer(kind=iintegers) nday
 
  pi=4.*datan(1.d0)

  nday  = JULIAN_DAY(year,month,day)

  delta = 23.5d0*pi/180.d0*cos(2*pi*(float(nday)-173.d0)/365.d0)
  theta = pi*(hour-12.d0)/12.d0
  phi_rad = phi*pi/180.d0
  sinh0 = sin(phi_rad)*sin(delta) + &
  & cos(phi_rad)*cos(delta)*cos(theta)
  sinh0=max(sinh0,0.d0) 

  END FUNCTION SINH0


  !> Function LONGWAVE_RADIATION returns incident 
  !! longwave (atmospheric) radiation on the earth surface
  FUNCTION LONGWAVE_RADIATION(tempair,humair,pressure,cloud_sky_fraction) result(LR)
  use UNITS, only : Pa_to_hPa
  use PHYS_CONSTANTS, only : Kelvin0, Rd_d_Rwv, sigma
  implicit none
  !> Surface air temperature, Celsius
  real(kind=ireals), intent(in) :: tempair 
  !> Surface air specific humidity, kg/kg
  real(kind=ireals), intent(in) :: humair  
  !> Surface atmospheric pressure, Pa
  real(kind=ireals), intent(in) :: pressure
  !> Cloud-sky fraction, 0-1, n/d
  real(kind=ireals), intent(in) :: cloud_sky_fraction 
  
  !> Output variables
  real(kind=ireals) :: LR
  !> Local variables
  real(kind=ireals) :: emis, a1, a2, a3, parpres, Rwv_d_Rd

  integer(kind=iintegers), parameter :: CAN = 1 !(Anstrom, 1915; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: CBR = 2 !(Brunt, 1932; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: CSW = 3 !(Swinbank, 1963; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: CIJ = 4 !(Idso and Jackson, 1969; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: CBT = 5 !(Brutsaert, 1975, 1982; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: CID = 6 !(Idso, 1981; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: CKZ = 7 !(Konzelmann et al., 1994; Marthews et al., Theor Appl Climatol, 2012)
  integer(kind=iintegers), parameter :: nCSE = CAN !switch for clear-sky emissivity parameterization

  Rwv_d_Rd = 1./Rd_d_Rwv
  parpres = humair*pressure*Rwv_d_Rd/(1. + humair*(Rwv_d_Rd - 1.))*Pa_to_hPa

  select case(nCSE)
  case(CAN)
    a1 = 0.83
    a2 = 0.18
    a3 = 0.067
    emis = a1 - a2*10**( - a3*parpres)
  case(CBR)
    a1 = 0.51
    a2 = 0.066
    emis = a1 + a2*sqrt(parpres)
  case(CSW)
    a1 = 0.0000092
    emis = a1*(tempair + Kelvin0)**2
  case(CIJ)
    a1 = 0.261
    a2 = 0.000777
    emis = 1. - a1*exp( - a2*tempair**2)
  case(CBT)
    a1 = 1.24
    emis = a1*(parpres/(tempair + Kelvin0))**(1./7.)
  case(CID)
    a1 = 0.7
    a2 = 0.0000595
    a3 = 1500.
    emis = a1 + a2*parpres*exp(a3/(tempair + Kelvin0))
  case(CKZ)
    a1 = 0.23
    a2 = 0.484
    emis = a1 + a2*(parpres/(tempair + Kelvin0))**(1./8.)
  end select

  !Correction of emissivity for cloudiness
  a1 = 0.22
  emis = emis*(1. + a1*cloud_sky_fraction)
  emis = min(emis,1._ireals)
  LR = emis*sigma*(tempair + Kelvin0)**4

  END FUNCTION LONGWAVE_RADIATION

  !> Function SHORTWAVE_RADIATION returns incident 
  !! shortwave (solar) radiation (direct+diffuse) on the earth surface
  FUNCTION SHORTWAVE_RADIATION(time_group,phi,cloud_sky_fraction) result(SR)
  use PHYS_CONSTANTS, only : solar_constant
  use ARRAYS, only : time_type
  implicit none
  !> Input variables
  type(time_type), intent(in) :: time_group !>The group of time/date variables, time assumed local, not GMT
  real(kind=ireals), intent(in) :: phi !>The latitude, deg.
  real(kind=ireals), intent(in) :: cloud_sky_fraction !> Cloud-sky fraction, 0-1, n/d
  !> Output variables
  real(kind=ireals) :: SR
  !> Local variables
  real(kind=ireals) :: sinsolh, trans_atm, SR_dir, SR_dif, a1, a2, SR_atten

  sinsolh = SINH0(time_group%year,time_group%month,time_group%day,time_group%hour,phi)
  trans_atm = 0.6 ! atmospheric transmissivity for direct solar radiation
  SR_atten = trans_atm**(1./sinsolh)
  !Direct solar radiation at the horizontal surface (Burger's law)
  SR_dir = solar_constant*SR_atten*sinsolh
  !Diffuse solar radiation at the horizontal surface (Liu and Jordan, Solar Energy, 1960; 
  !Jemaa et al., Energy Procedia, 2013)
  a1 = 0.271
  a2 = 0.294
  SR_dif = solar_constant*(a1 - a2*SR_atten)*sinsolh
  SR = SR_dir + SR_dif
  !Correction of global radiation by cloudiness (Kasten and Czeplak, Solar Energy, 1980; 
  !Jemaa et al., Energy Procedia, 2013)
  a1 = 0.75
  a2 = 3.4
  SR = SR*(1.-a1*cloud_sky_fraction**a2)

  END FUNCTION SHORTWAVE_RADIATION


  real(kind=ireals) FUNCTION QS(phase,t,p)
  
! QS - specific humidity, kg/kg, for saturated water vapour

  implicit none
  real(kind=ireals) t,p,aw,bw,ai,bi,a,b,es
  integer(kind=iintegers) phase
!phase = 1 - liquid, 0 - ice
aw = 7.6326    
bw = 241.9    
ai = 9.5  
bi = 265.5 
a  = phase*aw+(1-phase)*ai
b  = phase*bw+(1-phase)*bi
es=610.7*10.**(a*t/(b+t))
QS=0.622*es/p
END FUNCTION

!> Derivative of saturated specific humidity on temperature
FUNCTION DQSDT(phase,t,p)
! QS - specific humidity, kg/kg, for saturated water vapour
implicit none
real(kind=ireals) :: t,p,aw,bw,ai,bi,a,b,qs,es
real(kind=ireals) :: DQSDT
integer(kind=iintegers) :: phase
!phase = 1 - liquid, 0 - ice
aw = 7.6326    
bw = 241.9    
ai = 9.5  
bi = 265.5 
a  = phase*aw+(1-phase)*ai
b  = phase*bw+(1-phase)*bi
es = 610.7*10.**(a*t/(b+t))
qs = 0.622*es/p
DQSDT = qs*a*b*log(10.)/(b+t)**2
END FUNCTION DQSDT


real(kind=ireals) FUNCTION ES(phase,t,p)

!     ES is pressure of saturated water vapour, Pa    

implicit none
real(kind=ireals) t,p,aw,bw,ai,bi,a,b
integer(kind=iintegers) phase
!phase = 1 - liquid, 0 - ice
aw = 7.6326    
bw = 241.9    
ai = 9.5  
bi = 265.5 
a  = phase*aw+(1-phase)*ai
b  = phase*bw+(1-phase)*bi
ES = 610.7*10.**(a*t/(b+t))
END FUNCTION


real(kind=ireals) FUNCTION MELTPNT(C,p,npar)
implicit none

real(kind=ireals), intent(in) :: C ! salinity, kg/kg
real(kind=ireals), intent(in) :: p ! pressure, Pa
integer(kind=iintegers), intent(in) :: npar 

real(kind=ireals), parameter :: dtdc = 66.7, pnorm = 101325
real(kind=ireals), parameter :: Pa2dbar = 1.d-4

select case(npar)
case(0)
  MELTPNT = 0.
case(1)
  MELTPNT = 0. - C*dtdc
case(2)
  ! Jackett et al. 2004 formula
  MELTPNT = FP_THETA(C*1.E+3_ireals, (p-pnorm)*Pa2dbar,'air-sat') ! convertion to 0/00
end select

END FUNCTION MELTPNT

FUNCTION FP_THETA(s,p,sat)

!   potential temperature freezing point of seawater, as in
!   Jackett, McDougall, Feistel, Wright and Griffies (2004), submitted JAOT
!
!   s                : salinity                               (psu)
!   p                : gauge pressure                         (dbar)
!                      (absolute pressure - 10.1325 dbar)
!   sat              : string variable
!                       'air-sat'  - air saturated water       
!                       'air-free' - air free water
!
!   fp_theta         : potential freezing temperature         (deg C, ITS-90)
!
!   check value      : fp_theta(35,200,'air-sat')   = -2.076426227617581 deg C
!                      fp_theta(35,200,'air-free') = -2.074408175943127 deg C
!
!   DRJ on 2/6/04


implicit real(kind=ireals)(a-h,o-z)

character*(*) sat


sqrts = sqrt(s)

tf_num =                          2.5180516744541290d-03    +     &
                              s*(-5.8545863698926184d-02    +     &
                          sqrts*( 2.2979985780124325d-03    -     &
                         sqrts *  3.0086338218235500d-04))  +     &
                              p*(-7.0023530029351803d-04    +     &
                              p*( 8.4149607219833806d-09    +     &
                             s *  1.1845857563107403d-11));

tf_den =                          1.0000000000000000d+00    +     &
                              p*(-3.8493266309172074d-05    +     &
                             p *  9.1686537446749641d-10)   +     &
                      s*s*sqrts*  1.3632481944285909d-06 


fp_theta = tf_num/tf_den;


if(sat.eq.'air-sat') then 
    fp_theta = fp_theta          -2.5180516744541290d-03    +     &
                             s *  1.428571428571429d-05
elseif(sat.eq.'air-free') then  
    continue
else  
    print *, '***   Error in fp_theta.f90: invalid third argument   ***'
    print *
    stop
endif


return
END FUNCTION FP_THETA
  
  FUNCTION WATER_FREEZE_MELT(temperature, grid_size, &
 & melting_point, switch)
 
! The function WATER_FREEZE_MELT checks, if the
! heat storage of a grid cell is larger or equal
! the latent heat, necessary for phase transition
  
  use PHYS_CONSTANTS, only : &
 & cw_m_row0, &
 & ci_m_roi, &
 & row0_m_Lwi, &
 & roi_m_Lwi
  use NUMERIC_PARAMS, only : &
 & min_ice_thick, &
 & min_water_thick, &
 & T_phase_threshold
 
  implicit none
  
! Input variables
  real(kind=ireals), intent(in) :: temperature ! Temperature, Celsius
  real(kind=ireals), intent(in) :: grid_size
  real(kind=ireals), intent(in) :: melting_point
  
  integer(kind=iintegers), intent(in) :: switch ! +1 for water -> ice transition
                                   ! -1 for ice -> water transition
  logical :: WATER_FREEZE_MELT
  
  if (switch == +1) then
    if ( ( melting_point - T_phase_threshold - temperature) * &
 &   cw_m_row0*grid_size > min_ice_thick*roi_m_Lwi .or. &
 &   ( melting_point - T_phase_threshold - temperature) > 0.2d0) &
 &   then
      WATER_FREEZE_MELT = .true.
    else
      WATER_FREEZE_MELT = .false.
    endif
  elseif (switch == -1) then
    if ( (temperature - T_phase_threshold - melting_point) * &
 &   ci_m_roi*grid_size > min_water_thick*row0_m_Lwi .or. &
 &   (temperature - T_phase_threshold - melting_point) > 0.2d0) &
 &   then
      WATER_FREEZE_MELT = .true.
    else
      WATER_FREEZE_MELT = .false.
    endif     
  else
    write(*,*) 'Wrong switch in WATER_FREEZE_MELT: STOP'
    STOP
  endif
  
  END FUNCTION WATER_FREEZE_MELT


  !> Subroutine TURB_SCALES calculates turbulent scales, both bulk and local
  SUBROUTINE TURB_SCALES(gsp, ls, bathymwater, wst, RadWater, &
  & k_turb_T_flux, T_massflux, row, eflux0_kinem, &
  & turb_density_flux, Buoyancy0, tau, kor, i_maxN, H_mixed_layer, maxN, w_conv_scale, &
  & T_conv_scale, Wedderburn, LakeNumber, Rossby_rad, ThermThick, ReTherm, RiTherm, trb)
 
  use DRIVING_PARAMS, only : M, eos, lindens
  use PHYS_CONSTANTS, only : g, row0, cw_m_row0, g_d_Kelvin0, niu_wat
  use ARRAYS_BATHYM, only : bathym, layers_type
  use ARRAYS_GRID, only : gridspacing_type
  use WATER_DENSITY, only : DDENS_DTEMP0, DDENS_DSAL0
  use NUMERIC_PARAMS, only : small_value
  use ARRAYS_WATERSTATE, only : waterstate_type
  use RADIATION, only : rad_type
  use ARRAYS_TURB, only : turb_type
  use WATER_DENSITY, only : SET_DDENS_DTEMP, SET_DDENS_DSAL

  implicit none

! Input variables
  type(gridspacing_type), intent(in) :: gsp
  type(layers_type), intent(in) :: ls
  type(bathym), intent(in) :: bathymwater(1:M+1)
  type(waterstate_type), intent(in) :: wst
  type(rad_type), intent(in) :: RadWater

  real(kind=ireals), intent(in) :: k_turb_T_flux(1:M)
  real(kind=ireals), intent(in) :: T_massflux(1:M)
  real(kind=ireals), intent(in) :: row(1:M+1)
  real(kind=ireals), intent(in) :: eflux0_kinem
  real(kind=ireals), intent(in) :: turb_density_flux
  real(kind=ireals), intent(in) :: Buoyancy0
  real(kind=ireals), intent(in) :: tau
  real(kind=ireals), intent(in) :: kor

! Output variables
  integer(kind=iintegers), intent(out) :: i_maxN

  real(kind=ireals), intent(out) :: H_mixed_layer
  real(kind=ireals), intent(out) :: maxN
  real(kind=ireals), intent(out) :: w_conv_scale
  real(kind=ireals), intent(out) :: T_conv_scale
  real(kind=ireals), intent(out) :: Wedderburn !Wedderburn number
  real(kind=ireals), intent(out) :: LakeNumber
  real(kind=ireals), intent(out) :: Rossby_rad !Internal Rossby radius
  real(kind=ireals), intent(out) :: ThermThick !Thermocline thickness
  real(kind=ireals), intent(out) :: ReTherm    !Reynolds number in thermocline
  real(kind=ireals), intent(out) :: RiTherm    !Richardson number in thermocline

  type(turb_type), intent(inout) :: trb
  
! External functions
  real(kind=ireals), external:: DZETA
  
  real(kind=ireals), parameter :: Ttherm0 = 14., Ttherm1 = 8. !temperatures of top and bottom of thermocline, C

! Auxilliary variables
  integer(kind=iintegers) :: maxlocat, i
  real(kind=ireals), allocatable :: bvf(:)
  real(kind=ireals) :: zv, zm, x, y, x1, x2
  real(kind=ireals) :: u1, u2, v1, v2, Sal1, Sal2

! Mixed-layer depth - depth of maximal heat flux
!  maxlocat = 1
!  do i = 2, M
!    if (k_turb_T_flux(i) + T_massflux(i) > &
!    & k_turb_T_flux(maxlocat) + T_massflux(maxlocat)) maxlocat = i
!  enddo
!  H_mixed_layer = dzeta_05int(maxlocat)*h1

! Mixed-layer depth - depth of maximal Bruent-Vaisala frequency
!  allocate (bvf(2:M-1))      
!  do i = 2, M-1
!    bvf(i) = g/row0*(row(i+1) - row(i-1))/(h1*(ddz(i-1) + ddz(i)))
!  enddo
!  maxlocat = maxloc(bvf,1) + lbound(bvf,1) - 1
!  deallocate (bvf)
!  H_mixed_layer = dzeta_int(maxlocat)*h1

  call MIXED_LAYER_CALC(row,gsp%ddz,gsp%ddz2,gsp%dzeta_int,gsp%dzeta_05int, &
  & ls%h1,M,i_maxN,H_mixed_layer,maxN)

  w_conv_scale = max( (Buoyancy0 - &
  & (RadWater%integr(1) - RadWater%integr(i_maxN))*g_d_Kelvin0/cw_m_row0) * &
  & H_mixed_layer, 0._ireals)**(1._ireals/3._ireals) 

  T_conv_scale = -eflux0_kinem/(w_conv_scale + small_value)

  Wedderburn = max(min(g*(row(M+1) - row(1))*H_mixed_layer*H_mixed_layer/&
  & ((tau+small_value)*max(bathymwater(1)%Lx,bathymwater(1)%Ly)),&
  & 1.e+2_ireals),1.e-2_ireals)

  ! Depth of volume center (zv) and of center of mass (zm)
  zm = 0.; x1 = 0.
  zv = 0.; x2 = 0.
  do i = 1, M+1
    y = gsp%ddz05(i-1)*ls%h1*bathymwater(i)%area_int
    x = y*gsp%z_full(i)
    zm = zm + row(i)*x
    x1 = x1 + row(i)*y
    zv = zv + x
    x2 = x2 + y
  enddo
  zm = zm/x1
  zv = zv/x2

  LakeNumber = (zm - zv)*ls%vol*row0*g*2.*H_mixed_layer / &
  &            (zv*(tau+small_value)*bathymwater(1)%area_int* &
  &            max(bathymwater(1)%Lx,bathymwater(1)%Ly))

  LakeNumber = max(min(LakeNumber,1.e+2_ireals),1.e-2_ireals)

  Rossby_rad = sqrt(g*max(row(M+1) - row(1),small_value)/row0*H_mixed_layer)/max(kor,small_value)

  ! Thermocline thickness
  x1 = 0.; x2 = ls%h1
  u1 = 0.; u2 = 0.
  v1 = 0.; v2 = 0.
  Sal1 = 0.; Sal2 = 0.
  do i = 1, M
    if (wst%Tw2(i) >= Ttherm0 .and. wst%Tw2(i+1) < Ttherm0) then
      y = gsp%ddz(i)*ls%h1
      x1 = gsp%z_full(i) + (wst%Tw2(i) - Ttherm0)*y/(wst%Tw2(i) - wst%Tw2(i+1))
      x = ( x1 - gsp%z_full(i) )/y
      u1 = wst%u2(i) + x*( wst%u2(i+1) - wst%u2(i) )
      v1 = wst%v2(i) + x*( wst%v2(i+1) - wst%v2(i) )
      Sal1 = wst%Sal2(i) + x*( wst%Sal2(i+1) - wst%Sal2(i) )
    endif
    if (wst%Tw2(i) >= Ttherm1 .and. wst%Tw2(i+1) < Ttherm1) then
      y = gsp%ddz(i)*ls%h1
      x2 = gsp%z_full(i) + (wst%Tw2(i) - Ttherm1)*y/(wst%Tw2(i) - wst%Tw2(i+1))
      x = ( x2 - gsp%z_full(i) )/y
      u2 = wst%u2(i) + x*( wst%u2(i+1) - wst%u2(i) )
      v2 = wst%v2(i) + x*( wst%v2(i+1) - wst%v2(i) )
      Sal2 = wst%Sal2(i) + x*( wst%Sal2(i+1) - wst%Sal2(i) )
    endif
  enddo
  ThermThick = x2 - x1 

  ReTherm = sqrt((u2 - u1)*(u2 - u1) + (v2 - v1)*(v2 - v1))*ThermThick / niu_wat

  RiTherm = g/row0*(DDENS_DTEMP0*(Ttherm1 - Ttherm0) + DDENS_DSAL0*(Sal2 - Sal1))*ThermThick / &
  & ( (u2 - u1)*(u2 - u1) + (v2 - v1)*(v2 - v1) + small_value)

  ! Rp: ratio of density increments by salinity and temperature,
  ! Rpdens : ratio of density fluxes caused by salinity and temperature fluxes
  do i = 1, M
    x1 = SET_DDENS_DTEMP(eos%par,lindens%par,0.5*(wst%Tw2(i+1) + wst%Tw2(i)), &
    & 0.5*(wst%Sal2(i+1) + wst%Sal2(i)))
    x2 = SET_DDENS_DSAL(eos%par,lindens%par, 0.5*(wst%Tw2(i+1) + wst%Tw2(i)), &
    & 0.5*(wst%Sal2(i+1) + wst%Sal2(i)))
    x = BOUND_FROM_BELOW(wst%Tw2(i+1) - wst%Tw2(i), small_value)
    trb%Rp(i) = - x2/x1*(wst%Sal2(i+1) - wst%Sal2(i))/x
    trb%Rpdens(i) = wst%lamsal(i)/wst%lamw(i)*trb%Rp(i)
  enddo
 
  END SUBROUTINE TURB_SCALES


  SUBROUTINE MIXED_LAYER_CALC(row,ddz,ddz2,dzeta_int,dzeta_05int,h,M, &
  & i_mixed_layer,H_mixed_layer,maxN)

  ! Subroutine calculates the mixed-layer depth

  use PHYS_CONSTANTS, only: &
  & g, &
  & row0

  implicit none

! Input variables
  real(kind=ireals), intent(in) :: row (1:M+1)
  real(kind=ireals), intent(in) :: ddz(1:M)
  real(kind=ireals), intent(in) :: ddz2(1:M-1)
  real(kind=ireals), intent(in) :: dzeta_int(1:M+1)
  real(kind=ireals), intent(in) :: dzeta_05int(1:M)
 
  real(kind=ireals), intent(in) :: h

  integer(kind=iintegers), intent(in) :: M

! Output variables
  integer(kind=iintegers), intent(out) :: i_mixed_layer
  real(kind=ireals), intent(out) :: H_mixed_layer
  real(kind=ireals), intent(out) :: maxN

! Local variables
  integer(kind=iintegers) :: i, ml
  real(kind=ireals) :: a, b, c, x1, x2

  real(kind=ireals), allocatable :: bvf(:)

! Mixed-layer depth - depth of maximal heat flux
!  maxlocat = 1
!  do i = 2, M
!    if (k_turb_T_flux(i) + T_massflux(i) > &
!    & k_turb_T_flux(maxlocat) + T_massflux(maxlocat)) maxlocat = i
!  enddo
!  H_mixed_layer = dzeta_05int(maxlocat)*h1

  allocate (bvf(1:M))      
  do i = 1, M
    bvf(i) = g/row0*(row(i+1) - row(i))/(h*ddz(i))
  enddo

  ! Mixed-layer depth - depth of maximal Bruent-Vaisala frequency
  ml = maxloc(bvf,1) !+ lbound(bvf,1) - 1
  maxN = sqrt(max(bvf(ml),0._ireals))

  if ((row(M) - row(1))/row0 < 1.E-3) ml = M

  !! Mixed-layer depth as a depth of sharp increase of static stability
  !ml = M
  !do i = ml, 1, -1
  !  if (abs(bvf(i)/bvf(ml)) < 7.E-1_ireals) then
  !    ml = i
  !    exit
  !  endif
  !enddo

  ! Maximum of quadratic function
  !if (ml >= 2 .and. ml <= M-1) then
  !  x1 = (bvf(ml+1) - bvf(ml)  )/(h*(dzeta_05int(ml+1) - dzeta_05int(ml  )))
  !  x2 = (bvf(ml)   - bvf(ml-1))/(h*(dzeta_05int(ml  ) - dzeta_05int(ml-1)))
  !  a =   ( x1 - x2 ) / (h*(dzeta_05int(ml+1) - dzeta_05int(ml-1)))
  !  b =   x1 - a*(h*(dzeta_05int(ml+1) + dzeta_05int(ml  )))
  !  H_mixed_layer = - 0.5*b/a
  !else
    H_mixed_layer = dzeta_05int(ml)*h
  !endif
  deallocate (bvf)

  i_mixed_layer = ml

  !if (H_mixed_layer < 0.05) then
  !  H_mixed_layer = dzeta_05int(M)*h
  !  i_mixed_layer = M
  !endif

  ! H_mixed_layer = dzeta_int(ml)*h

  END SUBROUTINE MIXED_LAYER_CALC

  !> Subroutine RICHARDSON_NUMBER returns Richardson number profile in the water column
  SUBROUTINE RICHARDSON_NUMBER(M,u,v,ddz,h,row,Ri)
  use PHYS_CONSTANTS, only : g, row0
  implicit none
  !Input varaibles
  integer(kind=iintegers), intent(in) :: M    !> Number of layers in water
  real(kind=ireals), intent(in ) :: u(1:M+1)  !> x-component of velocity, m/s
  real(kind=ireals), intent(in ) :: v(1:M+1)  !> y-component of velocity, m/s
  real(kind=ireals), intent(in ) :: row(1:M+1)!> water density, kg/m**3
  real(kind=ireals), intent(in ) :: ddz(1:M)  !> non-dim thickness of cells, n/d
  real(kind=ireals), intent(in ) :: h         !> maximal depth of a water body, m
  real(kind=ireals), intent(out) :: Ri(1:M)   !> Richardson number, n/d
  !Local variables
  integer(kind=iintegers) :: i
  real(kind=ireals) :: dudz2dvdz2
  do i = 1, M
    dudz2dvdz2 = max( ((u(i+1)-u(i))**2 + (v(i+1)-v(i))**2)/(ddz(i)*h), small_value)
    Ri(i) = g/row0*(row(i+1)-row(i))/dudz2dvdz2
  enddo
  END SUBROUTINE RICHARDSON_NUMBER

  !> The function W_SEDIM calculates the sedimentation speed
  !! of organic particles, following Song et al., 2008, 
  !! DOI: 10.3882/j. issn. 1674-2370.2008.01.005
  FUNCTION W_SEDIM(d,ind)
     
  use PHYS_CONSTANTS, only : g, niu_wat

  implicit none

  ! Input variables
  !> Particle diameter, m
  real(kind=ireals), intent(in) :: d
  !> Switch between \f$Re\rightarrow 0\f$ and \f$Re \geq 10^5\f$ limits
  integer(kind=iintegers), intent(in) :: ind

  ! Output variables
  !> Sedimentation speed, m/s
  real(kind=ireals) :: W_SEDIM

  ! Local variables
  real(kind=ireals), parameter :: A = 30., B = 1.25 
  real(kind=ireals), parameter :: delta = 0.25 ! Density parameter for organic particles, Avnimelech et al. 2001

  real(kind=ireals), save :: A1, B1
  logical, save :: firstcall = .true.

  if (firstcall) then
    A1 = 4.*delta*g/(3.*A*niu_wat) 
    B1 = 4.*delta*g/(3.*B)
  endif

  if     (ind == 1) then
    W_SEDIM = A1*d*d
  elseif (ind == 2) then
    W_SEDIM = sqrt(B1*d)
  endif

  if (firstcall) firstcall = .false.
  END FUNCTION W_SEDIM


  FUNCTION SHORTRAD(shortrad0,z)

  implicit none

  real(kind=ireals), intent(in) :: shortrad0 ! Radiation at the surface
  real(kind=ireals), intent(in) :: z ! depth, m

! Shortwave radiation in water column from Arctic ocean model by N.Yakovlev
! Penetrating into ocean radiation parameters (Paulson&Simpson, 1977).
  real(kind=ireals), parameter :: Ra = 0.68 ! Fraction of light frequency radiation
  real(kind=ireals), parameter :: dzi1 = 1.2 ! Penetration of high frequency, m
  real(kind=ireals), parameter :: dzi2 = 28. ! Penetration of low frequency, m

  real(kind=ireals) :: SHORTRAD

  SHORTRAD = shortrad0*(Ra*exp(-z/dzi1)+(1.-Ra)*exp(-z/dzi2))

  END FUNCTION SHORTRAD


  FUNCTION WATER_ALBEDO(sinh0)
  implicit none

  real(kind=ireals) :: WATER_ALBEDO

! Input variables
  real(kind=ireals), intent(in) :: sinh0

! Local variables
  real(kind=ireals), parameter :: const1 = 0.05d0
  real(kind=ireals), parameter :: const2 = 0.15d0

  WATER_ALBEDO = const1/(sinh0 + const2)

  END FUNCTION WATER_ALBEDO


  FUNCTION F_AI(T,h)
! Ice albedo, from Arctic ocean model by N.Yakovlev
  implicit none 

! Input variables
  real(kind=ireals), intent(in) :: T, h

  real(kind=ireals) :: F_AI
  real(kind=ireals), parameter :: aow = 0.10, & ! AOMIP
  &                     ai = 0.65 ! Boulder Ice CCSM3 (Ice version 4, 2002) 

  if (T .LT. -1.0)then ! -1C - see BoulderIce 
    if (h .LE. 50.)then ! see BoulderIce - this is a good approx.
      F_ai=aow +(ai-aow)*h/50.
    else
      F_ai= ai
    end if
  else
    if(h.LE.50.)then ! see BoulderIce - this is a good approx.
      F_ai=aow +(ai-0.075*(T+1.)-aow)*h/50. 
    else
      F_ai= ai -0.075*(T+1.)
    end if
  end if

  return
  END FUNCTION F_AI
 
  
  FUNCTION F_AS(T)
! Snow albedo, from Arctic ocean model by N.Yakovlev
  if(T .LT. -1.0)then ! -1C - see BoulderIce

    F_as=0.80 ! Old Aomip, close to CCSM3

  else

    F_as=0.80 -0.1*(T+1.0) ! Pinto, 1999, SHEBA + Boulder Ice, CCSM3

  end if

  return
  END FUNCTION F_AS


  FUNCTION NETLWRAD(T,Ta,epsa,cloud,emis)
! Net longwave radiation at the surface, Rosati&Miyakoda, 1988
  use PHYS_CONSTANTS, only: &
  & sigma

  implicit none

  real(kind=ireals), intent(in) :: T ! Surface temperature, K
  real(kind=ireals), intent(in) :: Ta ! Air temperature, K
  real(kind=ireals), intent(in) :: epsa ! Water vapor pressure, Pa
  real(kind=ireals), intent(in) :: cloud ! Cloudiness, fraction
  real(kind=ireals), intent(in) :: emis ! Emissivity, fraction

  real(kind=ireals) :: NETLWRAD

  real(kind=ireals) :: AA, BB

  AA = emis*sigma*(0.39-0.005*sqrt(epsa))*(1.-0.8*cloud)
  BB = 4.*emis*sigma
  NETLWRAD = - (AA*T + BB*(T-Ta))*T**3 ! TC - corrected

  END FUNCTION NETLWRAD


  FUNCTION EXTINCT_SNOW(snow_density)
  implicit none

  real(kind=ireals) :: EXTINCT_SNOW

! Input variables
  real(kind=ireals), intent(in) :: snow_density ! Snow density, kg/m**3

! Local variables
  real(kind=ireals), parameter :: const1 = 0.13d0
  real(kind=ireals), parameter :: const2 = 3.4d0
  real(kind=ireals), parameter :: extinct_snow_max = 1.d+7

  EXTINCT_SNOW = exp(-(const1*snow_density+const2) )
  
! For Sparkling2002-2005 experiment
! EXTINCT_SNOW = exp(-extinct_snow_max)


  END FUNCTION EXTINCT_SNOW


  real(kind=ireals) FUNCTION UNFRWAT(T,i)
  use DRIVING_PARAMS
  use ARRAYS_SOIL, only : WLM0, WLM7
  
!CALCULATION OF LIQUID WATER CONTENT IN FREEZING SOIL
!T - DEG. C; WLM0,WLM7,WLMAX - KG/KG

  implicit none
  real(kind=ireals) T
  integer(kind=iintegers) i

  unfrwat = (WLM0(i)-WLM7(i))*exp(T/3.) + WLM7(i)
  !unfrwat = 0 
  
  RETURN
  END FUNCTION UNFRWAT


  real(kind=ireals) FUNCTION WL_MAX(por,rosdry,wi,mh)
  use PHYS_CONSTANTS, only: &
  & row0, &
  & roi
  use METH_OXYG_CONSTANTS, only : &
  & densmh
  implicit none
  real(kind=ireals), intent(in):: por,rosdry,wi,mh
  real(kind=ireals) :: work

  if (mh > 0.) then
    work = densmh
  else
    work = roi
  endif

  WL_MAX = por*row0/(rosdry*(1 - por)) - wi*row0/work
  END FUNCTION WL_MAX


  real(kind=ireals) FUNCTION WI_MAX(por,rosdry,mh)
  use PHYS_CONSTANTS, only: &
  & roi
  use METH_OXYG_CONSTANTS, only : &
  & densmh
  implicit none
  real(kind=ireals), intent(in):: por,rosdry,mh
  real(kind=ireals) :: work

  if (mh > 0.) then
    work = densmh
  else
    work = roi
  endif

  WI_MAX = por*work/(rosdry*(1 - por))
  END FUNCTION WI_MAX

  
  FUNCTION SOIL_COND_JOHANSEN(wl,wi,rosdry,por)
  
  use PHYS_CONSTANTS, only: &
  & row0, roi, &
  & row0_d_roi, roi_d_row0, &
  & lamw0, lami
  
  implicit none
  
  real(kind=ireals) :: SOIL_COND_JOHANSEN
  
! Input variables
  real(kind=ireals), intent(in) :: wl, wi ! liquid water and ice content 
                                ! (in respect to dry soil mass), kg/kg
  real(kind=ireals), intent(in) :: rosdry ! dry soil (soil particles) density, kg/m**3
  real(kind=ireals), intent(in) :: por ! soil porosity, m**3/m**3
  
! Local variables
  real(kind=ireals) :: water_vol_ratio, ice_vol_ratio ! water and ice volume ratio 
                                            ! (in respect to bulk soil volume), m**3/m**3
  real(kind=ireals) :: waterice_vol_ratio ! total volume ratio of water and ice, m**3/m**3
  real(kind=ireals) :: Kersten ! Kersten number, n/d
  real(kind=ireals) :: CK_const, CK_const1, CK_const2
  real(kind=ireals) :: lambda_sat, lambda_dry ! heat conduction coefficients for saturated
                                    ! and dry soil, W/(m*K)
                                    
  real(kind=ireals) :: water_sat_ratio, ice_sat_ratio                       
                   
  real(kind=ireals), save :: CK_consts(1:4) ! Cote and Konrad (2005) constants, n/d
  real(kind=ireals), save :: CK_consts1(1:3) ! Cote and Konrad (2005) constants, W/(m*K)
  real(kind=ireals), save :: CK_consts2(1:3) ! Cote and Konrad (2005) constants, n/d
  real(kind=ireals), save :: quartz_ratio ! quartz content of the total solids content
  real(kind=ireals), save :: lambda_quartz ! heat conduction coefficient for quartz, W/(m*K)
  real(kind=ireals), save :: lambda_othmin ! heat conduction coefficient for non-quartz
                                 ! minerals, W/(m*K)
  real(kind=ireals), save :: lambda_solids ! heat conduction coefficient for soild part
                                 ! of the soil, W/(m*K)
  
  logical, save :: firstcall = .true.
  
  if (firstcall) then
    CK_consts(1) = 4.6d0 ! for gravel and coarse sand
    CK_consts(2) = 3.55d0 ! for medium and fine sand
    CK_consts(3) = 1.9d0 ! silty and clay soils
    CK_consts(4) = 0.6d0 ! organic fibrous soils
    
    CK_consts1(1) = 1.7d0 ! for crashed rocks
    CK_consts1(2) = 0.75d0 ! for mineral soils
    CK_consts1(3) = 0.3d0 ! organic fibrous soils
    
    CK_consts2(1) = 1.8d0 ! for crashed rocks
    CK_consts2(2) = 1.2d0 ! for mineral soils
    CK_consts2(3) = 0.87d0 ! organic fibrous soils
    
    quartz_ratio = 0.1d0
    lambda_quartz = 7.7d0
    if (quartz_ratio > 0.2d0) then
      lambda_othmin = 2.d0
    else
      lambda_othmin = 3.d0
    endif
    lambda_solids = lambda_quartz**quartz_ratio * &
    &               lambda_othmin**(1.d0-quartz_ratio)
  endif
  
! Convertation from mass ratios to volume ratios      
  water_vol_ratio = wl / &
  & (por*(wl + row0/roi*wi + row0/rosdry))
  ice_vol_ratio = wi / &
  & (por*(wi + roi/row0*wl + roi/rosdry))
  
  waterice_vol_ratio = water_vol_ratio + ice_vol_ratio
  
  CK_const = CK_consts(3) ! silty and clay soils are assumed
  Kersten = CK_const*waterice_vol_ratio / &
  & (1.d0 + (CK_const - 1.d0)*waterice_vol_ratio)
  
  water_sat_ratio = por*water_vol_ratio/waterice_vol_ratio
  ice_sat_ratio = por*ice_vol_ratio/waterice_vol_ratio
  
! This is the original formula from Johansen (1975) extended for the case
! with the ice content
  lambda_sat = lambda_solids**(1.d0 - por) * &
  & lamw0**(water_sat_ratio)*lami**(ice_sat_ratio)
  
  CK_const1 = CK_consts1(2) ! mineral soils are assumed
  CK_const2 = CK_consts2(2) ! mineral soils are assumed
! Cote and Konrad (2005) formula for heat conduction coefficient of dry soil
  lambda_dry = CK_const1*10.d0**(-CK_const2*por)
  
! The normalized soil conductivity concept by Johansen (1975)      
  SOIL_COND_JOHANSEN = (lambda_sat - lambda_dry)*Kersten + lambda_dry
  
  if (firstcall) firstcall = .false.
  END FUNCTION SOIL_COND_JOHANSEN


  FUNCTION PRESMHYDRDISS(T,S)

  implicit none

  real(kind=ireals), intent(in) :: T ! Temperature, K
  real(kind=ireals), intent(in) :: S ! Salinity, ppt

  real(kind=ireals) :: PRESMHYDRDISS ! Pressure of lower limit of methane hydrate stability, Pa 
                           ! (Tishchenko et al. 2005)

  real(kind=ireals), parameter :: c1 = 1.6444866d+3, c2 = 0.1374178, c3 = 5.4979866d+4
  real(kind=ireals), parameter :: c4 = 2.64118188d+2, c5 = 1.1178266d+4, c6 = 7.67420344
  real(kind=ireals), parameter :: c7 = 4.515213d-3, c8 = 2.04872879d+5, c9 = 2.17246046d+3
  real(kind=ireals), parameter :: c10 = 1.70484431d+2, c11 = 0.118594073, c12 = 7.0581304d-5
  real(kind=ireals), parameter :: c13 = 3.09796169d+3, c14 = 33.2031996 

  real(kind=ireals) :: logpres

  logpres =  - c1 - c2*T + &
  & c3/T + c4*log(T) + S* &
  & (c5 + c6*T - c7*T**2 - c8/T - c9*log(T)) + S**2* &
  & (c10 + c11*T - c12*T**2 - c13/T - c14*log(T))
  PRESMHYDRDISS = exp(logpres)*1.d+6 !Converting to Pa

  END FUNCTION PRESMHYDRDISS



  FUNCTION TEMPMHYDRDISS(p)

! The temperature (K) of methane hydrate dissociation in seawater 
! at 35 ppt salinity according to (Dickens and Quinby-Hunt, 1994, GRL)
  use PHYS_CONSTANTS, only : &
  & Kelvin0

  implicit none

  real(kind=ireals), intent(in) :: p ! Pressure, Pa

  real(kind=ireals) :: TEMPMHYDRDISS

  real(kind=ireals), parameter ::  c1 = 3.79E-3, c2 = 2.83E-4

  TEMPMHYDRDISS = 1./(c1 - c2*log10(p*1.E-6)) - Kelvin0

  END FUNCTION TEMPMHYDRDISS


  FUNCTION MELTINGPOINT(S,p,methhydr)

  use DRIVING_PARAMS, only : nmeltpoint

  implicit none

  real(kind=ireals), intent(in) :: S, p 
  real(kind=ireals), intent(in) :: methhydr
  real(kind=ireals) :: MELTINGPOINT

  if (methhydr > 0.) then
    MELTINGPOINT = TEMPMHYDRDISS(p)
  else
    MELTINGPOINT = MELTPNT(S,p,nmeltpoint%par)
  endif

  END FUNCTION MELTINGPOINT


  !> Function calculates ice salinity at the phase change front
  !! according to V.L.Tsurikov (Nazintsev and Panov, 2000)
  FUNCTION SALICEBOT(dhbdt,salwat,por)
  use PHYS_CONSTANTS, only : row0, roi
  implicit none
  !>Input/output variables
  real(kind=ireals), intent(in) :: dhbdt !> the rate of water freeze at the phase change front, m/s
  real(kind=ireals), intent(in) :: salwat !> water salinity below the phase change front, kg/kg
  real(kind=ireals), intent(in) :: por  !> ice porosity, m**3/m**3
  real(kind=ireals) :: SALICEBOT

  !>Local variables
  real(kind=ireals), parameter :: alpha = 7., beta = 7., gamma_ = 10.3
  real(kind=ireals) :: w
  
  w = max(dhbdt,0._ireals)*3.6E+6 !Converting from m/s to mm/h

  SALICEBOT = (row0*por + (1.-por)*roi) * &
  &           salwat*alpha*sqrt(w)/(beta*sqrt(w) + gamma_) 

  END FUNCTION SALICEBOT


  FUNCTION REACPOT_ARRHEN &
  & (delta_E, temp, temp0, reacpot_arrhen0)

! The FUNCTION REACPOT_ARRHEN calculates the reaction potential
! according to Arrhenius equation

  use PHYS_CONSTANTS, only : &
  & R_univ

  implicit none      
  
! Input variables
  real(kind=ireals), intent(in) :: delta_E ! activation energy, J/mol
  real(kind=ireals), intent(in) :: temp    ! temperature, Kelvin
  real(kind=ireals), intent(in) :: temp0   ! reference temperature, Kelvin
  real(kind=ireals), intent(in) :: reacpot_arrhen0 ! reaction potential at the
                                         ! reference temeperature, mol/(m**3*s)
                                         
! Output variables
  real(kind=ireals) :: REACPOT_ARRHEN
  
! Local variables and constants
  
  REACPOT_ARRHEN = &
  & reacpot_arrhen0*exp(delta_E/R_univ*(1./temp0 - 1./temp))
  
  END FUNCTION REACPOT_ARRHEN
  
  
  FUNCTION HENRY_CONST(henry_const0, temp_dep, temp_ref, temp) !, radius)
  
  ! Function HENRY_CONST calculates the Henry constant of a substance for a given temperature
  
  use PHYS_CONSTANTS, only : &
  & surf_tension_wat, R_univ

  implicit none
  
  real(kind=ireals) :: HENRY_CONST
  
  ! Input variables
  real(kind=ireals), intent(in) :: henry_const0 ! Henry constant at the reference temperature, mol/(m**3*Pa)
  real(kind=ireals), intent(in) :: temp_dep ! Temperature dependence (enthalpy solution devided by universal gas constant), K
  real(kind=ireals), intent(in) :: temp_ref ! Reference temperature, K
  real(kind=ireals), intent(in) :: temp ! Temperature, K
!  real(kind=ireals), intent(in) :: radius ! Curvature radius, m, positive for drops, negative for bubbles and zero for flat surface
  
  HENRY_CONST = henry_const0*exp(temp_dep*(1./temp-1./temp_ref))
! Effect of bubble surface curvatue on saturation pressure
!  if (radius /= 0.d0) HENRY_CONST = HENRY_CONST* &
!  & exp(-2.*surf_tension_wat*mol_vol_water/(radius*R_univ*temp))
  
  END FUNCTION HENRY_CONST


  !> Function HC_CORR_CARBEQUIL returns a correction multiplier for 
  !! CO_2 Henry constant to get solubulity, taking into account
  !! carbonate equilibrium.
  !! \f[
  !! HC\_CORR\_CARBEQUIL = (1+k_1 10^{pH}+k_1*k_2 10^{2pH})
  !! \f]
  !! Concomitantly, it is a ratio of DIC to CO_2 molar concenatrations.
  !! DIC molar concentration is a molar concentration of C atoms in DIC.
  FUNCTION HC_CORR_CARBEQUIL(temp,pH)

  implicit none

  !Input variables
  !>Water temperature, deg. Kelvin
  real(kind=ireals), intent(in) :: temp 
  !>pH
  real(kind=ireals), intent(in) :: pH 

  real(kind=ireals) :: HC_CORR_CARBEQUIL

  !Local variables
  real(kind=ireals), parameter :: k1_0 = 4.3E-7, k2_0 = 4.7E-11
  real(kind=ireals), parameter :: Eact_d_RT_1 = 921.4, Eact_d_RT_2 = 1787.4
  real(kind=ireals), parameter :: temp_ref = 298.

  real(kind=ireals) :: k1,k2

  k1 = k1_0*exp( - Eact_d_RT_1 * (1./temp - 1./temp_ref) )
  k2 = k2_0*exp( - Eact_d_RT_2 * (1./temp - 1./temp_ref) )
  HC_CORR_CARBEQUIL = (1. + k1*10._ireals**(pH) + k1*k2*10._ireals**(2*pH))

  END FUNCTION HC_CORR_CARBEQUIL


  FUNCTION DIFF_WATER_OXYGEN(temp_C)
  
  ! Function DIFF_WATER_OXYGEN calculates the molecular diffusivity 
  ! of oxygen dissolved in liquid water, m**2/s
  ! (Broecker and Peng, 1974)
  
  implicit none
  
  real(kind=ireals) :: DIFF_WATER_OXYGEN
  
  ! Input variables
  real(kind=ireals), intent(in) :: temp_C ! temperature, degrees Celsius

  ! Local variables
  real(kind=ireals), parameter :: const1 = 1.11d-9
  real(kind=ireals), parameter :: const2 = 4.96d-11
  
  DIFF_WATER_OXYGEN = const1 + const2*temp_C
   
  END FUNCTION DIFF_WATER_OXYGEN

  
  FUNCTION DIFF_WATER_METHANE(temp_C)
  
  ! Function DIFF_WATER_METHANE calculates the molecular diffusivity 
  ! of methane dissolved in liquid water, m**2/s
  ! (Broecker and Peng, 1974)
  
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
  
  
  FUNCTION DIFF_AIR_METHANE(temp_C)
  
  ! Function DIFF_WATER_METHANE calculates the molecular diffusivity 
  ! of methane dissolved in liquid water, m**2/s
  ! (Lerman, 1979)
  
  implicit none
  
  real(kind=ireals) :: DIFF_AIR_METHANE
  
  ! Input variables
  real(kind=ireals), intent(in) :: temp_C ! temperature, degrees Celsius

  ! Local variables
  real(kind=ireals), parameter :: const1 = 1.875d-5
  real(kind=ireals), parameter :: const2 = 1.3d-7
  
  DIFF_AIR_METHANE = const1 + const2*temp_C
   
  END FUNCTION DIFF_AIR_METHANE

  
  FUNCTION DIFF_WATER_CARBDI(temp_C)
  
  ! Function DIFF_WATER_CARBDI calculates the molecular diffusivity 
  ! of crabon dioxide dissolved in liquid water, m**2/s
  ! (Broecker and Peng, 1974)
  
  implicit none
  
  real(kind=ireals) :: DIFF_WATER_CARBDI
  
  ! Input variables
  real(kind=ireals), intent(in) :: temp_C ! temperature, degrees Celsius

  ! Local variables
  real(kind=ireals), parameter :: const1 = 9.39d-10
  real(kind=ireals), parameter :: const2 = 2.671d-11
  real(kind=ireals), parameter :: const3 = 4.095d-13
  
  DIFF_WATER_CARBDI = const1 + const2*temp_C + const3*temp_C*temp_C
   
  END FUNCTION DIFF_WATER_CARBDI

  
  FUNCTION DIFF_AIR_CARBDI(temp_C)
  
  ! Function DIFF_WATER_CARBDI calculates the molecular diffusivity 
  ! of carbon dioxide dissolved in liquid water, m**2/s
  ! (Lerman, 1979)
  
  implicit none
  
  real(kind=ireals) :: DIFF_AIR_CARBDI
  
  ! Input variables
  real(kind=ireals), intent(in) :: temp_C ! temperature, degrees Celsius

  ! Local variables
  real(kind=ireals), parameter :: const1 = 1.35d-5
  real(kind=ireals), parameter :: const2 = 0.9d-7
        
  DIFF_AIR_CARBDI = const1 + const2*temp_C
   
  END FUNCTION DIFF_AIR_CARBDI
  
  
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
  
  
  FUNCTION SCHMIDT_NUMBER_OXYGEN(tempC)
  
  implicit none
  
  real(kind=ireals) :: SCHMIDT_NUMBER_OXYGEN
  
  ! Input variables      
  real(kind=ireals), intent(in) :: tempC
  
  ! Local variables
  real(kind=ireals), parameter :: const1 = 1.8006d+3
  real(kind=ireals), parameter :: const2 = -1.201d+2
  real(kind=ireals), parameter :: const3 = 3.7818
  real(kind=ireals), parameter :: const4 = -4.7608d-2
  
  SCHMIDT_NUMBER_OXYGEN = &
  & const1 + const2*tempC + const3*tempC**2 + const4*tempC**3
  
  END FUNCTION SCHMIDT_NUMBER_OXYGEN
  
  
  FUNCTION SCHMIDT_NUMBER_CARBDI(tempC)
  
  implicit none
  
  real(kind=ireals) :: SCHMIDT_NUMBER_CARBDI
  
  ! Input variables      
  real(kind=ireals), intent(in) :: tempC
  
  ! Local variables
  real(kind=ireals), parameter :: const1 = 1.911d+3
  real(kind=ireals), parameter :: const2 = -1.137d+2
  real(kind=ireals), parameter :: const3 = 2.967
  real(kind=ireals), parameter :: const4 = -2.943d-2
  
  SCHMIDT_NUMBER_CARBDI = &
  & const1 + const2*tempC + const3*tempC**2 + const4*tempC**3
  
  END FUNCTION SCHMIDT_NUMBER_CARBDI
  
  
  FUNCTION GAS_WATATM_FLUX &
  & (tempC,wind10,surf_conc,partial_pressure,henry_const,gasindic,eps)

  use PHYS_CONSTANTS, only: &
  & niu_wat
        
  ! Subroutine GAS_WATATM_FLUX calculates the upward gas flux at the water-air interface, mol/(m**2*s)
  ! following formulations by (McGillis et al., 2000; Cole and Caraco, 1998; Riera et al., 1999) 
  ! and others (described in Wania, 2007)
        
  implicit none
  
  real(kind=ireals) :: GAS_WATATM_FLUX
  
  ! Input variables
  real(kind=ireals), intent(in) :: tempC ! water surface temperature, Celsius
  real(kind=ireals), intent(in) :: wind10 ! wind speed at 10 m above the surface, m/s
  real(kind=ireals), intent(in) :: surf_conc ! gas concentration at the water surface, mol/m**3
  real(kind=ireals), intent(in) :: partial_pressure ! partial pressure of a gas in the atmosphere, Pa
  real(kind=ireals), intent(in) :: henry_const ! Henry constant of a gas, mol/(m**3*Pa)
  real(kind=ireals), intent(in) :: eps  ! kinetic energy dissipation
  
  integer(kind=iintegers), intent(in) :: gasindic ! gas indicator: methane - 8, oxygen - 9, carbon dioxide - 10
        
  ! Local constants
  real(kind=ireals), parameter :: constvel1 = 5.75d-6 !(Cole and Coraco et al.,1998) ! 3.78d-8 ! SI units
  real(kind=ireals), parameter :: constvel2 = 5.97d-7 !(Cole and Coraco et al.,1998) ! 1.36d-9 ! SI units
  real(kind=ireals), parameter :: c1 = 0.5 ! (JOUNI J. HEISKANEN et al., 2014)
  ! real(kind=ireals), parameter :: c1 = 1.2 (MacIntyre et al., 2010)
  real(kind=ireals), parameter :: Schmidt_scale = 600.
  real(kind=ireals), parameter :: Schmidt_number_min = 1.
  real(kind=ireals), parameter :: wind10_power = 1.7
  
  integer(kind=iintegers), parameter :: methane_indic = 8
  integer(kind=iintegers), parameter :: oxygen_indic = 9
  integer(kind=iintegers), parameter :: carbdi_indic = 10
  integer(kind=iintegers), parameter :: COLE_CARACO = 1 ! model depending on wind-speed (Cole and Coraco et al.,1998)
  integer(kind=iintegers), parameter :: SURFACE_RENEWAL = 2 ! surface renewal model (Soloviev and Schluessel, 1994;&
  ! & MacIntyre et al.,1995, 2001; Zappa et al., 2007; Turney and Banerjee,2008)
  
  ! Local variables
  real(kind=ireals) :: Schmidt_number
  real(kind=ireals) :: piston_velocity, piston_velocity600
  real(kind=ireals) :: surf_conc_atmequil

  integer(kind=iintegers) :: piston_velocity_model ! choice of model

  piston_velocity_model = SURFACE_RENEWAL

  if (piston_velocity_model == COLE_CARACO) then
    piston_velocity600 = constvel1 + constvel2*wind10**wind10_power
  elseif (piston_velocity_model == SURFACE_RENEWAL) then
    piston_velocity600 = c1*(eps*niu_wat)**0.25/sqrt(Schmidt_scale)
  endif
  
  ! piston_velocity600 = constvel1 + constvel2*wind10**wind10_power      
  if     (gasindic == methane_indic) then
    Schmidt_number = max(SCHMIDT_NUMBER_METHANE(tempC),Schmidt_number_min)
  elseif (gasindic == oxygen_indic) then
    Schmidt_number = max(SCHMIDT_NUMBER_OXYGEN(tempC),Schmidt_number_min)
  elseif (gasindic == carbdi_indic) then
    Schmidt_number = max(SCHMIDT_NUMBER_CARBDI(tempC),Schmidt_number_min)
  endif
  if (Schmidt_number <= 0) then
    print *, 'Schmidt_number=', Schmidt_number, 'ss=',Schmidt_scale, &
&   'tempC=',tempC
  endif
    
  piston_velocity = piston_velocity600*sqrt(Schmidt_scale/Schmidt_number)
  surf_conc_atmequil = partial_pressure*henry_const
  GAS_WATATM_FLUX = piston_velocity*(surf_conc-surf_conc_atmequil)
  
  END FUNCTION GAS_WATATM_FLUX
  
  
  FUNCTION CHARNOCK_Z0(velfrict)

  use PHYS_CONSTANTS, only : &
  & g, niu_atm, const2 => charnock_const
  
  implicit none

  real(kind=ireals) :: CHARNOCK_Z0

  ! Input variables
  real(kind=ireals), intent(in) :: velfrict

  ! Local variables
  real(kind=ireals), parameter :: const1 = 0.111
  ! real(kind=ireals), parameter :: const2 = 0.0144
  ! real(kind=ireals), parameter :: const2 = 0.35
  real(kind=ireals), parameter :: roughness_min = 1.d-5
  real(kind=ireals), parameter :: roughness_max = 1.1d-1

  CHARNOCK_Z0 = &
  & min(max(const1*niu_atm/velfrict + const2*velfrict**2/g, &
  & roughness_min),roughness_max)

  END FUNCTION CHARNOCK_Z0


  FUNCTION DOMWAVESPEED(ufr,fetch)

! Calculating fetch-dependent dominant wave speed,
! following Wuest and Lorke (2003)

  use PHYS_CONSTANTS, only : &
  & g

  implicit none
  real(kind=ireals) :: DOMWAVESPEED

  ! Input variables
  real(kind=ireals), intent(in) :: ufr ! Friction velocity, m/s
  real(kind=ireals), intent(in) :: fetch ! Wind fetch, m

  ! Locals
  real(kind=ireals), parameter :: C1 = 0.051, C2 = 0.96
  real(kind=ireals), save :: C3
  logical, save :: firstcall = .true.

  if (firstcall) then
    C3 = (C1/C2)**(2./3.)
  endif

  DOMWAVESPEED = C3*(ufr*g*fetch)**(1./3.)

  if (firstcall) firstcall = .false.
  END FUNCTION DOMWAVESPEED


  FUNCTION H13(momentum_flux,fetch)

! Significant wave height, Hasselman et al. (1973)

  use PHYS_CONSTANTS, only : &
  & g, roa0

  implicit none
  
  real(kind=ireals) :: H13
 
! Input variables
  real(kind=ireals), intent(in) :: momentum_flux, fetch

! Locals
  real(kind=ireals), parameter :: const = 0.051

  H13 = const*sqrt(momentum_flux*fetch/(g*roa0))

  END FUNCTION H13



  FUNCTION LOGFLUX(vel, ds, z, z0, z0s, mult, ind)

  ! Computes a flux of substance in neutral (logarythmic) surface layer
  ! LOGFLUX is positive in the direction in respect to which the difference
  ! ds is taken

  use PHYS_CONSTANTS, only : &
  & kappa

  implicit none

  ! Input variables
  real(kind=ireals), intent(in) :: vel, ds, z, z0, z0s, mult
  integer(kind=iintegers), intent(in) :: ind
  
  ! Output varaible
  real(kind=ireals) :: LOGFLUX

  ! Local variables
  real(kind=ireals), save :: kappa2
  logical, save :: firstcall = .true.

  if (firstcall) kappa2 = kappa*kappa

  if (ind == 1) then ! return flux
    LOGFLUX = - mult*kappa2*vel*ds/(log(z/z0)*log(z/z0s))
  elseif (ind == 2) then ! return exchange coefficient
    LOGFLUX =   mult*kappa2*vel/(log(z/z0)*log(z/z0s))
  endif

  if (firstcall) firstcall = .false.
  END FUNCTION LOGFLUX


  FUNCTION TEMPWATR(kturb,rad,Tsoil,exchcoef,Tw,Tflux)

  ! Water temperature near lateral bottom from radial temperature
  ! profile scaling

  implicit none

  ! Input variables
  real(kind=ireals), intent(in) :: kturb,rad,Tsoil,exchcoef,Tw,Tflux

  ! Output variable
  real(kind=ireals) :: TEMPWATR

  ! Local variables
  real(kind=ireals) :: C = 1.e+1

  TEMPWATR = Tw + C*rad/kturb*(Tsoil*exchcoef - Tflux)/ & ! 
  &          (1 + C*rad/kturb*       exchcoef)


  END FUNCTION TEMPWATR


  !> Subroutine calculates background eddy diffusivity after (Hondzo and Stefan, 1993)
  SUBROUTINE DIFFMIN_HS(area,wst,gsp,ls,M)
  use PHYS_CONSTANTS, only : row0, g, cw_m_row0
  use ARRAYS_GRID, only : gridspacing_type
  use ARRAYS_WATERSTATE, only : waterstate_type
  use ARRAYS_BATHYM, only : layers_type
  use DRIVING_PARAMS, only : backdiff0
  implicit none
  !Input/output variables
  real(kind=ireals)     , intent(in) :: area
  type(waterstate_type) , intent(in) :: wst
  type(gridspacing_type), intent(in) :: gsp
  type(layers_type)     , intent(in) :: ls
  integer(kind=iintegers), intent(in) :: M
  !Local variables
  integer(kind=iintegers) :: i
  real(kind=ireals) :: x, y

  real(kind=ireals), parameter :: C1 = 8.17E-4, C2 = 0.56, C3 = -0.43, xmin = 7.E-5
  real(kind=ireals), save :: z !A multiplier in expression for background diffusivity
  logical, save :: firstcall = .true.

  if (firstcall) then
    if (backdiff0%par >= 0.) then
      z = backdiff0%par 
    else
      z = C1 !default value
    endif
  endif

  do i = 1, M
    x = g/row0*(wst%row(i+1)-wst%row(i))/(ls%h1*gsp%ddz(i))
    if (x <= 0.) then
      wst%lamw_back(i) = 0.
    else
      x = max(x, xmin)
      y = area*1.E-6 ! converting from m**2 to km**2
      wst%lamw_back(i) = z*(y)**C2 * x**C3
      wst%lamw_back(i) = wst%lamw_back(i) * 1.E-4 !converting from cm**2/s to m**2/s
      wst%lamw_back(i) = wst%lamw_back(i) * cw_m_row0
      wst%lamw_back(i) = wst%lamw_back(i) * 5.E-1 !Calibration multiplyer
    endif
  enddo

  if (firstcall) firstcall = .false.
  END SUBROUTINE DIFFMIN_HS


  !> Subroutine calculates background eddy diffusivity from KPP model 
  !! (Zhang et al., HESS, 2019 and references therein)
  SUBROUTINE DIFFMIN_KPP(wst,gsp,ls,M) 
  use PHYS_CONSTANTS, only : row0, g, cw_m_row0
  use ARRAYS_GRID, only : gridspacing_type
  use ARRAYS_WATERSTATE, only : waterstate_type
  use ARRAYS_BATHYM, only : layers_type
  use DRIVING_PARAMS, only : backdiff0
  implicit none
  !Input/output variables
  type(waterstate_type) , intent(inout) :: wst
  type(gridspacing_type), intent(in) :: gsp
  type(layers_type)     , intent(in) :: ls
  integer(kind=iintegers), intent(in) :: M
  !Local variables
  logical, save :: firstcall = .true.
  real(kind=ireals), parameter :: k0_default = 1.E-5, Ri0 = 7.E-1, p = 3.
  real(kind=ireals), save :: k0
  real(kind=ireals) :: x, y, Ri, dz
  integer(kind=iintegers) :: i !Loop index

  if (firstcall) then
    if (backdiff0%par >= 0.) then
      k0 = backdiff0%par 
    else
      k0 = k0_default !default value
    endif
  endif

  do i = 1, M
    dz = ls%h1*gsp%ddz(i)
    !Brunt-Vaisala frequency squared
    x = g/row0*(wst%row(i+1) - wst%row(i)) / dz
    !Shear frequency squared
    y = ( (wst%u1(i+1) - wst%u1(i))**2 + (wst%v1(i+1) - wst%v1(i))**2 ) / dz**2
    !Gradient Richardson number
    Ri = x / (y + small_value)
    if     (Ri <= 0.) then
      wst%lamw_back(i) = k0
    elseif (Ri < Ri0) then
      wst%lamw_back(i) = k0*(1. - (Ri/Ri0)**2)**p
    else
      wst%lamw_back(i) = 0.
    endif
  enddo
  wst%lamw_back(:) = wst%lamw_back(:) * cw_m_row0

  if (firstcall) firstcall = .false.
  END SUBROUTINE DIFFMIN_KPP

  !> Function HORIZCONC returns a value of dissolved concentration in the mixed layer
  !! at distance r from center of circular lake; according to 
  !! (DelSontro et al., Ecosystems, 2018) extended by inclusion of linear sink term
  FUNCTION HORIZCONC(Caver,r,rL,kdiff,kexch,ksink,hML) result(Cr)
  use NUMERICS, only : BESSI !Bessel function
  implicit none
  !Input/output variables
  real(kind=ireals), intent(in) :: Caver !> horizontally averaged concentration in the ML
  real(kind=ireals), intent(in) :: r     !> distance form lake center, m
  real(kind=ireals), intent(in) :: rL    !> lake radius, m
  real(kind=ireals), intent(in) :: kdiff !> effective diffusion coefficient in horizontal, m**2/s
  real(kind=ireals), intent(in) :: kexch !> gas exchange koefficient with the atmosphere (piston velocity), m/s
  real(kind=ireals), intent(in) :: ksink !> the rate of concentration sink in the ML, s**(-1)
  real(kind=ireals), intent(in) :: hML   !> mixed-layer (ML) depth
  real(kind=ireals)             :: Cr    !> concentration at the distance r from lake center
  !Local variables
  real(kind=ireals) :: x
  x = sqrt((kexch/hML+ksink)/kdiff)
  Cr = 0.5*Caver*rL*BESSI(0_4,x*r)/BESSI(1_4,x*rL)
  END FUNCTION HORIZCONC

  END MODULE PHYS_FUNC
