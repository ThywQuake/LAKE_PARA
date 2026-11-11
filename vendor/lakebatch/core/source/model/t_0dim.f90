MODULE ZERODIM_MODEL_LAKE_MOD

contains
SUBROUTINE ZERODIM_MODEL &
& (h, dt, &
& shortwave, longwave, tempair, humair, pressure, &
& uwind, vwind, zref, hw_input, xlew_input, cdmw_input, surfrad_input, &
& cloud, botflux, Temp0)

! Subroutine T_0DIM calculates the lake temperature
! on the new timestep according to zero-dimensional model
! (well mixed lake)
use LAKE_DATATYPES, only : ireals, iintegers

use PHYS_CONSTANTS, only : &
& cw_m_row0, &
& albedoofwater, &
& albedoofwater_lw, &
& emissivityofwater, &
& sigma, &
& roughness0, &
& Kelvin0, &
& pref, &
& Rd, Rd_d_cp, Rd_d_Rwv, &
& esatsurf0, &
& aMagw, bMagw, &
& Lwv, cp

use DRIVING_PARAMS, only : PBLpar, missing_value

use PHYS_FUNC, only : &
& NETLWRAD

use SURF_SCHEME1_MOD, only : DRAGVL

implicit none

! Input/output variables

real(kind=ireals), intent(in) :: h ! Lake depth, m
real(kind=ireals), intent(in) :: dt ! Timestep, s
real(kind=ireals), intent(in) :: shortwave ! Shortwave radiation flux, W/m**2
real(kind=ireals), intent(in) :: longwave ! Longwave radiation flux, W/m**2
real(kind=ireals), intent(in) :: tempair ! Air temperature, Celsius
real(kind=ireals), intent(in) :: humair ! Specific humidity, kg/kg
real(kind=ireals), intent(in) :: pressure ! Air pressure, Pa
real(kind=ireals), intent(in) :: uwind ! X-wind component, m/s
real(kind=ireals), intent(in) :: vwind ! Y-wind component, m/s
real(kind=ireals), intent(in) :: zref ! The height of measurements 
                            ! (or of the lowermost atmospheric model level), m
real(kind=ireals), intent(in) :: hw_input ! Sensible heat flux, W/m**2, computed outside
real(kind=ireals), intent(in) :: xlew_input ! Latent heat flux, W/m**2, computed outside
real(kind=ireals), intent(in) :: cdmw_input ! Momentum exchange coefficient, m/s, computed outside
real(kind=ireals), intent(in) :: surfrad_input ! Surface radiation, W/m**2, computed outside
real(kind=ireals), intent(in) :: cloud ! Cloudiness, fraction
real(kind=ireals), intent(in) :: botflux ! Heat flux at the bottom, W/m**2, computed outside
real(kind=ireals), intent(inout) :: Temp0 ! Lake temperature, Celsius

! Local variables and parameters
real(kind=ireals), parameter :: minwind = 1.d-2
real(kind=ireals), parameter :: relhums = 1.
integer(kind=iintegers), parameter :: itdrag = 10

real(kind=ireals) :: eflux, hflux, Elatent, surfrad
real(kind=ireals) :: wr, TET2, TET1, ro, esatsurf, humsurf
real(kind=ireals) :: c_u, c_t
real(kind=ireals) :: bx(7), bix(11)
real(kind=ireals) :: longwave1, epsa

wr       = max(sqrt(uwind**2 + vwind**2), minwind)
TET2     = (tempair + Kelvin0)*(pref/pressure)**Rd_d_cp
TET1     = (Temp0 + Kelvin0)*(pref/pressure)**Rd_d_cp  
ro       = pressure/(Rd*(tempair + Kelvin0))  
esatsurf = esatsurf0*10.**(aMagw*Temp0/(bMagw + Temp0))
humsurf  = Rd_d_Rwv/pressure*esatsurf*relhums

! Businger-Dayer, Beljaars parameterization for exchange coefficients
bx(1) = wr
bx(2) = TET2
bx(3) = TET1
bx(4) = humair 
bx(5) = humsurf
bx(6) = zref
bx(7) = roughness0
call DRAGVL (bx, bix, itdrag)
c_u = bix(8)
c_t = bix(9)
hflux   = - cp*ro*c_u*c_t*wr*(TET2 - TET1)
Elatent = - Lwv*ro*c_u*c_t*wr*(humair - humsurf) 
surfrad = sigma*emissivityofwater*(Temp0 + Kelvin0)**4

!if (PBLpar%par == -1) then
! Sensible, latent heat and momentum fluxes are prescribed outside from lake model
!  hflux   = hw_input
!  Elatent = xlew_input
!  tau = cdmw_input*ro*wr
!endif

if (longwave == missing_value) then
! Longwave radiation is missing in atmospheric forcing
  epsa = humair*pressure/Rd_d_Rwv
  longwave1 = NETLWRAD(Temp0+Kelvin0,tempair+Kelvin0,epsa,cloud,emissivityofwater) + surfrad
else
  longwave1 = longwave
endif

eflux = shortwave*(1. - albedoofwater) + longwave1*(1. - albedoofwater_lw) - &
& surfrad - hflux - Elatent
!eflux = shortwave*(1. - albedoofwater) + longwave - surfrad_input - hw_input - xlew_input
Temp0 = Temp0 + dt*(eflux - botflux)/(cw_m_row0*h)
!Temp0 = Temp0 + dt*eflux/(cw_m_row0*h)

END SUBROUTINE ZERODIM_MODEL

END MODULE ZERODIM_MODEL_LAKE_MOD
