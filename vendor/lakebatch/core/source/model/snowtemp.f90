MODULE SNOWTEMP_LAKE

use SNOW_LAKE, only : SNOW_CALC, ADDPRECIPTOSNOW

contains
SUBROUTINE SNOWTEMP(ix,iy,nx,ny,year,month,day,hour,snowmass, &
& snowmass_init,a,b,c,d,Temp,phi,fetch,dt)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

! SNOWTEMP calculates temperature profile in the snow cover

use LAKE_DATATYPES, only : ireals, iintegers
use NUMERIC_PARAMS    
use ATMOS
use PHYS_CONSTANTS
use DRIVING_PARAMS
use ARRAYS
use ARRAYS_BATHYM, only : hs1
use BL_MOD_LAKE
use SNOWSOIL_MOD

implicit none


!-----------------------------MAIN VARIABLES-------------------------------
!   arrays: Tsn - snow temperature, C
!   lams - thermal conductivity of snow
!    
!   rofresh - density of fresh snow, kg/m**3     
!   snowmass - snow mass, kg/m**2

!   Boundary conditions:
!   t2 - air temperature (2 m), C  
 
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: nx, ny
integer(kind=iintegers), intent(in) :: year, month, day

real(kind=ireals)   , intent(in) :: hour
real(kind=ireals)   , intent(in) :: phi
!real(kind=ireals)   , intent(in) :: extice
real(kind=ireals)   , intent(in) :: fetch
real(kind=ireals), intent(inout) :: a(1:nveclen), b(1:nveclen), &
&                                   c(1:nveclen), d(1:nveclen), Temp(1:nveclen)


real(kind=ireals) :: xx
real(kind=ireals) :: dzsn(ms)
real(kind=ireals) :: rofresh
real(kind=ireals) :: t2
real(kind=ireals) :: snowmass
real(kind=ireals) :: pheat
real(kind=ireals) :: pmelt
real(kind=ireals) :: snowmass_init
real(kind=ireals) :: CCT(ML)
real(kind=ireals) :: dt


integer(kind=iintegers) :: i
integer(kind=iintegers) :: iyear
integer(kind=iintegers) :: imonth
integer(kind=iintegers) :: iday  

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_SNOWTEMP)
include 'lb_ctx_snowtemp.inc'

do i = itop,ms
  T(i) = Tsn(i)    
enddo

t2 = tempair
Erad = shortwave*(1-albedoofsnow)
Elatent = xlew

call addPrecipToSnow(t2, T(itop), snowmass, iyear, imonth, iday, &
& CCT, pmelt, pheat,dt)

call snow_calc(ix,iy,t2,Erad,T(itop),snowmass,iyear, &
& imonth, iday, CCt, pmelt, pheat,dt)
hs1 = hsnow

if (flag_snow_init == 1) then
! snowmass_init = snowmass  - (precip*dt*row-snmelt*dt*row-
! & Elatent/Liv*dt)
  flag_snow_init = 0
endif 

totalevaps = totalevaps + Elatent/(row0*Lwv)*dt
totalprecips = totalprecips + precip*dt
totalmelts = totalmelts + snmelt*dt

!if (hs1 == 0) hs1=0.00001

END SUBROUTINE SNOWTEMP


SUBROUTINE SNOW_COND_HEAT_COEF

use LAKE_DATATYPES, only : ireals, iintegers

use ARRAYS, only : &
& flag_snow_init, &
& totalmelts, &
& totalprecips, &
& totalevaps

use ARRAYS_BATHYM, only : hs1
use ARRAYS_WATERSTATE, only: Ti1

use ATMOS, only : &
& tempair

use PHYS_CONSTANTS, only : &
& row0, &
& ci, &
& cw, &
& lamair

use NUMERIC_PARAMS, only : &
& ms, &
& ML

use BL_MOD_LAKE
use SNOWSOIL_MOD

implicit none

real(kind=ireals) :: xx
real(kind=ireals) :: dzsn(ms)
real(kind=ireals) :: rofresh
real(kind=ireals) :: CCT(ML)

integer(kind=iintegers) :: i


if (flag_snow_init == 1) then
  itop = ms-2 
  hs1 = 0.02
  rofresh = 67.9 + 51.3*exp(tempair/2.6)
  do i = itop, ms-1
    WL(i) = 0.0
    dens(i) = rofresh 
    dz(i) = 0.01
    T(i) = tempair
  end do
  wl(ms) = 0.
  dens(ms) = rofresh
  T(ms) = Ti1(1)
  dz(ms) = 0.5d0*dz(ms-1)
  do i=1,itop-1
    T(i)=0
    wl(i)=0
    dens(i)=0
    dz(i)=0
  enddo
  totalmelts=0.
  totalprecips=0.
  totalevaps=0.
end if


do i = itop, ms-2
  dzsn(i) = (dz(i)+dz(i+1))*0.5d0
enddo
do i = itop, ms-1
  !densav = (dens(i) + dens(i+1))/2
  !lams(i) =  0.419*(6.*(densav/row)**4+1.9*(densav/row)+0.05)
  lams(i) = 0.419*(6.*(dens(i)/row0)**4+1.9*(dens(i)/row0)+0.05)
  !lams(i) = 3.E-6*dens(i)**2 - 1.06E-5*dens(i) + lamair !(Riche and Schneebeli, 2013, The Cryosphere)
  !lams(i) = 2.5E-6*dens(i)**2 - 1.23E-4*dens(i) + lamair !(Calonne et al., 2011, GRL)
  xx = wl(i)*row0/(dens(i)*dz(i)) 
  cs(i) = ci*(1-xx)+cw*xx
end do


END SUBROUTINE SNOW_COND_HEAT_COEF

END MODULE SNOWTEMP_LAKE
