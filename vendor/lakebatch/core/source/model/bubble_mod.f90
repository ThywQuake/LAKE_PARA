MODULE BUBBLE_LAKE_MOD

use LAKE_DATATYPES, only : ireals, iintegers

contains
SUBROUTINE BUBBLE(M,Mbot,ddz,ddz05,dzeta_int,qwater,Tw,Sal,h1,atmpres, &
& DIC,oxyg,ngasb,febul0,dt, &
& frac0,fbbleflx_ch4,fbbleflx_co2,fbbleflx_o2)

! Subroutine BUBBLE calculates evolution of gases in a single bubble
! according to (McGinnis et al., 2006) and normalized bubble fluxes of gases
! (Stepanenko et al., 2016, submitted to GMD)

use PHYS_CONSTANTS, only : &
& g, row0, Kelvin0, R_univ !,DEFINE_PHYS_CONSTANTS

use METH_OXYG_CONSTANTS, only : &
& Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, &
& Henry_const0_co2, Henry_temp_dep_co2, &
& Henry_const0_o2,  Henry_temp_dep_o2, &
& Henry_const0_n2,  Henry_temp_dep_n2, &
& Henry_const0_ar,  Henry_temp_dep_ar, &
& molmass_ch4, molmass_co2, molmass_o2, &
& molmass_n2, molmass_ar, &
& n2_ocean_ref, ar_ocean_ref, &
& pH

use PHYS_FUNC, only : &
& HENRY_CONST, &
& PRESMHYDRDISS, HC_CORR_CARBEQUIL

implicit none

! Input variables
integer(kind=iintegers), intent(in) :: M ! Number of layers in water
integer(kind=iintegers), intent(in) :: Mbot !The number of water layer at the bottom 

real(kind=ireals), intent(in) :: ddz(1:M) ! Dzeta-grid spacing, n/d
real(kind=ireals), intent(in) :: ddz05(0:M) ! Dzeta-grid spacing, n/d
real(kind=ireals), intent(in) :: dzeta_int(1:M+1) ! 
real(kind=ireals), intent(in) :: qwater(1:M+1) ! Methane concentration in water, mol/m**3
real(kind=ireals), intent(in) :: DIC(1:M+1) ! Carbon dioxide concentration in water, mol/m**3
real(kind=ireals), intent(in) :: oxyg(1:M+1)   ! Oxygen concentration in water, mol/m**3
real(kind=ireals), intent(in) :: Tw(1:M+1) ! Water temperature, Celsius
real(kind=ireals), intent(in) :: Sal(1:M+1) ! Water salinity, kg/kg
real(kind=ireals), intent(in) :: h1 ! Lake/reservoir depth, m
real(kind=ireals), intent(in) :: atmpres ! Atmospheric pressure, Pa

integer(kind=iintegers), intent(in) :: ngasb ! The number of gases considered in a bubble

real(kind=ireals), intent(in) :: febul0 ! Bottom ebullition flux of methane, mol/(m**2*s)
real(kind=ireals), intent(in) :: dt ! Timestep, s

real(kind=ireals), intent(inout) :: frac0(1:ngasb) ! Molar fractions of gases in a bubble

real(kind=ireals), intent(out) :: fbbleflx_ch4(0:M+1) ! Methane upward bubble flux, scaled by bottom flux, n/d
real(kind=ireals), intent(out) :: fbbleflx_co2(0:M+1) ! Carbon upward dioxide bubble flux, scaled by bottom flux, n/d
real(kind=ireals), intent(out) :: fbbleflx_o2 (0:M+1) ! Oxygen upward bubble flux, scaled by bottom flux, n/d

! Local variables
real(kind=ireals), save :: pi,pi43r 
real(kind=ireals), allocatable :: mols(:,:), q(:,:), ndiv(:), TwK(:)
real(kind=ireals), parameter :: radius0 = 2.d-3 ! Initial bubble radius, m
real(kind=ireals), parameter :: small_number = 1.d-20
real(kind=ireals), parameter :: radius_max = 5.d-3 ! Maximal bubble radius, m
real(kind=ireals) :: qmid, rhog, ppres, vb, radius, molconc, mols0, xx, dddz
real(kind=ireals) :: henry(1:ngasb), diff(1:ngasb), parpres(1:ngasb)
real(kind=ireals) :: molmass(1:ngasb)

real(kind=ireals), external :: DZETA

integer(kind=iintegers) :: i, k


logical, save :: firstcall = .true.
logical :: test, HSZ

!CALL DEFINE_PHYS_CONSTANTS
if (firstcall) then
  pi = 4.*datan(1.d0)
  pi43r = 4./3.*pi/R_univ
endif

! 1 - methane
! 2 - carbon dioxide
! 3 - oxygen
! 4 - nitrogen
! 5 - argon
molmass(1) = molmass_ch4
molmass(2) = molmass_co2
molmass(3) = molmass_o2
molmass(4) = molmass_n2
molmass(5) = molmass_ar

frac0(1) = 1.- (real(ngasb) - 1.)*small_number ! The molar fraction of methane in bubble at the bottom
frac0(2:ngasb) = 0. + small_number

allocate(mols(1:Mbot,1:ngasb),q(1:Mbot,1:ngasb),ndiv(1:Mbot),TwK(1:Mbot))

TwK(1:Mbot) = Tw(1:Mbot) + Kelvin0

q(1:Mbot,1) = qwater(1:Mbot)
!Converting from DIC to CO_2(aq)
do i = 1, Mbot
  q(i,2) = DIC(i)/HC_CORR_CARBEQUIL(TwK(i),pH) 
enddo
q(1:Mbot,3) = oxyg  (1:Mbot)
q(1:Mbot,4) = n2_ocean_ref
q(1:Mbot,5) = ar_ocean_ref

mols0 = pi43r*(h1*g*row0 + atmpres)*radius0**3/TwK(Mbot)
! The amount of gases in the bottom bubble, mols
mols(Mbot,1:ngasb) = mols0*frac0(1:ngasb)

ndiv(Mbot) = 1.
vb = 2.d-1 ! Estimate for velocity for drag coefficient calculation in BUBBLEVEL at i = Mbot-1
! Euler method for ODE solution
eulcyc : do i = Mbot-1, 1, -1
! qmid = 0.5*(qwater(i) + qwater(i+1))
  ppres = dzeta_int(i+1)*h1*g*row0 + atmpres ! total pressure in a bubble
  radius = (sum(mols(i+1,1:ngasb))*TwK(i+1)/(ppres*pi43r))**(1./3.)
  parpres(1:ngasb) = mols(i+1,1:ngasb)*TwK(i+1)/(pi43r*radius**3)
  rhog = sum(parpres(1:ngasb)*molmass(1:ngasb))*1.d-3/(R_univ*TwK(i+1))

  vb = BUBBLEVEL(radius,rhog,vb)

  henry(1) = HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, TwK(i+1))
  henry(2) = HENRY_CONST(Henry_const0_co2, Henry_temp_dep_co2, Henry_temp_ref, TwK(i+1))
  henry(3) = HENRY_CONST(Henry_const0_o2,  Henry_temp_dep_o2,  Henry_temp_ref, TwK(i+1))
  henry(4) = HENRY_CONST(Henry_const0_n2,  Henry_temp_dep_n2,  Henry_temp_ref, TwK(i+1))
  henry(5) = HENRY_CONST(Henry_const0_ar,  Henry_temp_dep_ar,  Henry_temp_ref, TwK(i+1))
  ! Checking for Hydrate Stability Zone, salinity converted to ppt's
  HSZ = (ppres >  PRESMHYDRDISS(TwK(i+1),Sal(i+1)*1.e+3_ireals)) 
!  if (HSZ) write(*,*) 'Hydrate Stability Zone ', dzeta_int(i+1)*h1
!  if (i==M) write(*,*) PRESMHYDRDISS(Tw(i+1)+Kelvin0,Sal(i+1)*1.d+3), ppres

  do k = 1, ngasb
    diff(k) = Dg(k,ngasb)
    if (mols(i+1,k) == 0.) then ! not a necessary requirement
      mols(i,k) = 0.
    else
      mols(i,k) = mols(i+1,k) - KLi(radius,vb,diff(k),HSZ)* &
      & (henry(k)*parpres(k) - q(i+1,k))*4.*pi*radius**2/vb*ddz(i)*h1
      if (febul0 > 0.) then
        if (i == Mbot-1) then
          dddz = 0.5*ddz(Mbot-1)
        else
          dddz = ddz05(i)
        endif
        ! Ensuring positiveness of q (gas concentration)
        xx = mols(i+1,k) + (q(i+1,k) - small_number)*dddz*h1*mols(Mbot,k)/ &
        & (dt*febul0*ndiv(i+1))
        mols(i,k) = min(mols(i,k),xx)
      endif
!      if (k == 1 .and. mols(i,k) > 2.*mols(i+1,k)) then
!        write(*,*) 'Bubble is growing!', KLi(radius,vb,diff(k),HSZ),henry(k)*parpres(k), q(i+1,k), vb,radius
!      endif
!      xx = KLi(radius,vb,diff(k),HSZ)
      mols(i,k) = max(mols(i,k),0._ireals)
!      if (i == M) write(*,*) henry(1)*parpres(1),q(M+1,1),vb,KLi(radius,vb,diff(1),HSZ)
!      if (i == M) then
!        xx = mols(i+1,k) + mols(M+1,k)*q(M+1,k)*h1*ddz(M)/(dt*febul)
!        mols(i,k) = min(mols(i,k),)
!      elseif (i == 1) then
!      else
!      endif
    endif
  enddo

! Splitting of large bubbles
  ppres = dzeta_int(i)*h1*g*row0 + atmpres ! total pressure in a bubble
  radius = (sum(mols(i,1:ngasb))*TwK(i)/(ppres*pi43r))**(1./3.)
  if (radius > radius_max) then
    mols(i,1:ngasb) = 0.5*mols(i,1:ngasb)
    ndiv(i) = 2.*ndiv(i+1)
  else
    ndiv(i) = ndiv(i+1)
  endif

! Checking if all gas amounts are zero
  test = (mols(i,1) == 0.)
  if (ngasb > 1 .and. test) then
    do k = 2, ngasb
      test = (test .and. mols(i,k) == 0.)
      if (.not. test) exit
    enddo
  endif

  if (test) then
    mols(1:i-1,1:ngasb) = 0.
    exit eulcyc
  endif

enddo eulcyc

!write(*,*) 'CO2 ', mols(:,2), 'N2 ', mols(:,4), 'CH4 ', mols(:,1), 'O2 ', mols(:,3)

fbbleflx_ch4(0) = mols(1,1)*ndiv(1)/mols(Mbot,1)
fbbleflx_co2(0) = mols(1,2)*ndiv(1)/mols(Mbot,2)
fbbleflx_o2 (0) = mols(1,3)*ndiv(1)/mols(Mbot,3)
do i = 1, Mbot-1 ! Velocity should be taken into account as well
  fbbleflx_ch4(i) = 0.5*(mols(i,1)*ndiv(i) + mols(i+1,1)*ndiv(i+1))/mols(Mbot,1)
  fbbleflx_co2(i) = 0.5*(mols(i,2)*ndiv(i) + mols(i+1,2)*ndiv(i+1))/mols(Mbot,2)
  fbbleflx_o2 (i) = 0.5*(mols(i,3)*ndiv(i) + mols(i+1,3)*ndiv(i+1))/mols(Mbot,3)
enddo
fbbleflx_ch4(Mbot) = 1.
fbbleflx_co2(Mbot) = 1.
fbbleflx_o2 (Mbot) = 1.

if (Mbot < M+1) then
  fbbleflx_ch4(Mbot+1:M+1) = 0.
  fbbleflx_co2(Mbot+1:M+1) = 0.
  fbbleflx_o2 (Mbot+1:M+1) = 0.
endif

!do i = 1, M
!  if (fbbleflx_ch4(i) > 10.) then
!    write(*,*) 'Bubble!',mols(1,1),ndiv(1),mols(M+1,1)
!    write(*,*) 'fbbleflx_ch4', fbbleflx_ch4(:)
!    write(*,*) 'mols', mols(:,1)
!    write(*,*) 'ndiv', ndiv(:)
!    write(*,*) 'qwater', qwater(:)
!    write(*,*) 'oxygen', oxyg(:)
!    write(*,*) 'DIC', DIC(:)
!    write(*,*) 'Tw', Tw(:)
!    write(*,*) 'Sal', Sal(:)
!    write(*,*) 'h1', h1
!    STOP
!    exit
!  endif
!enddo

!write(*,*) 'Exceeded! ', fbbleflx_ch4(:)

! Debug
!fbbleflx_ch4(:) = 0.
!fbbleflx_co2(:) = 0.
!fbbleflx_o2(:)  = 0.

deallocate(mols,q,ndiv,TwK)
if (firstcall) firstcall = .false.
END SUBROUTINE BUBBLE


FUNCTION KLi(radius,vb,diff,HSZ)


implicit none

! Liquid-side mass transfer coefficient, m/s

real(kind=ireals), intent(in) :: radius, vb, diff
logical, intent(in) :: HSZ

real(kind=ireals) :: KLi
real(kind=ireals) :: d ! Diamater, cm
real(kind=ireals) :: n ! for dirty bubbles 2./3.
real(kind=ireals), parameter :: m_to_cm = 1.d+2
real(kind=ireals), parameter :: coef1 = 1.13d-2, coef2 = 6.5d-2, coef3 = 6.94d-2

! HSZ - hydrate stability zone
if (HSZ) then
  n = 2./3.
else
  n = 0.5
endif

d = 2.*radius*m_to_cm

if (d < 0.5d0) then
  KLi = coef1*sqrt(vb*m_to_cm/(0.45+0.2*d))*diff**n
elseif (d >= 0.5d0 .and. d < 1.3d0) then
  KLi = coef2*diff**n
elseif (d >= 1.3d0) then
  KLi = coef3*diff**n*d**(-0.25d0) 
endif

END FUNCTION KLi


FUNCTION Dg(ng,ngasb)


use PHYS_CONSTANTS, only : &
& row0, niu_wat

use METH_OXYG_CONSTANTS, only : &
& molvol_ch4, &
& molvol_co2, &
& molvol_o2, &
& molvol_n2, &
& molvol_ar

implicit none

! Molecular diffusion coefficient in water, cm**2/s

integer(kind=iintegers), intent(in) :: ng ! Gas identifier
integer(kind=iintegers), intent(in) :: ngasb ! Total number of gases

real(kind=ireals) :: Dg

real(kind=ireals), parameter :: m3_to_cm3 = 1.d+6
real(kind=ireals), parameter :: kg_d_ms_to_centipoises = 1.d+3
real(kind=ireals), parameter :: small_value = 1.d-20

real(kind=ireals) :: miu
real(kind=ireals), allocatable, save :: molvols(:)

logical, save :: firstcall = .true.

if (firstcall) then
  allocate (molvols(1:ngasb))
  molvols(1) = molvol_ch4
  molvols(2) = molvol_co2
  molvols(3) = molvol_o2
  molvols(4) = molvol_n2
  molvols(5) = molvol_ar
endif

miu = row0*niu_wat*kg_d_ms_to_centipoises
Dg = 13.26d-5/(miu**1.14*(molvols(ng)*m3_to_cm3)**0.589) !+ small_value
! small_value is added to avoid zero diffusivity when there is zero concentration

if (firstcall) firstcall = .false.
END FUNCTION Dg


FUNCTION BUBBLEVEL(radius,rhog,vb_)

! Bubble velocity

use PHYS_CONSTANTS, only : &
& g, row0, niu_wat, surf_tension_wat

implicit none

! Input variables
real(kind=ireals), intent(in) :: radius, rhog
real(kind=ireals), intent(in) :: vb_

real(kind=ireals) :: BUBBLEVEL

! Local variables
real(kind=ireals), save :: Re, Cd, vb0, radius0
real(kind=ireals) :: d
logical, save :: firstcall = .true.

vb0 = max(1.d-2,vb_)
radius0 = max(1.d-3,radius)

Re = vb0*2.*radius0/niu_wat
Cd = 24./Re + 3./sqrt(Re) + 0.34

d = 2.*radius
if (d < 2.6d-3) then ! These equations should be combined!
  BUBBLEVEL = sqrt(4.*d*g*(1.-rhog/row0)/(3.*Cd))
else
  BUBBLEVEL = sqrt(2.*surf_tension_wat/(d*(row0 + rhog)) + 0.5*g*d)
endif

if (firstcall) firstcall = .false.
END FUNCTION BUBBLEVEL


SUBROUTINE BUBBLEFLUXAVER(ix,iy,is)

! Subroutine adds bubble flux from particular soil column to
! the horizontally averaged bubble flux

use ARRAYS_GRID, only : gs

use ARRAYS_BATHYM, only : bathymwater, bathymsoil
use ARRAYS_METHANE, only : fbbleflx_ch4, fbbleflx_ch4_sum, &
& fracb0, febul0, febul_lake_bot
use ARRAYS_OXYGEN, only : fbbleflx_co2, fbbleflx_co2_sum, &
& fbbleflx_o2 , fbbleflx_o2_sum, fco2ebul_bot

implicit none

integer(kind=iintegers), intent(in) :: ix, iy, is

integer(kind=iintegers) :: i
real(kind=ireals) :: x, y

do i = 0, gs%M+1
  if     (i == 0  ) then
    y = bathymwater(1)%area_int
  elseif (i == gs%M+1) then
    y = bathymwater(gs%M+1)%area_int
  else
    y = bathymwater(i)%area_half
  endif
  x = max(min(bathymsoil(is,ix,iy)%area_int - bathymsoil(is+1,ix,iy)%area_int, &
  &           y                             - bathymsoil(is+1,ix,iy)%area_int),0._ireals)
  !Adding area of the deepest column
  if (is == gs%nsoilcols) x = x + bathymsoil(is+1,ix,iy)%area_int
  fbbleflx_ch4_sum(i) = fbbleflx_ch4_sum(i) + &
  & febul0(is) *                       fbbleflx_ch4(i,is) * x
  fbbleflx_co2_sum(i) = fbbleflx_co2_sum(i) + &
  & febul0(is) * fracb0(2)/fracb0(1) * fbbleflx_co2(i,is) * x
  fbbleflx_o2_sum (i) = fbbleflx_o2_sum (i) + &
  & febul0(is) * fracb0(3)/fracb0(1) * fbbleflx_o2 (i,is) * x
enddo

!Total bottom ebullition fluxes
if (is /= gs%nsoilcols) then
  x = febul0(is)*(bathymsoil(is,ix,iy)%area_int - bathymsoil(is+1,ix,iy)%area_int)
else
  x = febul0(is)* bathymsoil(is,ix,iy)%area_int
endif
febul_lake_bot = febul_lake_bot + x 
fco2ebul_bot   = fco2ebul_bot   + x*fracb0(2)/fracb0(1) 


END SUBROUTINE BUBBLEFLUXAVER


END MODULE BUBBLE_LAKE_MOD
