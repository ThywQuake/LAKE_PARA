MODULE SOIL_LAKE_MOD

use LAKE_DATATYPES, only : ireals, iintegers
use NUMERICS, only : PROGONKA, STEP
use NUMERIC_PARAMS, only : nveclen

contains
SUBROUTINE COMSOILFORLAKE
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

!COMSOILFORLAKE specifies parameters of soil according to soil type

use NUMERIC_PARAMS
use DRIVING_PARAMS
use ARRAYS
use ARRAYS_SOIL
implicit none

real(kind=ireals) :: b(1:Num_Soil)
real(kind=ireals) :: psi_max(1:Num_Soil)
real(kind=ireals) :: Porosity(1:Num_Soil)
real(kind=ireals) :: gamma_max(1:Num_Soil)
real(kind=ireals) :: lambda_max(1:Num_Soil)
real(kind=ireals) :: W_0(1:Num_Soil)
real(kind=ireals) :: W_m(1:Num_Soil)
real(kind=ireals) :: pow
  
integer(kind=iintegers) :: i, j
logical :: firstcall

data firstcall /.true./

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_COMSOILFORLAKE)
include 'lb_ctx_comsoilforlake.inc'
if (firstcall) then
! allocate (AL(1:ns+2),DLT(1:ns+2),DVT(1:ns+2),DL(1:ns+2),
!& ALV(1:ns+2),DV(1:ns+2),Z(1:ns+2),T(1:ns+2),WL(1:ns+2),
!& WV(1:ns+2),WI(1:ns+2),dens(1:ns+2))
endif

!I=1 ! SAND
!I=2 ! LOAMY SAND
!I=3 ! SANDY LOAM
!I=4 ! LOAM
!I=5 ! SILT LOAM
!I=6 ! SANDY CLAY LOAM
!I=7 ! CLAY LOAM
!I=8 ! SILTY CLAY LOAM
!I=9 ! SANDY CLAY
!I=10! SILTY CLAY
!I=11! CLAY
!----------------------------------------------------------------
!  NN    b      psi_max  Por   gamma_max lambda_max W_0     W_m
!----------------------------------------------------------------
!   1   4.05    3.50    0.395   0.01760   0.29800   0.02    0.01
!   2   4.38    1.78    0.410   0.01560   0.13900   0.05    0.02
!   3   4.90    7.18    0.435   0.00340   0.13700   0.08    0.03
!   4   5.39    14.6    0.451   0.00069   0.06190   0.20    0.08
!   5   5.30    56.6    0.485   0.00072   0.20400   0.18    0.07
!   6   7.12    8.63    0.420   0.00063   0.03070   0.13    0.06
!   7   8.52    36.1    0.476   0.00024   0.03720   0.27    0.12
!   8   7.75    14.6    0.477   0.00017   0.01240   0.24    0.11
!   9   10.4    6.16    0.426   0.00021   0.00641   0.23    0.10
!  10   10.4    17.4    0.492   0.00010   0.00755   0.30    0.15
!  11   11.4    18.6    0.482   0.00013   0.00926   0.40    0.20
!-----------------------------------------------------------------

zsoil(1) = 0.

if (UpperLayer > 0.) then
  pow = log(UpperLayer/depth%par)/log(1.d0/float(ns-1))
  do i = 2, ns
    zsoil(i) = (1.*(i-1)/(ns-1))**pow*depth%par
  end do
else
  do i = 2, ns
    zsoil(i) = (1.*(i-1)/(ns-1))*depth%par
  end do
end if

do i = 1, ns-1
  dzs(i) = zsoil(i+1) - zsoil(i)
end do

do i = 1, ns
  if (i == 1) then
    dzss(i) = 0.5d0*dzs(i)
  elseif (i == ns) then
    dzss(i) = 0.5d0*dzs(i-1)
  else
    dzss(i) = 0.5d0*(dzs(i-1) + dzs(i))
  endif 
end do 

!SoilType = 7

b(1) = 4.05
b(2) = 4.38
b(3) = 4.90
b(4) = 5.39
b(5) = 5.30
b(6) = 7.12
b(7) = 8.52
b(8) = 7.75
b(9) = 10.4
b(10)= 10.4
b(11)= 11.4

psi_max( 1) = 3.50/100. 
psi_max( 2) = 1.78/100. 
psi_max( 3) = 7.18/100. 
psi_max( 4) = 14.6/100. 
psi_max( 5) = 56.6/100. 
psi_max( 6) = 8.63/100. 
psi_max( 7) = 36.1/100. 
psi_max( 8) = 14.6/100. 
psi_max( 9) = 6.16/100. 
psi_max(10) = 17.4/100. 
psi_max(11) = 18.6/100. 

Porosity( 1) = 0.395
Porosity( 2) = 0.410
Porosity( 3) = 0.435
Porosity( 4) = 0.451
Porosity( 5) = 0.485
Porosity( 6) = 0.420
Porosity( 7) = 0.476
Porosity( 8) = 0.477
Porosity( 9) = 0.426
Porosity(10) = 0.492
Porosity(11) = 0.482

gamma_max( 1) = 0.01760/100. 
gamma_max( 2) = 0.01560/100. 
gamma_max( 3) = 0.00340/100. 
gamma_max( 4) = 0.00069/100. 
gamma_max( 5) = 0.00072/100. 
gamma_max( 6) = 0.00063/100. 
gamma_max( 7) = 0.00024/100. 
gamma_max( 8) = 0.00017/100. 
gamma_max( 9) = 0.00021/100. 
gamma_max(10) = 0.00010/100. 
gamma_max(11) = 0.00013/100. 

lambda_max( 1) = 0.29800/10000. 
lambda_max( 2) = 0.13900/10000.
lambda_max( 3) = 0.13700/10000.
lambda_max( 4) = 0.06190/10000.
lambda_max( 5) = 0.20400/10000.
lambda_max( 6) = 0.03070/10000.
lambda_max( 7) = 0.03720/10000.
lambda_max( 8) = 0.01240/10000.
lambda_max( 9) = 0.00641/10000.
lambda_max(10) = 0.00755/10000.
lambda_max(11) = 0.00926/10000.

W_0( 1) = 0.02
W_0( 2) = 0.05
W_0( 3) = 0.08
W_0( 4) = 0.20
W_0( 5) = 0.18
W_0( 6) = 0.13
W_0( 7) = 0.27
W_0( 8) = 0.24
W_0( 9) = 0.23
W_0(10) = 0.30
W_0(11) = 0.40

W_m( 1) = 0.01
W_m( 2) = 0.02
W_m( 3) = 0.03
W_m( 4) = 0.08
W_m( 5) = 0.07
W_m( 6) = 0.06
W_m( 7) = 0.12
W_m( 8) = 0.11
W_m( 9) = 0.10
W_m(10) = 0.15
W_m(11) = 0.20

if (SoilType%par > 0) then
  do j = 1, ns
    BH(j)= b(SoilType%par)   ! PARAMETER B, DIMENSOINLESS
    PSIMAX(j) = - psi_max(SoilType%par) ! SAT. WATER POTENTIAL, M.    
    POR(j)    = Porosity(SoilType%par)  ! POROSITY, DIMENSIONLESS
    FLWMAX(j) = gamma_max(SoilType%par) ! SAT. HYDR. CONDUCTIVITY, M/S
    DLMAX(j)  = lambda_max(SoilType%par)! SAT. WATER DIFFUSIVITY, ?KG/(M*SEC)
    WLM0(j)   = W_0(SoilType%par) ! MAXIMAL UNFREEZING WATER AT 0C
    WLM7(j)   = W_m(SoilType%par) ! MAXIMAL UNFREEZING WATER AT T<<0C
  end do
else
  write(*,*) 'LAKE: Soil type is not given: STOP'
  STOP
!do j = 1, ns
! BH(j)= BH_soil! PARAMETER B, DIMENSOINLESS
! PSIMAX(j) = PSIMAX_soil ! SAT. WATER POTENTIAL, M.    
! POR(j)    = POR_soil    ! POROSITY, DIMENSIONLESS
! FLWMAX(j) = FLWMAX_soil ! SAT. HYDR. CONDUCTIVITY, M/S
! DLMAX(j)  = DLMAX_soil  ! SAT. WATER DIFFUSIVITY, ?KG/(M*SEC)
! WLM0(j)   = WLM0_soil   ! MAXIMAL UNFREEZING WATER AT 0C
! WLM7(j)   = WLM7_soil   ! MAXIMAL UNFREEZING WATER AT T<<0C
!end do
end if

rosdry(:) = 1.2E+3

if (firstcall) firstcall=.false.   
RETURN
END SUBROUTINE COMSOILFORLAKE


SUBROUTINE SOILFORLAKE(nn,dt,a,b,c,d,is)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

!SOILFORLAKE calculates profiles of temperature, liquid and solid water 
!content in the soil under a lake

use PHYS_CONSTANTS 
use METH_OXYG_CONSTANTS!, only : &
!& methhydrdiss, &
!& alphamh, &
!& nch4_d_nh2o, &
!& molmass_h2o
use DRIVING_PARAMS
use ARRAYS
use ARRAYS_SOIL
use ARRAYS_METHANE
use ARRAYS_BATHYM
use PHYS_FUNC!, only : &
!& MELTPNT, &
!& TEMPMHYDRDISS, &
!& UNFRWAT, &
!& WL_MAX, &
!& MELTINGPOINT
use ATMOS!, only : &
!& pressure

implicit none

integer(kind=iintegers), intent(in) :: nn 

real(kind=ireals), intent(in) :: dt

!integer(kind=iintegers), intent(in) :: ix, iy
!integer(kind=iintegers), intent(in) :: nx, ny
integer(kind=iintegers), intent(in) :: is ! Number of soil column 

!integer(kind=iintegers), intent(in) :: year, month, day

!real(kind=ireals)   , intent(in) :: hour
!real(kind=ireals)   , intent(in) :: phi
!real(kind=ireals)   , intent(in) :: extwat, extice
!real(kind=ireals)   , intent(in) :: fetch
    
real(kind=ireals), intent(inout) :: a(1:nn)
real(kind=ireals), intent(inout) :: b(1:nn)
real(kind=ireals), intent(inout) :: c(1:nn)
real(kind=ireals), intent(inout) :: d(1:nn)
!real(kind=ireals) :: Temp(1:nveclen)

real(kind=ireals) :: dzmean
real(kind=ireals) :: ma, mi
real(kind=ireals) :: wflow, wfhigh
real(kind=ireals) :: lammoist1, lammoist2
real(kind=ireals) :: surplus
real(kind=ireals) :: Potphenergy
real(kind=ireals) :: watavailtofreeze
real(kind=ireals) :: filtr_low
real(kind=ireals) :: dhwfsoil1, dhwfsoil2, dhwfsoil3, dhwfsoil4
real(kind=ireals) :: max1
real(kind=ireals) :: work, mhsep
real(kind=ireals), pointer :: hicemelt

real(kind=ireals) :: alp(10)
real(kind=ireals) :: psiit(10)

real(kind=ireals), allocatable :: pressoil(:)

integer(kind=iintegers) :: i, j, k
integer(kind=iintegers) :: iter, iter3

logical :: firstcall

data psiit/6,3,7,2,5,4,8,1,0,0/ !/3,2,4,1,0,0,0,0,0,0/  
data firstcall /.true./

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_SOILFORLAKE)
include 'lb_ctx_soilforlake.inc'

!open (133, file=dir_out//'\dhwfsoil.dat', status = 'unknown')   

!PARAMETERS OF ITERATIONAL PROCESS
ma = 10.5 !5.5
mi = 4.5
do i = 1, 8 
  alp(i) = 2*(mi+ma-(ma-mi)*cos((2*psiit(i)-1)/16*pi))**(-1)
enddo

allocate (pressoil(1:ns))
do i = 1, ns
  pressoil(i) = pressure + row0*g*(h1 + zsoil(i))
enddo

if (tricemethhydr%par > 0.) then
  hicemelt => methhydrdiss
else
  hicemelt => Lwi
endif
   
!do i=1, ns
! wlmax_i = POR(i)*ROW0/(rosdry(i)*(1-por(i))) 
! BB1 = BH(i) + 2.
! csoil(i) = cr+WL1(i)*CW+WI1(i)*CI
! rosoil(i) = rosdry(i)*(1-por(i))*(1+wi1(i)+wl1(i)) 
! ARG = (WL1(i)+WI1(i))/wlmax_i
! ARG = max(ARG,1.d-2)
! PSI = PSIMAX(i)*(ARG)**(-BH(i))
! PF = log(-PSI*100.)/log(10.d0)
! IF(PF>5.1) THEN
!  lamsoil(i) = 4.1E-4*418.
! ELSE
!  lamsoil(i) = exp(-PF-2.7)*418.
! END IF
! wlmax_i = POR(i)*ROW0/(rosdry(i)*(1-por(i))) - wi1(i)*row0_d_roi
! ARG = (WL1(i))/wlmax_i
! ARG = max(ARG,1.d-2)
! lammoist(i) = dlmax(i)*ARG**BB1
!end do


!do i=1, ns-1
! wlmax_i=WL_MAX(por(i+1),rosdry(i+1),wi1(i+1))
! if (wl1(i+1)/wlmax_i>0.98.or.wlmax_i<0.01) then
!  filtr(i) = 0
! else
!  wlmax_i = (POR(i)+POR(i+1))*ROW0/((rosdry(i)+rosdry(i+1))*&
!&  (1-(por(i)+por(i+1))/2)) - (wi1(i)+wi1(i+1))/2*row0_d_roi
!  filtr(i) = (flwmax(i+1)+flwmax(i))/2*((wl1(i+1)+&
!&  wl1(i))/2/wlmax_i)**(bh(i+1)+bh(i)+3)
! endif
!enddo

!SPLITTING-UP METHOD FOR TEMPERATURE EQUATION:
!STEP 1: HEAT DIFFUSION

!do i = 1, ns-1
! lammoist1 = 0.5*(lammoist(i)+lammoist(i+1))
! wsoil(i) = ( - lammoist1*(wl1(i+1)-wl1(i))/dzs(i) + &
! & filtr(i) ) *0.5*(rosdry(i)+rosdry(i+1))* &
! & (1-0.5*(por(i)+por(i+1)) )/row0
!enddo


!SPLITTING-UP METHOD FOR MOISTURE EQUATION:
!STEP 1: MOISTURE DIFFUSION

Wflow = 0.
dhwfsoil=0.
dhwfsoil1=0.
dhwfsoil2=0.
dhwfsoil3=0.
dhwfsoil4=0.

if ((l1 /= 0.and. h1 == 0.).or.ls1/=0.or. &
  & (hs1/=0.and.l1==0.and.h1==0)) then
  Wfhigh = 0
!c(1)=1
!b(1)=1
!d(1)=0 
  c(1) = -1-dt*(lammoist(1)+lammoist(2))/(dzs(1)**2)
  b(1) =   -dt*(lammoist(1)+lammoist(2))/(dzs(1)**2)
  d(1) =   -wl1(1,is)
endif
   
if (h1 /= 0.and.ls1 == 0) then
  c(1)=1
  b(1)=0
  d(1)=WL_MAX(por(1),rosdry(1),wi1(1,is),tricemethhydr%par)
!dhwfsoil1 = dt*(wl1(2)-wl1(1))*rosdry(1)*(1-por(1))/row0
!& /dzs(1)*(lammoist(1)+lammoist(2))/2 !-
!&  (WL_MAX(por(1),rosdry(1),wi1(1))-wl1(1)) VS,23.09.07
!& *rosdry(1)*(1-por(1))*dzss(1)/row0  VS,23.09.07
endif    
  
lammoist1 = (lammoist(ns-1)+lammoist(ns))/2
!c(ns)=-lammoist1/dzs(ns-1)
!a(ns)=-lammoist1/dzs(ns-1)
!d(ns)=Wflow
c(ns) = -1-2*dt*lammoist1/dzs(ns-1)**2 
a(ns) = -2*dt*lammoist1/dzs(ns-1)**2 
d(ns) = -wl1(ns,is)

do i=2,ns-1
  lammoist2 = (lammoist(i)+lammoist(i+1))/2
  lammoist1 = (lammoist(i-1)+lammoist(i))/2
  dzmean = (dzs(i-1)+dzs(i))/2
  a(i)=-dt/dzmean*lammoist1/dzs(i-1)
  b(i)=-dt/dzmean*lammoist2/dzs(i)
  c(i)=-dt/dzmean*(lammoist1/dzs(i-1) + &
  & lammoist2/dzs(i))-1
  d(i)=-wl1(i,is)
 !wl2(i) = wl1(i)+dt*(lammoist2/dzs(i)*(wl1(i+1)-wl1(i))-
!& lammoist1/dzs(i-1)*(wl1(i)-wl1(i-1)))/dzmean
enddo   
    
call PROGONKA(nveclen,a,b,c,d,wl2,1,ns)  
    
if (h1 /= 0.and.ls1 == 0) then 
  lammoist1 = (lammoist(1)+lammoist(2))/2.
  dhwfsoil1 = -(wl2(1)*(dzs(1)/(2*dt)+lammoist1/dzs(1)) - &
  & wl2(2)*lammoist1/dzs(1)-wl1(1,is)*dzs(1)/(2*dt)) * &
  & rosdry(1)*(1-por(1))/row0*dt
endif 
!STEP 2 OF SPLITTING-UP METHOD FOR MOISTURE EQUATION: 
!EVOLUTION OF MOISTURE DUE TO GRAVITATIONAL INFILTRATION  
 
do i=2,ns-1
  wl3(i) = wl2(i) + dt*(filtr(i-1)-filtr(i))/dzss(i)
enddo
 
if ((h1==0.and.l1/=0).or.ls1 /= 0) then
  wl3(1) = wl2(1) + dt*(-filtr(1)+0)/dzss(1)
endif

if (h1/=0.and.ls1 == 0) then
  wl3(1) = WL_MAX(por(1),rosdry(1),wi1(1,is),tricemethhydr%par)
  dhwfsoil2 = - filtr(1)*rosdry(1)*(1-por(1))/row0*dt 
endif

filtr_low = 0
wl3(ns) = wl2(ns) + dt*(-filtr_low+filtr(ns-1))/dzss(ns)

do i=1,ns
  if (wl3(i)>WL_MAX(por(i),rosdry(i),wi1(i,is),tricemethhydr%par)) then
    surplus=wl3(i)-WL_MAX(por(i),rosdry(i),wi1(i,is),tricemethhydr%par)
    cy1:do j=1,ns
      if (WL_MAX(por(j),rosdry(j),wi1(j,is),tricemethhydr%par)-wl3(j)>0) then
        if (WL_MAX(por(j),rosdry(j),wi1(j,is),tricemethhydr%par) - wl3(j) > &
        & surplus*rosdry(i)*(1-por(i))*dzss(i) / &
        & (rosdry(j)*(1-por(j))*dzss(j))) then
          wl3(j)=wl3(j)+surplus*rosdry(i)*(1-por(i))*dzss(i) / &
          & (rosdry(j)*(1-por(j))*dzss(j))
          surplus=0
          exit cy1
        else
          surplus = surplus - &
          & (WL_MAX(por(j),rosdry(j),wi1(j,is),tricemethhydr%par)-wl3(j)) * &
          & (rosdry(j)*(1-por(j))*dzss(j)) / &
          & (rosdry(i)*(1-por(i))*dzss(i))
          wl3(j)=WL_MAX(por(j),rosdry(j),wi1(j,is),tricemethhydr%par) 
        endif
      endif
    enddo cy1
    dhwfsoil3 = dhwfsoil3 + surplus*rosdry(i)*(1-por(i))*dzss(i)/row0
    wl3(i) = WL_MAX(por(i),rosdry(i),wi1(i,is),tricemethhydr%par)
  endif
  if(wl3(i)<0) then
    if (i/=1) then 
      do j=i-1,1,-1
        if (wl3(j)>-wl3(i)*rosdry(i)*(1-por(i))*dzss(i) / &
        & (rosdry(j)*(1-por(j))*dzss(j))) then
          wl3(j)=wl3(j)+wl3(i)*rosdry(i)*(1-por(i))*dzss(i) / &
          & (rosdry(j)*(1-por(j))*dzss(j))
          goto 1
        endif
      enddo
    endif
    if (i/=ns) then 
      do j=i+1,ns
        if (wl3(j)>-wl3(i)*rosdry(i)*(1-por(i))*dzss(i) / &
        & (rosdry(j)*(1-por(j))*dzss(j))) then
          wl3(j)=wl3(j)+wl3(i)*rosdry(i)*(1-por(i))*dzss(i) / &
          & (rosdry(j)*(1-por(j))*dzss(j))
          goto 1
        endif
      enddo 
    endif
 !dhwfsoil = dhwfsoil - abs(wl4(i))*rosdry(i)*(1-por(i))*dzss(i)/row
1   wl3(i)=0.
  endif
enddo


! STEP 3 OF SPLITTING-UP METHOD : PHASE PROCESSES
mhsep = (1. + tricemethhydr%par*alphamh)
20 c2: do i = 1, ns
  work = MELTINGPOINT(Sals1(i,is)/wl3(i),pressoil(i),tricemethhydr%par)
  if (wi1(i,is) > 0 .and. Tsoil2(i) > work + 0.01) then
    Potphenergy = (Tsoil2(i) - (work + 0.01)) * &
    & csoil(i)*rosoil(i)*dzss(i)
    if (Potphenergy >= wi1(i,is)*rosdry(i)*dzss(i)*(1-por(i))*hicemelt) then
      wl4(i,is) = wl3(i) + wi1(i,is)/mhsep
      wi2(i,is) = 0.d0
      Tsoil3(i,is) = work + 0.01 + &
      & (Potphenergy-(wi1(i,is)*rosdry(i)*dzss(i)* &
      & (1 - por(i))*hicemelt))/(csoil(i)*rosoil(i)*dzss(i))
    else
      wl4(i,is) = wl3(i) + Potphenergy / &
      & (rosdry(i)*dzss(i)*(1 - por(i))*hicemelt)/mhsep
      wi2(i,is) = wi1(i,is) - Potphenergy / &
      & (rosdry(i)*dzss(i)*(1 - por(i))*hicemelt)
      Tsoil3(i,is) = work + 0.01
    endif
  else
    if (wl3(i) >= 0 .and. Tsoil2(i) < work - 0.01) then
      Potphenergy = - (Tsoil2(i) - work + 0.01) * &
      & csoil(i)*rosoil(i)*dzss(i)
      watavailtofreeze = (wl3(i) - UNFRWAT(Tsoil2(i),i)) * &
      & rosdry(i)*dzss(i)*(1 - por(i))*mhsep*hicemelt
      if (watavailtofreeze < 0) then 
        !cc = csoil(i)*rosoil(i)*dzss(i)
        !cc1 = rosdry(i)*dzss(i)*(1-por(i))*Lwi
        !bb = (-Potphenergy-0.1*cc-wl3(i)*cc1)/cc
        !aa = cc1/cc
        !call phase_iter(aa,bb,i,Tsoil3(i))
        Tsoil3(i,is) = Tsoil2(i) + watavailtofreeze/(csoil(i)*rosoil(i)*dzss(i))
        wi2(i,is) = wi1(i,is) + (wl3(i) - UNFRWAT(Tsoil2(i),i))*mhsep
        wl4(i,is) = UNFRWAT(Tsoil2(i),i)
        cycle c2  
      endif
      if (Potphenergy >= watavailtofreeze) then
        Tsoil3(i,is) = work - 0.01 - &
        & (Potphenergy - watavailtofreeze) / &
        & (csoil(i)*rosoil(i)*dzss(i))
        wi2(i,is) = wi1(i,is) + (wl3(i) - UNFRWAT(Tsoil2(i),i))*mhsep
        wl4(i,is) = UNFRWAT(Tsoil2(i),i)
      else
        wl4(i,is) = wl3(i) - Potphenergy/(rosdry(i)*dzss(i)*(1 - por(i))*hicemelt)/mhsep
        wi2(i,is) = wi1(i,is) + Potphenergy/(rosdry(i)*dzss(i)*(1 - por(i))*hicemelt)
        Tsoil3(i,is) = work - 0.01
      endif
    else
      wl4(i,is) = wl3(i)
      wi2(i,is) = wi1(i,is)
      Tsoil3(i,is) = Tsoil2(i)
    endif
  endif
enddo c2

max1 = 0.
do i = 1, ns
  if (Tsoil3(i,is) < &
    & MELTINGPOINT(Sals1(i,is)/wl4(i,is),pressoil(i),tricemethhydr%par) - 0.01 &
    &.and. abs(wl4(i,is) - UNFRWAT(Tsoil3(i,is),i)) > max1) then
    max1 = abs(wl4(i,is) - UNFRWAT(Tsoil3(i,is),i))
  endif
enddo    
  
if (max1 > 0.01 .and. iter3 < 20) then
  !wl3=(wl4+wl3)/2.
  !wi1=(wi2+wi1)/2.
  !Tsoil2=(Tsoil3+Tsoil2)/2.
  wl3 = wl3 + alp(iter+1)*(wl4(:,is)-wl3)
  wi1(:,is) = wi1(:,is) + &
  & alp(iter+1)*(wi2(:,is) - wi1(:,is))
  Tsoil2 = Tsoil2 + alp(iter+1)*(Tsoil3(:,is)-Tsoil2)
  iter = iter + 1
  iter3 = iter3 + 1
  if(iter == 8) iter = 0
  goto 20
else
  iter = 0
  iter3 = 0 
endif   

!evol=0

do i = 1, ns
  if (wi2(i,is) < 0) then
    wl4(i,is) = wl4(i,is) + wi2(i,is)/mhsep
    wi2(i,is) = 0
  endif
enddo

! Methane generation due to methane hydrate dissociation
do i = 1, ns
  methgenmh(i) = - tricemethhydr%par*(wi2(i,is) - wi1(i,is))*1.d+3 / &
  & (dt*mhsep*molmass_h2o)*nch4_d_nh2o*rosdry(i)*(1. - por(i))
enddo

do i=1,ns
  if (wl4(i,is) > &
  & POR(i)*ROW0/(rosdry(i)*(1-por(i))) - wi2(i,is)*row0_d_roi) then
    surplus = wl4(i,is) - &
    & (POR(i)*ROW0/(rosdry(i)*(1-por(i))) - wi2(i,is)*row0_d_roi)
    c3:do j=1,ns
      if (POR(j)*ROW0/(rosdry(j)*(1-por(j))) - wi2(j,is)*row0_d_roi - &
      & wl4(j,is) > 0) then
        if (POR(j)*ROW0/(rosdry(j)*(1-por(j))) - wi2(j,is)*row0_d_roi - &
        & wl4(j,is) > surplus*rosdry(i)*(1 - por(i))*dzss(i) / &
        & (rosdry(j)*(1-por(j))*dzss(j))) then
          wl4(j,is)=wl4(j,is)+surplus*rosdry(i)*(1-por(i))*dzss(i) / &
          & (rosdry(j)*(1-por(j))*dzss(j))
          surplus=0
          exit c3
        else
          surplus=surplus-(POR(j)*ROW0/(rosdry(j)*(1-por(j))) - &
          & wi2(j,is)*row0_d_roi-wl4(j,is)) * &
          & (rosdry(j)*(1-por(j))*dzss(j)) / &
          & (rosdry(i)*(1-por(i))*dzss(i))
          wl4(j,is) = &
          & POR(j)*ROW0/(rosdry(j)*(1-por(j))) - wi2(j,is)*row0_d_roi 
        endif
      endif
    enddo c3
    dhwfsoil4 = dhwfsoil4 + surplus*rosdry(i)*(1-por(i))*dzss(i)/row0
    wl4(i,is) = &
    & POR(i)*ROW0/(rosdry(i)*(1-por(i))) - wi2(i,is)*row0_d_roi 
  endif
  if(wl4(i,is) < 0) then
    if (i/=1) then 
      do j=i-1,1,-1
        if (wl4(j,is) > - wl4(i,is)*rosdry(i)*(1-por(i))*dzss(i) / &
        & (rosdry(j)*(1-por(j))*dzss(j))) then
          wl4(j,is) = wl4(j,is) + &
          & wl4(i,is)*rosdry(i)*(1-por(i))*dzss(i) / &
          & (rosdry(j)*(1-por(j))*dzss(j))
          goto 2
        endif
      enddo
    endif
    if (i/=ns) then 
      do j=i+1,ns
        if (wl4(j,is) > -wl4(i,is)*rosdry(i)*(1-por(i))*dzss(i) / &
        & (rosdry(j)*(1-por(j))*dzss(j))) then
          wl4(j,is) = wl4(j,is) + &
          & wl4(i,is)*rosdry(i)*(1-por(i))*dzss(i) / &
          & (rosdry(j)*(1-por(j))*dzss(j))
          goto 2
        endif
      enddo
    endif
    !dhwfsoil = dhwfsoil - abs(wl4(i))*rosdry(i)*(1-por(i))*dzss(i)/row
2   wl4(i,is) = 0
  endif
enddo

  !!!!! II. FREEZING AND MELTING IN CASE OF TSOIL<MELTING POINT (O C)!!!!!

    !do i=1, ns
    ! if (Tsoil3(i)<-1.) then
    !   wi3(i) = wi2(i) + (wl4(i)-unfrwat(Tsoil3(i),i))
  !   wl5(i) = unfrwat(Tsoil3(i),i)
    !   Tsoil4(i) = Tsoil3(i) + & 
    !   (wl4(i)-unfrwat(Tsoil3(i),i))*rosdry(i)*dzss(i)*(1-por(i))*Lwi/&
    !   (csoil(i)*rosoil(i)*dzss(i))
    ! else
    !   wi3(i) = wi2(i)
    !   wl5(i) = wl4(i)
    !   Tsoil4(i) = Tsoil3(i)
  ! end if
    ! if (wl5(i)>POR(i)*ROW/(rosdry(i)*(1-por(i))) - wi3(i)*row/roi) then
    !  surplus = wl5(i)-(POR(i)*ROW/(rosdry(i)*(1-por(i))) - wi3(i)*row/roi)
    !  dhwfsoil = dhwfsoil + surplus*rosdry(i)*(1-por(i))*dzss(i)/row
    !  wl5(i) = POR(i)*ROW/(rosdry(i)*(1-por(i))) - wi3(i)*row/roi 
  ! endif
    ! if(wl5(i)<0) then 
    !  dhwfsoil = dhwfsoil - abs(wl5(i))*rosdry(i)*(1-por(i))*dzss(i)/row
    !  wl5(i)=0
    ! endif
  !enddo
   
   ! evol=0
!   evol1=0

!   do i=1,ns
!    evol=evol+(Tsoil3(i)-Tsoil1(i))*rosoil(i)*csoil(i)*dzss(i)
!    evol1=evol1+(wi2(i)-wi1(i))*rosdry(i)*(1-por(i))*dzss(i)*Lwi
!   enddo

dhwfsoil = dhwfsoil1 + dhwfsoil2 + dhwfsoil3 + dhwfsoil4

 if (any(Tsoil3(:,is) < -10.)) then
  print*, 'SOILFORLAKE: Tsoil11 = ', Tsoil1
  print*, 'SOILFORLAKE: Tsoil12 = ', Tsoil2
  print*, 'SOILFORLAKE: Tsoil13 = ', Tsoil3
  print*, 'SOILFORLAKE: wl1 = ', wl1
  print*, 'SOILFORLAKE: wl2 = ', wl2
  print*, 'SOILFORLAKE: wl3 = ', wl3
  print*, 'SOILFORLAKE: wl4 = ', wl4
  print*, 'SOILFORLAKE: wi1 = ', wi1
  print*, 'SOILFORLAKE: wi2 = ', wi2
endif
 
Tsoil1(:,is) = Tsoil3(:,is)
wl1   (:,is) = wl4   (:,is)
wi1   (:,is) = wi2   (:,is)

! Diagnostic calculation of air content, kg/kg 
! (air mass per unit mass of the dry soil)
! assuming that the air occupies all the pore space 
! free from liquid water and ice
do i = 1, ns
  wa(i) = por(i)*roa0/((1.-por(i))*rosdry(i)) - &
  & roa0_d_row0*wl1(i,is) - roa0_d_roi*wi1(i,is)
  wa(i) = max(wa(i), 0.d0)
enddo

deallocate(pressoil)

if (firstcall) firstcall=.false.
RETURN
END SUBROUTINE SOILFORLAKE


SUBROUTINE SOIL_COND_HEAT_COEF(is)

use ARRAYS_SOIL, only : rosdry,csoil,rosoil,lamsoil, &
lammoist,filtr,wsoil,por,bh,wl1,wi1,psimax,flwmax,dlmax,dzs
use ARRAYS_GRID, only : nsoilcols
use DRIVING_PARAMS, only : ns, tricemethhydr
use PHYS_CONSTANTS, only : &
& row0, &
& cw, &
& ci, &
& roi, &
& row0_d_roi
use PHYS_FUNC, only : &
& WL_MAX, &
& SOIL_COND_JOHANSEN
use METH_OXYG_CONSTANTS, only : &
& cpmethhydr


implicit none

! Input variables
integer(kind=iintegers), intent(in) :: is

real(kind=ireals), parameter :: cr = 0.2*4180.d0

real(kind=ireals) :: wlmax_i
real(kind=ireals) :: bb1
real(kind=ireals) :: arg
real(kind=ireals) :: psi
real(kind=ireals) :: pf
real(kind=ireals) :: lammoist1
real(kind=ireals), save, pointer :: cimh

integer(kind=iintegers) :: i
logical, save :: firstcall = .true.


if (firstcall) then
  if (tricemethhydr%par > 0.) then
    cimh => cpmethhydr
  else
    cimh => ci
  endif
endif

!rosdry(:) = 1200. !2400. 
   
do i = 1, ns
  wlmax_i = POR(i)*ROW0/(rosdry(i)*(1-por(i))) 
  BB1 = BH(i) + 2.d0
  csoil(i) = (cr + wl1(i,is)*CW + wi1(i,is)*cimh) !*1.d-3 !1.d-3 - for sensitivity experiment
  rosoil(i) = rosdry(i)*(1 - por(i))*(1 + wi1(i,is) + wl1(i,is)) 
!  ARG = (WL1(i)+WI1(i))/wlmax_i
!  ARG = max(ARG,1.d-2)
!  PSI = PSIMAX(i)*(ARG)**(-BH(i))
!  PF = log(-PSI*100.)/log(10.d0) 
!  if(PF>5.1) then
!    lamsoil(i) = 4.1E-4*418.
!  else
!    lamsoil(i) = exp(-PF-2.7)*418.
!  end if
  lamsoil(i) = SOIL_COND_JOHANSEN(wl1(i,is),wi1(i,is),rosdry(i),por(i))
  wlmax_i = WL_MAX(por(i),rosdry(i),wi1(i,is),tricemethhydr%par)
  ARG = (wl1(i,is))/wlmax_i
  ARG = max(ARG,1.d-2)
  lammoist(i) = dlmax(i)*ARG**BB1
end do

do i = 1, ns-1
  wlmax_i = WL_MAX(por(i+1),rosdry(i+1),wi1(i+1,is),tricemethhydr%par)
  if (wlmax_i < 0.01) then !wl1(i+1,is)/wlmax_i>0.98.or.
    filtr(i) = 0
  else
    wlmax_i = WL_MAX(0.5*(por(i)+por(i+1)),0.5*(rosdry(i)+rosdry(i+1)), &
    & 0.5*(wi1(i,is) + wi1(i+1,is)),tricemethhydr%par)
    filtr(i) = 0.5*(flwmax(i+1)+flwmax(i))*(0.5*(wl1(i+1,is) + &
    & wl1(i,is))/wlmax_i)**(bh(i+1)+bh(i)+3)
  endif
enddo

do i = 1, ns-1
  lammoist1 = 0.5*(lammoist(i)+lammoist(i+1))
  wsoil(i) = ( - lammoist1*(wl1(i+1,is) - wl1(i,is))/dzs(i) + &
  & filtr(i) ) *0.5*(rosdry(i)+rosdry(i+1))* &
  & (1. - 0.5*(por(i)+por(i+1)) )/row0
enddo 

if (firstcall) firstcall = .false.
END SUBROUTINE SOIL_COND_HEAT_COEF


SUBROUTINE SOILCOLSTEMP(nn,gs,gsp,dt,ls,ftot,ch4_pres_atm, &
                       & ddz,ddzi,zsoilcols, &
                       & wst,SR,a,b,c,d,add_to_winter, &
                       & bathymwater,bathymice, &
                       & bathymdice,bathymsoil, &
                       & soilflux,fdiffbot, contr) 

! Subroutine calculates temperature in soil columns,
! excepting for the lowest column, assuming 
! nsoilcols = M+1

use LAKE_DATATYPES, only : &
& iintegers, ireals

use DRIVING_PARAMS, only : soilcolconjtype

use T_SOLVER_MOD, only : &
& T_DIFF

use ATMOS, only : &
& pressure, wind10

use PHYS_CONSTANTS, only : &
& cw_m_row0, z0_bot

use PHYS_FUNC, only : &
& LOGFLUX, TEMPWATR

use ARRAYS_BATHYM, only : bathym, dhw, dhw0, layers_type
use ARRAYS_SOIL, only : csoil, rosoil, lamsoil, Tsoil3, Tsoil2, Tsoil1, &
& wl4,wi2,wa,Sals1,rosdry,por,lsm,dzs,zsoil,rosoil
use ARRAYS_GRID, only : ddz05, gridsize_type, gridspacing_type
use ARRAYS_METHANE, only : veg,TgrAnn,methgenmh,qwater,lammeth, &
& fplant,febul0, fdiff_lake_surf, &
& febul_lake_bot, febul_lake_surf, &
& plant_sum,bull_sum,oxid_sum,rprod_sum, anox, &
& rprod_total_oldC,rprod_total_newC, &
& ice_meth_oxid_total, &
& h_talik,tot_ice_meth_bubbles,rootss 
use ARRAYS_WATERSTATE, only : Tw2, waterstate_type
use ARRAYS_TURB, only: eps1
use ARRAYS_OXYGEN, only : sodbot
use ARRAYS, only : gas, dt_inv

use METHANE_LAKE_MOD, only : METHANE

use METH_OXYG_CONSTANTS, only : CH4_exp_growth

implicit none

! Input variables

type(gridsize_type), intent(inout) :: gs
type(gridspacing_type), intent(in) :: gsp
type(layers_type), intent(in) :: ls

real(kind=ireals),      intent(in) :: dt   !> timestep, st
real(kind=ireals),      intent(inout) :: ftot
real(kind=ireals),      intent(in) :: ch4_pres_atm
real(kind=ireals),      intent(in) :: ddz(1:gs%M), ddzi(1:gs%Mice), zsoilcols(1:gs%nsoilcols+1) 
type(waterstate_type),  intent(in) :: wst
real(kind=ireals),      intent(in) :: SR(0:gs%M+1) ! Shortwave radiation
type(bathym),           intent(in) :: bathymwater(1:gs%M+1), bathymice(1:gs%Mice+1) 
type(bathym),           intent(in) :: bathymdice(1:gs%Mice+1), bathymsoil(1:gs%nsoilcols+1)

integer(kind=iintegers), intent(in) :: contr(1:2)

integer(kind=iintegers), intent(in) :: nn
real(kind=ireals), intent(inout), dimension(1:nn) :: a, b, c, d

logical, intent(inout) :: add_to_winter

! Output variables
real(kind=ireals),      intent(inout) :: soilflux(1:gs%nsoilcols), fdiffbot(1:gs%nsoilcols)

! Local variables
integer(kind=iintegers), parameter :: bcswitch = 1 !1 - continuity of flux and temperature
                                                   !2 - continuity of flux, calculated from surface layer theory

integer(kind=iintegers) :: is, i ! Number of soil column
real(kind=ireals) :: Tsoilsurf(1:gs%nsoilcols), qsoilsurf(1:gs%nsoilcols)
real(kind=ireals) :: exchcoef, Tws, qwaters
real(kind=ireals), allocatable :: work(:), work1(:)
integer(kind=iintegers), allocatable :: iwork(:)
real(kind=ireals) :: xx

if ((ls%l1 /= 0. .or. ls%ls1 /= 0) .and. bcswitch == 2) then
  print*, "bcswitch == 2 is not operational when &
  &l1 /= 0. .or. ls1 /= 0", ls%l1, ls%ls1
  STOP
endif


if (bcswitch == 2) then
  allocate(work1(1:gs%nsoilcols)) 
  allocate(iwork(1:gs%nsoilcols))
  if     (soilcolconjtype == 1) then
    forall(is=1:gs%nsoilcols-1) work1(is) = bathymsoil(is)%dzSLc
    forall(is=1:gs%nsoilcols-1) iwork(is) = bathymsoil(is)%icent
  elseif (soilcolconjtype == 2) then
    forall(is=1:gs%nsoilcols-1) work1(is) = bathymsoil(is)%dzSL
    forall(is=1:gs%nsoilcols-1) iwork(is) = bathymsoil(is)%itop
  endif
endif


if (contr(1) == 1) then
  ! Heat equation in soil columns
  if (bcswitch == 1) then ! Continuity of flux and temperature across soil-water interface
    if (soilcolconjtype == 1) then
      do is = 1, gs%nsoilcols-1
        i = bathymsoil(is)%icent
        Tsoilsurf(is) = wst%Tw1(i)
      enddo
    elseif (soilcolconjtype == 2) then
      call TSURFSOILCOL(gs%M,gs%Mice,gs%nsoilcols,ls%h1,ls%l1,ls%ls1, &
                       & ddz,ddzi,zsoilcols, &
                       & wst%Tw1,wst%Ti1,wst%Tis1, &
                       & bathymwater,bathymice,bathymdice,bathymsoil,&
                       & Tsoilsurf)
    endif
    
    do is = 1, gs%nsoilcols-1
      call SOIL_COND_HEAT_COEF(is)
      call T_DIFF(1,Tsoilsurf(is),dt,0,0,0,0,is,.false.)
      soilflux(is) = &
      & csoil(1)*rosoil(1)*dt_inv*0.5*dzs(1)*( Tsoil2(1) - Tsoil1(1,is) ) - &
      & 0.25* ( lamsoil(1) + lamsoil(2) ) * &
      & ( Tsoil2(2) + Tsoil1(2,is) - Tsoil2(1) - Tsoil1(1,is) )/dzs(1)
      call SOILFORLAKE(nveclen,dt,a,b,c,d,is)
    enddo
  elseif (bcswitch == 2) then ! continuity of flux, calculated from surface layer theory above soil surface
    do is = 1, gs%nsoilcols-1
      call SOIL_COND_HEAT_COEF(is)
      i = iwork(is)
      soilflux(is) = LOGFLUX(sqrt(wst%u1(i)*wst%u1(i) + wst%v1(i)*wst%v1(i)), &
      & Tsoil1(1,is) - wst%Tw1(i), work1(is), z0_bot, z0_bot, cw_m_row0, 1) + SR(i)
      call T_DIFF(1,soilflux(is),dt,0,0,0,0,is,.true.)
      call SOILFORLAKE(nveclen,dt,a,b,c,d,is)
    enddo
  endif
endif

if (contr(2) == 1) then
  ! Methane equation in soil columns
  if (bcswitch == 1) then ! Continuity of flux and concentration across soil-water interface
    if     (soilcolconjtype == 1) then
      do is = 1, gs%nsoilcols-1
        i = bathymsoil(is)%icent
        qsoilsurf(is) = qwater(i,1)
      enddo
    elseif (soilcolconjtype == 2) then
      allocate (work(1:gs%Mice+1)); work = 0.
      call TSURFSOILCOL(gs%M,gs%Mice,gs%nsoilcols,ls%h1,ls%l1,ls%ls1, &
                       & ddz,ddzi,zsoilcols, &
                       & qwater(1,1),work,work, &
                       & bathymwater,bathymice,bathymdice,bathymsoil,&
                       & qsoilsurf)
      deallocate (work)
    endif
    
    allocate (work(0:gs%M+1))
    do is = 1, gs%nsoilcols-1
      gs%isoilcol = is
      call METHANE &
      & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,is),rosoil, &
      & work, work, &
      & wl4(1,is),wi2(1,is),wa,Sals1, &
      & rootss,rosdry,por,veg,qsoilsurf(is)*por(1),TgrAnn, methgenmh, &
      & ddz,ddz05, wst, lammeth, &
      & gsp%z_half(bathymsoil(is)%icent), & 
      & ls, dhw, dhw0, .false., .false., lsm, bathymwater, &
      & fplant, febul0(is), fdiffbot(is), ftot, fdiff_lake_surf, &
      & febul_lake_bot, febul_lake_surf, &
      & plant_sum,bull_sum,oxid_sum,rprod_sum, &
      & anox,gs,gsp,dt,eps1(1),sodbot, &
      & rprod_total_oldC, rprod_total_newC, ice_meth_oxid_total, &
      & h_talik,tot_ice_meth_bubbles,add_to_winter) 
    enddo
    deallocate(work)
  elseif (bcswitch == 2) then! continuity of flux, calculated from surface layer theory above soil surface
    allocate (work(0:gs%M+1))
    do is = 1, gs%nsoilcols-1
      i = iwork(is)
      ! Exchange coefficient
      !exchcoef = LOGFLUX(sqrt(wst%u1(i)*wst%u1(i) + wst%v1(i)*wst%v1(i)), &
      !& qsoil(1,is)/por(1) - qwater(i,1), bathymsoil(is)%dzSL, z0_bot, z0_bot, 1._ireals,2)
      ! Methane concentration in water from similarity profile
      !qwaters = TEMPWATR(lammeth(i),bathymwater(i)%rad_int,qsoil(1,is)/por(1),exchcoef, &
      !& qwater(i,1), lammeth(i)*(qwater(i+1,1) - qwater(i,1))/(ddz(i)*h1))
      ! Methane flux at the bottom
      ! Taking into account depletion of methane in top sediments
      if (CH4_exp_growth /= 0.) then
        xx = CH4_exp_growth*0.5*dzs(1) / ( (exp(CH4_exp_growth*0.5*dzs(1)) - 1.) * por(1) )
      else
        xx = 1./por(1)
      endif
      fdiffbot(is) = LOGFLUX(sqrt(wst%u1(i)*wst%u1(i) + wst%v1(i)*wst%v1(i)), &
      & gas%qsoil(1,is)*xx - qwater(i,1), work1(is), &
      & z0_bot, z0_bot, 1._ireals,1)

      gs%isoilcol = is
      call METHANE &
      & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,is),rosoil, &
      & work, work, &
      & wl4(1,is),wi2(1,is),wa,Sals1, &
      & rootss,rosdry,por,veg,fdiffbot(is),TgrAnn, methgenmh, &
      & ddz,ddz05, wst, lammeth, &
      & gsp%z_half(bathymsoil(is)%icent), & 
      & ls, dhw, dhw0, .false., .true., lsm, bathymwater, &
      & fplant, febul0(is), fdiffbot(is), ftot, fdiff_lake_surf, &
      & febul_lake_bot, febul_lake_surf, &
      & plant_sum,bull_sum,oxid_sum,rprod_sum, &
      & anox,gs,gsp,dt,eps1(1),sodbot, &
      & rprod_total_oldC, rprod_total_newC, ice_meth_oxid_total, &
      & h_talik,tot_ice_meth_bubbles,add_to_winter) 
    enddo
    deallocate(work)
  endif

endif

if (bcswitch == 2) then
  deallocate(work1) 
  deallocate(iwork)
endif

END SUBROUTINE SOILCOLSTEMP


SUBROUTINE TSURFSOILCOL(M,Mice,nsoilcols,h1,l1,ls1,ddz,ddzi,zsoilcols, &
                       & Tw1,Ti1,Tis1, &
                       & bathymwater,bathymice,bathymdice,bathymsoil, &
                       & Tsoilsurf)

! Subroutine TSURFSOILCOL calculates mean temperature
! of a lake cross-section over specified depth intervals


use NUMERIC_PARAMS!, only : &
!& small_value, pi

use ARRAYS_BATHYM, bathymwater_=>bathymwater,bathymice_=>bathymice, &
& bathymsoil_=>bathymsoil,bathymdice_=>bathymdice,&
& h1_=>h1,l1_=>l1,ls1_=>ls1!, only : &
!& bathym

implicit none

!Input variables
integer(kind=iintegers), intent(in) :: M, Mice, nsoilcols
real(kind=ireals),    intent(in) :: h1, l1, ls1
real(kind=ireals),    intent(in) :: ddz(1:M), ddzi(1:Mice)
real(kind=ireals),    intent(in) :: zsoilcols(1:nsoilcols+1)
real(kind=ireals),    intent(in) :: Tw1(1:M+1)
real(kind=ireals),    intent(in) :: Ti1(1:Mice+1), Tis1(1:Mice+1)
type(bathym), intent(in) :: bathymwater(1:M+1)  , bathymice(1:Mice+1) 
type(bathym), intent(in) :: bathymdice(1:Mice+1), bathymsoil(1:nsoilcols+1)

!Output variables
real(kind=ireals), intent(out) :: Tsoilsurf(1:nsoilcols)

! Local variables
integer(kind=iintegers) :: i, j, j0, j1, nl 

real(kind=ireals), allocatable :: Temp(:), z(:), per(:), dlx(:), dly(:), aell(:), bell(:)
real(kind=ireals) :: pers,dz,aells,bells,areas,area,dlxs,dlys
!type(bathym),      allocatable :: bathymall(:)

logical, save :: firstcall = .true.

! Water, ice and deep ice
if (l1  > small_value .and. &
  & h1  > small_value .and. &
  & ls1 > small_value) then

  nl = 2*Mice + M
  allocate (Temp(1:nl+1), z(0:nl+1), per(0:nl+1), dlx(0:nl+1), dly(0:nl+1))
  allocate (aell(0:nl+1), bell(0:nl+1))
!  allocate (bathymall(1:nl+1))

  ! Ice top
  z(0)    = 0.                  ; z(1)   = 0.5*ddzi(1)*l1
  aell(0)  = 0.5*bathymice(1)%Lx ; aell(1) = 0.5*bathymice(1)%Lx_half
  bell(0)  = 0.5*bathymice(1)%Ly ; bell(1) = 0.5*bathymice(1)%Ly_half
  Temp(1) = Ti1(1)
  j = 2

  ! Ice interior
  do i = 2, Mice
    Temp(j) = Ti1(i)
    aell(j) = 0.5*bathymice(i)%Lx_half
    bell(j) = 0.5*bathymice(i)%Ly_half   
    z(j)  = z(j-1) + 0.5*(ddzi(i-1) + ddzi(i))*l1
    j = j + 1
  enddo

  ! Ice-water interface
  Temp(j) = Ti1(Mice+1)
  z(j) = z(j-1) + 0.5*(ddzi(Mice)*l1 + ddz(1)*h1)
  aell(j) = 0.5*bathymwater(1)%Lx_half
  bell(j) = 0.5*bathymwater(1)%Ly_half
  j = j + 1

  ! Water interior
  do i = 2, M
    Temp(j) = Tw1(i)
    aell(j) = 0.5*bathymwater(i)%Lx_half
    bell(j) = 0.5*bathymwater(i)%Ly_half
    z(j) = z(j-1) + 0.5*(ddz(i-1) + ddz(i))*h1
    j = j + 1
  enddo

  ! Water-deepice interface
  Temp(j) = Tw1(M+1)
  z(j) = z(j-1) + 0.5*(ddz(M)*h1 + ddzi(1)*ls1)
  aell(j) = 0.5*bathymdice(1)%Lx_half
  bell(j) = 0.5*bathymdice(1)%Ly_half
  j = j + 1

  ! Deepice interior
  do i = 2, Mice
    Temp(j) = Tis1(i)
    aell(j) = 0.5*bathymdice(i)%Lx_half
    bell(j) = 0.5*bathymdice(i)%Ly_half
    z(j) = z(j-1) + 0.5*(ddzi(i-1) + ddzi(i))*ls1
    j = j + 1
  enddo 

  ! Deepice bottom
  Temp(j) = Tis1(Mice+1)
  z(j) = z(j-1) + 0.5*ddzi(Mice)*ls1
  aell(j) = 0.5*bathymdice(Mice+1)%Lx
  bell(j) = 0.5*bathymdice(Mice+1)%Ly

  z(:) = z(:) + zsoilcols(nsoilcols+1) - h1 - l1 - ls1
endif

! Water and ice
if (l1  > small_value .and. &
  & h1  > small_value .and. &
  & ls1 < small_value) then

  nl = Mice + M
  allocate (Temp(1:nl+1), z(0:nl+1), per(0:nl+1), dlx(0:nl+1), dly(0:nl+1))
  allocate (aell(0:nl+1), bell(0:nl+1))
!  allocate (bathymall(1:nl+1))

  ! Ice top
  z(0)    = 0.                  ; z(1)   = 0.5*ddzi(1)*l1
  aell(0)  = 0.5*bathymice(1)%Lx ; aell(1) = 0.5*bathymice(1)%Lx_half
  bell(0)  = 0.5*bathymice(1)%Ly ; bell(1) = 0.5*bathymice(1)%Ly_half
  Temp(1) = Ti1(1)
  j = 2

  do i = 2, Mice
    Temp(j) = Ti1(i)
    aell(j) = 0.5*bathymice(i)%Lx_half
    bell(j) = 0.5*bathymice(i)%Ly_half   
    z(j)  = z(j-1) + 0.5*(ddzi(i-1) + ddzi(i))*l1
    j = j + 1
  enddo

  ! Ice-water interface
  Temp(j) = Ti1(Mice+1)
  z(j) = z(j-1) + 0.5*(ddzi(Mice)*l1 + ddz(1)*h1)
  aell(j) = 0.5*bathymwater(1)%Lx_half
  bell(j) = 0.5*bathymwater(1)%Ly_half
  j = j + 1

  ! Water interior
  do i = 2, M
    Temp(j) = Tw1(i)
    aell(j) = 0.5*bathymwater(i)%Lx_half
    bell(j) = 0.5*bathymwater(i)%Ly_half
    z(j) = z(j-1) + 0.5*(ddz(i-1) + ddz(i))*h1
    j = j + 1
  enddo

  ! Water bottom
  Temp(j) = Tw1(M+1)
  z(j) = z(j-1) + 0.5*ddz(M)*h1
  aell(j) = 0.5*bathymwater(M+1)%Lx
  bell(j) = 0.5*bathymwater(M+1)%Ly

  z(:) = z(:) + zsoilcols(nsoilcols+1) - h1 - l1
endif

! Water and deep ice
if (l1  < small_value .and. &
  & h1  > small_value .and. &
  & ls1 > small_value) then

  nl = Mice + M
  allocate (Temp(1:nl+1), z(0:nl+1), per(0:nl+1), dlx(0:nl+1), dly(0:nl+1))
  allocate (aell(0:nl+1), bell(0:nl+1))
!  allocate (bathymall(1:nl+1))

  ! Water top
  z(0)    = 0.                    ; z(1)   = 0.5*ddz(1)*h1
  aell(0)  = 0.5*bathymwater(1)%Lx ; aell(1) = 0.5*bathymwater(1)%Lx_half
  bell(0)  = 0.5*bathymwater(1)%Ly ; bell(1) = 0.5*bathymwater(1)%Ly_half
  Temp(1) = Tw1(1)
  j = 2

  ! Water interior
  do i = 2, M
    Temp(j) = Tw1(i)
    aell(j) = 0.5*bathymwater(i)%Lx_half
    bell(j) = 0.5*bathymwater(i)%Ly_half
    z(j) = z(j-1) + 0.5*(ddz(i-1) + ddz(i))*h1
    j = j + 1
  enddo

  ! Water-deepice interface
  Temp(j) = Tw1(M+1)
  z(j) = z(j-1) + 0.5*(ddz(M)*h1 + ddzi(1)*ls1)
  aell(j) = 0.5*bathymdice(1)%Lx_half
  bell(j) = 0.5*bathymdice(1)%Ly_half
  j = j + 1

  ! Deep ice interior
  do i = 2, Mice
    Temp(j) = Tis1(i)
    aell(j) = 0.5*bathymdice(i)%Lx_half
    bell(j) = 0.5*bathymdice(i)%Ly_half
    z(j) = z(j-1) + 0.5*(ddzi(i-1) + ddzi(i))*ls1
    j = j + 1
  enddo

  ! Deepice bottom
  Temp(j) = Tis1(Mice+1)
  z(j) = z(j-1) + 0.5*ddzi(Mice)*ls1
  aell(j) = 0.5*bathymdice(Mice+1)%Lx
  bell(j) = 0.5*bathymdice(Mice+1)%Ly

  z(:) = z(:) + zsoilcols(nsoilcols+1) - h1 - ls1
endif

! Water only
if (l1  < small_value .and. &
  & h1  > small_value .and. &
  & ls1 < small_value) then

  nl = M
  allocate (Temp(1:nl+1), z(0:nl+1), per(0:nl+1), dlx(0:nl+1), dly(0:nl+1))
  allocate (aell(0:nl+1), bell(0:nl+1))
!  allocate (bathymall(1:nl+1))

  ! Water top
  z(0)    = 0.                    ; z(1)   = 0.5*ddz(1)*h1
  aell(0)  = 0.5*bathymwater(1)%Lx ; aell(1) = 0.5*bathymwater(1)%Lx_half
  bell(0)  = 0.5*bathymwater(1)%Ly ; bell(1) = 0.5*bathymwater(1)%Ly_half
  Temp(1) = Tw1(1)
  j = 2

  do i = 2, M
    Temp(j) = Tw1(i)
    aell(j) = 0.5*bathymwater(i)%Lx_half
    bell(j) = 0.5*bathymwater(i)%Ly_half
    z(j) = z(j-1) + 0.5*(ddz(i-1) + ddz(i))*h1
    j = j + 1
  enddo

  ! Water bottom
  Temp(j) = Tw1(M+1)
  z(j) = z(j-1) + 0.5*ddz(M)*h1
  aell(j) = 0.5*bathymwater(M+1)%Lx
  bell(j) = 0.5*bathymwater(M+1)%Ly

  z(:) = z(:) + zsoilcols(nsoilcols+1) - h1
endif

!! Cross-section parameters calculation, assuming elliptic shape
!do i = 0, nl+1
!  per(i) = ELLAR(aell(i),bell(i))
!enddo
!! Assuming cross-section area reduces with depth
!do i = 0, nl
!  dlx(i) = aell(i) - aell(i+1) 
!  dly(i) = bell(i) - bell(i+1)
!  dlx(i) = sqrt(( z(i+1) - z(i) ) * ( z(i+1) - z(i) ) + dlx(i)*dlx(i))
!  dly(i) = sqrt(( z(i+1) - z(i) ) * ( z(i+1) - z(i) ) + dly(i)*dly(i))
!enddo

do i = 1, nsoilcols-1 ! Except for the lowest soil column
  ! Seeking the gridcells falling into the depth interval of current soil column
  if (zsoilcols(i) < z(0)) then
    j0 = 1
  else
    do j = 0, nl
      if (z(j) <= zsoilcols(i) .and. z(j+1) > zsoilcols(i)) then
        j0 = j+1
        exit
      endif
    enddo 
  endif
  do j = j0, nl
    if (z(j) <= zsoilcols(i+1) .and. z(j+1) > zsoilcols(i+1)) then
      j1 = j+1
      exit
    endif
  enddo
  
  if (j1 <= j0) print*, 'Warning: water does not flood all soil columns'

  ! Integrating temperature over the range of depths of current soil column,
  ! assuming at least two numerical layers overlay with this range.
  ! The bottom temperature is areally-averaged.
  areas = 0.
  Tsoilsurf(i) = 0.

  ! Top numerical layer, overlapping with the soil column depth interval
  if (zsoilcols(i) > z(j0-1)) then
    ! Soil column top is inside the numerical layer
    !dz = z(j0) - zsoilcols(i)
    aells = 0.5*bathymsoil(i)%Lx; bells = 0.5*bathymsoil(i)%Ly
    !pers  = ELLAR(aells,bells)
    !dlxs  = sqrt((aells - aell(j0))*(aells - aell(j0)) + dz*dz)
    !dlys  = sqrt((bells - bell(j0))*(bells - bell(j0)) + dz*dz)
    !area  = 0.25*(per(j0) + pers)*(dlxs + dlys)
    area  = pi*(aells*bells - aell(j0)*bell(j0))
    Tsoilsurf(i) = Tsoilsurf(i) + Temp(j0)*area !(zsoilcols(i+1) - z(j1-1))
    areas = areas + area
  else
    ! Soil column top is outside the numerical layer (j0 = 1)
    ! area = 0.25*(per(j0-1) + per(j0))*(dlx(j0-1) + dly(j0-1))
    area  = pi*(aell(j0-1)*bell(j0-1) - aell(j0)*bell(j0))
    Tsoilsurf(i) = Tsoilsurf(i) + Temp(j0)*area  !(z(j) - z(j-1))
    areas = areas + area
  endif

  ! Interior numerical layers
  if (j0+1 <= j1-1) then
    do j = j0+1, j1-1
      !area = 0.25*(per(j-1) + per(j))*(dlx(j-1) + dly(j-1))
      area  = pi*(aell(j-1)*bell(j-1) - aell(j)*bell(j))
      Tsoilsurf(i) = Tsoilsurf(i) + Temp(j)*area  !(z(j) - z(j-1))
      areas = areas + area
    enddo
  endif

 ! Lowest numerical layer, overlapping with the soil column depth interval
  !dz = zsoilcols(i+1) - z(j1-1)
  aells = 0.5*bathymsoil(i+1)%Lx; bells = 0.5*bathymsoil(i+1)%Ly
  !pers  = ELLAR(aells,bells)
  !dlxs  = sqrt((aell(j1-1) - aells)*(aell(j1-1) - aells) + dz*dz)
  !dlys  = sqrt((bell(j1-1) - bells)*(bell(j1-1) - bells) + dz*dz)
  !area  = 0.25*(per(j1-1) + pers)*(dlxs + dlys)
  area  = pi*(aell(j1-1)*bell(j1-1) - aells*bells)
  Tsoilsurf(i) = Tsoilsurf(i) + Temp(j1)*area !(zsoilcols(i+1) - z(j1-1))
  areas = areas + area

  ! Averaging
  Tsoilsurf(i) = Tsoilsurf(i)/areas

enddo

deallocate(Temp, z, per, dlx, dly, aell, bell)
!deallocate(bathymall)

if (firstcall) firstcall = .false.


contains
FUNCTION ELLAR(a,b)
! Approximate formula for the ellipse perimeter
implicit none

real(kind=ireals) :: ELLAR

real(kind=ireals), intent(in) :: a, b

ELLAR = pi*(3.*(a + b) - sqrt( (3.*a + b)*(a + 3.*b) ) )

END FUNCTION ELLAR


END SUBROUTINE TSURFSOILCOL


SUBROUTINE LATERHEAT(ix,iy,gs,ls,btmw,btmi,btmdi,btms,gsp,soilflux,onlywater,lsh)

!Subroutine LATERHEAT calculates heat sources from lateral heat exchange
! with soil columns for water, ice and deep ice layers

use ARRAYS_BATHYM,ls_=>ls!, only : bathym, layers_type
use ARRAYS_GRID,gs_=>gs,gsp_=>gsp!, only : gridsize_type, gridspacing_type
use ARRAYS_SOIL,lsh_=>lsh,soilflux_=>soilflux!, only : lsh_type

use NUMERIC_PARAMS!, only : &
!& small_value

implicit none

! Input variables
integer(kind=iintegers), intent(in) :: ix, iy
type(gridsize_type),     intent(in) :: gs ! Gridsizes
type(layers_type),       intent(in) :: ls ! Layers thicknesses
type(bathym),            intent(in) :: btmw(1:gs%M+1), btmi(1:gs%Mice+1) !Bathymetries
type(bathym),            intent(in) :: btmdi(1:gs%Mice+1), btms(1:gs%nsoilcols+1) ! Bathymetries 
type(gridspacing_type),  intent(in) :: gsp ! Grid spacing

real(kind=ireals), intent(in) :: soilflux(1:gs%nsoilcols) ! Heat flux into soil, W/m**2

logical, intent(in) :: onlywater ! Indicates whether the lateral heat influx is calculated for water column only

! Input/output variables

! Output variables
type(lsh_type), intent(inout) :: lsh ! Heat source from soil columns for water, ice and deep ice

! Local variables
real(kind=ireals) :: offset
real(kind=ireals), allocatable :: z(:), area(:)
integer(kind=iintegers) :: i, j, i0, i1


! Water layer exists
if (ls%h1 > small_value) then

  offset = maxval(gsp%zsoilcols) - ls%h1 - STEP(ls%ls1 - small_value)*ls%ls1
  allocate ( z(0:gs%M+1), area(0:gs%M+1) )
  z(0) = offset ; area(0) = btmw(1)%area_int
  z(1:gs%M) = offset + gsp%z_half(1:gs%M) ; area(1:gs%M) = btmw(1:gs%M)%area_half
  z(gs%M+1) = offset + ls%h1 ; area(gs%M+1) = btmw(gs%M+1)%area_int

  lsh%water(:) = 0.
  waterscan : do j = 1, gs%nsoilcols-1

    if ( OVERLAY(z(0),z(gs%M+1),gsp%zsoilcols(j,ix,iy),gsp%zsoilcols(j+1,ix,iy)) ) then

      if (z(0) <= gsp%zsoilcols(j,ix,iy)) then
        i0 = -1
        do i = 1, gs%M+1
          if ( z(i-1) <  gsp%zsoilcols(j,ix,iy) .and. &
          &    z(i)   >  gsp%zsoilcols(j,ix,iy) ) then
            i0 = i
            exit
          endif
        enddo
      else
        i0 = -1
      endif
      if (i0 /= -1) then
        lsh%water(i0) = lsh%water(i0) + 1./btmw(i0)%area_int * &
        & ( area(i0) - btms(j)%area_int ) / (ls%h1*gsp%ddz05(i0-1)) * soilflux(j)
      endif 

      if (z(gs%M+1) >= gsp%zsoilcols(j+1,ix,iy)) then
        i1 = -1
        do i = 1, gs%M+1
          if ( z(i-1) <  gsp%zsoilcols(j+1,ix,iy) .and. &
          &    z(i)   >  gsp%zsoilcols(j+1,ix,iy) ) then
            i1 = i
            exit
          endif
        enddo
      else
        i1 = -1
      endif
      if (i1 /= -1) then
        lsh%water(i1) = + 1./btmw(i1)%area_int * &
        & ( btms(j+1)%area_int - area(i1-1) ) / (ls%h1*gsp%ddz05(i1-1)) * soilflux(j)
      endif

      forall ( i = 1:gs%M+1, z(i-1) >= gsp%zsoilcols(j,ix,iy) &
      & .and. z(i) <= gsp%zsoilcols(j+1,iy,iy) ) &
      & lsh%water(i) = + 1./btmw(i)%area_int * &
      & ( area(i) - area(i-1) )/ (ls%h1*gsp%ddz05(i-1)) * soilflux(j)

      if (z(gs%M+1) <= gsp%zsoilcols(j,ix,iy)) exit waterscan

    endif
  enddo waterscan

  deallocate(z,area)
endif

! Ice layer exists
if (ls%l1 > small_value .and. (.not. onlywater) ) then

  offset = maxval(gsp%zsoilcols) - ls%l1 &
  & - STEP(ls%ls1 - small_value)*ls%ls1 - STEP(ls%h1 - small_value)*ls%h1
  allocate ( z(0:gs%Mice+1), area(0:gs%Mice+1) )
  z(0) = offset ; area(0) = btmi(1)%area_int
  !print*, 'z0', z(0), ls%l1, gsp%ddzi05(gs%Mice), gsp%ddzi05(0)
  do i = 1, gs%Mice+1 
    z(i) = z(i-1) + gsp%ddzi05(i-1)*ls%l1
  enddo  
  area(1:gs%Mice) = btmi(1:gs%Mice)%area_half  
  area(gs%Mice+1) = btmi(gs%Mice+1)%area_int

  lsh%ice(:) = 0.
  icescan : do j = 1, gs%nsoilcols-1

    if ( OVERLAY(z(0),z(gs%Mice+1),gsp%zsoilcols(j,ix,iy),gsp%zsoilcols(j+1,ix,iy)) ) then

      if (z(0) <= gsp%zsoilcols(j,ix,iy)) then
        i0 = -1
        do i = 1, gs%Mice+1
          if ( z(i-1) <  gsp%zsoilcols(j,ix,iy) .and. &
          &    z(i)   >  gsp%zsoilcols(j,ix,iy) ) then
            i0 = i
            exit
          endif
        enddo
      else
        i0 = -1
      endif
      if (i0 /= -1) then
        lsh%ice(i0) = lsh%ice(i0) + 1./btmi(i0)%area_int * &
        & ( area(i0) - btms(j)%area_int ) / (ls%l1*gsp%ddzi05(i0-1)) * soilflux(j)
      endif 

      if (z(gs%Mice+1) >= gsp%zsoilcols(j+1,ix,iy)) then
        i1 = -1
        do i = 1, gs%Mice+1
          if ( z(i-1) <  gsp%zsoilcols(j+1,ix,iy) .and. &
          &    z(i)   >  gsp%zsoilcols(j+1,ix,iy) ) then
            i1 = i
            exit
          endif
        enddo
      else
        i1 = -1
      endif
      if (i1 /= -1) then
        lsh%ice(i1) = + 1./btmi(i1)%area_int * &
        & ( btms(j+1)%area_int - area(i1-1) ) / (ls%l1*gsp%ddzi05(i1-1)) * soilflux(j)
      endif

      forall ( i = 1:gs%Mice+1, z(i-1) >= gsp%zsoilcols(j,ix,iy) &
      & .and. z(i) <= gsp%zsoilcols(j+1,iy,iy) ) &
      & lsh%ice(i) = + 1./btmi(i)%area_int * &
      & ( area(i) - area(i-1) )/ (ls%l1*gsp%ddzi05(i-1)) * soilflux(j)

      if ( z(gs%Mice+1) <= gsp%zsoilcols(j,ix,iy) ) exit icescan

    endif
  enddo icescan

  deallocate(z,area)
endif

! Deep ice layer exists
if (ls%ls1 > small_value .and. (.not. onlywater) ) then

  offset = maxval(gsp%zsoilcols) - ls%ls1
  allocate ( z(0:gs%Mice+1), area(0:gs%Mice+1) )
  z(0) = offset ; area(0) = btmdi(1)%area_int
  do i = 1, gs%Mice+1
    z(i) = z(i-1) + gsp%ddzi05(i-1)*ls%ls1
  enddo
  area(1:gs%Mice) = btmdi(1:gs%Mice)%area_half  
  area(gs%Mice+1) = btmdi(gs%Mice+1)%area_int

  lsh%dice(:) = 0.
  dicescan : do j = 1, gs%nsoilcols-1

    if ( OVERLAY(z(0),z(gs%Mice+1),gsp%zsoilcols(j,ix,iy),gsp%zsoilcols(j+1,ix,iy)) ) then

      if (z(0) <= gsp%zsoilcols(j,ix,iy)) then
        i0 = -1
        do i = 1, gs%Mice+1
          if ( z(i-1) <  gsp%zsoilcols(j,ix,iy) .and. &
          &    z(i)   >  gsp%zsoilcols(j,ix,iy) ) then
            i0 = i
            exit
          endif
        enddo
      else
        i0 = -1
      endif
      if (i0 /= -1) then
        lsh%dice(i0) = lsh%dice(i0) + 1./btmdi(i0)%area_int * &
        & ( area(i0) - btms(j)%area_int ) / (ls%ls1*gsp%ddzi05(i0-1)) * soilflux(j)
      endif 

      if (z(gs%Mice+1) >= gsp%zsoilcols(j+1,ix,iy)) then
        i1 = -1
        do i = 1, gs%Mice+1
          if ( z(i-1) <  gsp%zsoilcols(j+1,ix,iy) .and. &
          &    z(i)   >  gsp%zsoilcols(j+1,ix,iy) ) then
            i1 = i
            exit
          endif
        enddo
      else
        i1 = -1
      endif
      if (i1 /= -1) then
        lsh%dice(i1) = + 1./btmdi(i1)%area_int * &
        & ( btms(j+1)%area_int - area(i1-1) ) / (ls%ls1*gsp%ddzi05(i1-1)) * soilflux(j)
      endif

      forall ( i = 1:gs%Mice+1, z(i-1) >= gsp%zsoilcols(j,ix,iy) &
      & .and. z(i) <= gsp%zsoilcols(j+1,iy,iy) ) &
      & lsh%dice(i) = + 1./btmdi(i)%area_int * &
      & ( area(i) - area(i-1) )/ (ls%ls1*gsp%ddzi05(i-1)) * soilflux(j)

      if ( z(gs%Mice+1) <= gsp%zsoilcols(j,ix,iy) ) exit dicescan

    endif
  enddo dicescan

  deallocate(z,area)
endif

contains
FUNCTION OVERLAY(x1,x2,y1,y2)
! Function OVERLAY checks, if two intervals overlay
! x2>x1, y2>y1
implicit none

logical :: OVERLAY
real(kind=ireals), intent(in) :: x1, x2, y1, y2 

OVERLAY = (y1 >= x1 .and. y2 <= x2) .or. &
        & (y1 <= x1 .and. y2 >= x2) .or. &
        & (y1 <= x1 .and. y2 >  x1) .or. &
        & (y1 <  x2 .and. y2 >= x2)

END FUNCTION OVERLAY
END SUBROUTINE LATERHEAT

!> Function AVERSOILCOL averages a property over soil columns using their areas
FUNCTION AVERSOILCOL(f,areas,nsoilcols) result(faver)
implicit none
!Input/output arguments
real(kind=ireals), intent(in) :: f(1:nsoilcols) !< A property to be averaged
real(kind=ireals), intent(in) :: areas(1:nsoilcols+1) !< Areas of soil columns, m
integer(kind=iintegers), intent(in) :: nsoilcols !< Number of soil columns, n/d
real(kind=ireals) :: faver
!Local variables
integer(kind=iintegers) :: is

if (nsoilcols > 1) then
  faver = 0.
  do is = 1, nsoilcols-1
    faver = faver + f(is)*(areas(is) - areas(is+1))
  enddo
  faver = faver + f(nsoilcols)*areas(nsoilcols)
  faver = faver/areas(1)
else
  faver = f(nsoilcols)
endif

END FUNCTION AVERSOILCOL

END MODULE SOIL_LAKE_MOD

