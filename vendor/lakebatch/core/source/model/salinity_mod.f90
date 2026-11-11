MODULE SALINITY_LAKE_MOD

use NUMERICS, only : PROGONKA, CHECK_PROGONKA, IND_STAB_FACT_DB, &
&                    nveclen        
use NUMERIC_PARAMS, only : nveclen, small_value
use LAKE_DATATYPES, only : ireals, iintegers
use DZETA_MOD, only : VARMEAN
use ARRAYS_BATHYM, only : layers_type
use ARRAYS_WATERSTATE, only : waterstate_type

contains

!> Subroutine solves salinity diffusion
SUBROUTINE S_DIFF(dt,ls,salice)

use ARRAYS, only : &
& deepice, water, &
& water_salinity_indic, soil_salinity_indic, &
& nstep
use ARRAYS_WATERSTATE, only : lamsal, Sal1, Sal2
use ARRAYS_BATHYM, only : h1, l1, dhw0, dhw, bathymwater
use ARRAYS_GRID, only : ddz, nsoilcols, ddzi05
use ARRAYS_SOIL, only : wsoil, Sals1, Sals2,rosdry,por,dzs

use DRIVING_PARAMS, only : &
& sedim, soilswitch, M, Mice, ns, salsoil, &
& saltice

use ATMOS, only: &
& Sflux0

use PHYS_FUNC, only: &
& w_sedim

use PHYS_CONSTANTS, only : &
& roi_d_row0, &
& salice0, &
& row0, &
& asalice

use T_SOLVER_MOD, only : &
& DIFF_COEF


implicit none
      
! S_diff solves salinity (mineralization) diffusion equation  

! Input variables
real(kind=ireals), intent(in) :: dt !> Time step, s
type(layers_type), intent(in) :: ls !> Data structure with sizes of physical layers
real(kind=ireals), intent(in) :: salice(1:Mice+1) !> Ice layer salinity, kg/kg

! Local variables
real(kind=ireals) :: Sflux1,Sflux_soil_bot,x,y,z,zz=0.
!real(kind=ireals), dimension(1:nveclen):: a,b,c,d,Sal
real(kind=ireals), allocatable :: a(:), b(:), c(:), d(:), Sal(:)

integer(kind=iintegers) :: i !loop index

allocate(a(1:nveclen), b(1:nveclen), c(1:nveclen), &
&        d(1:nveclen), Sal(1:nveclen))
a(:) = 0.; b(:) = 0.; c(:) = 0.; d(:) = 0

! Sflux1 --- salinity flux at the bottom boundary       
! Sflux0 --- salinity flux at the top boundary       

Sflux1 = 0.d0
Sflux_soil_bot = 0.d0
if (ls%l1 > 0.) then
! Salinity flux to water = flux from atmosphere + flux from melting/freezing of saline ice
!  if (ls%dhilow < 0.) then
!    x = max(ls%dhilow,-l1)
!  else
!    x = dhilow
!  endif
!  salbot = Sal1(1)*poricebot
! Currently, the ice melted on the top of ice layer is assumed to have zero salinity
  Sflux0 = Sflux0 + (ls%dhwhigh)/dt   * ( salice(     1)/row0 - Sal1(1) ) + &
  &                 (ls%dhwlow) /dt   * ( salice(Mice+1)/row0 - Sal1(1) ) + &
  &                 (ls%dhwsnmelt + ls%dhip)/dt * ( 0.        - Sal1(1) )
  !zz = zz + (ls%dhwsnmelt + ls%dhip)*1000.
  !if (ls%dhwsnmelt /= 0. .or. ls%dhip /= 0.) print*, zz, ls%hs1, ls%dhwsnmelt, ls%dhip
  if (saltice%par == 1) then
    ! Flux of salty water in ice cover through cracks to water layer
    x = 0.
    do i = 1, Mice
      x = x + asalice * salice(i) * ddzi05(i-1) * l1/row0
    enddo
    Sflux0 = Sflux0 + x
  endif
else 
  Sflux0 = Sflux0 + (ls%dhwp + ls%dhwe)     /dt*(0. - Sal1(1)) 
endif
if (ls%ls1 > 0.) Sflux1 = Sflux1 - ls%dhwls /dt*( salice0 - Sal1(M+1) )

! It is allowed for atmospheric aerosol to immediately go to 
! water instead of first contaminating the snow cover.
! if (ice == 1) Sflux0 = 0.d0

! It is assumed, that the water freezing to ice, melting from ice and snow,
! the rain are all freshwater sinks/sources. It is not true in fact, as far as
! the acid rain may occur and contaminated snow thaw. For these cases
! the present scheme must be updated.

! Defines, if gravitational sedimentation of tracer is taken into account
! sedim = 1 it is taken into account
! sedim = 0 it is neglected
! sedim = 0

if (water == 1) then
  call DIFF_COEF(nveclen,a,b,c,d,2,M,2,M,water_salinity_indic,dt)
  x = 0.5*( - bathymwater(1)%area_half*lamsal(1)/(h1*ddz(1)* &
  & bathymwater(1)%area_int) + dhw0/(2.d0*dt) )
  c(1)   = x - ddz(1)*h1/(2.d0*dt) ! - dhw0/(2*dt) is wrong sign!
  b(1)   = x
  d(1)   = - ddz(1)*h1*Sal1(1)/(2.d0*dt) - Sflux0 + x*(Sal1(2) - Sal1(1))
  if (deepice == 1) then
!   Case water,deepice and soil; upper ice and snow are allowed       
!   Salinity diffusion in water       
    x = 0.5*( - bathymwater(M)%area_half*lamsal(M)/(ddz(M)*h1* &
    & bathymwater(M+1)%area_int) + (dhw-dhw0)/(2.d0*dt) )
    c(M+1) = x - ddz(M)*h1/(2.d0*dt) 
    a(M+1) = x
    d(M+1) = - Sal1(M+1)*ddz(M)*h1/(2.d0*dt) + Sflux1 - x*(Sal1(M+1) - Sal1(M))
    call PROGONKA (nveclen,a,b,c,d,Sal,1,M+1)
    Sal (1:M+1) = max(Sal(1:M+1),0._ireals)
    Sal2(1:M+1) = Sal(1:M+1)
    if (soilswitch%par == 1 .and. salsoil%par == 1) then
!     Salinity diffusion in soil
      call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,2,ns-1,soil_salinity_indic,dt)
      x = 0.5*dt*wsoil(1)/dzs(1)
      c(1) = - 1.d0 - x
      b(1) = x
      d(1) = - Sals1(1,nsoilcols) + x*(Sals1(2,nsoilcols) + Sals1(1,nsoilcols))
      x = 0.5*wsoil(ns-1)*dt/dzs(ns-1)
      c(ns) = - 1.d0 + x
      a(ns) = - x
      d(ns) = - Sals1(ns,nsoilcols) + 2.d0*dt*Sflux_soil_bot/dzs(ns-1) - &
      & x*(Sals1(ns,nsoilcols) + Sals1(ns-1,nsoilcols))
      !print*, CHECK_PROGONKA(ns,a,b,c,d,Sal)
      call PROGONKA (nveclen,a,b,c,d,Sal,1,ns)
      Sal(1:ns) = max(Sal(1:ns),0._ireals)
      Sals2(1:ns,nsoilcols) = Sal(1:ns)
    else
      Sals2(:,nsoilcols) = Sals1(:,nsoilcols)
    endif
  else
!   Case water, soil; upper ice and snow are allowed       
    if (soilswitch%par == 1 .and. salsoil%par == 1) then
!------WATER-SOIL INTERFACE-------------------------
      x = 0.5*( 0.5*(dhw-dhw0)/dt - bathymwater(M)%area_half*lamsal(M)/ &
      & (ddz(M)*h1*bathymwater(M+1)%area_int) )
      y = rosdry(1)*(1 - por(1))/row0
      z = 0.25*wsoil(1)
      a(M+1) = x
      b(M+1) = z*y
      c(M+1) = - ( 0.5*(dzs(1) + ddz(M)*h1)/dt + z - x ) 
      d(M+1) = - Sal1(M+1)*0.5*(dzs(1) + ddz(M)*h1)/dt - &
      & x*(Sal1(M+1) - Sal1(M)) + z*Sal1(1) + z*y*Sals1(2,nsoilcols)
      call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,M+2,M+ns-1,soil_salinity_indic,dt)
      x = 0.5*wsoil(ns-1)*dt/(dzs(ns-1))
      c(M+ns) = - 1.d0 + x
      a(M+ns) = - x
      d(M+ns) = - Sals1(ns,nsoilcols) + 2*dt*Sflux_soil_bot/dzs(ns-1) - &
      & x*(Sals1(ns,nsoilcols) + Sals1(ns-1,nsoilcols))
      !print*, CHECK_PROGONKA(M+ns,a,b,c,d,Sal)
      !print*, 'a', a(1:M+ns)
      !read*
      !print*, 'b', b(1:M+ns)
      !read*
      !print*, 'c', c(1:M+ns)
      !read*
      !print*, 'd', d(1:M+ns)
      !read*
      call PROGONKA (nveclen,a,b,c,d,Sal,1,M+ns)
      Sal(1:M+ns) = max(Sal(1:M+ns),0._ireals)
      Sals2(2:ns,nsoilcols) = Sal(M+2:M+ns)
      Sals2(1,nsoilcols) = Sal(M+1)/y !mind y!
      Sal2(1:M+1) = Sal(1:M+1)
    else ! Zero salinity flux at the bottom
      x = 0.5*( - bathymwater(M)%area_half*lamsal(M)/(ddz(M)*h1* &
      & bathymwater(M+1)%area_int) + (dhw-dhw0)/(2.d0*dt) )
      c(M+1) = x - ddz(M)*h1/(2.d0*dt) 
      a(M+1) = x
      d(M+1) = - Sal1(M+1)*ddz(M)*h1/(2.d0*dt) + Sflux1 - x*(Sal1(M+1) - Sal1(M))
      call PROGONKA (nveclen,a,b,c,d,Sal,1,M+1)
      Sal (1:M+1) = max(Sal(1:M+1),0._ireals)
      Sal2(1:M+1) = Sal(1:M+1)
    endif
  endif
  !if (sedim%par == 1) call SAL_SEDIM(ddz,h1,dt,Sal2)
else
! Case of the soil and the ice above     
  if (soilswitch%par == 1 .and. salsoil%par == 1) then
    call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,2,ns-1,soil_salinity_indic,dt)
    x = 0.5*dt*wsoil(1)/dzs(1)
    c(1) = - 1.d0 - x
    b(1) = x
    d(1) = - Sals1(1,nsoilcols) + x*(Sals1(1,nsoilcols) + Sals1(2,nsoilcols))
    x = 0.5*wsoil(ns-1)*dt/(dzs(ns-1))
    c(ns) = - 1.d0 + x
    a(ns) = - x
    d(ns) = - Sals1(ns,nsoilcols) + 2.d0*dt*Sflux_soil_bot/dzs(ns-1) - &
    & x*(Sals1(ns,nsoilcols) + Sals1(ns-1,nsoilcols))
    !print*, CHECK_PROGONKA(ns,a,b,c,d,Sal)
    call PROGONKA (nveclen,a,b,c,d,Sal,1,ns)
    Sal(1:ns) = max(Sal(1:ns),0._ireals)
    Sals2(1:ns,nsoilcols) = Sal(1:ns)
  else
    Sals2(:,nsoilcols) = Sals1(:,nsoilcols)
  endif
endif 

deallocate(a,b,c,d,Sal)

!print*, VARMEAN(Sal2,bathymwater,1_iintegers)

!print*, 'Sals2', Sals2(:,nsoilcols)
!read*
!print*, 'Sals1', Sals1
!read*
!print*, 'Sal2', Sal2
!read*
!print*, 'Sal1', Sal1
!read*

END SUBROUTINE S_DIFF


!SUBROUTINE SAL_SEDIM(ddz,h1,dt,Sal)
!
!! The subroutine SAL_SEDIM updates the salinity profile
!! due to gravitational sedimentation
!
!use DRIVING_PARAMS, only : M
!use PHYS_FUNC!, only: &
!!& W_SEDIM
!
!implicit none
!
!! Input variables
!real(kind=ireals), intent(in) :: ddz  (M) ! Spacing of dzeta-coordinate grid
!      
!real(kind=ireals), intent(in) :: h1 ! Lake depth, m
!real(kind=ireals), intent(in) :: dt ! Timestep,   sec
!
!! Input/output variables
!real(kind=ireals), intent(inout) :: Sal (M+1) ! Salinity at main levels, kg/kg
!
!! Local variables
!real(kind=ireals) :: a(M+1) 
!real(kind=ireals) :: b(M+1) 
!real(kind=ireals) :: c(M+1) 
!real(kind=ireals) :: d(M+1) 
!
!real(kind=ireals) :: w_sediment (M) !Speed of gravitational sedimentation, m/s
!
!real(kind=ireals) x ! Help variable
!
!integer i ! loop index
!
!logical indstab
!logical ind_bound
!
!! The speed of gravitational sedimentation, positive downwards
!do i = 1, M
!  w_sediment(i) = W_SEDIM()
!enddo
!
!! Boundary conditions at the top boundary (dzeta = 0)
!x = 0.5d0*ddz(1)*h1/dt
!c(1) = - 0.5d0*w_sediment(1) - x
!b(1) =   0.5d0*w_sediment(1)
!d(1) = - Sal(1)*x
!
!! Boundary conditions at the bottom boundary (dzeta = 1)
!x = 0.5d0*ddz(M)*h1/dt
!a(M+1) = - 0.5d0*w_sediment(M)
!c(M+1) =   0.5d0*w_sediment(M) - x
!d(M+1) = - Sal(M+1)*x
!
!! The coefficients of tridiagonal matrix
!do i = 2, M
!  x = 0.5d0*(ddz(i-1) + ddz(i))*h1/dt
!  a(i) = - 0.5d0*w_sediment(i-1)
!  c(i) =   0.5d0*w_sediment(i-1) - 0.5d0*w_sediment(i) - x
!  b(i) =   0.5d0*w_sediment(i)
!  d(i) = - Sal(i)*x
!enddo
!  
!ind_bound = .true.
!call IND_STAB_FACT_DB(a,b,c,1,M+1,indstab,ind_bound)
!call PROGONKA(a,b,c,d,Sal,1,M+1)
!if (.not.indstab) then
!  print*,'Info: Unstable factorization method in SAL_SEDIM'
!  print*,'The accuracy flag is', CHECK_PROGONKA(M+1,a,b,c,d,Sal)
!  if (.not.CHECK_PROGONKA(M+1,a,b,c,d,Sal) ) STOP
!endif
!
!RETURN
!END SUBROUTINE SAL_SEDIM


!> Subroutine updates ice bulk salinity to the next timestep
SUBROUTINE UPDATE_ICE_SALINITY(dt,ls,wst)

use PHYS_CONSTANTS, only : row0_d_roi, poricebot, &
& row0, ci_m_roi, row0_m_Lwi, asalice, roi
use ARRAYS, only : ice_sal_indic
use DRIVING_PARAMS, only : Mice, nmeltpoint
use T_SOLVER_MOD, only : DIFF_COEF
use PHYS_FUNC, only : MELTPNT, SALICEBOT 
use ATMOS, only : pressure
use ARRAYS_GRID, only : ddzi

implicit none

! Input variables
real(kind=ireals), intent(in) :: dt !> Time step, s
type(layers_type), intent(in) :: ls !> Data structure with sizes of physical layers

! Input/output variables
type(waterstate_type), intent(inout) :: wst !> Data structure with 1D profiles

!Local variables
integer(kind=iintegers), parameter :: nitermax = 100
real   (kind=ireals)   , parameter :: resid_max = 1.e-2
real   (kind=ireals)   , parameter :: sal_min = 1.e-2
real   (kind=ireals)   , parameter :: denssal_max = 300. !maximal salt concentration in pores, promille
real(kind=ireals) :: work1, work2, work3, del, resid1, resid2, resid3
real   (kind=ireals)   , allocatable :: salprev(:)
integer(kind=iintegers) :: i, j !Loop indices
integer(kind=iintegers) :: iter

!real(kind=ireals), dimension(1:nveclen):: a,b,c,d
real(kind=ireals), allocatable :: a(:), b(:), c(:), d(:)
      
allocate(a(1:nveclen), b(1:nveclen), &
&        c(1:nveclen), d(1:nveclen))
a(:) = 0.; b(:) = 0.; c(:) = 0.; d(:) = 0

!dh = ls%dhihigh - row0_d_roi*(ls%snmeltice + ls%dhiimp) ! Melting of ice at the top, negative

!salice = salice - ls%dhi/(ls%l1)*salice + &
!&      (dh)/(ls%l1) * ( salice ) + &
!!&      row0_d_roi*ls%snmeltice/(ls%l1) * ( 0. ) + &
!&      row0_d_roi*ls%dhiimp   /(ls%l1) * ( Sal1(1) ) + &
!&      (ls%dhilow)/ (ls%l1)* &
!&      ( 0.5*(1. + sign(1._ireals,ls%dhilow ) )*( Sal1(1) )   + &
!&        0.5*(1. - sign(1._ireals,ls%dhilow ) )*( salice  ) ) + &
!&      (ls%dhis)/(ls%l1) * ( salice ) !+ &
!!&      (ls%dhip)/(ls%l1) * ( 0. )

allocate(salprev(1:Mice+1))
forall(i=1:Mice+1) salprev(i) = wst%salice(i)


call DIFF_COEF(nveclen,a,b,c,d,2,Mice,2,Mice,ice_sal_indic,dt)
!1. Zero ice salinity at top of ice
!c(1) = 1.
!b(1) = 0.
!d(1) = 0.
!2. Salinity equation discretized at the top boundary
b(1) =       - 0.5*ls%dhw0/(ls%h1*dt*ddzi(1))
c(1) = 1./dt + b(1) !+ asalice*0.5
d(1) = wst%salice(1)*(1./dt - asalice) - & !asalice*0.5
&              0.5*ls%dhw0/(ls%h1*dt*ddzi(1))*(wst%salice(2) - wst%salice(1)) 
! Bottom boundary condition: salinity in ice pores coincides with that in water
if (ls%h1 > 0.) then
  ! Water-ice interface
  a(Mice+1) = 0.
  c(Mice+1) = 1.
  !d(Mice+1) = wst%Sal1(1)*row0*poricebot
  d(Mice+1) = max(SALICEBOT((ls%dhw - ls%dhw0)/dt, wst%Sal1(1),wst%porice(Mice+1)), &
  &               wst%Sal1(1)*row0*poricebot) !salinity dependence on the ice growth
else
  ! Ice-soil interface: zero salinity gradient
  a(Mice+1) = 1.
  c(Mice+1) = 1.
  d(Mice+1) = 0.
endif
call PROGONKA (nveclen,a,b,c,d,wst%salice,1,Mice+1)
forall(i=1:Mice) wst%salice(i) = max(wst%salice(i), sal_min)

! Brine salt concentration at the ice bottom equals to the water salinity below
wst%porice(Mice+1) = wst%salice(Mice+1)/(wst%Sal1(1)*row0+small_value)

! Rescaling of water-filled porosity, after update of bulk ice salinity.
! In-pore salt concentration is assumed constant under applying metric terms
! and flux of alloy into water layer through cracks
do i = 1, Mice
  wst%porice(i) = wst%porice(i)*wst%salice(i)/salprev(i)
enddo

! Adjustment of ice temperature and salty water content
do j = 2, Mice

  ifadjust : if (wst%Ti2(j) < MELTPNT(wst%salice(j)/(wst%porice(j)*row0), &
  &                        pressure,nmeltpoint%par) ) then

    ! Salty alloy in ice may only freeze, no melting of pure ice into alloy is allowed

    !work = wst%Ti2(j)
    !cyc : do i = 1, nitermax 
    !  work2 = (wst%porice(j) - (work - wst%Ti2(j)) * &
    !  & wst%ci_m_roi_v(j)/row0_m_Lwi)**(-1.) * wst%salice(j) !, small_value)
    !  print*, '1', work2, wst%porice(j), wst%salice(j)
    !  work3 = MELTPNT(work2/row0,pressure,nmeltpoint%par) !Assuming atmospheric pressure in ice
    !  del = abs(work3-work)
    !  if (del > 1.e-4) then !convergence criterium on temperature
    !    !print*, i,j,del, work3, work, work2
    !    !read*
    !    work = 0.2*work3 + 0.8*work
    !  else
    !    !print*, i,j,del, work3, work, work2
    !    !read*
    !    wst%Ti2(j) = work3
    !    wst%porice(j) = wst%salice(j)/work2
    !    exit cyc
    !  endif
    !  print*, del
    !enddo cyc

    !work1 = 0.5*salprev(j)/wst%porice(j)
    work1 = 0.5*wst%salice(j)/wst%porice(j)
    resid1 = RESID(work1)
    !print*, 'resid1', resid1
    work2 = work1*2.
    resid2 = RESID(work2)
    do while ( resid1 * resid2 > 0._ireals)
      !print*, 'resid2', resid2
      work2 = work2 * 2.
      resid2 = RESID(work2)
      !print*, 'Tfr2=', MELTPNT(work2/(row0),pressure,nmeltpoint%par)
      if (work2/row0 > denssal_max) then
        print*, 'No adjustment made for ice salt:', &
        & 'sal1=',work1, 'sal2=',work2, 'res1=',resid1, 'res2=',resid2
        print*, 'salice=', & !wst%salice(j), ', porosity=', wst%porice(j), &
        & ', in-pore salinity: ', wst%salice(j)/(wst%porice(j)*row0)*1000., '(0\00)', &
        & 'Ti=', wst%Ti2(j), ', Tfr1=', MELTPNT(wst%salice(j)/(wst%porice(j)*row0), &
        &                              pressure,nmeltpoint%par), &
        &                    ', Tfr2=', MELTPNT(work2/(row0), &
        &                              pressure,nmeltpoint%par), &
        &                    j
        read*
        !exit ifadjust
      endif
    enddo
    work3  = 0.5*(work1 + work2)
    resid3 =  RESID(work3)
    resid1 = resid3+1.
    resid2 = resid3+1.
    !CHORDE METHOD TO FIND IN-PORE SALINITY
    iter = 0
    if (resid1 /= 0._ireals .and. resid2 /= 0._ireals) then
      cy1:do while (abs(resid3) > resid_max)
        iter = iter + 1
        if (resid1 /= resid3) then
          resid1 = RESID(work1)
        endif
        if (resid2 /= resid3) then
          resid2 = RESID(work2)
        endif
        work3 = (work1*resid2 - work2*resid1)/(resid2 - resid1)
        resid3 = RESID(work3)
        if (resid1 * resid3 < 0.) then
          work2 = work3
          resid2 = resid3
        elseif (resid2 * resid3 < 0.) then
          work1 = work3
          resid1 = resid3
        endif
        if (iter > nitermax) then
          !write (n_unit,*) resid3
          print*, 'iterations exceeded: ', work1, work2, work3, resid1, resid2, resid3
          read*
          exit cy1
        endif
      enddo cy1
    elseif (resid1 == 0._ireals) then
      work3 = work1
    elseif (resid2 == 0._ireals) then
      work3 = work2
    endif

    wst%porice(j) = wst%salice(j)/work3
    wst%Ti2   (j) = MELTPNT(wst%salice(j)/(wst%porice(j)*row0), &
    &                       pressure,nmeltpoint%par)
    print*, 'resid=',resid3,' niter=', iter, ' por=', wst%porice(j), ' Ti=',wst%Ti2(j)
  else
    !print*, 'No adjustment needed: salice=', & !wst%salice(j), ', porosity=', wst%porice(j), &
    !& ', in-pore salinity: ', wst%salice(j)/(wst%porice(j)*row0)*1000., '(0\00)', &
    !& 'Ti=', wst%Ti2(j), ', Tfr=', MELTPNT(wst%salice(j)/(wst%porice(j)*row0), &
    !&                        pressure,nmeltpoint%par), j
  endif ifadjust
enddo

deallocate(a, b, c, d)
deallocate(salprev)

contains
!> Residual of equation for in-pore salinity at the next timestep
FUNCTION RESID(saldens)
implicit none
real(kind=ireals), intent(in) :: saldens
real(kind=ireals) :: RESID
!print*, saldens
RESID = wst%porice(j) - (MELTPNT(saldens/row0,pressure,nmeltpoint%par) - wst%Ti2(j)) * &
& wst%ci_m_roi_v(j)/row0_m_Lwi - wst%salice(j)/(saldens + small_value)
END FUNCTION RESID

END SUBROUTINE UPDATE_ICE_SALINITY


END MODULE SALINITY_LAKE_MOD
