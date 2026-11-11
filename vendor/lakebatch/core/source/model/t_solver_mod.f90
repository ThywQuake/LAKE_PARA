MODULE T_SOLVER_MOD

use NUMERICS, only : PROGONKA

use LAKE_DATATYPES, only : ireals, iintegers

contains

!-----------------------------------------------------------------------------------
!TEMPERATURE EQUATION SOLVER
!-----------------------------------------------------------------------------------

!>T_SOLVER implements iterations to find lake surface temperature 
!!at the next time step
SUBROUTINE T_SOLVER(ix,iy,nx,ny,year,month,day,hour,phi, &
& RadWater, RadIce, SurfTemp, fetch, dt)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

use DRIVING_PARAMS
use ARRAYS
use ARRAYS_BATHYM
use ARRAYS_WATERSTATE
use ARRAYS_GRID
use INOUT_PARAMETERS, only : &
& lake_subr_unit_min, &
& lake_subr_unit_max
use INOUT, only : CHECK_UNIT
use RADIATION, only : rad_type
use HEATBALANCE_MOD, only : &
& HEATBALANCE, &
& TEMPLINHEATBALEQ

use LB_SAVED_LOCALS, only : Tsurf_prev => tsolver_Tsurf_prev, surftyp_prev => tsolver_surftyp_prev, &
 & total_Bal => tsolver_total_Bal ! lakebatch restart
implicit none

!Input variables
integer(kind=iintegers), intent(in) :: ix, iy !< lon-lat coordinate indices of the lake
integer(kind=iintegers), intent(in) :: nx, ny !< dimensions of lon-lat grid

integer(kind=iintegers), intent(in) :: year, month, day !< year, month and day

real(kind=ireals), intent(in) :: hour !< hour
real(kind=ireals), intent(in) :: phi  !< latitude, degrees
real(kind=ireals), intent(in) :: fetch
real(kind=ireals), intent(in) :: dt
real(kind=ireals), intent(in) :: SurfTemp !> Surface temperature, Kelvin (prescribed, used as Dirichlet b.c.)

type(rad_type), intent(in) :: RadWater !< radiation data structure for water layer
type(rad_type), intent(in) :: RadIce   !< radiation data structure for ice layer


!Local variables

integer(kind=iintegers), parameter :: Tbc_net_heat_flux   = 1 !b.c. for temperature: heat balance equation
integer(kind=iintegers), parameter :: Tbc_Dirichlet_const_cuette = 2 !b.c. for temperature: Dirichlet constant (Cuette flow)
integer(kind=iintegers), parameter :: Tbc_Dirichlet_var   = 3 !b.c. for temperature: Dirichlet as prescribed time series


! Tsurf_prev, surftyp_prev, total_Bal now live in LB_SAVED_LOCALS (lakebatch restart)
integer(kind=iintegers), allocatable :: surftyp(:,:)

real(kind=ireals) :: dt_scan
real(kind=ireals) :: Tsurf1, Tsurf2, Tsurf3
real(kind=ireals) :: Bal1, Bal2, Bal3
real(kind=ireals) :: resid_Bal_max = 1. !1.d-2

integer(kind=iintegers) :: iter
integer(kind=iintegers) :: maxiter, Tbc_switch
integer(kind=iintegers) :: n_unit = lake_subr_unit_min

logical :: firstcall = .true.
logical :: iterate

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_T_SOLVER)
include 'lb_ctx_t_solver.inc'

if (firstcall) then
  n_unit = lbc_open_unit(path(1:len_trim(path))//'results/debug/iter_T.dat')
  if (n_unit < 0) then ! not yet opened by another point of this process
    n_unit = lake_subr_unit_min ! CHECK_UNIT searches from here, as on a fresh start
    call CHECK_UNIT(lake_subr_unit_min,lake_subr_unit_max,n_unit)
    open (n_unit,file=path(1:len_trim(path))// &
    & 'results/debug/iter_T.dat')
  endif
  maxiter = 20
  dt_scan = 5.

  if (.not. allocated(Tsurf_prev)) then ! may already be restored (lakebatch restart)
    allocate (Tsurf_prev(nx,ny))
    allocate (surftyp_prev(nx,ny))
  endif
  ! (nx,ny): shared by all points; each point runs this first-call block
  if (.not. allocated(surftyp)) allocate (surftyp(nx,ny))

  Tbc_switch = Tbc_net_heat_flux
  !Tbc_switch = Tbc_Dirichlet_var
  if (cuette%par == 1 .or. cuette%par == 11) Tbc_switch = Tbc_Dirichlet_const_cuette

endif

!Setting radiation, aerodynamic and thermophysical parameters of the lake surface
if     (snow  == 1) then
  surftyp(ix,iy) = snow_indic
elseif (ice   == 1) then
  surftyp(ix,iy) = ice_indic
elseif (water == 1) then
  surftyp(ix,iy) = water_indic
else
  surftyp(ix,iy) = soil_indic
endif
call SURF_CHAR(surftyp(ix,iy),year,month,day,hour,phi,fetch)

Tbc_if : if (Tbc_switch == Tbc_net_heat_flux) then

  ! Heat balance equation as b.c.s for temperature
    
    if (nstep > 1 .and. surftyp(ix,iy) == surftyp_prev(ix,iy)) then
      Tsurf2 = Tsurf_prev(ix,iy) - 10.
    else
      Tsurf2 = -90.
    endif
  
    !SCANNING INTERVAL OF SURFACE TEMPERATURE [-90 C, ...]
    
    iterate = .true.
    Bal1 = 1.; Bal2 = 1.
    do while (Bal1 * Bal2 > 0 .and. iterate)
      Tsurf1 = Tsurf2
      Tsurf2 = Tsurf2 + dt_scan
      call T_DIFF(1_iintegers,Tsurf1,dt,snow,ice,water,deepice,nsoilcols,.false.)
      Bal1 = HEATBALANCE(Tsurf1,surftyp(ix,iy),RadWater,RadIce,fetch,dt) 
      call T_DIFF(1_iintegers,Tsurf2,dt,snow,ice,water,deepice,nsoilcols,.false.)
      Bal2 = HEATBALANCE(Tsurf2,surftyp(ix,iy),RadWater,RadIce,fetch,dt)
      if (Tsurf2 > 80.) then
        !print*, 'Severe: iterations for the surface temperature &
        !&do not converge: at the point', ix, iy, 'STOP', 'nstep = ', nstep, &
        !& 'h1 = ', h1, 'l1 = ', l1, 'hs1 = ', hs1, 'lamw = ', lamw
        !print*, 'Temperature limit in scan process is exceeded' !: STOP'
        ! If iterations do not converge, solve linearized heat balance equation
        Tsurf3 = TEMPLINHEATBALEQ(albedo,albedo_lw,emissivity,Tsurf_prev(ix,iy),relhums,surftyp(ix,iy))
        iterate = .false.
        write(n_unit,*) year,month,day,hour,Tsurf3
      endif
    enddo
    
    iter = 0
    Bal3 = resid_Bal_max + 1.
    
    !CHORDE METHOD TO FIND SURFACE TEMPERATURE
    if (Bal1 /= 0.d0 .and. Bal2 /= 0.d0 .and. iterate) then
      cy1:do while (abs(Bal3) > resid_Bal_max)
        iter = iter + 1
        if (Bal1 /= Bal3) then
          call T_DIFF(1_iintegers,Tsurf1,dt,snow,ice,water,deepice,nsoilcols,.false.)
          Bal1 = HEATBALANCE(Tsurf1,surftyp(ix,iy),RadWater,RadIce,fetch,dt) 
        endif
        if (Bal2 /= Bal3) then
          call T_DIFF(1_iintegers,Tsurf2,dt,snow,ice,water,deepice,nsoilcols,.false.)
          Bal2 = HEATBALANCE(Tsurf2,surftyp(ix,iy),RadWater,RadIce,fetch,dt) 
        endif
        Tsurf3 = (Tsurf1*Bal2 - Tsurf2*Bal1)/(Bal2 - Bal1)
        call T_DIFF(1_iintegers,Tsurf3,dt,snow,ice,water,deepice,nsoilcols,.false.)
        Bal3 = HEATBALANCE(Tsurf3,surftyp(ix,iy),RadWater,RadIce,fetch,dt)
        if (Bal1 * Bal3 < 0.) then
          Tsurf2 = Tsurf3
          Bal2 = Bal3
        elseif (Bal2 * Bal3 < 0.) then
          Tsurf1 = Tsurf3
          Bal1 = Bal3
        endif
        if (iter > maxiter) then
          write (n_unit,*) Bal3
          exit cy1
        endif
      enddo cy1
    elseif (Bal1 == 0.d0) then
      Tsurf3 = Tsurf1
    elseif (Bal2 == 0.d0) then
      Tsurf3 = Tsurf2
    endif
    
    call T_DIFF(0_iintegers,Tsurf3,dt,snow,ice,water,deepice,nsoilcols,.false.)
    
    Tsurf_prev(ix,iy) = Tsurf3
    surftyp_prev(ix,iy) = surftyp(ix,iy)
    
    total_Bal = total_Bal + Bal3

elseif (Tbc_switch == Tbc_Dirichlet_var .and. SurfTemp /= missing_value) then

  !Using prescribed surface temperature as Dirichlet boundary condition
  Tsurf3 = SurfTemp
  call T_DIFF(1_iintegers,Tsurf3,dt,snow,ice,water,deepice,nsoilcols,.false.)
  Bal3 = HEATBALANCE(Tsurf3,surftyp(ix,iy),RadWater,RadIce,fetch,dt)
  call T_DIFF(0_iintegers,Tsurf3,dt,snow,ice,water,deepice,nsoilcols,.false.)

elseif (Tbc_switch == Tbc_Dirichlet_const_cuette) then

  ! B.c.s for temperature for Cuette flow
  Tsurf3 = Tw1(1)
  call T_DIFF(1_iintegers,Tsurf3,dt,snow,ice,water,deepice,nsoilcols,.false.)
  Bal3 = HEATBALANCE(Tsurf3,surftyp(ix,iy),RadWater,RadIce,fetch,dt)

endif Tbc_if

if (Tsurf3 /= Tsurf3) print*, 'T_solver: Tsurf3 = NaN!'
if (abs(Tsurf3) > 100.) then
  print*, 'T_solver: Tsurf3 = ', Tsurf3,snow,ice,water,deepice
endif

if (firstcall) firstcall = .false.
END SUBROUTINE T_SOLVER


SUBROUTINE T_DIFF(surf,Tsurf,dt,snow,ice,water,deepice,isoilcol,ifflux)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

!T_DIFF calculates temperature profile in water, soil, ice and snow,
!with specified temparature at the surface 

use NUMERIC_PARAMS
use PHYS_CONSTANTS
use DRIVING_PARAMS
use ARRAYS, &
& snow0  => snow,  ice0     => ice, &
& water0 => water, deepice0 => deepice
use ARRAYS_WATERSTATE
use ARRAYS_BATHYM
use ARRAYS_SOIL
use ARRAYS_GRID
use ARRAYS_TURB
use PHYS_FUNC, only: &
& MELTPNT 
use BL_MOD_LAKE, only : snmelt
use SNOWSOIL_MOD, only : T, Tsn, cs, dens, dz, itop, lams
use RADIATION, only : RadWater, RadIce, RadDeepIce

implicit none

!Input variables
real(kind=ireals)   , intent(in) :: dt, Tsurf
integer(kind=iintegers), intent(in) :: surf
integer(kind=iintegers), intent(in) :: snow, ice, water, deepice
integer(kind=iintegers), intent(in) :: isoilcol

logical, intent(in) :: ifflux

! Local variables
!real(kind=ireals), dimension(1:nveclen) :: a,b,c,d,Temp
real(kind=ireals), allocatable :: a(:), b(:), c(:), d(:), Temp(:)
real(kind=ireals) :: cc, xx
integer(kind=iintegers) :: i,j

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_T_DIFF)
include 'lb_ctx_t_diff.inc'

allocate(a(1:nveclen), b(1:nveclen), c(1:nveclen), &
&        d(1:nveclen), Temp(1:nveclen))

!Meltpnt - Melting point temperature, C degrees 
!Meltpnt=0.
!ddz=1./float(M)VS,06.2007

if (surf == 1) then
  if (snow == 1) then ! Snow exists
    c(itop) = 1.
    b(itop) = 0.
    d(itop) = Tsurf
    if (itop <= ms-2) then
      call DIFF_COEF(nveclen,a,b,c,d,itop+1,ms-1,itop+1,ms-1,snow_indic,dt) 
    endif
    if (ice == 1) then ! Snow and ice exist
!----------------SNOW-ICE INTERFACE
      a(ms) = - 0.25*(lams(ms-1) + lams(ms))/dz(ms-1)
      b(ms) = 0.5*( - lami_v(1)/(ddzi(1)*l1) + 0.5*ci_m_roi_v(1)*dhi0*dt_inv)
      c(ms) = a(ms) + b(ms) - 0.5*dt_inv*(cs(ms)*dens(ms)*dz(ms) + &
      & ci_m_roi_v(1)*ddzi(1)*l1)
      d(ms) = - Ti1(1)*0.5*dt_inv*(cs(ms)*dens(ms)*dz(ms) + &
      & ci_m_roi_v(1)*ddzi(1)*l1) - SR_botsnow + RadIce%integr(1) + &
      & a(ms)*T(ms-1) - (a(ms) + b(ms))*Ti1(1) + b(ms)*Ti1(2) - &
      & (snmeltice + dhiimp)*Lwi*row0*dt_inv
!-----------------------------------
      call DIFF_COEF(nveclen,a,b,c,d,2,Mice,ms+1,ms+Mice-1,ice_indic,dt) 
      if (water == 1) then ! Snow, ice and water exist
        a(ms+Mice) = 0.
        c(ms+Mice) = 1.
        d(ms+Mice) = Meltpnt(Sal2(1),preswat(1),nmeltpoint%par)
        call PROGONKA (nveclen,a,b,c,d,Temp,itop,ms+Mice)
        do i = itop, ms
          Tsn(i) = Temp(i)
        enddo
        do i = ms, ms+Mice
          Ti2(i-ms+1) = Temp(i)
        enddo 
      else
        if (soilswitch%par == 1) then ! No water, only snow, ice and soil
!----------------ICE-SOIL INTERFACE-------------------------
          xx = bathymice(Mice+1)%area_int/bathymice(Mice)%area_half
          a(ms+Mice) = 0.5*( - & !bathymice(Mice+1)%area_int/bathymice(Mice)%area_int * &
          &  lami_v(Mice)/(ddzi(Mice)*l1) + xx*0.5*ci_m_roi_v(Mice+1)*(dhi-dhi0)*dt_inv)
          b(ms+Mice) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
          cc = 0.5*dt_inv*(csoil(1)*rosoil(1)*dzs(1) + xx*ci_m_roi_v(Mice+1)*ddzi(Mice)*l1)
          c(ms+Mice) = a(ms+Mice) + b(ms+Mice) - cc
          d(ms+Mice) = - Ti1(Mice+1)*cc - RadIce%integr(Mice) + &
          & a(ms+Mice)*Ti1(Mice) - (a(ms+Mice) + b(ms+Mice))*Ti1(Mice+1) + &
          & b(ms+Mice)*Tsoil1(2,isoilcol)
!----------------------------------------------------------- 
          call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,ms+Mice+1,ms+Mice+ns-2,soil_indic,dt,isoilcol)
          c(ms+Mice+ns-1) = 1.
          a(ms+Mice+ns-1) = 1.
          d(ms+Mice+ns-1) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1) + lamsoil(ns))
          call PROGONKA (nveclen,a,b,c,d,Temp,itop,ms+Mice+ns-1)
          do i = ms+Mice, ms+Mice+ns-1
            Tsoil2(i-ms-Mice+1) = Temp(i)
          enddo
        elseif (soilswitch%par == 0) then ! No water and soil, only snow and ice exist
          xx = bathymice(Mice)%area_half/bathymice(Mice+1)%area_int
          c(ms+Mice) = - xx*0.5*lami_v(Mice)/(ddzi(Mice)*l1) - &
          & 0.5*ci_m_roi_v(Mice+1)*ddzi(Mice)*l1*dt_inv + &
          & 0.25*ci_m_roi*(dhi-dhi0)*dt_inv
          a(ms+Mice) = 0.5*( - xx*lami_v(Mice)/(ddzi(Mice)*l1) + &
          & 0.5*ci_m_roi_v(Mice+1)*(dhi-dhi0)*dt_inv)
          d(ms+Mice) = - 0.5*Ti1(Mice+1)*ci_m_roi_v(Mice+1)*ddzi(Mice)*l1*dt_inv - &
          & xx*RadIce%integr(Mice) + soilbotflx%par + &  
          ! soilbotflx is interpreted as top soil boundary flux in this case
          & (xx*0.5*lami_v(Mice)/(ddzi(Mice)*l1) - &
          & 0.25*ci_m_roi_v(Mice+1)*(dhi-dhi0)*dt_inv)*Ti1(Mice+1) + a(ms+Mice)*Ti1(Mice)
          call PROGONKA (nveclen,a,b,c,d,Temp,itop,ms+Mice)
          Tsoil2(:) = Tsoil1(:,isoilcol)
        endif
        do i = itop, ms
          Tsn(i) = Temp(i)
        enddo
        do i = ms, ms+Mice
          Ti2(i-ms+1) = Temp(i)
        enddo 
      endif
    else ! There is no ice, but snow extists
      if (soilswitch%par == 1) then ! Only snow and soil exist
!-------------------SNOW-SOIL INTERFACE-------------------------
        a(ms) = - 0.25*(lams(ms-1) + lams(ms))/dz(ms-1)
        b(ms) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
        c(ms) = a(ms) + b(ms) - 0.5*dt_inv*(cs(ms)*dens(ms)*dz(ms) + &
        & csoil(1)*rosoil(1)*dzs(1))
        d(ms) = - Tsoil1(1,isoilcol)*0.5*dt_inv*(cs(ms)*dens(ms)*dz(ms) + &
        & csoil(1)*rosoil(1)*dzs(1)) - SR_botsnow + &
        & a(ms)*T(ms-1) - (a(ms) + b(ms))*Tsoil1(1,isoilcol) + b(ms)*Tsoil1(2,isoilcol)
!---------------------------------------------------------------
        call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,ms+1,ms+ns-2,soil_indic,dt,isoilcol) 
        c(ms+ns-1) = 1.
        a(ms+ns-1) = 1.
        d(ms+ns-1) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1)+lamsoil(ns))
        call PROGONKA (nveclen,a,b,c,d,Temp,itop,ms+ns-1)
        do i = ms, ms+ns-1
          Tsoil2(i-ms+1) = Temp(i)
        enddo
      else if (soilswitch%par == 0) then ! Only ice exists
        a(ms) = - 0.25*(lams(ms-1) + lams(ms))/dz(ms-1)
        c(ms) = a(ms) - 0.5*cs(ms)*dens(ms)*dz(ms)*dt_inv
        d(ms) = - T(ms)*0.5*cs(ms)*dens(ms)*dz(ms)*dt_inv - &
        & SR_botsnow + soilbotflx%par + & ! soilbotflx is interpreted as top soil boundary flux in this case
        & a(ms)*(T(ms-1) - T(ms))
        call PROGONKA (nveclen,a,b,c,d,Temp,itop,ms)
        Tsoil2(:) = Tsoil1(:,isoilcol)
      endif
      do i = itop, ms
        Tsn(i) = Temp(i)
      enddo
    endif
  elseif (ice == 1) then ! There is no snow, but ice exists
    b(1) = 0.
    c(1) = 1.
    d(1) = Tsurf
    call DIFF_COEF(nveclen,a,b,c,d,2,Mice,2,Mice,ice_indic,dt)  
    if (water == 1) then ! Ice and water exist
      a(Mice+1) = 0.
      c(Mice+1) = 1.
      d(Mice+1) = Meltpnt(Sal2(1),preswat(1),nmeltpoint%par)
      call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+1)
      do i = 1, Mice+1
        Ti2(i) = Temp(i)
      enddo
    else ! Ice exists, but not water
      if (soilswitch%par == 1) then ! Only ice and soil exist
!----------------ICE-SOIL INTERFACE-------------------------
        xx = bathymice(Mice+1)%area_int/bathymice(Mice)%area_half
        a(Mice+1) = 0.5*( - & !bathymice(Mice+1)%area_int/bathymice(Mice)%area_int * &
        & lami_v(Mice)/(ddzi(Mice)*l1) + xx*0.5*ci_m_roi_v(Mice+1)*(dhi-dhi0)*dt_inv)
        b(Mice+1) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
        cc = 0.5*dt_inv*(csoil(1)*rosoil(1)*dzs(1) + xx*ci_m_roi_v(Mice+1)*ddzi(Mice)*l1)
        c(Mice+1) = a(Mice+1) + b(Mice+1) - cc 
        d(Mice+1) = - Ti1(Mice+1)*cc - RadIce%integr(Mice) + &
        & a(Mice+1)*Ti1(Mice) - (a(Mice+1) + b(Mice+1))*Ti1(Mice+1) + &
        & b(Mice+1)*Tsoil1(2,isoilcol)
!-----------------------------------------------------------
        call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,Mice+2,Mice+ns-1,soil_indic,dt,isoilcol) 
        c(Mice+ns) = 1.
        a(Mice+ns) = 1.
        d(Mice+ns) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1)+lamsoil(ns))
        call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+ns)
        do i = Mice+1, Mice+ns
          Tsoil2(i-Mice) = Temp(i)
        enddo
      elseif (soilswitch%par == 0) then  ! Only ice exists
        xx = bathymice(Mice)%area_half/bathymice(Mice+1)%area_int
        c(Mice+1) = - xx*0.5*lami_v(Mice)/(ddzi(Mice)*l1) + 0.5*dt_inv*ci_m_roi_v(Mice+1)* &
        & ( - ddzi(Mice)*l1 + 0.5*(dhi - dhi0) ) 
        a(Mice+1) = 0.5*( - xx*lami_v(Mice)/(ddzi(Mice)*l1) + &
        & 0.5*dt_inv*ci_m_roi_v(Mice+1)*(dhi-dhi0))
        d(Mice+1) = - Ti1(Mice+1)*0.5*dt_inv*ci_m_roi_v(Mice+1)*ddzi(Mice)*l1 - &
        & xx*RadIce%integr(Mice) + soilbotflx%par + & 
        ! soilbotflx is interpreted as top soil boundary flux in this case
        & (xx*0.5*lami_v(Mice)/(ddzi(Mice)*l1) - &
        & 0.25*dt_inv*ci_m_roi_v(Mice+1)*(dhi - dhi0))*Ti1(Mice+1) + &
        & a(Mice+1)*Ti1(Mice)
        call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+1)
        Tsoil2(:) = Tsoil1(:,isoilcol)
      endif
      do i = 1, Mice+1
        Ti2(i) = Temp(i)
      enddo
    endif
  elseif (water == 1) then  ! There is no snow and ice, but water exists
    b(1) = 0.
    c(1) = 1.
    d(1) = Tsurf
    call DIFF_COEF(nveclen,a,b,c,d,2,M,2,M,water_indic,dt) 
    if (deepice == 1) then ! Water and deep ice exist
      a(M+1) = 0.
      c(M+1) = 1.
      d(M+1) = Meltpnt(Sal2(M+1),preswat(M+1),nmeltpoint%par)
      call PROGONKA(nveclen,a,b,c,d,Temp,1,M+1) 
      do i = 1, M+1
        Tw2(i) = Temp(i)
      enddo
    else ! There is water, and no deep ice
      if (soilswitch%par == 1) then ! Water and soil exist only
!---------------------WATER-SOIL INTERFACE------------------
        xx = bathymwater(M+1)%area_int/bathymwater(M)%area_half
        a(M+1) = 0.5*( - & !bathymwater(M+1)%area_int/bathymwater(M)%area_int * &
        & lamw(M)/(ddz(M)*h1) + xx*0.5*dt_inv*cw_m_row0*(dhw-dhw0)) + &
        & xx*0.5d0*cw_m_row0*PEMF(M)
        b(M+1) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
        cc = 0.5*dt_inv*(csoil(1)*rosoil(1)*dzs(1) + xx*cw_m_row0*ddz(M)*h1)
        c(M+1) = a(M+1) + b(M+1) - cc - xx*cw_m_row0*PEMF(M)
        d(M+1) = - Tw1(M+1)*cc - RadWater%integr(M) - &
        & xx*cw_m_row0*PEMF(M)*pt_down_f(M) + &
        & a(M+1)*Tw1(M) - (a(M+1) + b(M+1))*Tw1(M+1) + b(M+1)*Tsoil1(2,isoilcol)
!-----------------------------------------------------------
        call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,M+2,M+ns-1,soil_indic,dt,isoilcol)
        c(M+ns) = 1.
        a(M+ns) = 1.
        d(M+ns) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1)+lamsoil(ns))
        call PROGONKA (nveclen,a,b,c,d,Temp,1,M+ns)
        do i = M+1, M+ns
          Tsoil2(i-M) = Temp(i)
        enddo
      elseif (soilswitch%par == 0) then  ! Only water exists
        if (cuette%par == 0) then
          xx = bathymwater(M)%area_half/bathymwater(M+1)%area_int
          a(M+1) = 0.5*( - xx*lamw(M)/(ddz(M)*h1) + 0.5*dt_inv*cw_m_row0*(dhw-dhw0)) + &
          & 0.5d0*cw_m_row0*PEMF(M)
          c(M+1) = a(M+1) - 0.5*dt_inv*cw_m_row0*ddz(M)*h1 - cw_m_row0*PEMF(M)
          d(M+1) = - Tw1(M+1)*0.5*dt_inv*cw_m_row0*ddz(M)*h1 - xx*RadWater%integr(M) - &
          & cw_m_row0*PEMF(M)*pt_down_f(M) + soilbotflx%par + & 
          ! soilbotflx is interpreted as top soil boundary flux in this case
          & 0.5*( - xx*lamw(M)/(ddz(M)*h1) + 0.5*dt_inv*cw_m_row0*(dhw-dhw0)) * &
          & (Tw1(M) - Tw1(M+1))
          call PROGONKA (nveclen,a,b,c,d,Temp,1,M+1)
          Tsoil2(:) = Tsoil1(:,isoilcol)
        elseif (cuette%par == 1 .or. cuette%par == 11) then
          !B.c.s for Cuette flow
          a(M+1) = 0.
          c(M+1) = 1.
          d(M+1) = Tw1(M+1)
          call PROGONKA (nveclen,a,b,c,d,Temp,1,M+1)
          Tsoil2(:) = Tsoil1(:,isoilcol)
        endif
      endif
      do i = 1, M+1
        Tw2(i) = Temp(i)
      enddo
    endif
  else ! There is no snow, ice and water: only soil exists
    if (ifflux) then ! In this case Tsurf is a heat flux, downwards
      cc = rosoil(1)*csoil(1)*0.5*dzs(1)*dt_inv
      xx = 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
      b(1) = xx
      c(1) = cc + xx
      d(1) = Tsurf + xx*Tsoil1(2,isoilcol) + (cc - xx)*Tsoil1(1,isoilcol)
    else
      b(1) = 0.
      c(1) = 1.
      d(1) = Tsurf
    endif
    call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,2,ns-1,soil_indic,dt,isoilcol) 
    c(ns) = 0.5
    a(ns) = 0.5
    d(ns) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1)+lamsoil(ns)) + &
    & a(ns)*Tsoil1(ns-1,isoilcol) - c(ns)*Tsoil1(ns,isoilcol)
    call PROGONKA (nveclen,a,b,c,d,Temp,1,ns)
    do i = 1, ns
      Tsoil2(i) = Temp(i)
    enddo
  endif
else   
  if (water == 0) then
    RETURN
  else
    if (ice == 1) then ! Ice and water exist
      b(1) = 0.
      c(1) = 1.
      d(1) = Meltpnt(Sal2(1),preswat(1),nmeltpoint%par)
      call DIFF_COEF(nveclen,a,b,c,d,2,M,2,M,water_indic,dt)  
      if (deepice == 1) then ! Ice, water and deep ice exist
        a(M+1) = 0.
        c(M+1) = 1.
        d(M+1) = Meltpnt(Sal2(M+1),preswat(M+1),nmeltpoint%par)
        call PROGONKA (nveclen,a,b,c,d,Temp,1,M+1)
        do i = 1, M+1
          Tw2(i) = Temp(i)
        enddo
        b(1) = 0.
        c(1) = 1.
        d(1) = Meltpnt(Sal2(M+1),preswat(M+1),nmeltpoint%par)
        call DIFF_COEF(nveclen,a,b,c,d,2,Mice,2,Mice,deepice_indic,dt)
        if (soilswitch%par == 1) then ! Ice, water, deep ice and soil exist
!----------------DEEPICE-SOIL INTERFACE-------------------------
          xx = bathymdice(Mice+1)%area_int/bathymdice(Mice)%area_half
          a(Mice+1) = 0.5*( - & !bathymdice(Mice+1)%area_int/bathymdice(Mice)%area_int * &
          & lami/(ddzi(Mice)*ls1) + xx*0.5*dt_inv*ci_m_roi*(dls-dls0))
          b(Mice+1) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
          cc = 0.5*dt_inv*(csoil(1)*rosoil(1)*dzs(1) + xx*ci_m_roi*ddzi(Mice)*ls1)
          c(Mice+1) = a(Mice+1) + b(Mice+1) - cc 
          d(Mice+1) = - Tis1(Mice+1)*cc - RadDeepIce%integr(Mice) + &
          & a(Mice+1)*Tis1(Mice) - (a(Mice+1) + b(Mice+1))*Tis1(Mice+1) &
          & + b(Mice+1)*Tsoil1(2,isoilcol)
!-----------------------------------------------------------
          call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,Mice+2,Mice+ns-1,soil_indic,dt,isoilcol)  
          c(Mice+ns) = 1.
          a(Mice+ns) = 1.
          d(Mice+ns) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1) + lamsoil(ns))
          call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+ns)
          do i = Mice+1, Mice+ns
            Tsoil2(i-Mice) = Temp(i)
          enddo
        elseif (soilswitch%par == 0) then  ! Ice, water, deep ice exist, but not soil
          xx = bathymdice(Mice)%area_half/bathymdice(Mice+1)%area_int
          c(Mice+1) = - xx*0.5*lami/(ddzi(Mice)*ls1) + 0.5*dt_inv*ci_m_roi* &
          & ( - ddzi(Mice)*ls1 + 0.5*(dls-dls0)) 
          a(Mice+1) = - xx*0.5*lami/(ddzi(Mice)*ls1) + 0.25*dt_inv*ci_m_roi*(dls-dls0)
          d(Mice+1) = - Tis1(Mice+1)*0.5*dt_inv*ci_m_roi*ddzi(Mice)*ls1 - &
          & xx*RadDeepIce%integr(Mice) + soilbotflx%par + & 
          ! soilbotflx is interpreted as top soil boundary flux in this case
          & (xx*0.5*lami/(ddzi(Mice)*ls1) - 0.25*dt_inv*ci_m_roi*(dls-dls0)) * &
          & (Tis1(Mice+1) - Tis1(Mice))
          call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+1)
          Tsoil2(:) = Tsoil1(:,isoilcol)
        endif
        do i = 1, Mice+1
          Tis2(i) = Temp(i)
        enddo
      else ! Ice and water, but no deep ice
        if (soilswitch%par == 1) then ! Ice, water and soil exist, no deep ice
!---------------------WATER-SOIL INTERFACE------------------
          xx = bathymwater(M+1)%area_int/bathymwater(M)%area_half
          a(M+1) = 0.5*( - & !bathymwater(M+1)%area_int/bathymwater(M)%area_int * &
          & lamw(M)/(ddz(M)*h1) + xx*0.5*dt_inv*cw_m_row0*(dhw-dhw0)) + &
          & xx*0.5d0*cw_m_row0*PEMF(M)
          b(M+1) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
          cc = 0.5*dt_inv*(csoil(1)*rosoil(1)*dzs(1) + xx*cw_m_row0*ddz(M)*h1)
          c(M+1) = a(M+1) + b(M+1) - cc - xx*cw_m_row0*PEMF(M)
          d(M+1) = - Tw1(M+1)*cc - RadWater%integr(M) - &
          & xx*cw_m_row0*PEMF(M)*pt_down_f(M) + &
          & a(M+1)*Tw1(M) - (a(M+1) + b(M+1))*Tw1(M+1) + b(M+1)*Tsoil1(2,isoilcol)
!-----------------------------------------------------------
          call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,M+2,M+ns-1,soil_indic,dt,isoilcol)
          c(M+ns) = 1.
          a(M+ns) = 1.
          d(M+ns) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1)+lamsoil(ns))
          call PROGONKA (nveclen,a,b,c,d,Temp,1,M+ns)
          do i = M+1, M+ns
            Tsoil2(i-M) = Temp(i)
          enddo
        elseif (soilswitch%par == 0) then  ! Ice, water exist, no soil and no deep ice
          xx = bathymwater(M)%area_half/bathymwater(M+1)%area_int
          a(M+1) = 0.5*( - xx*lamw(M)/(ddz(M)*h1) + 0.5*dt_inv*cw_m_row0*(dhw-dhw0)) + &
          & 0.5d0*cw_m_row0*PEMF(M)
          c(M+1) = a(M+1) - 0.5*dt_inv*cw_m_row0*ddz(M)*h1 - cw_m_row0*PEMF(M)
          d(M+1) = - Tw1(M+1)*0.5*dt_inv*cw_m_row0*ddz(M)*h1 - xx*RadWater%integr(M) - &
          & cw_m_row0*PEMF(M)*pt_down_f(M) + soilbotflx%par + & 
          ! soilbotflx is interpreted as top soil boundary flux in this case
          & 0.5*( - xx*lamw(M)/(ddz(M)*h1) + 0.5*dt_inv*cw_m_row0*(dhw-dhw0)) * &
          & (Tw1(M) - Tw1(M+1))
          call PROGONKA (nveclen,a,b,c,d,Temp,1,M+1)
          Tsoil2(:) = Tsoil1(:,isoilcol)
        endif
        do i = 1, M+1
          Tw2(i) = Temp(i)
        enddo
      endif
    elseif (deepice == 1) then ! Water and deep ice exist, no top ice
      b(1) = 0.
      c(1) = 1.
      d(1) = Meltpnt(Sal2(M+1),preswat(M+1),nmeltpoint%par)
      call DIFF_COEF(nveclen,a,b,c,d,2,Mice,2,Mice,deepice_indic,dt)
      if (soilswitch%par == 1) then ! Water, deep ice and soil, no top ice
!----------------DEEPICE-SOIL INTERFACE-------------------------
        xx = bathymdice(Mice+1)%area_int/bathymdice(Mice)%area_half
        a(Mice+1) = 0.5*( - & !bathymdice(Mice+1)%area_int/bathymdice(Mice)%area_int * &
        & lami/(ddzi(Mice)*ls1) + xx*0.5*dt_inv*ci_m_roi*(dls-dls0))
        b(Mice+1) = - 0.25*(lamsoil(1) + lamsoil(2))/dzs(1)
        cc = 0.5*dt_inv*(csoil(1)*rosoil(1)*dzs(1) + xx*ci_m_roi*ddzi(Mice)*ls1)
        c(Mice+1) = a(Mice+1) + b(Mice+1) - cc 
        d(Mice+1) = - Tis1(Mice+1)*cc - RadDeepIce%integr(Mice) + &
        & a(Mice+1)*Tis1(Mice) - (a(Mice+1) + b(Mice+1))*Tis1(Mice+1) &
        & + b(Mice+1)*Tsoil1(2,isoilcol)
!-----------------------------------------------------------
        call DIFF_COEF(nveclen,a,b,c,d,2,ns-1,Mice+2,Mice+ns-1,soil_indic,dt,isoilcol)  
        c(Mice+ns) = 1.
        a(Mice+ns) = 1.
        d(Mice+ns) = - soilbotflx%par*dzs(ns-1)/(lamsoil(ns-1)+lamsoil(ns))
        call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+ns)
        do i = Mice+1, Mice+ns
          Tsoil2(i-Mice) = Temp(i)
        enddo
      elseif (soilswitch%par == 0) then  ! Water , deep ice exist, no soil and top ice
        xx = bathymdice(Mice)%area_half/bathymdice(Mice+1)%area_int
        c(Mice+1) = - xx*0.5*lami/(ddzi(Mice)*ls1) + 0.5*dt_inv*ci_m_roi * &
        & ( - ddzi(Mice)*ls1 + 0.5*(dls-dls0)) 
        a(Mice+1) = - xx*0.5*lami/(ddzi(Mice)*ls1) + 0.25*dt_inv*ci_m_roi*(dls-dls0)
        d(Mice+1) = - Tis1(Mice+1)*ci_m_roi*0.5*dt_inv*ddzi(Mice)*ls1 - &
        & RadDeepIce%integr(Mice) + soilbotflx%par + & 
        ! soilbotflx is interpreted as top soil boundary flux in this case
        & ( - xx*0.5*lami/(ddzi(Mice)*ls1) + 0.25*dt_inv*ci_m_roi*(dls-dls0))* &
        & (Tis1(Mice) - Tis1(Mice+1))
        call PROGONKA (nveclen,a,b,c,d,Temp,1,Mice+1)
        Tsoil2(:) = Tsoil1(:,isoilcol)
      endif
      do i = 1, Mice+1
        Tis2(i) = Temp(i)
      enddo
    endif
  endif
endif

deallocate(a, b, c, d, Temp)

END SUBROUTINE T_DIFF
    

!---------------------------------------------------------------------------------


SUBROUTINE DIFF_COEF(nn,a,b,c,d,n0,n1,m0,m1,substr,dt,isoilcol)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

!-------------------DEFINES COEFFICIENTS FOR SOLVING A THERMAL DIFFUSIVITY-----------------
!-------------------EQUATION BY FACTORIZATION METHOD----------------------------------------

use NUMERIC_PARAMS
use PHYS_CONSTANTS  
use DRIVING_PARAMS
use ARRAYS
use ARRAYS_BATHYM
use ARRAYS_SOIL
use ARRAYS_GRID
use ARRAYS_METHANE
use ARRAYS_OXYGEN
use ARRAYS_WATERSTATE
use ARRAYS_TURB
use SNOWSOIL_MOD, only : itop, Tsn, T, cs, lams, dz, dens
use RADIATION, only : RadWater, RadIce, RadDeepIce

implicit none

!Input variables
real(kind=ireals), intent(in) :: dt

integer(kind=iintegers), intent(in) :: n0, n1
integer(kind=iintegers), intent(in) :: m0, m1

integer(kind=iintegers), intent(in) :: substr

integer(kind=iintegers), intent(in), optional :: isoilcol

integer(kind=iintegers), intent(in) :: nn
real(kind=ireals), intent(out), dimension(1:nn) :: a, b, c, d

!External functions
real(kind=ireals), external :: DZETA
real(kind=ireals), external :: DZETAI

!Local variables
!real(kind=ireals), parameter :: kreg = 1.d-8 !Regularization coefficient for PROGONKA stability

real(kind=ireals) :: lam1, lam2
real(kind=ireals) :: dzmean
real(kind=ireals) :: aa, bb

integer(kind=iintegers) :: i, j

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_DIFF_COEF)
include 'lb_ctx_diff_coef.inc'

if (m1-m0 /= n1-n0) then
  print*, 'Error: diff_coef'
  STOP
endif

SELECT CASE (substr)
!CASE 1: SOIL
  case (soil_indic)
    do i = m0, m1
      j = i - m0 + n0
      lam2 = 0.5d0*(lamsoil(j) + lamsoil(j+1))
      lam1 = 0.5d0*(lamsoil(j-1) + lamsoil(j))
      dzmean = 0.5d0*(dzs(j-1) + dzs(j))
      a(i) = - 0.5d0*lam1/dzs(j-1)
      b(i) = - 0.5d0*lam2/dzs(j)
      c(i) = a(i) + b(i) - dt_inv*(csoil(j)*rosoil(j)*dzmean)
      d(i) = - Tsoil1(j,isoilcol)*dt_inv*(csoil(j)*rosoil(j)*dzmean) + &
      & a(i)     *    Tsoil1(j-1,isoilcol) - &
      & (a(i) + b(i))*Tsoil1(j,  isoilcol) + &
      & b(i)     *    Tsoil1(j+1,isoilcol)
    enddo 
!CASE 2: ICE
  case (ice_indic)
    do i = m0, m1
      j = i - m0 + n0
      a(i) = 0.5d0*( - lami_v(j-1)*bathymice(j-1)%area_half / &
      & (ddzi(j-1)*l1*bathymice(j)%area_int) + &
      & ci_m_roi_v(j)*dhi*dzetai_int(j)*dt_inv05 - &
      & ci_m_roi_v(j)*dhi0*dt_inv05)
      b(i) = 0.5d0*( - lami_v(j  )*bathymice(j  )%area_half / &
      & (ddzi(j  )*l1*bathymice(j)%area_int) - &
      & ci_m_roi_v(j)*dhi*dzetai_int(j)*dt_inv05 + &
      & ci_m_roi_v(j)*dhi0*dt_inv05)  
      c(i) = a(i) + b(i) - dt_inv*ddzi05(j-1)*l1*ci_m_roi_v(j)
      d(i) = - Ti1(j)*dt_inv*ddzi05(j-1)*l1*ci_m_roi_v(j) + &
      & RadIce%integr(j) - RadIce%integr(j-1) + a(i)*Ti1(j-1) + b(i)*Ti1(j+1) - &
      & (a(i) + b(i))*Ti1(j) - lsh%ice(j)*ddzi05(j-1)*l1
    enddo
!CASE 2.1: ICE salinity (no diffusion)
  case (ice_sal_indic)
    do i = m0, m1
      j = i - m0 + n0
      a(i) = 0.5d0*(dhi*dzetai_int(j)*dt_inv05 - dhi0*dt_inv05)
      b(i) = 0.5d0*( - dhi*dzetai_int(j)*dt_inv05 + dhi0*dt_inv05)  
      c(i) = a(i) + b(i) - dt_inv*ddzi05(j-1)*l1
      d(i) = - salice(j)*dt_inv*ddzi05(j-1)*l1 + &
      &  a(i)*salice(j-1) + b(i)*salice(j+1) - &
      & (a(i) + b(i))*salice(j) - asalice*salice(j)*ddzi05(j-1)*l1
    enddo
!CASE   3: SNOW  
  case (snow_indic)
    do i = m0, m1
      j = i - m0 + n0
      lam2 = 0.5d0*(lams(j) + lams(j+1))
      lam1 = 0.5d0*(lams(j-1) + lams(j))
      dzmean = 0.5d0*(dz(j-1) + dz(j))
      a(i) = - 0.5d0*lam1/dz(j-1)
      b(i) = - 0.5d0*lam2/dz(j)
      c(i) = a(i) + b(i) - dt_inv*(cs(j)*dens(j)*dzmean)
      d(i) = - T(j)*dt_inv*(cs(j)*dens(j)*dzmean) + &
      & a(i)*T(j-1) - (a(i) + b(i))*T(j) + b(i)*T(j+1)
    enddo 
!CASE 4: WATER
  case (water_indic)
    do i = m0, m1
      j = i - m0 + n0
      aa = 0.5d0*( - area_half(j-1)*lamw(j-1)/(ddz(j-1)*h1*area_int(j)) + &
      & cw_m_row0*dhw*dzeta_int(j)*dt_inv05 - &
      & cw_m_row0*dhw0*dt_inv05)
      a(i) = aa + cw_m_row0*0.5d0*PEMF(j-1)
      bb = 0.5d0*( - area_half(j)*lamw(j)/(ddz(j)*h1*area_int(j)) - &
      & cw_m_row0*dhw*dzeta_int(j)*dt_inv05 + &
      & cw_m_row0*dhw0*dt_inv05)
      b(i) = bb - cw_m_row0*0.5d0*PEMF(j)
      c(i) = aa + bb - &
      & dt_inv*ddz05(j-1)*h1*cw_m_row0 + &
      & cw_m_row0*0.5d0*(PEMF(j)-PEMF(j-1))
      d(i) = - Tw1(j)*dt_inv*ddz05(j-1)*h1*cw_m_row0 + &
      & (area_half(j)*RadWater%integr(j) - &
      & area_half(j-1)*RadWater%integr(j-1))/area_int(j) - &
      & cw_m_row0*PEMF(j-1)*pt_down_f(j-1) + &
      & cw_m_row0*PEMF(j)*pt_down_f(j) + &

      & aa*Tw1(j-1) + bb*Tw1(j+1) - &

      & (aa + bb)*Tw1(j) - &

      & lsh%water(j)*ddz05(j-1)*h1 
    enddo 
!CASE 5: DEEP ICE
  case (deepice_indic)
    do i = m0, m1
      j = i - m0 + n0
      a(i) = 0.5d0*( - lami*bathymdice(j-1)%area_half / &
      & (ddzi(j-1)*ls1*bathymdice(j)%area_int) + &
      & ci_m_roi*dls*dzetai_int(j)*dt_inv05 - &
      & ci_m_roi*dls0*dt_inv05)
      b(i) = 0.5d0*( - lami*bathymdice(j  )%area_half/ &
      & (ddzi(j  )*ls1*bathymdice(j)%area_int) - &
      & ci_m_roi*dls*dzetai_int(j)*dt_inv05 + &
      & ci_m_roi*dls0*dt_inv05)
      c(i) = a(i) + b(i) - &
      & dt_inv*ddzi05(j-1)*ls1*ci_m_roi
      d(i) = - Tis1(j)*dt_inv*ddzi05(j-1)*ls1*ci_m_roi + &
      & RadDeepIce%integr(j) - RadDeepIce%integr(j-1) + &
      & a(i)*Tis1(j-1) + b(i)*Tis1(j+1) - &
      & (a(i) + b(i))*Tis1(j) - &
      & lsh%dice(j)*ddzi05(j-1)*ls1
    enddo
!CASE 6: SALINITY IN WATER
 case (water_salinity_indic)
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - area_half(j-1)*lamsal(j-1)/(ddz(j-1)*h1*area_int(j)) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05)
     b(i) = 0.5d0*( - area_half(j)  *lamsal(j)  /(ddz(j)  *h1*area_int(j)) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05)
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - Sal1(j)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*Sal1(j-1) + b(i)*Sal1(j+1) - &
     & (a(i) + b(i))*Sal1(j)
   enddo 
!CASE 7: SALINITY IN THE SOIL  
 case (soil_salinity_indic)
   do i = m0, m1
     j = i - m0 + n0
     a(i) = - 0.5d0*wsoil(j-1)*dt/(dzs(j-1) + dzs(j))
     b(i) = 0.5d0*wsoil(j)*dt/(dzs(j-1) + dzs(j))   
     c(i) = - 1.d0 - a(i) - b(i) !+ 2.*kreg
     !a(i) = a(i) !+ kreg
     !b(i) = b(i) !+ kreg
     d(i) = - Sals1(j,nsoilcols) + &
     & a(i)*Sals1(j-1,nsoilcols) + b(i)*Sals1(j+1,nsoilcols) + &
     & (a(i) + b(i))*Sals1(j,nsoilcols)
   enddo 
!CASE 8: METHANE IN WATER     
 case (water_methane_indic)
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - area_half(j-1)*lammeth(j-1)/(ddz(j-1)*h1*area_int(j) ) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05 )
     b(i) = 0.5d0*( - area_half(j)  *lammeth(j)  /(ddz(j)  *h1*area_int(j) ) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05 )
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - qwater(j,1)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*qwater(j-1,1) + b(i)*qwater(j+1,1) - &
     & (a(i) + b(i))*qwater(j,1) - &
     & lsm%water(j)*ddz05(j-1)*h1
   enddo
!CASE 9: OXYGEN IN WATER
 case (water_oxygen_indic)
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - area_half(j-1)*lamoxyg(j-1)/(ddz(j-1)*h1*area_int(j)) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05)
     b(i) = 0.5d0*( - area_half(j)  *lamoxyg(j)  /(ddz(j)  *h1*area_int(j)) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05) 
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - oxyg(j,1)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*oxyg(j-1,1) + b(i)*oxyg(j+1,1) - &
     & (a(i) + b(i))*oxyg(j,1) - &
     & lso%water(j)*ddz05(j-1)*h1
   enddo
!CASE 10: CARBON DIOXIDE IN WATER
 case (water_carbdi_indic)
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - area_half(j-1)*lamcarbdi(j-1)/(ddz(j-1)*h1*area_int(j)) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05)
     b(i) = 0.5d0*( - area_half(j)  *lamcarbdi(j)  /(ddz(j)  *h1*area_int(j)) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05) 
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - DIC(j,1)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*DIC(j-1,1) + b(i)*DIC(j+1,1) - &
     & (a(i) + b(i))*DIC(j,1) - &
     & lsc%water(j)*ddz05(j-1)*h1
   enddo          
!CASE 11: DISSOLVED CARBON SPECIES IN WATER, ordinary diffusion
 case (water_carbon_indic)
   ! No morphometry effects included
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - lamcarbdi(j-1)/(ddz(j-1)*h1) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05)
     b(i) = 0.5d0*( - lamcarbdi(j)  /(ddz(j)  *h1) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05) 
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - workc(j)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*workc(j-1) + b(i)*workc(j+1) - &
     & (a(i) + b(i))*workc(j)
   enddo
!CASE 12: PARTICLES IN WATER, ordinary diffusion
 case (water_particles_indic)
   ! No morphometry effects included
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - lampart(j-1)/(ddz(j-1)*h1) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05)
     b(i) = 0.5d0*( - lampart(j)  /(ddz(j)  *h1) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05) 
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - workc(j)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*workc(j-1) + b(i)*workc(j+1) - &
     & (a(i) + b(i))*workc(j)
   enddo
!CASE 13: PARTICLES IN WATER, area-weighted diffusion
 case (water_particles_morph_indic)
   ! Morphometry effects included
   do i = m0, m1
     j = i - m0 + n0
     a(i) = 0.5d0*( - area_half(j-1)*lampart(j-1)/(ddz(j-1)*h1*area_int(j)) + &
     & dhw*dzeta_int(j)*dt_inv05 - dhw0*dt_inv05)
     b(i) = 0.5d0*( - area_half(j  )*lampart(j)  /(ddz(j)  *h1*area_int(j)) - &
     & dhw*dzeta_int(j)*dt_inv05 + dhw0*dt_inv05) 
     c(i) = a(i) + b(i) - dt_inv*ddz05(j-1)*h1  
     d(i) = - workc(j)*dt_inv*ddz05(j-1)*h1 + &
     & a(i)*workc(j-1) + b(i)*workc(j+1) - &
     & (a(i) + b(i))*workc(j)
   enddo
END SELECT
  
END SUBROUTINE DIFF_COEF


SUBROUTINE SURF_CHAR(surftyp,year,month,day,hour,phi,fetch)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals
use DRIVING_PARAMS
use PHYS_CONSTANTS
use ARRAYS
use ATMOS, only: &
& WIND, &
& VELFRICT_PREV
use SFCFLX, only: &
& SFCFLX_ROUGHNESS    
use PHYS_FUNC, only: &
& SINH0, &
& WATER_ALBEDO, &
& CHARNOCK_Z0, &
& H13, DOMWAVESPEED

!----------------DEFINES SURFACE CHARACTERISTICS:-------------------
!----------------roughness,emissivity,albedo,relative humidity,-----
!----------------coefficients in Magnus formula---------------------


implicit none

integer(kind=iintegers), intent(in) :: surftyp
integer(kind=iintegers), intent(in) :: year
integer(kind=iintegers), intent(in) :: month
integer(kind=iintegers), intent(in) :: day
real(kind=ireals)   , intent(in) :: hour
real(kind=ireals)   , intent(in) :: phi
real(kind=ireals)   , intent(in) :: fetch

real(kind=8) :: x1,x2,x3,x4,x5 !For interface with FLake's subroutines

 
SAVE 

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_SURF_CHAR)
include 'lb_ctx_surf_char.inc'
    
select case (surftyp)
  case (1)
    albedo = albedoofsoil
    albedo_lw = albedoofsoil_lw
    roughness = 0.05
    aM = aMagw 
    bM = bMagw
    emissivity = emissivityofsoil
    relhums = 0.5
  case (2) 
    albedo = albedoofice
    albedo_lw = albedoofice_lw
    emissivity = emissivityofice
    aM = aMagi
    bM = bMagi 
    roughness = 0.00001
    relhums = 0.7
  case (3)
    albedo = albedoofsnow
    albedo_lw = albedoofsnow_lw
    emissivity = emissivityofsnow
    aM = aMagi
    bM = bMagi 
    roughness = 0.001
    relhums = 0.7
  case (4)
    velfrict_prev = max(velfrict_prev, 1.d-2)
!   roughness = roughness0
!   roughness = 2.68282E-6*wind**4 - 5.321223E-5*wind**3 + 0.00038836*wind**2 - &
!   & 0.00119916*wind + 0.0013668
!   Charnock formula
!    roughness = CHARNOCK_Z0(velfrict_prev)
!   Charnock formula extended for limited fetch (FLake)
    call SfcFlx_roughness (dble(fetch), dble(wind), dble(velfrict_prev), 0._8, &
    & x1, x2, x3, x4, x5)
    roughness = x3
!!   Fetch dependent roughness following Drennan et al., 2003
!    x1 = 3.35
!    roughness = H13(roa0*velfrict_prev**2,fetch)*x1* &
!    & (velfrict_prev/DOMWAVESPEED(velfrict_prev,fetch))**3.4
    aM = aMagw
    bM = bMagw
    if (varalb%par == 0) then
      albedo = albedoofwater
    elseif (varalb%par == 1) then
      albedo = WATER_ALBEDO( SINH0(year,month,day,hour,phi) )
      !albedo = (dirdif()*0.05/(sinh0()+0.15)+0.05)/ &
      !(1+dirdif())
    endif
    albedo_lw = albedoofwater_lw ! alternative: 1 - emissivityofwater
    emissivity = emissivityofwater 
    relhums = 1. !0.9
end select  

END SUBROUTINE SURF_CHAR


END MODULE T_SOLVER_MOD
