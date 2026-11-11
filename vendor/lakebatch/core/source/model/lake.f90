MODULE LAKE_MOD

type ISIMIP_Lake_type
  ! Type for ISIMIP-Lake output
  sequence
  integer :: istratstart, istratend, istratdur
  integer :: icestart, icetend, icedur
  real :: thermodepth
  real :: surftemp, bottemp
  real :: lakeicefrac
  real :: icethick
  real :: snowthick
  real :: waterthick
  real :: icetemp
  real :: snowtemp
  real :: sensheatf
  real :: latentheatf
  real :: momf
  real :: lwup
  real :: swup
  real :: lakeheatf
  real :: albedo
  real :: extcoeff
  real :: sedheatf
  real, allocatable :: watertemp(:)
  real, allocatable :: zwatertemp(:)
end type ISIMIP_Lake_type

contains
SUBROUTINE INIT_LAKE(nx0,ny0,nx,ny,fnmap1,dt)

!The subroutine INIT_LAKE implements
!initialisation of the model (but not initial conditions)

use LAKE_DATATYPES, only : ireals, iintegers
use PHYS_PARAMETERS
use NUMERIC_PARAMS
use PHYS_CONSTANTS
use DRIVING_PARAMS 
use ATMOS
use ARRAYS
use ARRAYS_GRID, only : gs
use TURB_CONST, only : &
& TURB_CONST_DERIVED_SET
use INOUT_PARAMETERS, only : &
& lake_subr_unit_min, &
& lake_subr_unit_max
use WATER_DENSITY
use SOIL_LAKE_MOD, only : COMSOILFORLAKE
use SURF_SCHEME1_MOD, only : PBLDAT_LAKE
use SURF_SCHEME_INM, only : PBLDAT_INM
use INOUT, only : fext2, readgrd_lake
use BL_MOD_LAKE, only : BL_MOD_LAKE_ALLOC
use SNOWSOIL_MOD, only : SNOWSOIL_MOD_ALLOC
use METH_OXYG_CONSTANTS, only : SET_METH_OXYG_CONSTANTS
use ALLOCARRAYS_MOD, only : ALLOCARRAYS

implicit none

!Input variables
!Reals
real(kind=ireals), intent(in) :: dt

!Integers
integer(kind=iintegers), intent(in) :: nx0, ny0, nx, ny

!Characters
character(len=*), intent(in) :: fnmap1

!Output variables

!Local variables
!Reals
real(kind=ireals), parameter :: hour_sec = 60.*60.
real(kind=ireals) :: x1

!Integers
integer(kind=iintegers) :: n_unit = lake_subr_unit_min
integer(kind=iintegers) :: i, j, dnx, dny

!Logicals
logical :: uniform_depth

!Characters
character(len=80) :: path2
character(len=80) :: fnmap

data uniform_depth /.true./
!data n_unit /999/

dnx = nx - nx0 + 1
dny = ny - ny0 + 1

fnmap = fnmap1

call DEFINE_PHYS_PARAMETERS
call DEFINE_PHYS_CONSTANTS
call DEFINE_DRIVING_PARAMS
call SET_METH_OXYG_CONSTANTS
call ALLOCARRAYS(dnx,dny)
call BL_MOD_LAKE_ALLOC
call SNOWSOIL_MOD_ALLOC

call PBLDAT_LAKE
call PBLDAT_INM
call COMSOILFORLAKE
call TURB_CONST_DERIVED_SET
call SET_DENS_CONSTS(eos%par)
call NUMERIC_PARAMS_SET(gs)

if (error_cov==1) then
  if (runmode%par == 2) then
    print*, 'The error covariance estimation algorithm works &
    &only in standalone runs: STOP'
    STOP
  endif
  if (assim == 1) then
    print*, 'assim = 1 and error_cov = 1: these regimes could not &
    &be turned on simultaneously: STOP'
    STOP
  endif
  if (dnx > 1 .or. dny > 1) then
    print*, 'Error covariance calculation is currently adjusted &
    &only for one-point stand-alone runs of the lake model: STOP'
    STOP
  endif
endif
if (assim==1.and.(dnx>1.or.dny>1)) then
  print*, 'Data assimilation is currently adjusted &
  &only for one-point stand-alone lake model: STOP'
  STOP
endif

if (dt_out%par<dt/hour_sec) then
  print*, 'dt_out must be larger or equal to timestep: STOP'
  STOP
endif
   
if (runmode%par == 2.and.(.not.uniform_depth)) then
  path2 = fext2(fnmap,'dep')
  call CHECK_UNIT(lake_subr_unit_min,lake_subr_unit_max,n_unit)
  open (n_unit, file=path(1:len_trim(path))// &
  & path2(1:len_trim(path2)), status='old')
  call READGRD_LAKE(n_unit,dep_2d,0,nx-1,0,ny-1)
  close (n_unit)
  dep_av = 0.
  x1 = 0.
  do i = 1, nx-1
    do j = 1, ny-1
      if (dep_2d(i,j) > 0.) then
        dep_av = dep_av + dep_2d(i,j)
        x1 = x1 + 1.
      endif
    enddo
  enddo
  dep_av = dep_av / x1
endif

END SUBROUTINE INIT_LAKE

!>MAIN SUBROUTINE OF THE MODEL 
!!Calculates all parameters characterizing the state of a lake (temperature, currents,
!!eddy diffusivity coefficients, snow, ice, sediments characteristics, biogeochemistry etc.) 
!!at the next timestep (i.e. implements evolution of variables 
!!from i-th time step to (i+1)-th )
!!In the atmospheric model must be called once each timestep,
!!or once each N timesteps, where N = dt_lake/dt_atmos 
!!(dt_lake - timestep in the lake model, dt_atmos - timestep the atmospheric model) 
SUBROUTINE LAKE &
& (tempair1,humair1,pressure1,uwind1,vwind1, &
& longwave1,netrad1,shortwave1,precip1,Sflux01, &
& ch4_pres_atm, co2_pres_atm, o2_pres_atm, zref1, dtl, &
& h10, l10, ls10, hs10, Ts0, &
& Tb0, Tbb0, h_ML0, extwat, extice, &
& kor_in, trib_inflow, Sals0, Salb0, fetch, &
& phi, lam, us0, vs0, Tm, &
& alphax, alphay, a_veg, c_veg, h_veg, &
& area_lake, cellipt, depth_area, tsw, hw1, &
& xlew1, cdmw1, surfrad1, cloud1, SurfTemp1, &
& ftot, fdiff_lake_surf1, h2_out, ix, iy, &
& nx0, ny0, nx, ny, nx_max, ny_max, &
& ndatamax, year, month, day, hour, &
& init_T, flag_assim, flag_print, outpath1, spinup_done, &
& dataread, lakeform, comm3d, rank_comm3d, coords, &
& parallel, icp, step_final, LakePhysISIMIP)

!OUTPUT VARIABLES:   
!tsw--- surface temperature of a lake, K;
!hw1--- sensible heat flux from lake, W/m**2; 
!xlew1   --- latent heat flux from lake, W/m**2; 
!cdmw    --- exchange coefficient, m/s ( == wind_speed*exchange_nondimensional_coeficient)

use LAKE_DATATYPES, only : ireals, iintegers
use OMP_LIB

use COMPARAMS, only: &
& num_ompthr, parparams

use NUMERIC_PARAMS
use DRIVING_PARAMS
use ATMOS
use PHYS_CONSTANTS
use ARRAYS
use ARRAYS_GRID
use ARRAYS_BATHYM
use ARRAYS_SOIL
use ARRAYS_TURB
use ARRAYS_WATERSTATE
use ARRAYS_METHANE
use ARRAYS_OXYGEN
use MODI_MASSFLUX_CONVECTION
use CONVECTPAR_LAKE_MOD, only : UnDenGrMix
use SALINITY_LAKE_MOD, only : S_DIFF, UPDATE_ICE_SALINITY
use TURB_LAKE_MOD, only : K_EPSILON, ED_TEMP_HOSTETLER, &
&                         ED_TEMP_HOSTETLER2, ZIRIVMODEL, &
&                         EDDY_DIFF_MOST_RICH
use TURB_CONST, only : min_diff
use DZETA_MOD, only : VARMEAN

use MOMENTUM_LAKE_MOD, only : &
& MOMENTUM_SOLVER

use VERTADV, only : &
& VERTADV_UPDATE

use METHANE_LAKE_MOD, only : METHANE, &
& METHANE_OXIDATION, METHANE_OXIC_PRODUCTION

use SOIL_LAKE_MOD, only : &
& SOILFORLAKE, &
& SOIL_COND_HEAT_COEF, &
& SOILCOLSTEMP, &
& LATERHEAT, &
& AVERSOILCOL

use INIT_VAR_MOD, only : &
& INIT_VAR

use T_SOLVER_MOD, only : &
& T_SOLVER

use WATER_DENSITY, only: &
& DENSITY_W

use PHYS_FUNC, only: &
& TURB_SCALES, &
& MELTPNT, &
& WATER_FREEZE_MELT, &
& TURB_DENS_FLUX, &
& SINH0, &
& WATER_ALBEDO, &
& EXTINCT_SNOW, &
& UNFRWAT, &
& WI_MAX, &
& WL_MAX, &
& MIXED_LAYER_CALC, &
& HC_CORR_CARBEQUIL, &
& DIFFMIN_HS, DIFFMIN_KPP, &
& LONGWAVE_RADIATION, &
& SHORTWAVE_RADIATION, &
& RICHARDSON_NUMBER

use EVOLUTION_VARIABLES, only : &
& UPDATE_CURRENT_TIMESTEP, &
& UPDATE_NEXT_TIMESTEP

use TURB_CONST, only : &
& lamTM, min_diff

use METH_OXYG_CONSTANTS, only : &
& r0_methprod_phot, r0_oldorg2_star, &
& ngasb, mf, &
& molm3tonM, &
& molm3toppm_co2,molm3toppm_o2,molm3toppm_ch4, &
& molm3tomgl_co2,molm3tomkgl_ch4,molm3tomgl_o2, &
& molm3tomcM, &
& photic_threshold, &
& pH, &
& molmass_p

use OXYGEN_MOD, only : &
& OXYGEN, OXYGEN_PRODCONS, &
& ADDOXPRODCONS, &
& CHLOROPHYLLA

use CARBON_DIOXIDE_LAKE_MOD, only : &
& CARBON_DIOXIDE

use PHOSPHORUS_MOD, only : &
& PHOSPHORUS

use TIMEVAR, only : &
& hour_sec, &
& day_sec, &
& year_sec, &
& month_sec

use OUT_MOD

use BUBBLE_LAKE_MOD, only : BUBBLE, BUBBLEFLUXAVER

use CONTROL_POINT_LAKE_MOD, only : CONTROL_POINT_OUT, CONTROL_POINT_IN
use LAKEBATCH_RESTART, only : RESTART_IN_PRESENT, RESTART_OUT_REQUESTED, RESTART_OPEN, RESTART_CLOSE
use LB_RESTART_IO, only : LB_MODULES_IO, lb_rec, lb_chk, lb_none, lbtag
use LB_POINT_CTX, only : LBC_SET_POINT, LBC_CLEAR_POINT, LBC_MODULES, lbc_begin, &
& LBC_SCOPE_LAKE, lbc_act, lbc_storing, lbc_stored, lbc_loading, lbc_mem, &
& lbc_puti, lbc_geti, lbc_lb, lbc_ub
#ifdef LB_DUMP
use LB_STATE_DUMP, only : LB_DUMP_MODULES
use EVOLUTION_VARIABLES, only : nstep_2d
#endif

use TRIBUTARIES, only : TRIBTEMP, LBC_TRIBUTARIES

use BATHYMSUBR, only : BATHYMETRY

use NUMERICS, only : ACCUMM, VALUE_IS_FINITE

use BL_MOD_LAKE

use SNOWSOIL_MOD
use SNOWTEMP_LAKE, only : SNOWTEMP, SNOW_COND_HEAT_COEF

use RADIATION, only : &
& RadWater, RadIce, RadDeepIce, &
& fracbands, nbands, extwatbands

use ZERODIM_MODEL_LAKE_MOD, only : ZERODIM_MODEL

implicit none

!Input variables
!Reals
real(kind=ireals), intent(in) :: tempair1   !> Air temperature, K
real(kind=ireals), intent(in) :: humair1    !> Specific humidity, kg/kg
real(kind=ireals), intent(in) :: pressure1  !> Air pressure, Pa
real(kind=ireals), intent(in) :: uwind1, vwind1 !> x- and y-components of wind speed, m/s
real(kind=ireals), intent(in) :: longwave1  !> Longwave downward radiation, W/m**2
real(kind=ireals), intent(in) :: netrad1    !> Net radiation, W/m**2
real(kind=ireals), intent(in) :: zref1      !> Height above the lake surface, where the forcing is given, m
real(kind=ireals), intent(in) :: shortwave1 !> Global shortwave radiation, W/m**2
real(kind=ireals), intent(in) :: precip1    !> Precipitation intensity, m/s
real(kind=ireals), intent(in) :: Sflux01    !> Salinity flux at the water surface by rain
real(kind=ireals), intent(in) :: cloud1     !> Cloud sky fraction, 0-1, n/d
real(kind=ireals), intent(in) :: SurfTemp1  !> Surface temperature, K

real(kind=ireals), intent(in) :: ch4_pres_atm, co2_pres_atm, o2_pres_atm
real(kind=ireals), intent(in) :: dtl !> Time step for the LAKE model, s

real(kind=ireals), intent(in), target :: hour

real(kind=ireals), intent(in) :: h10, l10, ls10, hs10
real(kind=ireals), intent(in) :: Ts0, Tb0, Tbb0
real(kind=ireals), intent(in) :: h_ML0
real(kind=ireals), intent(in) :: extwat, extice
real(kind=ireals), intent(in) :: kor_in


real(kind=ireals), intent(in) :: trib_inflow(1:nvartribext) !> The group of tributary data: discharge, ...
real(kind=ireals), intent(in) :: Sals0, Salb0
real(kind=ireals), intent(in) :: fetch
real(kind=ireals), intent(in) :: phi, lam
real(kind=ireals), intent(in) :: us0, vs0
real(kind=ireals), intent(in) :: Tm
real(kind=ireals), intent(inout) :: alphax, alphay
real(kind=ireals), intent(in) :: a_veg, c_veg, h_veg
real(kind=ireals), intent(in) :: area_lake, cellipt
real(kind=ireals), intent(in) :: depth_area(1:ndatamax,1:2)

!Integers

!> Indicator of the lake cross-section form
!! 1 - ellipse
!! 2 - rectangle
integer(kind=iintegers), intent(in) :: lakeform

!> Coordinates of the current lake on a horizontal mesh
integer(kind=iintegers), intent(in) :: ix, iy

!> Dimensions of the horizontal mesh
integer(kind=iintegers), intent(in) :: nx, ny
integer(kind=iintegers), intent(in) :: nx0, ny0, nx_max, ny_max
integer(kind=iintegers), intent(in) :: ndatamax
integer(kind=iintegers), intent(in), target :: year, month, day
integer(kind=iintegers), intent(in) :: init_T
!> MPI parameters
integer(kind=iintegers), intent(in) :: comm3d, rank_comm3d, coords(1:3) 
!> Control point (CP) switch: 0 - do nothing
!!                            1 - write CP                             
!!                            2 - read CP as initial conditions                                           
integer(kind=iintegers), intent(in) :: icp 

                                           
!Characters
character(len=60), intent(in) :: outpath1

!Logicals
logical, intent(in) :: flag_assim
logical, intent(in) :: flag_print
logical, intent(in) :: spinup_done 
logical, intent(in) :: dataread
logical, intent(in) :: parallel
!> Indicator of the last time step
logical, intent(in) :: step_final

!Output variables
!Reals
real(kind=ireals), intent(inout) :: hw1, xlew1, cdmw1, surfrad1
real(kind=ireals), intent(out) :: tsw
real(kind=ireals), intent(out) :: ftot, fdiff_lake_surf1
real(kind=ireals), intent(out) :: h2_out

type(ISIMIP_Lake_type), intent(inout), optional :: LakePhysISIMIP

!------------------------ MAIN VARIABLES ------------------------
!  arrays:    Tw1 and Tw2 - temperature of water, C
! Ti1 and Ti2 - temperature of ice, C
! T - temperature of snow, C
!  functions of time:h1 and h2 - thickness of water, m
! l1 and l2 - thickness of ice, m
! hs1 - thickness of snow, m  
! flag_ice_surf - shows if ice is melting on the top surface, n/d  
!  constants: cw - specific heat of water, J/(kg*K) 
! ci - specific heat of ice, J/(kg*K)
! row0 - density of water, kg/m**3
! roi - density of ice, kg/m**3
! lamw - thermal conductivity of water, J*sec/(m**3*K)
! lami - thermal conductivity of ice, J*sec/(m**3*K)
! L - specific heat of melting, J/kg
! Lwv - specific heat of evaporation, J/kg
! Liv - specific heat of sublimation, J/kg    
! ddz - size of grid element (space), m
! dt - size of grid element (time), sec
! kstratwater - coefficient for linear initial conditions in water
! kstratice - coefficient for linear initial conditions in ice 
!  boundary conditions:
! eFlux - total heat flux on the upper surface,
!   shortwave(1-A)-T**4+longwave-H-LE, J/(m**2*sec)
! Elatent - latent heat flux, J/(m**2*sec) 
! Erad - shortwave radiation, penetrated below a surface, J/(m**2*sec)
! tempair - air temperature (2 m), C
! precip - precipitation, m/sec
!(M) number of layers in water and snow   

!Local variables
!Reals
real(kind=ireals), allocatable :: work1(:), work2(:), work3(:), work4(:), work5(:), work6(:)
real(kind=ireals), allocatable :: radflux(:)
real(kind=ireals) :: l2, h2, ls2
real(kind=ireals) :: totalevap, totalprecip, totalmelt, totalpen, totalwat
real(kind=ireals) :: snowmass, snowmass_init
real(kind=ireals) :: totalerad, totalhflux
real(kind=ireals) :: h_ML0zv
real(kind=ireals) :: x, xx, y, yy, zz, zzz, vol_, vol_2, tt, ttt, ww, ww1, www
real(kind=ireals) :: kor
real(kind=ireals) :: Ti10 ! Initial temperature of the ice surface (if l10 > 0)

!real(kind=ireals), dimension(1:nveclen) :: a, b, c, d 
real(kind=ireals), allocatable :: a(:), b(:), c(:), d(:) 

real(kind=ireals) :: dt
real(kind=ireals) :: b0
real(kind=ireals) :: tau_air, tau_i, tau_gr
real(kind=ireals) :: eflux0_kinem_water
real(kind=ireals) :: totmeth0, totmethc, metracc = 0.

!real(kind=ireals), dimension(1:nveclen) :: Temp
real(kind=ireals), allocatable :: Temp(:)

real(kind=ireals) :: flux1, flux2

real(kind=ireals) :: dTisnmelt, dTiimp, dTmax, fracsn, fracimp

real(kind=ireals) :: calibr_par_min(1:2), calibr_par_max(1:2)
!data calibr_par_min /1.7d-8/5., 1.d-10/5./
!data calibr_par_max /1.7d-8*5., 1.d-10*5./
real(kind=ireals) :: step_calibr(1:2)


!Integers    
integer(kind=iintegers) :: i, j, k, nn = 0, i2, j2
integer(kind=iintegers) :: i_ML
integer(kind=iintegers) :: flag_snow
integer(kind=iintegers) :: nstep_meas
integer(kind=iintegers) :: n_1cm, n_5cm
integer(kind=iintegers) :: layer_case
integer(kind=iintegers) :: ndec
integer(kind=iintegers) :: npoint_calibr(1:2) = 12
integer(kind=iintegers) :: dnx, dny
integer(kind=iintegers) :: nstep_cp_written = 0
integer(kind=iintegers) :: iwork(1:2)

!Logicals
logical :: uniform_depth
logical :: flag
logical :: accum = .false.
logical :: add_to_winter
logical :: firstcall = .true.
! lakebatch restart (lb_* names are not part of the saved state)
integer :: lb_u, lb_rk
integer(8) :: lb_nb, lb_lb(7), lb_ub(7)
logical :: lb_reading
#ifdef LB_DUMP
integer :: lb_dump_step = -2
character(len=32) :: lb_env
#endif
logical :: multsoil
logical :: worklog, worklog1, indTMprev = .false.
logical :: control_point_read = .false.

!Characters
character(len=60) :: workchar

data uniform_depth /.true./
data nstep_meas /0/
 
SAVE
  
! lakebatch multi-point: this point's copy of everything carried between calls
! (module state and LAKE's saved locals, see lb_point_ctx.f90); off for one point
call LBC_SET_POINT(ix, iy, nx, ny)
call LBC_MODULES()
call LBC_TRIBUTARIES()
call lbc_begin(LBC_SCOPE_LAKE)
include 'lb_ctx_lake.inc'

allocate(a(1:nveclen),b(1:nveclen),c(1:nveclen), &
&        d(1:nveclen),Temp(1:nveclen))
allocate(work2(1:nsoilcols+1))

! lakebatch restart: continue exactly where the saved run stopped
if (firstcall .and. RESTART_IN_PRESENT()) then
  lb_reading = .true.
  call RESTART_OPEN(lb_reading, lb_u)
  call LB_MODULES_IO(lb_u, lb_reading)
  include 'lb_restart_lake.inc'
  call RESTART_CLOSE(lb_reading, lb_u)
endif

#ifdef LB_DUMP
! Diagnostic: dump all module variables and LAKE locals at the start of the
! call whose saved step counter equals $LB_DUMP_STEP
if (lb_dump_step == -2) then
  call get_environment_variable('LB_DUMP_STEP', lb_env)
  lb_dump_step = -1
  if (len_trim(lb_env) > 0) read(lb_env,*) lb_dump_step
endif
if (lb_dump_step >= 0 .and. allocated(nstep_2d)) then
  if (nstep_2d(ix,iy) == lb_dump_step) then
    open(newunit=lb_u, file='lb_dump.bin', access='stream', form='unformatted', status='replace')
    call LB_DUMP_MODULES(lb_u)
    include 'lb_dump_lake.inc'
    close(lb_u)
    lb_dump_step = -1
  endif
endif
#endif

!BODY OF PROGRAM

! Getting the number of cores
num_ompthr = OMP_GET_NUM_PROCS()
! MPI parameters
parparams%comm3d = comm3d
parparams%rank_comm3d = rank_comm3d
parparams%coords = coords
parparams%parallel = parallel

!call TIMEC(0)

time_group%year  => year
time_group%month => month
time_group%day   => day
time_group%hour  => hour

gs%ix = ix
gs%iy = iy

dnx = nx - nx0 + 1
dny = ny - ny0 + 1


!if (firstcall)  then
!  if (calibrate_parameter) then
!    ! Calibration runs are performed only when ny = 1
!    if (npoint_calibr(1)*npoint_calibr(2) /= dnx) then
!      write(*,*) 'npoint_calibr(1)*npoint_calibr(2) /= dnx in LAKE: STOP'
!      STOP
!    else
!      ! Calibration of constants r0_methprod_phot and r0_oldorg2_star
!      calibr_par1 => r0_methprod_phot
!      calibr_par2 => r0_oldorg2_star
!      cost_function => febultot
!      
!!      calibr_par_min(1) = calibr_par1/2. ; calibr_par_max(1) = calibr_par1*2.
!      calibr_par_min(1) = 2.25d-8 ; calibr_par_max(1) = 3.d-8
!!      calibr_par_min(2) = calibr_par2/sqrt(2.) ; calibr_par_max(2) = calibr_par2*sqrt(2.)
!      calibr_par_min(2) = 6.d-11 ; calibr_par_max(2) = 8.d-11
!      step_calibr(1:2) = (calibr_par_max(1:2) - calibr_par_min(1:2))/ &
!      & real(npoint_calibr(1:2)-1)
!    endif
!  endif
!endif
!
!if (calibrate_parameter) then
!  if (mod(ix,npoint_calibr(1)) == 0) then
!    i = npoint_calibr(1)
!    j = ix/npoint_calibr(1)
!  else
!    i = mod(ix,npoint_calibr(1))
!    j = 1 + ix/npoint_calibr(1)
!  endif
!  calibr_par1(:) = calibr_par_min(1) + (i-1)*step_calibr(1)
!  calibr_par2    = calibr_par_min(2) + (j-1)*step_calibr(2)
!endif

!Checking if forcing data is read
if ( (.not.dataread) .and. PBLpar%par /= 0) then
  write(*,*) 'Atmospheric forcing is not available while PBLpar /= 0: STOP'
  STOP
endif

tempair = tempair1 - Kelvin0
humair = humair1
pressure = pressure1
uwind = uwind1
vwind = vwind1
zref  = zref1
precip = precip1
Sflux0 = Sflux01*(roa0/row0)
outpath = outpath1
cloud = cloud1
if (SurfTemp1 /= missing_value) then
  SurfTemp = SurfTemp1 - Kelvin0
else
  SurfTemp = missing_value
endif

hw_input = hw1
xlew_input = xlew1
cdmw_input = cdmw1
surfrad_input = surfrad1

if (kor_in == -999.d0) then
  kor = 2.d0*omega*sin(phi*pi/180.d0)
else
  kor = kor_in
endif

!extwatarr(:) = extwat
allocate(radflux(1:nbands))
if (nbands > 1) then
  forall (i=1:M+1) RadWater%extinct(i,1:nbands) = extwatbands(1:nbands)
  RadWater  %extinct(:,2) = extwat ! extwat is treated as extinction coefficient for PAR
else
  RadWater  %extinct(:,:) = extwat
endif
RadIce    %extinct(:,:) = extice
RadDeepIce%extinct(:,:) = extice
if (ifrad%par == 1) then
  longwave  = longwave1
  shortwave = max(shortwave1,0._ireals)
elseif (ifrad%par == 0) then
  longwave  = 0.e0_ireals
  shortwave = 0.e0_ireals
endif
! If atmospheric radiation is missing, net radiation to be used to compute this variable
if (longwave1 /= missing_value) then
  netrad = missing_value
else
  netrad = netrad1
endif

!Incident global shortwave radiation is computed by empirical formulae, if missing value is provided 
!in the LAKE arguments
if (shortwave1 == missing_value) then
  if (cloud1 /= missing_value) then
    shortwave = SHORTWAVE_RADIATION(time_group,phi,cloud/10.)
  else
    print*, 'Global solar radiaion and cloud fraction are not provided: STOP'
    STOP
  endif
endif

if (longwave1 == missing_value) then
  if (firstcall) write(*,*) 'Atmospheric radiation is missing in input file and will be calculated &
  &from net longwave radiation, net radiation or empirical formulae'
  if (cloud1 == missing_value) then
    if (firstcall) write(*,*) 'Cloudiness data is missing: atmospheric radation to be computed &
    &from net radiation' 
    if (netrad1 == missing_value) then
      STOP 'Net radiation data is missing!: STOP'
    endif
  else
    longwave = LONGWAVE_RADIATION(tempair,humair,pressure,cloud/10.)
  endif
endif

if (uwind == 0) uwind = minwind
if (vwind == 0) vwind = minwind
wind = sqrt(uwind**2+vwind**2)
!wind10 = wind*log(10./roughness0)/log(zref/roughness0)
!Control of the input atmospheric forcing
flag = .false.
if (tempair > 60. .or. tempair < -90. .or. (.not. VALUE_IS_FINITE(tempair) )) then
  print*, 'The air temperature ', tempair, 'is illegal: STOP'
  flag = .true.
elseif (abs(humair) > 1. .or. (.not. VALUE_IS_FINITE(humair) )) then
  print*, 'The air humidity ', humair, 'is illegal: STOP'
  flag = .true.
elseif (pressure > 110000 .or. pressure < 20000. .or. (.not. VALUE_IS_FINITE(pressure) )) then
  print*, 'The air pressure ', pressure, 'is illegal: STOP'
  flag = .true.
elseif (abs(uwind) > 200. .or. (.not. VALUE_IS_FINITE(uwind) )) then
  print*, 'The x-component of wind ', uwind, 'is illegal: STOP'
  flag = .true.
elseif (abs(vwind) > 200. .or. (.not. VALUE_IS_FINITE(vwind) )) then
  print*, 'The y-component of wind ', vwind, 'is illegal: STOP'
  flag = .true.
elseif (abs(longwave) > 1000. .or. (.not. VALUE_IS_FINITE(longwave) )) then
  print*, 'The longwave radiation ',longwave,'is illegal: STOP'
  flag = .true.
elseif (abs(shortwave) > 1400. .or. (.not. VALUE_IS_FINITE(shortwave) )) then
  print*, 'The shortwave radiation ', shortwave, 'is illegal: STOP'
  flag = .true.
elseif (abs(precip) > 1. .or. (.not. VALUE_IS_FINITE(precip) )) then
  print*, 'The atmospheric precipitation ',precip, &
  & 'is illegal: STOP'
  flag = .true.
endif
if (flag) then
  write(*,*) 'LAKE: atmospheric forcing is incorrect: Temperature = ', tempair, &
  & 'Humidity = ', humair, &
  & 'Pressure = ', pressure, &
  & 'Uwind = ', uwind, &
  & 'Vwind = ', vwind, &
  & 'Longwave = ', longwave, &
  & 'Shortwave = ', shortwave, &
  & 'Precip = ', precip
  STOP
endif

dt = dtl
dt05 = 0.5d0*dt
dt_keps = dt/real(nstep_keps%par)
dt_inv = 1.d0/dt
dt_inv05 = 0.5d0*dt_inv

! Logical, indicating multiple soil columns configuration
multsoil = (nsoilcols > 1 .and. depth_area(1,2) > 0. .and. soilswitch%par == 1)

if (init(ix,iy) == 0_iintegers) then
  ! Specification of initial profiles   
  
  rosoil(:) = 1200. ! Bulk soil density for initialization
  call INIT_VAR &
  ( M, Mice, ns, ms, ml, nsoilcols, ndatamax, &
  & init_T, skin%par, zero_model%par, &
  & h10, l10, hs10, ls10, tempair, Ts0, Tb0, Tm, Tbb0, &
  & Sals0, Salb0, us0, vs0, h_ML0, &
  & rosoil, rosdry, por, depth_area, ip, isp, &
  
  & flag_snow, flag_snow_init, itop, nstep, itherm, &
  & h1, l1, hs1, ls1, &
  & hx1, hx2, hy1, hy2, &
  & hx1t, hx2t, hy1t, hy2t, &
  & hx1ml, hx2ml, hy1ml, hy2ml, &
  & veg, snmelt, snowmass, cdmw, velfrict_prev, &
  & roughness, eflux0_kinem, Elatent, &
  & totalevap, totalmelt, totalprecip, totalwat, totalpen, &
  & time, dhwfsoil, dhw, dhw0, dhi, dhi0, dls0, &
  
  & E1, eps1, zsoilcols(1,ix,iy), Tsoil1, wi1, wl1, Sals1, &
  & rootss, qsoil, TgrAnn, qwater(1,1), &
  & oxyg(1,1), DIC(1,1), phosph, DOC, POCL, POCD, oxygsoil, Sal1, &
  & u1, v1, Tw1, Tskin, T_0dim, Eseiches, &
  & Ti1, salice, porice, Tis1, z_full, ddz, &
  & dz, T, wl, dens, lamw, &
  & dzeta_int, zsoil, pressure )

  Sals2(1:ns,1:nsoilcols) = Sals1(1:ns,1:nsoilcols) !Salinity is currently not calculated in lateral columns 
      
  !if (runmode%par == 2 .and. (uniform_depth .eqv. .false.)) then
  !  if (ix <= nx-1 .and. iy <= ny-1) then
  !    h1 = dep_2d(ix,iy)
  !  else
  !    h1 = dep_av
  !  endif  
  !  if (h1 <= 0) then
  !    print*, 'Negative or zero initial depth at ix=',ix, &
  !    & 'iy=',iy,'h1=',h1,':terminating program'
  !    STOP 
  !  endif
  !endif

endif

if (init(ix,iy) == 0_iintegers .and. icp == 2_iintegers .and. &
  & (.not. control_point_read) ) then
  ! Read control point for inital conditions of all lakes of the domain
  call CONTROL_POINT_IN(nx,nx0,nx_max,ny,ny0,ny_max,gs,parparams)
  print*, 'Control point is read'
  control_point_read = .true.
  !call CONTROL_POINT_OUT(year,month,day,hour, &
  !& nx,nx0,nx_max,ny,ny0,ny_max,gs,parparams,time)
endif


if (init(ix,iy) /= 0_iintegers .or. &
  & (init(ix,iy) == 0_iintegers .and. icp == 2_iintegers )) then
  call UPDATE_CURRENT_TIMESTEP ( &
  & ix, iy, dnx, dny, M, Mice, ns, ms, ml, nsoilcols, &
  & l1, h1, hx1, hx2, hy1, hy2, ls1, hs1, &
  & hx1t, hx2t, hy1t, hy2t, &
  & hx1ml, hx2ml, hy1ml, hy2ml, &
  & gradpx_tend, gradpy_tend, &
  & u1, v1, &
  & E1, eps1, k_turb_T_flux, &
  & Tsoil1, Sals1, wi1, wl1, &
  & Tw1, Sal1, lamw, &
  & Tskin(1), &
  & Ti1, Tis1, &
  & dz, T, wl, dens, &
  & qwater(1,1), qsoil, &
  & oxyg(1,1), oxygsoil, &
  & DIC(1,1), DOC, POCL, POCD, phosph, &
  & snmelt, snowmass, &
  & cdmw, &
  & time, &
  & dhwfsoil, &
  & Elatent, &
  & dhw, dhw0, &
  & dhi, dhi0, dls0, &
  & velfrict_prev, &
  & roughness, &
  & eflux0_kinem, &
  & tot_ice_meth_bubbles, &
  & febul0, Eseiches, salice, porice, &
  
  & flag_snow, flag_snow_init, &
  & itop, &
  & nstep, i_maxN, itherm )
endif

!Checking temperature for NaNs and extreme values
do i = 1, M+1
  if (Tw1(i) /= Tw1(i) .or. abs(Tw1(i)) > 100.) then
    print*, 'Tw1(i) = ', Tw1(i), ', nstep = ', nstep, &
    & ', init(ix,iy) = ', init(ix,iy), ', ix = ', ix, ', iy = ', iy
    print*, 'Other coordinatess: ', nx, ny, nx0, ny0, nx_max, ny_max
    print*, 'Terminating LAKE model ...'
    stop
  endif
enddo

!At the next timestep after control point output, the control point may be written again
!Writing the control point for all lakes of the domain
!Note: the lakes state is written for the beginning of the timestep
if (icp == 1_iintegers .and. (nstep /= nstep_cp_written)) then
  !print*, 'Before CP output: ', icp, nstep, nstep_cp_written, rank_comm3d, ix, iy
  call CONTROL_POINT_OUT(year,month,day,hour, &
  & nx,nx0,nx_max,ny,ny0,ny_max,gs,parparams,time)
  nstep_cp_written = nstep
endif


! Updating bathymetry 
call BATHYMETRY(M,Mice,ns,ix,iy,ndatamax,month,day,lakeform,hour,dt, &
               & depth_area,area_lake,cellipt, &
               & multsoil,trib_inflow,dhwtrib,vol,botar)
! Tributary inflow (may be overwritten later in BATHYMETRY and/or in TRIBTEMP)
if (trib_inflow(1) /= -9999.) then
  dhwtrib = trib_inflow(1)*dt/area_int(1)
else
  dhwtrib = (h10 - h1)/10. ! 10 - arbitrary value, the "time" of depth damping towards initial depth
endif

time = time + dt
nstep = nstep + 1

layer_case = 1
if (h1 >  0  .and. l1 >  0) layer_case = 2 
if (h1 == 0  .and. l1 >  0) layer_case = 3
if (h1 == 0  .and. l1 == 0) layer_case = 4

if (layer_case == 1 .and. &
& WATER_FREEZE_MELT(Tw1(1), 0.5*ddz(1)*h1, &
& Meltpnt(Sal1(1),preswat(1),nmeltpoint%par), +1) .and. &
& h1 - min_ice_thick * roi/row0 > min_water_thick) then
  layer_case = 2
endif 

if (layer_case == 3 .and. &
& WATER_FREEZE_MELT(Ti1(Mice+1), 0.5*ddzi(Mice)*l1, Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par), -1) .and. &
& l1 - min_water_thick * row0_d_roi > min_ice_thick) then
  layer_case = 2
endif  

if (layer_case == 3 .and. &
& WATER_FREEZE_MELT(Ti1(1), 0.5*ddzi(1)*l1, Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par), -1) .and. &
& l1 - min_water_thick * row0_d_roi > min_ice_thick .and. flag_snow == 0) then
  h1 = min_water_thick ; dhw = 0.; dhw0 = 0.
  Tw1 = Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par) + T_phase_threshold 
  Sal1 = 1.d-5
  ls1 = l1 - min_water_thick * row0_d_roi
  Tis1 = Ti1
  l2 = 0
  Ti2 = 0
  porice(:) = poricebot * saltice%par
  salice(:) = porice(:) * Sal1(1) * row0
  if (ls1 < 0.009999) print*, 'Thin layer! - ls'
  layer_case = 1 
endif

  
if1: IF (layer_case == 1) THEN 

!---------------------------- CASE 1: LIQUID WATER LAYER AT THE TOP ("Summer")-----------------------


!Check the bottom layer water temperature
x = Meltpnt(Sal1(M+1),preswat(M+1),nmeltpoint%par)
if (WATER_FREEZE_MELT(Tw1(M+1), 0.5*ddz(M)*h1, x, +1) .and. &
& ls1 == 0 .and. h1 - min_ice_thick * roi/row0 > min_water_thick) then
  ls1 = min_ice_thick ; dls = 0.; dls0 = 0.
  Tis1 = x - T_phase_threshold
  h1 = h1 - min_ice_thick*roi/row0
endif   

! Updating bathymetry 
call BATHYMETRY(M,Mice,ns,ix,iy,ndatamax,month,day,lakeform,hour,dt, &
                & depth_area,area_lake,cellipt, &
                & multsoil,trib_inflow,dhwtrib,vol,botar)

!Check the liquid temperature to be above melting point
do i = 2, M
  Tw1(i) = max(Tw1(i),Meltpnt(Sal1(i),preswat(i),nmeltpoint%par))
enddo

!Calculation of water density profile at the current timestep
call DENSITY_W(M,eos%par,lindens%par,Tw1,Sal1,preswat,row)
call MIXED_LAYER_CALC(row,ddz,ddz2,dzeta_int,dzeta_05int,h1,M,i_ML,H_mixed_layer,maxN)

!The salinity flux at the surface are passed to
!TURB_DENS_FLUX as zeros since they are currently      ______
!not taken into account in calculation of density flux w'row'	
 
turb_density_flux = TURB_DENS_FLUX(eflux0_kinem,0.e0_ireals,Tw1(1),0.e0_ireals)
Buoyancy0 = g/row0*turb_density_flux

! Updating tributaries data and adding inflow of all substances
if (tribheat%par > 0 .and. .not. (tribheat%par == 4 .and. trib_inflow(1) == -9999.) ) then
  call TRIBTEMP(time,dt,h1,h10,dhwtrib,trib_inflow, &
  &             z_full,area_int,area_half,gsp,gas,wst,spinup_done,ix,iy,nx,ny)
  ! Adding tendency due to mean vertical velocity
  call VERTADV_UPDATE(M,dt,h1,gsp,bathymwater,wst,gas)
endif

!if (massflux%par == 1) then
!!Updating density profile for massflux convection
!!Updating z-grid properties 
!  do i = 1, M
!    dz_full (i) = ddz(i)  *h1
!    z_half  (i) = dzeta_05int(i)*h1
!  enddo
!  call MASSFLUX_CONVECTION( &
!  & nstep,dt,z_half,z_full,dz_full, &
!  & turb_density_flux,eflux0_kinem, &
!! The latent heat flux and salinity flux at the surface are passed as zeros,
!! as far as currently they are not taken 
!! into account in massflux conection parameterization
!  & 0.e0_ireals,0.e0_ireals, &
!! The surface wind stress components are passed as zeros,
!! as far as currently they are not taken 
!! into account in massflux conection parameterization
!  & 0.e0_ireals,0.e0_ireals, &
!  & u1,v1,w1,E1(1:M), &
!  & u1,v1,w1,row,Tw1,Sal1, &
!  & PEMF,PDENS_DOWN,PT_DOWN,PSAL_DOWN,zinv,1,M)
!! Debugging purposes only
!! PEMF = 0.e0_ireals
!! call TEMP_MASSFLUX(PEMF,PT_DOWN,ddz,h1,dt,Tw2,T_massflux)
!  do i = 1, M
!    pt_down_f(i) = 0.5d0*(PT_DOWN(i) + PT_DOWN(i+1))
!  enddo
!endif

! The solar radiation, absorbed by the water surface
if (varalb%par == 0) then
  Erad = shortwave*(1-albedoofwater)
elseif (varalb%par == 1) then
  Erad = shortwave* &
  & (1 - WATER_ALBEDO( SINH0(year,month,day,hour,phi) ) ) ! Note: the time here must be a local one
endif  ! time, not UTC

!Radiation fluxes in layers
x = 1._ireals - sabspen*skin%par
!Partitioning of shortwave energy between wavelength bands
forall(i=1:nbands) radflux(i) = fracbands(i)*Erad*(1 - sabs0)*x
call RadWater%RAD_UPDATE(ddz05(0:M)*h1,radflux) !(/Erad*(1 - sabs0)*x/))
! Calculating photic zone depth (<1% of surface irradiance)
if (RadWater%integr(1) > 0) then
  H_photic_zone = h1
  do i = 1, M
    if (RadWater%integr(i)/RadWater%integr(1) < photic_threshold) then
      H_photic_zone = z_full(i)
      exit
    endif
  enddo
else
  H_photic_zone = 0.
endif
if (ls1 /= 0.e0_ireals) then
  ! xx - radiation penetrated through the water - deep ice interface
  forall (i=1:nbands) radflux(i) = RadWater%flux(M+1,i)*(1-albedoofice)
  call RadDeepIce%RAD_UPDATE(ddzi05(0:Mice)*ls1, radflux) !(/xx/))
endif

!Calculation of the layer's thickness increments
if (ls1 == 0) then
  ice = 0; snow = 0; water = 1; deepice = 0
  dhwp = precip*dt
  dhwe = - Elatent/(Lwv*row0)*dt
  dhw = dhwp + dhwe + dhwfsoil + dhwtrib
  dhw0 = dhwp + dhwe
  dls = 0.
else
  ice = 0; snow = 0; water = 1; deepice = 1
  flux1 = - lamw(M)*(Tw1(M+1)-Tw1(M))/(ddz(M)*h1) + RadWater%integr(M) + &
  & cw_m_row0*(dhw-dhw0)*(Tw1(M+1)-Tw1(M))/(2.*dt) + &
  & cw_m_row0*PEMF(M)*(pt_down_f(M)-0.5d0*(Tw1(M+1)+Tw1(M)) )
  flux2 = - lami*(Tis1(2)-Tis1(1))/(ddzi(1)*ls1) + RadDeepIce%integr(1) + &
  & ci_m_roi*dls0*(Tis1(2)-Tis1(1))/(2.*dt)
  dhwls = dt*(flux1 - flux2)/(row0_m_Lwi)
  !dhwls = min(max(dt*(flux1 - flux2)/(row0_m_Lwi),-ddz(M)*h1),ddzi(1)*ls1*roi_d_row0)
!  dhwls = -lamw(M)*dt/(ddz(M)*h1*row0_m_Lwi)*(Tw1(M+1) - Tw1(M))+
!&  lami*dt/(ddz(1)*ls1*row0_m_Lwi)*(Tis1(2) - Tis1(1))+
!&  (1-albedoofice)*Erad*(1-sabs)*exp(-extwat*h1)/(row0_m_Lwi)*dt 
  dls = - dhwls*row0_d_roi
  dls0 = dls
  dhwp = precip*dt
  dhwe = - Elatent/(Lwv*row0)*dt
  dhw = dhwp + dhwe + dhwfsoil + dhwls + dhwtrib
  dhw0 = dhwp + dhwe
endif

!Calculation of water current's velocities and turbulent characteristics
if (Turbpar%par == 2 .or. Turbpar%par == 5) then
  call MOMENTUM_SOLVER(ix, iy, dnx, dny, ndatamax, &
  & year, month, day, hour, kor, a_veg, c_veg, h_veg, &
  & alphax, alphay, dt, b0, tau_air, tau_i, tau_gr, tau_wav, fetch, depth_area)
endif

if (soilswitch%par == 1) then
  call SOIL_COND_HEAT_COEF(nsoilcols)
endif
if (multsoil) then 
! Calculation of heat sources from heat fluxes from mutliple soil columns
  iwork = (/1,0/)
  call SOILCOLSTEMP(nveclen,gs,gsp,dt,ls,ftot,ch4_pres_atm,ddz,ddzi,zsoilcols(1,ix,iy), &
                   & wst,RadWater%integr,a,b,c,d,add_to_winter, &
                   & bathymwater,bathymice, &
                   & bathymdice,bathymsoil(1,ix,iy), &
                   & soilflux(1,ix,iy),fdiffbot,iwork)
  call LATERHEAT(ix,iy,gs,ls, &
  & bathymwater,bathymice,bathymdice,bathymsoil, &
  & gsp,soilflux(1,ix,iy),.false.,lsh)
endif
! Calculation of the whole temperature profile
call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
& RadWater, RadIce, SurfTemp, fetch, dt)
if (soilswitch%par == 1) then
  call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
endif

! Diagnostic calculations
do i = 1, M
  k_turb_T_flux(i) = - 0.5*lamw(i)/(cw_m_row0) * &
  & ( Tw2(i+1) + Tw1(i+1) - Tw2(i) - Tw1(i)) / (ddz(i)*h1)
  T_massflux(i) = PEMF(i)*(pt_down_f(i) - 0.5d0*(Tw2(i+1) + Tw2(i)))
enddo

!Assuming all dissolved species diffuse with the same molecular diffusivity as salinity
lamsal(:)    = (lamw(:) - lamw0)/cw_m_row0 * alsal    + lamsal0
lammeth(:)   = (lamw(:) - lamw0)/cw_m_row0 * almeth   + lamsal0
lamoxyg(:)   = (lamw(:) - lamw0)/cw_m_row0 * aloxyg   + lamsal0
lamcarbdi(:) = (lamw(:) - lamw0)/cw_m_row0 * alcarbdi + lamsal0
lampart  (:) = (lamw(:) - lamw0)/cw_m_row0 * alcarbdi !no molecular diffusion for particles

! Salinity diffusion in water and soil
CALL S_DIFF(dt,ls,salice)

! Oxygen model precedes methane module, in order sedimentary 
! oxygen demand to be available for the latter
call CHLOROPHYLLA(z_full, H_mixed_layer, H_photic_zone, extwat, RadWater, gas, M, Chl_a, itroph)
call OXYGEN_PRODCONS(gs, gsp, wst, bathymsoil, gas, area_int, area_half, ddz05, &
& h1, dt, Tsoil3, Chl_a, dzs, por, oxygsoil, itroph, &
& prodox, resp, bod, sod, sodbot)

if (soilswitch%par == 1) then
  ! Calculation of methane in all soil columns, except for the lowest one
  if (multsoil) then
    iwork = (/0,1/)
    call SOILCOLSTEMP(nveclen,gs,gsp,dt,ls,ftot,ch4_pres_atm,ddz,ddzi,zsoilcols(1,ix,iy), &
                     & wst,RadWater%integr,a,b,c,d,add_to_winter, &
                     & bathymwater,bathymice, &
                     & bathymdice,bathymsoil(1,ix,iy), &
                     & soilflux(1,ix,iy), fdiffbot, iwork)
    methsoilflux(1:nsoilcols-1) = fdiffbot(1:nsoilcols-1)
  endif
  call BUBBLE_BLOCK
  ! Calculation of methane sources from methane fluxes from mutliple soil columns
  if (multsoil) then 
    !call LATERHEAT(ix,iy,gs,ls, &
    !& bathymwater,bathymice,bathymdice,bathymsoil, &
    !& gsp,soilflux(1,ix,iy),.false.,lsh)
    call LATERHEAT(ix,iy,gs,ls, &
    & bathymwater,bathymice,bathymdice,bathymsoil, &
    & gsp,methsoilflux,.true.,lsm)
    call LATERHEAT(ix,iy,gs,ls, &
    & bathymwater,bathymice,bathymdice,bathymsoil, &
    & gsp,co2soilflux, .true.,lsc)
    call LATERHEAT(ix,iy,gs,ls, &
    & bathymwater,bathymice,bathymdice,bathymsoil, &
    & gsp,o2soilflux,  .true.,lso)
  endif
endif
gs%isoilcol = nsoilcols
call METHANE &
& (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
& fbbleflx_ch4(0,nsoilcols),wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
& rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
& ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false., &
& lsm, bathymwater, &
& fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
& febul_lake_bot, febul_lake_surf, &
& plant_sum,bull_sum,oxid_sum,rprod_sum, &
& anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
& ice_meth_oxid_total, &
& h_talik,tot_ice_meth_bubbles,add_to_winter)
qmethane(1:M+1   ) = qwater(1:M+1,2)
qmethane(M+2:M+ns) = qsoil (2:ns,nsoilcols)
work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 

call ADDOXPRODCONS(M,i_maxN,H_mixed_layer,H_photic_zone,prodox,resp,bod,sod,RadWater,dt,ls,gas)
call OXYGEN (gs, Tw2, fbbleflx_o2_sum, fbbleflx_o2(0,nsoilcols), lamoxyg, gsp, fracb0, pressure, wind10, &
& o2_pres_atm, ls, bathymwater, lso, dt, febul0(nsoilcols), eps1(1), sodbot, oxyg, soilswitch%par, fo2)
call CARBON_DIOXIDE (gs, Tw2, fbbleflx_co2_sum, fbbleflx_co2(0,nsoilcols), fco2ebul_bot, fco2ebul_surf, &
& lamcarbdi, gsp, bathymwater, lsc, fracb0, &
& pressure, wind10, co2_pres_atm, febul0(nsoilcols), ls, dt, eps1(1), sodbot, gas, soilswitch%par, fco2)
call PHOSPHORUS(M,dt,ls,gsp,bathymwater,gas,lsp,lamcarbdi)
call METHANE_OXIDATION(M, i_maxN, dt, ddz, h1, bathymwater, oxyg(1,2), qwater(1,2), DIC(1,2), Tw2, &
& methoxid,qwateroxidtot, qwateroxidML)
if (oxicmethprod%par == 1) call METHANE_OXIC_PRODUCTION(dt,H_mixed_layer,extwat,Chl_a,methoxid,prodox,gs,gsp,gas)

!Calculation of water density profile at the next timestep
call DENSITY_W(M,eos%par,lindens%par,Tw2,Sal2,preswat,row2)

if (Turbpar%par == 2) then
  do i = 1, nstep_keps%par
    call K_EPSILON(ix,iy,dnx,dny, year, month, day, hour, &
    & kor, a_veg, c_veg, h_veg, dt_keps, &
    & b0, tau_air, tau_wav, tau_i, tau_gr, roughness, fetch) 
  enddo
else
! Calculation of the of eddy difusivity for temperature
  SELECT CASE (Turbpar%par)
    CASE (5)
      call RICHARDSON_NUMBER(M,u2,v2,ddz,h1+dhw,row2,Ri)
      allocate(work1(1:M))
      call EDDY_DIFF_MOST_RICH(M,Ri,2_iintegers,u2,v2,dzeta_05int,ddz,h1+dhw,work1,KT)
      deallocate(work1)
      KT(:) = cw_m_row0*KT(:)
    CASE (8)
      ! Hostetler formulation
      call ED_TEMP_HOSTETLER &
      & (wind,velfrict_prev,zref,phi,z_full,row,KT,M)
      KT(:) = cw_m_row0*KT(:)
      call UnDenGrMix(M,eos%par,lindens%par,ddz*h1,dzs(1),csoil(1),rosoil(1),wst,gas)
    CASE (9)
      ! Modified Hostetler formulation for shallow lake
      call ED_TEMP_HOSTETLER2 &
      & (wind,velfrict_prev,zref,phi,z_full,row,KT,M)
      KT(:) = cw_m_row0*KT(:)
    CASE (1)
      do i = 1, M
        KT(i) = (k2(i)-niu_wat)*cw_m_row0 
      enddo
    CASE DEFAULT
      KT = lamTM*k2*cw_m_row0 
  ENDSELECT 
endif

!Background vertical diffusivity
select case(backdiff%par)
case(0)
  lamw_back(:) = 0.
case(1)
  !Background heat conductivity following (Fang and Stefan, 1996)
  call DIFFMIN_HS(area_lake,wst,gsp,ls,M) 
case(2)
  !Background heat conductivity from KPP model (e.g., Zhang et al., HESS, 2019)
  call DIFFMIN_KPP(wst,gsp,ls,M) 
case(3)
  !Constant background vertical diffusivity 
  lamw_back(:) = max(backdiff0%par,0._ireals) 
end select

! Heat conductance at the next timestep
do i = 1, M
  lamw(i) = lamw0 + KT(i) + lamw_back(i) 
enddo

!The scales of turbulence
call TURB_SCALES(gsp,ls, bathymwater, wst, RadWater, &
& k_turb_T_flux, T_massflux, row, eflux0_kinem, &
& turb_density_flux, Buoyancy0, tau_air-tau_wav, kor, i_maxN, H_mixed_layer,maxN, w_conv_scale, &
& T_conv_scale, Wedderburn, LakeNumber, Rossby_rad, ThermThick, ReTherm, RiTherm, trb)

if (zero_model%par == 1) then
! Note that zero model is impemented currently only for
! 1. one-point simulation
! 2. open water conditions
! Not operational, when net radiation is given instead of atmospheric radiation
  call ZERODIM_MODEL &
  & (h1, dt, &
  & shortwave, longwave, tempair, humair, pressure, &
  & uwind, vwind, zref, hw_input, xlew_input, cdmw_input, surfrad_input, &
  & cloud, botflux, T_0dim)
endif
  
h2 = h1 + dhw
ls2 = ls1 + dls
l2 = 0.

ENDIF if1   

if2: IF (layer_case == 2) THEN

!---------------------------CASE 2: WATER AND ICE ("Winter")------------------------------

!call TIMEC(1)

! Creation of the initial thin ice layer
if (l1 == 0.) then
  l1 = min_ice_thick; dhi = 0.; dhi0 = 0.; dhilow = 0.; dhihigh = 0.
  h1 = h1 - min_ice_thick * roi/row0
  x = Meltpnt(Sal1(1),0.e0_ireals,nmeltpoint%par)
  if (tempair < 0.) then    
    do i = 1, Mice+1
      Ti1(i) = tempair + float(i-1)/float(Mice) * (x - tempair)
    enddo 
  else 
    Ti1(:) = x - T_phase_threshold
  endif
  porice(:) = poricebot * saltice%par  !Initial ice salinity
  salice(:) = porice(:) * Sal1(1) * row0  !Initial ice porosity
end if 

!Check for water temperature to be higher than melting point
do i = 2, M
  Tw1(i) = max(Tw1(i),Meltpnt(Sal1(i),preswat(i),nmeltpoint%par))
enddo

!Creation of the initial thin water layer
if (h1 == 0.) then
  h1 = min_water_thick; dhw = 0.; dhw0 = 0.
  Sal1(:) = 1.d-5
  Tw1(:) = Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par) + T_phase_threshold
  l1 = l1 - min_water_thick * row0_d_roi
end if

!Creation of the initial thin bottom ice layer
x = Meltpnt(Sal1(M+1),preswat(M+1),nmeltpoint%par)
if (WATER_FREEZE_MELT(Tw1(M+1), 0.5*ddz(M)*h1, x, +1) .and. &
& ls1 == 0 .and. h1 - min_ice_thick * roi/row0 > min_water_thick) then
  ls1 = min_ice_thick; dls = 0.; dls0 = 0.
  Tis1 = x - T_phase_threshold
  h1 = h1 - min_ice_thick * roi/row0
endif

! Updating bathymetry 
call BATHYMETRY(M,Mice,ns,ix,iy,ndatamax,month,day,lakeform,hour,dt, &
                & depth_area,area_lake,cellipt, &
                & multsoil,trib_inflow,dhwtrib,vol,botar)


!Calculation of water density profile at the current timestep
call DENSITY_W(M,eos%par,lindens%par,Tw1,Sal1,preswat,row)

! Note: the salinity effects and partial ice coverage 
! are currently neglected in density flux at the ice-water interface
eflux0_kinem_water = - lamw(1)*(Tw1(2)-Tw1(1))/(ddz(1)*h1*cw_m_row0) ! The first-order approximation
turb_density_flux = TURB_DENS_FLUX(eflux0_kinem_water,0.e0_ireals,Tw1(1),0.e0_ireals)
Buoyancy0 = g/row0*turb_density_flux

!Currently the EDMF parameterization of convection is not implemented in the code
!during 'winter' (if the ice layer is present over
!water layer). This seems to be reasonable, since the temperature profile
!is stable in this case - until spring convection under thin ice
!develops.

PEMF = 0.e0_ireals
pt_down_f = 0.e0_ireals

! Updating tributaries data and adding inflow of all substances
if (tribheat%par > 0 .and. .not. (tribheat%par == 4 .and. trib_inflow(1) == -9999.)) then
  call TRIBTEMP(time,dt,h1,h10,dhwtrib,trib_inflow, &
  &             z_full,area_int,area_half,gsp,gas,wst,spinup_done,ix,iy,nx,ny)
  ! Adding tendency due to mean vertical velocity
  call VERTADV_UPDATE(M,dt,h1,gsp,bathymwater,wst,gas)
endif

! Specification of volumetric heat capacity and heat conductance for ice
if   (saltice%par == 0) then
  !Pure ice
  ci_m_roi_v(:) = ci_m_roi
  lami_v    (:) = lami
elseif (saltice%par == 1) then
  !Ice with por * saltwater%pares filled with water
  do i = 1, Mice+1
    ci_m_roi_v(i) = ci*roi*(1. - porice(i)) + cw*row0*porice(i)
  enddo
  do i = 1, Mice
    x = 0.5*(porice(i) + porice(i+1))
    lami_v(i)      = lami  *(1. - x) + lamw0  *x
  enddo
endif


!CASE 2.1: WATER, ICE AND SNOW 

if (flag_snow == 1) then

  call MIXED_LAYER_CALC(row,ddz,ddz2,dzeta_int,dzeta_05int,h1,M,i_ML,H_mixed_layer,maxN)

! Radiation fluxes in physical layers
  SR_botsnow = shortwave*(1-albedoofsnow)
  if (flag_snow_init /= 1) then
    do i = itop, ms-1
      SR_botsnow = SR_botsnow*EXTINCT_SNOW(dens(i))**dz(i)
    enddo
  endif
  forall (i=1:nbands) radflux(i) = fracbands(i)*SR_botsnow ! assuming the spectrum does not change in snow
  call RadIce  %RAD_UPDATE(ddzi05(0:Mice)*l1, radflux) !(/SR_botsnow/))
  forall (i=1:nbands) radflux(i) = RadIce%flux(Mice+1,i)
  call RadWater%RAD_UPDATE(ddz05 (0:M   )*h1, radflux) !(/RadIce%flux(Mice+1,1)/))
! Calculating photic zone depth (<1% of surface irradiance)
  if (RadWater%integr(1) > 0) then
    H_photic_zone = h1
    do i = 1, M
      if (RadWater%integr(i)/RadWater%integr(1) < photic_threshold) then
        H_photic_zone = z_full(i)
        exit
      endif
    enddo
  else
    H_photic_zone = 0.
  endif
  if (ls1 /= 0.e0_ireals) then
    forall (i=1:nbands) radflux(i) = RadWater%flux(M+1,i)
    call RadDeepIce%RAD_UPDATE(ddzi05(0:Mice)*ls1, radflux) ! (/RadWater%flux(M+1,1)/))
  endif


! Calculation of the layer's thickness increments
  flux1 = - lami_v(Mice)*(Ti1(Mice+1)-Ti1(Mice))/(ddzi(Mice)*l1) + RadIce%integr(Mice) + &
  & ci_m_roi_v(Mice+1)*(dhi-dhi0)*(Ti1(Mice+1)-Ti1(Mice))/(2.d0*dt)
  flux2 = - lamw(1)*(Tw1(2)-Tw1(1))/(ddz(1)*h1) + RadWater%integr(1) + &
  & cw_m_row0*dhw0*(Tw1(2)-Tw1(1))/(2.*dt) + &
  & cw_m_row0*PEMF(1)*(pt_down_f(1)-0.5d0*(Tw1(2)+Tw1(1)) )
  ! Limiting the freezing/melting rate by the thicknesses of water and ice
  dhwlow = min(max(dt*(flux1 - flux2)/(row0_m_Lwi),-ddz(1)*h1),ddzi(Mice)*l1*roi_d_row0)
  dhilow = - dhwlow*row0_d_roi
  if (saltice%par == 1) then
    dhwlow = dhwlow * (1. + porice(Mice+1)*row0_d_roi / (1. - porice(Mice+1)) )
    dhilow = dhilow * (1. / (1. - porice(Mice+1)) )
  endif
  x = Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par)
  ! Melting point is that of fresh water, because it is pure ice that melts at the ice top
  if (Ti1(1) > x + T_phase_threshold) then 
    dhwhigh = (Ti1(1) - x - T_phase_threshold) * &
    & ci_m_roi_v(1)*l1*ddzi(1)*0.5d0/(row0_m_Lwi)
    dhihigh = - dhwhigh*row0_d_roi
    Ti1(1) = x + T_phase_threshold
    if (saltice%par == 1) then
      dhwhigh = dhwhigh * (1. + porice(1)*row0_d_roi / (1. - porice(1)) )
      dhihigh = dhihigh * (1. / (1. - porice(1)) )
    endif
  else
    dhwhigh = 0.e0_ireals
    dhihigh = 0.e0_ireals
  endif
  !Treatment of water freezing at the snow-ice interface
  !Porosity and salinity assumed zero at the ice top, so this parameterization is valid
  !for fresh ice only.
  !Freezing snowmelt
  call SNOW_COND_HEAT_COEF() !Needed for cs update
  y = 0.5*(cs(ms)*dens(ms)*dz(ms) + ci_m_roi_v(1)*ddzi(1)*l1)! Heat capacity of snow-ice interface
  dTmax = max((1. - snmeltwat)*snmelt*dt*row0*Lwi/y, 0._ireals)
  dTisnmelt = min(dTmax, x + T_phase_threshold - Ti1(1)) ! x -- melting point!
  dTisnmelt = max(dTisnmelt, 0._ireals)
  fracsn = dTisnmelt/(dTmax + small_value) !Fraction of snowmelt, frozen at the ice top
  snmeltice = dTmax*y/(row0*Lwi)*fracsn
  dhihigh = dhihigh + snmeltice*row0_d_roi ! Adding snowmelt, frozen at the top of ice

  !Assessing impoundment of ice from above by water penetrated through cracks
  !Porosity and salinity assumed zero at the ice top (fresh ice is assumed!)
  dTmax = max((snowmass + l1*(roi - row0))*Lwi/y, 0._ireals)
  dTiimp = min(dTmax, x + T_phase_threshold - Ti1(1) - dTisnmelt) ! x -- melting point!
  dTiimp = max(dTiimp, 0._ireals)
  fracimp = dTiimp/(dTmax + small_value)
  dhiimp = icetopfr * dTmax*y/(Lwi*row0)*fracimp
  dhihigh = dhihigh + dhiimp*row0_d_roi
  dhwhigh = dhwhigh - dhiimp

  !The showmelt water reaching the water column
  dhwsnmelt = snmeltwat*snmelt*dt + (1. - snmeltwat)*snmelt*dt*(1. - fracsn)
  if (ls1 /= 0) then
    ice = 1; snow = 1; water = 1; deepice = 1
    flux1 = - lamw(M)*(Tw1(M+1)-Tw1(M))/(ddz(M)*h1) + RadWater%integr(M) + &
    & cw_m_row0*(dhw-dhw0)*(Tw1(M+1)-Tw1(M))/(2.d0*dt) + &
    & cw_m_row0*PEMF(M)*(pt_down_f(M)-0.5d0*(Tw1(M+1)+Tw1(M)) )
    flux2 = - lami*(Tis1(2)-Tis1(1))/(ddzi(1)*ls1) + RadDeepIce%integr(1) + &
    & ci_m_roi*dls0*(Tis1(2)-Tis1(1))/(2.d0*dt)
    dhwls = dt*(flux1 - flux2)/(row0_m_Lwi)
    !dhwls = min(max(dt*(flux1 - flux2)/(row0_m_Lwi),-ddz(M)*h1),ddzi(1)*ls1*roi_d_row0)
    dls = - dhwls*row0_d_roi
    dhw = dhwhigh + dhwlow  + dhwfsoil + dhwls + dhwtrib
    dhw = dhw + dhwsnmelt
    dhw0 = dhwlow + dhwhigh 
    dhw0 = dhw0 + dhwsnmelt
    dls0 = dls
  else
    ice = 1; snow = 1; water = 1; deepice = 0
    dls = 0.
    dhwls = 0.
    dhw =  dhwhigh + dhwlow + dhwfsoil + dhwtrib 
    dhw = dhw + dhwsnmelt
    dhw0 = dhwlow + dhwhigh
    dhw0 = dhw0 + dhwsnmelt
  endif
  dhi = dhilow + dhihigh 
  dhi0 = dhihigh

  if (soilswitch%par == 1) then
    call SOIL_COND_HEAT_COEF(nsoilcols)
  endif
  ! Calculation of heat sources from heat fluxes from mutliple soil columns
  if (multsoil) then
    iwork = (/1,0/)
    call SOILCOLSTEMP(nveclen,gs,gsp,dt,ls,ftot,ch4_pres_atm,ddz,ddzi,zsoilcols(1,ix,iy), &
                     & wst,RadWater%integr,a,b,c,d,add_to_winter, &
                     & bathymwater,bathymice, &
                     & bathymdice,bathymsoil(1,ix,iy), &
                     & soilflux(1,ix,iy), fdiffbot, iwork)
    call LATERHEAT(ix,iy,gs,ls, &
    & bathymwater,bathymice,bathymdice,bathymsoil, &
    & gsp,soilflux(1,ix,iy),.false.,lsh)
  endif
  ! Calculation of the whole temperature profile 
  call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
  & RadWater, RadIce, SurfTemp, fetch, dt)
  if (soilswitch%par == 1) then
    call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
  endif

  !Assuming all dissolved species diffuse with the same molecular diffusivity as salinity
  lamsal(:)    = (lamw(:) - lamw0)/cw_m_row0 * alsal    + lamsal0
  lammeth(:)   = (lamw(:) - lamw0)/cw_m_row0 * almeth   + lamsal0
  lamoxyg(:)   = (lamw(:) - lamw0)/cw_m_row0 * aloxyg   + lamsal0
  lamcarbdi(:) = (lamw(:) - lamw0)/cw_m_row0 * alcarbdi + lamsal0
  lampart  (:) = (lamw(:) - lamw0)/cw_m_row0 * alcarbdi !no molecular diffusion for particles

  ! Salinity profile in water and soil
  call S_DIFF(dt,ls,salice)
  if (saltice%par == 1) call UPDATE_ICE_SALINITY(dt,ls,wst)
  if (soilswitch%par == 1) then
    ! Calculation of methane in all soil columns, except for the lowest one
    if (multsoil) then
      iwork = (/0,1/)
      call SOILCOLSTEMP(nveclen,gs,gsp,dt,ls,ftot,ch4_pres_atm,ddz,ddzi,zsoilcols(1,ix,iy), &
                       & wst,RadWater%integr,a,b,c,d,add_to_winter, &
                       & bathymwater,bathymice, &
                       & bathymdice,bathymsoil(1,ix,iy), &
                       & soilflux(1,ix,iy), fdiffbot, iwork)
      methsoilflux(1:nsoilcols-1) = fdiffbot(1:nsoilcols-1)
    endif
  endif
  call SNOWTEMP(ix,iy,dnx,dny,year,month,day,hour,snowmass, &
  & snowmass_init,a,b,c,d,Temp,phi,fetch,dt)

  ! Oxygen model precedes methane module, in order sedimentary 
  ! oxygen demand to be available for the latter
  call CHLOROPHYLLA(z_full, H_mixed_layer, H_photic_zone, extwat, RadWater, gas, M, Chl_a, itroph)
  call OXYGEN_PRODCONS(gs, gsp, wst, bathymsoil, gas, area_int, area_half, ddz05, &
  & h1, dt, Tsoil3, Chl_a, dzs, por, oxygsoil, itroph, &
  & prodox, resp, bod, sod, sodbot)

  if (soilswitch%par == 1) then
    call BUBBLE_BLOCK
    ! Calculation of methane sources from methane fluxes from mutliple soil columns
    if (multsoil) then 
      call LATERHEAT(ix,iy,gs,ls, &
      & bathymwater,bathymice,bathymdice,bathymsoil, &
      & gsp,methsoilflux,.true.,lsm)
      call LATERHEAT(ix,iy,gs,ls, &
      & bathymwater,bathymice,bathymdice,bathymsoil, &
      & gsp,co2soilflux, .true.,lsc)
      call LATERHEAT(ix,iy,gs,ls, &
      & bathymwater,bathymice,bathymdice,bathymsoil, &
      & gsp,o2soilflux,  .true.,lso)
    endif 
  endif
  gs%isoilcol = nsoilcols
  call METHANE &
  & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
  & fbbleflx_ch4(0,nsoilcols),wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
  & rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
  & ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false. ,&
  & lsm, bathymwater, &
  & fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
  & febul_lake_bot, febul_lake_surf, &
  & plant_sum,bull_sum,oxid_sum,rprod_sum, &
  & anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
  & ice_meth_oxid_total, &
  & h_talik,tot_ice_meth_bubbles,add_to_winter)
  qmethane(1:M+1) = qwater(1:M+1,2)
  qmethane(M+2:M+ns) = qsoil(2:ns,nsoilcols)
  work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
  fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 

  call ADDOXPRODCONS(M,i_maxN,H_mixed_layer,H_photic_zone,prodox,resp,bod,sod,RadWater,dt,ls,gas)
  call OXYGEN (gs, Tw2, fbbleflx_o2_sum, fbbleflx_o2(0,nsoilcols), lamoxyg, gsp, fracb0, pressure, wind10, &
  & o2_pres_atm, ls, bathymwater, lso, dt, febul0(nsoilcols), eps1(1), sodbot, oxyg, soilswitch%par, fo2)
  call CARBON_DIOXIDE (gs, Tw2, fbbleflx_co2_sum, fbbleflx_co2(0,nsoilcols), fco2ebul_bot, fco2ebul_bot, &
  & lamcarbdi, gsp, bathymwater, lsc, fracb0, &
  & pressure, wind10, co2_pres_atm, febul0(nsoilcols), ls, dt, eps1(1), sodbot, gas, soilswitch%par, fco2)
  call PHOSPHORUS(M,dt,ls,gsp,bathymwater,gas,lsp,lamcarbdi)
  call METHANE_OXIDATION(M, i_maxN, dt, ddz, h1, bathymwater, oxyg(1,2), qwater(1,2), DIC(1,2), Tw2, &
  & methoxid,qwateroxidtot, qwateroxidML)
  if (oxicmethprod%par == 1) call METHANE_OXIC_PRODUCTION(dt,H_mixed_layer,extwat,Chl_a,methoxid,prodox,gs,gsp,gas)

!Calculation of water current's velocities and turbulent characteristics
  if (Turbpar%par /= 1) then
  
    if (Turbpar%par == 2 .or. Turbpar%par == 5) then
      call MOMENTUM_SOLVER(ix, iy, dnx, dny, ndatamax, &
      & year, month, day, hour, kor, a_veg, c_veg, h_veg, &
      & alphax, alphay, dt, b0, tau_air, tau_i, tau_gr, tau_wav, fetch, depth_area)  
    endif

    !Calculation of water density profile at the current timestep
    call DENSITY_W(M,eos%par,lindens%par,Tw2,Sal2,preswat,row2)

    if (Turbpar%par == 2) then
      do i = 1, nstep_keps%par
        call K_EPSILON(ix,iy,dnx,dny, year, month, day, hour, &
        & kor, a_veg, c_veg, h_veg, dt_keps, &
        & b0, tau_air, tau_wav, tau_i, tau_gr, roughness, fetch) 
      enddo
    else
!     Calculation of the of eddy difusivity for temperature
      SELECT CASE (Turbpar%par)
        CASE (5)
          call RICHARDSON_NUMBER(M,u2,v2,ddz,h1+dhw,row2,Ri)
          allocate(work1(1:M))
          call EDDY_DIFF_MOST_RICH(M,Ri,2_iintegers,u2,v2,dzeta_05int,ddz,h1+dhw,work1,KT)
          deallocate(work1)
          KT(:) = cw_m_row0*KT(:)
        CASE (8)
          ! Hostetler formulation
          !call ED_TEMP_HOSTETLER &
          !& (wind,velfrict_prev,zref,phi,z_full,row,KT,M)
          KT(:) = 0.
          call UnDenGrMix(M,eos%par,lindens%par,ddz*h1,dzs(1),csoil(1),rosoil(1),wst,gas)
        CASE (9)
          ! Modified Hostetler formulation for shallow lake
          ! call ED_TEMP_HOSTETLER2 &
          ! & (wind,velfrict_prev,zref,phi,z_full,row,KT,M)
          KT(:) = 0. !cw_m_row0*KT(:)
        CASE (1)
          do i = 1, M
            KT(i) = (k2(i)-niu_wat)*cw_m_row0 
          enddo
        CASE DEFAULT
          KT = lamTM*k2*cw_m_row0 
      ENDSELECT 
    endif

    !Background vertical diffusivity
    !select case(backdiff%par)
    !case(0)
      lamw_back(:) = 0.
    !case(1)
    !  call DIFFMIN_HS(area_lake,wst,gsp,ls,M) !Calculating background heat conductance
    !end select

    do i = 1, M
      lamw(i) = lamw0 + KT(i) + lamw_back(i) !+KC(i)
    enddo  

  else
    lamw = lamw0*3.d0
  endif
  
  
  h2 = h1 + dhw
  l2 = l1 + dhi 
  ls2 = ls1 + dls
  
else

!CASE 2.2: WATER AND ICE WITHOUT SNOW

  call MIXED_LAYER_CALC(row,ddz,ddz2,dzeta_int,dzeta_05int,h1,M,i_ML,H_mixed_layer,maxN)

!Radiation fluxes in layers
  Erad = shortwave*(1-albedoofice)
  forall (i=1:nbands) radflux(i) = Erad*fracbands(i)
  call RadIce  %RAD_UPDATE(ddzi05(0:Mice)*l1, radflux)! (/Erad/))
  forall (i=1:nbands) radflux(i) = RadIce%flux(Mice+1,i)
  call RadWater%RAD_UPDATE(ddz05 (0:M   )*h1, radflux)!(/RadIce%flux(Mice+1,1)/))
! Calculating photic zone depth (<1% of surface irradiance)
  if (RadWater%integr(1) > 0) then
    H_photic_zone = h1
    do i = 1, M
      if (RadWater%integr(i)/RadWater%integr(1) < photic_threshold) then
        H_photic_zone = z_full(i)
        exit
      endif
    enddo
  else
    H_photic_zone = 0.
  endif
  if (ls1 /= 0.e0_ireals) then
    forall (i=1:nbands) radflux(i) = RadWater%flux(M+1,i)
    call RadDeepIce%RAD_UPDATE(ddzi05(0:Mice)*ls1, radflux) !(/RadWater%flux(M+1,1)/))
  endif

!Calculation of the layer's thickness increments
  x = Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par)
  ! Melting point is that of fresh water, because it is pure ice that melts at the ice top
  if (Ti1(1) > x + T_phase_threshold) then
    dhwhigh = (Ti1(1) - x - T_phase_threshold) * &
    & ci_m_roi_v(1)*ddzi(1)*l1*0.5d0/(row0_m_Lwi)
    dhis = 0. 
    dhihigh = - dhwhigh*row0_d_roi   
    if (saltice%par == 1) then
      dhwhigh = dhwhigh * (1. + porice(1)*row0_d_roi / (1. - porice(1)) )
      dhihigh = dhihigh * (1. / (1. - porice(1)) )
    endif
    Ti1(1) = x + T_phase_threshold
  endif

! Creation of the initial thin snow layer - to be considered at the next timestep
  if (precip > 0. .and. tempair < 0. .and. l1 > 0.05) then
    snowmass = snowmass + precip*dt*row0
    if (snowmass/rofrsn >= 3.*dzmin) then
      flag_snow = 1
      hs1 = snowmass/rofrsn + small_value
      itop = ms-2
    endif
    dhip = 0.
  else
    dhip = precip*row0_d_roi*dt
  end if
  if (tempair > 0. .and. snowmass > 0.) then
    dhip = dhip + snowmass/row0
    snowmass = 0.
  endif

  flux1 = - lami_v(Mice)*(Ti1(Mice+1)-Ti1(Mice))/(ddzi(Mice)*l1)+RadIce%integr(Mice) + &
  &  ci_m_roi_v(Mice+1)*(dhi-dhi0)*(Ti1(Mice+1)-Ti1(Mice))/(2.d0*dt)
  flux2 = -lamw(1)*(Tw1(2)-Tw1(1))/(ddz(1)*h1) + RadWater%integr(1) + &
  &  cw_m_row0*dhw0*(Tw1(2)-Tw1(1))/(2.*dt) + &
  &  cw_m_row0*PEMF(1)*(pt_down_f(1)-0.5d0*(Tw1(2)+Tw1(1)) )
  ! Limiting the freezing/melting rate by the thicknesses of water and ice
  dhwlow = min(max(dt*(flux1 - flux2)/(row0_m_Lwi),-ddz(1)*h1),ddzi(Mice)*l1*roi_d_row0)
  dhilow = - dhwlow*row0_d_roi
  !Correction of layers increments in a case of porous and saline ice
  if (saltice%par == 1) then
    dhwlow = dhwlow * (1. + porice(Mice+1)*row0_d_roi / (1. - porice(Mice+1)) )
    dhilow = dhilow * (1. / (1. - porice(Mice+1)) )
  endif

  if (Ti1(1) < Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par) + T_phase_threshold) then
    dhis = - Elatent/(roi*Liv)*dt
    dhwhigh = 0
    dhihigh = 0
  endif
  dhi = dhihigh + dhilow + dhis + dhip
  dhi0 = dhis + dhihigh + dhip
  if (ls1 /=0 ) then
    ice = 1; snow = 0; water = 1; deepice = 1 
    flux1 = - lamw(M)*(Tw1(M+1)-Tw1(M))/(ddz(M)*h1) + RadWater%integr(M) + &
    & cw_m_row0*(dhw-dhw0)*(Tw1(M+1)-Tw1(M))/(2.d0*dt) + &
    & cw_m_row0*PEMF(M)*(pt_down_f(M)-0.5d0*(Tw1(M+1)+Tw1(M)) )
    flux2 = - lami*(Tis1(2)-Tis1(1))/(ddzi(1)*ls1) + RadDeepIce%integr(1) + &
    & ci_m_roi*dls0*(Tis1(2)-Tis1(1))/(2.d0*dt)
    dhwls = dt*(flux1 - flux2)/(row0_m_Lwi)
    !dhwls = min(max(dt*(flux1 - flux2)/(row0_m_Lwi),-ddz(M)*h1),ddzi(1)*ls1*roi_d_row0)
    dls = - dhwls*row0_d_roi
    dls0 = dls
    dhw = dhwhigh + dhwfsoil + dhwls + dhwlow + dhwtrib
    dhw0 = dhwhigh + dhwlow
  else
    ice = 1; snow = 0; water = 1; deepice = 0
    dhw = dhwhigh + dhwfsoil + dhwlow + dhwtrib
    dhw0 = dhwhigh + dhwlow
    dls = 0.
  endif

  if (soilswitch%par == 1) then
    call SOIL_COND_HEAT_COEF(nsoilcols)
  endif
  ! Calculation of heat sources from heat fluxes from mutliple soil columns
  if (multsoil) then 
    iwork = (/1,0/)
    call SOILCOLSTEMP(nveclen,gs,gsp,dt,ls,ftot,ch4_pres_atm,ddz,ddzi,zsoilcols(1,ix,iy), &
                     & wst,RadWater%integr,a,b,c,d,add_to_winter, &
                     & bathymwater,bathymice, &
                     & bathymdice,bathymsoil(1,ix,iy), &
                     & soilflux(1,ix,iy), fdiffbot, iwork)
    call LATERHEAT(ix,iy,gs,ls, &
    & bathymwater,bathymice,bathymdice,bathymsoil, &
    & gsp,soilflux(1,ix,iy),.false.,lsh)
  endif
  ! Calculation of the whole temperature profile
  call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
  & RadWater, RadIce, SurfTemp, fetch, dt)
  if (soilswitch%par == 1) then
    call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
  endif

  !Assuming all dissolved species diffuse with the same molecular diffusivity as salinity
  lamsal(:)    = (lamw(:) - lamw0)/cw_m_row0 * alsal    + lamsal0
  lammeth(:)   = (lamw(:) - lamw0)/cw_m_row0 * almeth   + lamsal0
  lamoxyg(:)   = (lamw(:) - lamw0)/cw_m_row0 * aloxyg   + lamsal0
  lamcarbdi(:) = (lamw(:) - lamw0)/cw_m_row0 * alcarbdi + lamsal0
  lampart  (:) = (lamw(:) - lamw0)/cw_m_row0 * alcarbdi !no molecular diffusion for particles

  ! Salinity profile in water and soil
  dhwsnmelt = 0.  !Needed in S_DIFF
  call S_DIFF(dt,ls,salice)
  if (saltice%par == 1) call UPDATE_ICE_SALINITY(dt,ls,wst)
  ! Oxygen model precedes methane module, in order sedimentary 
  ! oxygen demand to be available for the latter
  call CHLOROPHYLLA(z_full, H_mixed_layer, H_photic_zone, extwat, RadWater, gas, M, Chl_a, itroph)
  call OXYGEN_PRODCONS(gs, gsp, wst, bathymsoil, gas, area_int, area_half, ddz05, &
  & h1, dt, Tsoil3, Chl_a, dzs, por, oxygsoil, itroph, &
  & prodox, resp, bod, sod, sodbot)

  if (soilswitch%par == 1) then
  ! Calculation of temperature in all soil columns, except for the lowest one
    if (multsoil) then
      iwork = (/0,1/)
      call SOILCOLSTEMP(nveclen,gs,gsp,dt,ls,ftot,ch4_pres_atm,ddz,ddzi,zsoilcols(1,ix,iy), &
                       & wst,RadWater%integr,a,b,c,d,add_to_winter, &
                       & bathymwater,bathymice, &
                       & bathymdice,bathymsoil(1,ix,iy), &
                       & soilflux(1,ix,iy), fdiffbot, iwork)
      methsoilflux(1:nsoilcols-1) = fdiffbot(1:nsoilcols-1)
    endif
    ! Calling bubble model
    call BUBBLE_BLOCK
    ! Calculation of methane sources from methane fluxes from mutliple soil columns
    if (multsoil) then 
      call LATERHEAT(ix,iy,gs,ls, &
      & bathymwater,bathymice,bathymdice,bathymsoil, &
      & gsp,methsoilflux,.true.,lsm)
      call LATERHEAT(ix,iy,gs,ls, &
      & bathymwater,bathymice,bathymdice,bathymsoil, &
      & gsp,co2soilflux, .true.,lsc)
      call LATERHEAT(ix,iy,gs,ls, &
      & bathymwater,bathymice,bathymdice,bathymsoil, &
      & gsp,o2soilflux,  .true.,lso)
    endif
  endif
  gs%isoilcol = nsoilcols
  call METHANE &
  & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
  & fbbleflx_ch4(0,nsoilcols),wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
  & rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
  & ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false., &
  & lsm, bathymwater, &
  & fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
  & febul_lake_bot, febul_lake_surf, &
  & plant_sum,bull_sum,oxid_sum,rprod_sum, &
  & anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
  & ice_meth_oxid_total, &
  & h_talik,tot_ice_meth_bubbles,add_to_winter)
  qmethane(1:M+1)    = qwater(1:M+1,2)
  qmethane(M+2:M+ns) = qsoil (2:ns,nsoilcols)
  work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
  fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 

  call ADDOXPRODCONS(M,i_maxN,H_mixed_layer,H_photic_zone,prodox,resp,bod,sod,RadWater,dt,ls,gas)
  call OXYGEN (gs, Tw2, fbbleflx_o2_sum, fbbleflx_o2(0,nsoilcols), lamoxyg, gsp, fracb0, pressure, wind10, &
  & o2_pres_atm, ls, bathymwater, lso, dt, febul0(nsoilcols), eps1(1), sodbot, oxyg, soilswitch%par, fo2)
  call CARBON_DIOXIDE (gs, Tw2, fbbleflx_co2_sum, fbbleflx_co2(0,nsoilcols), fco2ebul_bot, fco2ebul_bot, &
  & lamcarbdi, gsp, bathymwater, lsc, fracb0, &
  & pressure, wind10, co2_pres_atm, febul0(nsoilcols), ls, dt, eps1(1), sodbot, gas, soilswitch%par, fco2)
  call PHOSPHORUS(M,dt,ls,gsp,bathymwater,gas,lsp,lamcarbdi)
  call METHANE_OXIDATION(M, i_maxN, dt, ddz, h1, bathymwater, oxyg(1,2), qwater(1,2), DIC(1,2), Tw2, &
  & methoxid,qwateroxidtot, qwateroxidML)
  if (oxicmethprod%par == 1) call METHANE_OXIC_PRODUCTION(dt,H_mixed_layer,extwat,Chl_a,methoxid,prodox,gs,gsp,gas)

!Calculation of water current's velocities and turbulent characteristics
  if (Turbpar%par /= 1) then
  
    if (Turbpar%par == 2 .or. Turbpar%par == 5) then
      call MOMENTUM_SOLVER(ix, iy, dnx, dny, ndatamax, &
      & year, month, day, hour, kor, a_veg, c_veg, h_veg, &
      & alphax, alphay, dt, b0, tau_air, tau_i, tau_gr, tau_wav, fetch, depth_area)  
    endif

    !Calculation of water density profile at the current timestep
    call DENSITY_W(M,eos%par,lindens%par,Tw2,Sal2,preswat,row2)

    if (Turbpar%par == 2) then
      do i = 1, nstep_keps%par
        call K_EPSILON(ix,iy,dnx,dny, &
        & year, month, day, hour, kor, a_veg, c_veg, h_veg, dt_keps, &
        & b0, tau_air, tau_wav, tau_i, tau_gr, roughness, fetch) 
      enddo
    else
!     Calculation of the of eddy difusivity for temperature
      SELECT CASE (Turbpar%par)
        CASE(5)
          call RICHARDSON_NUMBER(M,u2,v2,ddz,h1+dhw,row2,Ri)
          allocate(work1(1:M))
          call EDDY_DIFF_MOST_RICH(M,Ri,2_iintegers,u2,v2,dzeta_05int,ddz,h1+dhw,work1,KT)
          deallocate(work1)
          KT(:) = cw_m_row0*KT(:)
        CASE (8)
          ! Hostetler formulation
          ! call ED_TEMP_HOSTETLER &
          ! & (wind,velfrict_prev,zref,phi,z_full,row,KT,M)
          KT(:) = 0.
          call UnDenGrMix(M,eos%par,lindens%par,ddz*h1,dzs(1),csoil(1),rosoil(1),wst,gas)
        CASE (9)
          ! Modified Hostetler formulation for shallow lake
          ! call ED_TEMP_HOSTETLER2 &
          ! & (wind,velfrict_prev,zref,phi,z_full,row,KT,M)
          KT(:) = 0.
        CASE (1)
          do i = 1, M
            KT(i) = (k2(i)-niu_wat)*cw_m_row0 
          enddo
        CASE DEFAULT
          KT = lamTM*k2*cw_m_row0 
      ENDSELECT 
    endif

    !Background vertical diffusivity
    !select case(backdiff%par)
    !case(0)
      lamw_back(:) = 0.
    !case(1)
    !  call DIFFMIN_HS(area_lake,wst,gsp,ls,M) !Calculating background heat conductance
    !end select

    do i = 1, M
      lamw(i) = lamw0 + KT(i) + lamw_back(i) !+KC(i)
    enddo 
    
  else
    lamw = lamw0*3.d0
  endif
  

  h2 = h1 + dhw
  l2 = l1 + dhi
  ls2 = ls1 + dls
 
endif

!Diagnostic calculations
do i = 1, M
  k_turb_T_flux(i) = - 0.5*lamw(i)/(cw_m_row0) * &
  & ( Tw2(i+1) + Tw1(i+1) - Tw2(i) - Tw1(i)) / (ddz(i)*h1)
  T_massflux(i) = PEMF(i)*(pt_down_f(i)-0.5d0*(Tw2(i+1)+Tw2(i)))
enddo


!The scales of turbulence
call TURB_SCALES(gsp,ls, bathymwater, wst, RadWater, &
& k_turb_T_flux, T_massflux, row, eflux0_kinem_water, &
& turb_density_flux, Buoyancy0, tau_air-tau_wav, kor, i_maxN, H_mixed_layer,maxN, w_conv_scale, &
& T_conv_scale, Wedderburn, LakeNumber, Rossby_rad, ThermThick, ReTherm, RiTherm, trb)
 
ENDIF if2

!CASE 3: ICE WITHOUT WATER

if3: IF (layer_case == 3) THEN

! Specification of volumetric heat capacity and heat conductance for ice
if   (saltice%par == 0) then
  !Pure ice
  ci_m_roi_v(:) = ci_m_roi
  lami_v    (:) = lami
elseif (saltice%par == 1) then
  !Ice with pores filled with water
  do i = 1, Mice+1
    ci_m_roi_v(i) = ci*roi*(1. - porice(i)) + cw*row0*porice(i)
  enddo
  do i = 1, Mice
    x = 0.5*(porice(i) + porice(i+1))
    lami_v(i)      = lami  *(1. - x) + lamw0  *x
  enddo
endif
   
!CASE 3.1: ICE WITH SNOW

if (flag_snow == 1) then

! Radiation fluxes
  SR_botsnow = shortwave*(1-albedoofsnow)
  if (flag_snow_init /= 1) then
    do i = itop, ms-1
      SR_botsnow = SR_botsnow*EXTINCT_SNOW(dens(i))**dz(i)
    enddo
  endif
  forall (i=1:nbands) radflux(i) = SR_botsnow*fracbands(i) !assuming spectrum does not change in snow
  call RadIce%RAD_UPDATE(ddzi05(0:Mice)*l1, radflux) !(/SR_botsnow/))

  x = Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par)
  ! Melting point is that of fresh water, because it is pure ice that melts at the ice top
  if (Ti1(1) > x + T_phase_threshold) then
    dhwhigh = (Ti1(1) - x - T_phase_threshold) * &
    & ci_m_roi_v(1)*l1*ddzi(1)*0.5d0/(row0_m_Lwi)
    dhif = - dhwhigh*row0_d_roi
    if (saltice%par == 1) then
      dhwhigh = dhwhigh * (1. + porice(1)*row0_d_roi / (1. - porice(1)) )
      dhif    = dhif * (1. / (1. - porice(1)) )
    endif
    Ti1(1) = x + T_phase_threshold
  else
    dhwhigh = 0
    dhif = 0
  endif
  dhwsnmelt = snmeltwat*snmelt*dt
  dhw = dhwsnmelt + dhwhigh + dhwtrib
  dhi = dhif
  dhi0 = dhi
  ice = 1; snow = 1; water = 0; deepice = 0

  call SNOW_COND_HEAT_COEF()
  if (soilswitch%par == 1) then
    call SOIL_COND_HEAT_COEF(nsoilcols)
  endif
  ! Calculation of heat sources from heat fluxes from mutliple soil columns
  if (multsoil) call LATERHEAT(ix,iy,gs,ls, &
  & bathymwater,bathymice,bathymdice,bathymsoil, &
  & gsp,soilflux(1,ix,iy),.false.,lsh)
  ! Temperature profile calculation
  call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
  & RadWater, RadIce, SurfTemp, fetch, dt)
  if (soilswitch%par == 1) then
    call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
  endif

  ! Salinity profile
  call S_DIFF(dt,ls,salice)
  if (saltice%par == 1) call UPDATE_ICE_SALINITY(dt,ls,wst)
  call SNOWTEMP(ix,iy,dnx,dny,year,month,day,hour,snowmass, &
  & snowmass_init,a,b,c,d,Temp,phi,fetch,dt)
  if (soilswitch%par == 1) then
    call BUBBLE_BLOCK
  endif
  gs%isoilcol = nsoilcols
  call METHANE &
  & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
  & fbbleflx_ch4(0,nsoilcols),wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
  & rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
  & ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false., &
  & lsm, bathymwater, &
  & fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
  & febul_lake_bot, febul_lake_surf, &
  & plant_sum,bull_sum,oxid_sum,rprod_sum, &
  & anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
  & ice_meth_oxid_total, &
  & h_talik,tot_ice_meth_bubbles,add_to_winter)
  qmethane(1:M+1)    = qwater(1:M+1,2)
  qmethane(M+2:M+ns) = qsoil (2:ns,nsoilcols)
  work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
  fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 


  l2 = l1 + dhi 
  h2 = h1 + dhw
  ls2 = 0.

else

!CASE 3.2: ICE WITHOUT SNOW !

  ! Radiation fluxes
  Erad = shortwave*(1-albedoofice)
  forall (i=1:nbands) radflux(i) = Erad*fracbands(i)
  call RadIce%RAD_UPDATE(ddzi05(0:Mice)*l1, radflux) !(/Erad/))

  if (precip > 0 .and. tempair < 0) then
    snowmass = snowmass + precip*dt*row0
    if (snowmass/rofrsn >= 3.*dzmin) then
      flag_snow = 1
      hs1 = snowmass/rofrsn + small_value
      itop = ms-2
    endif
    !flag_snow = 1
    !hs1 = 0.02
    dhip = 0.
  else
    dhip = precip*row0_d_roi*dt
  end if
  x = Meltpnt(0.e0_ireals,0.e0_ireals,nmeltpoint%par)
  if (Ti1(1) > x + T_phase_threshold) then
    dhw = (Ti1(1) - x - T_phase_threshold) * &
    & ci_m_roi_v(1)*l1*ddzi(1)*0.5d0/(row0_m_Lwi) + dhwtrib
    dhihigh = - dhw*row0_d_roi
    if (saltice%par == 1) then
      dhwhigh = dhwhigh * (1. + porice(1)*row0_d_roi / (1. - porice(1)) )
    endif
    dhis = 0
    Ti1(1) = x + T_phase_threshold
  else
    dhis = - Elatent/(roi*Liv)*dt
    dhw = 0.
    dhihigh = 0
  endif
  dhi = dhis + dhihigh + dhip
  dhi0 = dhi
  ice = 1; snow = 0; water = 0; deepice = 0 

  ! Temperature profile caluclation
  if (soilswitch%par == 1) then
    call SOIL_COND_HEAT_COEF(nsoilcols)
  endif
  ! Calculation of heat sources from heat fluxes from mutliple soil columns
  if (multsoil) call LATERHEAT(ix,iy,gs,ls, &
  & bathymwater,bathymice,bathymdice,bathymsoil, &
  & gsp,soilflux(1,ix,iy),.false.,lsh)
  call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
  & RadWater, RadIce, SurfTemp, fetch, dt)
  if (soilswitch%par == 1) then
    call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
  endif

  ! Salinity profile
  call S_DIFF(dt,ls,salice)
  if (saltice%par == 1) call UPDATE_ICE_SALINITY(dt,ls,wst)
  if (soilswitch%par == 1) then
    call BUBBLE_BLOCK
  endif
  gs%isoilcol = nsoilcols
  call METHANE &
  & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
  & fbbleflx_ch4(0,nsoilcols), wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
  & rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
  & ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false., &
  & lsm, bathymwater, &
  & fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
  & febul_lake_bot, febul_lake_surf, &
  & plant_sum,bull_sum,oxid_sum,rprod_sum, &
  & anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
  & ice_meth_oxid_total, &
  & h_talik,tot_ice_meth_bubbles,add_to_winter)
  qmethane(1:M+1)    = qwater(1:M+1,2)
  qmethane(M+2:M+ns) = qsoil (2:ns,nsoilcols)
  work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
  fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 

   
  l2 = l1 + dhi
  h2 = h1 + dhw
  ls2 = 0.
 
endif

ENDIF if3 

! CASE 4: SNOW AND SOIL WITHOUT ICE AND WATER
    
if4: IF (layer_case==4) THEN
  print*, 'The non-operational case: &
  &there is no water and ice layers: STOP'
  STOP
  if (flag_snow == 1) then
    ice = 0; snow = 1; water = 0; deepice = 0

    ! Temperature profile caluclation
    call SNOW_COND_HEAT_COEF()
    if (soilswitch%par == 1) then
      call SOIL_COND_HEAT_COEF(nsoilcols)
    endif
    call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
    & RadWater, RadIce, SurfTemp, fetch, dt)
    if (soilswitch%par == 1) then
      call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
    endif

    ! Salinity profile
    call S_DIFF(dt,ls,salice)
    call SNOWTEMP(ix,iy,dnx,dny,year,month,day,hour,snowmass, &
    & snowmass_init,a,b,c,d,Temp,phi,fetch,dt)
    if (soilswitch%par == 1) then
      call BUBBLE_BLOCK 
    endif
    gs%isoilcol = nsoilcols
    call METHANE &
    & (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
    & fbbleflx_ch4(0,nsoilcols),wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
    & rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
    & ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false., &
    & lsm, bathymwater, &
    & fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
    & febul_lake_bot, febul_lake_surf, &
    & plant_sum,bull_sum,oxid_sum,rprod_sum, &
    & anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
    & ice_meth_oxid_total, &
    & h_talik,tot_ice_meth_bubbles,add_to_winter)
    qmethane(1:M+1)    = qwater(1:M+1,2)
    qmethane(M+2:M+ns) = qsoil (2:ns,nsoilcols)
    work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
    fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 
  else
    ice = 0; snow = 0; water = 0; deepice = 0
    
    ! Temperature profile calculation
    if (soilswitch%par == 1) then
      call SOIL_COND_HEAT_COEF(nsoilcols)
    endif
    call T_SOLVER(ix,iy,dnx,dny,year,month,day,hour,phi, &
    & RadWater, RadIce, SurfTemp, fetch, dt)

    ! Salnity profile
    call S_DIFF(dt,ls,salice)
    if (soilswitch%par == 1) then
      call SOILFORLAKE(nveclen,dt,a,b,c,d,nsoilcols)
      call BUBBLE_BLOCK 
    endif
    gs%isoilcol = nsoilcols
    call METHANE(gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil3(1,nsoilcols),rosoil,fbbleflx_ch4_sum, &
    & fbbleflx_ch4(0,nsoilcols),wl4(1,nsoilcols),wi2(1,nsoilcols),wa,Sals1, &
    & rootss,rosdry,por,veg,0._ireals,TgrAnn, methgenmh, &
    & ddz,ddz05,wst, lammeth, h1, ls, dhw, dhw0, .true., .false., &
    & lsm, bathymwater, &
    & fplant, febul0(nsoilcols), fdiffbot(nsoilcols), ftot, fdiff_lake_surf, &
    & febul_lake_bot, febul_lake_surf, &
    & plant_sum,bull_sum,oxid_sum,rprod_sum, &
    & anox,gs,gsp,dt,eps1(1),sodbot,rprod_total_oldC,rprod_total_newC, &
    & ice_meth_oxid_total, &
    & h_talik,tot_ice_meth_bubbles,add_to_winter)
    qmethane(1:M+1)    = qwater(1:M+1,2)
    qmethane(M+2:M+ns) = qsoil (2:ns,nsoilcols)
    work2(1:nsoilcols+1) = bathymsoil(1:nsoilcols+1,ix,iy)%area_int
    fdiff_lake_bot = AVERSOILCOL(fdiffbot,work2,nsoilcols) 
  endif 
ENDIF if4


! Calculation of volume deficit (only if dead volume is specified), diagnostic
if (deadvol%par > 0. .and. h2 < deadvol%par) then
  voldef = voldef + area_int(1)*(deadvol%par - h2)
  h2 = deadvol%par
endif
   

if (h2 < min_water_thick .and. ls2 /= 0 .and. l2 == 0) then
  ls2 = ls2 + h2*row0_d_roi
  if (ls2 < 0.5*min_ice_thick) then
    ls2 = 0; Tis2 = 0
  endif
  h2 = 0.
  Tw2 = 0.
  Sal2 = 0.
endif

if ((h2 < min_water_thick .and. h2 > 0) .and. l2 /= 0) then
  l2 = l2 + h2*row0_d_roi
  if (l2 < 0.5*min_ice_thick) then
    l2 = 0; Ti2 = 0
  endif
  h2 = 0.
  Tw2 = 0.
  Sal2 = 0. 
end if
    
if (h2 < min_water_thick .and. l2 /= 0 .and. ls2 /= 0) then
  Ti2 = Ti2*l2/(l2+ls2) + Tis2*ls2/(l2+ls2)
  l2 = l2 + h2*row0_d_roi + ls2 
  ls2 = 0.
  Tis2 = 0.
  h2 = 0 
  Tw2 = 0.
  Sal2 = 0.
endif

if (l2 < 0.5*min_ice_thick .and. h2 /= 0) then
  h2 = h2 + l2*roi/row0
  if (h2 < min_water_thick) then 
    h2 = 0; Tw2 = 0; Sal2 = 0.
  endif
  l2 = 0.
  Ti2 = 0.
end if

if (h2 < min_water_thick .and. l2 == 0) then
  h2 = 0.
  Tw2 = 0.
  Sal2 = 0. 
end if

if (l2 < 0.5*min_ice_thick .and. h2 == 0) then
  l2 = 0.
  Ti2 = 0.
end if

if ((hs1 < dzmin .and. hs1 > 0.) .or. &
& (hs1 > 0. .and. l2 == 0. .and. h2 /= 0.)) then
! h2 = h2 + (totalprecips-totalevaps-totalmelts)
  h2 = h2 + snowmass/row0 
  if (h2 < min_water_thick) then
    l2 = l2 + row0*h2/roi
    h2 = 0; Tw2 = 0; Sal2 = 0.
    if (l2 < 0.5*min_ice_thick) then
      l2 = 0; Ti2 = 0
    endif
  endif
  hs1 = 0.
  snowmass = 0.
  flag_snow = 0
  flag_snow_init = 1
end if

if (hs1==0. .and. flag_snow==1) then
  flag_snow = 0
  flag_snow_init = 1
endif

if (ls2 < 0.5*min_ice_thick .and. h2 /= 0) then
  h2 = h2 + ls2*roi/row0
  if (h2 < min_water_thick) then 
    h2 = 0; Tw2 = 0; Sal2 = 0.
  endif
  ls2 = 0
  Tis2 = 0.
endif 

if (ls2 > 0 .and. h2 == 0) then
  l2 = ls2
  Ti2 = Tis2
  if (l2 < 0.5*min_ice_thick) then
    l2 = 0; Ti2 = 0
  endif
  ls2 = 0
  Tis2 = 0.
endif

if (outflpar%par == 2 .and. tribheat%par > 0) then
  print*, 'LAGREFFL not operational: STOP'
  STOP
endif

!Checking temperature for NaNs and extreme values
do i = 1, M+1
  if (Tw2(i) /= Tw2(i) .or. abs(Tw2(i)) > 100.) then
    print*, 'Tw2(i) = ', Tw2(i), 'Tw1(i) = ', Tw1(i), &
    & 'i = ',i,', nstep = ', nstep, &
    & ', init(ix,iy) = ', init(ix,iy), ', ix = ', ix, ', iy = ', iy
    print*, 'Other coordinates: ', nx, ny, nx0, ny0, nx_max, ny_max
    print*, 'Layers: ', h1, l1, hs1, ls1,h2, l2, hs1, ls2
    print*, 'Meteo: ', pressure, tempair, humair, uwind, vwind, shortwave, longwave, &
    & precip
    print*, 'Tw2=', Tw2
  endif
enddo


!! Deepened temperature maximum diagnostics
!! Output of temperature equation terms
!if (firstcall)   then
!!  open(1234,file='maxT.dat',status='unknown')
!!  write(1234,*), 'nstep dT/dt -dT_{ML}/dt -ddz/dt/dzT flux1/dz -flux2/dz flux1 -flux2 &
!!  &S1/dz -S2/dz S1 -S2 botheat z_{ML} dz T_mean  Tmax-T_{ML} locTmax'
!  nn = 0
!  allocate (work3(1:17)); work3 = 0.; 
!!  allocate (work4(1:17)); work4 = 0.
!endif
!!write(12345,*) tau/(wind*wind*1.273)
!!if (mod(nstep,60*6) == 0_iintegers) then
!! Identifying the location of the heated layer
!worklog1 = .false.
!i = 1
!c1:do
!  i = i + 1
!  !Averaging over the mixed layer
!  vol_ = 0.
!  do k = 1, i-1
!    vol_ = vol_ + ddz05(k-1)*h1*area_int(k)
!  enddo
!  ttt = 0.
!  do k = 1, i-1
!    ttt = ttt + Tw2(k)*ddz05(k-1)*h1*area_int(k)
!  enddo
!  ttt = ttt/vol_
!  if (Tw2(i) > ttt + 0.01 .and. z_full(i) > 1.) then
!  !if (KT(i-1)/cw_m_row0 < min_diff*100.5 .and. Tw2(i+1) > Tw2(i) .and. z_full(i) > 1.) then
!    !Upper boundary
!    x = 0.5*(z_full(i-1) + z_full(i))
!    j = i
!    c2:do 
!      j = j + 1
!      if (j > M+1) exit c2
!      if (Tw2(j) < ttt) then
!        !Lower boundary
!        xx = 0.5*(z_full(j) + z_full(j-1))
!        worklog1 = .true.
!        exit c2
!      endif
!    enddo c2
!    if (worklog1) then
!      ! Averaging over the temperature maximum layer
!      vol_2 = 0.
!      do k = i, j-1
!        vol_2 = vol_2 + ddz05(k-1)*h1*area_int(k)
!      enddo
!      zzz = 0.
!      do k = i, j-1
!        zzz = zzz + Tw2(k)*ddz05(k-1)*h1*area_int(k)
!      enddo
!      zzz = zzz/vol_2
!      zzz = zzz - ttt
!    endif
!    i2 = i
!    j2 = j
!    exit c1
!  endif
!  if (i == M+1) exit c1
!enddo c1
!worklog = .false.
!i = 1
!c3:do
!  i = i + 1
!  !Averaging over the mixed layer
!  vol_ = 0.
!  do k = 1, i-1
!    vol_ = vol_ + ddz05(k-1)*h1*area_int(k)
!  enddo
!  tt = 0.
!  do k = 1, i-1
!    tt = tt + Tw1(k)*ddz05(k-1)*h1*area_int(k)
!  enddo
!  tt = tt/vol_
!  if (Tw1(i) > tt + 0.01 .and. z_full(i) > 1.) then
!  !if (KT(i-1)/cw_m_row0 < min_diff*100.5 .and. Tw1(i+1) > Tw1(i) .and. z_full(i) > 1.) then
!    !Upper boundary
!    y = 0.5*(z_full(i-1) + z_full(i))
!    j = i
!    c4:do 
!      j = j + 1
!      if (j > M+1) exit c4
!      if (Tw1(j) < tt) then
!        !Lower boundary
!        yy = 0.5*(z_full(j) + z_full(j-1))
!        worklog = .true.
!        exit c4
!      endif
!    enddo c4
!    if (worklog) then
!      ! Averaging over the temperature maximum layer
!      vol_ = 0.
!      do k = i, j-1
!        vol_ = vol_ + ddz05(k-1)*h1*area_int(k)
!      enddo
!      zz = 0.
!      do k = i, j-1
!        zz = zz + Tw1(k)*ddz05(k-1)*h1*area_int(k)
!      enddo
!      zz = zz/vol_
!      zz = zz - tt
!    endif
!    exit c3
!  endif
!  if (i == M+1) exit c3
!enddo c3
!
!if (worklog .and. worklog1 .and. (j-1 > i+1) .and. yy < 4.) then
!
!  !Initializing TM mean temperature
!  if (.not. indTMprev) then
!    work3(1) = zz 
!    work3(2:12) = 0.
!    !print*, zz
!    !print*, Tw1(i:j-1) - tt
!    !print*, Tw1
!    !read*
!    indTMprev = .true.
!  endif
!
!  ! Volume change term
!  ww1 = 0.
!  if (vol_2 /= vol_) then
!    !print*, i,i2,j,j2
!    if (i2 > i) then
!      do k = i, i2-1
!        ww1 = ww1 - (Tw2(k) - ttt)*ddz05(k-1)*h1*area_int(k)
!      enddo
!    elseif (i2 < i) then
!      do k = i2, i-1
!        ww1 = ww1 + (Tw2(k) - ttt)*ddz05(k-1)*h1*area_int(k)
!      enddo
!    endif
!    if (j2 > j) then
!      do k = j, j2-1
!        ww1 = ww1 + (Tw2(k) - ttt)*ddz05(k-1)*h1*area_int(k)
!      enddo
!    elseif (j2 < j) then
!      do k = j2, j-1
!        ww1 = ww1 - (Tw2(k) - ttt)*ddz05(k-1)*h1*area_int(k)
!      enddo
!    endif
!  endif
!
!!!  ! Summing increments of temperature in the maximum
!   if (l1 > 0.) then
!     nn = nn + 1
!     work3(1) = work3(1) + lamw(1)*(Tw2(2)-Tw2(1))/(ddz(1)*h1)
!   endif
!  work3(1) = work3(1) + (zzz - zz)
!  work3(2) = work3(2) - (ttt - tt)
!  work3(3) = work3(3) - (vol_2 - vol_)/(vol_)*zzz + ww1/(vol_)
!  work3(4) = work3(4) + 1./vol_*area_half(i-1)*k_turb_T_flux(i-1)*dt
!  work3(5) = work3(5) - 1./vol_*area_half(j-1)*k_turb_T_flux(j-1)*dt
!  work3(6) = work3(6) + k_turb_T_flux(i-1)*dt
!  work3(7) = work3(7) - k_turb_T_flux(j-1)*dt
!  work3(8) = work3(8) + 1./vol_*area_half(i-1)*SR(i-1)/cw_m_row0*dt
!  work3(9) = work3(9) - 1./vol_*area_half(j-1)*SR(j-1)/cw_m_row0*dt
!  work3(10) = work3(10) + SR(i-1)/cw_m_row0*dt
!  work3(11) = work3(11) - SR(j-1)/cw_m_row0*dt
!  ww = 0.
!  do k = i, j-1
!    ww = ww + lsh%water(k)*area_int(k)*ddz05(k-1)
!  enddo
!  work3(12) = work3(12) + ww*h1/cw_m_row0/vol_*dt
!  work3(13) = work3(13) + y
!  work3(14) = work3(14) + yy-y
!  work3(15) = work3(15) + zz
!  work3(16) = work3(16) + maxval(Tw1(i:j-1))-tt
!  work3(17) = work3(17) + (maxloc(Tw1,1) - i + 0.5)*ddz(i)*h1 !Assuming regular mesh
!
!  ! Time averaging
!  work4(1:17) = work4(1:17) + work3(1:17)
!
!  !www = (zzz - zz)/dt + (ttt - tt)/dt + (vol_2 - vol_)/(dt*vol_)*zzz - ww1/(vol_*dt) &
!  !& - 1./vol_*area_half(i-1)*k_turb_T_flux(i-1) &
!  !& + 1./vol_*area_half(j-1)*k_turb_T_flux(j-1) &
!  !& - 1./vol_*area_half(i-1)*SR(i-1)/cw_m_row0 &
!  !& + 1./vol_*area_half(j-1)*SR(j-1)/cw_m_row0 &
!  !& - ww*h1/cw_m_row0/vol_
!  !if (abs(www/(zzz - zz)*dt) > 1.E-3) then
!  !  print*, (zzz - zz)/dt, www, (vol_2 - vol_)/(dt*vol_)*zzz, - ww1/(vol_*dt) 
!  !  read*
!  !endif
!
!  if (abs(work3(1) - zzz) > 1.e-10) then
!    print*, 'TMdiag', nstep, zz, zzz, work3(1) - zzz
!    read*
!  endif
!
!  if (mod(nstep,60*6) == 0_iintegers) then
!    write(1234,*) nstep, (zzz - zz)/dt, - (ttt - tt)/dt, &
!    !& - (area_half(j-1)*(xx - yy) - area_half(i-1)*(x - y))/(dt*vol_)*zz, &
!    & - (vol_2 - vol_)/(dt*vol_)*zzz + ww1/(vol_*dt), &
!    & + 1./vol_*area_half(i-1)*k_turb_T_flux(i-1), &
!    & - 1./vol_*area_half(j-1)*k_turb_T_flux(j-1), &
!    & + k_turb_T_flux(i-1), - k_turb_T_flux(j-1), &
!    &   1./vol_*area_half(i-1)*SR(i-1)/cw_m_row0, &
!    & - 1./vol_*area_half(j-1)*SR(j-1)/cw_m_row0, &
!    &   SR(i-1)/cw_m_row0, - SR(j-1)/cw_m_row0, &
!    &   ww*h1/cw_m_row0/vol_, &
!    &   y, yy-y, zz, maxval(Tw1(i:j-1))-tt, &
!    &   (maxloc(Tw1(i:j-1),1)-0.5)*ddz(i)*h1 !Assuming regular mesh
!  endif 
!
!else
!
!  indTMprev = .false.
!
!endif
!
!if (step_final) then
!!  open(12345,file='maxTav.dat',status='unknown')
!!  write(12345,*), 'nstep dT/dt -dT_{ML}/dt -ddz/dt/dzT flux1/dz -flux2/dz flux1 -flux2 &
!!  &S1/dz -S2/dz S1 -S2 botheat z_{ML} dz T_mean  Tmax-T_{ML} locTmax'
!!  write(12345,*) nstep, work3(1:17)/real(nn)
!!  write(12345,*) nstep, work3(1:17)
!!  write(12345,*) nstep, work4(1:17)/real(nn)
!!  !print*, 'resid = ', work3(1) - work3(2) - work3(3) - work3(4) - work3(5) - work3(8) - work3(9) - work3(12)
!!  close(12345)
!  print*, 'Mean under-ice heat flux towards ice is', work3(1)/real(nn), ' W/m**2' 
!  deallocate(work3)
!  read*
!!   deallocate(work4)
!endif
!!!endif

!i = maxloc(Tw2,1)
!if (i > 1 .and. i < M+1 .and. (mod(nstep,60) == 0_iintegers) ) then
!  write(1234,*) cw_m_row0*(Tw2(i) - Tw1(i))/dt,  &
!  & 0.5/h1**2*( lamw(i)*(Tw2(i+1) + Tw1(i+1) - Tw2(i) - Tw1(i) )/ddz(i) - &
!  & lamw(i-1)*( Tw2(i) + Tw1(i) - Tw2(i-1) - Tw1(i-1) )/ddz(i-1) )/ddz05(i-1), &
!  & lsh%water(i), (SR(i) - SR(i+1))/(h1*ddz05(i-1))
!endif

!! On-screen diagnostics of TKE budget
!i = M-40 !Level in thermocline
!if (month == 10 .and. nn == 0) then
!  xx = tau/row0*u1(1)
!  yy = - H_mixed_layer*lamw0/cw_m_row0*g/row0*(row(i+1)-row(i))/(ddz(i)*h1)
!  zz = - sum(eps1(1:i_maxN)*ddz(1:i_maxN))*h1
!  nn = 1
!elseif (nn > 0) then
!  nn = nn + 1
!  xx  = ACCUMM(nn, xx, tau/row0*u1(1))
!  yy  = ACCUMM(nn, yy, - H_mixed_layer*lamw0/cw_m_row0*g/row0*(row(i+1)-row(i))/(ddz(i)*h1))
!  zz  = ACCUMM(nn, zz, - sum(eps1(1:i_maxN)*ddz(1:i_maxN))*h1)
!  !print*, xx,yy,zz
!  !read*
!endif
!if (mod(nstep,100) == 0) then
!  !i = M-40 !Level in thermocline
!  print*, tau/row0*u1(1), -H_mixed_layer*lamw0/cw_m_row0*g/row0*(row(i+1)-row(i))/(ddz(i)*h1), - &
!  & sum(eps1(1:i_maxN)*ddz(1:i_maxN))*h1, &
!  & tau/row0*u1(1) - H_mixed_layer*lamw0/cw_m_row0*g/row0*(row(i+1)-row(i))/(ddz(i)*h1) - &
!  & sum(eps1(1:i_maxN)*ddz(1:i_maxN))*h1
!  print*, xx + yy + zz
!endif

h1    = h2
l1    = l2
ls1   = ls2
Tw1   = Tw2
Ti1   = Ti2
Tis1  = Tis2
Sal1  = Sal2
Sals1 = Sals2
qwater(:,1) = qwater(:,2)
oxyg  (:,1) = oxyg  (:,2)
DIC(:,1) = DIC(:,2)
u1    = u2
v1    = v2
Tskin(1) = Tskin(2)
    
! CALCULATION OF SUMMARY FLUXES !
totalpen = totalpen + dhwfsoil 
totalevap = totalevap + Elatent/(row0*Lwv)*dt
totalprecip = totalprecip + precip*dt
totalhflux = totalhflux + hflux*dt
totalerad = totalerad + erad*dt

if (l1 == 0) tsw = Tw1(1) + Kelvin0
if (l1 /=0 .and. flag_snow == 0) tsw = Ti1(1) + Kelvin0
if (flag_snow == 1) tsw = T(itop) + Kelvin0

if (init(ix,iy) == 0) then
  init(ix,iy) = 1  
endif

!VALUES AT NEXT TIME STEP (time + dt) IN CURRENT POINT (ix,iy)

call UPDATE_NEXT_TIMESTEP ( &
& ix, iy, dnx, dny, M, Mice, ns, ms, ml, nsoilcols, &
& l1, h1, hx1, hx2, hy1, hy2, ls1, hs1, &
& hx1t, hx2t, hy1t, hy2t, &
& hx1ml, hx2ml, hy1ml, hy2ml, &
& gradpx_tend, gradpy_tend, &
& u1, v1, &
& E1, eps1, k_turb_T_flux, &
& Tsoil1, Sals1, wi1, wl1, &
& Tw1, Sal1, lamw, &
& Tskin(1), &
& Ti1, Tis1, &
& dz, T, wl, dens, &
& qwater(1,1), qsoil, &
& oxyg(1,1), oxygsoil, &
& DIC(1,1), DOC, POCL, POCD, phosph, &
& snmelt, snowmass, &
& cdmw, &
& time, &
& dhwfsoil, &
& Elatent, &
& dhw, dhw0, &
& dhi, dhi0, dls0, &
& velfrict, &
& roughness, &
& eflux0_kinem, &
& tot_ice_meth_bubbles, &
& febul0, Eseiches, salice, porice, &

& flag_snow, flag_snow_init, &
& itop, &
& nstep, i_maxN, itherm )

!At the next timestep after control point output, the control point may be written again
!if (ix == nx - nx0 + 1 .and. iy == ny - ny0 + 1 .and. &
!&  cpwrite_done .eqv. .true.) cpwrite_done = .false.
!Writing the control point for all lakes of the domain
!if (icp == 1_iintegers .and. (nstep /= nstep_cp_written)) then
!  call CONTROL_POINT_OUT(year,month,day,hour, &
!  & nx,nx0,nx_max,ny,ny0,ny_max,gs,parparams,time)
!  nstep_cp_written = nstep
!endif


hw1 = hw
xlew1 = xlew
cdmw1 = cdmw
surfrad1 = surfrad
h2_out = h2
fdiff_lake_surf1 = fdiff_lake_surf

!Diagnoctics
if (dyn_pgrad%par == 5 .or. dyn_pgrad%par == 6) then !diagnostic computation of turbulence in a river scenario
  call ZIRIVMODEL(M,h1,sign(1._ireals,tauxw)*sqrt(abs(tauxw)/row0), &
  & g*tan(pi*alphax/180.),z_half,z_full,eps_ziriv,viscturb_ziriv,speed_ziriv)
endif

if (accum .eqv. .false. .and. &
&   year  == year_accum_begin  .and. &
&   month == month_accum_begin .and. &
&   day   == day_accum_begin   .and. &
&   hour  >= hour_accum_begin) then
  accum = .true.
  allocate(work1(1:2))
  work1 = VARMEAN(qwater(1,1),bathymwater,11_iintegers)
  totmeth0 = work1(2) * (h2 - dhw)*dzeta_05int(M)*mf ! Depth at previous timestep
  deallocate(work1)
endif

if (accum .eqv. .true. .and. &
&   year  == year_accum_end  .and. &
&   month == month_accum_end .and. &
&   day   == day_accum_end   .and. &
&   hour  >= hour_accum_end) accum = .false.

if (accum) then
! Note: variables accumulation is implemented currently only for one-point simulations!
  call ACCUM_VAR &
  & (dt, l1, fbbleflx_ch4_sum(0), fbbleflx_ch4_sum(M+1), & 
  & fdiffbot(nsoilcols), fdiff_lake_surf, qwateroxidtot, &
  & ice_meth_oxid_total, &
  & rprod_total_newC(nsoilcols), rprod_total_oldC(nsoilcols), &
  & febultot, febulbottot, fdifftot, fdiff_lake_surftot, &
  & rprod_total_newC_integr, rprod_total_oldC_integr, &
  & methoxidwat, ice_meth_oxid, add_to_winter)
  allocate(work1(1:2))
  work1 = VARMEAN(qwater(1,2),bathymwater,11_iintegers)
  totmethc = work1(2) * h2**dzeta_05int(M)*mf
  deallocate(work1)
!  metracc = metracc + qwater(M+1,2)*(dhw - dhw0) + dhw0*qwater(1,2)
!  call ACCUM_VAR & ! two-meth
!  & (dt, l1, febul2, fdiff2, fdiff_lake_surf2, & ! two-meth
!  & rprod_total_newC, rprod_total_oldC, & ! two-meth
!  & febultot2, fdifftot2, fdiff_lake_surftot2, & ! two-meth
!  & rprod_total_newC_integr2, rprod_total_oldC_integr2) ! two-meth
endif

!Diagnosing the pools and sources (carbon etc.)
allocate(work1(1:2))
!Pools
work1 = VARMEAN(DIC (1,1),bathymwater,grid=1); pools_lake%pools(npool_DIC ) = work1(1)
work1 = VARMEAN(DOC (1,1),bathymwater,grid=1); pools_lake%pools(npool_DOC ) = work1(1)
work1 = VARMEAN(DOC (1,2),bathymwater,grid=1); pools_lake%pools(npool_DOC ) = &
&                                              pools_lake%pools(npool_DOC ) + work1(1)
work1 = VARMEAN(POCL(1  ),bathymwater,grid=1); pools_lake%pools(npool_POCL) = work1(1)
work1 = VARMEAN(POCD(1  ),bathymwater,grid=1); pools_lake%pools(npool_POCD) = work1(1)
pools_lake%pools(:) = pools_lake%pools(:)/area_int(1)
!Sources/sinks
work1 = VARMEAN(prodox,bathymwater,grid=1); srcs_lake%sources(nsrc_prodox) = work1(1)
work1 = VARMEAN(resp  ,bathymwater,grid=1); srcs_lake%sources(nsrc_resp  ) = work1(1)
work1 = VARMEAN(bod   ,bathymwater,grid=1); srcs_lake%sources(nsrc_BOD   ) = work1(1)
work1 = VARMEAN(sod   ,bathymwater,grid=1); srcs_lake%sources(nsrc_SOD   ) = work1(1)
srcs_lake%sources(:) = srcs_lake%sources(:)/area_int(1)
deallocate(work1)

if (flag_print) then ! The output in ASCII files 
  if (monthly_out%par == 1) call MON_OUT(ix,iy,dnx,dny,year,month,day,hour,time,zgrid_out,ngrid_out%par,tsw)
  if (daily_out%par   == 1) call DAY_OUT(ix,iy,dnx,dny,year,month,day,hour)
  if (hourly_out%par  == 1) call HOUR_OUT(ix,iy,dnx,dny,year,month,day,hour,time)
  if (everystep%par   >  0) call EVERYSTEP_OUT(ix,iy,dnx,dny)
  if (time_series%par == 1) then
    call SERIES_OUT(ix,iy,dnx,dny,year,month,day,hour,tsw)
    allocate(work1(1:M+ns))
    ndec = -1 ! Number of decimal digits, negative meaning exponential format
    i = 1
    !Interpolating diffusivity to layers' interfaces, result is stored in work1
    !call LININTERPOL (z_half,lamsal,M,z_full,work1,M+1,flag) 
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  Tw1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, & ! ngrid_out%par
    &  outpath, 'water_temp', i, ndec, .false.)  !, row, work1 - last optional argument!
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  Ti1, z_full_ice, Mice+1, &
    &  zgridice_out, ngridice_out%par, & !ngridsoil_out%par, &
    &  outpath, 'ice_temp', i, ndec, .false.)
    ndec = 4
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  Tsoil3(1,nsoilcols), zsoil, ns, &
    &  zgridsoil_out, ngridsoil_out%par, & !ngridsoil_out%par, &
    &  outpath, 'soil_temp', i, ndec, .false.)
    z_watersoil(1:M+1) = z_full(1:M+1)    
    do j = 2, ns
      z_watersoil(M+j) = h1 + zsoil(j)
    enddo
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  rowc, z_full, M+1, &
    &  zgrid_out, -1, &
    &  outpath, 'water_dens_layers', i, ndec, .false.)
    ndec = -1
    i = i + 2
    work1(1:M+ns) = qmethane(1:M+ns)*molm3tonM
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_watersoil, M+ns, &
    &  zgridsoil_out, -1, &
    &  outpath, 'methane_water_soil', i, ndec, .false.)
    ndec = -1
    i = i + 2
    work1(1:M+1) = oxyg(1:M+1,1)*molm3tomgl_o2 !molm3tonM
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'oxygen_water', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  Chl_a, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'chla', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  prodox, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'prodox', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  sod, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'sod', i, ndec, .false.)
    ndec = -1
    i = i + 2
    work1(1:M+1) = qwater(1:M+1,1)*molm3tomcM !molm3tomkgl_ch4 ! molm3tonM
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'methane_water', i, ndec, .false.)
    if (nsoilcols > 1) then
      ndec = -1
      i = i + 2
      ! Methane production in soil columns
      work1(1:nsoilcols) = 0.5*(zsoilcols(1:nsoilcols,ix,iy) + zsoilcols(2:nsoilcols+1,ix,iy))
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  rprod_total_newC, work1, nsoilcols, &
      &  work1, nsoilcols, &
      &  outpath, 'rprod_soil', i, ndec, .false.)
    endif 
    ndec = -1
    i = i + 2
    do j = 1, M+1
      work1(j) = DIC(j,1) / HC_CORR_CARBEQUIL(Tw1(j)+Kelvin0,pH) * molm3tomcM !molm3tomgl_co2 !molm3tonM
    enddo
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'co2_water', i, ndec, .false.)


    if (carbon_model%par == 2) then

      ndec = -1
      i = i + 2
      work1(1:M+1) = phosph(1:M+1)*molmass_p !mg/l of phosphorus
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  work1, z_full, M+1, &
      &  zgrid_out, ngrid_out%par, &
      &  outpath, 'phosph_water', i, ndec, .false.)

      ndec = -1
      i = i + 2
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  DOC, z_full, M+1, &
      &  zgrid_out, ngrid_out%par, &
      &  outpath, 'DOC', i, ndec, .false.)

      ndec = -1
      i = i + 2
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  POCL, z_full, M+1, &
      &  zgrid_out, ngrid_out%par, &
      &  outpath, 'POCL', i, ndec, .false.)

      ndec = -1
      i = i + 2
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  POCD, z_full, M+1, &
      &  zgrid_out, ngrid_out%par, &
      &  outpath, 'POCD', i, ndec, .false.)
    endif

    ndec = -1
    i = i + 2    
    work1(1:M) = fbbleflx_ch4_sum(1:M) 
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_half, M, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'fbblflx_ch4', i, ndec, .false.)
    ndec = -1
    i = i + 2
    work1(1:M) = fbbleflx_co2_sum(1:M)
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_half, M, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'fbblflx_co2', i, ndec, .false.)
    ndec = -1
    i = i + 2
    work1(1:M) = fbbleflx_o2_sum(1:M)
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  work1, z_half, M, &
    &  zgrid_out, ngrid_out%par, &
    &  outpath, 'fbblflx_o2', i, ndec, .false.)
    deallocate(work1)
    ndec = 4
    i = i + 2    
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  wl1, zsoil, ns, &
    &  zgridsoil_out, ngridsoil_out%par, &
    &  outpath, 'wl_soil', i, ndec, .false.)
    ndec = 4
    i = i + 2    
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  wi1, zsoil, ns, &
    &  zgridsoil_out, ngridsoil_out%par, &
    &  outpath, 'wi_soil', i, ndec, .false.)    
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  Sal1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'sal_water', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  lamw/cw_m_row0, z_half, M, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'lamw', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  eps1, z_half, M, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'epsilon', i, ndec, .false.)
    if (dyn_pgrad%par == 5 .or. dyn_pgrad%par == 6) then
      ndec = -1
      i = i + 2
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  eps_ziriv, z_half, M, &
      &  zgrid_out, ngrid_out%par, & !ngrid_out
      &  outpath, 'epsziriv', i, ndec, .false.)
      ndec = -1
      i = i + 2
      call PROFILE_OUTPUT &
      & (ix, iy, dnx, dny, &
      &  year, month, day, hour, &
      &  time, dt_out%par, &
      &  viscturb_ziriv, z_half, M, &
      &  zgrid_out, ngrid_out%par, & !ngrid_out
      &  outpath, 'viscturbziriv', i, ndec, .false.)
    endif
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  Ri, z_half, M, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'Ri', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  trb%Rp, z_half, M, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'Rp', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  trb%Rpdens, z_half, M, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'Rpdens', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  u1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'u', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  v1, z_full, M+1, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'v', i, ndec, .false.)
    ndec = -1
    i = i + 2
    call PROFILE_OUTPUT &
    & (ix, iy, dnx, dny, &
    &  year, month, day, hour, &
    &  time, dt_out%par, &
    &  w(1:M), z_half, M, &
    &  zgrid_out, ngrid_out%par, & !ngrid_out
    &  outpath, 'w', i, ndec, .true.)
  endif

  ! Output time series of scalars in selected points
  if ( rtemp%arr1(1,1) /= missing_value .and. dyn_pgrad%par == 4 ) then 
      call TEMPLOC(M,3_iintegers,1_iintegers,itherm,h1,dt_out%par,time,hx1ml,hy1ml,Tw1,gsp,gs,bathymwater,rtemp, &
    &            'temploc',firstcall)
      call TEMPLOC(M,3_iintegers,2_iintegers,itherm,h1,dt_out%par,time,hx1ml,hy1ml,qwater(1,1),gsp,gs,bathymwater,rtemp, &
    &            'ch4loc',firstcall)
      call TEMPLOC(M,3_iintegers,3_iintegers,itherm,h1,dt_out%par,time,hx1ml,hy1ml,DIC(1,1),gsp,gs,bathymwater,rtemp, &
    &            'co2loc',firstcall)
  endif

endif
 

if (runmode%par == 1)  then
  if (nscreen%par > 0 .and. mod(nstep,nscreen%par)==0) then

    write (*,'(3a25)') 'timestep', 'N of point', &
    & 'Surface temperature'
    write (*,'(2i25,f25.1)') nstep, ix, tsw
    write (*,'(5a8)') 'Year', 'Month', 'Day', 'Hour', 'Months'
    i = year 
    if (year<1000) i = year+1000
    write (*,'(3i8,2f8.2)') i, month, day, hour, time/(month_sec)

    !write(*,'(a)')
    !write(*,*) 'Methane generation, transport and oxidation rates'
    !write(*,*) 'Total bottom diff, Total surface diff, Total bottom ebul, Total surface ebul'
    !write(*,'(a,4(2x,e12.5))') 'Summer ', fdifftot(1), fdiff_lake_surftot(1), febulbottot(1), febultot(1)
    !write(*,'(a,4(2x,e12.5))') 'Winter ', fdifftot(2), fdiff_lake_surftot(2), febulbottot(2), febultot(2)
    !write(*,*) 'Total prod from "new" org, Total prod from "old" org, Total oxidation in water, &
    !& Total oxidation in ice (for winter)'
    !write(*,'(a,3(2x,e12.5))') 'Summer ', &
    !& rprod_total_newC_integr(1), rprod_total_oldC_integr(1), methoxidwat(1)
    !write(*,'(a,4(2x,e12.5))') 'Winter ', &
    !& rprod_total_newC_integr(2), rprod_total_oldC_integr(2), methoxidwat(2), ice_meth_oxid
    !write(*,'(a,(2x,e12.5))') 'Change of integral methane amount in water column and ice cover', &
    !& totmethc - totmeth0 + tot_ice_meth_bubbles*mf
    !write(*,'(a)')
    !write(*,*) 'Checking methane balance residuals:'   
    !write(*,'(a,1(2x,e12.5))') 'Total production in soil - total bottom flux ', &
    !& sum(rprod_total_newC_integr(1:2)) + sum(rprod_total_oldC_integr(1:2)) - &
    !& sum(febulbottot(1:2)) - sum(fdifftot(1:2))
    !write(*,'(a,1(2x,e12.5))') 'Total bottom flux - total surface flux - total oxidation - & 
    !& methane total column amount change', &
    !& sum(febulbottot(1:2)) + sum(fdifftot(1:2)) - &
    !& sum(febultot(1:2)) - sum(fdiff_lake_surftot(1:2)) - &
    !& sum(methoxidwat(1:2)) - ice_meth_oxid - &
    !& (totmethc - totmeth0 + tot_ice_meth_bubbles*mf)


    open(12345,file=outpath(1:len_trim(outpath))//'lastscrout.dat')
    write(12345,'(a)')
    write(12345,*) 'Methane generation, transport and oxidation rates'
    write(12345,*) 'Total bottom diff, Total surface diff, Total bottom ebul, Total surface ebul'
    write(12345,'(a,4(2x,e12.5))') 'Summer ', fdifftot(1), fdiff_lake_surftot(1), febulbottot(1), febultot(1)
    write(12345,'(a,4(2x,e12.5))') 'Winter ', fdifftot(2), fdiff_lake_surftot(2), febulbottot(2), febultot(2)
    write(12345,*) 'Total prod from "new" org, Total prod from "old" org, Total oxidation in water, &
    & Total oxidation in ice (for winter)'    
    write(12345,'(a,3(2x,e12.5))') 'Summer ', &
    & rprod_total_newC_integr(1), rprod_total_oldC_integr(1), methoxidwat(1)
    write(12345,'(a,4(2x,e12.5))') 'Winter ', &
    & rprod_total_newC_integr(2), rprod_total_oldC_integr(2), methoxidwat(2), ice_meth_oxid
    write(12345,'(a,(2x,e12.5))') 'Change of integral methane amount in water column and ice cover', &
    & totmethc - totmeth0 + tot_ice_meth_bubbles*mf
    write(12345,'(a)')
    write(12345,*) 'Checking methane balance residuals:'   
    write(12345,'(a,1(2x,e12.5))') 'Total production in soil - total bottom flux ', &
    & sum(rprod_total_newC_integr(1:2)) + sum(rprod_total_oldC_integr(1:2)) - &
    & sum(febulbottot(1:2)) - sum(fdifftot(1:2))
    write(12345,'(a,1(2x,e12.5))') 'Total bottom flux - total surface flux - total oxidation - & 
    & methane total column amount change', &
    & sum(febulbottot(1:2)) + sum(fdifftot(1:2)) - &
    & sum(febultot(1:2)) - sum(fdiff_lake_surftot(1:2)) - &
    & sum(methoxidwat(1:2)) - ice_meth_oxid - &
    & (totmethc - totmeth0 + tot_ice_meth_bubbles*mf)
    close(12345)

  endif
endif
  

!write(workchar,'(i5)') ncomp
!workchar = '(a6,'//workchar(1:len_trim(workchar))//'(1x,f6.2))'
!write(*,fmt = workchar) 'Timing', comptime(1:ncomp)/60. ! Converting to minutes
!if (l1 > 0.) print*, T(itop), Ti1(1), Ti1(Mice+1)

if (present(LakePhysISIMIP)) then
  !ISIMIP-Lake output
  x  = 0.5*(sign(1._ireals,l2  - small_value) + 1.) !1 -- ice , 0 -- no ice
  xx = 0.5*(sign(1._ireals,hs1 - small_value) + 1.) !1 -- snow, 0 -- no snow
  LakePhysISIMIP%istratstart = missing_value
  LakePhysISIMIP%istratend = missing_value
  LakePhysISIMIP%istratdur = missing_value
  LakePhysISIMIP%icestart = missing_value
  LakePhysISIMIP%icetend = missing_value
  LakePhysISIMIP%icedur = missing_value
  LakePhysISIMIP%thermodepth = H_mixed_layer
  LakePhysISIMIP%surftemp = Tw2(1)+Kelvin0  !sum(Tw2(1:i_ML)*ddz05(0:i_ML-1))*h1/H_mixed_layer
  LakePhysISIMIP%bottemp  = Tw2(M+1)+Kelvin0!sum(Tw2(i_ML+1:M+1)*ddz05(i_ML:M))*h1/(h1-H_mixed_layer)
  LakePhysISIMIP%lakeicefrac = x
  LakePhysISIMIP%icethick = l2
  LakePhysISIMIP%snowthick = hs1
  LakePhysISIMIP%waterthick = h2
  LakePhysISIMIP%icetemp =  (Ti2(1)  + Kelvin0)*x  + (1-x) *missing_value
  LakePhysISIMIP%snowtemp = (T(itop) + Kelvin0)*xx + (1-xx)*missing_value
  !if (xx > 0.) print*, "snowtemp=", LakePhysISIMIP%snowtemp
  LakePhysISIMIP%sensheatf = hflux
  LakePhysISIMIP%latentheatf = Elatent
  LakePhysISIMIP%momf = tau
  LakePhysISIMIP%lwup = surfrad
  LakePhysISIMIP%swup = swup
  LakePhysISIMIP%lakeheatf = - swup - surfrad - hflux - Elatent + &
  & longwave + shortwave 
  LakePhysISIMIP%albedo = albedo
  LakePhysISIMIP%extcoeff = extwat
  LakePhysISIMIP%sedheatf = - botflux
  forall (i=1:M+1) LakePhysISIMIP%watertemp(i) = Tw2(i)+Kelvin0
  forall (i=1:M+1) LakePhysISIMIP%zwatertemp(i) = z_full(i)
endif

deallocate(radflux)
deallocate(a,b,c,d)
deallocate(work2)
deallocate(Temp)
nullify(time_group%year,time_group%month,time_group%day,time_group%hour)

! lakebatch restart: save the state after the last step
if (step_final .and. RESTART_OUT_REQUESTED()) then
  lb_reading = .false.
  call RESTART_OPEN(lb_reading, lb_u)
  call LB_MODULES_IO(lb_u, lb_reading)
  include 'lb_restart_lake.inc'
  call RESTART_CLOSE(lb_reading, lb_u)
endif

if (firstcall) firstcall = .false.

call LBC_CLEAR_POINT()

!FORMATS!
7   format (f7.3, 4i5,37f7.3) 
60  format (f5.2, 2e16.5, 2f8.3, e15.5, f11.5, f7.1)
! 61  format (i5, f6.2, 2e16.5, 2f8.3, e15.5, f11.5, f7.1)
61  format (f6.2, f5.2, 2e16.5, 2f8.3, e15.5, f11.5, f7.1)
62  format (f6.2, f5.2, 2f12.3, 2f8.3, f11.5, f7.1,3f12.4,2f7.2,3f7.1)
80  format (f7.3, 10f9.2)
90  format (f7.3, 12f9.2)   
100 format (a5, 2a16, 2a8, a15, a11, a7) 


contains
SUBROUTINE BUBBLE_BLOCK
! The block of code invoking procedures, that calculate the bubble flux
implicit none

! Calling bubble model
fbbleflx_ch4_sum(0:M+1) = 0.
fbbleflx_o2_sum (0:M+1) = 0.
fbbleflx_co2_sum(0:M+1) = 0.
febul_lake_bot = 0.
fco2ebul_bot = 0.
if (ifbubble%par == 1) then
  call BUBBLE(M,M+1,ddz,ddz05,dzeta_int,qwater(:,1),Tw1,Sal1,h1,pressure, &
  & DIC(:,1),oxyg(:,1),ngasb,febul0(nsoilcols),dt, &
  & fracb0,fbbleflx_ch4(0,nsoilcols), &
  & fbbleflx_co2(0,nsoilcols),fbbleflx_o2(0,nsoilcols))
endif 
if (multsoil) then
  do i = 1, nsoilcols-1
    if (ifbubble%par == 1) then
      call BUBBLE(M,bathymsoil(i,ix,iy)%ibot,ddz,ddz05,dzeta_int, &
      & qwater(:,1),Tw1,Sal1,z_half(isoilcolc(i)),pressure, &
      & DIC(:,1),oxyg(:,1),ngasb,febul0(i),dt, &
      & fracb0,fbbleflx_ch4(0,i), &
      & fbbleflx_co2(0,i),fbbleflx_o2(0,i)) 
    else
      fbbleflx_ch4(1:bathymsoil(i,ix,iy)%ibot,i) = 1.; fbbleflx_ch4(bathymsoil(i,ix,iy)%ibot+1:M+1,i) = 0.
      fbbleflx_co2(1:bathymsoil(i,ix,iy)%ibot,i) = 1.; fbbleflx_co2(bathymsoil(i,ix,iy)%ibot+1:M+1,i) = 0.
      fbbleflx_o2 (1:bathymsoil(i,ix,iy)%ibot,i) = 1.; fbbleflx_o2 (bathymsoil(i,ix,iy)%ibot+1:M+1,i) = 0. 
    endif
    ! Adding bubble flux from i-th soil column to the horizontally averaged bubble flux
    call BUBBLEFLUXAVER(ix,iy,i)
    ! Adding bubble fluxes to fluxes of CH_4, CO_2 and O_2 at soil columns tops
    ! Sedimentary oxygen demand is added to CO_2 and O_2 as sources in OXYGEN_MOD module
    methsoilflux(i) =    methsoilflux(i) - febul0(i)
    co2soilflux (i) =  - febul0(i)*fracb0(2)/fracb0(1)
    o2soilflux  (i) =  - febul0(i)*fracb0(3)/fracb0(1)
  enddo
endif

END SUBROUTINE BUBBLE_BLOCK

END SUBROUTINE LAKE


END MODULE LAKE_MOD
