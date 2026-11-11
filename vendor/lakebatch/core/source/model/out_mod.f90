MODULE OUT_MOD

use LAKE_DATATYPES, only : ireals, iintegers
use INOUT, only : CHECK_UNIT
use DRIVING_PARAMS, only : missing_value
use GRID_OPERATIONS_LAKE, only : LININTERPOL
  use TIME_LAKE_MOD, only : DATEMINUS, TIMESTR

character(len=60), save :: outpath
real(kind=ireals), allocatable, save :: series_voldef0(:,:)

contains

!> Subroutine MON_OUT performs monthly output in ASCII files
SUBROUTINE MON_OUT(ix,iy,nx,ny,year,month,day,hour,time,z_out,nout,tsw)

use ARRAYS_WATERSTATE, only : Tw1, Sal1
use ARRAYS_TURB, only : E1, E2, eps1, S, Gen, Gen_seiches, row, TKE_turb_trans, TF, KT
use ARRAYS, only : u1, v1
use ARRAYS_GRID, only : dzeta_int, z_full, z_half
use ARRAYS_BATHYM, only : h1, l1, voldef, vol, area_int
use ARRAYS_METHANE, only : fbbleflx_ch4_sum, fplant, fdiff_lake_surf
use DRIVING_PARAMS, only : M
use PHYS_CONSTANTS, only : cw_m_row0, Kelvin0

use INOUT_PARAMETERS, only : &
& lake_mon_out_unit_min, &
& lake_mon_out_unit_max

implicit none

! Input variables
! Reals
real(kind=ireals), intent(in) :: hour !< Hour of day
real(kind=ireals), intent(in) :: time !< Time from start of integration, sec
real(kind=ireals), intent(in) :: z_out(1:nout) !< The model output grid
real(kind=ireals), intent(in) :: tsw !< Surface temperature, Kelvin (water, ice or snow)

! Integers
integer(kind=iintegers), intent(in) :: year  !< Year
integer(kind=iintegers), intent(in) :: month !< Month of year
integer(kind=iintegers), intent(in) :: day   !< Day of month
integer(kind=iintegers), intent(in) :: ix, iy !< Indices of a lake on the grid
integer(kind=iintegers), intent(in) :: nx, ny !< Grid sizes
integer(kind=iintegers), intent(in) :: nout

! Local variables
! Reals
real(kind=ireals), allocatable :: accum_var (:,:,:,:)
real(kind=ireals), allocatable :: var       (:,:,:,:)
real(kind=ireals), allocatable :: accum_var_scalar (:,:,:)
real(kind=ireals), allocatable :: var_scalar       (:,:,:)
real(kind=ireals), allocatable :: tsteps    (:,:)
real(kind=ireals), allocatable :: Profile_out(:,:)
real(kind=ireals), allocatable :: work(:)
real(kind=ireals), allocatable :: voldef0(:,:), voldef0y(:,:)
real(kind=ireals) :: work1, work2
real(kind=ireals), allocatable :: valmax(:,:,:)

real(kind=ireals), external:: DZETA

! Integers
integer(kind=iintegers), parameter :: n_var = 11, n_var_scalar = 9, n_var_max = 3
integer(kind=iintegers), parameter :: n_var1 = 4 ! Number of vector variables defined at cell interfaces
integer(kind=iintegers), allocatable :: month_old(:,:)
integer(kind=iintegers), allocatable :: out_unita(:,:)
integer(kind=iintegers) :: out_unit = lake_mon_out_unit_min
integer(kind=iintegers) :: i, j ! Loop indices

! Characters
character :: month1*2
character :: year1*4
character :: day1*2
character :: hour1*2
character :: timestring*6
character :: coords_point*6
character :: formatline*50

! Logicals
logical :: firstcall, flag
logical, allocatable :: firstcallixiy(:,:)

data firstcall /.true./

SAVE

if (firstcall) then
  allocate (firstcallixiy(1:nx,1:ny))
  allocate (out_unita(1:nx,1:ny))
  allocate (month_old(1:nx, 1:ny))
  allocate (tsteps    (1:nx, 1:ny))  
  allocate (var      (1:n_var, 1:nx, 1:ny, 1:M+1) )
  allocate (accum_var(1:n_var, 1:nx, 1:ny, 1:M+1) )
  allocate (var_scalar      (1:n_var_scalar, 1:nx, 1:ny) )
  allocate (accum_var_scalar(1:n_var_scalar, 1:nx, 1:ny) )
  ! var_scalar(9) is never assigned below; without this it was written as
  ! uninitialized memory (column 11 of monthly/scalars)
  var_scalar(:,:,:) = 0.d0
  allocate (valmax(1:n_var_max,1:nx,1:ny))
  allocate (work(1:M+1))
  firstcallixiy(:,:) = .true.
  out_unita(:,:) = lake_mon_out_unit_min
  month_old(:,:) = month
  tsteps   (:,:) = 0.d0  
  accum_var(:,:,:,:) = 0.d0
  accum_var_scalar(:,:,:) = 0.d0
  if (.not. allocated(voldef0)) then
    allocate(voldef0(nx,ny), voldef0y(nx,ny))
  endif
  voldef0 = 0.
  voldef0y = 0.
  valmax(:,:,:) = 0.
endif

if (firstcallixiy(ix,iy)) then
  call CHECK_UNIT(lake_mon_out_unit_min,lake_mon_out_unit_max,out_unita(ix,iy))
  write (coords_point,'(2i3)') ix, iy
  open(out_unita(ix,iy),file=outpath(1:len_trim(outpath))//'monthly/'// &
  & 'scalars'//coords_point//'.dat', status='unknown')
  write (out_unita(ix,iy),*) '1 - year' 
  write (out_unita(ix,iy),*) '2 - month'
  write (out_unita(ix,iy),*) '3 - surface temperature, Celsius' 
  write (out_unita(ix,iy),*) '4 - water thickness, m' 
  write (out_unita(ix,iy),*) '5 - ice thickness, m' 
  write (out_unita(ix,iy),*) '6 - volume, m**3'
  write (out_unita(ix,iy),*) '7 - volume deficit, m**3'
  write (out_unita(ix,iy),*) '8 - mean diffusive CH4 flux, mol/(m**2*s)'
  write (out_unita(ix,iy),*) '9 - mean ebullition CH4 flux, mol/(m**2*s)'
  write (out_unita(ix,iy),*) '10 - mean plant CH4 flux, mol/(m**2*s)'
  write (out_unita(ix,iy),*) '11 - maximal monthly depth, m'
  write (out_unita(ix,iy),*) '12 - maximal monthly ice thickness, m'
  write (out_unita(ix,iy),*) '13 - maximal monthly volume, m**3'
  close (out_unita(ix,iy))
endif

! Variables located at cell interfaces
var(1,ix,iy,1:M+1) = Tw1 (1:M+1)
var(2,ix,iy,1:M+1) = Sal1(1:M+1)
var(3,ix,iy,1:M+1) = row (1:M+1)
var(4,ix,iy,1:M+1) = sqrt(u1 (1:M+1)*u1 (1:M+1) + v1 (1:M+1)*v1 (1:M+1) )
! Variables located at cell centers
var(5,ix,iy,1:M)   = E1  (1:M)
var(6,ix,iy,1:M)   = TF(1:M)*E2(1:M) !eps1(1:M)
var(7,ix,iy,1:M)   = S   (1:M)
var(8,ix,iy,1:M)   = Gen (1:M)
var(9,ix,iy,1:M)   = Gen_seiches (1:M)
var(10,ix,iy,1:M)  = TKE_turb_trans (1:M)
var(11,ix,iy,1:M)  = KT (1:M)/cw_m_row0 !Kinematic heat conductance, m**2/s

!Scalar variables
var_scalar(1,ix,iy) = tsw - Kelvin0
var_scalar(2,ix,iy) = h1
var_scalar(3,ix,iy) = l1
var_scalar(4,ix,iy) = vol
var_scalar(5,ix,iy) = voldef - voldef0(ix,iy)
voldef0(ix,iy) = voldef

var_scalar(6,ix,iy) = fdiff_lake_surf
var_scalar(7,ix,iy) = fbbleflx_ch4_sum(0)/area_int(1)
var_scalar(8,ix,iy) = fplant

valmax(1,ix,iy) = max(h1,valmax(1,ix,iy)) ! Calculating monthly maximum
valmax(2,ix,iy) = max(l1,valmax(2,ix,iy)) 
valmax(3,ix,iy) = max(vol,valmax(3,ix,iy))

accum_var(:,ix,iy,:) = accum_var(:,ix,iy,:) + var(:,ix,iy,:)
accum_var_scalar(:,ix,iy) = accum_var_scalar(:,ix,iy) + &
& var_scalar(:,ix,iy)
tsteps(ix,iy) = tsteps(ix,iy) + 1._ireals

if (month_old(ix,iy) /= month) then

  accum_var(:,ix,iy,:) = accum_var(:,ix,iy,:)/tsteps(ix,iy)
  accum_var_scalar(:,ix,iy) = accum_var_scalar(:,ix,iy)/tsteps(ix,iy) ! excepting volume deficit

  call DATEMINUS(1,year,month,day,hour,year1,month1,day1,hour1)
  call TIMESTR(6,year1,month1,day1,hour1,timestring)

  call CHECK_UNIT(lake_mon_out_unit_min,lake_mon_out_unit_max,out_unit)
  write (coords_point,'(2i3)') ix, iy
  open(out_unit,file=outpath(1:len_trim(outpath))//'monthly/'// &
  & 'Profiles'//coords_point//timestring//'.dat', status='unknown')
  write (out_unit,*) '1 - depth, m' 
  write (out_unit,*) '2 - temperature, C'
  write (out_unit,*) '3 - salinity, kg/kg' 
  write (out_unit,*) '4 - density, kg/m**3'
  write (out_unit,*) '5 - absolute value of velocity, m/s'
  write (out_unit,*) '6 - turbulent kinetic energy, m**2/s**2'
  write (out_unit,*) '7 - disspation rate, m**2/s**3'
  write (out_unit,*) '8 - TKE production by buoyancy, m**2/s**3'
  write (out_unit,*) '9 - TKE production by shear   , m**2/s**3'
  write (out_unit,*) '10 - TKE production by seiches (Goudsmit paramaterization)  , m**2/s**3'
  write (out_unit,*) '11 - TKE turbulent transport  , m**2/s**3'
  write (out_unit,*) '12 - kinematic heat conductance, m**2/s'
  formatline = '(f12.6,f9.3,e12.4,f9.3,8e12.4)'
  if (nout > 0) then
    allocate (Profile_out(1:nout,1:n_var))
    ! Interpolating to given output levels
    do j = 1, n_var1
      work(1:M+1) = accum_var(j,ix,iy,1:M+1)
      call LININTERPOL (z_full,work,M+1,z_out,Profile_out(1,j),nout,flag)
    enddo
    do j = n_var1+1, n_var
      work(1:M) = accum_var(j,ix,iy,1:M)
      call LININTERPOL (z_half,work,M,z_out,Profile_out(1,j),nout,flag)
    enddo
    j = nout
    if (z_out(nout) > z_full(M+1)) then
      do while (z_out(j) > z_full(M+1))
        j = j - 1
      enddo
    endif
    ! Writing to file
    do i = 1, j 
      write (unit = out_unit, fmt = formatline) &
      & -z_out(i), Profile_out(i,1:n_var)
    enddo
  else
    allocate (Profile_out(1:M+1,1:n_var))
    forall (i = 1:M+1, j = 1:n_var1) Profile_out(i,j) = accum_var(j,ix,iy,i)
    ! Interpolating from cell centers to cell interfaces
    do j = n_var1+1, n_var
      work(1:M) = accum_var(j,ix,iy,1:M)
      call LININTERPOL (z_half,work,M,z_full,Profile_out(1,j),M+1,flag)
    enddo
    ! Writing to file
    do i = 1, M+1 
      write (unit = out_unit, fmt = formatline) &
      & -dzeta_int(i)*h1, Profile_out(i,1:n_var)
    enddo
  endif
  deallocate(Profile_out)
  close(out_unit)

  read(year1,*) work1
  read(month1,*) work2
  open(out_unita(ix,iy),file=outpath(1:len_trim(outpath))//'monthly/'// &
  & 'scalars'//coords_point//'.dat', status='old', position='append')
  write (out_unita(ix,iy),fmt=*) & !, fmt = formatline) &
  & int(work1), int(work2), accum_var_scalar(1:n_var_scalar,ix,iy), &
  & valmax(:,ix,iy)
  close(out_unita(ix,iy))
  
  tsteps(ix,iy) = 0.d0
  accum_var(:,ix,iy,:) = 0.d0
  accum_var_scalar(1:4,ix,iy) = 0.d0
  if (month_old(ix,iy) == 12) accum_var_scalar(5,ix,iy) = 0.d0
  valmax(:,ix,iy) = 0.d0

  month_old(ix,iy) = month
endif

if (firstcall) firstcall = .false.
if (firstcallixiy(ix,iy)) firstcallixiy(ix,iy) = .false.
END SUBROUTINE MON_OUT


SUBROUTINE DAY_OUT(ix,iy,nx,ny,year,month,day,hour)

use ARRAYS_GRID, only : dzeta_int
use ARRAYS_WATERSTATE, only : Tw1, Sal1
use ARRAYS_TURB, only : E1, eps1, KT, Ri
use ARRAYS_BATHYM, only : h1
use ARRAYS, only : u1, v1
use ATMOS , only : discharge
use DRIVING_PARAMS, only : M
use PHYS_CONSTANTS, only : cw_m_row0

use INOUT_PARAMETERS, only : &
& lake_day_out_unit_min, &
& lake_day_out_unit_max

implicit none

! Input variables
! Reals
real(kind=ireals), intent(in) :: hour

! Integers
integer(kind=iintegers), intent(in) :: year, month, day
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: nx, ny

! Local variables
! Reals
real(kind=ireals), allocatable :: tsteps   (:,:)
real(kind=ireals), allocatable :: accum_var(:,:,:,:)
real(kind=ireals), allocatable :: accum_var_scalar (:,:,:)
real(kind=ireals), allocatable :: var_scalar (:,:,:)
real(kind=ireals), allocatable :: var      (:,:,:,:)

real(kind=ireals), external :: DZETA

real(kind=ireals) :: work1, work2, work3

! Integers
integer(kind=iintegers), allocatable :: day_old(:,:)
integer(kind=iintegers), allocatable :: out_unita(:,:)
integer(kind=iintegers) :: i ! Loop index
integer(kind=iintegers) :: out_unit = lake_day_out_unit_min
integer(kind=iintegers) :: nvars = 7
integer(kind=iintegers) :: nvars_scalar = 2

character :: month1*2
character :: year1*4
character :: day1*2
character :: hour1*2
character :: timestring*8
character :: coords_point*6

logical :: firstcall
logical, allocatable :: firstcallixiy(:,:)

data firstcall /.true./
      
SAVE

if (firstcall) then
  allocate (firstcallixiy(1:nx,1:ny))
  allocate (out_unita(1:nx,1:ny))
  allocate (day_old   (1:nx, 1:ny))
  allocate (tsteps    (1:nx, 1:ny))  
  allocate (var       (1:nvars, 1:nx, 1:ny, 1:M+1) )
  allocate (var_scalar(1:nvars_scalar, 1:nx, 1:ny ) )
  allocate (accum_var (1:nvars, 1:nx, 1:ny, 1:M+1) )
  allocate (accum_var_scalar (1:nvars_scalar, 1:nx, 1:ny) )
  firstcallixiy(:,:) = .true.
  out_unita(:,:) = lake_day_out_unit_min
  day_old  (:,:) = day
  tsteps   (:,:) = 0.d0  
  accum_var(:,:,:,:) = 0.d0
  accum_var_scalar(:,:,:) = 0.d0
  var       (:,:,:,:) = 0.d0
  var_scalar(:,:,:) = 0.d0
endif

if (firstcallixiy(ix,iy)) then
  call CHECK_UNIT(lake_day_out_unit_min,lake_day_out_unit_max,out_unita(ix,iy))
  write (coords_point,'(2i3)') ix, iy
  open(out_unita(ix,iy),file=outpath(1:len_trim(outpath))//'daily/'// &
  & 'fluxes'//coords_point//'.dat', status='unknown')
  write (out_unita(ix,iy),*) '1 - year' 
  write (out_unita(ix,iy),*) '2 - month'
  write (out_unita(ix,iy),*) '3 - discharge in x direction, m**3/s'
  write (out_unita(ix,iy),*) '4 - discharge in y direction, m**3/s' 
endif

!Vector variables
var(1,ix,iy,1:M+1) = Tw1 (1:M+1)
var(2,ix,iy,1:M+1) = Sal1(1:M+1)
var(3,ix,iy,1:M)   = E1  (1:M)
var(4,ix,iy,1:M)   = eps1(1:M)
var(5,ix,iy,1:M)   = KT(1:M)/cw_m_row0
var(6,ix,iy,1:M+1) = sqrt(u1(1:M+1)**2 + v1(1:M+1)**2)
var(7,ix,iy,1:M)   = Ri(1:M)

!Scalar variables
var_scalar(1,ix,iy) = discharge(1)
var_scalar(2,ix,iy) = discharge(2)

accum_var(:,ix,iy,:) = accum_var(:,ix,iy,:) + var(:,ix,iy,:)
accum_var_scalar(:,ix,iy) = accum_var_scalar(:,ix,iy) + var_scalar(:,ix,iy)
tsteps(ix,iy) = tsteps(ix,iy) + 1.d0

if (day_old(ix,iy)/=day) then
  accum_var(:,ix,iy,:) = accum_var(:,ix,iy,:)/tsteps(ix,iy)
  call DATEMINUS (2,year,month,day,hour,year1,month1,day1,hour1)
  call TIMESTR (8,year1,month1,day1,hour1,timestring)
  call CHECK_UNIT(lake_day_out_unit_min,lake_day_out_unit_max,out_unit)
  write (coords_point, '(2i3)') ix, iy
  open(out_unit, file=outpath(1:len_trim(outpath))//'daily/'// &
  & 'Profiles'//coords_point//timestring//'.dat',status='unknown')
  write (out_unit,*) '1 - depth, m' 
  write (out_unit,*) '2 - temperature, C'
  write (out_unit,*) '3 - salinity, kg/kg' 
  write (out_unit,*) '4 - turbulent kinetic energy, m**2/s**2'
  write (out_unit,*) '5 - dissipation rate, m**2/s**3'
  write (out_unit,*) '6 - kinematic heat conductance, m**2/s'
  write (out_unit,*) '7 - modulus of horizontal velocity, m/s'
  write (out_unit,*) '8 - Richardson number, n/d'
  do i = 1, M+1 
    write (out_unit,'(f12.6,f11.5,6e12.4)') &
    &  -dzeta_int(i)*h1, accum_var(1:nvars,ix,iy,i)
  enddo
  close(out_unit)

  accum_var_scalar(:,ix,iy) = accum_var_scalar(:,ix,iy)/tsteps(ix,iy)
  read(year1,*) work1
  read(month1,*) work2
  read(day1,*) work3
  write (out_unita(ix,iy),fmt=*) & !, fmt = formatline) &
  & int(work1), int(work2), int(work3), accum_var_scalar(1:nvars_scalar,ix,iy)

  day_old(ix,iy) = day
  accum_var(:,ix,iy,:) = 0.d0
  accum_var_scalar(:,ix,iy) = 0.d0
  tsteps(ix,iy) = 0.d0
endif

if (firstcall) firstcall=.false.
if (firstcallixiy(ix,iy)) firstcallixiy(ix,iy) = .false.
END SUBROUTINE DAY_OUT


SUBROUTINE HOUR_OUT(ix,iy,nx,ny,year,month,day,hour,time)

use ARRAYS_WATERSTATE, only : Tw1, Sal1
use ARRAYS_TURB!, only : E1, eps1, row, H_mixed_layer, H_entrainment, Wedderburn
use ARRAYS_BATHYM, only : h1
use ARRAYS_GRID, only : dzeta_int, dzeta_05int
use ARRAYS, only : u1, v1
use TIMEVAR, only : hour_sec
use ARRAYS_OXYGEN, only: oxyg, DIC
use ARRAYS_METHANE, only : qwater
use PHYS_FUNC, only : HC_CORR_CARBEQUIL
use PHYS_CONSTANTS, only : Kelvin0
use METH_OXYG_CONSTANTS, only : &
& molm3tomgl_o2, molm3tomcM, molm3tomcM, pH

use DRIVING_PARAMS, only : M, scale_output

use INOUT_PARAMETERS, only : &
& lake_hour_out_unit_min, &
& lake_hour_out_unit_max

implicit none

! Input variables
! Reals
real(kind=ireals), intent(in) :: hour
real(kind=ireals), intent(in) :: time

! Integers
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: nx, ny
integer(kind=iintegers), intent(in) :: year, month, day

! Local variables
! Reals
real(kind=ireals), allocatable :: accum_var (:,:,:,:)
real(kind=ireals), allocatable :: var       (:,:,:,:)
real(kind=ireals), allocatable :: var_scalar(:,:,:)
real(kind=ireals), allocatable :: accum_var_scalar(:,:,:)

real(kind=ireals), external :: DZETA

! Integers
integer(kind=iintegers), parameter :: n_var = 30
integer(kind=iintegers), parameter :: n_var_scalar = 10
integer(kind=iintegers), allocatable :: hour_old(:,:)
integer(kind=iintegers), allocatable :: tsteps(:,:)
integer(kind=iintegers), allocatable :: n_unit(:,:)
integer(kind=iintegers) :: i
integer(kind=iintegers) :: out_unit = lake_hour_out_unit_min

character :: month1*2
character :: year1*4
character :: day1*2
character :: hour1*2
character :: timestring*10
character :: coords_point*6
character :: format_char*100

logical :: firstcall = .true.
logical, allocatable :: firstcall_ixiy(:,:)

      
SAVE

if (firstcall) then
  allocate (accum_var (n_var,1:nx,1:ny,1:M+1) )
  allocate (var       (n_var,1:nx,1:ny,1:M+1) )
  allocate (accum_var_scalar (n_var_scalar,1:nx,1:ny) )
  allocate (var_scalar       (n_var_scalar,1:nx,1:ny) )  
  allocate (tsteps    (1:nx,1:ny) )
  allocate (hour_old  (1:nx,1:ny) )
  allocate (n_unit(1:nx,1:ny))
  allocate (firstcall_ixiy(1:nx,1:ny))
  accum_var(:,:,:,:) = 0.d0
  accum_var_scalar(:,:,:) = 0.d0
  var(:,:,:,:) = 0.d0
  var_scalar(:,:,:) = 0.d0
  tsteps(:,:) = 0
  hour_old(:,:) = int(hour)
  firstcall_ixiy(:,:) = .true.
  n_unit(:,:) = out_unit + 1
endif

var(1,ix,iy,1:M)   = PEMF          (1:M)
var(2,ix,iy,1:M+1) = PDENS_DOWN    (1:M+1)
var(3,ix,iy,1:M+1) = PT_DOWN       (1:M+1)
var(4,ix,iy,1:M+1) = PSAL_DOWN     (1:M+1)
var(5,ix,iy,1:M)   = k_turb_T_flux (1:M)
var(6,ix,iy,1:M)   = T_massflux    (1:M)
var(7,ix,iy,1:M+1) = row           (1:M+1)
var(8,ix,iy,1:M+1) = Tw1           (1:M+1)
var(9,ix,iy,1:M+1) = Sal1          (1:M+1)
var(10,ix,iy,1:M)  = E1            (1:M)
var(11,ix,iy,1:M)  = eps1          (1:M)

if (scale_output%par == 1) then
  var(12,ix,iy,1)    = H_mixed_layer ! Scale
  var(13,ix,iy,1)    = Buoyancy0     ! Scale
  var(14,ix,iy,1)    = w_conv_scale  ! Scale
  var(15,ix,iy,1)    = T_conv_scale  ! Scale
elseif (scale_output%par == 0) then
  var(12,ix,iy,1)    = 1.d0 ! No scaling
  var(13,ix,iy,1)    = 1.d0 ! No scaling
  var(14,ix,iy,1)    = 1.d0 ! No scaling
  var(15,ix,iy,1)    = 1.d0 ! No scaling
else
  print*, 'Scale_output is ', scale_output%par, &
  & ', must be 0 or 1: STOP'  
  STOP
endif
      
var(16,ix,iy,1:M)  = S              (1:M)
var(17,ix,iy,1:M)  = Gen            (1:M)
var(18,ix,iy,1:M)  = Gen_seiches    (1:M)
var(19,ix,iy,1:M)  = TKE_turb_trans (1:M)
      
var(20,ix,iy,2:M)  = k3_mid         (2:M)

var(21,ix,iy,1:M+1)  = u1(1:M+1)
var(22,ix,iy,1:M+1)  = v1(1:M+1)
var(23,ix,iy,1:M+1)  = sqrt(u1(1:M+1)*u1(1:M+1) + v1(1:M+1)*v1(1:M+1))

!Biogeochemical substances
var(24,ix,iy,1:M+1)  = oxyg(1:M+1,1)*molm3tomgl_o2
do i = 1, M+1 
  var(25,ix,iy,i)  = DIC(i,1) / HC_CORR_CARBEQUIL(Tw1(i)+Kelvin0,pH) * molm3tomcM
enddo
var(26,ix,iy,1:M+1)  = qwater(1:M+1,1)*molm3tomcM

!Coefficients of turbulent viscosity
var(27,ix,iy,1:M)   = k2             (1:M  )
var(28,ix,iy,1:M)   = viscturb_ziriv (1:M  )
var(29,ix,iy,1:M)   = eps_ziriv      (1:M  )
var(30,ix,iy,1:M+1) = speed_ziriv    (1:M+1)

var_scalar(1,ix,iy) = S_integr_positive
var_scalar(2,ix,iy) = S_integr_negative
var_scalar(3,ix,iy) = Gen_integr
var_scalar(4,ix,iy) = eps_integr
var_scalar(5,ix,iy) = TKE_balance
var_scalar(6,ix,iy) = H_entrainment
var_scalar(7,ix,iy) = TKE_turb_trans_integr
var_scalar(8,ix,iy) = Wedderburn
var_scalar(9,ix,iy) = LakeNumber
var_scalar(10,ix,iy) = Rossby_rad

accum_var_scalar(:,ix,iy) = accum_var_scalar(:,ix,iy) + var_scalar(:,ix,iy)
accum_var(:,ix,iy,:) = accum_var(:,ix,iy,:) + var(:,ix,iy,:)
tsteps(ix,iy) = tsteps(ix,iy) + 1

if (int(hour)/=hour_old(ix,iy)) then
  accum_var(:,ix,iy,:) = accum_var(:,ix,iy,:) / real(tsteps(ix,iy))
  accum_var_scalar(:,ix,iy) = accum_var_scalar(:,ix,iy) / real(tsteps(ix,iy))
  call DATEMINUS(3,year,month,day,hour,year1,month1,day1,hour1)
  call TIMESTR(10,year1,month1,day1,hour1,timestring)
  call CHECK_UNIT(lake_hour_out_unit_min,lake_hour_out_unit_max,out_unit)
  write (coords_point, '(2i3)') ix, iy
  
! Writing to the file Profiles<yyyymmddhh>.dat  
  open(out_unit, file=outpath(1:len_trim(outpath))// &
  & 'hourly/'//'Profiles'//coords_point//timestring//'.dat', &
  &  status='unknown')
  write (out_unit,*) '1 - depth, normalized' 
  write (out_unit,*) '2 - temperature, normalized'
  write (out_unit,*) '3 - salinity, kg/kg' 
  write (out_unit,*) '4 - water density, kg/m**3'  
  write (out_unit,*) '5 - turbulent kinetic energy, normalized'
  write (out_unit,*) '6 - dissipation rate, normalized'
  write (out_unit,*) '7 - eddy viscosity for TKE, m**2/s'
  write (out_unit,*) '8  - mass flux, m/s'
  write (out_unit,*) '9  - downdraft temperature, C' 
  write (out_unit,*) '10 - downdraft salinity, kg/kg' 
  write (out_unit,*) '11 - downdraft density, kg/m**3'
  write (out_unit,*) '12 - x-component of speed, m/s'
  write (out_unit,*) '13 - y-component of speed, m/s'
  write (out_unit,*) '14 - modulus of speed, m/s'
  write (out_unit,*) '15 - oxygen, mg/l'
  write (out_unit,*) '16 - co2, mcmol/l'
  write (out_unit,*) '17 - ch4, mcmol/l'
  write (out_unit,*) '18 - eddy viscosity from k-epsilon model, m**2/s'
  write (out_unit,*) '19 - eddy viscosity from Ziriv model, m**2/s'
  write (out_unit,*) '20 - dissipation rate from Ziriv model, m**2/s**3'
  write (out_unit,*) '21 - longitudinal speed from Ziriv model, m/s'
  do i=1,M 
    write (out_unit,'(f14.6,f12.5,e10.3,f10.3,3e10.3,4e16.7,3f11.5,3e16.7,3e12.3,f11.5)')      &
    & -dzeta_int(i)*h1                         /accum_var(12,ix,iy,1),  &
    & (accum_var(8,ix,iy,i) - scale_output%par*maxval(accum_var(8,ix,iy,:)) ) &
    &  /accum_var(15,ix,iy,1), &
    & accum_var(9,ix,iy,i),                                               &
    & accum_var(7,ix,iy,i),                                               &
    & accum_var(10,ix,iy,i)    / (accum_var(13,ix,iy,1)*accum_var(12,ix,iy,1) /       &
    &  accum_var(14,ix,iy,1) ),                                           &
    & accum_var(11,ix,iy,i)    / accum_var(13,ix,iy,1),                   &
    & accum_var(20,ix,iy,i),                                              & !accum_var(10,i)**2 / accum_var(11,i),
    & accum_var(1,ix,iy,i),                                               &
    & accum_var(3,ix,iy,i),                                               &
    & accum_var(4,ix,iy,i),                                               &
    & accum_var(2,ix,iy,i),                                               &
    & accum_var(21,ix,iy,i),                                              &
    & accum_var(22,ix,iy,i),                                              &
    & accum_var(23,ix,iy,i),                                              &
    & accum_var(24,ix,iy,i),                                              &
    & accum_var(25,ix,iy,i),                                              &
    & accum_var(26,ix,iy,i),                                              &
    & accum_var(27,ix,iy,i),                                              &
    & accum_var(28,ix,iy,i),                                              &
    & accum_var(29,ix,iy,i),                                              &
    & accum_var(30,ix,iy,i)
  enddo 
  close (out_unit)
  
! Writing to the file EDMF_profiles<yyyymmddhh>.dat
  open(out_unit, file=outpath(1:len_trim(outpath))//  &
  & 'hourly/'//'EDMF_profiles'//coords_point//timestring//'.dat',   &
  & status='unknown')
  write (out_unit,*) '1 - depth, normalized by mixed layer depth' 
  write (out_unit,*) '2 - "k - flux" of temperature, normalized'
  write (out_unit,*) '3 - mass flux of temperature,  normalized'
  write (out_unit,*) '4 - total flux of temperature, normalized'
  write (out_unit,'( 4(i10,6x) )') 1,2,3,4
  do i=1, M
    write (out_unit,'(f18.6,3e16.7)')                   &
    & -dzeta_05int(i)*h1/ accum_var(12,ix,iy,1),          &
    & accum_var(5,ix,iy,i)/(accum_var(14,ix,iy,1)*accum_var(15,ix,iy,1)), &
    & accum_var(6,ix,iy,i)/(accum_var(14,ix,iy,1)*accum_var(15,ix,iy,1)), &
    & (accum_var(5,ix,iy,i) + accum_var(6,ix,iy,i) ) /              &
    & (accum_var(14,ix,iy,1)*accum_var(15,ix,iy,1))
  enddo
  close (out_unit)
  
! Writing to the file TKE_budget<yyyymmddhh>.dat
  open(out_unit, file=outpath(1:len_trim(outpath))//    &
  & 'hourly/'//'TKE_budget'//coords_point//timestring//'.dat',        &
  &  status='unknown')
  write (out_unit,*) '1 - depth, normalised with mixed layer depth'
  write (out_unit,*) '2 - shear production, normalised with surface buoyancy flux'
  write (out_unit,*) '3 - seiches production, normalised with surface buoyancy flux'
  write (out_unit,*) '4 - buoyancy source,  normalised with surface buoyancy flux'
  write (out_unit,*) '5 - dissipation rate, normalised with surface buoyancy flux'
  write (out_unit,*) '6 - turbulent transport, normalised with surface buoyancy flux'
  write (out_unit,'( 6(i10,6x) )') 1,2,3,4,5,6
  do i = 1, M
    write (out_unit,'(f18.6,5e16.7)')          &
    & -dzeta_05int(i)*h1/ accum_var(12,ix,iy,1), &
    & accum_var(17,ix,iy,i) / accum_var(13,ix,iy,1), &
    & accum_var(18,ix,iy,i) / accum_var(13,ix,iy,1), &
    & accum_var(16,ix,iy,i) / accum_var(13,ix,iy,1), &
    & accum_var(11,ix,iy,i) / accum_var(13,ix,iy,1), &
    & accum_var(19,ix,iy,i) / accum_var(13,ix,iy,1)
  enddo
!  write (out_unit, '(a)')
  close (out_unit)
  
  if (firstcall_ixiy(ix,iy)) then
    call CHECK_UNIT(lake_hour_out_unit_min,lake_hour_out_unit_max,n_unit(ix,iy))
    open(n_unit(ix,iy), file=outpath(1:len_trim(outpath))//           &
    & 'hourly/'//'TKE_integr_conv'//coords_point//timestring//'.dat', &
    &  status='unknown')
    write (n_unit(ix,iy),*) 'Col. 1 - year'
    write (n_unit(ix,iy),*) 'Col. 2 - month'
    write (n_unit(ix,iy),*) 'Col. 3 - day'
    write (n_unit(ix,iy),*) 'Col. 4 - hour'
    write (n_unit(ix,iy),*) 'Col. 5 - the time from the start of integration, hours'
    write (n_unit(ix,iy),*) 'Col. 6 - H_entrainment, m    '
    write (n_unit(ix,iy),*) 'Col. 7 - B_integr+,   m**3/s**3'
    write (n_unit(ix,iy),*) 'Col. 8 - B_integr-,   m**3/s**3'
    write (n_unit(ix,iy),*) 'Col. 9 - S_integr,    m**3/s**3'
    write (n_unit(ix,iy),*) 'Col. 10 - eps_integr,  m**3/s**3'
    write (n_unit(ix,iy),*) 'Col. 11 - TKE turb trans integr,  m**3/s**3'
    write (n_unit(ix,iy),*) 'Col. 12 - TKE_balance, m**3/s**3'
    write (n_unit(ix,iy),*) 'Col. 13 - Wedderburn number, n/d'
    write (n_unit(ix,iy),*) 'Col. 14 - Lake number, n/d'
    write (n_unit(ix,iy),*) 'Col. 15 - Internal Rossby radius, m'
    firstcall_ixiy(ix,iy) = .false.
  endif
  
  format_char = '(3i7,f8.2,f13.2,f10.2,9e15.6)'
  write (n_unit(ix,iy), format_char) &
  &                    year,month,day,hour,time/hour_sec, &
  &                    accum_var_scalar(6,ix,iy), & 
  &                    accum_var_scalar(1,ix,iy), &
  &                    accum_var_scalar(2,ix,iy), &
  &                    accum_var_scalar(3,ix,iy), &
  &                    accum_var_scalar(4,ix,iy), &
  &                    accum_var_scalar(7,ix,iy), &
  &                    accum_var_scalar(5,ix,iy), &
  &                    accum_var_scalar(8,ix,iy), &
  &                    accum_var_scalar(9,ix,iy), &
  &                    accum_var_scalar(10,ix,iy)
  
  hour_old(ix,iy) = int(hour)
  accum_var(:,ix,iy,:) = 0.d0
  accum_var_scalar(:,ix,iy) = 0.d0
  tsteps(ix,iy)  = 0
   
endif

if (firstcall) firstcall = .false.
!if (firstcall_ixiy(ix,iy)) firstcall_ixiy(ix,iy) = .false.
END SUBROUTINE HOUR_OUT          
      


SUBROUTINE EVERYSTEP_OUT(ix,iy,nx,ny)

use ARRAYS!, only : time, nstep

use ARRAYS_WATERSTATE!, only : Tw1, Sal1 
use ARRAYS_BATHYM!, only : h1
use ARRAYS_GRID!, only : dzeta_int, M

use ARRAYS_TURB!, only : &
! & S_integr_positive, &
! & S_integr_negative, &
! & Gen_integr, &
! & eps_integr, &
! & TKE_balance, &
! & E_integr, &
! & H_entrainment, &
! & E1, eps1, &
! & k_turb_T_flux

use DRIVING_PARAMS, only : everystep, M

use INOUT_PARAMETERS, only : &
& lake_everystep_out_unit_min, &
& lake_everystep_out_unit_max

implicit none

! Input variables
! Integers
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: nx, ny

! Local variables
! Reals
real(kind=ireals), external :: DZETA
real(kind=ireals), parameter :: ACC = 1.d-20
real(kind=ireals), parameter :: hour_sec = 60.*60.

! Integers
integer(kind=iintegers), allocatable :: n_unit(:,:,:)
integer(kind=iintegers) :: i ! Loop index

! Characters
character :: coords_point*6
character :: format_char*100

! Logicals
logical, allocatable :: firstcall(:,:)

      
SAVE

if (.not.allocated(firstcall)) then
  allocate (firstcall(1:nx, 1:ny) )
  allocate (n_unit   (1:2, 1:nx, 1:ny) )
  firstcall = .true.
  n_unit = lake_everystep_out_unit_min
endif  
      
if (firstcall(ix,iy)) then
  write (coords_point, '(2i3)') ix, iy
  
  if (everystep%par /= 2) then
    call CHECK_UNIT(lake_everystep_out_unit_min,lake_everystep_out_unit_max, &
    & n_unit(1,ix,iy))
    open (n_unit(1,ix,iy), file=outpath(1:len_trim(outpath))//'everystep/'// &
    & 'Profiles'//coords_point//'.dat',  status='unknown')
    write (n_unit(1,ix,iy),*) '1 - depth, m' 
    write (n_unit(1,ix,iy),*) '2 - temperature, C'
    write (n_unit(1,ix,iy),*) '3 - salinity, kg/kg' 
    write (n_unit(1,ix,iy),*) '4 - turbulent kinetic energy, m**2/s**2'
    write (n_unit(1,ix,iy),*) '5 - disspation rate, m**2/s**3'
    write (n_unit(1,ix,iy),*) '6 - eddy diffusivity (TKE**2/dissipation), m**2/s'
    write (n_unit(1,ix,iy),*) '7 - k-flux of temperature, K*m/s'
  endif
    
  call CHECK_UNIT(lake_everystep_out_unit_min,lake_everystep_out_unit_max, &
  & n_unit(2,ix,iy))
  open (n_unit(2,ix,iy), file=outpath(1:len_trim(outpath))//'everystep/'// &
  & 'TKE_integr'//coords_point//'.dat',  status='unknown')
  write (n_unit(2,ix,iy),*) '1 - timestep              '
  write (n_unit(2,ix,iy),*) '2 - time,        hours    '
  write (n_unit(2,ix,iy),*) '3 - B_integr+,   m**3/s**3'
  write (n_unit(2,ix,iy),*) '4 - B_integr-,   m**3/s**3'
  write (n_unit(2,ix,iy),*) '5 - S_integr,    m**3/s**3'
  write (n_unit(2,ix,iy),*) '6 - eps_integr,  m**3/s**3'
  write (n_unit(2,ix,iy),*) '7 - TKE_balance, m**3/s**3'
  write (n_unit(2,ix,iy),*) '8 - E_integr,    m**3/s**2'
endif

if (everystep%par /= 2) then      
  write (n_unit(1,ix,iy),*) 'nstep = ', nstep
  format_char = '(f10.6,f9.3,5e12.4)'
  do i=1,M 
    write (n_unit(1,ix,iy), format_char) &
    & -dzeta_int(i)*h1,Tw1(i),Sal1(i),E1(i),eps1(i),  &
    & E1(i)**2/(eps1(i)+ACC), k_turb_T_flux(i)
  enddo
endif

format_char = '(i10, 2f10.2, 6e15.6)'
write(n_unit(2,ix,iy), format_char) &
& nstep, time/hour_sec, H_entrainment, S_integr_positive, S_integr_negative, &
& Gen_integr, eps_integr, TKE_balance, E_integr
      
if (firstcall(ix,iy)) firstcall(ix,iy)=.false.
END SUBROUTINE EVERYSTEP_OUT
      
      
SUBROUTINE SERIES_OUT(ix,iy,nx,ny,year,month,day,hour,tsw)

use DRIVING_PARAMS, only : M, ns, Mice, dt_out, time_series_aver
use NUMERIC_PARAMS , only : &
& ms, small_value

use ARRAYS_TURB, only : row, H_mixed_layer, H_entrainment, &
& signwaveheight, w_conv_scale, u_star, Ri_bulk, maxN, i_maxN, &
& ThermThick, ReTherm, RiTherm, TKE_budget_terms

use ARRAYS_WATERSTATE, only : Tw1, Ti1, lamw, salice

use ARRAYS_GRID, only : nsoilcols

use ARRAYS_BATHYM, only : &
& h1,hx1,hx2,hy1,hy2, &
& hx1t,hx2t,hy1t,hy2t, &
& l1,hs1,ls1,  &
& voldef, vol,     &
& area_int, bathymwater

use ARRAYS_METHANE, only : &
& fbbleflx_ch4_sum, &
& fbbleflx_ch4,    &
& febul0, fplant,   &
& fdiff_lake_surf, &
& fdiff_lake_bot, &
& febul_lake_surf, &
& febul_lake_bot, &
& fdiffbot, foutflow, finflow, &
& qwater, qsoil, &
& rprod_total_newC, &
& rprod_total_oldC, &
& h_talik, lammeth, &
& qwateroxidtot, qwateroxidML

use ARRAYS_OXYGEN, only : &
& fbbleflx_co2_sum, &
& fbbleflx_co2,    &
& fbbleflx_o2_sum, &
& fbbleflx_o2, &
& oxyg, &
& fco2, fo2, fco2ebul_bot, fco2ebul_surf, fPOCD_sedim, &
& pools_lake, srcs_lake, &
& npool_DIC, npool_DOC, npool_POCL, npool_POCD, &
& nsrc_prodox, nsrc_resp, nsrc_BOD, nsrc_SOD

use ARRAYS_SOIL, only : Tsoil1, lsm

use ARRAYS, only:  &
& Tskin,       &
& zinv,            &
& time,            &
& T_0dim,           &
& roughness,emissivity,albedo,aM,bM,relhums, & 
& gas

use ATMOS,  only:    &
& eflux0_kinem,      &
& eflux0,            &
& turb_density_flux, &
& hw, xlew, botflux, &
& tau, windwork, surfrad, &
& discharge, velfrict_bot, &
& shortwave

use INOUT_PARAMETERS, only : &
& lake_series_out_unit_min, &
& lake_series_out_unit_max

use METH_OXYG_CONSTANTS, only : &
& molmass_ch4, mfs

use TIMEVAR, only : &
hour_sec, day_sec, hour_min

use PHYS_CONSTANTS, only : &
Kelvin0, cw_m_row0, roa0, sabs, row0

use DZETA_MOD, only : VARMEAN

use ARRAYS_GRID, only : gsp

use SNOWSOIL_MOD, only : Tsn, itop

use TRIBUTARIES, only : inout_fluxes, &
& nDIC, nDOC, nPOCL, nPOCD, nmeth, noxyg, nphosph

use LB_NCOUT, only : LB_ASCII_OUT, LB_HEADER, LB_ROW

implicit none

! Input variables
! Reals
real(kind=ireals), intent(in) :: tsw
real(kind=ireals), intent(in) :: hour

! Integers
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: nx, ny
integer(kind=iintegers), intent(in) :: year, month, day

! Local variables
! Reals

real(kind=ireals), parameter :: small_number = 1.d-5

real(kind=ireals) :: T_mean
real(kind=ireals) :: methfluxML, methfluxshalsed

real(kind=ireals), allocatable :: tsteps(:,:), work(:)
real(kind=ireals), allocatable :: var_scalar (:,:,:) 
real(kind=ireals), allocatable :: accum_var_scalar(:,:,:)

! Integers
integer(kind=iintegers), parameter :: nfiles = 6
integer(kind=iintegers), parameter :: numaccum = 5 ! The number of scalar variables being accumulated (averaged)
integer(kind=iintegers), allocatable :: n_unit(:,:,:)
integer(kind=iintegers), allocatable :: count_out(:,:)
integer(kind=iintegers) :: i, n

! Characters
character :: coords_point*6
character :: format_char*100, workchar*30, workchar1*30

! Logicals
logical, allocatable :: firstcall(:,:)

type :: work_aver_out_type
  integer(kind=iintegers)              :: nvals
  real   (kind=ireals   ), allocatable :: vals(:), vals_sum(:)
  character(len=100)                   :: format_char
end type work_aver_out_type
type(work_aver_out_type), allocatable :: work_aver_out(:,:,:)

SAVE

if (.not.allocated(firstcall)) then
  allocate ( firstcall (1:nx, 1:ny) )
  allocate ( count_out (1:nx, 1:ny) )
  allocate ( tsteps    (1:nx, 1:ny) )  
  allocate ( n_unit        (1:nfiles, 1:nx, 1:ny) )
  allocate ( work_aver_out (1:nfiles, 1:nx, 1:ny) )
  allocate ( var_scalar       (1:numaccum, 1:nx, 1:ny) )
  allocate ( accum_var_scalar (1:numaccum, 1:nx, 1:ny) )

  tsteps(:,:) = 0.d0
  var_scalar(:,:,:) = 0.d0
  accum_var_scalar(:,:,:) = 0.d0

  firstcall(:,:) = .true.
  n_unit = lake_series_out_unit_min
endif  
            
if_firstcall_ixiy : if (firstcall(ix,iy)) then

  write (coords_point, '(2i3)') ix, iy
  write (workchar    , '(a)'  ) '(3i7,f10.4,f15.4,' !formats for year, month, day, hour, the time from start (hours)
  n = len_trim(workchar)

  i = 1
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(i,ix,iy))
  open (n_unit(i,ix,iy),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & 'layers'//coords_point//'.dat',  status='unknown')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 1 - year')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 2 - month')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 3 - day')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 4 - hour')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 5 - the time from the start of integration, hours')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 6 - water layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 7 - W mixed layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 8 - E mixed layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 9 - S mixed layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 10 - N mixed layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 11 - W lower layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 12 - E lower layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 13 - S lower layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 14 - N lower layer thickness, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 15 - ice layer thickness,   m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 16 - snow layer thickness,  m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 17 - bottom ice thickness,  m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 18 - reservoir volume,  m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 19 - volume deficit (accumulated),  m**3')

  work_aver_out(i,ix,iy)%nvals = 14
  work_aver_out(i,ix,iy)%format_char = workchar(1:n)//'f10.4,8e12.4,3f10.4,1x,e12.4,1x,f10.1)'
  

  i = 2
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(i,ix,iy))
  open (n_unit(i,ix,iy),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & 'T_fluxes'//coords_point//'.dat',status='unknown')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 1 - year')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 2 - month')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 3 - day')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 4 - hour')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 5 - the time from the start of integration, hours')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 6 - surface temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 7 - water skin temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 8 - water surface temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 9 - mean temperature of water coloumn, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 10 - maximal temperature in the water coloumn, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 11 - zero-dimensional model temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 12 - upper ice surface temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 13 - upper snow surface temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 14 - sensible heat flux,    W/m**2')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 15 - latent heat flux,      W/m**2')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 16 - downward heat flux at the upper lake surface, W/m**2')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 17 - downward heat flux at the lake bottom, W/m**2')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 18 - friction velocity at the surface (waterside), m/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 19 - friction velocity at the bottom, m/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 20 - wind work at the water surface, W/m**2')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 21 - albedo of the lake-atmosphere interface, n/d')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 22 - shortwave radiation penetrated below surface, W/m**2')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 23 - significant wave height, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 24 - bottom ice salinity, kg/kg')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 25 - discharge in x direction, m**3/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 26 - discharge in y direction, m**3/s')
  call LB_HEADER(n_unit(i,ix,iy),'Cols. 27... - top soil columns temperature, m')

  work_aver_out(i,ix,iy)%nvals = 21 + nsoilcols
  format_char = workchar(1:n)//'15e14.5,2f11.2,f11.4,f6.2,2e14.5'
  ! Formatting for possible multiple soil columns in a lake
  write(workchar1,'(i2)') nsoilcols
  format_char = format_char(1:len_trim(format_char))//','// &
  &             workchar1  (1:len_trim(workchar1)  )//'f11.4)'
  work_aver_out(i,ix,iy)%format_char = format_char

 
  i = 3 
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(i,ix,iy))
  open (n_unit(i,ix,iy),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & 'conv_series'//coords_point//'.dat',status='unknown')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 1 - year')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 2 - month')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 3 - day')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 4 - hour')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 5 - the time from the start of integration, hours')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 6 - surface temperature, C')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 7 - surface density, kg/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 8 - heat flux downwards at the surface, K*m/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 9 - turbulent density flux at the surface, kg/m**2/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 10 - inversion depth (mass flux diagnostics),    m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 11 - mixed layer depth,                          m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 12 - entrainment depth,                          m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 13 - convective velocity scale,                m/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 14 - friction velocity,                        m/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 15 - -w_star/u_star,                            m/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 16 - bulk Richardson number,                    n/d')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 17 - maximal Brunt-Vaisala frequency,       s**(-1)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 18 - minimal thermal conductance,            m**2/s')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 19 - thermocline thickness,                       m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 20 - Reynolds number in thermocline ,           n/d')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 21 - bulk Richardson number in thermocline,     n/d')

  work_aver_out(i,ix,iy)%nvals = 16
  work_aver_out(i,ix,iy)%format_char = workchar(1:n)//'2f10.4,2e15.5,5f10.4,7e15.5)'
      

  i = 4
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(i,ix,iy))
  open (n_unit(i,ix,iy),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & 'methane_series'//coords_point//'.dat',status='unknown')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 1 - year')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 2 - month')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 3 - day')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 4 - hour')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 5 - the time from the start of integration, hours')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 6 - the talik depth, m')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 7 - lake surface methane concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 8 - lake bottom methane concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 9 - soil bottom methane concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 10 - lake surface oxygen concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 11 - lake bottom oxygen concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 12 - total methane production due to young C decomposition, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 13 - total methane production due to old C decomposition, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 14 - methane ebullition flux averaged over the lake bottom, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 15 - methane ebullition flux at the surface, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 16 - methane plant-mediated flux at the lake bottom, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 17 - methane diffusion flux averaged over the lake bottom, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 18 - methane turbulent flux at the lake surface, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 19 - methane ebullition flux averaged over the lake bottom, upwards, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 20 - methane ebullition flux at the surface, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 21 - methane plant-mediated flux at the lake bottom, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 22 - methane diffusion flux averaged over lake bottom, upwards, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 23 - methane turbulent flux at the lake surface, upwards, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 24 - methane turbulent flux at the bottom of mixed layer normalized by &
  & surface area, upwards, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 25 - methane flux from sediments in the mixed layer normalized by &
  & surface area, upwards, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 26 - methane bubble flux at the bottom of the mixed layer normalized by &
  & surface area, upwards, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 27 - total methane oxidation in water normalized by &
  & surface area, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 28 - methane oxidation in mixed layer normalized by &
  & surface area, mg/(m**2*day)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 29 - co2 turbulent flux at the lake surface, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 30 - co2 ebullition flux at the surface, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 31 - oxygen turbulent flux at the lake surface, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 32 - oxygen ebullition flux at the surface, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 33 - methane flux by inlets, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 34 - methane flux through outlet, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 35..35+ns-1 - methane ebullition flux at the surface from different soil columns, mg/(m**2*day)')

  work_aver_out(i,ix,iy)%nvals = 29 + nsoilcols
  format_char = workchar(1:n)//'f8.2,28e15.5'
  ! Formatting for possible multiple soil columns in a lake
  write(workchar1,'(i2)') nsoilcols
  format_char = format_char(1:len_trim(format_char))//','// &
  &             workchar1  (1:len_trim(workchar1)  )//'e15.5)'
  work_aver_out(i,ix,iy)%format_char = format_char

 
  i = 5 
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(i,ix,iy))
  open (n_unit(i,ix,iy),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & 'TKE_budget_terms_series'//coords_point//'.dat',status='unknown')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 1  - year')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 2  - month')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 3  - day')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 4  - hour')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 5  - the time from the start of integration, hours')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 6  - TKE tendency, m**2/s**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 7  - TKE turbulent transport, m**2/s**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 8  - TKE shear production, m**2/s**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 9  - TKE buoyancy production/sink, m**2/s**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 10 - TKE dissipation rate, m**2/s**3')

  work_aver_out(i,ix,iy)%nvals = 5
  work_aver_out(i,ix,iy)%format_char = workchar(1:n)//'5E15.5)'

  i = 6
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(i,ix,iy))
  open (n_unit(i,ix,iy),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & 'carbon_series'//coords_point//'.dat',status='unknown')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 1  - year')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 2  - month')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 3  - day')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 4  - hour')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 5  - the time from the start of integration, hours')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 6  - lake surface DIC concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 7  - lake bottom DIC concentration, mol/m**3')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 8  - DIC pool in a lake, normalized by surface area, mol/(m**2)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 9  - DOC pool in a lake, normalized by surface area, mol/(m**2)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 10 - POCL pool in a lake, normalized by surface area, mol/(m**2)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 11 - POCD pool in a lake, normalized by surface area, mol/(m**2)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 12 - DIC production due to aerobic heterotrophic respiration in sediments (SOD), mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 13 - DIC production due to aerobic POCD decay in the water column, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 14 - DIC production due to autotrophic respiration by POCL in the water column, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 15 - DIC sink due to photosynthesis in the water column, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 16 - POCD sedimentation to a lake bottom, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 17 - CO2 turbulent flux at the lake surface, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 18 - CO2 ebullition flux at the surface, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 19 - CO2 ebullition flux averaged over the lake bottom, upwards, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 20 - CO2 plant-mediated flux at the lake bottom, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 21 - DIC flux by inlets, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 22 - DOC flux by inlets, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 23 - POCL flux by inlets, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 24 - DIC flux through outlet, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 25 - DOC flux through outlet, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 26 - POCL flux through outlet, normalized by surface area, mol/(m**2*s)')
  call LB_HEADER(n_unit(i,ix,iy),'Col. 27 - POCD flux through outlet, normalized by surface area, mol/(m**2*s)')

  work_aver_out(i,ix,iy)%nvals = 22
  work_aver_out(i,ix,iy)%format_char = workchar(1:n)//'22e15.5)'

  do i = 1, nfiles
    allocate ( work_aver_out(i,ix,iy)%vals     (work_aver_out(i,ix,iy)%nvals) )
    allocate ( work_aver_out(i,ix,iy)%vals_sum (work_aver_out(i,ix,iy)%nvals) )
    work_aver_out           (i,ix,iy)%vals_sum (:) = 0.
  enddo

  if (.not. allocated(series_voldef0)) allocate(series_voldef0(nx,ny))
  series_voldef0(ix,iy) = 0.
  count_out(ix,iy) = int(time/(dt_out%par*hour_sec))

endif if_firstcall_ixiy

i = 1
work_aver_out(i,ix,iy)%vals(1)  = h1
work_aver_out(i,ix,iy)%vals(2)  = hx1
work_aver_out(i,ix,iy)%vals(3)  = hx2
work_aver_out(i,ix,iy)%vals(4)  = hy1
work_aver_out(i,ix,iy)%vals(5)  = hy2
work_aver_out(i,ix,iy)%vals(6)  = hx1t
work_aver_out(i,ix,iy)%vals(7)  = hx2t
work_aver_out(i,ix,iy)%vals(8)  = hy1t
work_aver_out(i,ix,iy)%vals(9)  = hy2t
work_aver_out(i,ix,iy)%vals(10) = l1
work_aver_out(i,ix,iy)%vals(11) = hs1
work_aver_out(i,ix,iy)%vals(12) = ls1
work_aver_out(i,ix,iy)%vals(13) = vol
work_aver_out(i,ix,iy)%vals(14) = voldef - series_voldef0(ix,iy)


allocate(work(1:2))
work = VARMEAN(Tw1,bathymwater,1_iintegers); T_mean = work(2)
deallocate(work)
i = 2
work_aver_out(i,ix,iy)%vals(1)  = tsw - Kelvin0
work_aver_out(i,ix,iy)%vals(2)  = Tskin(1)
work_aver_out(i,ix,iy)%vals(3)  = Tw1(1)
work_aver_out(i,ix,iy)%vals(4)  = T_mean
work_aver_out(i,ix,iy)%vals(5)  = maxval(Tw1(1:M+1))
work_aver_out(i,ix,iy)%vals(6)  = T_0dim
work_aver_out(i,ix,iy)%vals(7)  = Ti1(1)
work_aver_out(i,ix,iy)%vals(8)  = Tsn(itop)
work_aver_out(i,ix,iy)%vals(9)  = hw
work_aver_out(i,ix,iy)%vals(10) = xlew
work_aver_out(i,ix,iy)%vals(11) = eflux0
work_aver_out(i,ix,iy)%vals(12) = botflux
work_aver_out(i,ix,iy)%vals(13) = sqrt(tau/row0)
work_aver_out(i,ix,iy)%vals(14) = velfrict_bot
work_aver_out(i,ix,iy)%vals(15) = windwork
work_aver_out(i,ix,iy)%vals(16) = albedo
work_aver_out(i,ix,iy)%vals(17) = shortwave*(1.-sabs)
work_aver_out(i,ix,iy)%vals(18) = signwaveheight
work_aver_out(i,ix,iy)%vals(19) = salice(Mice+1)
work_aver_out(i,ix,iy)%vals(20:21            ) = discharge(1:2)
work_aver_out(i,ix,iy)%vals(22:22+nsoilcols-1) = Tsoil1(1,1:nsoilcols)

i = 3
work_aver_out(i,ix,iy)%vals(1)  = Tw1(1)
work_aver_out(i,ix,iy)%vals(2)  = row(1)
work_aver_out(i,ix,iy)%vals(3)  = eflux0_kinem
work_aver_out(i,ix,iy)%vals(4)  = turb_density_flux
work_aver_out(i,ix,iy)%vals(5)  = zinv
work_aver_out(i,ix,iy)%vals(6)  = H_mixed_layer
work_aver_out(i,ix,iy)%vals(7)  = H_entrainment
work_aver_out(i,ix,iy)%vals(8)  = w_conv_scale
work_aver_out(i,ix,iy)%vals(9)  = u_star
work_aver_out(i,ix,iy)%vals(10) = min(max(- w_conv_scale/(u_star + small_value),-1.e+3_ireals),1.e+3_ireals)
work_aver_out(i,ix,iy)%vals(11) = min(max(Ri_bulk, -1.e+3_ireals),1.e+3_ireals)
work_aver_out(i,ix,iy)%vals(12) = maxN
work_aver_out(i,ix,iy)%vals(13) = minval(lamw(max(i_maxN-3,1_iintegers):min(i_maxN+3,M)))/cw_m_row0
work_aver_out(i,ix,iy)%vals(14) = ThermThick
work_aver_out(i,ix,iy)%vals(15) = ReTherm
work_aver_out(i,ix,iy)%vals(16) = RiTherm

! Methane flux at the bottom of mixed layer
if (h1 > 0.) then
  methfluxML = lammeth(i_maxN)*(qwater(i_maxN+1,2) - qwater(i_maxN,2))/(gsp%ddz(i_maxN)*h1)*&
  & bathymwater(i_maxN)%area_half/bathymwater(1)%area_int
  ! Methane flux from shallow sediments (i.e. sediments in the mixed layer)
  methfluxshalsed = 0.
  do i = i_maxN, 1, -1
    methfluxshalsed = methfluxshalsed + bathymwater(i)%area_int * lsm%water(i) * gsp%ddz05(i-1)*h1
  enddo
  methfluxshalsed = methfluxshalsed/bathymwater(1)%area_int
else
  methfluxML = 0.
  methfluxshalsed = 0.
endif
i = 4
work_aver_out(i,ix,iy)%vals(1)  = h_talik
work_aver_out(i,ix,iy)%vals(2)  = qwater(1,1)
work_aver_out(i,ix,iy)%vals(3)  = qwater(M+1,1)
work_aver_out(i,ix,iy)%vals(4)  = qsoil(ns,nsoilcols)
work_aver_out(i,ix,iy)%vals(5)  = oxyg(1,1)
work_aver_out(i,ix,iy)%vals(6)  = oxyg(M+1,1)
work_aver_out(i,ix,iy)%vals(7)  = rprod_total_newC(nsoilcols)
work_aver_out(i,ix,iy)%vals(8)  = rprod_total_oldC(nsoilcols)
work_aver_out(i,ix,iy)%vals(9)  = febul_lake_bot
work_aver_out(i,ix,iy)%vals(10) = febul_lake_surf
work_aver_out(i,ix,iy)%vals(11) = fplant
work_aver_out(i,ix,iy)%vals(12) = fdiff_lake_bot
work_aver_out(i,ix,iy)%vals(13) = fdiff_lake_surf
work_aver_out(i,ix,iy)%vals(14) = febul_lake_bot*mfs
work_aver_out(i,ix,iy)%vals(15) = febul_lake_surf*mfs
work_aver_out(i,ix,iy)%vals(16) = fplant*mfs
work_aver_out(i,ix,iy)%vals(17) = fdiff_lake_bot*mfs
work_aver_out(i,ix,iy)%vals(18) = fdiff_lake_surf*mfs
work_aver_out(i,ix,iy)%vals(19) = methfluxML*mfs
work_aver_out(i,ix,iy)%vals(20) = methfluxshalsed*mfs
work_aver_out(i,ix,iy)%vals(21) = fbbleflx_ch4_sum(i_maxN)/area_int(1)*mfs
work_aver_out(i,ix,iy)%vals(22) = qwateroxidtot*mfs
work_aver_out(i,ix,iy)%vals(23) = qwateroxidML*mfs
work_aver_out(i,ix,iy)%vals(24) = fco2
work_aver_out(i,ix,iy)%vals(25) = fbbleflx_co2_sum(0)/area_int(1)
work_aver_out(i,ix,iy)%vals(26) = fo2
work_aver_out(i,ix,iy)%vals(27) = fbbleflx_o2_sum(0) /area_int(1)
work_aver_out(i,ix,iy)%vals(28) = finflow
work_aver_out(i,ix,iy)%vals(29) = foutflow
work_aver_out(i,ix,iy)%vals(30:30+nsoilcols-1) = fbbleflx_ch4(0,1:nsoilcols)*febul0(1:nsoilcols)*mfs

i = 5
work_aver_out(i,ix,iy)%vals(1:5) = TKE_budget_terms(1:5)

i = 6
work_aver_out(i,ix,iy)%vals(1)  = gas%DIC(1  ,1)
work_aver_out(i,ix,iy)%vals(2)  = gas%DIC(M+1,1)
work_aver_out(i,ix,iy)%vals(3)  = pools_lake%pools(npool_DIC ) !DIC pool in a lake, normalized by surface area, mol/(m**2)
work_aver_out(i,ix,iy)%vals(4)  = pools_lake%pools(npool_DOC ) !DOC pool in a lake, normalized by surface area, mol/(m**2)
work_aver_out(i,ix,iy)%vals(5)  = pools_lake%pools(npool_POCL) !POCL pool in a lake, normalized by surface area, mol/(m**2)
work_aver_out(i,ix,iy)%vals(6)  = pools_lake%pools(npool_POCD) !POCD pool in a lake, normalized by surface area, mol/(m**2)
work_aver_out(i,ix,iy)%vals(7)  = srcs_lake%sources(nsrc_SOD) !DIC production due to aerobic respiration in sediments (SOD), mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(8)  = srcs_lake%sources(nsrc_BOD) !DIC production due to aerobic POCD decay in the water column (BOD), mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(9)  = srcs_lake%sources(nsrc_resp) !DIC production due to respiration by POCL in the water column, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(10) = - srcs_lake%sources(nsrc_prodox) !DIC sink due to photosynthesis in the water column, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(11) = - fPOCD_sedim !POCD sedimentation to a lake bottom, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(12) = fco2 !CO2 turbulent flux at the lake surface, upwards, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(13) = fco2ebul_surf !CO2 ebullition flux at the surface, upwards, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(14) = fco2ebul_bot  !CO2 ebullition flux averaged over the lake bottom, upwards, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(15) = 0. !CO2 plant-mediated flux at the lake bottom, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(16) = inout_fluxes%fluxes_in (nDIC )!DIC flux by inlets, normalized by surface area, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(17) = inout_fluxes%fluxes_in (nDOC )!DOC flux by inlets, normalized by surface area, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(18) = inout_fluxes%fluxes_in (nPOCL)!POCL flux by inlets, normalized by surface area, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(19) = -inout_fluxes%fluxes_out(nDIC )!DIC flux through outlet, normalized by surface area, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(20) = -inout_fluxes%fluxes_out(nDOC )!DOC flux through outlet, normalized by surface area, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(21) = -inout_fluxes%fluxes_out(nPOCL)!POCL flux through outlet, normalized by surface area, mol/(m**2*s)
work_aver_out(i,ix,iy)%vals(22) = -inout_fluxes%fluxes_out(nPOCD)!POCD flux through outlet, normalized by surface area, mol/(m**2*s)

if (time_series_aver%par == 1) then
  tsteps(ix,iy) = tsteps(ix,iy) + 1.d0
else
  do i = 1, nfiles
    work_aver_out(i,ix,iy)%vals_sum(:) = 0.
  enddo
endif
do i = 1, nfiles
  work_aver_out  (i,ix,iy)%vals_sum(:) = &
  & work_aver_out(i,ix,iy)%vals_sum(:) + work_aver_out(i,ix,iy)%vals(:)
enddo

! Writing to files
if (int(time/(dt_out%par*hour_sec)) > count_out(ix,iy) .or. &
  & abs(time - (count_out(ix,iy) + 1)*dt_out%par*hour_sec) < small_number) then

  if (time_series_aver%par == 1) then
    do i = 1, nfiles
      work_aver_out(i,ix,iy)%vals_sum(:) = work_aver_out(i,ix,iy)%vals_sum(:)/tsteps(ix,iy) 
    enddo
    tsteps(ix,iy) = 0.d0
  endif

  do i = 1, nfiles
    if (LB_ASCII_OUT()) &
    & write (n_unit(i,ix,iy),work_aver_out(i,ix,iy)%format_char) &
    & year,month,day,hour,time/hour_sec,work_aver_out(i,ix,iy)%vals_sum(:)
    call LB_ROW(n_unit(i,ix,iy),year,month,day,hour,time/hour_sec,work_aver_out(i,ix,iy)%vals_sum(:))
    work_aver_out(i,ix,iy)%vals_sum(:) = 0.
  enddo

  count_out(ix,iy) = count_out(ix,iy) + 1 
endif 
      
if (firstcall(ix,iy)) firstcall(ix,iy)=.false.
END SUBROUTINE SERIES_OUT


SUBROUTINE PROFILE_OUTPUT &
& (ix, iy, nx, ny, &
&  year, month, day, hour, &
&  time, dt_out, &
&  Profile, z_prof, nprof, &
&  z_out, nout, &
&  outpath1, filename, &
&  unitindex, ndec, last, Profile2, Profile3)

! The subroutine WATER_TEMP_OUT writes the water temperature profiles
! with the timestep dt_out

use INOUT_PARAMETERS, only : &
& lake_series_out_unit_min, &
& lake_series_out_unit_max

use TIMEVAR, only : &
hour_sec, day_sec, hour_min

use LB_NCOUT, only : LB_ASCII_OUT, LB_HEADER, LB_ASCII_HEADER, LB_PROFILE_ROW

implicit none

! Input variables
! Reals
real(kind=ireals), intent(in) :: Profile(1:nprof) ! The profile
real(kind=ireals), intent(in) :: z_prof(1:nprof) ! The levels of the profile elements
real(kind=ireals), intent(in) :: hour ! The hour
real(kind=ireals), intent(in) :: z_out(1:nout) ! The output grid
real(kind=ireals), intent(in) :: time, dt_out ! The time from beginning of integration
                                    ! The timestep of output
! Integers
integer(kind=iintegers), intent(in) :: ix, iy ! The current coordinates of lake point
integer(kind=iintegers), intent(in) :: nx, ny ! The maximal coordinates of lake points
integer(kind=iintegers), intent(in) :: year, month, day ! Year, month and day
integer(kind=iintegers), intent(in) :: nprof ! The number of layers in water, (M+1) - number of levels
integer(kind=iintegers), intent(in) :: nout
integer(kind=iintegers), intent(in) :: unitindex
integer(kind=iintegers), intent(in) :: ndec ! Number of decimal digits

character(len=*), intent(in) :: outpath1
character(len=*), intent(in) :: filename

logical, intent(in) :: last

real(kind=ireals), intent(in), optional :: Profile2(1:nprof), Profile3(1:nprof) ! Optional profiles

! Local variables
! Reals

real(kind=ireals), parameter :: small_number = 1.d-5
real(kind=ireals), allocatable :: Profile_out(:,:)


! Integers
integer(kind=iintegers), parameter :: unitindex_max = 100
integer(kind=iintegers), allocatable, save :: n_unit(:,:,:)
integer(kind=iintegers), allocatable, save :: count_out(:,:)
integer(kind=iintegers) :: i, k !Loop index
integer(kind=iintegers) :: nvars
!integer(kind=iintegers) :: nform(1:2)

! Characters
character(len=6) :: coords_point
character(len=100) :: formats(1:2)

! Logicals
logical, allocatable, save :: firstcall(:,:)
logical :: flag


if (.not.allocated(firstcall)) then
  allocate (firstcall(1:nx, 1:ny) )
  allocate (count_out(1:nx, 1:ny) )
  allocate (n_unit   (1:unitindex_max, 1:nx, 1:ny) )
  firstcall(:,:) = .true.
  n_unit = lake_series_out_unit_min
endif  

nvars = 1
if (present(Profile2)) nvars = 2
if (present(Profile2) .and. present(Profile3)) nvars = 3
            
if (firstcall(ix,iy)) then
  write (coords_point, '(2i3)') ix, iy

  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(unitindex,ix,iy))
  open (n_unit(unitindex,ix,iy), &
  & file=outpath1(1:len_trim(outpath1))//'time_series/'// &
  & filename(1:len_trim(filename))//coords_point//'.dat',  status='unknown')
  call LB_HEADER(n_unit(unitindex,ix,iy),'Col. 1 - year')
  call LB_HEADER(n_unit(unitindex,ix,iy),'Col. 2 - month')
  call LB_HEADER(n_unit(unitindex,ix,iy),'Col. 3 - day')
  call LB_HEADER(n_unit(unitindex,ix,iy),'Col. 4 - hour')
  call LB_HEADER(n_unit(unitindex,ix,iy),&
  & 'Col. 5 - the time from the start of integration, days') !& 'Col. 5 - the time from the start of integration, days'
  call LB_HEADER(n_unit(unitindex,ix,iy),&
  & 'Col. [6...] - Depth1, Var1, Depth2, Var2, ..., DepthN, VarN')
 
  call CHECK_UNIT(lake_series_out_unit_min,lake_series_out_unit_max, &
  & n_unit(unitindex+1,ix,iy))
  open (n_unit(unitindex+1,ix,iy), &
  & file=outpath1(1:len_trim(outpath1))//'time_series/'// &
  & filename(1:len_trim(filename))//coords_point//'f2'//'.dat',  status='unknown')
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),'Col. 1 - year')
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),'Col. 2 - month')
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),'Col. 3 - day')
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),'Col. 4 - hour')
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),&
  & 'Col. 5 - the time from the start of integration, days') !& 'Col. 5 - the time from the start of integration, days'
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),&
  & 'Col. 6 - Depth, m')
  call LB_ASCII_HEADER(n_unit(unitindex+1,ix,iy),&
  & 'Col. 7 - Var, m')
  
  count_out(ix,iy) = int(time/(dt_out*hour_sec))
endif

      
if (int(time/(dt_out*hour_sec))>count_out(ix,iy) .or. &
  & abs(time - (count_out(ix,iy) + 1)*dt_out*hour_sec) < small_number) then
  if (nout > 0) then
    allocate (Profile_out(1:nout,1:nvars))
    call LININTERPOL (z_prof,Profile,nprof,z_out,Profile_out(1,1),nout,flag,.false.)
    if (nvars >= 2) call LININTERPOL (z_prof,Profile2,nprof,z_out,Profile_out(1,2),nout,flag,.false.)
    if (nvars == 3) call LININTERPOL (z_prof,Profile3,nprof,z_out,Profile_out(1,3),nout,flag,.false.)
    k = nout
    call DEFFORMAT
    call LB_PROFILE_ROW(n_unit(unitindex,ix,iy),year,month,day,hour,time/day_sec, &
    &                   z_out(1:nout),Profile_out(1:nout,1:nvars))
    if (LB_ASCII_OUT()) &
    & write (unit=n_unit(unitindex,ix,iy),fmt=formats(1)) year,month,day,hour, &
    &                                   time/day_sec,       & !max((hour - floor(hour+small_number))*hour_min,0.d0), & 
    &                                   (z_out(i),Profile_out(i,1:nvars), i = 1, nout)
    if (LB_ASCII_OUT()) then
    do i = 1, nout
      write (unit=n_unit(unitindex+1,ix,iy),fmt=formats(2)) year,month,day,hour, &
      &                                   time/day_sec,       & !max((hour - floor(hour+small_number))*hour_min,0.d0), & 
      &                                   -z_out(i),Profile_out(i,1:nvars)
    enddo
    endif
    deallocate (Profile_out)
  else
    allocate (Profile_out(1:nprof,1:nvars))
    Profile_out(1:nprof,1) = Profile(1:nprof)
    if (nvars >= 2) Profile_out(1:nprof,2) = Profile2(1:nprof)
    if (nvars == 3) Profile_out(1:nprof,3) = Profile3(1:nprof)
    k = nprof
    call DEFFORMAT
    call LB_PROFILE_ROW(n_unit(unitindex,ix,iy),year,month,day,hour,time/day_sec, &
    &                   z_prof(1:nprof),Profile_out(1:nprof,1:nvars))
    if (LB_ASCII_OUT()) &
    & write (unit=n_unit(unitindex,ix,iy),fmt=formats(1)) year,month,day,hour, &
    &                                   time/day_sec,       &  ! max((hour - floor(hour+small_number))*hour_min,0.d0), & 
    &                                   (z_prof(i),Profile_out(i,1:nvars), i = 1, nprof)
    if (LB_ASCII_OUT()) then
    do i = 1, nprof
      write (unit=n_unit(unitindex+1,ix,iy),fmt=formats(2)) year,month,day,hour, &
      &                                   time/day_sec,       & ! max((hour - floor(hour+small_number))*hour_min,0.d0), & 
      &                                   -z_prof(i),Profile_out(i,1:nvars)
    enddo
    endif
  endif
  if (last) count_out(ix,iy) = count_out(ix,iy) + 1 
endif 
      
if (firstcall(ix,iy) .and. last) firstcall(ix,iy) = .false.

contains
SUBROUTINE DEFFORMAT

implicit none

type charl
  character(len=20) :: ch
  integer(kind=iintegers) :: len
end type charl

type(charl) :: nc1, nc2, nc3, nv, ymdh

ymdh%ch = '3i7,f8.2,' !'3i7,f8.2,' ! year,month,day,hour
ymdh%len = len_trim(ymdh%ch)
write(nc1%ch,'(i3)') k     ; nc1%len = len_trim(nc1%ch)
write(nv%ch, '(i3)') nvars ; nv%len  = len_trim(nv%ch)
if (ndec >= 0) then
  write(nc2%ch,'(i3)') 6 + ndec ; nc2%len = len_trim(nc2%ch)
  write(nc3%ch,'(i3)') ndec     ; nc3%len = len_trim(nc3%ch)
  formats(1) = '('//ymdh%ch(1:ymdh%len)//'f13.4,'//nc1%ch(1:nc1%len)// &
  & '(f10.2, '//nv%ch(1:nv%len)//'f'//nc2%ch(1:nc2%len)//'.'//nc3%ch(1:nc3%len)//'))'
  formats(2) = '('//ymdh%ch(1:ymdh%len)//'f13.4,f10.2,'//nv%ch(1:nv%len)//'f'//nc2%ch(1:nc2%len)//'.'// &
  & nc3%ch(1:nc3%len)//')'
else
  formats(1) = '('//ymdh%ch(1:ymdh%len)//'f13.4,'//nc1%ch(1:nc1%len)//'(f10.2, '//nv%ch(1:nv%len)//'E23.10E3))'
  formats(2) = '('//ymdh%ch(1:ymdh%len)//'f13.4,f10.2,'//nv%ch(1:nv%len)//'E23.10E3)'
endif

END SUBROUTINE DEFFORMAT

!100 format (3i7,f8.2,f13.4,<k>(f10.2, f<6+ndec>.<ndec>))
!101 format (3i7,f8.2,f13.4,f10.2,f<6+ndec>.<ndec>)
!102 format (3i7,f8.2,f13.4,<k>(f10.2, E15.7))
!103 format (3i7,f8.2,f13.4,f10.2,E15.7)
END SUBROUTINE PROFILE_OUTPUT


SUBROUTINE ACCUM_VAR &
& (dt, l1, febul, febul0, fdiff, fdiff_lake_surf, qwateroxidtot, & 
& ice_meth_oxid_total, &
& rprod_total_newC, rprod_total_oldC, &
& febultot, febulbottot, fdifftot, fdiff_lake_surftot, &
& rprod_total_newC_integr, rprod_total_oldC_integr, &
& methoxidwat, ice_meth_oxid, add_to_winter)

use METH_OXYG_CONSTANTS, only : &
& molmass_ch4, mfs, mf

implicit none

! Input/output variables

real(kind=ireals), intent(in) :: dt
real(kind=ireals), intent(in) :: l1
real(kind=ireals), intent(in) :: febul, febul0
real(kind=ireals), intent(in) :: fdiff
real(kind=ireals), intent(in) :: fdiff_lake_surf
!real(kind=ireals), intent(in) :: febul(1:2) ! two-meth
!real(kind=ireals), intent(in) :: fdiff(1:2) ! two-meth
!real(kind=ireals), intent(in) :: fdiff_lake_surf(1:2) ! two-meth
real(kind=ireals), intent(in) :: rprod_total_newC
real(kind=ireals), intent(in) :: rprod_total_oldC
real(kind=ireals), intent(in) :: qwateroxidtot
real(kind=ireals), intent(in) :: ice_meth_oxid_total

logical, intent(inout) :: add_to_winter
real(kind=ireals), intent(inout) :: febultot(1:2), febulbottot(1:2)
real(kind=ireals), intent(inout) :: fdifftot(1:2)
real(kind=ireals), intent(inout) :: fdiff_lake_surftot(1:2)
real(kind=ireals), intent(inout) :: rprod_total_newC_integr(1:2)
real(kind=ireals), intent(inout) :: rprod_total_oldC_integr(1:2)
real(kind=ireals), intent(inout) :: methoxidwat(1:2)
real(kind=ireals), intent(inout) :: ice_meth_oxid

!real(kind=ireals), intent(inout) :: febultot(1:2,1:2) ! two-meth
!real(kind=ireals), intent(inout) :: fdifftot(1:2,1:2) ! two-meth
!real(kind=ireals), intent(inout) :: fdiff_lake_surftot(1:2,1:2) ! two-meth
!real(kind=ireals), intent(inout) :: rprod_total_newC_integr(1:2,1:2) ! two-meth
!real(kind=ireals), intent(inout) :: rprod_total_oldC_integr(1:2,1:2) ! two-meth

! Local variables

real(kind=ireals), parameter :: day_sec = 24.*60.*60.

integer(kind=iintegers) :: i ! loop index

if (l1 == 0. .and. (.not. add_to_winter)) then
  i = 1
else
  i = 2
  if (add_to_winter) add_to_winter = .false.
endif

febultot(i) = febultot(i) + febul*mfs*dt/day_sec
febulbottot(i) = febulbottot(i) + febul0*mfs*dt/day_sec
fdifftot(i) = fdifftot(i) - fdiff*mfs*dt/day_sec
fdiff_lake_surftot(i) = fdiff_lake_surftot(i) - fdiff_lake_surf*mfs*dt/day_sec
methoxidwat(i) = methoxidwat(i) - qwateroxidtot*mfs*dt/day_sec
rprod_total_newC_integr(i) = rprod_total_newC_integr(i) + rprod_total_newC*mfs*dt/day_sec 
rprod_total_oldC_integr(i) = rprod_total_oldC_integr(i) + rprod_total_oldC*mfs*dt/day_sec 

ice_meth_oxid = ice_meth_oxid + ice_meth_oxid_total*mf

!do i = 1, 2 ! two-meth
!  if (l1 == 0.) then ! two-meth
!    febultot(1,i) = febultot(1,i) + febul(i)*mfs*dt/day_sec ! two-meth
!    fdifftot(1,i) = fdifftot(1,i) - fdiff(i)*mfs*dt/day_sec ! two-meth
!    fdiff_lake_surftot(1,i) = fdiff_lake_surftot(1,i) - fdiff_lake_surf(i)*mfs*dt/day_sec ! two-meth
!    rprod_total_newC_integr(1,i) = rprod_total_newC_integr(1,i) + rprod_total_newC*mfs*dt/day_sec  ! two-meth
!    rprod_total_oldC_integr(1,i) = rprod_total_oldC_integr(1,i) + rprod_total_oldC*mfs*dt/day_sec  ! two-meth
!  else ! two-meth
!    febultot(2,i) = febultot(2,i) + febul(i)*mfs*dt/day_sec ! two-meth
!    fdifftot(2,i) = fdifftot(2,i) - fdiff(i)*mfs*dt/day_sec ! two-meth
!    fdiff_lake_surftot(2,i) = fdiff_lake_surftot(2,i) - fdiff_lake_surf(i)*mfs*dt/day_sec ! two-meth
!    rprod_total_newC_integr(2,i) = rprod_total_newC_integr(2,i) + rprod_total_newC*mfs*dt/day_sec  ! two-meth
!    rprod_total_oldC_integr(2,i) = rprod_total_oldC_integr(2,i) + rprod_total_oldC*mfs*dt/day_sec  ! two-meth
!  endif ! two-meth
!enddo ! two-meth

END SUBROUTINE ACCUM_VAR    


!> Subroutine TEMPLOC calculates the temperature and other scalars in a specified location
!! of a lake given the vertical profile of horizontally-averaged field 
!! and deviations of heights of constant-density layers
SUBROUTINE TEMPLOC(M,nvars,nvar,itherm,h,dt_out,time,hx1ml,hy1ml,Tw,gsp,gs,bathymwater,rtemp, &
&                  basename,firstcall_)

!Currently implemented for single-lake output only.

use NUMERIC_PARAMS, only : pi
use ARRAYS_GRID, only : gridspacing_type, gridsize_type
use ARRAYS_BATHYM, only : bathym, aki, bki
use INOUT_PARAMETERS, only : lake_misc_unit_min, lake_misc_unit_max
use TIMEVAR, only : hour_sec
use DRIVING_PARAMS, only : grarr1

implicit none

!Input variables
integer(kind=iintegers), intent(in) :: M !> Number of grid layers (number of levels - (M+1) )
integer(kind=iintegers), intent(in) :: nvars !> Number of scalar fields
integer(kind=iintegers), intent(in) :: nvar !> Number of scalar field
integer(kind=iintegers), intent(in) :: itherm(1:M+2) !> Grid levels between constant-density layers

real(kind=ireals), intent(in) :: h !> Lake depth, m
real(kind=ireals), intent(in) :: dt_out !> Output interval, s
real(kind=ireals), intent(in) :: time !> Time, s
real(kind=ireals), intent(in) :: hx1ml(1:M+1), hy1ml(1:M+1) !> Deviations of thickness of 
                                                            !! constant-density layers
real(kind=ireals), intent(in) :: Tw(1:M+1) !> Profile of horizontally-averaged scalar

type(gridspacing_type), intent(in) :: gsp !> Grid spacing group
type(gridsize_type), intent(in) :: gs !> Grid size group
type(bathym), intent(in) :: bathymwater(1:M+1) !> Bathymetry group
type(grarr1), intent(in) :: rtemp !> Coordinates of a location in a lake

character(len=*), intent(in) :: basename
logical, intent(in) :: firstcall_ != .true.

!Output variables
!real(kind=ireals), intent(out) :: Temp !> Temperature in the point set by coordinates rtemp

!Local variables
real(kind=ireals), allocatable :: Temp(:) ! Scalar in the point set by coordinates rtemp
real(kind=ireals), allocatable :: dep(:,:), Hlars(:)
real(kind=ireals) :: amplx, amply, amplx0, amply0, sumh, sumh1, dsumh, &
& larthick, dlarthick, alpha, Lx, Ly, h_, x, y, z
real(kind=ireals), save :: tcount = 0.

integer(kind=iintegers) :: i, j, k !Loop indices
integer(kind=iintegers) :: nlevs
integer(kind=iintegers), allocatable, save :: out_unit(:)


character :: coords_point*6

if (.not. allocated(out_unit)) then
  allocate(out_unit(1:nvars)) 
  out_unit = lake_misc_unit_min
endif
if (firstcall_) then
  call CHECK_UNIT(lake_misc_unit_min,lake_misc_unit_max,out_unit(nvar))
  write (coords_point,'(2i3)') gs%ix, gs%iy
  open(out_unit(nvar),file=outpath(1:len_trim(outpath))//'time_series/'// &
  & basename(1:len_trim(basename))//coords_point//'.dat', status='unknown')
endif

if (int(time/(dt_out*hour_sec)) > tcount) then

  allocate (Temp(1:rtemp%numarr))
  Temp(:) = missing_value

  ! Counting number of constant-density layers
  nlevs = 0
  do i = 2, M+2
    if (itherm(i) < 0) then
      exit
    else
      nlevs = nlevs + 1
    endif
  enddo
  allocate (dep(1:nlevs,1:2))

  allocate (Hlars(1:nlevs))
  forall (i=1:nlevs) Hlars(i) = h*sum(gsp%ddz05(itherm(i):itherm(i+1)-1))


  ! Cycle by locations of output
  do k = 1, rtemp%numarr

    do i = 1, nlevs
      amplx0 = 0.; amply0 = 0.
      if (i < nlevs) then
        do j = nlevs, i+1, -1
          amplx0 = amplx0 - aki(j,i)*0.5*pi*hx1ml(j)
          amply0 = amply0 - bki(j,i)*0.5*pi*hy1ml(j)
        enddo
      endif
      amplx = amplx0 - 0.5*hx1ml(i)*pi
      amply = amply0 - 0.5*hy1ml(i)*pi
      if (i == nlevs) then
        x = 0.
      else
        x = sum(Hlars(i+1:nlevs))
      endif
      if (i == 1) then
        Lx = bathymwater(1)%Lx
        Ly = bathymwater(1)%Ly       
      else
        Lx = bathymwater(itherm(i))%Lx_half
        Ly = bathymwater(itherm(i))%Ly_half
      endif
      dep(i,2) = x + amplx0*sin(pi*rtemp%arr1(1,k)/Lx) + amply0*sin(pi*rtemp%arr1(2,k)/Ly)
      dep(i,2) = h - dep(i,2)
      dep(i,1) = x + Hlars(i) + &
      &              amplx *sin(pi*rtemp%arr1(1,k)/Lx) + amply *sin(pi*rtemp%arr1(2,k)/Ly)
      dep(i,1) = h - dep(i,1)
      !if (i == 1) print*, 'i=1:', Hlars(i), 0.5*hy1ml(i)*pi, dep(i,1:2)
    enddo
    !if (k == 7 .or. k == 3) then
    !print*, 'hx1ml=', hx1ml(1:10)
    !do i = 1, nlevs
    !  print*, 'k=', k, 'dep1,dep2', i, dep(i,1:2)
    !enddo
    !endif
    !!read*

    ! Computing temperature at a given depth as linear interpolation between model levels
    seekl:do i = 1, nlevs
      if ( i /= nlevs) then 
        if ( rtemp%arr1(3,k) < dep(i,2) .and. rtemp%arr1(3,k) > dep(i+1,1) ) then
          z = abs(rtemp%arr1(3,k) - dep(i,2)) + abs(rtemp%arr1(3,k) - dep(i+1,1))
          x = abs(rtemp%arr1(3,k) - dep(i,2))/z
          y = abs(rtemp%arr1(3,k) - dep(i+1,1))/z
          Temp(k) = y*Tw(itherm(i+1)) + x*Tw(itherm(i+1)+1)
          !if (k == 7 .and. Temp(k) > 16.) print*, 'Temploc1: ', dep(i,2), dep(i+1,1)
          exit seekl
        endif
        if ( rtemp%arr1(3,k) > dep(i,2) .and. rtemp%arr1(3,k) < dep(i+1,1) ) then
          z = abs(rtemp%arr1(3,k) - dep(i,2)) + abs(rtemp%arr1(3,k) - dep(i+1,1))
          x = abs(rtemp%arr1(3,k) - dep(i,2))/z
          y = abs(rtemp%arr1(3,k) - dep(i+1,1))/z
          Temp(k) = y*Tw(itherm(i+1)) + x*Tw(itherm(i+1)+1)
          !if (k == 7 .and. Temp(k) > 16.) print*, 'Temploc2: ', dep(i,2), dep(i+1,1)
          exit seekl
        endif
      endif
      if (rtemp%arr1(3,k) > dep(i,1) .and. rtemp%arr1(3,k) <= dep(i,2)) then
        y = h*(dep(i,2) - dep(i,1))/Hlars(i)
        ! Near top of the layer
        if (i /= 1) then
          if ( rtemp%arr1(3,k) < dep(i,1) + y*gsp%ddz(itherm(i))*0.5 ) then
            if (dep(i-1,2)-dep(i-1,1) > 0.) then
              x = 0.5*gsp%ddz(itherm(i))*h*(dep(i-1,2)-dep(i-1,1))/Hlars(i-1)
              Temp(k) = Tw(itherm(i)) + (Tw(itherm(i)+1) - Tw(itherm(i))) * &
              & (rtemp%arr1(3,k) - dep(i,1) + x)/(x + y*gsp%ddz(itherm(i))*0.5)
            else
              Temp(k) = 0.5*(Tw(itherm(i)) + Tw(itherm(i)+1))
            endif
            !if (k == 7 .and. Temp(k) > 16.) print*, 'Temploc3: ', dep(i,2), dep(i,1)
            exit seekl
          endif
        endif
        ! Near bottom of the layer
        if (i /= nlevs) then
          if ( rtemp%arr1(3,k) > dep(i,2) - y*gsp%ddz(itherm(i+1))*0.5 ) then
            if (dep(i+1,2)-dep(i+1,1) > 0.) then
              x = 0.5*gsp%ddz(itherm(i+1))*h*(dep(i+1,2)-dep(i+1,1))/Hlars(i+1)
              Temp(k) = Tw(itherm(i+1)) + (Tw(itherm(i+1)+1) - Tw(itherm(i+1))) * &
              & (rtemp%arr1(3,k) - dep(i,2) + y*gsp%ddz(itherm(i+1))*0.5) / &
              & (x + y*gsp%ddz(itherm(i+1))*0.5)
            else
              Temp(k) = 0.5*(Tw(itherm(i+1)) + Tw(itherm(i+1)+1))
            endif
            !if (k == 7 .and. Temp(k) > 16.) print*, 'Temploc5: ', x, y, dep(i,2), i
            exit seekl
          endif
        endif
        x = dep(i,1) 
        if (i /= 1) x = x + y*gsp%ddz(itherm(i))*0.5
        j = itherm(i)+1
        ! Linear interpolation between numerical levels
        seekl_1 : do
          if (x                <  rtemp%arr1(3,k) .and. &
            & x + y*gsp%ddz(j) >= rtemp%arr1(3,k)) then
            Temp(k) = Tw(j) + (Tw(j+1) - Tw(j)) * &
            & (rtemp%arr1(3,k) - x)/(y*gsp%ddz(j))
            !if (k == 7 .and. Temp(k) > 16.) print*, 'Temploc4: ', x, y
            exit seekl_1
          endif
          x = x + y*gsp%ddz(j)
          j = j + 1
        enddo seekl_1
      endif
    enddo seekl

    !if (k == 7 .and. Temp(k) > 16.) then
    !  print*, 'In Temploc: STOP'
    !  read* !STOP
    !endif

    !sumh = 0.
    !cyc1 : do i = M+1, 1, -1
    !  if1 : if (itherm(i+1) > 0) then
    !    Lx = bathymwater(itherm(2))%Lx!bathymwater(itherm(i+1))%Lx ! Assuming cross-section 
    !                                                               ! to be constant with depth
    !    Ly = bathymwater(itherm(2))%Ly!bathymwater(itherm(i+1))%Ly
    !    !if (abs(rtemp%arr1(1,k)) > 0.5*Lx .or. abs(rtemp%arr1(2,k)) > 0.5*Ly) exit if1
    !    amplx = - 0.5*hx1ml(i)*pi !/(Lx*Ly)
    !    amply = - 0.5*hy1ml(i)*pi !/(Lx*Ly)
    !    larthick = h*sum(gsp%ddz05(itherm(i):itherm(i+1)-1)) ! Equilibrium thickness of constant-density layer
    !    dlarthick = amplx*sin(pi*rtemp%arr1(1,k)/Lx) + amply*sin(pi*rtemp%arr1(2,k)/Ly)
    !    sumh = sumh + larthick + dlarthick
    !    !print*, 'ay', hy1ml
    !  endif if1
    !  if (sumh >= h - rtemp%arr1(3,k)) then
    !    !print*, itherm(i)+1, amplx
    !    h_ = h - rtemp%arr1(3,k)
    !    sumh1 = sumh
    !    cyclrs : do j = itherm(i)+1, itherm(i+1)
    !      dsumh = h*gsp%ddz05(j-1)*(1. + dlarthick/larthick) 
    !      if ( h_ <= sumh1 .and. h_ > sumh1 - 0.5*dsumh ) then
    !         alpha = (h_ - sumh1 + 0.5*dsumh)/dsumh !Assuming regular grid and the same thickness deviation
    !                                                !in adjacent layers
    !         Temp(k) = Tw(max(j-1,1))*alpha + Tw(j)*(1. - alpha)
    !         !print*, '1', j, nvar
    !         exit cyclrs
    !      elseif ( h_ <= sumh1 - 0.5*dsumh .and. h_ > sumh1 - dsumh ) then
    !         alpha = (sumh1 - 0.5*dsumh - h_)/dsumh !Assuming regular grid and the same thickness deviation
    !                                                !in adjacent layers
    !         Temp(k) = Tw(min(j+1,M+1))*alpha + Tw(j)*(1. - alpha) 
    !         !print*, '2', j, nvar
    !         exit cyclrs
    !      endif
    !      sumh1 = sumh1 - dsumh
    !    enddo cyclrs
    !    exit cyc1
    !  endif
    !enddo cyc1

  enddo
  
  write(out_unit(nvar),*) time/hour_sec, Temp(1:rtemp%numarr)
  
  if (nvar == nvars) tcount = tcount + 1.

  !print*, Temp
  !read*

  deallocate(Temp)
  deallocate(dep)
  deallocate(Hlars)
endif

!if (firstcall) firstcall = .false.
END SUBROUTINE TEMPLOC


END MODULE OUT_MOD
