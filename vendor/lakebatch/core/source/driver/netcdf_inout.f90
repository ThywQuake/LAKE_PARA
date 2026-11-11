#ifdef NETCDF_LIB
MODULE NETCDF_LAKE_DRIVER
contains
SUBROUTINE NETCDF_READ_FORCING &
 & (filename1, rewind_netcdf, npoints, &
 & lat, lon, &
 & prec, I_atm, Q_atm_lw, U_a, V_a, &
 & T_a, q_a, P_a)

use netcdf 
use DRIVER_DATATYPES, only : ireals, iintegers

implicit none 

! Input variables

character(len=*), intent(in) :: filename1
integer(kind=iintegers), intent(in) :: npoints

! Input/output variables

logical, intent(inout) :: rewind_netcdf

! Output variables

real(kind=ireals), intent(out) :: prec(:,:)
real(kind=ireals), intent(out) :: I_atm(:,:)
real(kind=ireals), intent(out) :: Q_atm_lw(:,:)
real(kind=ireals), intent(out) :: U_a(:,:)
real(kind=ireals), intent(out) :: V_a(:,:)
real(kind=ireals), intent(out) :: T_a(:,:)
real(kind=ireals), intent(out) :: q_a(:,:)
real(kind=ireals), intent(out) :: P_a(:,:)

real(kind=ireals), intent(out) :: lat(:)
real(kind=ireals), intent(out) :: lon(:)

! Local variables

real(kind=ireals), allocatable, save :: prec_in(:,:)
real(kind=ireals), allocatable, save :: I_atm_in(:,:)
real(kind=ireals), allocatable, save :: Q_atm_lw_in(:,:)
real(kind=ireals), allocatable, save :: U_a_in(:,:)
real(kind=ireals), allocatable, save :: V_a_in(:,:)
real(kind=ireals), allocatable, save :: T_a_in(:,:)
real(kind=ireals), allocatable, save :: q_a_in(:,:)
real(kind=ireals), allocatable, save :: P_a_in(:,:)

real(kind=ireals), allocatable, save :: latr(:)
real(kind=ireals), allocatable, save :: lonr(:)

real(4), allocatable :: values_2d(:,:)
real(4), allocatable :: values_1d(:) 

integer(kind=iintegers) :: status_nc
integer(kind=iintegers) :: ncid
integer(kind=iintegers) :: varid
integer(kind=iintegers) :: dimid 
integer(kind=iintegers), save :: tstep
integer(kind=iintegers), save :: xdim
integer(kind=iintegers) :: start(2)
integer(kind=iintegers) :: countt(2)
integer(kind=iintegers) :: i
integer(kind=iintegers) :: j
integer(kind=iintegers), save :: nforc

character(len=20) :: filename

logical, save :: firstcall
data firstcall /.true./

if (firstcall) then
  filename = adjustl(filename1)

! Opening the netCDF file
  status_nc=nf90_open(filename,nf90_nowrite,ncid)
  call ncerror(status_nc,'opening file')

! Inquiring time variable ID
  status_nc=nf90_inq_dimid(ncid,'tstep',dimid)
  call ncerror(status_nc,'getting tstep id')

! Inquiring time variable dimension
  status_nc=nf90_inquire_dimension(ncid=ncid,dimid=dimid,len=tstep)
  call ncerror(status_nc,'getting tstep dim')

!  status_nc=nf90_inq_dimid(ncid,'x',dimid)
!  call ncerror(status_nc,'getting x id')

! Inquiring the spatial coordinate ID
  status_nc=nf90_inq_dimid(ncid,'land',dimid)
  call ncerror(status_nc,'getting land id')

! Inquiring the spatial coordinate dimension
  status_nc=nf90_inquire_dimension(ncid=ncid,dimid=dimid,len=xdim)
  call ncerror(status_nc,'getting xdim dim')

  if (xdim /= npoints) then
    print*, 'The number of points in setup file does not match &
    & the number of points in netcdf file: STOP'
    STOP
  endif

! Allocating relevant arrays
  allocate(values_2d(xdim,tstep))
  allocate(values_1d(1:xdim))

  allocate                                                                                &
   &( prec_in(xdim,tstep)           , I_atm_in(xdim,tstep)    , Q_atm_lw_in(xdim,tstep),  &
   &  U_a_in(xdim,tstep)            , T_a_in(xdim,tstep)      , q_a_in(xdim,tstep)     ,  &
   &  P_a_in(xdim,tstep)            , V_a_in(xdim,tstep) )         

  allocate (latr(1:xdim))
  allocate (lonr(1:xdim))

  call READVAR_1D('lat')
  latr(1:xdim) = values_1d(1:xdim)

  call READVAR_1D('lon')
  lonr(1:xdim) = values_1d(1:xdim)

  call READVAR_2D('Wind_E')
  U_a_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Wind_N')
  V_a_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Tair')
  T_a_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Qair')
  q_a_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('PSurf')
  P_a_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('SWdown')
  I_atm_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('LWdown')
  Q_atm_lw_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Rainf')
  prec_in(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Snowf')
  prec_in(1:xdim,1:tstep) = prec_in(1:xdim,1:tstep) + values_2d(1:xdim,1:tstep)

  status_nc=nf90_close(ncid)
  call ncerror(status_nc,'closing file')

  deallocate(values_2d)
  deallocate(values_1d)

  nforc = 0

endif

if (rewind_netcdf) then
  nforc = 1
  rewind_netcdf = .false.
else
  nforc = nforc + 1
endif

if (nforc+1 > tstep) then
  print*, 'The abnormal termination in driver: the netcdf forcing finished'
  STOP
endif

prec    (:,1:2) = prec_in    (:,nforc:nforc+1)
I_atm   (:,1:2) = I_atm_in   (:,nforc:nforc+1)
Q_atm_lw(:,1:2) = Q_atm_lw_in(:,nforc:nforc+1)
U_a     (:,1:2) = U_a_in     (:,nforc:nforc+1)
V_a     (:,1:2) = V_a_in     (:,nforc:nforc+1)
T_a     (:,1:2) = T_a_in     (:,nforc:nforc+1)
q_a     (:,1:2) = q_a_in     (:,nforc:nforc+1)
P_a     (:,1:2) = P_a_in     (:,nforc:nforc+1)

lat(1:xdim) = latr(1:xdim)
lon(1:xdim) = lonr(1:xdim)

!print*, 'Finished NETCDF_READ successfully!'

if (firstcall) firstcall = .false.
CONTAINS
 
SUBROUTINE NCERROR(STATUS,STRING)

IMPLICIT NONE

INTEGER,INTENT(IN) :: STATUS
CHARACTER(LEN=*),INTENT(IN),OPTIONAL :: STRING

IF ( STATUS /= 0 ) THEN
  PRINT*, 'Error on NETCDF_READ'
  PRINT *, TRIM(NF90_STRERROR(STATUS))
  IF( PRESENT(STRING) ) PRINT*,TRIM(STRING)
  STOP
ENDIF

END SUBROUTINE NCERROR


SUBROUTINE READVAR_1D(namevar)

implicit none

character(len=*), intent(in) :: namevar

integer(kind=iintegers) :: start_1d(1)
integer(kind=iintegers) :: count_1d(1)

call NCERROR(NF90_INQ_VARID(ncid,namevar,varid))
print*, NAMEVAR//' varid is ', varid 

start_1d = (/1/)
count_1d =(/xdim/)
call NCERROR (NF90_GET_VAR(ncid,varid,values_1d,start_1d,count_1d), &
& 'reading var '//NAMEVAR)

END SUBROUTINE READVAR_1D


SUBROUTINE READVAR_2D(namevar)

implicit none

character(len=*), intent(in):: namevar

call NCERROR(NF90_INQ_VARID(ncid,namevar,varid))
print*, namevar//' varid is ',varid 

start=(/1,1/)
countt=(/xdim,tstep/)
call NCERROR(NF90_GET_VAR(ncid,varid,values_2d,start,countt), &
& 'reading var '//NAMEVAR)

END SUBROUTINE READVAR_2D


END SUBROUTINE NETCDF_READ_FORCING


SUBROUTINE NETCDF_WRITE_SURF &
 & (Sens_heat_flux,Lat_heat_flux,Long_net_rad,Short_net_rad,Surf_temp, &
 &  Time,npoints,Nfile,Nproc,filename1,filename_netcdf_file,close_log)

use netcdf
use DRIVER_DATATYPES, only : ireals, iintegers

implicit none

! Input variables

! Integers
integer(kind=iintegers), intent(in) :: Nfile
integer(kind=iintegers), intent(in) :: npoints
integer(kind=iintegers), intent(in) :: Nproc

! Reals
real(kind=ireals), intent(in) :: Sens_heat_flux(1:npoints)
real(kind=ireals), intent(in) :: Lat_heat_flux(1:npoints)
real(kind=ireals), intent(in) :: Long_net_rad(1:npoints)
real(kind=ireals), intent(in) :: Short_net_rad(1:npoints)
real(kind=ireals), intent(in) :: Surf_temp(1:npoints)

real(kind=ireals), intent(in) :: Time 

! Characters
character(len=*), intent(in) :: filename1

! Logicals
logical, intent(in) :: close_log

! Output variables
! Characters
character(len=60), intent(out) :: filename_netcdf_file ! Omitting extension '.nc'

! Local variables
! Integers
integer(kind=iintegers), parameter :: Nfiles = 2
integer(kind=iintegers), save :: ncid(1:Nfiles)
integer(kind=iintegers), save :: dimids(1:2,1:Nfiles)
integer(kind=iintegers), save :: varids(1:6,1:Nfiles)
integer(kind=iintegers), save :: startt(1:2,1:Nfiles)
integer(kind=iintegers), save :: start(1:1,1:Nfiles)

! Characters
character(len=60) :: filename
character(len=4) :: nfile_suffix

! Logicals
logical, save :: firstcall(1:Nfiles)
data firstcall(1:2) /2*.true./

if (firstcall(Nfile)) then
  write(nfile_suffix, '(2i2)') Nproc, Nfile
  filename = 'results/'//adjustl(filename1)
  filename_netcdf_file = filename(1:len_trim(filename))//'/netcdf/surf'//nfile_suffix
  filename = filename(1:len_trim(filename))//'/netcdf/surf'//nfile_suffix//'.nc'
  print*, 'filename = ', filename
!  STOP 
  call NCERROR(NF90_CREATE(filename,nf90_clobber,ncid(Nfile)))
  call NCERROR(NF90_DEF_DIM(ncid(Nfile), 'x'    , npoints       , dimids(1,Nfile)))
  call NCERROR(NF90_DEF_DIM(ncid(Nfile), 'tstep', nf90_unlimited, dimids(2,Nfile)))

  call NCERROR(NF90_DEF_VAR(ncid(Nfile),'Sens_heat_flux',nf90_double, &
  & dimids(1:2,Nfile),varids(1,Nfile)))
  call NCERROR(NF90_DEF_VAR(ncid(Nfile),'Lat_heat_flux' ,nf90_double, &
  & dimids(1:2,Nfile),varids(2,Nfile)))
  call NCERROR(NF90_DEF_VAR(ncid(Nfile),'Long_net_rad'  ,nf90_double, &
  & dimids(1:2,Nfile),varids(3,Nfile)))
  call NCERROR(NF90_DEF_VAR(ncid(Nfile),'Short_net_rad' ,nf90_double, &
  & dimids(1:2,Nfile),varids(4,Nfile)))
  call NCERROR(NF90_DEF_VAR(ncid(Nfile),'Surf_temp'     ,nf90_double, &
  & dimids(1:2,Nfile),varids(5,Nfile)))
  call NCERROR(NF90_DEF_VAR(ncid(Nfile),'Time'          ,nf90_double, &
  & dimids(2,Nfile),varids(6,Nfile)))

  call NCERROR(NF90_PUT_ATT(ncid(Nfile), varids(1,Nfile), 'Units', 'W/m**2'))
  call NCERROR(NF90_PUT_ATT(ncid(Nfile), varids(2,Nfile), 'Units', 'W/m**2'))
  call NCERROR(NF90_PUT_ATT(ncid(Nfile), varids(3,Nfile), 'Units', 'W/m**2'))
  call NCERROR(NF90_PUT_ATT(ncid(Nfile), varids(4,Nfile), 'Units', 'W/m**2'))
  call NCERROR(NF90_PUT_ATT(ncid(Nfile), varids(5,Nfile), 'Units', 'K'))
  call NCERROR(NF90_PUT_ATT(ncid(Nfile), varids(6,Nfile), 'Units', 'sec'))

  call NCERROR(NF90_ENDDEF(ncid(Nfile)))
  startt(1,Nfile) = 1
  startt(2,Nfile) = 0
  start(1,Nfile) = 0
endif

startt(2,Nfile) = startt(2,Nfile) + 1
start(1,Nfile) = start(1,Nfile) + 1
call NCERROR(NF90_PUT_VAR(ncid=ncid(Nfile),varid=varids(1,Nfile), &
 & values=Sens_heat_flux(1:npoints),start=startt(1:2,Nfile)))
call NCERROR(NF90_PUT_VAR(ncid=ncid(Nfile),varid=varids(2,Nfile), &
 & values=Lat_heat_flux(1:npoints),start=startt(1:2,Nfile)))
call NCERROR(NF90_PUT_VAR(ncid=ncid(Nfile),varid=varids(3,Nfile), &
 & values=Long_net_rad(1:npoints),start=startt(1:2,Nfile)))
call NCERROR(NF90_PUT_VAR(ncid=ncid(Nfile),varid=varids(4,Nfile), &
 & values=Short_net_rad(1:npoints),start=startt(1:2,Nfile)))
call NCERROR(NF90_PUT_VAR(ncid=ncid(Nfile),varid=varids(5,Nfile), &
 & values=Surf_temp(1:npoints),start=startt(1:2,Nfile)))
call NCERROR(NF90_PUT_VAR(ncid=ncid(Nfile),varid=varids(6,Nfile), &
 & values=Time,start=start(1:1,Nfile)))
if (close_log) call NCERROR(NF90_CLOSE(ncid(Nfile)))

if (firstcall(Nfile)) firstcall(Nfile) = .false.
RETURN

CONTAINS
SUBROUTINE NCERROR(STATUS,STRING)

IMPLICIT NONE

INTEGER, INTENT(IN) :: STATUS
CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: STRING

IF ( STATUS /= 0 ) THEN
  PRINT *, TRIM(NF90_STRERROR(STATUS))
  print*, 'Error in NETCDF_WRITE'
  IF( PRESENT(STRING) ) PRINT*, TRIM(STRING)
  STOP
ENDIF

END SUBROUTINE NCERROR

END SUBROUTINE NETCDF_WRITE_SURF
END MODULE NETCDF_LAKE_DRIVER
#endif

