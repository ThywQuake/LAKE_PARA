#ifdef NETCDF_LIB
MODULE NETCDF_POSTPROCESS_LAKE
contains
SUBROUTINE NETCDF_POSTPROCESS_SURF &
& (filename1,moving_average_window,mean_cycle_period)

use netcdf 
use DRIVER_DATATYPES, only : ireals, iintegers

implicit none 

! Input variables

integer(kind=iintegers), intent(in) :: moving_average_window
integer(kind=iintegers), intent(in) :: mean_cycle_period

character(len=*), intent(in) :: filename1

! Local variables

real(kind=ireals), allocatable :: Sens_heat_flux(:,:)
real(kind=ireals), allocatable :: Lat_heat_flux (:,:)
real(kind=ireals), allocatable :: Long_net_rad  (:,:)
real(kind=ireals), allocatable :: Short_net_rad (:,:)
real(kind=ireals), allocatable :: Surf_temp     (:,:)
real(kind=ireals), allocatable :: Time          (:)

real(kind=ireals), allocatable :: Sens_heat_flux_mov_aver(:,:)
real(kind=ireals), allocatable :: Lat_heat_flux_mov_aver (:,:)
real(kind=ireals), allocatable :: Long_net_rad_mov_aver  (:,:)
real(kind=ireals), allocatable :: Short_net_rad_mov_aver (:,:)
real(kind=ireals), allocatable :: Surf_temp_mov_aver     (:,:)

real(kind=ireals), allocatable :: Sens_heat_flux_mean_cycle(:,:)
real(kind=ireals), allocatable :: Lat_heat_flux_mean_cycle (:,:)
real(kind=ireals), allocatable :: Long_net_rad_mean_cycle  (:,:)
real(kind=ireals), allocatable :: Short_net_rad_mean_cycle (:,:)
real(kind=ireals), allocatable :: Surf_temp_mean_cycle     (:,:)

real(4), allocatable :: values_2d(:,:)
real(4), allocatable :: values_1d(:) 


integer(kind=iintegers) :: varids(1:6)
integer(kind=iintegers) :: dimids(1:2)

integer(kind=iintegers) :: ncid
integer(kind=iintegers) :: varid
integer(kind=iintegers) :: dimid 
integer(kind=iintegers) :: tstep
integer(kind=iintegers) :: xdim
integer(kind=iintegers) :: i
integer(kind=iintegers) :: j

character(len=60) :: filename

logical, save :: firstcall
data firstcall /.true./

!if (firstcall) then
  filename = filename1(1:len_trim(filename1))//'.nc'

! Opening the netCDF file
  call NCERROR(NF90_OPEN(filename,nf90_nowrite,ncid),'opening file')

! Inquiring time variable ID
  call NCERROR(NF90_INQ_DIMID(ncid,'tstep',dimid),'getting tstep id')

! Inquiring time variable dimension
  call NCERROR(NF90_INQUIRE_DIMENSION(ncid=ncid,dimid=dimid,len=tstep),'getting tstep dim')

! Inquiring the spatial coordinate ID
  call NCERROR(NF90_INQ_DIMID(ncid,'x',dimid),'getting land id')

! Inquiring the spatial coordinate dimension
  call NCERROR(NF90_INQUIRE_DIMENSION(ncid=ncid,dimid=dimid,len=xdim),'getting xdim dim')

! Allocating relevant arrays
  allocate(values_2d(xdim,tstep))
  allocate(values_1d(1:xdim))

  allocate (Sens_heat_flux(1:xdim,1:tstep))
  allocate (Lat_heat_flux (1:xdim,1:tstep))
  allocate (Long_net_rad  (1:xdim,1:tstep))
  allocate (Short_net_rad (1:xdim,1:tstep))
  allocate (Surf_temp     (1:xdim,1:tstep))
  allocate (Time                (1:tstep))

  call READVAR_2D('Sens_heat_flux')
  Sens_heat_flux(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Lat_heat_flux')
  Lat_heat_flux(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Long_net_rad')
  Long_net_rad(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Short_net_rad')
  Short_net_rad(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_2D('Surf_temp')
  Surf_temp(1:xdim,1:tstep) = values_2d(1:xdim,1:tstep)

  call READVAR_1D('Time')
  Time(1:tstep) = values_1d(1:tstep)

  call NCERROR(nf90_close(ncid), 'closing file')

  deallocate(values_2d)
  deallocate(values_1d)

!endif

  if (moving_average_window > 0) then
    allocate (Sens_heat_flux_mov_aver(1:xdim,1:tstep))
    allocate (Lat_heat_flux_mov_aver (1:xdim,1:tstep))
    allocate (Long_net_rad_mov_aver  (1:xdim,1:tstep))
    allocate (Short_net_rad_mov_aver (1:xdim,1:tstep))
    allocate (Surf_temp_mov_aver     (1:xdim,1:tstep))

    do i = 1, xdim
      call SIMPLE_MOVING_AVERAGE(Sens_heat_flux(i,1:tstep), &
      & Sens_heat_flux_mov_aver(i,1:tstep), tstep, moving_average_window)
      call SIMPLE_MOVING_AVERAGE(Lat_heat_flux(i,1:tstep), &
      & Lat_heat_flux_mov_aver(i,1:tstep), tstep, moving_average_window)
      call SIMPLE_MOVING_AVERAGE(Long_net_rad(i,1:tstep), &
      & Long_net_rad_mov_aver(i,1:tstep), tstep, moving_average_window)
      call SIMPLE_MOVING_AVERAGE(Short_net_rad(i,1:tstep), &
      & Short_net_rad_mov_aver(i,1:tstep), tstep, moving_average_window)
      call SIMPLE_MOVING_AVERAGE(Surf_temp(i,1:tstep), &
      & Surf_temp_mov_aver(i,1:tstep), tstep, moving_average_window)
    enddo
    filename = filename1(1:len_trim(filename1))//'_mov_aver'//'.nc'
    call NETCDF_WRITE_MOV_AVER
  endif


 
  if (mean_cycle_period > 0) then
    allocate (Sens_heat_flux_mean_cycle(1:xdim,1:mean_cycle_period))
    allocate (Lat_heat_flux_mean_cycle (1:xdim,1:mean_cycle_period))
    allocate (Long_net_rad_mean_cycle  (1:xdim,1:mean_cycle_period))
    allocate (Short_net_rad_mean_cycle (1:xdim,1:mean_cycle_period))
    allocate (Surf_temp_mean_cycle     (1:xdim,1:mean_cycle_period))
    if (moving_average_window > 0) then
      do i = 1, xdim
        call MEAN_CYCLE_CALC(Sens_heat_flux_mov_aver(i,1:tstep), & 
        & Sens_heat_flux_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Lat_heat_flux_mov_aver(i,1:tstep), &
        & Lat_heat_flux_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Long_net_rad_mov_aver(i,1:tstep), &
        & Long_net_rad_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Short_net_rad_mov_aver(i,1:tstep), &
        & Short_net_rad_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Surf_temp_mov_aver(i,1:tstep), &
        & Surf_temp_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
      enddo
    else
      do i = 1, xdim
        call MEAN_CYCLE_CALC(Sens_heat_flux(i,1:tstep), &
        & Sens_heat_flux_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Lat_heat_flux(i,1:tstep), &
        & Lat_heat_flux_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Long_net_rad(i,1:tstep), &
        & Long_net_rad_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Short_net_rad(i,1:tstep), &
        & Short_net_rad_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
         call MEAN_CYCLE_CALC(Surf_temp(i,1:tstep), &
        & Surf_temp_mean_cycle(i,1:mean_cycle_period), &
        & tstep, mean_cycle_period)
      enddo
    endif
    filename = filename1(1:len_trim(filename1))//'_mean_cycle'//'.nc'
    call NETCDF_WRITE_MEAN_CYCLE
  endif

deallocate (Sens_heat_flux)
deallocate (Lat_heat_flux )
deallocate (Long_net_rad  )
deallocate (Short_net_rad )
deallocate (Surf_temp     )
deallocate (Time          )

deallocate (Sens_heat_flux_mov_aver)
deallocate (Lat_heat_flux_mov_aver )
deallocate (Long_net_rad_mov_aver  )
deallocate (Short_net_rad_mov_aver )
deallocate (Surf_temp_mov_aver     )

deallocate (Sens_heat_flux_mean_cycle)
deallocate (Lat_heat_flux_mean_cycle )
deallocate (Long_net_rad_mean_cycle  )
deallocate (Short_net_rad_mean_cycle )
deallocate (Surf_temp_mean_cycle     )

!print*, 'Finished NETCDF_POSTPROCESS_SURF successfully!'
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

! Local variables
integer(kind=iintegers) :: start_2d(1:2)
integer(kind=iintegers) :: count_2d(1:2)

call NCERROR(NF90_INQ_VARID(ncid,namevar,varid))
print*, namevar//' varid is ',varid 

start_2d=(/1,1/)
count_2d=(/xdim,tstep/)
call NCERROR(NF90_GET_VAR(ncid,varid,values_2d,start_2d,count_2d), &
& 'reading var '//NAMEVAR)

END SUBROUTINE READVAR_2D


SUBROUTINE NETCDF_WRITE_MOV_AVER

call NCERROR(NF90_CREATE(filename,nf90_clobber,ncid))
call NCERROR(NF90_DEF_DIM(ncid, 'x'    , xdim          , dimids(1)))
call NCERROR(NF90_DEF_DIM(ncid, 'tstep', nf90_unlimited, dimids(2)))

call NCERROR(NF90_DEF_VAR(ncid,'Sens_heat_flux_mov_aver',nf90_double, &
& dimids(1:2),varids(1)))
call NCERROR(NF90_DEF_VAR(ncid,'Lat_heat_flux_mov_aver' ,nf90_double, &
& dimids(1:2),varids(2)))
call NCERROR(NF90_DEF_VAR(ncid,'Long_net_rad_mov_aver'  ,nf90_double, &
& dimids(1:2),varids(3)))
call NCERROR(NF90_DEF_VAR(ncid,'Short_net_rad_mov_aver' ,nf90_double, &
& dimids(1:2),varids(4)))
call NCERROR(NF90_DEF_VAR(ncid,'Surf_temp_mov_aver'     ,nf90_double, &
& dimids(1:2),varids(5)))
call NCERROR(NF90_DEF_VAR(ncid,'Time'                   ,nf90_double, &
& dimids(2)  ,varids(6)))

call NCERROR(NF90_PUT_ATT(ncid, varids(1), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(2), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(3), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(4), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(5), 'Units', 'K'))
call NCERROR(NF90_PUT_ATT(ncid, varids(6), 'Units', 'sec'))

call NCERROR(NF90_ENDDEF(ncid))


call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(1), &
 & values=Sens_heat_flux_mov_aver(1:xdim,1:tstep)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(2), &
 & values=Lat_heat_flux_mov_aver(1:xdim,1:tstep)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(3), &
 & values=Long_net_rad_mov_aver(1:xdim,1:tstep)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(4), &
 & values=Short_net_rad_mov_aver(1:xdim,1:tstep))) 
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(5), & 
 & values=Surf_temp_mov_aver(1:xdim,1:tstep)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(6), &
 & values=Time))

call NCERROR(NF90_CLOSE(ncid))

END SUBROUTINE NETCDF_WRITE_MOV_AVER


SUBROUTINE NETCDF_WRITE_MEAN_CYCLE

call NCERROR(NF90_CREATE(filename,nf90_clobber,ncid))
call NCERROR(NF90_DEF_DIM(ncid, 'x'    , xdim          , dimids(1)))
call NCERROR(NF90_DEF_DIM(ncid, 'tstep', nf90_unlimited, dimids(2)))

call NCERROR(NF90_DEF_VAR(ncid,'Sens_heat_flux_mean_cycle',nf90_double, &
& dimids(1:2),varids(1)))
call NCERROR(NF90_DEF_VAR(ncid,'Lat_heat_flux_mean_cycle' ,nf90_double, &
& dimids(1:2),varids(2)))
call NCERROR(NF90_DEF_VAR(ncid,'Long_net_rad_mean_cycle'  ,nf90_double, &
& dimids(1:2),varids(3)))
call NCERROR(NF90_DEF_VAR(ncid,'Short_net_rad_mean_cycle' ,nf90_double, &
& dimids(1:2),varids(4)))
call NCERROR(NF90_DEF_VAR(ncid,'Surf_temp_mean_cycle'     ,nf90_double, &
& dimids(1:2),varids(5)))
call NCERROR(NF90_DEF_VAR(ncid,'Time'                     ,nf90_double, &
& dimids(2)  ,varids(6)))

call NCERROR(NF90_PUT_ATT(ncid, varids(1), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(2), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(3), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(4), 'Units', 'W/m**2'))
call NCERROR(NF90_PUT_ATT(ncid, varids(5), 'Units', 'K'))
call NCERROR(NF90_PUT_ATT(ncid, varids(6), 'Units', 'sec'))

call NCERROR(NF90_ENDDEF(ncid))


call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(1), &
 & values=Sens_heat_flux_mean_cycle(1:xdim,1:mean_cycle_period)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(2), &
 & values=Lat_heat_flux_mean_cycle(1:xdim,1:mean_cycle_period)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(3), &
 & values=Long_net_rad_mean_cycle(1:xdim,1:mean_cycle_period)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(4), &
 & values=Short_net_rad_mean_cycle(1:xdim,1:mean_cycle_period))) 
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(5), & 
 & values=Surf_temp_mean_cycle(1:xdim,1:mean_cycle_period)))
call NCERROR(NF90_PUT_VAR(ncid=ncid,varid=varids(6), &
 & values=Time(1:mean_cycle_period)))

call NCERROR(NF90_CLOSE(ncid))

END SUBROUTINE NETCDF_WRITE_MEAN_CYCLE

END SUBROUTINE NETCDF_POSTPROCESS_SURF
END MODULE NETCDF_POSTPROCESS_LAKE
#endif
