MODULE DRIVER_INTERFACES_MOD


INTERFACE NETCDF_READ_FORCING
  SUBROUTINE NETCDF_READ_FORCING &
   & (filename1, rewind_netcdf, npoints, &
   & lat, lon, &
   & prec, I_atm, Q_atm_lw, U_a, V_a, &
   & T_a, q_a, P_a)

! Input variables

  use DRIVER_DATATYPES, only : ireals, iintegers

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

  END SUBROUTINE NETCDF_READ_FORCING
END INTERFACE NETCDF_READ_FORCING


INTERFACE NETCDF_WRITE_SURF
  SUBROUTINE NETCDF_WRITE_SURF &
   & (Sens_heat_flux,Lat_heat_flux,Long_net_rad,Short_net_rad,Surf_temp, &
   &  Time,npoints,Nfile,Nproc,filename1,filename_netcdf_file,close_log)

  use DRIVER_DATATYPES, only : ireals, iintegers
!  use netcdf
!  implicit none

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
  character(len=60), intent(out) :: filename_netcdf_file

  END SUBROUTINE NETCDF_WRITE_SURF
END INTERFACE NETCDF_WRITE_SURF

END MODULE DRIVER_INTERFACES_MOD
