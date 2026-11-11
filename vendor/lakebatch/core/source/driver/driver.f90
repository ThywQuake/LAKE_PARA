!-----ONE-DIMENSIONAL LAKE MODEL LAKE1.1
!-----Moscow State University, Scientific Research Computing Center
!-----Moscow, 119992, GSP-2, Vorobievy Gory, SRCC MSU 
!-----Victor M. Stepanenko, stepanen@srcc.msu.ru
!-----Release: December 2011

  PROGRAM CALL_LAKE
! This program calls the LAKE model.
! It is needed only in stand-alone runs; in interactive regime with
! atmospheric model it must be cut from the code.
  use DRIVER_DATATYPES, only : ireals, iintegers

  !use INOUT,ireals_2=>ireals,iintegers_2=>iintegers!, only : CHECK_UNIT
  use DRIVER_INTERFACES_MOD
  use DRIVER_PARAMETERS,ireals_3=>ireals,iintegers_3=>iintegers

  use ARRAYS, only : &
  & cost_function, calibr_par1, calibr_par2, &
  & calibrate_parameter

  use LAKE_MOD, only : INIT_LAKE, LAKE      
  use LAKEBATCH_RESTART, only : RESTART_IN_PRESENT, RESTART_OUT_REQUESTED, RESTART_CLOCK_IO
  use LB_NCOUT, only : LB_NC_CLOSE

  use PHYS_CONSTANTS, only : &
  & sigma, Rd_d_Rwv, &
  & emissivityofwater, &
  & emissivityofice, &  
  & emissivityofsnow 
 
  use PHYS_FUNC, only : &
  & NETLWRAD

  use LAKE_INTERACTION, only : LAKETRIBINT, OUTFLOW_DISCHARGE
  use TIME_LAKE_MOD, only : JULIAN_DATE

#ifdef MPI
      use MPI
#endif

  implicit none

  integer(kind=iintegers), parameter :: n_select_call = 20
  integer(kind=iintegers), parameter :: ndatamax = 100
  integer(kind=iintegers), parameter :: n_netcdf_surf_files_max = 10
  integer(kind=iintegers), parameter :: nvartribext = 4 !number of tributary variables
       
  real(kind=ireals), parameter :: day_sec = 24.*60.*60.
  real(kind=ireals), parameter :: hour_sec = 60.*60.
  real(kind=ireals), parameter :: Kelvin0 = 273.15d0
  real(kind=ireals), parameter :: row0 = 1.d+3
  real(kind=ireals), parameter :: omega = 7.29d-5 ! Angular velocity of the Earth's rotation, sec**(-1)
  real(kind=ireals), parameter :: pi = 3.141592654d0
  
  real(kind=ireals), parameter :: ch4_pres_atm0 = 1.75d-1 ! atmospheric partial methane pressure, Pa
  real(kind=ireals), parameter :: o2_pres_atm0 = 21278. ! atmospheric partial oxygen pressure, Pa
  real(kind=ireals), parameter :: co2_pres_atm0 = 39.01 ! atmospheric partial carbon dioxide pressure, Pa
  
  real(kind=ireals), allocatable :: ta(:), qa(:), pa(:), ua(:), va(:)
  real(kind=ireals), allocatable :: atm_rad(:)
  real(kind=ireals), allocatable :: Rad_bal(:)
  real(kind=ireals), allocatable :: tsurf(:)
  real(kind=ireals), allocatable :: wind(:)
  real(kind=ireals), allocatable :: solar_rad(:)
  real(kind=ireals), allocatable :: precip(:)
  real(kind=ireals), allocatable :: ts(:)
  real(kind=ireals), allocatable :: SensFlux(:), LatentFlux(:)
  real(kind=ireals), allocatable :: Ustar(:)
  real(kind=ireals), allocatable :: exch_coef(:)
  real(kind=ireals), allocatable :: surfrad(:)
  real(kind=ireals), allocatable :: ftot(:)
  real(kind=ireals), allocatable :: fdiff_lake_surf(:)
  real(kind=ireals), allocatable :: Long_net_rad(:)
  real(kind=ireals), allocatable :: Short_net_rad(:)
  real(kind=ireals), allocatable :: Sflux(:)
  real(kind=ireals), allocatable :: cloud(:)
  real(kind=ireals), allocatable :: SurfTemp(:)

  real(kind=ireals), allocatable :: lat(:), lon(:)
 
  real(kind=ireals), allocatable :: h10_2d(:,:), h_2d(:,:), l10_2d(:,:), ls10_2d(:,:), hs10_2d(:,:)
  real(kind=ireals), allocatable :: Ts0_2d(:,:), Tb0_2d(:,:), Tbb0_2d(:,:)
  real(kind=ireals), allocatable :: h_ML0_2d(:,:)
  real(kind=ireals), allocatable :: extwat_2d(:,:), extice_2d(:,:)
  real(kind=ireals), allocatable :: kor_2d(:,:)
  real(kind=ireals), allocatable :: trib_inflow_2d(:,:), effl_outflow_2d(:,:,:)
  real(kind=ireals), allocatable :: Sals0_2d(:,:), Salb0_2d(:,:)
  real(kind=ireals), allocatable :: fetch_2d(:,:)
  real(kind=ireals), allocatable :: phi_2d(:,:), lam_2d(:,:)
  real(kind=ireals), allocatable :: us0_2d(:,:), vs0_2d(:,:)
  real(kind=ireals), allocatable :: Tm_2d(:,:)
  real(kind=ireals), allocatable :: alphax_2d(:,:), alphay_2d(:,:)
  real(kind=ireals), allocatable :: a_veg_2d(:,:), c_veg_2d(:,:), h_veg_2d(:,:)
  real(kind=ireals), allocatable :: area_lake_2d(:,:), cellipt_2d(:,:)
  real(kind=ireals), allocatable :: depth_area_2d(:,:,:,:)
  real(kind=ireals), allocatable :: select_extwat_2d(:,:)
  real(kind=ireals), allocatable :: select_h10_2d(:,:)
 
  real(kind=ireals), allocatable :: dhdt(:)
  
  real(kind=ireals), allocatable :: T_snow_in(:)
  real(kind=ireals), allocatable :: T_ice_in(:)
  real(kind=ireals), allocatable :: T_mnw_in(:)
  real(kind=ireals), allocatable :: T_wML_in(:)
  real(kind=ireals), allocatable :: T_bot_in(:)
  real(kind=ireals), allocatable :: T_B1_in(:)
  real(kind=ireals), allocatable :: C_T_in(:)
  real(kind=ireals), allocatable :: h_snow_in(:)
  real(kind=ireals), allocatable :: h_ice_in(:)
  real(kind=ireals), allocatable :: h_ML_in(:)
  real(kind=ireals), allocatable :: H_B1_in(:)
  real(kind=ireals), allocatable :: T_sfc_p(:)

  real(kind=ireals), allocatable :: T_snow_out(:)
  real(kind=ireals), allocatable :: T_ice_out(:)
  real(kind=ireals), allocatable :: T_mnw_out(:)
  real(kind=ireals), allocatable :: T_wML_out(:)
  real(kind=ireals), allocatable :: T_bot_out(:)
  real(kind=ireals), allocatable :: T_B1_out(:)
  real(kind=ireals), allocatable :: C_T_out(:)
  real(kind=ireals), allocatable :: h_snow_out(:)
  real(kind=ireals), allocatable :: h_ice_out(:)
  real(kind=ireals), allocatable :: h_ML_out(:)
  real(kind=ireals), allocatable :: H_B1_out(:)
  real(kind=ireals), allocatable :: T_sfc_n(:)

  real(kind=ireals), allocatable :: sens_flux_Flake(:)
  real(kind=ireals), allocatable :: latent_flux_Flake(:)
  real(kind=ireals), allocatable :: Long_net_rad_Flake(:)
  real(kind=ireals), allocatable :: Short_net_rad_Flake(:)

  real(kind=ireals) :: select_extwat(1:n_select_call)
  real(kind=ireals) :: select_h10(1:n_select_call)
  real(kind=ireals) :: depth_area(1:ndatamax,1:2)
 
  real(kind=ireals) :: h10, l10, ls10, hs10
  real(kind=ireals) :: Ts0, Tb0, Tbb0
  real(kind=ireals) :: h_ML0
  real(kind=ireals) :: extwat, extice
  real(kind=ireals) :: kor
  real(kind=ireals) :: trib_inflow(1:nvartribext), effl_outflow(1:ndatamax)
  real(kind=ireals) :: widthchan, lengthchan
  real(kind=ireals) :: hbot(1:2), hbotchan(1:2) ! assuming two interacting lakes
  real(kind=ireals) :: Sals0, Salb0
  real(kind=ireals) :: fetch
  real(kind=ireals) :: phi, lam
  real(kind=ireals) :: us0, vs0
  real(kind=ireals) :: Tm
  real(kind=ireals) :: alphax, alphay
  real(kind=ireals) :: a_veg, c_veg, h_veg
  real(kind=ireals) :: area_lake, cellipt
  real(kind=ireals) :: spinup_period
  real(kind=ireals) :: cp_period
  
  real(kind=ireals) :: zref 
  real(kind=ireals) :: time
  real(kind=ireals) :: t_Flake, t_Lake
  real(kind=ireals) :: t1, t2
  real(kind=ireals) :: hour
  real(kind=ireals) :: depth_bs
  real(kind=ireals) :: dMsnowdt_in
  real(kind=ireals) :: kor_FLake
  real(kind=ireals) :: epsa
  real(kind=ireals) :: surfrad_FLake
  real(kind=ireals), pointer :: emissivity

  integer(kind=iintegers) :: i, j
  integer(kind=iintegers) :: ix, iy
  integer(kind=iintegers) :: nx, ny
  integer(kind=iintegers) :: i_year
  integer(kind=iintegers) :: year
  integer(kind=iintegers) :: month
  integer(kind=iintegers) :: day
  integer(kind=iintegers) :: N_netcdf
  integer(kind=iintegers) :: Nproc
  integer(kind=iintegers) :: icp !> Control point handler
  integer(kind=iintegers) :: icp_counter !> Control point counter

#ifdef mpi
  integer(kind=iintegers) :: rank_mpi
  integer(kind=iintegers) :: ierr_mpi
  integer(kind=iintegers) :: size_mpi
#endif

  logical :: flag_assim
  logical :: flag_print
  logical :: first_netcdf_surf_write
  logical :: close_netcdf_write
  logical :: select_call_log
  logical :: call_lake_log
  logical :: spinup_done
  logical :: no_spinup
  logical :: multi_lake_single_forcing
  logical :: output_laststep = .true.
  logical :: vartrib_inflow
  logical :: dataread
              
  real(kind=ireals) :: hour0
  real(kind=ireals) :: tinteg
  real(kind=ireals) :: dt
  real(kind=ireals) :: height_T_q
  real(kind=ireals) :: height_u
  real(kind=ireals) :: interval
  real(kind=ireals) :: tribinfl

  real(kind=ireals), allocatable :: calibrdata(:,:)
  real(kind=ireals), allocatable :: workcalibr(:,:)

  integer(kind=iintegers) :: init_T
  integer(kind=iintegers) :: control_point
  integer(kind=iintegers) :: year0
  integer(kind=iintegers) :: month0
  integer(kind=iintegers) :: day0
  integer(kind=iintegers) :: forc_format
  integer(kind=iintegers) :: form
  integer(kind=iintegers) :: spinup_times
  integer(kind=iintegers) :: rad
  integer(kind=iintegers) :: N_header_lines
  integer(kind=iintegers) :: N_coloumns
  integer(kind=iintegers) :: N_Year
  integer(kind=iintegers) :: N_Month
  integer(kind=iintegers) :: N_Day
  integer(kind=iintegers) :: N_Hour
  integer(kind=iintegers) :: N_Precip
  integer(kind=iintegers) :: N_Uspeed
  integer(kind=iintegers) :: N_Vspeed
  integer(kind=iintegers) :: N_Temp
  integer(kind=iintegers) :: N_Hum
  integer(kind=iintegers) :: N_Pres
  integer(kind=iintegers) :: N_SWdown
  integer(kind=iintegers) :: N_LWdown
  integer(kind=iintegers) :: N_NetRad
  integer(kind=iintegers) :: N_SurfTemp
  integer(kind=iintegers) :: N_SensFlux, N_LatentFlux, N_Ustar, N_surfrad, N_cloud
  integer(kind=iintegers) :: npoints, lakinterac, lakeform
  integer(kind=iintegers) :: nstep_ncout
  integer(kind=iintegers) :: nstep
  integer(kind=iintegers) :: nstep_final
  integer(kind=iintegers) :: call_Flake
  integer(kind=iintegers) :: moving_average_window
  integer(kind=iintegers) :: mean_cycle_period
  integer(kind=iintegers) :: nstep_out_Flake
  integer(kind=iintegers) :: Flake_output_unit = driver_subr_unit_min
  integer(kind=iintegers) :: laststep_unitout = driver_subr_unit_min
  integer(kind=iintegers) :: ixproc_min, ixproc_max

  integer(kind=iintegers) :: select_call(1:n_select_call)

  character(len=60) :: filenames_netcdf_surf(1:n_netcdf_surf_files_max)
  
  character(len=60) :: dataname
  character(len=60) :: outpath
  character(len=100) :: format_screen ! Work variable
  character(len=60) :: work_character

  data time    /0./
  data i_year  /0/
  data nstep /0/
  data t_Lake  /0./
  data t_Flake /0./
  data flag_assim /.false./
  data first_netcdf_surf_write /.true./
  data close_netcdf_write /.false./
  data select_call_log /.false./
  data spinup_done /.false./
  data Nproc /0/
      
#ifdef mpi
  call MPI_INIT(ierr_mpi)
  call MPI_COMM_RANK(MPI_COMM_WORLD,rank_mpi,ierr_mpi)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,size_mpi,ierr_mpi)
  Nproc = rank_mpi
#endif

  call DEFPARDRIVER &
 & (1,hour0, tinteg, dt, height_T_q, height_u, interval, &
 &  h10, select_h10, l10, ls10, hs10, Ts0, Tb0, Tbb0, h_ML0, &
 &  extwat, select_extwat, extice, &
 &  kor, trib_inflow(1), effl_outflow, &
 &  widthchan, lengthchan, hbot, hbotchan, &
 &  Sals0, Salb0, fetch, phi, lam, us0, vs0, &
 &  Tm, alphax, alphay, a_veg, c_veg, h_veg, area_lake, cellipt, depth_area, &
 &  year0, month0, day0, npoints, lakinterac, lakeform, select_call, forc_format, form, &
 &  spinup_times, spinup_period, cp_period, rad, &
 &  N_header_lines, N_coloumns, N_Year, N_Month, &
 &  N_Day, N_Hour, N_Precip, N_Uspeed, N_Vspeed, &
 &  N_Temp, N_Hum, N_Pres, N_SWdown, N_LWDown, N_NetRad, &
 &  N_SensFlux, N_LatentFlux, N_Ustar, N_surfrad, N_cloud, N_SurfTemp, nstep_ncout, &
 &  init_T, control_point, call_Flake, moving_average_window, mean_cycle_period, &
 &  nstep_out_Flake, dataname)

  nx = npoints
  if (lakinterac > 1) then
    ! The case of two interacting lakes:
    ! interacting lakes are assumed to be separated by
    ! short distance, so that atmospheric forcing is the same (see the code below)
    ny = lakinterac
  else
    ny = 1
  endif

  vartrib_inflow = (trib_inflow(1) == -999.) ! Tributary inflow varying with time
 
  allocate (ta(1:npoints) )
  allocate (qa(1:npoints) )
  allocate (pa(1:npoints) )
  allocate (ua(1:npoints) )
  allocate (va(1:npoints) )
  allocate (atm_rad(1:npoints) )
  allocate (Rad_bal(1:npoints) )
  allocate (tsurf(1:npoints) )
  allocate (wind(1:npoints) )
  allocate (solar_rad(1:npoints) )
  allocate (precip(1:npoints) )
  allocate (ts(1:npoints) )
  allocate (SensFlux(1:npoints) )
  allocate (LatentFlux(1:npoints) )
  allocate (Ustar(1:npoints) )
  allocate (exch_coef(1:npoints) )
  allocate (surfrad(1:npoints) )
  allocate (ftot(1:npoints) )      
  allocate (fdiff_lake_surf(1:npoints) )      
  allocate (Long_net_rad(1:npoints) )
  allocate (Short_net_rad(1:npoints) )
  allocate (Sflux(1:npoints) )
  allocate (cloud(1:npoints) )
  allocate (SurfTemp(1:npoints) )

  allocate (lat(1:npoints))
  allocate (lon(1:npoints))

! Allocation and assigning initial fields
  allocate (h10_2d(1:npoints,1:ny))    ; h10_2d(:,1) = h10
  allocate (h_2d(1:npoints,1:ny))      ; h_2d(:,1) = h10
  allocate (l10_2d(1:npoints,1:ny))    ; l10_2d(:,1) = l10
  allocate (ls10_2d(1:npoints,1:ny))   ; ls10_2d(:,1) = ls10
  allocate (hs10_2d(1:npoints,1:ny))   ; hs10_2d(:,1) = hs10
  allocate (Ts0_2d(1:npoints,1:ny))    ; Ts0_2d(:,1) = Ts0
  allocate (Tb0_2d(1:npoints,1:ny))    ; Tb0_2d(:,1) = Tb0
  allocate (Tbb0_2d(1:npoints,1:ny))   ; Tbb0_2d(:,1) = Tbb0
  allocate (h_ML0_2d(1:npoints,1:ny))  ; h_ML0_2d(:,1) = h_ML0
  allocate (extwat_2d(1:npoints,1:ny)) ; extwat_2d(:,1) = extwat
  allocate (extice_2d(1:npoints,1:ny)) ; extice_2d(:,1) = extice
  allocate (kor_2d(1:npoints,1:ny))    ; kor_2d(:,1) = kor
  allocate (trib_inflow_2d(1:npoints,1:ny)) ; trib_inflow_2d(:,1) = trib_inflow(1)
  allocate (effl_outflow_2d(1:ndatamax,1:npoints,1:ny)) ;
  do i = 1, npoints 
    effl_outflow_2d(1:ndatamax,i,1) = effl_outflow(1:ndatamax)
  enddo
  allocate (Sals0_2d(1:npoints,1:ny)) ; Sals0_2d(:,1) = Sals0
  allocate (Salb0_2d(1:npoints,1:ny)) ; Salb0_2d(:,1) = Salb0
  allocate (fetch_2d(1:npoints,1:ny)) ; fetch_2d(:,1) = fetch
  allocate (phi_2d(1:npoints,1:ny))   ; phi_2d(:,1) = phi
  allocate (lam_2d(1:npoints,1:ny))   ; lam_2d(:,1) = lam
  allocate (us0_2d(1:npoints,1:ny))   ; us0_2d(:,1) = us0
  allocate (vs0_2d(1:npoints,1:ny))   ; vs0_2d(:,1) = vs0
  allocate (Tm_2d(1:npoints,1:ny))    ; Tm_2d(:,1) = Tm
  allocate (alphax_2d(1:npoints,1:ny)) ; alphax_2d(:,1) = alphax
  allocate (alphay_2d(1:npoints,1:ny)) ; alphay_2d(:,1) = alphay
  allocate (a_veg_2d(1:npoints,1:ny))  ; a_veg_2d(:,1) = a_veg
  allocate (c_veg_2d(1:npoints,1:ny))  ; c_veg_2d(:,1) = c_veg
  allocate (h_veg_2d(1:npoints,1:ny))  ; h_veg_2d(:,1) = h_veg
  allocate (area_lake_2d(1:npoints,1:ny)) ; area_lake_2d(:,1) = area_lake
  allocate (cellipt_2d  (1:npoints,1:ny)) ; cellipt_2d  (:,1) = cellipt
  allocate (depth_area_2d(1:ndatamax,1:2,1:npoints,1:ny))
  do i = 1, npoints
    depth_area_2d(1:ndatamax,1:2,i,1) = depth_area(1:ndatamax,1:2)
  enddo
  allocate (select_extwat_2d(1:n_select_call,1:ny)) ; select_extwat_2d(:,1) = select_extwat(:)
  allocate (select_h10_2d(1:n_select_call,1:ny)) ; select_h10_2d(:,1) = select_h10(:)
  allocate (dhdt(1:ny)); dhdt(:) = 0. ! The level change rate due to flow between lakes

! Allocation and assigning initial fields for Flake variables (interacting lakes are not operational for FLake)
  if (call_Flake == 1) then
    depth_bs = 5.d0 ! the depth of bottom sediments layer in Flake model
    allocate (T_snow_in(1:npoints) ) ; T_snow_in(:) = 0.d0+Kelvin0
    allocate (T_ice_in(1:npoints) )  ; T_ice_in(:) = 0.d0 +Kelvin0
    allocate (T_mnw_in(1:npoints) ) 
    T_mnw_in(:) = (Ts0*h_ML0 + 0.5d0*(Ts0+Tb0)*(h10-h_ML0))/h10 + Kelvin0
    allocate (T_wML_in(1:npoints) )  ; T_wML_in(:) = Ts0 + Kelvin0
    allocate (T_bot_in(1:npoints) )  ; T_bot_in(:) = Tb0 + Kelvin0
    allocate (T_B1_in(1:npoints) )   ; T_B1_in(:) = Tbb0 + Kelvin0
    allocate (C_T_in(1:npoints) )    ; C_T_in(:) = 0.5d0
    allocate (h_snow_in(1:npoints) ) ; h_snow_in(:) = hs10
    allocate (h_ice_in(1:npoints) )  ; h_ice_in(:) = l10
    allocate (h_ML_in(1:npoints) )   ; h_ML_in(:) = h_ML0
    allocate (H_B1_in(1:npoints) )   ; H_B1_in(:) = 0.5d0*depth_bs
    allocate (T_sfc_p(1:npoints) )   ; T_sfc_p(:) = Ts0 + Kelvin0

    allocate (T_snow_out(1:npoints) )
    allocate (T_ice_out(1:npoints) )
    allocate (T_mnw_out(1:npoints) )
    allocate (T_wML_out(1:npoints) )
    allocate (T_bot_out(1:npoints) )
    allocate (T_B1_out(1:npoints) )
    allocate (C_T_out(1:npoints) )
    allocate (h_snow_out(1:npoints) )
    allocate (h_ice_out(1:npoints) )
    allocate (h_ML_out(1:npoints) )
    allocate (H_B1_out(1:npoints) )
    allocate (T_sfc_n(1:npoints) )
    
    allocate (sens_flux_Flake(1:npoints) )
    allocate (latent_flux_Flake(1:npoints) )
    allocate (Long_net_rad_Flake(1:npoints) )
    allocate (Short_net_rad_Flake(1:npoints) )
  endif

  if (ny > 1) then
    ! Reading data for interacting lakes
    do j = 2, ny ! currently only ny == 2 is operational
      call DEFPARDRIVER &
      & (j,hour0, tinteg, dt, height_T_q, height_u, interval, &
      &  h10, select_h10, l10, ls10, hs10, Ts0, Tb0, Tbb0, h_ML0, &
      &  extwat, select_extwat, extice, &
      &  kor, trib_inflow(1), effl_outflow, &
      &  widthchan, lengthchan, hbot, hbotchan, &
      &  Sals0, Salb0, fetch, phi, lam, us0, vs0, &
      &  Tm, alphax, alphay, a_veg, c_veg, &
      &  h_veg, area_lake, cellipt, depth_area, &
      &  year0, month0, day0, npoints, lakinterac, &
      &  lakeform, select_call, forc_format, form, &
      &  spinup_times, spinup_period, cp_period, rad, &
      &  N_header_lines, N_coloumns, N_Year, N_Month, &
      &  N_Day, N_Hour, N_Precip, N_Uspeed, N_Vspeed, &
      &  N_Temp, N_Hum, N_Pres, N_SWdown, N_LWDown, N_NetRad, &
      &  N_SensFlux, N_LatentFlux, N_Ustar, N_surfrad, N_cloud, N_SurfTemp, nstep_ncout, &
      &  init_T, control_point, call_Flake, moving_average_window, mean_cycle_period, &
      &  nstep_out_Flake, dataname)
      h10_2d(:,j) = h10
      h_2d(:,j) = h10
      l10_2d(:,j) = l10
      ls10_2d(:,j) = ls10
      hs10_2d(:,j) = hs10
      Ts0_2d(:,j) = Ts0
      Tb0_2d(:,j) = Tb0
      Tbb0_2d(:,j) = Tbb0
      h_ML0_2d(:,j) = h_ML0
      extwat_2d(:,j) = extwat
      extice_2d(:,j) = extice
      kor_2d(:,j) = kor
      trib_inflow_2d(:,j) = trib_inflow(1)
      do i = 1, npoints
        effl_outflow_2d(1:ndatamax,i,j) = effl_outflow(1:ndatamax)
      enddo
      Sals0_2d(:,j) = Sals0
      Salb0_2d(:,j) = Salb0
      fetch_2d(:,j) = fetch
      phi_2d(:,j) = phi
      lam_2d(:,j) = lam
      us0_2d(:,j) = us0
      vs0_2d(:,j) = vs0
      Tm_2d(:,j) = Tm
      alphax_2d(:,j) = alphax
      alphay_2d(:,j) = alphay
      a_veg_2d(:,j) = a_veg
      c_veg_2d(:,j) = c_veg
      h_veg_2d(:,j) = h_veg
      area_lake_2d(:,j) = area_lake
      cellipt_2d  (:,j) = cellipt
      do i = 1, npoints
        depth_area_2d(1:ndatamax,1:2,i,j) = depth_area(1:ndatamax,1:2)
      enddo
      ! Note that select_call must be the same in all driver files
      select_extwat_2d(:,j) = select_extwat(:)
      select_h10_2d(:,j) = select_h10(:)
    enddo
  endif
  
 
  do i = 1, n_select_call
    if (select_call(i) /= -1) then
      select_call_log = .true.
      exit
    endif
  enddo

! The values of physical properties for selected points
  if (select_call_log) then
    do iy = 1, ny
      do ix = 1, npoints
        c1 : do i = 1, n_select_call
          if (ix == select_call(i)) then
!           Water extinction coefficient, m**(-1)
            if (select_extwat_2d(i,iy) /= -1) then
              extwat_2d(ix,iy) = select_extwat_2d(i,iy)
            endif
!           Initial water depth, m
            if (select_h10_2d(i,iy) /= -1) then
              h10_2d(ix,iy) = select_h10_2d(i,iy)
            endif
            exit c1
          endif
        enddo c1
      enddo
    enddo
  endif

!  write(*,*) 'Selected points (1) and extinction coefficients (2)'
!  format_screen = '(" (1) ", i4, " (2) ", f6.3)'
!  do i = 1, n_select_call
!    write (*, format_screen) select_call(i), select_extwat(i)
!  enddo
!
!  write(*,*) 'Selected points (1) and initial depths (2)'
!  format_screen = '(" (1) ", i4, " (2) ", f7.1)'
!  do i = 1, n_select_call
!    write (*, format_screen) select_call(i), select_h10(i)
!  enddo


#ifdef mpi
  if (select_call_log) then
!   Distributing the lake points between MPI-processes
    iy = rank_mpi + 1 
    do i = 1, n_select_call
      if (i < iy .or. mod(i-iy,size_mpi) /= 0) then 
        select_call(i) = -1
      endif
    enddo
    do i = 1, n_select_call
      if (select_call(i) /= -1) then
        write(*,*) &
 &      'Process rank = ', rank_mpi, &
 &      'Lake point number = ', select_call(i)
      endif
    enddo
!    STOP
  else
    ixproc_min = npoints/size_mpi*rank_mpi + 1    
    if (rank_mpi < size_mpi-1) then 
      ixproc_max = npoints/size_mpi*(rank_mpi + 1)
    else
      ixproc_max = npoints
    endif
    write(*,*) 'My rank is ', rank_mpi, 'My points are ', ixproc_min, ixproc_max
!    print*, 'MPI implementation is not yet operational for simulation without &
! &   selected lake points'
!    STOP
  endif
#endif

  if (call_Flake == 1 .and. nstep_out_Flake > 0) then
! Opens the file for Flake output
    call CHECK_UNIT(driver_subr_unit_min,driver_subr_unit_max,Flake_output_unit)
    open (unit = Flake_output_unit,file = 'results/Flake_output.dat', &
    & status = 'unknown')
    write (unit = Flake_output_unit, fmt = '(14a12)') &
 &   'nstep', 'T_snow_out', 'T_ice_out', &
 &   'T_mnw_out',  'T_wML_out', &
 &   'T_bot_out',  'T_B1_out', &
 &   'h_snow_out', 'h_ice_out', &
 &   'h_ML_out',   'H_B1_out', &
 &   'T_sfc_n',    'SensFlux', &
 &   'LatentFlux'
  endif
  
  if (output_laststep) then
    call CHECK_UNIT(driver_subr_unit_min,driver_subr_unit_max,laststep_unitout)
    open (unit = laststep_unitout, &
    & file = 'results/laststep_output.dat', status = 'unknown')
  endif

!  if (npoints > 1 .and. forc_format == 0) then
!    print*,'Warning: ASCII file forces the multipoint simulation'
!    multi_lake_single_forcing = .true.
!  else
!    multi_lake_single_forcing = .false.
!  endif


  tsurf = 293.d0
  Sflux = 0.d0 !1.d-7
      
! Transforming tinteg units from days to seconds
  tinteg = tinteg*day_sec
  spinup_period = spinup_period*day_sec
  zref = height_T_q
  
  year  = year0
  month = month0
  day   = day0
  hour  = hour0

  nstep_final = int(tinteg/dt)

  call INIT_LAKE(1_iintegers,1_iintegers,nx,ny,'sam_map',dt)
  icp_counter = 0
  cp_period = cp_period*day_sec !converting days to seconds

  ! lakebatch restart: continue the saved run's clock exactly
  if (RESTART_IN_PRESENT()) call RESTART_CLOCK_IO(.true., year, month, day, hour, time)

  timecyc : do nstep = 1, nstep_final

   time = time + dt

   if (vartrib_inflow) then ! Inflow variable with time
     call TRIB_INFLOW_UPDATE(time,interval,dataname,trib_inflow(1))
     trib_inflow_2d(:,:) = trib_inflow(1)
   endif

   if (forc_format == 0) then ! Reading input file in ASCII format

     call METEODATA_ASCII &
 &    (time, dt, tsurf(1), &
 &    height_T_q, height_u, interval, &
 &    spinup_times, spinup_period, form, rad, &
 &    N_header_lines, N_coloumns, N_Year, N_Month, &
 &    N_Day, N_Hour, N_Precip, N_Uspeed, N_Vspeed, &
 &    N_Temp, N_Hum, N_Pres, N_SWdown, N_LWDown, N_NetRad, &
 &    N_SensFlux, N_LatentFlux, N_Ustar, N_surfrad, &
 &    N_cloud, N_SurfTemp, dataname, &
 &    ta, qa, pa, ua, va, atm_rad, Rad_bal, solar_rad, precip, &
 &    SensFlux, LatentFlux, Ustar, surfrad, cloud, SurfTemp, &
 &    outpath, spinup_done, npoints, dataread)

 elseif (forc_format == 1) then ! Reading input file in netcdf format
#ifdef netcdf_lib     
     call METEODATA_NETCDF &
 &    (time, dt, npoints, spinup_times, spinup_period, &
 &     height_T_q, height_u, interval, &
 &     dataname, &

 &     lat, lon, &
 
 &     ta, qa, pa, ua, va, atm_rad, solar_rad, precip, &
 &     outpath,spinup_done)
     
     ! Interacting lakes are supposed to be close to each other, so as to be forced by same atmospheric state
     phi_2d(1:npoints,1:ny) = lat(1:npoints)
     lam_2d(1:npoints,1:ny) = lon(1:npoints)
#endif
 endif
     
!_var_in: ta - air temperature at z=zref, K  
!_var_in: qa - specific humidity at z=zref, kg/kg
!_var_in: pa - atmospheric pressure, Pa
!_var_in: ua - wind speed at z=zref, m/s
!_var_in: atm_rad - downward atmospheric radiation, W/m**2
!_var_in: Rad_bal - net radiation, W/m**2
!_var_in: solar_rad - net solar radiation, W/m**2
!_var_in: precip - precipitation rate in terms of liquid water, m/sec
!_var_in: zref - first atmospheric model level, or level of measurements, m
!_var_out: ts - surface temperature, K
!_var_out: SensFlux - sensible heat flux, W/m**2
!_var_out: LatentFlux - latent heat flux, W/m**2
!_var_out: exch_coef - exchange coefficient (tau=ro*exch_coef*ua) W/m**2
!_var_in: dt - time step, s
!_var_in: ix - x-number of current model grid point at the surface
!_var_in: iy - y-number of current model grid point at the surface
!_var_in: nx - number of grid points in x direction 
!_var_in: ny - number of grid points in y direction 
! CPU time counter

       
! The assimilation algorithm is not used since it is not operational
  flag_assim = .false.

  if (time > spinup_times*spinup_period) then
    no_spinup = .true.
  else
    no_spinup = .false.
  endif
  
  if (spinup_done) then
    year = year0
    month = month0
    day = day0
    hour = hour0
  endif
  call JULIAN_DATE(year,month,day,hour,dt)

  call cpu_time(t1)

  ixcyc : do ix = 1, npoints

    if (select_call_log) then
      call_lake_log = .false.
      cL : do i = 1, n_select_call
        call_lake_log = (ix == select_call(i))
        if (call_lake_log) exit cL
      enddo cL
    else
      call_lake_log = .true.
#ifdef mpi
           if (ix < ixproc_min .or. ix > ixproc_max) call_lake_log = .false.
#endif           
     endif
   
     if (call_lake_log) then
 
     flag_print = no_spinup

!     if (multi_lake_single_forcing) then
!       ta(ix) = ta(1)
!       qa(ix) = qa(1)
!       pa(ix) = pa(1)
!       ua(ix) = ua(1)
!       va(ix) = va(1)
!       atm_rad(ix) = atm_rad(1)
!       solar_rad(ix) = solar_rad(1)
!       precip(ix) = precip(1)
!       ! Optional variables
!       cloud(ix) = cloud(1)
!       SensFlux(ix) = SensFlux(1)
!       LatentFlux(ix) = LatentFlux(1)
!       Ustar(ix) = Ustar(1)             
!       surfrad(ix) = surfrad(1)                        
!     endif

     exch_coef(ix) = Ustar(ix)*Ustar(ix) / &
     & (max(sqrt(ua(ix)*ua(ix) + va(ix)*va(ix)), small_number))

     ! h_2d - water layer thickness (lake depth)
     if (lakinterac > 1) then
       call LAKETRIBINT(area_lake_2d,h_2d,hbot,hbotchan,widthchan, &
       & lengthchan,lakinterac,dhdt)
     endif

     !Control point (CP)
     select case (control_point)
     case (0)
       !Do nothing with CP
       icp = 0 
     case (1)
       !Start with standard initial conditions, writing CP with a given timestep
       if (nstep == 1) then
         icp = 0
       else
         !Check CP timestep
         if (floor(time/cp_period) > icp_counter) then
           icp = 1
           icp_counter = icp_counter + 1
         else
           icp = 0
         endif
       endif
     case (2)
       !Start with control point, writing CP with a given timestep
       if (nstep == 1) then
         icp = 2 !Read CP as initial conditions
       else
         !Check CP timestep
         if (floor(time/cp_period) > icp_counter) then
           icp = 1
           icp_counter = icp_counter + 1
         else
           icp = 0
         endif
       endif
     end select

     iycyc : do iy = 1, ny

       tribinfl = trib_inflow_2d(ix,iy) 
       if (lakinterac > 1) then
         tribinfl = tribinfl - &
         & OUTFLOW_DISCHARGE(ndatamax,effl_outflow_2d(1,ix,iy),h_2d(ix,iy)) + &
         & dhdt(iy) ! Adding interaction of two lakes
       endif
       trib_inflow(1)             = tribinfl
       trib_inflow(2:nvartribext) = missing_value

       call LAKE(ta(ix),qa(ix),pa(ix),ua(ix),va(ix),atm_rad(ix),Rad_bal(ix), &
       &  solar_rad(ix),precip(ix),Sflux(ix), &
       &  ch4_pres_atm0, co2_pres_atm0, o2_pres_atm0, zref, dt, &
       &  h10_2d(ix,iy), l10_2d(ix,iy), ls10_2d(ix,iy), hs10_2d(ix,iy), Ts0_2d(ix,iy), &
       &  Tb0_2d(ix,iy), Tbb0_2d(ix,iy), h_ML0_2d(ix,iy), extwat_2d(ix,iy), &
       &  extice_2d(ix,iy), kor_2d(ix,iy), trib_inflow, Sals0_2d(ix,iy), &
       &  Salb0_2d(ix,iy), fetch_2d(ix,iy), phi_2d(ix,iy), lam_2d(ix,iy), &
       &  us0_2d(ix,iy), vs0_2d(ix,iy), Tm_2d(ix,iy), alphax_2d(ix,iy), &
       &  alphay_2d(ix,iy), a_veg_2d(ix,iy), c_veg_2d(ix,iy), h_veg_2d(ix,iy), &
       &  area_lake_2d(ix,iy), cellipt_2d(ix,iy), depth_area_2d(1,1,ix,iy), &
       &  ts(ix),SensFlux(ix),LatentFlux(ix), exch_coef(ix), surfrad(ix), &
       &  cloud(ix), SurfTemp(ix), ftot(ix), fdiff_lake_surf(ix), h_2d(ix,iy), &
       &  ix,iy,1_iintegers,1_iintegers,nx,ny,nx,ny,ndatamax,year,month,day,hour, &
       &  init_T,flag_assim,flag_print, outpath, spinup_done, dataread, lakeform, &
       &  -999_iintegers,0_iintegers,(/0_iintegers,0_iintegers,0_iintegers/),.false., &
       &  icp, nstep == nstep_final)

     enddo iycyc
 
     !if (nstep==nstep_final .and. output_laststep) then
     !  calibrif : if (calibrate_parameter) then
#ifdef mpi           
     !   if (.not. associated(calibr_par1)) &
     !   & write(*,*) 'calibr_par1 not associated', ix, nstep, rank_mpi 
#endif
     !   i = ubound(cost_function, dim = 1) - &
     !   &   lbound(cost_function, dim = 1) + 1
     !   
     !   if (.not.allocated(calibrdata)) then
     !     allocate(calibrdata(1:npoints,1:i+2))
     !     calibrdata = 0
     !   endif
     !                
     !   calibrdata(ix,1) = calibr_par1
     !   calibrdata(ix,2) = calibr_par2
     !   calibrdata(ix,3:i+2) = cost_function(:)
             
#ifdef mpi
     !   if (ix == ixproc_max) then
     !     write(*,*) 'Process rank ', rank_mpi, &
     !     & ' entered calibrdata output'
     !     allocate (workcalibr(1:npoints,1:i+2))
     !     call MPI_REDUCE (calibrdata, workcalibr, npoints*(i+2), &
     !     & MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr_mpi)
     !     calibrdata = workcalibr
     !     deallocate (workcalibr)
     !     if (rank_mpi == 0) then
#else        
     !    if (ix == npoints) then
#endif             
     !      !do j = 1, npoints
     !      !  write (laststep_unitout,'(2e12.4,<i>e12.4)') calibrdata(j,1:2+i)
     !      !enddo
     !      deallocate (calibrdata)
           
     !      write (*,*) 'Calibration data written'
#ifdef mpi               
     !      endif
#endif               
     !    endif
         
     !  endif calibrif
     !endif
 
     if (call_Flake == 1) then

     iy = 1 ! interacting lakes are not operational with FLake

     if (ta(ix) > Kelvin0) then
       dMsnowdt_in = precip(ix)*row0
     else
       dMsnowdt_in = 0.d0
     endif

     if (kor_2d(ix,iy) == -999.d0) then
       kor_Flake = 2.d0*omega*sin(phi_2d(ix,iy)/180.d0*pi)
     else
       kor_Flake = kor_2d(ix,iy)
     endif
 
!     write(*,*) ix,
! &    dMsnowdt_in, solar_rad(ix), atm_rad(ix), zref, zref,
! &    sqrt(ua(ix)**2+va(ix)**2), ta(ix), qa(ix), pa(ix),
                         
! &    h10_2d(ix,iy), fetch_2d(ix,iy),depth_bs,Tbb0_2d(ix,iy)+Kelvin0,
! &    kor_Flake, dt,
 
! &    T_snow_in(ix),T_ice_in(ix),T_mnw_in(ix),T_wML_in(ix),
! &    T_bot_in(ix), T_B1_in(ix),
! &    C_T_in(ix), h_snow_in(ix), h_ice_in(ix), h_ML_in(ix), 
! &    H_B1_in(ix), T_sfc_p(ix)
!     read*

     if (atm_rad(ix) == missing_value) then
       ! Longwave radiation is missing in atmospheric forcing
       if     (h_snow_in(ix) > 0.) then
         emissivity => emissivityofsnow
       elseif (h_ice_in (ix) > 0.) then
         emissivity => emissivityofice
       else
         emissivity => emissivityofwater
       endif
       surfrad_FLake = emissivity*sigma*T_sfc_p(ix)**4
       epsa = qa(ix)*pa(ix)/Rd_d_Rwv
       atm_rad(ix) = NETLWRAD(T_sfc_p(ix),ta(ix),epsa,cloud(ix),emissivity) + surfrad_FLake
     endif

     call FLAKE_INTERFACE &
 &   (dMsnowdt_in, solar_rad(ix), atm_rad(ix), zref, zref, &
 &    sqrt(ua(ix)**2+va(ix)**2), ta(ix),qa(ix),pa(ix), &
                         
 &    h10_2d(ix,iy), fetch_2d(ix,iy),depth_bs,Tbb0_2d(ix,iy)+Kelvin0, & ! 50.
 &    kor_Flake, extwat_2d(ix,iy), extice_2d(ix,iy), dt, &
 
 &    T_snow_in(ix),T_ice_in(ix),T_mnw_in(ix),T_wML_in(ix), &
 &    T_bot_in(ix), T_B1_in(ix), &
 &    C_T_in(ix), h_snow_in(ix), h_ice_in(ix), h_ML_in(ix), &
 &    H_B1_in(ix), T_sfc_p(ix), &

 &    T_snow_out(ix),T_ice_out(ix),T_mnw_out(ix),T_wML_out(ix), &
 &    T_bot_out(ix), T_B1_out(ix), &
 &    C_T_out(ix), h_snow_out(ix), h_ice_out(ix), h_ML_out(ix), &
 &    H_B1_out(ix), T_sfc_n(ix), &
 
 &    sens_flux_Flake(ix), latent_flux_Flake(ix), &
 &    Short_net_rad_Flake(ix), Long_net_rad_Flake(ix) )
      
!      print*, dMsnowdt_in, solar_rad(ix), atm_rad(ix), zref, zref, &
! &    sqrt(ua(ix)**2+va(ix)**2), ta(ix),qa(ix),pa(ix), &
!                         
! &    h10_2d(ix,iy), fetch_2d(ix,iy),depth_bs,Tbb0_2d(ix,iy)+Kelvin0, & ! 50.
! &    kor_Flake, extwat_2d(ix,iy), extice_2d(ix,iy), dt, &
! 
! &    T_snow_in(ix),T_ice_in(ix),T_mnw_in(ix),T_wML_in(ix), &
! &    T_bot_in(ix), T_B1_in(ix), &
! &    C_T_in(ix), h_snow_in(ix), h_ice_in(ix), h_ML_in(ix), &
! &    H_B1_in(ix), T_sfc_p(ix), &
!
! &    T_snow_out(ix),T_ice_out(ix),T_mnw_out(ix),T_wML_out(ix), &
! &    T_bot_out(ix), T_B1_out(ix), &
! &    C_T_out(ix), h_snow_out(ix), h_ice_out(ix), h_ML_out(ix), &
! &    H_B1_out(ix), T_sfc_n(ix), &
! 
! &    sens_flux_Flake(ix), latent_flux_Flake(ix), &
! &    Short_net_rad_Flake(ix), Long_net_rad_Flake(ix)
!      read*
!      if (mod(nstep,100)==0) then
!        write(*,*) 'Mixed layer depth from Flake ', h_ML_out(ix)
!      endif

! Updating Flake variables
      T_snow_in(ix) = T_snow_out(ix)
      T_ice_in(ix)  = T_ice_out(ix)
      T_mnw_in(ix)  = T_mnw_out(ix)
      T_wML_in(ix)  = T_wML_out(ix)
      T_bot_in(ix)  = T_bot_out(ix)
      T_B1_in(ix)   = T_B1_out(ix)
      C_T_in(ix)    = C_T_out(ix)
      h_snow_in(ix) = h_snow_out(ix)
      h_ice_in(ix)  = h_ice_out(ix)
      h_ML_in(ix)   = h_ML_out(ix)
      H_B1_in(ix)   = H_B1_out(ix)
      T_sfc_p(ix)   = T_sfc_n(ix)

      if (nstep_out_Flake > 0 .and. &
 &      (nstep == 1 .or. mod(nstep,nstep_out_Flake) == 0)) then
        write (unit = Flake_output_unit, fmt = '(i12, 13f12.4)') &
 &       nstep, &
 &       T_snow_out(ix), T_ice_out(ix), &
 &       T_mnw_out(ix),  T_wML_out(ix), &
 &       T_bot_out(ix),  T_B1_out(ix), &
 &       h_snow_out(ix), h_ice_out(ix), &
 &       h_ML_out(ix),   H_B1_out(ix), &
 &       T_sfc_n(ix),    sens_flux_Flake(ix), &
 &       latent_flux_Flake(ix)
      endif
      
     endif
     endif

 
   enddo ixcyc

   call cpu_time(t2)

   t_Lake = t_Lake + t2 - t1

   if (spinup_done) then
     spinup_done = .false.
   endif

#ifdef netcdf_lib
   if (no_spinup.and.nstep_ncout/=-1.and. &
 &  (nstep==1.or.mod(nstep,nstep_ncout)==0.or.nstep==nstep_final)) then
     if (first_netcdf_surf_write) filenames_netcdf_surf(:)='none'
     if (nstep==nstep_final) close_netcdf_write = .true. ! The last call of Lake - final write to netcdf
! Writing the Lake model output to netcdf         
     N_netcdf = 1
     call NETCDF_WRITE_SURF &
 &    (SensFlux,LatentFlux,Long_net_rad,Short_net_rad, &
 &     ts,time,npoints,N_netcdf,Nproc,dataname, &
 &     work_character,close_netcdf_write)
     if (first_netcdf_surf_write) filenames_netcdf_surf(N_netcdf) = work_character
     if (call_Flake == 1) then
! Writing the Flake model output to netcdf              
       N_netcdf = 2
       call NETCDF_WRITE_SURF &
 &      (sens_flux_Flake,latent_flux_Flake, &
 &       Long_net_rad_Flake,Short_net_rad_Flake,T_sfc_n, &
 &       time,npoints,N_netcdf,Nproc,dataname, &
 &       work_character,close_netcdf_write) 
       if (first_netcdf_surf_write) filenames_netcdf_surf(N_netcdf) = work_character
! Screen output of Flake variables
!       do i = 1, n_select_call
!         if (select_call(i) /= -1) then
!           write(*,*) 'Flake temperature at point ', 
! &         select_call(i), 'is ', T_sfc_n(select_call(i))
!         endif
!       enddo
!      read*
     endif
     if (first_netcdf_surf_write) first_netcdf_surf_write = .false.
   endif
#endif

!   wind=sqrt(ua**2+va**2)
   if (flag_assim) flag_assim=.false.

   tsurf=ts

   if (year-year0>i_year) then
     print*, i_year+1, 'years left'
     i_year=i_year+1
   endif

  enddo timecyc

  ! lakebatch restart: save the clock with the model state
  if (RESTART_OUT_REQUESTED()) call RESTART_CLOCK_IO(.false., year, month, day, hour, time)
  ! lakebatch NetCDF output: flush and close the file
  call LB_NC_CLOSE
 
  print*, 'The integration finished successfully'
  print*, 'The time of integration is ', t_Lake, 'sec  '

! Postprocessing
#ifdef netcdf
  if (moving_average_window > 0 .or. mean_cycle_period > 0) then
    if (nstep_ncout > 0) then
      do i = 1, n_netcdf_surf_files_max
        if (filenames_netcdf_surf(i) /= 'none') then
          call NETCDF_POSTPROCESS_SURF &
 &         (filenames_netcdf_surf(i), &
 &         moving_average_window,mean_cycle_period)
        endif
      enddo
      write(*,*) 'The postprocessing is implemented'
    else
      write(*,*) 'Postprocessing is omitted: there has not been &
 &     netcdf surface output'
    endif
  endif
#endif
 
  if (call_Flake == 1 .and. nstep_out_Flake > 0) then
    close(Flake_output_unit)
  endif
 
#ifdef mpi
  call MPI_FINALIZE(ierr_mpi)
  write(*,*) 'The process ', rank_MPI, ' finished successfully'
#endif

700   format (f10.1, 3f9.2)
770   format (f6.1, 4f9.1, 2e10.2)
  
  END PROGRAM CALL_LAKE


      SUBROUTINE METEODATA_ASCII &
     & (time, dt, tsurf, &
     &  height_T_q, height_u, interval, &
     &  spinup_times, spinup_period, form, rad, &
     &  N_header_lines, N_coloumns, N_Year, N_Month, &
     &  N_Day, N_Hour, N_Precip, N_Uspeed, N_Vspeed, &
     &  N_Temp, N_Hum, N_Pres, N_SWdown, N_LWDown, N_NetRad, &
     &  N_SensFlux, N_LatentFlux, N_Ustar, N_surfrad, N_cloud, N_SurfTemp, &
     &  dataname, &
     &  ta, qa, pa, ua, va, atm_rad, Rad_bal, solar_rad, precip, &
     &  SensFlux, LatentFlux, Ustar, surfrad, cloud, SurfTemp, &
     &  outpath,spinup_done,npoints,dataread)

      use DRIVER_DATATYPES!, only : ireals, iintegers

      use INOUT_DRIVER_PARAMETERS!, only : &
      !& driver_file_unit_min, &
      !& driver_file_unit_max

      !use INOUT,ireals_=>ireals,iintegers_=>iintegers!, only : CHECK_UNIT

      use DRIVER_PARAMETERS,small_number_=>small_number!, only : &
      !& missing_value

!     Subroutine METEODATA_ASCII reads the file with data on atmospheric variables
!     in near surface air layer. 
!     It is needed only in stand-alone runs; in interactive regime with
!     atmospheric model it must be cut from the code.

      implicit none

!     Input variables
      real(kind=ireals), intent(in) :: time
      real(kind=ireals), intent(in) :: dt
      real(kind=ireals), intent(in) :: Tsurf
      real(kind=ireals), intent(in) :: height_T_q
      real(kind=ireals), intent(in) :: height_u
      real(kind=ireals), intent(in) :: interval
      real(kind=ireals), intent(in) :: spinup_period 
      
      integer(kind=iintegers), intent(in) :: form
      integer(kind=iintegers), intent(in) :: rad
      integer(kind=iintegers), intent(in) :: N_header_lines
      integer(kind=iintegers), intent(in) :: N_coloumns
      integer(kind=iintegers), intent(in) :: N_Year
      integer(kind=iintegers), intent(in) :: N_Month
      integer(kind=iintegers), intent(in) :: N_Day
      integer(kind=iintegers), intent(in) :: N_Hour
      integer(kind=iintegers), intent(in) :: N_Precip
      integer(kind=iintegers), intent(in) :: N_Uspeed
      integer(kind=iintegers), intent(in) :: N_Vspeed
      integer(kind=iintegers), intent(in) :: N_Temp
      integer(kind=iintegers), intent(in) :: N_Hum
      integer(kind=iintegers), intent(in) :: N_Pres
      integer(kind=iintegers), intent(in) :: N_SWdown
      integer(kind=iintegers), intent(in) :: N_LWdown
      integer(kind=iintegers), intent(in) :: N_NetRad
      integer(kind=iintegers), intent(in) :: N_SensFlux, N_LatentFlux, N_Ustar, N_surfrad, N_cloud
      integer(kind=iintegers), intent(in) :: N_SurfTemp
      integer(kind=iintegers), intent(in) :: spinup_times
      integer(kind=iintegers), intent(in) :: npoints
      
      character(len=60), intent(in) :: dataname
      
      logical, intent(inout) :: spinup_done

!     Output variables     
      real(kind=ireals), intent(out) :: ta(1:npoints)
      real(kind=ireals), intent(out) :: qa(1:npoints)
      real(kind=ireals), intent(out) :: pa(1:npoints)
      real(kind=ireals), intent(out) :: ua(1:npoints)
      real(kind=ireals), intent(out) :: va(1:npoints)
      real(kind=ireals), intent(out) :: atm_rad(1:npoints)
      real(kind=ireals), intent(out) :: Rad_bal(1:npoints)
      real(kind=ireals), intent(out) :: solar_rad(1:npoints)
      real(kind=ireals), intent(out) :: precip(1:npoints)
      real(kind=ireals), intent(out) :: SensFlux(1:npoints), LatentFlux(1:npoints), &
      & Ustar(1:npoints), surfrad(1:npoints), cloud(1:npoints), SurfTemp(1:npoints)
      
      character(len=60), intent(out) :: outpath

      logical, intent(out) :: dataread
      
!     Local variables      
      real(kind=ireals), parameter :: day_sec = 24.*60.*60.
      real(kind=ireals), parameter :: hour_sec = 60.*60.
      real(kind=ireals), parameter :: small_number = 1.d-5
      
     
      real(kind=ireals), allocatable :: taf(:)
      real(kind=ireals), allocatable :: qaf(:)
      real(kind=ireals), allocatable :: paf(:)
      real(kind=ireals), allocatable :: uaf(:)
      real(kind=ireals), allocatable :: vaf(:)
      real(kind=ireals), allocatable :: atm_radf(:)
      real(kind=ireals), allocatable :: solar_radf(:)
      real(kind=ireals), allocatable :: precipf(:)
      real(kind=ireals), allocatable :: Rad_balf(:)
      real(kind=ireals), allocatable :: SensFluxf(:), LatentFluxf(:) 
      real(kind=ireals), allocatable :: Ustarf(:), surfradf(:), cloudf(:)
      real(kind=ireals), allocatable :: SurfTempf(:)

      real(kind=ireals), allocatable :: ta_old(:)
      real(kind=ireals), allocatable :: qa_old(:)
      real(kind=ireals), allocatable :: pa_old(:)
      real(kind=ireals), allocatable :: ua_old(:)
      real(kind=ireals), allocatable :: va_old(:)
      real(kind=ireals), allocatable :: atm_rad_old(:)
      real(kind=ireals), allocatable :: solar_rad_old(:)
      real(kind=ireals), allocatable :: Rad_bal_old(:)
      real(kind=ireals), allocatable :: precip_old(:)
      real(kind=ireals), allocatable :: SensFlux_old(:), LatentFlux_old(:)
      real(kind=ireals), allocatable :: Ustar_old(:), surfrad_old(:), cloud_old(:)
      real(kind=ireals), allocatable :: SurfTemp_old(:)

      real(kind=ireals) :: z0_wind
      real(kind=ireals) :: sigma
      real(kind=ireals) :: ti_old

      real(kind=ireals) :: xx

      integer(kind=iintegers), parameter :: label_error_forcing_file = 123

      integer(8) :: ti_int
      integer(8) :: ti_int_old = -1
       
      integer(kind=iintegers) :: ispin = 1      
      integer(kind=iintegers) :: i, inpoints, ireaderr, k = 1

      integer(kind=iintegers), allocatable :: nunit(:)

      character(len=60) :: datafile
      character(len=30), allocatable :: workinput(:)
      character(len=3) :: numch

      logical :: firstcallM1 = .true.

      SAVE

      z0_wind = 0.0001
      sigma = 5.67e-8

      if_firstcall : if (firstcallM1) then
        allocate (nunit(1:npoints))
!       print*, 'The lake is  ', dataname
        do inpoints = 1, npoints
          nunit(inpoints) = driver_file_unit_min + inpoints
          if (npoints > 1) then
            write(numch,'(i3.3)') inpoints
            datafile = dataname(1:len_trim(dataname))//numch//'.dat'
          else
            datafile = dataname(1:len_trim(dataname))//'.dat'
          endif
          datafile = 'meteo/'//datafile
          outpath = 'results/'//dataname
          outpath = outpath(1:len_trim(outpath))//'/'
          call CHECK_UNIT(driver_file_unit_min,driver_file_unit_max,nunit(inpoints))
          open (unit=nunit(inpoints),file=datafile(1:len_trim(datafile)), &
          & status='old') 
        enddo
 
        allocate(taf(1:npoints))
        allocate(qaf(1:npoints))
        allocate(paf(1:npoints))
        allocate(uaf(1:npoints))
        allocate(vaf(1:npoints))
        allocate(atm_radf(1:npoints))
        allocate(solar_radf(1:npoints))
        allocate(precipf(1:npoints))
        allocate(Rad_balf(1:npoints))
        allocate(SensFluxf(1:npoints))
        allocate(LatentFluxf(1:npoints))
        allocate(Ustarf(1:npoints))
        allocate(surfradf(1:npoints))
        allocate(cloudf(1:npoints))
        allocate(SurfTempf(1:npoints))

        allocate(ta_old(1:npoints))
        allocate(qa_old(1:npoints))
        allocate(pa_old(1:npoints))
        allocate(ua_old(1:npoints))
        allocate(va_old(1:npoints))
        allocate(atm_rad_old(1:npoints))
        allocate(solar_rad_old(1:npoints))
        allocate(Rad_bal_old(1:npoints))
        allocate(precip_old(1:npoints))
        allocate(SensFlux_old(1:npoints))
        allocate(LatentFlux_old(1:npoints))
        allocate(Ustar_old(1:npoints))
        allocate(surfrad_old(1:npoints))
        allocate(cloud_old(1:npoints))
        allocate(SurfTemp_old(1:npoints))

        if (form == 0) then
!       This is a free form, adjusted in setup file
         allocate(workinput(1:N_coloumns))
         do inpoints = 1, npoints
           if (N_header_lines>0) then
             do i = 1, N_header_lines
               read (nunit(inpoints),'(a)', iostat = ireaderr, err = 123)
             enddo
           endif
           READ(nunit(inpoints),*,err = 123) (workinput(i),i=1,N_coloumns)           
           read(workinput(N_SWdown),*) solar_rad_old(inpoints)
           read(workinput(N_Precip),*) precip_old(inpoints)
           read(workinput(N_Temp),  *) ta_old(inpoints)
           read(workinput(N_Uspeed),*) ua_old(inpoints)
           read(workinput(N_Vspeed),*) va_old(inpoints)
           read(workinput(N_Pres),  *) pa_old(inpoints)
           read(workinput(N_Hum),   *) qa_old(inpoints)
           dataread = .true.
           ! Optional variables
           atm_rad_old   (inpoints) = EXMISS(N_LWdown)
           Rad_bal_old   (inpoints) = EXMISS(N_NetRad)
           SensFlux_old  (inpoints) = EXMISS(N_SensFlux)
           LatentFlux_old(inpoints) = EXMISS(N_LatentFlux)
           Ustar_old     (inpoints) = EXMISS(N_Ustar)
           surfrad_old   (inpoints) = EXMISS(N_surfrad)
           cloud_old     (inpoints) = EXMISS(N_cloud)
           SurfTemp_old  (inpoints) = EXMISS(N_SurfTemp)
           !if (rad == 1) then
           !  atm_rad_old(inpoints) = xx
           !elseif (rad == 2) then
           !  Rad_bal_old(inpoints) = xx
           !endif
         enddo 
        else
          write(*,*) 'Only form=0 of ASCII file is enabled: STOP'
          STOP
        endif
        
        taf(1:npoints)         = ta_old(1:npoints) 
        paf(1:npoints)         = pa_old(1:npoints) 
        qaf(1:npoints)         = qa_old(1:npoints) 
        uaf(1:npoints)         = ua_old(1:npoints) 
        vaf(1:npoints)         = va_old(1:npoints) 
        solar_radf(1:npoints)  = solar_rad_old(1:npoints)
        atm_radf(1:npoints)    = atm_rad_old(1:npoints)
        Rad_balf(1:npoints)    = Rad_bal_old (1:npoints)
        precipf(1:npoints)     = precip_old(1:npoints)
        SensFluxf(1:npoints)   = SensFlux_old(1:npoints)
        LatentFluxf(1:npoints) = LatentFlux_old(1:npoints)
        Ustarf(1:npoints)      = Ustar_old(1:npoints)
        surfradf(1:npoints)    = surfrad_old(1:npoints)
        cloudf(1:npoints)      = cloud_old(1:npoints)
        SurfTempf(1:npoints)   = SurfTemp_old(1:npoints)

        firstcallM1=.false.
 
      endif if_firstcall

      if (spinup_period > 0.) then 
        if (int(time/spinup_period) >= ispin .and. &
          & ispin <= spinup_times) then
          ispin = ispin + 1
          do inpoints = 1, npoints
            rewind nunit(inpoints) 
          enddo
          spinup_done = .true.
        endif  
      endif

      ti_int = int((time + small_number)/(hour_sec*interval))
      if (ti_int - ti_int_old /= 0) then
       ti_int_old    = ti_int
       ti_old        = ti_int_old*hour_sec*interval

       ta_old(:)         = taf(:)
       pa_old(:)         = paf(:)
       qa_old(:)         = qaf(:)
       ua_old(:)         = uaf(:)
       va_old(:)         = vaf(:)
       solar_rad_old(:)  = solar_radf(:)
       atm_rad_old(:)    = atm_radf(:)
       Rad_bal_old(:)    = Rad_balf(:)
       precip_old(:)     = precipf(:)
       SensFlux_old(:)   = SensFluxf(:)
       LatentFlux_old(:) = LatentFluxf(:)
       Ustar_old(:)      = Ustarf(:)
       surfrad_old(:)    = surfradf(:)
       cloud_old(:)      = cloudf(:)
       SurfTemp_old(:)   = SurfTempf(:)

       if (form == 0) then
         do inpoints = 1, npoints
           if (spinup_done .and.  N_header_lines > 0) then
             do i = 1, N_header_lines
               read (nunit(inpoints),'(a)', err = 123)
             enddo
           endif
           read(nunit(inpoints),*, iostat = ireaderr, err = 123) (workinput(i),i=1,N_coloumns)
           read(workinput(N_SWdown),*) solar_radf(inpoints)
           read(workinput(N_Precip),*) precipf(inpoints)
           read(workinput(N_Temp),  *) taf(inpoints)
           read(workinput(N_Uspeed),*) uaf(inpoints)
           read(workinput(N_Vspeed),*) vaf(inpoints)
           read(workinput(N_Pres),  *) paf(inpoints)
           read(workinput(N_Hum),   *) qaf(inpoints)
           dataread = .true.
           ! Optional variables
           atm_radf    (inpoints) = EXMISS(N_LWdown)
           Rad_balf    (inpoints) = EXMISS(N_NetRad)
           SensFluxf   (inpoints) = EXMISS(N_SensFlux)
           LatentFluxf (inpoints) = EXMISS(N_LatentFlux)
           Ustarf      (inpoints) = EXMISS(N_Ustar)
           surfradf    (inpoints) = EXMISS(N_surfrad)
           cloudf      (inpoints) = EXMISS(N_cloud)
           SurfTempf   (inpoints) = EXMISS(N_SurfTemp)
           !if (rad==1) then
           !  atm_radf(inpoints) = xx
           !elseif (rad==2) then
           !  Rad_balf(inpoints) = xx
           !endif
         enddo
       else
         write(*,*) 'Only form=0 of ASCII file is enabled: STOP'
         STOP
       endif
      end if

!      print*, taf, qaf, ta_old, qa_old, hour_sec*interval, time, ti_old
     

      do inpoints = 1, npoints
        ta(inpoints) = ta_old(inpoints) + &
        & (time - ti_old)*(taf(inpoints)-ta_old(inpoints))/(hour_sec*interval)
        pa(inpoints) = pa_old(inpoints) + &
        & (time - ti_old)*(paf(inpoints)-pa_old(inpoints))/(hour_sec*interval)
        qa(inpoints) = qa_old(inpoints) + &
        & (time - ti_old)*(qaf(inpoints)-qa_old(inpoints))/(hour_sec*interval)
        ua(inpoints) = ua_old(inpoints) + &
        & (time - ti_old)*(uaf(inpoints)-ua_old(inpoints))/(hour_sec*interval)
        va(inpoints) = va_old(inpoints) + &
        & (time - ti_old)*(vaf(inpoints)-va_old(inpoints))/(hour_sec*interval)
        solar_rad(inpoints) = solar_rad_old(inpoints) + (time - ti_old) * &
        &       (solar_radf(inpoints)-solar_rad_old(inpoints))/(hour_sec*interval)
        atm_rad(inpoints)   = atm_rad_old(inpoints)+  (time - ti_old) * &
        &       (atm_radf(inpoints)-atm_rad_old(inpoints))    /(hour_sec*interval)
        Rad_bal(inpoints)   = Rad_bal_old(inpoints)+  (time - ti_old) * &
        &       (Rad_balf(inpoints)-Rad_bal_old(inpoints))    /(hour_sec*interval)

        ! Optional parameters
        SensFlux(inpoints) = SensFlux_old(inpoints) + (time - ti_old)* &
        & (SensFluxf(inpoints) - SensFlux_old(inpoints))/(hour_sec*interval)
        LatentFlux(inpoints) = LatentFlux_old(inpoints) + (time - ti_old) * &
        & (LatentFluxf(inpoints) - LatentFlux_old(inpoints))/(hour_sec*interval)
        Ustar(inpoints) = Ustar_old(inpoints) + (time - ti_old) * &
        & (Ustarf(inpoints) - Ustar_old(inpoints))/(hour_sec*interval)
        surfrad(inpoints) = surfrad_old(inpoints) + (time - ti_old) * &
        & (surfradf(inpoints) - surfrad_old(inpoints))/(hour_sec*interval)
        cloud(inpoints) = cloud_old(inpoints) + (time - ti_old) * &
        & (cloudf(inpoints) - cloud_old(inpoints))/(hour_sec*interval)
        SurfTemp(inpoints) = SurfTemp_old(inpoints) + (time - ti_old) * &
        & (SurfTempf(inpoints) - SurfTemp_old(inpoints))/(hour_sec*interval)

!       The units of precipitation are m/sec
        precip(inpoints) = precipf(inpoints)

        if (height_T_q /= height_u) then
          ua(inpoints) = ua(inpoints)*log(height_T_q/Z0_WIND)/log(height_u/Z0_WIND)
          va(inpoints) = va(inpoints)*log(height_T_q/Z0_WIND)/log(height_u/Z0_WIND)
        endif

        !if (rad==2) then
!       !  For water surface
        !  atm_rad(inpoints) = Rad_bal(inpoints) - solar_rad(inpoints)*0.93 + 0.95*sigma*tsurf**4. ! Tsurf is not vectorized!
        !endif

      enddo

      RETURN

123  if (ireaderr == -1) then
        write(*,*) 'End of atmospheric forcing ASCII file encountered'
      else
        write(*,*) 'An error while reading ASCII forcing file. &
        & iostat = ', ireaderr
      endif
      !write(*,*) 'STOP'
      !STOP

      contains

      FUNCTION EXMISS(N)
      implicit none
      integer(kind=iintegers), intent(in) :: N
      real(kind=ireals) :: EXMISS

      if (N > 0) then
        read(workinput(N),*) EXMISS
      else
        EXMISS = missing_value
      endif

      END FUNCTION EXMISS
      
      END SUBROUTINE METEODATA_ASCII 
    
      

      SUBROUTINE TRIB_INFLOW_UPDATE(time,interval,dataname,trib_inflow)

!     Procedure updates inflow discharge, m**3/s
!     Not vectorized (for multiple lakes).

      use DRIVER_DATATYPES!, only : ireals, iintegers
      use INOUT_DRIVER_PARAMETERS!, only : &
      !& driver_subr_unit_min, &
      !& driver_subr_unit_max

      use INOUT,ireals_=>ireals,iintegers_=>iintegers!, only : CHECK_UNIT

      implicit none
    
!     Input variables
      real(kind=ireals), intent(in) :: time, interval
      character(len=*), intent(in) :: dataname

!     Output variables
      real(kind=ireals), intent(inout) :: trib_inflow

!     Local variables
      real(kind=ireals), parameter :: hour_sec = 60.*60.
      real(kind=ireals), save :: timec = 0.d0
      integer(kind=iintegers), save :: nunit = driver_subr_unit_min
      logical :: firstcall = .true.

      if (firstcall) then
        call CHECK_UNIT(driver_subr_unit_min,driver_subr_unit_max,nunit)
        open (unit=nunit,file='meteo/'//dataname(1:len_trim(dataname))//&
        &'_trib_inflow.dat', status='old')
      endif

      if (time > timec) then
        timec = timec + interval*hour_sec
        read(nunit,*) trib_inflow
      endif

      if (firstcall) firstcall = .false.
      END SUBROUTINE TRIB_INFLOW_UPDATE



#ifdef netcdf_lib
      SUBROUTINE METEODATA_NETCDF &
     & (time, dt, npoints, spinup_times, spinup_period, &
     &  height_T_q, height_u, interval, &
     &  dataname, &

     &  lat, lon, &
     
     &  ta, qa, pa, ua, va, atm_rad, solar_rad, precip, &
     &  outpath,spinup_done)

!     Subroutine METEODATA_NETCDF reads the netcdf file with data on atmospheric variables
!     in surface air layer. 
!     It is needed only in stand-alone runs; in interactive regime with
!     atmospheric model it must be cut from the code.

      use DRIVER_DATATYPES!, only : ireals, iintegers
      use DRIVER_INTERFACES_MOD
      implicit none

!     Input variables
      real(kind=ireals), intent(in) :: time
      real(kind=ireals), intent(in) :: dt
      real(kind=ireals), intent(in) :: height_T_q
      real(kind=ireals), intent(in) :: height_u
      real(kind=ireals), intent(in) :: interval
      real(kind=ireals), intent(in) :: spinup_period

      integer(kind=iintegers), intent(in) :: npoints
      integer(kind=iintegers), intent(in) :: spinup_times
      
      character(len=60), intent(in) :: dataname
      
!     Output variables     
      real(kind=ireals), intent(out) :: ta(1:npoints)
      real(kind=ireals), intent(out) :: qa(1:npoints)
      real(kind=ireals), intent(out) :: pa(1:npoints)
      real(kind=ireals), intent(out) :: ua(1:npoints)
      real(kind=ireals), intent(out) :: va(1:npoints)
      real(kind=ireals), intent(out) :: atm_rad(1:npoints)
      real(kind=ireals), intent(out) :: solar_rad(1:npoints)
      real(kind=ireals), intent(out) :: precip(1:npoints)
      real(kind=ireals), intent(out) :: lat(1:npoints)
      real(kind=ireals), intent(out) :: lon(1:npoints)
      
      character(len=60), intent(out) :: outpath

      logical, intent(inout) :: spinup_done      

!     Local variables      
      real(kind=ireals), parameter :: day_sec = 24.*60.*60.
      real(kind=ireals), parameter :: hour_sec = 60.*60.
      
      real(kind=ireals), allocatable, save :: prec(:,:)
      real(kind=ireals), allocatable, save :: I_atm(:,:)
      real(kind=ireals), allocatable, save :: Q_atm_lw(:,:)
      real(kind=ireals), allocatable, save :: U_a(:,:)
      real(kind=ireals), allocatable, save :: V_a(:,:)
      real(kind=ireals), allocatable, save :: T_a(:,:)
      real(kind=ireals), allocatable, save :: q_a(:,:)
      real(kind=ireals), allocatable, save :: P_a(:,:)

      real(kind=ireals), save :: time_old
      
      real(kind=ireals), save :: sigma = 5.67d-8 ! Stefan-Boltzman constant
      real(kind=ireals), save :: z0_wind = 1.d-3 ! Reference water surface roughness, m
      real(kind=ireals), save :: row0 = 1.d+3 ! Reference water density, kg/m**3

      integer(kind=iintegers) :: xdim
      integer(kind=iintegers), save :: ispin
      integer(kind=iintegers) :: i ! Loop index

      character(len=60), save :: datafile

      logical, save :: firstcallM1
      logical, save :: rewind_netcdf
      
      data firstcallM1 /.true./
      data rewind_netcdf /.false./
      data ispin /1/
      data time_old    /0./

      if (firstcallM1) then
      
        if (dt > interval*hour_sec) then
          print*, 'The time step of the model is greater than &
     &     the forcing timstep: STOP'
          STOP
        endif
        
        if (dmod(interval*hour_sec,dt) /= 0.) then
          print*, 'The forcing timestep is not dividable by model &
     &     timestep: STOP'
          STOP
        endif
              
        print*, 'The lake is  ', dataname
        datafile = dataname(1:len_trim(dataname))//'.nc'
        datafile = 'meteo/'//datafile
        outpath = 'results/'//dataname
        outpath = outpath(1:len_trim(outpath))//'/'
        
        allocate (prec(1:npoints, 1:2))
        allocate (I_atm(1:npoints, 1:2))
        allocate (Q_atm_lw(1:npoints, 1:2))
        allocate (U_a(1:npoints, 1:2))
        allocate (V_a(1:npoints, 1:2))
        allocate (T_a(1:npoints, 1:2))
        allocate (q_a(1:npoints, 1:2))
        allocate (P_a(1:npoints, 1:2))
        
        call NETCDF_READ_FORCING &
     &   (datafile, rewind_netcdf, npoints, &
     &   lat, lon, &
     &   prec, I_atm, Q_atm_lw, U_a, V_a, &
     &   T_a, q_a, P_a)
      endif

      if (int(time/spinup_period)>=ispin .and. ispin<=spinup_times) then
        ispin=ispin+1
        rewind_netcdf = .true.
        spinup_done = .true.
      endif

      if (rewind_netcdf) print*, 'NETCDF rewinded'

      if (dmod(time,interval*hour_sec) == 0) then
        call NETCDF_READ_FORCING &
     &   (datafile, rewind_netcdf, npoints, &
     &   lat, lon, &
     &   prec, I_atm, Q_atm_lw, U_a, V_a, &
     &   T_a, q_a, P_a)
        time_old  = time
      endif

      ta(:) = T_a(:,1) + (time - time_old)*(T_a(:,2)-T_a(:,1))/ &
     & (hour_sec*interval)
      pa(:) = P_a(:,1) + (time - time_old)*(P_a(:,2)-P_a(:,1))/ &
     & (hour_sec*interval)
      qa(:) = Q_a(:,1) + (time - time_old)*(Q_a(:,2)-Q_a(:,1))/ &
     & (hour_sec*interval)
      ua(:) = U_a(:,1) + (time - time_old)*(U_a(:,2)-U_a(:,1))/ &
     & (hour_sec*interval)
      va(:) = V_a(:,1) + (time - time_old)*(V_a(:,2)-V_a(:,1))/ &
     & (hour_sec*interval)
      
      solar_rad(:) = I_atm(:,1) + (time - time_old) * &
     &       (I_atm(:,2) - I_atm(:,1)) / (hour_sec*interval)
      atm_rad(:)   = Q_atm_lw(:,1)+  (time - time_old) * &
     &       (Q_atm_lw(:,2) - Q_atm_lw(:,1)) / (hour_sec*interval)

!     Converting precipitation units from kg/(m**3*sec) to m/sec 
      precip(:) = prec(:,2)/row0 

      if (height_T_q /= height_u) then
        ua(:)=ua(:)*log(height_T_q/Z0_WIND)/log(height_u/Z0_WIND)
      endif

      if (firstcallM1) firstcallM1 = .false.
      RETURN
      END SUBROUTINE METEODATA_NETCDF
#endif     

