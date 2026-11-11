 MODULE DRIVER_PARAMETERS

 use DRIVER_DATATYPES!, only : ireals, iintegers

 use INOUT_DRIVER_PARAMETERS!, only : &
 !& driver_subr_unit_min, &
 !& driver_subr_unit_max

 real(kind=ireals), parameter :: small_number = 1.d-5

 type, private :: realpararr
   !sequence
   real(kind=ireals) :: par(1:100) = -999. ! 100 - ndatamax
   logical :: ok = .false.
   character(len=30) :: name
 end type

 integer(kind=iintegers), parameter, private :: n_select_call = 20
 integer(kind=iintegers), parameter, private :: ndatamax = 100

 integer(kind=iintegers), private :: nunit = driver_subr_unit_min
 character(len=200), private :: line = 'begin_file'

 real(kind=ireals), private :: hour0 ; logical, private :: ok_hour0 = .false.
 real(kind=ireals), private :: tinteg ; logical, private :: ok_tinteg = .false.
 real(kind=ireals), private :: dt ; logical, private :: ok_dt = .false.
 real(kind=ireals), private :: height_T_q ; logical, private :: ok_height_T_q = .false.
 real(kind=ireals), private :: height_u ; logical, private :: ok_height_u = .false.
 real(kind=ireals), private :: interval ; logical, private :: ok_interval = .false.
 
 real(kind=ireals), private :: h10 ; logical, private :: ok_h10 = .false.
 real(kind=ireals), private :: l10 ; logical, private :: ok_l10 = .false.
 real(kind=ireals), private :: ls10 ; logical, private :: ok_ls10 = .false.
 real(kind=ireals), private :: hs10 ; logical, private :: ok_hs10 = .false.
 real(kind=ireals), private :: Ts0 ; logical, private :: ok_Ts0 = .false.
 real(kind=ireals), private :: Tb0 ; logical, private :: ok_Tb0 = .false.
 real(kind=ireals), private :: Tbb0 ; logical, private :: ok_Tbb0 = .false.
 real(kind=ireals), private :: h_ML0 ; logical, private :: ok_h_ML0 = .false.
 real(kind=ireals), private :: extwat ; logical, private :: ok_extwat = .false.
 real(kind=ireals), private :: extice ; logical, private :: ok_extice = .false.
 real(kind=ireals), private :: kor ; logical, private :: ok_kor = .false.
 real(kind=ireals), private :: trib_inflow ; logical, private :: ok_trib_inflow = .false.
 real(kind=ireals), private :: hbot(1:2), hbotchan(1:2), widthchan, lengthchan ! Assuming two interacting lakes
 real(kind=ireals), private :: Sals0 ; logical, private :: ok_Sals0 = .false.
 real(kind=ireals), private :: Salb0 ; logical, private :: ok_Salb0 = .false.
 real(kind=ireals), private :: fetch ; logical, private :: ok_fetch = .false.
 real(kind=ireals), private :: phi ; logical, private :: ok_phi = .false.
 real(kind=ireals), private :: lam ; logical, private :: ok_lam = .false.
 real(kind=ireals), private :: us0 ; logical, private :: ok_us0 = .false.
 real(kind=ireals), private :: vs0 ; logical, private :: ok_vs0 = .false.
 real(kind=ireals), private :: Tm ; logical, private :: ok_Tm = .false.
 real(kind=ireals), private :: alphax ; logical, private :: ok_alphax = .false.
 real(kind=ireals), private :: alphay ; logical, private :: ok_alphay = .false.
 real(kind=ireals), private :: a_veg ; logical, private :: ok_a_veg = .false.
 real(kind=ireals), private :: c_veg ; logical, private :: ok_c_veg = .false.
 real(kind=ireals), private :: h_veg ; logical, private :: ok_h_veg = .false.
 real(kind=ireals), private :: area_lake ; logical, private :: ok_area_lake = .false.
 real(kind=ireals), private :: cellipt   ; logical, private :: ok_cellipt = .false.
 real(kind=ireals), private :: depth_area(1:ndatamax,1:2) ; logical, private :: ok_depth_area = .false.
 real(kind=ireals), private :: spinup_period ; logical, private :: ok_spinup_period = .false.
 real(kind=ireals), private :: cp_period ; logical, private :: ok_cp_period = .false.

 real(kind=ireals), private :: select_extwat(1:n_select_call)
 real(kind=ireals), private :: select_h10(1:n_select_call)

 real(kind=ireals), parameter :: missing_value = -999.

 integer(kind=iintegers), private :: control_point ; logical, private :: ok_control_point = .false.
 integer(kind=iintegers), private :: init_T ; logical, private :: ok_init_T = .false.
 integer(kind=iintegers), private :: year0 ; logical, private :: ok_year0 = .false.
 integer(kind=iintegers), private :: month0 ; logical, private :: ok_month0 = .false.
 integer(kind=iintegers), private :: day0 ; logical, private :: ok_day0 = .false.
 integer(kind=iintegers), private :: forc_format ; logical, private :: ok_forc_format = .false.
 integer(kind=iintegers), private :: form ; logical, private :: ok_form = .false.
 integer(kind=iintegers), private :: spinup_times ; logical, private :: ok_spinup_times = .false.
 integer(kind=iintegers), private :: rad ; logical, private :: ok_rad = .false.
 integer(kind=iintegers), private :: N_header_lines ; logical, private :: ok_N_header_lines = .false.
 integer(kind=iintegers), private :: N_coloumns ; logical, private :: ok_N_coloumns = .false.
 integer(kind=iintegers), private :: N_Year ; logical, private :: ok_N_Year = .false.
 integer(kind=iintegers), private :: N_Month ; logical, private :: ok_N_Month = .false.
 integer(kind=iintegers), private :: N_Day ; logical, private :: ok_N_Day = .false.
 integer(kind=iintegers), private :: N_Hour ; logical, private :: ok_N_Hour = .false.
 integer(kind=iintegers), private :: N_Precip ; logical, private :: ok_N_Precip = .false.
 integer(kind=iintegers), private :: N_Uspeed ; logical, private :: ok_N_Uspeed = .false.
 integer(kind=iintegers), private :: N_Vspeed ; logical, private :: ok_N_Vspeed = .false.
 integer(kind=iintegers), private :: N_Temp ; logical, private :: ok_N_Temp = .false.
 integer(kind=iintegers), private :: N_Hum ; logical, private :: ok_N_Hum = .false.
 integer(kind=iintegers), private :: N_Pres ; logical, private :: ok_N_Pres = .false.
 integer(kind=iintegers), private :: N_SWdown ; logical, private :: ok_N_SWdown = .false.
 integer(kind=iintegers), private :: N_LWdown ; logical, private :: ok_N_LWdown = .false.
 integer(kind=iintegers), private :: N_NetRad ; logical, private :: ok_N_NetRad = .false.
 integer(kind=iintegers), private :: N_SensFlux ; logical, private :: ok_N_SensFlux = .false.
 integer(kind=iintegers), private :: N_LatentFlux ; logical, private :: ok_N_LatentFlux = .false.
 integer(kind=iintegers), private :: N_Ustar ; logical, private :: ok_N_Ustar = .false.      
 integer(kind=iintegers), private :: N_surfrad ; logical, private :: ok_N_surfrad = .false.
 integer(kind=iintegers), private :: N_cloud ; logical, private :: ok_N_cloud = .false.
 integer(kind=iintegers), private :: N_SurfTemp ; logical, private :: ok_N_SurfTemp = .false.
 integer(kind=iintegers), private :: npoints ; logical, private :: ok_npoints = .false.
 integer(kind=iintegers), private :: lakinterac ; logical, private :: ok_lakinterac = .false.
 integer(kind=iintegers), private :: lakeform ; logical, private :: ok_lakeform = .false.
 integer(kind=iintegers), private :: nstep_ncout ; logical, private :: ok_nstep_ncout = .false.
 integer(kind=iintegers), private :: call_FLake ; logical, private :: ok_call_FLake = .false.
 integer(kind=iintegers), private :: moving_average_window ; logical, private :: ok_moving_average_window = .false.
 integer(kind=iintegers), private :: mean_cycle_period ; logical, private :: ok_mean_cycle_period = .false.
 integer(kind=iintegers), private :: nstep_out_Flake ; logical, private :: ok_nstep_out_Flake = .false.
 
 integer(kind=iintegers), private :: select_call(1:n_select_call)

 character(len=60), private :: dataname ; logical, private :: ok_dataname = .false.
 
 type(realpararr), private :: effl_outflow

 contains
 SUBROUTINE DEFPARDRIVER &
& (ndrivinf,hour0_outt, tinteg_outt, dt_outt, height_T_q_outt, height_u_outt, interval_outt, &
&  h10_outt, select_h10_outt, l10_outt, ls10_outt, hs10_outt, Ts0_outt, &
&  Tb0_outt, Tbb0_outt, h_ML0_outt, &
&  extwat_outt, select_extwat_outt, extice_outt, &
&  kor_outt, trib_inflow_outt, effl_outflow_outt, &
&  widthchan_outt, lengthchan_outt, hbot_outt, hbotchan_outt, &
&  Sals0_outt, Salb0_outt, fetch_outt, phi_outt, &
&  lam_outt, us0_outt, vs0_outt, &
&  Tm_outt, alphax_outt, alphay_outt, a_veg_outt, c_veg_outt, &
&  h_veg_outt, area_lake_outt, cellipt_outt, depth_area_outt, &
&  year0_outt, month0_outt, day0_outt, npoints_outt, lakinterac_outt, lakeform_outt, &
&  select_call_outt, forc_format_outt, form_outt, &
&  spinup_times_outt, spinup_period_outt, cp_period_outt, rad_outt, &
&  N_header_lines_outt, N_coloumns_outt, N_Year_outt, N_Month_outt, &
&  N_Day_outt, N_Hour_outt, N_Precip_outt, N_Uspeed_outt, N_Vspeed_outt, &
&  N_Temp_outt, N_Hum_outt, N_Pres_outt, N_SWdown_outt, N_LWDown_outt, N_NetRad_outt, &
&  N_SensFlux_outt, N_LatentFlux_outt, N_Ustar_outt, N_surfrad_outt, &
&  N_cloud_outt, N_SurfTemp_outt, nstep_ncout_outt, &
&  init_T_outt, control_point_outt, call_Flake_outt, moving_average_window_outt, &
&  mean_cycle_period_outt, nstep_out_Flake_outt, dataname_outt)
      
!DEFPARDRIVER reads file with driving parameters

 implicit none
 
 integer(kind=iintegers), intent(in) :: ndrivinf

 real(kind=ireals), intent(out) :: hour0_outt
 real(kind=ireals), intent(out) :: tinteg_outt
 real(kind=ireals), intent(out) :: dt_outt
 real(kind=ireals), intent(out) :: height_T_q_outt
 real(kind=ireals), intent(out) :: height_u_outt
 real(kind=ireals), intent(out) :: interval_outt
 
 real(kind=ireals), intent(out) :: h10_outt
 real(kind=ireals), intent(out) :: l10_outt
 real(kind=ireals), intent(out) :: ls10_outt
 real(kind=ireals), intent(out) :: hs10_outt
 real(kind=ireals), intent(out) :: Ts0_outt
 real(kind=ireals), intent(out) :: Tb0_outt
 real(kind=ireals), intent(out) :: Tbb0_outt
 real(kind=ireals), intent(out) :: h_ML0_outt
 real(kind=ireals), intent(out) :: extwat_outt
 real(kind=ireals), intent(out) :: extice_outt
 real(kind=ireals), intent(out) :: kor_outt
 real(kind=ireals), intent(out) :: trib_inflow_outt, effl_outflow_outt(1:ndatamax)
 real(kind=ireals), intent(out) :: widthchan_outt, lengthchan_outt
 real(kind=ireals), intent(out) :: hbot_outt(1:2), hbotchan_outt(1:2) ! Assuming two interacting lakes
 real(kind=ireals), intent(out) :: Sals0_outt
 real(kind=ireals), intent(out) :: Salb0_outt
 real(kind=ireals), intent(out) :: fetch_outt
 real(kind=ireals), intent(out) :: phi_outt
 real(kind=ireals), intent(out) :: lam_outt
 real(kind=ireals), intent(out) :: us0_outt
 real(kind=ireals), intent(out) :: vs0_outt
 real(kind=ireals), intent(out) :: Tm_outt
 real(kind=ireals), intent(out) :: alphax_outt
 real(kind=ireals), intent(out) :: alphay_outt
 real(kind=ireals), intent(out) :: a_veg_outt
 real(kind=ireals), intent(out) :: c_veg_outt
 real(kind=ireals), intent(out) :: h_veg_outt
 real(kind=ireals), intent(out) :: area_lake_outt
 real(kind=ireals), intent(out) :: cellipt_outt
 real(kind=ireals), intent(out) :: depth_area_outt(1:ndatamax,1:2)
 real(kind=ireals), intent(out) :: spinup_period_outt
 real(kind=ireals), intent(out) :: cp_period_outt

 real(kind=ireals), intent(out) :: select_extwat_outt(1:n_select_call)
 real(kind=ireals), intent(out) :: select_h10_outt(1:n_select_call)

 integer(kind=iintegers), intent(out) :: init_T_outt
 integer(kind=iintegers), intent(out) :: control_point_outt
 integer(kind=iintegers), intent(out) :: year0_outt
 integer(kind=iintegers), intent(out) :: month0_outt
 integer(kind=iintegers), intent(out) :: day0_outt
 integer(kind=iintegers), intent(out) :: forc_format_outt
 integer(kind=iintegers), intent(out) :: form_outt
 integer(kind=iintegers), intent(out) :: spinup_times_outt
 integer(kind=iintegers), intent(out) :: rad_outt
 integer(kind=iintegers), intent(out) :: N_header_lines_outt
 integer(kind=iintegers), intent(out) :: N_coloumns_outt
 integer(kind=iintegers), intent(out) :: N_Year_outt
 integer(kind=iintegers), intent(out) :: N_Month_outt
 integer(kind=iintegers), intent(out) :: N_Day_outt
 integer(kind=iintegers), intent(out) :: N_Hour_outt
 integer(kind=iintegers), intent(out) :: N_Precip_outt
 integer(kind=iintegers), intent(out) :: N_Uspeed_outt
 integer(kind=iintegers), intent(out) :: N_Vspeed_outt
 integer(kind=iintegers), intent(out) :: N_Temp_outt
 integer(kind=iintegers), intent(out) :: N_Hum_outt
 integer(kind=iintegers), intent(out) :: N_Pres_outt
 integer(kind=iintegers), intent(out) :: N_SWdown_outt
 integer(kind=iintegers), intent(out) :: N_LWdown_outt
 integer(kind=iintegers), intent(out) :: N_NetRad_outt
 integer(kind=iintegers), intent(out) :: N_SensFlux_outt
 integer(kind=iintegers), intent(out) :: N_LatentFlux_outt
 integer(kind=iintegers), intent(out) :: N_Ustar_outt      
 integer(kind=iintegers), intent(out) :: N_surfrad_outt  
 integer(kind=iintegers), intent(out) :: N_cloud_outt  
 integer(kind=iintegers), intent(out) :: N_SurfTemp_outt  
 integer(kind=iintegers), intent(out) :: npoints_outt
 integer(kind=iintegers), intent(out) :: lakinterac_outt
 integer(kind=iintegers), intent(out) :: lakeform_outt
 integer(kind=iintegers), intent(out) :: nstep_ncout_outt
 integer(kind=iintegers), intent(out) :: call_FLake_outt
 integer(kind=iintegers), intent(out) :: moving_average_window_outt
 integer(kind=iintegers), intent(out) :: mean_cycle_period_outt
 integer(kind=iintegers), intent(out) :: nstep_out_Flake_outt
 
 integer(kind=iintegers), intent(out) :: select_call_outt(1:n_select_call)

 character(len=60), intent(out) :: dataname_outt
 
!Local variables      
 integer(kind=iintegers) :: i !Loop index
 character(len=60)  :: driver_file, chwork
 
 logical, save :: firstcall = .true.
 logical, save :: all_par_set = .true.
 
 if (firstcall .or. ndrivinf > 1) then
  call CHECK_UNIT(driver_subr_unit_min,driver_subr_unit_max,nunit)            
  open  (nunit,file='driver_file.dat',status='old')
  read  (nunit,'(a)') line
  read  (nunit,'(a)') driver_file
  close (nunit)
  if (ndrivinf > 1) then
    ! Defining the position before file extension 
    i = 1
    do while (driver_file(i:i) /= '.') 
      i = i + 1
    enddo
    chwork = driver_file(i:len(driver_file)) ! saving extension
    write(driver_file(i:i+2),'(i3.3)') ndrivinf
    driver_file(i+3:len(driver_file)) = chwork(1:len_trim(chwork))
  endif
  open (nunit,file=driver_file,status='old')
  do while (line /= 'end')
    read (nunit,'(a)') line
    call READPARDRIVER()
  enddo
  close (nunit)
  

 if (.not.PAR_SET_REAL8(hour0,ok_hour0,hour0_outt)) &
 & call WRITE_PAR_NOT_SET('hour0')
 if (.not.PAR_SET_REAL8(tinteg,ok_tinteg,tinteg_outt)) &
 & call WRITE_PAR_NOT_SET('tinteg')
 if (.not.PAR_SET_REAL8(dt,ok_dt,dt_outt)) &
 & call WRITE_PAR_NOT_SET('dt')
 if (.not.PAR_SET_REAL8(height_T_q,ok_height_T_q,height_T_q_outt)) &
 & call WRITE_PAR_NOT_SET('height_T_q')
 if (.not.PAR_SET_REAL8(height_u,ok_height_u,height_u_outt)) &
 & call WRITE_PAR_NOT_SET('height_u')
 if (.not.PAR_SET_REAL8(interval,ok_interval,interval_outt)) &
 & call WRITE_PAR_NOT_SET('interval')
 
 if (.not.PAR_SET_REAL8(h10,ok_h10,h10_outt)) &
 & call WRITE_PAR_NOT_SET('h10')
 if (.not.PAR_SET_REAL8(l10,ok_l10,l10_outt)) &
 & call WRITE_PAR_NOT_SET('l10')
 if (.not.PAR_SET_REAL8(ls10,ok_ls10,ls10_outt)) &
 & call WRITE_PAR_NOT_SET('ls10')
 if (.not.PAR_SET_REAL8(hs10,ok_hs10,hs10_outt)) &
 & call WRITE_PAR_NOT_SET('hs10')
 if (.not.PAR_SET_REAL8(Ts0,ok_Ts0,Ts0_outt)) &
 & call WRITE_PAR_NOT_SET('Ts0')
 if (.not.PAR_SET_REAL8(Tb0,ok_Tb0,Tb0_outt)) &
 & call WRITE_PAR_NOT_SET('Tb0')
 if (.not.PAR_SET_REAL8(Tbb0,ok_Tbb0,Tbb0_outt)) &
 & call WRITE_PAR_NOT_SET('Tbb0')
 if (.not.PAR_SET_REAL8(h_ML0,ok_h_ML0,h_ML0_outt)) &
 & call WRITE_PAR_NOT_SET('h_ML0')
 if (.not.PAR_SET_REAL8(extwat,ok_extwat,extwat_outt)) &
 & call WRITE_PAR_NOT_SET('extwat')
 if (.not.PAR_SET_REAL8(extice,ok_extice,extice_outt)) &
 & call WRITE_PAR_NOT_SET('extice')
 if (.not.PAR_SET_REAL8(kor,ok_kor,kor_outt)) &
 & call WRITE_PAR_NOT_SET('kor')
 if (.not.PAR_SET_REAL8(trib_inflow,ok_trib_inflow,trib_inflow_outt)) &
 & call WRITE_PAR_NOT_SET('trib_inflow')
 if (.not.PAR_SET_REAL8_ARR(effl_outflow%par,effl_outflow%ok,effl_outflow_outt,ndatamax)) &
 & call WRITE_PAR_NOT_SET('effl_outflow')
 if (.not.PAR_SET_REAL8(Sals0,ok_Sals0,Sals0_outt)) &
 & call WRITE_PAR_NOT_SET('Sals0')
 if (.not.PAR_SET_REAL8(Salb0,ok_Salb0,Salb0_outt)) &
 & call WRITE_PAR_NOT_SET('Salb0')
 if (.not.PAR_SET_REAL8(fetch,ok_fetch,fetch_outt)) &
 & call WRITE_PAR_NOT_SET('fetch')
 if (.not.PAR_SET_REAL8(phi,ok_phi,phi_outt)) &
 & call WRITE_PAR_NOT_SET('phi')
 if (.not.PAR_SET_REAL8(lam,ok_lam,lam_outt)) &
 & call WRITE_PAR_NOT_SET('lam')
 if (.not.PAR_SET_REAL8(us0,ok_us0,us0_outt)) &
 & call WRITE_PAR_NOT_SET('us0')
 if (.not.PAR_SET_REAL8(vs0,ok_vs0,vs0_outt)) &
 & call WRITE_PAR_NOT_SET('vs0')
 if (.not.PAR_SET_REAL8(Tm,ok_Tm,Tm_outt)) &
 & call WRITE_PAR_NOT_SET('Tm')
 if (.not.PAR_SET_REAL8(alphax,ok_alphax,alphax_outt)) &
 & call WRITE_PAR_NOT_SET('alphax')
 if (.not.PAR_SET_REAL8(alphay,ok_alphay,alphay_outt)) &
 & call WRITE_PAR_NOT_SET('alphay')
 if (.not.PAR_SET_REAL8(a_veg,ok_a_veg,a_veg_outt)) &
 & call WRITE_PAR_NOT_SET('a_veg')
 if (.not.PAR_SET_REAL8(c_veg,ok_c_veg,c_veg_outt)) &
 & call WRITE_PAR_NOT_SET('c_veg')
 if (.not.PAR_SET_REAL8(h_veg,ok_h_veg,h_veg_outt)) &
 & call WRITE_PAR_NOT_SET('h_veg')
 if (.not.PAR_SET_REAL8(area_lake,ok_area_lake,area_lake_outt)) &
 & call WRITE_PAR_NOT_SET('area_lake')
  if (.not.PAR_SET_REAL8(cellipt,ok_cellipt,cellipt_outt)) &
 & call WRITE_PAR_NOT_SET('cellipt')

 depth_area_outt = depth_area ! Not obligatory data
 
 if (.not.PAR_SET_REAL8(spinup_period,ok_spinup_period,spinup_period_outt)) &
 & call WRITE_PAR_NOT_SET('spinup_period')
 if (.not.PAR_SET_REAL8(cp_period,ok_cp_period,cp_period_outt)) &
 & call WRITE_PAR_NOT_SET('cp_period')

 select_extwat_outt = select_extwat
 select_h10_outt = select_h10

 widthchan_outt = widthchan
 lengthchan_outt = lengthchan
 hbot_outt = hbot
 hbotchan_outt = hbotchan

 if (.not.PAR_SET_INTEGER4(init_T,ok_init_T,init_T_outt)) &
 & call WRITE_PAR_NOT_SET('init_T')
 if (.not.PAR_SET_INTEGER4(control_point,ok_control_point,control_point_outt)) &
 & call WRITE_PAR_NOT_SET('control_point')
 if (.not.PAR_SET_INTEGER4(year0,ok_year0,year0_outt)) &
 & call WRITE_PAR_NOT_SET('year0')
 if (.not.PAR_SET_INTEGER4(month0,ok_month0,month0_outt)) &
 & call WRITE_PAR_NOT_SET('month0')
 if (.not.PAR_SET_INTEGER4(day0,ok_day0,day0_outt)) &
 & call WRITE_PAR_NOT_SET('day0')
 if (.not.PAR_SET_INTEGER4(forc_format,ok_forc_format,forc_format_outt)) &
 & call WRITE_PAR_NOT_SET('forc_format')
 if (.not.PAR_SET_INTEGER4(form,ok_form,form_outt)) &
 & call WRITE_PAR_NOT_SET('form')
 if (.not.PAR_SET_INTEGER4(spinup_times,ok_spinup_times, spinup_times_outt)) &
 & call WRITE_PAR_NOT_SET('spinup_times')
 if (.not.PAR_SET_INTEGER4(rad, ok_rad, rad_outt)) &
 & call WRITE_PAR_NOT_SET('rad')
 if (.not.PAR_SET_INTEGER4(N_header_lines, ok_N_header_lines, N_header_lines_outt)) &
 & call WRITE_PAR_NOT_SET('N_header_lines')
 if (.not.PAR_SET_INTEGER4(N_coloumns, ok_N_coloumns, N_coloumns_outt)) &
 & call WRITE_PAR_NOT_SET('N_coloumns')
 if (.not.PAR_SET_INTEGER4(N_Year, ok_N_Year, N_Year_outt)) &
 & call WRITE_PAR_NOT_SET('N_Year')
 if (.not.PAR_SET_INTEGER4(N_Month, ok_N_Month, N_Month_outt)) &
 & call WRITE_PAR_NOT_SET('N_Month')
 if (.not.PAR_SET_INTEGER4(N_Day, ok_N_Day, N_Day_outt)) &
 & call WRITE_PAR_NOT_SET('N_Day')
 if (.not.PAR_SET_INTEGER4(N_Hour, ok_N_Hour, N_Hour_outt)) &
 & call WRITE_PAR_NOT_SET('N_Hour')
 if (.not.PAR_SET_INTEGER4(N_Precip, ok_N_Precip, N_Precip_outt)) &
 & call WRITE_PAR_NOT_SET('N_Precip')
 if (.not.PAR_SET_INTEGER4(N_Uspeed, ok_N_Uspeed, N_Uspeed_outt)) &
 & call WRITE_PAR_NOT_SET('N_Uspeed')
 if (.not.PAR_SET_INTEGER4(N_Vspeed, ok_N_Vspeed, N_Vspeed_outt)) &
 & call WRITE_PAR_NOT_SET('N_Vspeed')
 if (.not.PAR_SET_INTEGER4(N_Temp, ok_N_Temp, N_Temp_outt)) &
 & call WRITE_PAR_NOT_SET('N_Temp')
 if (.not.PAR_SET_INTEGER4(N_Hum, ok_N_Hum, N_Hum_outt)) &
 & call WRITE_PAR_NOT_SET('N_Hum')
 if (.not.PAR_SET_INTEGER4(N_Pres, ok_N_Pres, N_Pres_outt)) &
 & call WRITE_PAR_NOT_SET('N_Pres')
 if (.not.PAR_SET_INTEGER4(N_SWdown, ok_N_SWdown, N_SWdown_outt)) &
 & call WRITE_PAR_NOT_SET('N_SWdown')
 if (.not.PAR_SET_INTEGER4(N_LWdown, ok_N_LWdown, N_LWdown_outt)) &
 & call WRITE_PAR_NOT_SET('N_LWdown')
 if (.not.PAR_SET_INTEGER4(N_NetRad, ok_N_NetRad, N_NetRad_outt)) &
 & call WRITE_PAR_NOT_SET('N_NetRad')
 if (.not.PAR_SET_INTEGER4(N_SensFlux, ok_N_SensFlux, N_SensFlux_outt)) &
 & call WRITE_PAR_NOT_SET('N_SensFlux')      
 if (.not.PAR_SET_INTEGER4(N_LatentFlux, ok_N_LatentFlux, N_LatentFlux_outt)) &
 & call WRITE_PAR_NOT_SET('N_LatentFlux')            
 if (.not.PAR_SET_INTEGER4(N_Ustar, ok_N_Ustar, N_Ustar_outt)) &
 & call WRITE_PAR_NOT_SET('N_Ustar')                  
 if (.not.PAR_SET_INTEGER4(N_surfrad, ok_N_surfrad, N_surfrad_outt)) &
 & call WRITE_PAR_NOT_SET('N_surfrad')
 if (.not.PAR_SET_INTEGER4(N_cloud, ok_N_cloud, N_cloud_outt)) &
 & call WRITE_PAR_NOT_SET('N_cloud')
 if (.not.PAR_SET_INTEGER4(N_SurfTemp, ok_N_SurfTemp, N_SurfTemp_outt)) &
 & call WRITE_PAR_NOT_SET('N_SurfTemp')
 if (.not.PAR_SET_INTEGER4(npoints, ok_npoints, npoints_outt)) &
 & call WRITE_PAR_NOT_SET('npoints')
 if (.not.PAR_SET_INTEGER4(lakinterac, ok_lakinterac, lakinterac_outt)) &
 & call WRITE_PAR_NOT_SET('lakinterac')
 if (.not.PAR_SET_INTEGER4(lakeform, ok_lakeform, lakeform_outt)) &
 & call WRITE_PAR_NOT_SET('lakeform')
 if (.not.PAR_SET_INTEGER4(nstep_ncout, ok_nstep_ncout, nstep_ncout_outt)) &
 & call WRITE_PAR_NOT_SET('nstep_ncout')
 if (.not.PAR_SET_INTEGER4(call_FLake, ok_call_FLake, call_FLake_outt)) &
 & call WRITE_PAR_NOT_SET('call_FLake')
 if (.not.PAR_SET_INTEGER4(moving_average_window, ok_moving_average_window, moving_average_window_outt)) &
 & call WRITE_PAR_NOT_SET('moving_average_window')
 if (.not.PAR_SET_INTEGER4(mean_cycle_period, ok_mean_cycle_period, mean_cycle_period_outt)) &
 & call WRITE_PAR_NOT_SET('mean_cycle_period')
 if (.not.PAR_SET_INTEGER4(nstep_out_Flake, ok_nstep_out_Flake, nstep_out_Flake_outt)) &
 & call WRITE_PAR_NOT_SET('nstep_out_Flake')
 
 select_call_outt = select_call

 if (ok_dataname) then
   dataname_outt = dataname
 else
   write (*,*) 'The parameter dataname is not set in driver file'
   all_par_set = .false.
 endif
 
 if (.not.all_par_set) then
   write (*,*) 'Not all necessary parameters are set &
   & in driver file: STOP'
   STOP
 endif
  
  firstcall = .false. 
 endif

 contains
 FUNCTION PAR_SET_REAL8(par,ok_par,par_outt)
 implicit none
 
 logical :: PAR_SET_REAL8
 
!Input variables
 real(kind=ireals), intent(in) :: par
 logical, intent(in) :: ok_par
 
!Output variables
 real(kind=ireals), intent(out) :: par_outt
       
 if (ok_par) par_outt = par
 PAR_SET_REAL8 = ok_par
 
 END FUNCTION PAR_SET_REAL8


 FUNCTION PAR_SET_REAL8_ARR(par,ok_par,par_outt,n)
 implicit none
 
 logical :: PAR_SET_REAL8_ARR
 
!Input variables
 integer(kind=iintegers), intent(in) :: n
 real(kind=ireals), intent(in) :: par(1:n)
 logical, intent(in) :: ok_par
 
!Output variables
 real(kind=ireals), intent(out) :: par_outt(1:n)
       
 if (ok_par) par_outt(:) = par(:)
 PAR_SET_REAL8_ARR = ok_par
 
 END FUNCTION PAR_SET_REAL8_ARR

 
 FUNCTION PAR_SET_INTEGER4(par,ok_par,par_outt)

 implicit none
 
 logical :: PAR_SET_INTEGER4
 
!Input variables
 integer(kind=iintegers), intent(in) :: par
 logical, intent(in) :: ok_par
 
!Output variables
 integer(kind=iintegers), intent(out) :: par_outt
       
 if (ok_par) par_outt = par
 PAR_SET_INTEGER4 = ok_par
 
 END FUNCTION PAR_SET_INTEGER4
 
 SUBROUTINE WRITE_PAR_NOT_SET(variable_name)

 !use DRIVER_DATATYPES!, only : ireals, iintegers
 implicit none
 
!Input variables
 character(len=*), intent(in) :: variable_name
 
 write (*,*) 'The parameter ', &
 & variable_name(1:len_trim(variable_name)), &
 & ' is not set in driver file'
 
 all_par_set = .false.
 
 END SUBROUTINE WRITE_PAR_NOT_SET
 END SUBROUTINE DEFPARDRIVER


 SUBROUTINE READPARDRIVER()
 !use DRIVER_DATATYPES!, only : ireals, iintegers
 
 implicit none

!Local variables
 real(kind=ireals), allocatable :: work1(:,:)
 
 integer(kind=iintegers) :: i
 integer(kind=iintegers) :: n2
 integer(kind=iintegers) :: n1
 integer(kind=iintegers) :: nselect

 logical, save :: firstcall = .true.
 
!External functions
! real(kind=ireals),    external :: getvarval
! integer(kind=iintegers), external :: igetvarval

 if (firstcall) then
   select_call(:) = -1
   select_extwat(:) = -1
   select_h10(:) = -1
   depth_area(:,:) = -1.
 endif

 i=1
 if (line(1:1)==' '.or.line(1:1)==char(1)) then
   do while (line(i:i)==' '.or.line(i:i)==char(1))
     i=i+1
   enddo
   n1=i
   do while (line(i:i)/=' '.and.line(i:i)/=char(1))
     i=i+1
   enddo
   n2=i-1
 else
   n1=1
   do while (line(i:i)/=' '.and.line(i:i)/=char(1))
     i=i+1
   enddo
   n2=i-1
 endif
 
 if (line(n1:n1)/='#') then
 
  SELECT CASE (line(n1:n2))

!The parameters of time integration

  CASE ('call_Flake')
   call_Flake &
 &           = igetvarval(n1,n2,line,'The year of start integr.') 
    ok_call_Flake = .true.
   CASE ('year0')
    year0    = igetvarval(n1,n2,line,'The year of start integr.') 
    ok_year0 = .true.
   CASE ('month0')
    month0   = igetvarval(n1,n2,line,'The month of start integ.')
    ok_month0 = .true.
   CASE ('day0')
    day0     = igetvarval(n1,n2,line,'The day of start integ.  ')
    ok_day0 = .true.
   CASE ('hour0')
    hour0    = getvarval (n1,n2,line,'Initial hour             ')
    ok_hour0 = .true.
   CASE ('tinteg')
    tinteg   = getvarval (n1,n2,line,'Integration time, days   ')
    ok_tinteg = .true.
   CASE ('spinup_times')
    spinup_times &
 &           = igetvarval(n1,n2,line,'Spinup, days             ')
    ok_spinup_times = .true.
   CASE ('spinup_period')
    spinup_period &
 &           = getvarval (n1,n2,line,'Spinup period, days      ')
    ok_spinup_period = .true.
   CASE ('cp_period')
    cp_period &
 &           = getvarval (n1,n2,line,'Contr. point period, days')
    ok_cp_period = .true.
   CASE ('dt')
    dt       = getvarval (n1,n2,line,'Timestep, sec            ')
    ok_dt = .true.

! The group of parameters of forcing file 

   CASE ('npoints')
    npoints  = igetvarval(n1,n2,line,'The number of points     ')
    ok_npoints = .true.
   CASE ('lakinterac')
    lakinterac &
    &        = igetvarval(n1,n2,line,'Switch for lake interact.')
    ok_lakinterac = .true.
    if (lakinterac > 1) then
      allocate (work1(3,2)) ! Assuming two interacting lakes
      call READPROFILE(nunit,3,2,work1)
      widthchan = work1(1,1)
      lengthchan = work1(1,2)
      hbot(1:2) = work1(2,1:2)
      hbotchan(1:2) = work1(3,1:2)
      deallocate(work1)
    endif
   CASE ('lakeform')
    lakeform = igetvarval(n1,n2,line,'The lake form            ')
    ok_lakeform = .true.
   CASE ('select_call')
    nselect  = igetvarval(n1,n2,line,'Number of selected points')
    allocate (work1(1:nselect,1))
    call READPROFILE(nunit,nselect,1,work1)
    select_call(1:nselect) = int(work1(1:nselect,1))
    deallocate(work1)
!    ok_nselect = .true.
   CASE ('forc_format')
    forc_format &
 &           = igetvarval(n1,n2,line,'Format of input file     ')
    ok_forc_format = .true.
   CASE ('form')
    form     = igetvarval(n1,n2,line,'Format of ASCII file     ')
    ok_form = .true.
   CASE ('dataname')
    read (line((n2+1):100),*) dataname
!    print*, 'dataname = ', dataname 
    ok_dataname = .true.
   CASE ('height_T_q')
    height_T_q &
 &           = getvarval (n1,n2,line,'Height of t&q measur-ts  ')
    ok_height_T_q = .true.
   CASE ('height_u')
    height_u = getvarval (n1,n2,line,'Height of wind measur-ts ')
    ok_height_u = .true.
   CASE ('interval')
    interval = getvarval (n1,n2,line,'Time interval of input   ')
    ok_interval = .true.
   CASE ('rad')
    rad      = igetvarval(n1,n2,line,'Input of radiation       ')
    ok_rad = .true.

! The group of input text file format parameters

   CASE ('N_header_lines')
    N_header_lines &
 &           = igetvarval(n1,n2,line,'N_header_lines           ')
    ok_N_header_lines = .true.
   CASE ('N_coloumns')
    N_coloumns &
 &           = igetvarval(n1,n2,line,'N_Coloumns               ')
    ok_N_coloumns = .true.
   CASE ('N_Year')
    N_Year   = igetvarval(n1,n2,line,'N_Year                   ')
    ok_N_Year = .true.
   CASE ('N_Month')
    N_Month  = igetvarval(n1,n2,line,'N_Month                  ')
    ok_N_Month = .true.
   CASE ('N_Day')
    N_Day    = igetvarval(n1,n2,line,'N_Day                    ')
    ok_N_Day = .true.
   CASE ('N_Hour')
    N_Hour   = igetvarval(n1,n2,line,'N_Hour                   ')
    ok_N_Hour = .true.
   CASE ('N_Precip')
    N_Precip = igetvarval(n1,n2,line,'N_Precip                 ')
    ok_N_Precip = .true.
   CASE ('N_Uspeed')
    N_Uspeed = igetvarval(n1,n2,line,'N_Uspeed                 ')
    ok_N_Uspeed = .true.
   CASE ('N_Vspeed')
    N_Vspeed = igetvarval(n1,n2,line,'N_Vspeed                 ')
    ok_N_Vspeed = .true.
   CASE ('N_Temp')
    N_Temp   = igetvarval(n1,n2,line,'N_Temp                   ')
    ok_N_Temp = .true.
   CASE ('N_Hum')
    N_Hum    = igetvarval(n1,n2,line,'N_Hum                    ')
    ok_N_Hum = .true.
   CASE ('N_Pres')
    N_Pres   = igetvarval(n1,n2,line,'N_Pres                   ')
    ok_N_Pres = .true.
   CASE ('N_SWdown')
    N_SWdown = igetvarval(n1,n2,line,'N_SWdown                 ')
    ok_N_SWdown = .true.
   CASE ('N_LWdown')
    N_LWdown = igetvarval(n1,n2,line,'N_LWdown                 ')
    ok_N_LWdown = .true.        
   CASE ('N_NetRad')
    N_NetRad = igetvarval(n1,n2,line,'N_NetRad                 ')
    ok_N_NetRad = .true.        
   CASE ('N_SensFlux')
    N_SensFlux &        
    &        = igetvarval(n1,n2,line,'N_SensFlux               ')
    ok_N_SensFlux = .true.        
   CASE ('N_LatentFlux')
    N_LatentFlux &
    &        = igetvarval(n1,n2,line,'N_LatentFlux             ')
    ok_N_LatentFlux = .true.                        
   CASE ('N_Ustar')
    N_Ustar &
    &        = igetvarval(n1,n2,line,'N_Ustar                  ')
    ok_N_Ustar = .true.                                
   CASE ('N_surfrad')
    N_surfrad &
    &        = igetvarval(n1,n2,line,'N_surfrad                ')
    ok_N_surfrad = .true.
   CASE ('N_cloud')
    N_cloud &
    &        = igetvarval(n1,n2,line,'N_cloud                  ')
    ok_N_cloud = .true.
   CASE ('N_SurfTemp')
    N_SurfTemp &
    &        = igetvarval(n1,n2,line,'N_SurfTemp               ')
    ok_N_SurfTemp = .true.
   CASE ('h10')
    h10      = getvarval (n1,n2,line,'Init. depth of lake, m   ')
    ok_h10 = .true.
   CASE ('select_h10')
    nselect  = igetvarval(n1,n2,line,'Number of selected depths')
    allocate (work1(1:nselect,1:2))
    call READPROFILE(nunit,nselect,2,work1)
    do i = 1, nselect
      if (int(work1(i,1)) /= select_call(i)) then
        write(*,*) 'select_h10 must be set in the order of selected points: &
        & terminating program'
        STOP
      endif
    enddo
    select_h10(1:nselect) = work1(1:nselect,2)
    deallocate(work1)
!    ok_nselect = .true.
   CASE ('l10')
    l10      = getvarval (n1,n2,line,'Init. ice thickness, m   ')
    ok_l10 = .true.
   CASE ('ls10')
    ls10     = getvarval (n1,n2,line,'Init. deep ice thick., m ')
    ok_ls10 = .true.
   CASE ('hs10')
    hs10     = getvarval (n1,n2,line,'Init. snow depth, m      ')
    ok_hs10 = .true.
   CASE ('Ts0')
    Ts0      = getvarval (n1,n2,line,'Init. temp. at lake surf.')
    ok_Ts0 = .true.
   CASE ('Tb0')
    Tb0      = getvarval (n1,n2,line,'Init. temp. at lake bot. ')
    ok_Tb0 = .true.
   CASE ('Tbb0')
    Tbb0     = getvarval (n1,n2,line,'Init. temp. at soil bot. ')
    ok_Tbb0 = .true.
   CASE ('h_ML0')
    h_ML0    = getvarval (n1,n2,line,'Init. mixed layer depth  ')
    ok_h_ML0 = .true.
   CASE ('Tm')
    Tm       = getvarval (n1,n2,line,'Mean coloumn temperature ')
    ok_Tm = .true.
   CASE ('init_T')
    init_T   = igetvarval(n1,n2,line,'Switch for temp. init.   ')
    ok_init_T = .true.
   CASE ('control_point')
    control_point   = &
    &          igetvarval(n1,n2,line,'Switch for control point ')
    ok_control_point = .true.
   CASE ('Sals0')
    Sals0    = getvarval (n1,n2,line,'Init. surface salinity   ')
    ok_Sals0 = .true.
   CASE ('Salb0')
    Salb0    = getvarval (n1,n2,line,'Init. bottom salinity    ')
    ok_Salb0 = .true.
   CASE ('us0')
    us0      = getvarval (n1,n2,line,'Init. surface X-speed    ')
    ok_us0 = .true.
   CASE ('vs0')
    vs0      = getvarval (n1,n2,line,'Init. surface Y-speed    ')
    ok_vs0 = .true.
   CASE ('kor')
    kor      = getvarval (n1,n2,line,'The Coriolis parameter   ')
    ok_kor = .true.
   CASE ('phi')
    phi      = getvarval (n1,n2,line,'Latitude, deg            ')
    ok_phi = .true.
   CASE ('lam')
    lam      = getvarval (n1,n2,line,'Longitude, deg           ') 
    ok_lam = .true.
   CASE ('extwat')
    extwat   = getvarval (n1,n2,line,'The rad extinct in water ')
    ok_extwat = .true.
   CASE ('select_extwat')
    nselect  = igetvarval(n1,n2,line,'Number of selected extwat')
    allocate (work1(1:nselect,1:2))
    call READPROFILE(nunit,nselect,2,work1)
    do i = 1, nselect
      if (int(work1(i,1)) /= select_call(i)) then
        write(*,*) 'select_extwat must be set in the order of selected points: &
        & terminating program'
        STOP
      endif
    enddo
    select_extwat(1:nselect) = work1(1:nselect,2)
    deallocate(work1)
!    ok_ = .true.
   CASE ('extice')
    extice   = getvarval (n1,n2,line,'The rad extinction in ice')
    ok_extice = .true.
   CASE ('alphax')
    alphax   = getvarval (n1,n2,line,'The x-slope angle        ')
    ok_alphax = .true.
   CASE ('alphay')
    alphay   = getvarval (n1,n2,line,'The y-slope angle        ')
    ok_alphay = .true.
   CASE ('c_veg')
    c_veg    = getvarval (n1,n2,line,'c_veg                    ')
    ok_c_veg = .true.
   CASE ('a_veg')
    a_veg    = getvarval (n1,n2,line,'a_veg                    ')
    ok_a_veg = .true.
   CASE ('h_veg')
    h_veg    = getvarval (n1,n2,line,'The vegetation height    ')
    ok_h_veg = .true.
   CASE ('fetch')
    fetch    = getvarval (n1,n2,line,'The fetch, m             ')
    ok_fetch = .true.
   CASE ('trib_inflow')
    trib_inflow &
 &           = getvarval (n1,n2,line,'Tributary inflow, m**3/s ')
    ok_trib_inflow = .true.
   CASE ('effl_outflow')
    i       = igetvarval (n1,n2,line,'Order of outflow dep.    ') + 2 ! n+1 coefficients of polynomial dependence + 
                                                                      ! height of outflow bottom above lake bottom
    call READPROFILE(nunit,1,i,effl_outflow%par)
    effl_outflow%ok = .true.
   CASE ('area_lake')
    area_lake= getvarval (n1,n2,line,'The area of lake, m**2   ')
    ok_area_lake = .true.
   CASE ('cellipt')
    cellipt  = getvarval (n1,n2,line,'The ellipse aspect ratio ')
    ok_cellipt = .true.
   CASE ('morphometry')
    nselect  = igetvarval(n1,n2,line,'Number of morphom. depths')
    if (nselect > ndatamax) then
      write(*,*) 'Too much morphometry levels: STOP'
      STOP
    endif
    allocate (work1(1:nselect,1:2))
    call READPROFILE(nunit,nselect,2,work1)
    depth_area(1:nselect,1:2) = work1(1:nselect,1:2)
    depth_area(nselect+1:ndatamax,1) = work1(nselect,1)
    depth_area(nselect+1:ndatamax,2) = work1(nselect,2)
    deallocate(work1)
   
!  The group of parameters for netcdf output from driver

   CASE ('nstep_ncout')
    nstep_ncout &
 &           = igetvarval(n1,n2,line,'Nstep for netcdf output  ')
    ok_nstep_ncout = .true.
   CASE ('nstep_out_Flake')
    nstep_out_Flake &
 &           = igetvarval(n1,n2,line,'Nstep for Flake output   ')
    ok_nstep_out_Flake = .true.

!  The group of parameters for postprocessing
   CASE ('moving_average_window')
    moving_average_window &
 &           = igetvarval(n1,n2,line,'Moving average window    ')
    ok_moving_average_window = .true.
   CASE ('mean_cycle_period')
    mean_cycle_period &
 &           = igetvarval(n1,n2,line,'Mean cycle period        ')
    ok_mean_cycle_period = .true.

   CASE DEFAULT
    if (.not.line(n1:n2)=='end') then
     print*,'Unknown keyword [',line(n1:n2),'] in driver file: STOP'
     STOP
    endif
   END SELECT

  endif

  if (firstcall) firstcall = .false.
  CONTAINS

  SUBROUTINE READPROFILE(iunit,N_rows,N_coloumns,work)
  !use DRIVER_DATATYPES!, only : ireals, iintegers
  implicit none

! Input variables:
  integer(kind=iintegers), intent(in) :: iunit
  integer(kind=iintegers), intent(in) :: N_rows
  integer(kind=iintegers), intent(in) :: N_coloumns

! Output variables:
  real(kind=ireals), intent(out) :: work(N_rows, N_coloumns)

! Local variables
  integer(kind=iintegers) :: i
  integer(kind=iintegers) :: j

  do i=1, N_rows
    read(iunit,*) (work(i,j), j=1,N_coloumns)
  enddo

  END SUBROUTINE READPROFILE
  END SUBROUTINE READPARDRIVER    

SUBROUTINE CHECK_UNIT(unit_min,unit_max,nunit)

! The subroutine CHECK_UNIT checks if the output unit is already occupied,
! and if yes, it returns the number of the free unit
implicit none

! Input variables
integer(kind=iintegers), intent(in) :: unit_min
integer(kind=iintegers), intent(in) :: unit_max

! Input/output variables
integer(kind=iintegers), intent(inout) :: nunit

! Local variables
logical :: unit_opened

inquire (unit=nunit,opened=unit_opened)

if (unit_opened) then
  do while (unit_opened)
!    write(*,*) 'The unit ', nunit, 'is attempted &
!    & to be connected to a file, while already connected: incrementing unit'
    nunit = nunit + 1
    inquire (unit=nunit,opened=unit_opened)
  enddo
!  STOP
endif

if (nunit < unit_min .or. nunit > unit_max) then
  write(*,*) 'Error on LAKE model: the bounds of permitted input/ouput &
  &unit numbers exceeded: STOP'
  STOP
endif

END SUBROUTINE CHECK_UNIT


FUNCTION GETVARVAL(n1,n2,line,name)
implicit none

real(kind=ireals) :: GETVARVAL

! Input variables
integer,       intent(in):: n2,n1
character*200, intent(in):: line
character(len=*),  intent(in):: name

! Local variables
real(kind=ireals) work

read (line((n2+1):100),*) work
print*, name//' = ', work

GETVARVAL = work      

RETURN
END FUNCTION GETVARVAL


FUNCTION IGETVARVAL(n1,n2,line,name)
implicit none

integer(kind=iintegers) :: IGETVARVAL

! Input variables
integer(kind=iintegers),    intent(in):: n2,n1
character*200, intent(in):: line
character(len=*),  intent(in):: name

! Local variables
integer(kind=iintegers) iwork

read (line((n2+1):100),*) iwork
print*, name//' = ', iwork

IGETVARVAL = iwork      

RETURN
END FUNCTION IGETVARVAL

  END MODULE DRIVER_PARAMETERS
     

