MODULE DRIVING_PARAMS

use LAKE_DATATYPES, only : ireals, iintegers

use INOUT_PARAMETERS, only : &
& lake_subr_unit_min, &
& lake_subr_unit_max

use INOUT, only : &
& CHECK_UNIT, GETVARVAL, IGETVARVAL

type, public :: realpar
  !sequence
  real(kind=ireals) :: par
  logical :: ok = .false.
  character(len=30) :: name
  character(len=21) :: fill
end type

type, public :: intpar
  !sequence
  integer(kind=iintegers) :: par
  logical :: ok = .false.
  character(len=30) :: name
end type

integer(kind=iintegers), parameter :: numder = 44 ! Number derived-type model integer controls handled in unified manner
integer(kind=iintegers), parameter :: numder_real = 25 ! The same, but real controls

real(kind=ireals), parameter :: missing_value = -999.

integer(kind=iintegers) :: soilcolconjtype

! Switches: 1. PBL parameterization
!  PBLpar =1 (Businger-Dayer formulas (Monin-Obukhov theory) for exchange coefficients)
!  PBLpar =11(Businger-Dayer formulas with shallow water correction after Panin(19..))
!  PBLpar =2 (formulation from NH3d)
!  PBLpar =21(formulation from NH3d with shallow water correction after Panin(19..))
!     2. Relative to water currents wind
!  relwind =0 (relative wind is off)
!  relwind =1 (relative wind is on)
!     3. Turbulent mixing parameterization
!  Turbpar =1 (analytyical profile of turbulent conductivity)
!  Turbpar =2 ("E-eps" parameterization of turbulent conductivity)  
!  Turbpar =3 : Nickuradze (NICK) formulation: Rodi (1993)    
!  Turbpar =4 : Parabolic (PARAB) formulation: Engelund (1976)
!  Turbpar =7 : RNG (re-normalization group) formulation: Simoes (1998)     

type(realpar), target :: dt_out
type(realpar), target :: sensflux0
type(realpar), target :: momflux0
type(realpar), target :: kwe
type(realpar), target :: d_surf
type(realpar), target :: d_bot
type(realpar), target :: depth
type(realpar), target :: thermokarst_meth_prod
type(realpar), target :: soilbotflx
type(realpar), target :: tricemethhydr
type(realpar), target :: deadvol
type(realpar), target :: soil_meth_prod
type(realpar), target :: c_d
type(realpar), target :: nManning
type(realpar), target :: horvisc
type(realpar), target :: zserout
type(realpar), target :: pgrad
type(realpar), target :: backdiff0
type(realpar), target :: VmaxCH4aeroboxid
type(realpar), target :: khsCH4
type(realpar), target :: khsO2
type(realpar), target :: r0methprod
type(realpar), target :: kc0_ud
type(realpar), target :: mubeta0_ud
type(realpar), target :: c1_pmax_ud
  
real(kind=ireals) :: dttribupdate = missing_value

!     The group of tributaries characteristics
type(intpar), target :: N_tribin
integer(kind=iintegers), parameter :: nvartribext = 4 !Driving parameter, not provided in user's files
integer(kind=iintegers), allocatable :: itribloc(:)
type(intpar), target :: N_triblev 
integer(kind=iintegers) :: N_tribout  = 0!; logical :: ok_N_tribout = .false.
type(intpar), target :: iefflloc !; logical :: ok_N_tribout = .false.
type(intpar), target :: tribheat 
real(kind=ireals), allocatable :: &
& U_tribin     (:,:), &
& U_tribout    (:,:), &
& T_tribin     (:,:), &
& Sal_tribin   (:,:), &
& meth_tribin  (:,:), &
& oxyg_tribin  (:,:), &
& DOC_tribin   (:,:), &
& POC_tribin   (:,:), &
& DIC_tribin   (:,:), &
& Ux_tribin    (:,:), &
& Uy_tribin    (:,:), &
& width_tribin (:,:), &
& width_tribout(:,:), &  
& disch_tribin (:,:), &
& disch_tribout(:,:)

! Group of initial profiles
type, public :: initprof
  !sequence
  integer(kind=iintegers) :: lenprof
  real(kind=ireals), allocatable :: zTinitprof(:)
  real(kind=ireals), allocatable :: Tinitprof(:), Sinitprof(:)
  real(kind=ireals), allocatable :: ch4initprof(:), co2initprof(:), o2initprof(:), phosphinitprof(:)
end type initprof
type(initprof) :: ip, isp

! Group of 1D arrays
type, public :: grarr1
  !sequence 
  logical :: ok
  integer(kind=iintegers) :: lenarr, numarr
  real(kind=ireals), allocatable :: arr1(:,:)
end type grarr1
type(grarr1) :: rtemp

! Output grid
real(kind=ireals), allocatable :: zgrid_out(:)
real(kind=ireals), allocatable :: zgridice_out(:)
real(kind=ireals), allocatable :: zgridsoil_out(:)

integer(kind=iintegers) :: nunit = lake_subr_unit_min

type(intpar), target :: runmode
type(intpar), target :: PBLpar
type(intpar), target :: waveenh
type(intpar), target :: momflxpart
type(intpar), target :: relwind
type(intpar), target :: Turbpar
type(intpar), target :: stabfunc
type(intpar), target :: kepsbc
type(intpar), target :: varalb
type(intpar), target :: skin
type(intpar), target :: massflux
type(intpar), target :: ifrad
type(intpar), target :: sedim
type(intpar), target :: salsoil
type(intpar), target :: nstep_keps ! The number of timesteps of k-epsilon parmeterization per on model timestep
type(intpar), target :: nscreen
type(intpar), target :: SoilType
type(intpar), target :: soilswitch
type(intpar), target :: saltice
type(intpar), target :: ifbubble
type(intpar), target :: carbon_model
type(intpar), target :: Tinitlength
type(intpar), target :: Tsoilinitlength
type(intpar), target :: dyn_pgrad
type(intpar), target :: botfric
type(intpar), target :: outflpar
type(intpar), target :: cuette
type(intpar), target :: backdiff
type(intpar), target :: oxicmethprod

type(intpar), target :: monthly_out
type(intpar), target :: daily_out
type(intpar), target :: hourly_out
type(intpar), target :: time_series
type(intpar), target :: time_series_aver
type(intpar), target :: everystep
type(intpar), target :: turb_out
type(intpar), target :: scale_output
type(intpar), target :: ngrid_out
type(intpar), target :: ngridice_out
type(intpar), target :: ngridsoil_out

type(intpar), target :: zero_model
type(intpar), target :: nsoilcols_

type(intpar), target :: omp
type(intpar), target :: eos
type(intpar), target :: lindens
type(intpar), target :: nmeltpoint

type, private :: intpointer
  type(intpar), pointer :: p
end type intpointer
type(intpointer) :: dertypepar(1:numder)

type, private :: realpointer
  type(realpar), pointer :: p
end type realpointer
type(realpointer) :: dertypepar_real(1:numder_real)

integer(kind=iintegers), target :: M ; logical :: ok_M = .false.
integer(kind=iintegers), target :: Mice ; logical :: ok_Mice = .false.
integer(kind=iintegers), target :: ns ; logical :: ok_ns = .false.

integer(kind=iintegers) :: assim ; logical :: ok_assim = .false.
integer(kind=iintegers) :: as_window ; logical :: ok_as_window = .false.
integer(kind=iintegers) :: error_cov ; logical :: ok_error_cov = .false.
integer(kind=iintegers) :: year_accum_begin, year_accum_end
integer(kind=iintegers) :: month_accum_begin, month_accum_end
integer(kind=iintegers) :: day_accum_begin, day_accum_end
integer(kind=iintegers) :: hour_accum_begin, hour_accum_end
logical :: ok_accum_begin, ok_accum_end

logical :: all_par_present = .true.

character(len=40) :: path ; logical :: ok_path = .false.
character(len=60) :: setupfile ! ; logical :: ok_setupfile = .false.
character(len=40) :: fileinflow  = 'filenamenotgiven', &
&                    fileoutflow = 'filenamenotgiven'

SAVE

contains
SUBROUTINE DEFINE_DRIVING_PARAMS()

!     DEFINE_driving_params reads files with driving parameters

implicit none

integer(kind=iintegers) :: i ! Loop index
character(len=200) :: line
logical :: firstcall
data firstcall, line /.true., 'begin file'/


if (firstcall) then

  !Driving parameters, not provided in the user's input files
  soilcolconjtype = 2

  dertypepar_real(1)%p => dt_out
  dertypepar_real(2)%p => sensflux0
  dertypepar_real(3)%p => momflux0
  dertypepar_real(4)%p => kwe
  dertypepar_real(5)%p => d_surf
  dertypepar_real(6)%p => d_bot
  dertypepar_real(7)%p => depth
  dertypepar_real(8)%p => thermokarst_meth_prod
  dertypepar_real(9)%p => soilbotflx
  dertypepar_real(10)%p => tricemethhydr
  dertypepar_real(11)%p => deadvol
  dertypepar_real(12)%p => soil_meth_prod
  dertypepar_real(13)%p => c_d
  dertypepar_real(14)%p => nManning
  dertypepar_real(15)%p => horvisc
  dertypepar_real(16)%p => zserout
  dertypepar_real(17)%p => pgrad
  dertypepar_real(18)%p => backdiff0
  dertypepar_real(19)%p => VmaxCH4aeroboxid
  dertypepar_real(20)%p => khsCH4
  dertypepar_real(21)%p => khsO2
  dertypepar_real(22)%p => r0methprod
  dertypepar_real(23)%p => kc0_ud
  dertypepar_real(24)%p => mubeta0_ud
  dertypepar_real(25)%p => c1_pmax_ud

  dertypepar(1)%p => stabfunc
  dertypepar(2)%p => tribheat
  dertypepar(3)%p => runmode
  dertypepar(4)%p => PBLpar
  dertypepar(5)%p => waveenh
  dertypepar(6)%p => relwind
  dertypepar(7)%p => Turbpar
  dertypepar(8)%p => kepsbc
  dertypepar(9)%p => varalb
  dertypepar(10)%p => skin
  dertypepar(11)%p => massflux
  dertypepar(12)%p => ifrad
  dertypepar(13)%p => sedim
  dertypepar(14)%p => nstep_keps
  dertypepar(15)%p => nscreen
  dertypepar(16)%p => SoilType
  dertypepar(17)%p => soilswitch
  dertypepar(18)%p => ifbubble
  dertypepar(19)%p => dyn_pgrad
  dertypepar(20)%p => monthly_out
  dertypepar(21)%p => daily_out
  dertypepar(22)%p => hourly_out
  dertypepar(23)%p => time_series
  dertypepar(24)%p => time_series_aver
  dertypepar(25)%p => everystep
  dertypepar(26)%p => turb_out
  dertypepar(27)%p => scale_output
  dertypepar(28)%p => zero_model
  dertypepar(29)%p => omp
  dertypepar(30)%p => outflpar
  dertypepar(31)%p => eos
  dertypepar(32)%p => nmeltpoint
  dertypepar(33)%p => momflxpart
  dertypepar(34)%p => nsoilcols_
  dertypepar(35)%p => cuette
  dertypepar(36)%p => iefflloc
  dertypepar(37)%p => salsoil
  dertypepar(38)%p => carbon_model
  dertypepar(39)%p => N_triblev
  dertypepar(40)%p => lindens
  dertypepar(41)%p => saltice
  dertypepar(42)%p => botfric
  dertypepar(43)%p => backdiff
  dertypepar(44)%p => oxicmethprod


  dt_out%name = 'dt_out' ; 
  sensflux0%name = 'sensflux0' ; 
  momflux0%name = 'momflux0' ; 
  kwe%name = 'kwe' ; 
  d_surf%name = 'd_surf' ; 
  d_bot%name = 'd_bot' ; 
  depth%name = 'soil_depth' ; 
  thermokarst_meth_prod%name = 'thermokarst_meth_prod'
  soilbotflx%name = 'soilbotflx'
  tricemethhydr%name = 'tricemethhydr'
  deadvol%name = 'deadvol'
  soil_meth_prod%name = 'soil_meth_prod'
  c_d%name = 'c_d'
  nManning%name = 'nManning'
  horvisc%name = 'horvisc'
  zserout%name = 'zserout'
  pgrad%name = 'pgrad'
  backdiff0%name = 'backdiff0'
  VmaxCH4aeroboxid%name = 'VmaxCH4aeroboxid'
  khsCH4%name = 'khsCH4'
  khsO2%name = 'khsO2'
  r0methprod%name = 'r0methprod'
  kc0_ud%name = 'kc0'
  mubeta0_ud%name = 'mubeta0'
  c1_pmax_ud%name = 'c1_pmax'

  stabfunc%name = 'stabfunc' ; 

  tribheat%name = 'tribheat' ; 
  iefflloc%name = 'iefflloc' ; 
  N_triblev%name = 'N_triblev' ; 

  runmode%name = 'runmode' ; 
  PBLpar%name = 'PBLpar' ; 
  waveenh%name = 'waveenh' ; 
  momflxpart%name = 'momflxpart' ; 
  relwind%name = 'relwind' ; 
  Turbpar%name = 'Turbpar' ; 
  kepsbc%name = 'kepsbc' ; 
  varalb%name = 'varalb' ; 
  skin%name = 'skin' ; 
  massflux%name = 'massflux' ; 
  ifrad%name = 'ifrad' ; 
  sedim%name = 'sedim' ; 
  salsoil%name = 'salsoil' ; 
  nstep_keps%name = 'nstep_keps' ; 
  nscreen%name = 'nscreen' ; 
  SoilType%name = 'soiltype' ; 
  soilswitch%name = 'soilswitch' ; 
  saltice%name = 'saltice' ; 
  ifbubble%name = 'ifbubble' ; 
  dyn_pgrad%name = 'dyn_pgrad' ; 
  botfric%name = 'botfric' ; 
  carbon_model%name = 'carbon_model' ; 
  outflpar%name = 'outflpar' ; 
  cuette%name = 'cuette' ; 
  backdiff%name = 'backdiff' ; 
  oxicmethprod%name = 'oxicmethprod' ;

  monthly_out%name = 'monthly' ; 
  daily_out%name = 'daily' ; 
  hourly_out%name = 'hourly' ; 
  time_series%name = 'time_series' ; 
  time_series_aver%name = 'time_series_aver' ; 
  everystep%name = 'everystep' ; 
  turb_out%name = 'turb_out' ; 
  scale_output%name = 'scale_output' ; 
  zero_model%name = 'zero_model' ; 
  nsoilcols_%name = 'nsoilcols' ;

  omp%name = 'omp' ; 
  eos%name = 'eos';
  lindens%name = 'lindens';
  nmeltpoint%name = 'nmeltpoint';
   
  call CHECK_UNIT(lake_subr_unit_min,lake_subr_unit_max,nunit)
  open  (nunit,file='setup_file.dat',status='old')
  read  (nunit,'(a)') line
  read  (nunit,'(a)') setupfile
  close (nunit)
  open  (nunit,file=setupfile,status='old')
  do while (line /= 'end')
    read (nunit,'(a)') line
    call READPAR(line)
  enddo
  close (nunit)
 
  do i = 1, numder_real
    if (.not. dertypepar_real(i)%p%ok) then
      write(*,*) 'The parameter ', &
      & dertypepar_real(i)%p%name(1:len_trim(dertypepar_real(i)%p%name)), &
      & ' is missing in setup file'
      if (all_par_present) all_par_present = .false.
    endif
  enddo

  do i = 1, numder
    if (.not. dertypepar(i)%p%ok) then
      write(*,*) 'The parameter ', &
      & dertypepar(i)%p%name(1:len_trim(dertypepar(i)%p%name)), &
      & ' is missing in setup file'
      if (all_par_present) all_par_present = .false.
    endif
  enddo

  if (.not.ok_M) then
    write(*,*) 'The parameter M is missing in setup file'
    all_par_present = .false.
  endif  
  if (.not.ok_Mice) then
    write(*,*) 'The parameter Mice is missing in setup file'
    all_par_present = .false.
  endif  
  if (.not.ok_ns) then
    write(*,*) 'The parameter ns is missing in setup file'
    all_par_present = .false.
  endif  
  if (.not.Tinitlength%ok) then
    write(*,*) 'The parameter T_profile is missing in setup file'
    all_par_present = .false.
  endif
  if (.not.Tsoilinitlength%ok) then
    write(*,*) 'The parameter T_soilprofile is missing in setup file'
    all_par_present = .false.
  endif
  if (.not.rtemp%ok) then
    write(*,*) 'The parameter rtemp is missing in setup file'
    all_par_present = .false.
  endif
  if (.not.N_tribin%ok) then
    write(*,*) 'The parameter N_tribin is missing in setup file'
    all_par_present = .false.
  endif
  if (.not.ngrid_out%ok) then
    write(*,*) 'The parameter ngrid_out is missing in setup file.'
    all_par_present = .false.
  endif
  if (.not.ngridice_out%ok) then
    write(*,*) 'The parameter ngridice_out is missing in setup file.'
    all_par_present = .false.
  endif
  if (.not.ngridsoil_out%ok) then
    write(*,*) 'The parameter ngridsoil_out is missing in setup file.'   
    all_par_present = .false.
  endif  
  if (.not.ok_assim) then
    write(*,*) 'The parameter assim is missing in setup file'
    all_par_present = .false.
  endif  
  if (.not.ok_error_cov) then
    write(*,*) 'The parameter error_cov is missing in setup file'
    all_par_present = .false.
  endif  
  if (.not.ok_path) then
    write(*,*) 'The parameter path is missing in setup file'
    all_par_present = .false.
  endif  
  if (.not.ok_accum_begin) then
    write(*,*) 'The parameter accum_begin is missing in setup file'
    all_par_present = .false.
  endif   
  if (.not.ok_accum_end) then
    write(*,*) 'The parameter accum_end is missing in setup file'
    all_par_present = .false.
  endif  
  
  if (.not.all_par_present) then
    write(*,*) 'Not all necessary parameters are present in setup &
    &file: STOP'
    STOP
  endif 
  
  firstcall = .false. 
 
endif

END SUBROUTINE DEFINE_driving_params


SUBROUTINE READPAR(LINE)
implicit none

character(len=200) :: line
character(len=10) :: chardate
integer(kind=iintegers) :: i, j, k
integer(kind=iintegers) :: n2
integer(kind=iintegers) :: n1
integer(kind=iintegers) :: ncol
real(kind=ireals), allocatable :: work(:,:)
logical :: casecheck

SAVE

i = 1
if (line(1:1) == ' ' .or. line(1:1) == char(1)) then
  do while (line(i:i) == ' ' .or. line(i:i) == char(1))
    i = i + 1
  enddo
  n1 = i
  do while (line(i:i) /= ' ' .and. line(i:i) /= char(1))
    i = i + 1
  enddo
  n2 = i - 1
else
  n1 = 1
  do while (line(i:i) /= ' ' .and. line(i:i) /= char(1))
    i = i + 1
  enddo
  n2 = i - 1
endif

lineread: if (line(n1:n1) /= '#') then

  casecheck = .true.
  do j = 1, numder_real
    if (line(n1:n2) == &
    & dertypepar_real(j)%p%name(1:len_trim(dertypepar_real(j)%p%name))) then
      dertypepar_real(j)%p%par = getvarval (n1,n2,line,line(n1:n2))
      dertypepar_real(j)%p%ok = .true.
      casecheck = .false.
      exit
    endif
  enddo
 
  do j = 1, numder
    if (line(n1:n2) == &
    & dertypepar(j)%p%name(1:len_trim(dertypepar(j)%p%name))) then
      dertypepar(j)%p%par = igetvarval (n1,n2,line,line(n1:n2))
      dertypepar(j)%p%ok = .true.
      casecheck = .false.
      exit
    endif
  enddo

  chk : if (casecheck) then

    SELECT CASE (line(n1:n2))

    CASE ('path')
     read (line((n2+1):100),*) path
     !        print*, 'path=', path
     ok_path = .true.

    !     The group of numerical scheme parameters
    CASE ('M')
     M        = igetvarval(n1,n2,line,'Number of layers in water')
     ok_M = .true.
    CASE ('Mice')
     Mice     = igetvarval(n1,n2,line,'Number of layers in ice  ')
     ok_Mice = .true.
    CASE ('ns')
     ns       = igetvarval(n1,n2,line,'Number of layers in soil ')
     ok_ns = .true.

    !     The group of initial conditions
    CASE ('T_profile')
     Tinitlength%par &
     &       = igetvarval (n1,n2,line,'Init. T-profile length   ') 
     ip%lenprof = Tinitlength%par
     allocate (ip%Tinitprof   (1:ip%lenprof) )
     allocate (ip%Sinitprof   (1:ip%lenprof) )
     allocate (ip%ch4initprof (1:ip%lenprof) )
     allocate (ip%co2initprof (1:ip%lenprof) )
     allocate (ip%o2initprof  (1:ip%lenprof) )
     allocate (ip%phosphinitprof  (1:ip%lenprof) )
     allocate (ip%zTinitprof  (1:ip%lenprof) )
     ncol = 7
     allocate (work(Tinitlength%par,ncol))
     call READPROFILE(nunit,Tinitlength%par,ncol,work) 
     ip%zTinitprof (:) = work (:,1)
     ip%Tinitprof  (:) = work (:,2)
     ip%Sinitprof  (:) = work (:,3)
     ip%ch4initprof(:) = work (:,4)
     ip%co2initprof(:) = work (:,5)
     ip%o2initprof (:) = work (:,6)
     ip%phosphinitprof(:) = work (:,7)
     deallocate(work)
     Tinitlength%ok = .true.
       
    !      The group of output controls

    CASE ('T_soilprofile')
     Tsoilinitlength%par &
     &       = igetvarval (n1,n2,line,'Init. T-soilprof length  ') 
     isp%lenprof = Tsoilinitlength%par
     allocate (isp%Tinitprof (1:isp%lenprof) )
     allocate (isp%zTinitprof(1:isp%lenprof) )
     ncol = 2
     allocate (work(Tsoilinitlength%par,ncol))
     call READPROFILE(nunit,Tsoilinitlength%par,ncol,work) 
     isp%zTinitprof (:) = work (:,1)
     isp%Tinitprof  (:) = work (:,2)
     deallocate(work)
     Tsoilinitlength%ok = .true.

    !      The group of output controls
    CASE ('accum_begin')
     j        = igetvarval(n1,n2,line,'Accumulation period begin')
     ok_accum_begin = .true.
     write (chardate,'(i10)') j
     read (chardate,'(i4,3i2)') year_accum_begin, month_accum_begin, &
     & day_accum_begin, hour_accum_begin
    CASE ('accum_end')
     j        = igetvarval(n1,n2,line,'Accumulation period end  ')
     ok_accum_end = .true.
     write (chardate,'(i10)') j
     read (chardate,'(i4,3i2)') year_accum_end, month_accum_end, &
     & day_accum_end, hour_accum_end 
    CASE ('ngrid_out')
     ngrid_out%par &
     &        = igetvarval(n1,n2,line,'Switch for output regrid ')
     allocate (zgrid_out  (1:ngrid_out%par) )
     ngrid_out%ok = .true.
     if (ngrid_out%par > 0) then
       ncol = 1
       allocate (work(1:ngrid_out%par,ncol))
       call READPROFILE(nunit,ngrid_out%par,ncol,work) 
       zgrid_out(:) = work (:,1)
       deallocate(work)
     endif
    CASE ('ngridice_out')
     ngridice_out%par &
     &        = igetvarval(n1,n2,line,'Switch ice output regrid ')
     allocate (zgridice_out  (1:ngridice_out%par) )
     ngridice_out%ok = .true.
     if (ngridice_out%par > 0) then
       ncol = 1
       allocate (work(1:ngridice_out%par,ncol))
       call READPROFILE(nunit,ngridice_out%par,ncol,work) 
       zgridice_out(:) = work (:,1)
       deallocate(work)
     endif
    CASE ('ngridsoil_out')
     ngridsoil_out%par &
     &        = igetvarval(n1,n2,line,'Switch soil out. regrid  ')
     allocate (zgridsoil_out  (1:ngridsoil_out%par) )
     ngridsoil_out%ok = .true.
     if (ngridsoil_out%par > 0) then
       ncol = 1
       allocate (work(1:ngridsoil_out%par,ncol))
       call READPROFILE(nunit,ngridsoil_out%par,ncol,work) 
       zgridsoil_out(:) = work (:,1)
       deallocate(work)
     endif
    CASE ('rtemp')
      rtemp%numarr &
      &       = igetvarval (n1,n2,line,'N. of points for T output')      
      rtemp%lenarr = 3
      allocate (rtemp%arr1(1:rtemp%lenarr,1:rtemp%numarr))
      allocate (work(1:rtemp%numarr,1:rtemp%lenarr))
      call READPROFILE(nunit,rtemp%numarr,rtemp%lenarr,work)
      forall(i=1:rtemp%lenarr,j=1:rtemp%numarr) rtemp%arr1(i,j) = work(j,i) 
      rtemp%ok = .true.
      deallocate (work)

   !     The group of physical properties

   !     The group of data assimilation controls

    CASE ('assim')
     assim    = igetvarval(n1,n2,line,'The assimilation switch  ')
     ok_assim = .true.
    CASE ('as_window')
     as_window= igetvarval(n1,n2,line,'Assimilation window      ')
     ok_as_window = .true.
    CASE ('error_cov')
     error_cov= igetvarval(n1,n2,line,'The switch for err_cov   ')
     ok_error_cov = .true.

!     The group of parameters of inflows and outflows

    !> @deprecated
    !! Keywords inflowprof and outflowprof
    !CASE ('inflowprof')
    !   allocate (width_tribin  (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (U_tribin      (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (T_tribin      (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (Sal_tribin    (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (meth_tribin   (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (DOC_tribin    (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (DIC_tribin    (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (POC_tribin    (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (Ux_tribin     (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (Uy_tribin     (1:N_tribin%par ,1:N_triblev%par) )
    !   allocate (disch_tribin  (1:N_tribin%par) )
    !   ncol = 11
    !   allocate (work(N_triblev%par,ncol))
    !   do k = 1, N_tribin%par
    !     call READPROFILE(nunit,N_triblev%par,ncol,work) 
    !     width_tribin(k,:) = work (:,2)
    !     U_tribin    (k,:) = work (:,3)
    !     T_tribin    (k,:) = work (:,4)
    !     Sal_tribin  (k,:) = work (:,5)
    !     Ux_tribin   (k,:) = work (:,6)
    !     Uy_tribin   (k,:) = work (:,7)
    !     DOC_tribin   (k,:) = work (:,8)
    !     POC_tribin   (k,:) = work (:,9)
    !     DIC_tribin   (k,:) = work (:,10)
    !     meth_tribin   (k,:) = work (:,11)
    !   enddo
    !   print*, 'The profiles in inflow are done'
    !   deallocate(work)
    !       !ok_inflowprof = .true.
    !CASE ('outflowprof')
    !   N_tribout = 1
    !   allocate (width_tribout  (1:N_tribout ,1:N_triblev%par) )
    !   allocate (U_tribout      (1:N_tribout ,1:N_triblev%par) )          
    !   ncol = 3
    !   allocate (work(N_triblev%par,ncol))
    !   call READPROFILE(nunit,N_triblev%par,ncol,work) 
    !      !In current version there is only ONE outflow 
    !   width_tribout(1,:) = work (:,2)
    !   U_tribout    (1,:) = work (:,3)
    !   print*, 'The profiles in ouflow are done'
    !   deallocate(work)
    !       !ok_outflowprof = .true.        
    CASE ('fileinflow')
      read (line((n2+1):200),*) fileinflow
      !allocate (width_tribin  (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (U_tribin      (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (T_tribin      (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (Sal_tribin    (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (meth_tribin   (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (DIC_tribin    (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (DOC_tribin    (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (POC_tribin    (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (Ux_tribin     (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (Uy_tribin     (1:N_tribin%par ,1:N_triblev%par) )
      !allocate (disch_tribin  (1:N_tribin%par) )
      !ok_fileinflow = .true.
    CASE ('fileoutflow')
      read (line((n2+1):200),*) fileoutflow
      N_tribout = 1
      !allocate (width_tribout  (1:N_tribout ,1:N_triblev%par) )
      !allocate (U_tribout      (1:N_tribout ,1:N_triblev%par) )          
      !ok_fileoutflow = .true.
    CASE ('dttribupdate')
      dttribupdate  = getvarval (n1,n2,line,'Tributaries update,  &
      &days')
    CASE ('N_tribin')
      N_tribin%par &
      &       = igetvarval (n1,n2,line,'Number of tributaries    ')      
      if (N_tribin%par > 0) then
        allocate (itribloc(1:N_tribin%par))
        read(nunit,*) itribloc(1:N_tribin%par)
      endif
      N_tribin%ok = .true.
    CASE DEFAULT
      if (.not.line(n1:n2)=='end') then
        print*,'Unknown keyword [',line(n1:n2),'] in setup file: STOP'
        STOP
      endif
    END SELECT

  endif chk

endif lineread

END SUBROUTINE READPAR


SUBROUTINE READPROFILE(iunit,N_rows,N_coloumns,work)
implicit none

!  Input variables:
integer(kind=iintegers), intent(in) :: iunit
integer(kind=iintegers), intent(in) :: N_rows
integer(kind=iintegers), intent(in) :: N_coloumns

!  Output variables:
real(kind=ireals), intent(out) :: work(N_rows, N_coloumns)

!  Local variables
integer(kind=iintegers) :: i, j

do i=1, N_rows
  read(iunit,*) (work(i,j), j=1,N_coloumns)
enddo

END SUBROUTINE READPROFILE


END MODULE DRIVING_PARAMS
