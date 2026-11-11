  MODULE INIT_VAR_MOD

  use LAKE_DATATYPES, only : ireals, iintegers
  use GRID_OPERATIONS_LAKE, only : LININTERPOL

  contains
  SUBROUTINE INIT_VAR &
  ( M, Mice, ns, ms, ml, nsoilcols, ndatamax, &
  & init_T, skin, zero_model, &
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
  
  & E1, eps1, zsoilcols, Tsoil1, wi1, wl1, Sals1, &
  & rootss, qsoil, TgrAnn, qwater, &
  & oxyg, DIC, phosph, DOC, POCL, POCD, &
  & oxygsoil, Sal1, u1, v1, Tw1, Tskin, T_0dim, Eseiches, &
  & Ti1, salice, porice, Tis1, z_full, ddz, &
  & dz, T, wl, dens, lamw, &
  & dzeta_int, zsoil, pressure)
  
  ! Subroutine INIT_VAR initializes the prognostic variables of the model
  
  use PHYS_FUNC, only : &
  & MELTPNT, &
  & UNFRWAT, &
  & WL_MAX, &
  & WI_MAX, &
  & HENRY_CONST, &
  & MELTINGPOINT
  
  use PHYS_CONSTANTS, only : &
  & row0, g, Kelvin0, R_univ, poricebot
  
  use METH_OXYG_CONSTANTS, only : &
  & ch4_atm0, o2_atm0, co2_atm0, &
  & rel_conc_ebul_crit, &
  & Henry_const0_ch4, &
  & Henry_temp_dep_ch4, &
  & Henry_temp_ref, &
  & molmass_p, &
  & r0_methprod_phot, &
  & r0_methprod_nonphot

  use DRIVING_PARAMS, only : &
  & tricemethhydr, &
  & nmeltpoint, &
  & initprof ! derived datatype

  use NUMERIC_PARAMS, only : &
  & small_value
  
  implicit none
  
  real(kind=ireals), save :: E_init
  real(kind=ireals), save :: eps_init
  
  ! Input variables
  integer, intent(in) :: M, Mice, ns, ms, ml, nsoilcols, ndatamax ! Grid dimensions
  integer, intent(in) :: init_T, skin, zero_model ! Driving parameters
  
  real(kind=ireals), intent(in) :: h10, l10, hs10, ls10
  real(kind=ireals), intent(in) :: tempair, Ts0, Tb0, Tm, Tbb0
  real(kind=ireals), intent(in) :: Sals0, Salb0
  real(kind=ireals), intent(in) :: us0, vs0
  real(kind=ireals), intent(in) :: h_ML0
  real(kind=ireals), intent(in) :: rosoil(1:ns), rosdry(1:ns), por(1:ns)
  real(kind=ireals), intent(in) :: depth_area(1:ndatamax,1:2) ! Data for lake bathymetry
  type(initprof), intent(in) :: ip, isp
  real(kind=ireals), intent(in) :: zsoil(1:ns)
  real(kind=ireals), intent(in) :: pressure
  real(kind=ireals), intent(in) :: dzeta_int(1:M+1)
  real(kind=ireals), intent(in) :: ddz(1:M)
  
  ! Output variables
  integer(kind=iintegers), intent(out) :: flag_snow, flag_snow_init, itop
  integer(kind=iintegers), intent(out) :: nstep
  integer(kind=iintegers), intent(inout), allocatable :: itherm(:)
  
  real(kind=ireals), intent(out) :: h1, l1, hs1, ls1
  real(kind=ireals), intent(out) :: hx1, hx2, hy1, hy2
  real(kind=ireals), intent(out) :: hx1t, hx2t, hy1t, hy2t
  real(kind=ireals), intent(inout), allocatable :: hx1ml(:), hx2ml(:), hy1ml(:), hy2ml(:)
  real(kind=ireals), intent(out) :: veg
  real(kind=ireals), intent(out) :: snmelt, snowmass
  real(kind=ireals), intent(out) :: cdmw, velfrict_prev
  real(kind=ireals), intent(out) :: roughness
  real(kind=ireals), intent(out) :: eflux0_kinem, Elatent
  real(kind=ireals), intent(out) :: totalevap, totalmelt, totalprecip, totalwat, totalpen
  real(kind=ireals), intent(out) :: time
  real(kind=ireals), intent(out) :: dhwfsoil
  real(kind=ireals), intent(out) :: dhw, dhw0, dhi, dhi0, dls0  
  
  real(kind=ireals), intent(out) :: E1(1:M+1), eps1(1:M+1)
  real(kind=ireals), intent(out) :: zsoilcols(1:nsoilcols+1)
  real(kind=ireals), intent(out) :: Tsoil1(1:ns,1:nsoilcols), Sals1(1:ns,1:nsoilcols)
  real(kind=ireals), intent(out) :: wi1   (1:ns,1:nsoilcols), wl1  (1:ns,1:nsoilcols)
  real(kind=ireals), intent(out) :: rootss(1:ns), qsoil(1:ns,1:nsoilcols), TgrAnn(1:ns), qwater(1:M+1)
  real(kind=ireals), intent(out) :: oxyg(1:M+1), DIC(1:M+1), phosph(1:M+1)
  real(kind=ireals), intent(out) :: DOC(1:M+1,1:2), POCL(1:M+1), POCD(1:M+1)
  real(kind=ireals), intent(out) :: oxygsoil(1:nsoilcols)
  real(kind=ireals), intent(out) :: Sal1(1:M+1)
  real(kind=ireals), intent(out) :: u1(1:M+1), v1(1:M+1)
  real(kind=ireals), intent(out) :: Tw1(1:M+1), Tskin(1:2)
  real(kind=ireals), intent(out) :: Ti1(1:Mice+1), Tis1(1:Mice+1)
  real(kind=ireals), intent(out) :: salice(1:Mice+1), porice(1:Mice+1)
  real(kind=ireals), intent(out) :: z_full(1:M+1)
  real(kind=ireals), intent(out) :: dz(1:ms), T(1:ml), wl(1:ml), dens(1:ms)
  real(kind=ireals), intent(out) :: lamw(1:M)
  real(kind=ireals), intent(out) :: T_0dim  
  real(kind=ireals), intent(out) :: Eseiches
  
  ! Local variables
  real(kind=ireals), parameter :: Ct_d_Ch = 2. ! Using results by West & Plug, 2007, may be estimated
                                     ! from 1.7 to 2.8 
  real(kind=ireals) :: work, work1
  real(kind=ireals), allocatable :: pressoil(:), workarr(:)
  
  real(kind=ireals) :: z, h_ML0zv, h_talik, ch4_bot
  real(kind=ireals) :: Ti10, Ti11
  integer(kind=iintegers) :: i, j, i_ML, i_talik, k
  integer(kind=iintegers) :: n_1cm, n_5cm

  logical :: flag
  logical, save :: firstcall = .true.

  logical, parameter :: soil_thermokarst_init = .false.
  real(kind=ireals), parameter :: relch4_0 = 0.4
  real(kind=ireals), parameter :: H_CH4zero = 1. !Depth in sediments where CH4 vanishes, m
  
  if (firstcall) then
    E_init = 10.d0**(-5.5)
    eps_init = 1.d-9
  endif

  h1 = h10

  hx1 = 0.; hx2 = 0.
  hy1 = 0.; hy2 = 0.
  hx1t = 0.; hx2t = 0.
  hy1t = 0.; hy2t = 0.

  if (allocated(hx1ml)) then
    hx1ml(1:M+1) = 0.; hx2ml(1:M+1) = 0.
    hy1ml(1:M+1) = 0.; hy2ml(1:M+1) = 0.
  endif

  if (allocated(itherm)) itherm(1:M+2) = -1

  l1 = l10
  hs1 = hs10
  ls1 = ls10
  do i = 1, M 
    E1(i) = E_init !+ float(i)/float(M)*(1.d-10  -  1.d-8)
    eps1(i) = eps_init !+ float(i)/float(M)*(1.d-18  -  1.d-14)
  enddo

  do i = 1, M+1
    z_full  (i) = dzeta_int(i)*h1
  enddo
  i_ML = int((M+1)*h_ML0/h1) ! The index of level of mixed-layer depth
  ! Water temperature initialization
  if (init_T == 2) then
    h_ML0zv = (Tm - 0.5*(Tb0 + Ts0))/(Ts0/h1 - 0.5*(Ts0 + Tb0)/h1)
    i_ML = int(M*h_ML0zv/h1)  
  endif 
  if (init_T == 1 .or. init_T == 2) then
    Tw1(1:max(i_ML,1)) = Ts0
    do i = max(i_ML+1,2), M+1 
      Tw1(i) = Ts0 + (Tb0-Ts0)*float(i-i_ML)/float(M+1-i_ML)
    enddo
    ! The initial temperature profile is given from the input file
  elseif (init_T == 3) then
    call LININTERPOL (ip%zTinitprof,ip%Tinitprof,ip%lenprof, &
    & z_full,Tw1,M+1,flag,.true.)
    call LININTERPOL (ip%zTinitprof,ip%Sinitprof,ip%lenprof, &
    & z_full,Sal1,M+1,flag,.true.)
    if (.not.flag) then
      print*, 'The error while interpolating the initial &
      &temperature profile: terminating program'
      STOP
    endif
  endif
 
  ! Lake depth intervals for the tops of soil columns.
  ! An alternative - to calculate from the vertical grid of depth-area data.
  if (nsoilcols > 1 .and. depth_area(1,2) >= 0.) then
    !work = h10/real(nsoilcols)
    work1 = maxval(depth_area(:,1))
    work = ( work1 - h10*0.5*ddz(M) )/real(nsoilcols-1)
    zsoilcols(nsoilcols+1) = work1
    do i = 1, nsoilcols
      zsoilcols(i) = (i-1)*work
    enddo
  endif


  Sals1(1:ns,1:nsoilcols) = Salb0*row0/( rosdry(1)*(1 - por(1)) ) ! assuming, that salinity in ground 
                                                                  ! is the same 
                                                                  ! as one in near bottom layer of water
  
  allocate(pressoil(1:ns))
  do i = 1, ns
    pressoil(i) = pressure + row0*g*(h10 + zsoil(i))
  enddo

! Temperature distribution in talik assuming homogeneous distribution of salinity in ground 
  flag = (h10 > 0. .and. Tb0 > MELTINGPOINT(Sals1(1,nsoilcols),pressoil(1),tricemethhydr%par) &
  & .and. Tbb0 < MELTINGPOINT(Sals1(ns,nsoilcols),pressoil(ns),tricemethhydr%par))
  if (flag .and. soil_thermokarst_init) then
  
!   Initializing the talik depth, assuming thermokarst lake, following West and Plug, 2007
    h_talik = min(Ct_d_Ch*h10,0.8*zsoil(ns))
    
    do i = 1, ns
      if (zsoil(i) >= h_talik) then
        i_talik = i ! i_talik is the number of the first level below the estimated talik depth
        exit
      endif
    enddo
    
!   Temperature distribution in the talik
    work = MELTINGPOINT(Sals1(i_talik,nsoilcols),pressoil(i_talik),tricemethhydr%par)
    do i = 1, i_talik
      Tsoil1(i,nsoilcols) = Tb0 + float(i-1)/float(i_talik-1) * (work - Tb0)
    enddo

!   Temperature distribution below the talik
    do i = i_talik+1, ns
      Tsoil1(i,nsoilcols) = work + float(i-i_talik)/float(ns-i_talik)*(Tbb0 - work)    
    enddo

  else
  
    h_talik = 0.

    ! Linear temperature profile in soil
    if (init_T == 1 .or. init_T == 2) then
      do i = 1, ns
        Tsoil1(i,nsoilcols) = Tb0 + float(i-1)/float(ns-1)*(Tbb0-Tb0)
      enddo 
    ! The initial soil temperature profile is given from the input file
    elseif (init_T == 3) then
      call LININTERPOL (isp%zTinitprof,isp%Tinitprof,isp%lenprof, &
      & zsoil,Tsoil1(1,nsoilcols),ns,flag,.true.)
      if (.not.flag) then
        print*, 'The error while interpolating the initial &
        &soil temperature profile: terminating program'
        STOP
      endif
    endif
    
  endif
  
  ! Ice and liquid water content initialization in soil
  do i = 1, ns
    if (Tsoil1(i,nsoilcols) > MELTINGPOINT(Sals1(i,nsoilcols),pressoil(i),tricemethhydr%par)) then
      wi1(i,nsoilcols) = 0.
      wl1(i,nsoilcols) = WL_MAX(por(i),rosoil(i),0.e0_ireals,tricemethhydr%par) - 0.01
    else
      wl1(i,nsoilcols) = UNFRWAT(Tsoil1(i,nsoilcols),i)
      wi1(i,nsoilcols) = WI_MAX(por(i),rosoil(i),tricemethhydr%par) - wl1(i,nsoilcols) - 0.01
    endif
  enddo

  ! Identical initialization of all soil columns
  if (nsoilcols > 1 .and. depth_area(1,2) >= 0.) then
    do i = 1, ns
      wl1   (i,1:nsoilcols-1) = wl1   (i,nsoilcols)
      wi1   (i,1:nsoilcols-1) = wi1   (i,nsoilcols)  
    enddo
    !Temperature in side soil columns is initialized as a linear function from
    !a water temperature at respective depth to Tbb0
    allocate(workarr(1:M+1))
    do j = 1, nsoilcols-1
      work = 0.5*(zsoilcols(j) + zsoilcols(j+1))
      workarr(1:M+1) = abs(z_full(1:M+1) - work)
      k = minloc(workarr,1)
      do i = 1, ns
        !Tsoil1(i,j) = Tw1(k) + float(i-1)/float(ns-1)*(Tbb0 - Tw1(k))
        Tsoil1(i,j) = Tw1(k) + (Tbb0 - Tw1(k))*(1. - exp( - 10.*float(i-1)/float(ns-1) ) )
      enddo
    enddo
    deallocate(workarr)
  endif

  
! Methane variables initialization  
  rootss(1:ns) = 0.e0_ireals ! No roots
  veg = 0.e0_ireals ! No vegetation at the bottom

  TgrAnn(1:ns) = Tsoil1(1:ns,nsoilcols)
  
!  qsoil(1) = ch4_bot
!  qsoil2(1,1:2) = 0.5*ch4_atm0 ! two-meth

! Initialization of methane in soil columns at the lake's slopes
  if (nsoilcols > 1 .and. depth_area(1,2) >= 0.) then
    do j = 1, nsoilcols-1
      call CINIT_CALC(ns,Tsoil1(1,j),zsoil,H_CH4zero,r0_methprod_phot(j),qsoil(1,j))
      do i = 1, ns
        if (zsoil(i) <= H_CH4zero) then
          qsoil(i,j) = min(relch4_0*rel_conc_ebul_crit*por(i) * &
          & (pressure + row0*g*(0.5*(zsoilcols(j) + zsoilcols(j+1)) + zsoil(i))) * & ! assuming hydrostatic pressure in soil
          & HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, &
          & Henry_temp_ref, Tsoil1(i,j) + Kelvin0),qsoil(i,j))
        else
          qsoil(i,j) = 0.
        endif
      enddo
    enddo
  endif

! Initialization of methane in deepest soil column
  j = nsoilcols
  call CINIT_CALC(ns,Tsoil1(1,j),zsoil,H_CH4zero,r0_methprod_phot(j),qsoil(1,j))
  do i = 1, ns
    if (zsoil(i) <= H_CH4zero) then
      qsoil(i,nsoilcols) = min(relch4_0*rel_conc_ebul_crit*por(i) * &
      & (pressure + row0*g*(h10 + zsoil(i))) * & ! assuming hydrostatic pressure in soil
      & HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, &
      & Henry_temp_ref, Tsoil1(i,nsoilcols) + Kelvin0),qsoil(i,nsoilcols))
    else
      qsoil(i,nsoilcols) = 0.
    endif
  enddo

  ch4_bot = qsoil(1,nsoilcols)/por(1)   ! Boundary condition
  if (init_T /= 3) then
    do i = 1, M+1
      if (i <= i_ML) then
        qwater(i) = ch4_atm0 ! Atmospheric concentration
      else
        qwater(i) = qwater(max(i_ML,1)) + &
        & (ch4_bot - qwater(max(i_ML,1)))*(h_ML0 - dzeta_int(i)*h10)/(h_ML0 - h10)
      endif
    enddo
  else
    call LININTERPOL(ip%zTinitprof,ip%ch4initprof,ip%lenprof, &
    & z_full,qwater,M+1,flag,.true.)
  endif

! Dissolved inorganic carbon initialization
  if (init_T /= 3) then
    DIC(1:M+1) = 1.*co2_atm0 ! Atmospheric concentration
  else
    call LININTERPOL (ip%zTinitprof,ip%co2initprof,ip%lenprof, &
    & z_full,DIC,M+1,flag,.true.)
  endif

! Inorganic dissolved phosphorus
  if (init_T /= 3) then
    phosph(:) = 100./molmass_p !0.01
  else
    call LININTERPOL (ip%zTinitprof,ip%phosphinitprof,ip%lenprof, &
    & z_full,phosph,M+1,flag,.true.)
  endif

! Oxygen initialization
  if (init_T /= 3) then
    do i = 1, M+1
      if (i <= i_ML) then
        oxyg(i) = o2_atm0 ! Atmospheric concentration
      else
        oxyg(i) = oxyg(max(i_ML,1)) - oxyg(max(i_ML,1))*(h_ML0 - dzeta_int(i)*h10)/(h_ML0 - h10)
      endif
      oxyg(i) = max(oxyg(i),small_value)
    enddo
  else
    call LININTERPOL (ip%zTinitprof,ip%o2initprof,ip%lenprof, &
    & z_full,oxyg,M+1,flag,.true.)
  endif
  oxygsoil(1:nsoilcols) = 0. !Zero initial oxygen concentration in the aerobic soil layer

  DOC (1:M+1,1:2) = 0.    !dissolved organic carbon
  POCL(1:M+1)     = 1.E-1 !particulate organic carbon living
  POCD(1:M+1)     = 0.    !particulate organic carbon dead

  u1(1:max(i_ML,1)) = us0
  v1(1:max(i_ML,1)) = vs0
  do i = max(i_ML+1,2), M+1
    u1(i) = us0 + (0.e0_ireals - us0)*float(i-i_ML)/float(M+1-i_ML)
    v1(i) = vs0 + (0.e0_ireals - vs0)*float(i-i_ML)/float(M+1-i_ML)
  enddo
 
  if (init_T /= 3) then
    Sal1(1:max(i_ML,1)) = Sals0
    do i = max(i_ML+1,2), M+1
      Sal1(i) = Sals0 + (Salb0-Sals0)*float(i-i_ML)/float(M+1-i_ML)
    enddo
  endif
  
  if (skin == 0) then
    Tskin(1:2) = 0.e0_ireals
  else
    Tskin(1) = Tw1(1)
  endif  
  
! Initial temperature for zero-dimensional model 
  if (zero_model == 1) then
    T_0dim = Tw1(1)*0.5*ddz(1) + Tw1(M+1)*0.5*ddz(M)
    do i = 2, M
      T_0dim = T_0dim + Tw1(i)*0.5*(ddz(i-1) + ddz(i) )
    enddo
  else
    T_0dim = 0.
  endif
 
  Ti1 = 0.
  salice(:) = 0.
  porice(:) = poricebot
  Tis1 = 0. 
  Ti10 = min(tempair,-1.d-1) ! Initial ice surface temperature, Celsius
  if (l1 /= 0) then
    if (h1 > 0.) then
!     Ice over water 
      Ti11 = MELTPNT(Sals0,0.e0_ireals,nmeltpoint%par)
    else
!     Ice over soil
      Ti11 = Tb0
    endif
    do i = 1, Mice+1
      Ti1(i)= Ti10 + (Ti11 - Ti10)*float(i-1)/float(Mice) ! Linear profile, if dzetai-grid is regular,
    enddo                                                 ! water layer underneath is assumed to exist
  endif
  
  if (hs1 == 0.) then   
    flag_snow = 0
    flag_snow_init = 1
    itop = 1
  else
    flag_snow = 1
    flag_snow_init = 0
    hs1 = max(2.d-2, hs1)
    hs1 = int(hs1/0.01)*0.01
    n_5cm = int(hs1/0.05)
    n_1cm = int((hs1-n_5cm*0.05)/0.01)
    if (n_1cm == 0) then
      n_1cm = 1 
      hs1 = hs1 + 0.01
    endif
    itop = ms - (n_1cm + n_5cm)
    do i = itop, itop + n_1cm - 1
      dz(i) = 0.01
      T(i) = min(tempair,-5.d0)
      wl(i) = 0
      dens(i) = 150.
    enddo
    do i = itop + n_1cm, ms-1
      dz(i) = 0.05
      T(i) = min(tempair,-5.d0)
      wl(i) = 0
      dens(i) = 150.
    enddo
    dz(ms) = 0.5d0*dz(ms-1)
    T(ms) = min(tempair,-5.d0)
    wl(ms) = 0.e0_ireals
    dens(ms) = 150.d0
  endif
 
  snmelt = 0.e0_ireals 
  snowmass = 0.e0_ireals 
  cdmw = 1.d-15
  velfrict_prev = 1.d-2 
  roughness = 1.d-3
  eflux0_kinem = 0.e0_ireals
  Elatent = 0.

  totalevap = 0 
  totalmelt = 0 
  totalprecip = 0. 
  totalwat = 0
  totalpen = 0
  time = 0
  nstep = 0
  dhwfsoil = 0.
  dhw = 0.
  dhw0 = 0.
  dhi = 0.
  dhi0 = 0.
  dls0 = 0.

  lamw(:) = 1.d+3
  
  Eseiches = 0. !initial seiche energy

  deallocate(pressoil)

  if (firstcall) firstcall = .false.
  END SUBROUTINE INIT_VAR

  !> Subroutine CINIT_CALC computes initial profile of methane concentration in
  !! sediments following analytical solution of stationary equation
  SUBROUTINE CINIT_CALC(n,Tsoil,z,H,r0_meth,Cinit)

  use METH_OXYG_CONSTANTS, only : &
  & forg0, lambda_new_org, tortuosity_coef, q100
  use PHYS_FUNC, only : DIFF_WATER_METHANE

  implicit none
  !Input/output variables
  real   (kind=ireals   ), intent(in ) :: z(n)     !< Soil level depths, m
  real   (kind=ireals   ), intent(in ) :: Tsoil(n) !< Soil temperature, Celsius
  real   (kind=ireals   ), intent(in ) :: H        !< Depth, where CH4 concentration and flux vanish, m
  real   (kind=ireals   ), intent(in ) :: r0_meth  !< The constant of methane production in sediments (soil)
  integer(kind=iintegers), intent(in ) :: n        !< Number of levels in soil
  real   (kind=ireals   ), intent(out) :: Cinit(n) !< CH4 concentration in soil, mol/m**3
  !Local variables
  integer(kind=iintegers) :: i
  real   (kind=ireals   ) :: ft, k, P0, Tsoil_mean

  Tsoil_mean = sum(Tsoil)/real(n) !%todo: replace by proper averaging
  k = tortuosity_coef*DIFF_WATER_METHANE(Tsoil_mean)
  ft = 0.5*(sign(1._ireals,Tsoil_mean) + 1.) 
  p0 = forg0*r0_meth*q100*(Tsoil_mean/10.)*ft

  do i = 1, n
    Cinit(i) = (p0/((lambda_new_org**2)*k))*exp(-lambda_new_org*z(i)) * &
    &          (exp(-lambda_new_org*(H-z(i)))*(1+lambda_new_org*(H-z(i)))-1)
  enddo     
  END SUBROUTINE CINIT_CALC

  END MODULE INIT_VAR_MOD
