MODULE OXYGEN_MOD

use NUMERICS, only : PROGONKA, STEP
use NUMERIC_PARAMS, only : nveclen, small_value
use LAKE_DATATYPES, only : ireals, iintegers
use ARRAYS_GRID, only : gridsize_type
use ARRAYS_BATHYM, only : layers_type, bathym 
use ARRAYS_GRID, only : gridspacing_type
use ARRAYS_SOIL, only : lsh_type
use METH_OXYG_CONSTANTS
use DRIVING_PARAMS, only : carbon_model
use UNITS, only : kg_to_g
use DIFFUSION_LAKE_MOD, only : DIFFUSION

real(4), parameter :: k_PA = 1.7 ! n/d, coefficient in Poole&Atkins formula
real(4), parameter :: extwat1 = k_PA/2., extwat2 = k_PA/3.5 ! Extinction coefficients
                                                          ! identifying eutrophic,
                                                          ! mesotrophic and oligotrophic lakes,
                                                          ! derived from Poole and Atkins (1929) 
                                                          ! formula and data from Stefan and Fang, 1994
real(4), parameter :: chla_aver_oligo = 2.E-3  !mg/l, Mean chlorophyll density in oligotrophic lake, Stefan and Fang, 1994
real(4), parameter :: chla_aver_meso  = 6.E-3  !mg/l, Mean chlorophyll density in mesotrophic lake, Stefan and Fang, 1994
real(4), parameter :: chla_aver_eu    = 15.E-3 !mg/l, Mean chlorophyll density in eutrophic lake, Stefan and Fang, 1994

contains
!> Subroutine OXYGEN calculates vertical diffusion of dissolved oxygen and interaction with bubbles
SUBROUTINE OXYGEN &
& (gs, Twater, fbbleflx_o2_sum, fbbleflx_o2, lamoxyg, gsp, fracb0, pressure, wind10, &
& o2_pres_atm, ls, bathymwater, lso, dt, febul0, eps_surf, sodbot, oxyg, soilswitch, Flux_atm)

use PHYS_FUNC, only : &
& HENRY_CONST, &
& GAS_WATATM_FLUX
use PHYS_CONSTANTS, only : kappa, z0_bot

use PHYS_CONSTANTS, only : &
& Kelvin0

use T_SOLVER_MOD, only : &
& DIFF_COEF

use ATMOS, only : velfrict_bot
use LB_SAVED_LOCALS, only : Foxyg1 => oxygen_Foxyg1, botO2 => oxygen_botO2 ! lakebatch restart

implicit none

!> Grid size group
type(gridsize_type),    intent(in) :: gs
!> Grid spacing group
type(gridspacing_type), intent(in) :: gsp
!> Group containing depths of physical layers
type(layers_type),      intent(in) :: ls
!> Bathymetry characteristics of a water body
type(bathym),           intent(in) :: bathymwater(1:gs%M+1)
!> Oxygen flux at the water-sediments interface
type(lsh_type),         intent(in) :: lso
!> Water temperature profile, Celsius
real(kind=ireals), intent(in) :: Twater(1:gs%M+1) 
!> The bubble oxygen flux, horizontally integrated, normalized by bottom value, n/d
real(kind=ireals), intent(inout) :: fbbleflx_o2_sum(0:gs%M+1)
!> The bubble oxygen flux above the deepest soil column, normalized by bottom value, n/d
real(kind=ireals), intent(in) :: fbbleflx_o2(0:gs%M+1) 
!> Vertical diffusivity for oxygen, m**2/s
real(kind=ireals), intent(in) :: lamoxyg(1:gs%M)
!> Molar fractions of gases in a bubble
real(kind=ireals), intent(in) :: fracb0(1:ngasb) 
!> Bottom methane flux, mol/m**2/s
real(kind=ireals), intent(in) :: febul0
!> Atmospheric pressure, Pa
real(kind=ireals), intent(in) :: pressure 
!> Wind speed at 10 m above the surface
real(kind=ireals), intent(in) :: wind10 
!> Oxygen partial pressure in the atmosphere, Pa
real(kind=ireals), intent(in) :: o2_pres_atm 
!> Time step, s
real(kind=ireals), intent(in) :: dt
!> TKE dissipation rate at the water surface, m**2/s**3
real(kind=ireals), intent(in) :: eps_surf 
!> Sedimentary oxygen demand (oxygen flux at the bottom), mol/(m**2*s)
real(kind=ireals), intent(in) :: sodbot 
!> Oxygen concentration in water, mol/m**3
real(kind=ireals), intent(inout) :: oxyg(1:gs%M+1,1:2)
!> Switch for soil (sediments)
integer(kind=iintegers), intent(in) :: soilswitch
!> Flux to the atmosphere, mol/m**2/s
real(kind=ireals), intent(out) :: Flux_atm

! Local variables
integer(kind=iintegers), parameter :: water_oxygen_indic = 9

real(kind=ireals), allocatable :: a(:), b(:), c(:), f(:), y(:)

! Foxyg1 (bottom oxygen flux) and botO2 now live in LB_SAVED_LOCALS (lakebatch restart)

real(kind=ireals) :: x, xx
real(kind=ireals) :: bcs(1:2)

integer(kind=iintegers) :: i
integer(kind=iintegers), parameter :: switchbotbc = 1 !1 - Neumann, 2 - Dirichlet
integer(kind=iintegers) :: bctype(1:2)

allocate (a(1:nveclen),b(1:nveclen),c(1:nveclen), &
&         f(1:nveclen),y(1:nveclen))
a(:) = 0.; b(:) = 0.; c(:) = 0.; f(:) = 0.; y(:) = 0.

Foxyg1 = sodbot
!Foxyg1 = kappa*oxyg(gs%M+1,1)*velfrict_bot/log(0.25*gsp%ddz(gs%M)*ls%h1/z0_bot) !note: explicit scheme for flux

if (ls%h1 > 0) then
! 1-st step of splitting-up scheme - diffusion, explicit scheme for flux
  if (ls%l1 == 0) then
    x = HENRY_CONST(Henry_const0_o2, Henry_temp_dep_o2, Henry_temp_ref, &
    & Twater(1) + Kelvin0) 
    Flux_atm = - GAS_WATATM_FLUX &
    & (Twater(1),wind10,oxyg(1,1),o2_pres_atm,x,water_oxygen_indic,eps_surf)
  else
    Flux_atm = 0.
  endif
  bctype(1) = 2
  bcs   (1) = Flux_atm
  if (ls%ls1 > 0) then  
!    ! Switching bottom b.c.
    select case (switchbotbc)
    case(1)
      bctype(2) = 2
      bcs   (2) = 0.
    case(2)
      bctype(2) = 1
      bcs   (2) = botO2
    end select
  else  
    ! Switching bottom b.c.
    select case (switchbotbc)
    case(1)
      bctype(2) = 2
      bcs   (2) = Foxyg1
    case(2)
      bctype(2) = 1
      bcs   (2) = botO2
    end select
  endif

  call DIFFUSION(gs%M+1,bctype,bcs,dt,ls%h1,ls%dhw,ls%dhw0, &
  & gsp%ddz,gsp%ddz05,gsp%dzeta_int,oxyg(1,1),oxyg(1,2),lamoxyg,&
  & lso%water,bathymwater)

  if (gs%nsoilcols == 1) then
    x = febul0*fracb0(3)/fracb0(1)
    fbbleflx_o2_sum(0)      = x*fbbleflx_o2(0)     *bathymwater(1)     %area_int
    fbbleflx_o2_sum(1:gs%M) = x*fbbleflx_o2(1:gs%M)*bathymwater(1:gs%M)%area_half
    fbbleflx_o2_sum(gs%M+1) = x*fbbleflx_o2(gs%M+1)*bathymwater(gs%M+1)%area_int
  endif

  if (soilswitch == 1) then
    ! 2-d step of splitting-up scheme - source due to bubble dissolution
    !x = dt*febul0*fracb0(3)/fracb0(1)
    oxyg(1,2) = oxyg(1,2) + 2.d0*dt/(ls%h1*gsp%ddz(1))*(fbbleflx_o2_sum(1) - fbbleflx_o2_sum(0)) / &
    & bathymwater(1)%area_int
    oxyg(1,2) = max(oxyg(1,2),small_value)
    do i = 2, gs%M
      oxyg(i,2) = oxyg(i,2) + dt/(ls%h1*gsp%ddz05(i-1))*(fbbleflx_o2_sum(i) - fbbleflx_o2_sum(i-1)) / &
      & bathymwater(i)%area_int
      oxyg(i,2) = max(oxyg(i,2),small_value)
    enddo
    oxyg(gs%M+1,2) = oxyg(gs%M+1,2) + 2.d0*dt/(ls%h1*gsp%ddz(gs%M)) * &
    & (fbbleflx_o2_sum(gs%M+1) - fbbleflx_o2_sum(gs%M)) / &
    & bathymwater(gs%M+1)%area_int
    oxyg(gs%M+1,2) = max(oxyg(gs%M+1,2),small_value)
  endif

endif 


deallocate (a, b, c, f, y)

END SUBROUTINE OXYGEN 


!> Calculates the rate of oxygen sources and sinks in the water column,
!! generally following Stefan and Fang, Ecological Modelling, 71 (1994) 37-68
!! V.S., 06.2014
SUBROUTINE OXYGEN_PRODCONS(gs,gsp,wst,bathymsoil,gas,area_int,area_half,ddz05, &
&                          h,dt,Tsoil,Chl_a,dzs,por,oxygsoil, &
&                          itroph,prodox,resp,bod,sod,sodbot)

use DRIVING_PARAMS, only : soilcolconjtype
use PHYS_CONSTANTS, only : &
& Wm22Em2, short2PAR, z0_bot

use TIMEVAR, only : &
& hour_sec, day_sec

use METH_OXYG_CONSTANTS, only : &
& molmass_o2, &
& k_O2_SOD, mubeta0, kc0, &
& thetaC_SOD, thetamu_SOD, &
& tortuosity_coef, &

& T0, &   ! Reference temperature, Celsius
& T00, &  ! Another reference temperature, Celsius
& c1_pmax, c2_pmax, &! Constants for temperature dependence,
                     ! (Stefan and Fang, 1994; Megard et al., 1984)
& k1_c1, k1_c2, &    ! (Megard et al., 1984)
& k2_c1, k2_c2, &
& YCHO2, &   ! mass ratio of chlorophyll-a to oxygen utilized in respiration
& k_r, &     ! day**(-1), (Brown and Barnwell, QUAL2E, 1987; Riley, 1988)
& theta_r, & ! temperature dependence of respiration (Ambrose et al., 1988)
& k_b, &     ! day**(-1), 1-st order decay coefficient, (Riley, 1988; Brown and Barnwell, 1987)
& theta_b1, theta_b2, &   ! temperature-dependence coefficient for biochemical oxygen demand
& mO2C_dec, mCChla_dec, & ! Stoichiometry constants for organics decay in the water column, 
                          ! for BOD, basing on carbonaceous BOD and neglecting 
                          ! nitropgenous BOD, see details in Stefan and Fang, 1994
& Sb20, & ! estimates: 0.5E+3 for oligotrophic, 1.5E+3 for eutrophic lakes, Stefan and Fang, 1994,
          ! mg/(m**2*day), reference sedimentary oxygen demand (SOD)
& theta_s1, theta_s2, &! temperature dependence coefficients for SOD, Stefan and Fang, 1994
& T_md, &              ! the temperature of maximal density

& alpha_POCL, tau_DOC_auto, tau_DOC_allo, tau_POCD, & !constants of Hanson et al. 2004 model

& halfsat_DIP_photosynt !half-saturation constant for soluble reactive phosphorus
                        !(~dissolved inorganic phosphorus) limitation in photosynthesis rate

use ARRAYS_GRID, only : &
& gridsize_type, gridspacing_type

use ARRAYS_WATERSTATE, only : waterstate_type
use RADIATION, only : RadWater
use ARRAYS_BATHYM, only : bathym
use ARRAYS, only : gas_type

use PHYS_FUNC, only : DIFF_WATER_OXYGEN, LOGFLUX

use RADIATION, only : nbands

implicit none

! Input variables
!> Grid sizes
type(gridsize_type), intent(in) :: gs
!> Grid spacings
type(gridspacing_type), intent(in) :: gsp
!> Water state variables
type(waterstate_type), intent(in) :: wst
!> Bathymetry of soil columns
type(bathym), intent(in) :: bathymsoil(1:gs%nsoilcols+1,1:gs%nx,1:gs%ny)
!> Carbon species in water and oxygen
type(gas_type), intent(in) :: gas

!> Lake cross-section area at cell interfaces
real(kind=ireals),    intent(in) :: area_int(1:gs%M+1) 
!> Lake cross-section area at cell centers
real(kind=ireals),    intent(in) :: area_half(1:gs%M)  
!> Spacing between cell centers, n/d
real(kind=ireals),    intent(in) :: ddz05(0:gs%M) 
!> Soil temperature, deg. Celsius
real(kind=ireals),    intent(in) :: Tsoil(1:gs%ns,1:gs%nsoilcols)  
!> Chlorophyll-a concentration, mg * l**(-1)
real(kind=ireals),    intent(in) :: Chl_a(1:gs%M+1) 
!> Soil grid spacing, m
real(kind=ireals),    intent(in) :: dzs  (1:gs%ns)  
!> Soil porosity, n/d
real(kind=ireals),    intent(in) :: por  (1:gs%ns)  
!> Mean oxygen concentration in top soil layer
real(kind=ireals),    intent(inout) :: oxygsoil (1:gs%nsoilcols)  
!> Lake depth, m
real(kind=ireals),    intent(in) :: h            
!> Timestep, s
real(kind=ireals),    intent(in) :: dt           
!> Trophic status (1 - oligotrophic, 2 - mesotrophic, 3 - eutrophic)
integer(kind=iintegers), intent(in) :: itroph 

! Output variables
!> Production of O2 by photosynthesis
real(kind=ireals), intent(out) :: prodox(1:gs%M+1) 
!> Sink due to respiration
real(kind=ireals), intent(out) :: resp(1:gs%M+1)   
!> Biochemical oxygen demand
real(kind=ireals), intent(out) :: bod(1:gs%M+1)    
!> Sediment oxygen demand (sink)
real(kind=ireals), intent(out) :: sod(1:gs%M+1)    
!> Sediment oxygen demand at the bottom, positive downwards
real(kind=ireals), intent(out) :: sodbot        

! Local variables

integer(kind=iintegers), parameter :: nSOD = 2 ! The switch for SOD models
integer(kind=iintegers), parameter :: nresp = 2 ! The switch for respiration model

real(4) :: Pmax  ! Maximum oxygen production by photosynthesis, h**(-1)
real(4) :: minL  ! Light limiting factor for photosynthesis
real(4) :: minN  ! Nutrient limiting factor for photosynthesis
real(4) :: PAR   ! Photosynthetically active radiation, Einstein /(m*m*h)
real(4) :: BOD_  ! detritus as oxygen equivalent, mg/l


real(4) :: k1_minL, k2_minL, kc, mubeta, oxysurf
real(kind=ireals) :: x, step10, step20, chla_aver, flux, diff
real(kind=ireals), allocatable :: PARz(:)

integer(kind=iintegers) :: i, j, k
integer(kind=iintegers), allocatable :: kcompl(:)

logical :: firstcall = .true.


if (nSOD == 3) then
  allocate (kcompl(1:gs%nsoilcols))
  kcompl(:) = 0
endif

! Calculating mean chlorophyll-a density
if     (itroph == 3) then 
! Eutrophic lake
  chla_aver = chla_aver_eu
elseif (itroph == 2) then
! Mesotrophic lake
  chla_aver = chla_aver_meso
elseif (itroph == 1) then
! Oligotrophic lake
  chla_aver = chla_aver_oligo
endif

allocate(PARz(0:gs%M+1))
if (nbands > 1) then
  PARz(0:gs%M+1) = RadWater%flux(0:gs%M+1,2) !2-d band assumed to be PAR
else
  PARz(0:gs%M+1) = RadWater%flux(0:gs%M+1,1)
endif

do i = 1, gs%M
  ! Oxygen production by photosynthesis
  Pmax = c1_pmax*c2_pmax**(wst%Tw2(i) - T0)
  k1_minL = k1_c1*k1_c2**(wst%Tw2(i) - T0)
  step10 = STEP(wst%Tw2(i) - T00) ! step function around 10 Celsius
  k2_minL = step10*k2_c2 + (1 - step10)*k2_c1
  if (i > 1) then
    PAR = 0.5*(PARz(i-1) + PARz(i))*Wm22Em2*hour_sec !*short2PAR !converting to Einstein /(m**2*hour)
  else
    PAR = PARz(0)*Wm22Em2*hour_sec !*short2PAR !converting to Einstein /(m**2*hour)
  endif
  minL = PAR*(1. + 2.*sqrt(k1_minL/k2_minL)) / (PAR + k1_minL + PAR*PAR/k2_minL)
  !minL = PAR/PAR_prod_max_warm*exp(1. - PAR/PAR_prod_max_warm)
  minN = gas%phosph(i)/(halfsat_DIP_photosynt + gas%phosph(i))
  prodox(i) = Pmax*minL*minN*(Chl_a(i) + chla_aver_oligo*0.1) !small number is added to avoid zero prodox at zero chl-a
  prodox(i) = prodox(i)/(hour_sec*molmass_o2) ! Converting from mg/(l*h) to mol/(m**3*s)
  if     (carbon_model%par == 1) then ! Stefan and Fang model for respiration and BOD
    ! Biochemical oxygen demand (BOD)
    BOD_ = Chla_aver*mO2C_dec*mCChla_dec ! stoichiometry details see in Stefan and Fang, 1994
    step20 = STEP(wst%Tw2(i) - T0) ! step function around 20 Celsius
    x = (theta_b1*step20 + theta_b2*(1-step20))*STEP(wst%Tw2(i) - T_md) ! using step function around 4 Celsius
    if (x == 0.) then
      bod(i) = 0.
    else
      bod(i) = k_b*x**(wst%Tw2(i) - T0)*BOD_
    endif
    bod(i) = bod(i)/(day_sec *molmass_o2) ! --||-- (Converting from mg/(l*day) to mol/(m**3*s))
  elseif (carbon_model%par == 2) then !Hanson et al. model for respiration and BOD
    !> @todo{introduce different timescale for degradation of allochtonous DOC}
    !Assuming all C atoms escaping POCD and DOC form CO2 
    bod(i) = (gas%POCD(i)/tau_POCD + & 
    & gas%DOC(i,1)/tau_DOC_auto + gas%DOC(i,2)/tau_DOC_allo) 
  endif
  ! Oxygen sink by respiration
  select case (nSOD)
  case(1)
    ! The model from Stefan and Fang (1994)
    resp(i) = 1./YCHO2*k_r*theta_r**(wst%Tw2(i) - T0)*Chl_a(i)
    resp(i) = resp(i)/(day_sec *molmass_o2) ! Converting from mg/(l*day) to mol/(m**3*s)
  case(2)
    ! The model from Hanson et al. (2004)
    resp(i) = alpha_POCL*prodox(i)
  end select
  ! Sedimentary oxygen demand (SOD)
  select case (nSOD)
  case(1) ! The model from Stefan and Fang (1994)
    x = step10*theta_s1 + (1 - step10)*theta_s2 !using step function
    sod(i) = Sb20*x**(wst%Tw2(i) - T0)/(day_sec *molmass_o2 * 1.E+3)
  case(2) ! The model from Walker and Sondgrass (1986)
    kc     = kc0*thetaC_SOD**(wst%Tw2(i) - 20.)
    mubeta = mubeta0*thetamu_SOD**(wst%Tw2(i) - 25.)
    sod(i) = kc*gas%oxyg(i,1) + mubeta*gas%oxyg(i,1)/(k_O2_SOD + gas%oxyg(i,1))
  case(3) ! New SOD model 
    k = gsp%ksoilcol(i) !The number of soil column intersecting i-th layer
    if (kcompl(k) == 0) then !checking if SOD calculated for k-th column
      if     (soilcolconjtype == 1) then
        j = bathymsoil(k,gs%ix,gs%iy)%icent
        x = bathymsoil(k,gs%ix,gs%iy)%dzSLc
      elseif (soilcolconjtype == 2) then
        j = bathymsoil(k,gs%ix,gs%iy)%itop
        x = bathymsoil(k,gs%ix,gs%iy)%dzSL
      endif
      ! Oxygen concentration at the soil surface from soil side
      diff = tortuosity_coef*DIFF_WATER_OXYGEN(Tsoil(1,k))
      oxysurf = oxygsoil(k) !OXSURF(Tsoil(1,k),0.5*dzs(1),diff,oxygsoil(k))
      ! Oxygen flux is calculated from logarithmic law
      sod(i) = LOGFLUX(sqrt(wst%u2(j)*wst%u2(j) + wst%v2(j)*wst%v2(j)), &
      & oxysurf/por(1) - gas%oxyg(j,1), x, z0_bot, z0_bot, 1._ireals, 1)
      ! Oxygen concentration is updated to the next time step
      call OXYGEN_TOPSOIL(Tsoil(1,k),sod(i),0.5*dzs(1),dt,oxygsoil(k))
      kcompl(k) = 1
    else
      sod(i) = sod(i-1) ! SOD flux is the same for all water layer intesecting the same soil column
    endif
  end select
  if (i > 1) then
    sod(i) = - sod(i) / area_int(i)*(area_half(i) - area_half(i-1)) / &
    & (h*ddz05(i-1))
  else
    sod(i) = - sod(i) / area_int(i)*(area_half(i) - area_int(i)) / &
    & (h*ddz05(i-1))
  endif
enddo
!sod   (:) = sod   (:)/(day_sec *molmass_o2 * 1.E+3) ! Converting from mg/(m*m*day) to mol/(m*m*s)

! Sedimentary oxygen demand at the bottom (flux)
select case (nSOD)
case(1) ! The model from Stefan and Fang (1994)
  step10 = STEP(wst%Tw2(gs%M+1) - 10.) ! step function around 10 Celsius
  x = step10*theta_s1 + (1 - step10)*theta_s2 ! using step function
  sodbot = Sb20*x**(wst%Tw2(gs%M+1) - T0)/(day_sec *molmass_o2 * 1.E+3) ! Converting from mg/(m*m*day) to mol/(m*m*s)
case(2) ! The model from Walker and Sondgrass (1986)
  kc = kc0*thetaC_SOD**(wst%Tw2(gs%M+1) - 20.)
  mubeta = mubeta0*thetamu_SOD**(wst%Tw2(gs%M+1) - 25.)
  sodbot = kc*gas%oxyg(gs%M+1,1) + mubeta*gas%oxyg(gs%M+1,1)/(k_O2_SOD + gas%oxyg(gs%M+1,1))
case(3) ! The new SOD model
  diff = tortuosity_coef*DIFF_WATER_OXYGEN(Tsoil(1,gs%nsoilcols))
  oxysurf = oxygsoil(gs%nsoilcols) !OXSURF(Tsoil(1,gs%nsoilcols),0.5*dzs(1),diff,oxygsoil(gs%nsoilcols))
  sodbot = LOGFLUX(sqrt(wst%u2(gs%M+1)*wst%u2(gs%M+1) + wst%v2(gs%M+1)*wst%v2(gs%M+1)), &
  & oxysurf/por(1) - gas%oxyg(gs%M+1,1), 0.25*gsp%ddz(gs%M)*h, z0_bot, z0_bot, 1._ireals, 1)
  call OXYGEN_TOPSOIL(Tsoil(1,gs%nsoilcols),sodbot,0.5*dzs(1),dt,oxygsoil(gs%nsoilcols))
end select


deallocate(PARz)
if (firstcall) firstcall = .false.
END SUBROUTINE OXYGEN_PRODCONS


SUBROUTINE CHLOROPHYLLA(z_full, H_mixed_layer, H_photic_zone, extwat, RadWater, &
& gas, M, Chl_a, itroph)
! This subroutine specifies the chlorophyll profile

use METH_OXYG_CONSTANTS, only : &
& C1_Chla_to_POCLphyto, C2_Chla_to_POCLphyto, &
& PAR_prod_max_warm, halfsat_DIP_photosynt, &
& molmass_ch2o, Chla_to_extinct, &
& k1_c1, k2_c2
use UNITS    , only : m3_to_l
use RADIATION, only : rad_type, nbands
use ARRAYS   , only : gas_type
use DRIVING_PARAMS, only : carbon_model
use TIMEVAR, only : hour_sec
use PHYS_CONSTANTS, only : Wm22Em2

implicit none

! Input variables
real(kind=ireals), intent(in) :: z_full(1:M+1) ! Depth at cell interfaces, m
real(kind=ireals), intent(in) :: H_mixed_layer ! Mixed layer depth, m
real(kind=ireals), intent(in) :: H_photic_zone ! Photic zone depth, m
real(kind=ireals), intent(in) :: extwat        ! Extinction coefficient, m**(-1)

type(rad_type)   , intent(inout) :: RadWater   ! Radiation fluxes in water
type(gas_type)   , intent(in) :: gas           ! Gases, organic and inorganic species

integer(kind=iintegers), intent(in) :: M ! Number of grid layers (cells)

! Output variables
real(kind=ireals), intent(out) :: Chl_a(1:M+1) ! Chlorophyll-a density, mg/l
integer(kind=iintegers), intent(out) :: itroph ! Trophic status

! Local variables
integer(kind=iintegers), save :: chlmodel = 2
real(4) :: depthact
real(4) :: chla_aver
real(kind=iintegers) :: phi_n, phi_l, Chla_to_POCL_phyto, PAR

integer(kind=iintegers) :: i, nb

logical, save :: firstcall = .true.

if (firstcall) then
  if (carbon_model%par == 1) chlmodel = 1
endif

select case (chlmodel)
case(1)
  ! Parameterization following Stefan and Fang (1994) with modifications:
  ! - no annual cycle
  if (extwat > extwat1) then 
  ! Eutrophic lake
    chla_aver = chla_aver_eu
    itroph = 3
  elseif (extwat <= extwat1 .and. extwat > extwat2) then
  ! Mesotrophic lake
    chla_aver = chla_aver_meso
    itroph = 2
  else
  ! Oligotrophic lake
    chla_aver = chla_aver_oligo
    itroph = 1
  endif
  depthact = max(H_mixed_layer,H_photic_zone)
  do i = 1, M+1
    Chl_a(i) = chla_aver*STEP(depthact - z_full(i)) ! step function
  enddo
case(2)
  ! Parameterization following Chapra (2008) and 
  ! (Sadeghian et al., Envir.Mod.Soft. 2018) 
  if (nbands > 1) then
    nb = 2
  else
    nb = 1
  endif
  do i = 1, M+1
    ! Optimal PAR for photosynthesis is a reference value for warm period
    if (i > 1) then
      PAR = 0.5*(RadWater%flux(i-1,nb) + RadWater%flux(i,nb))
    else
      PAR = RadWater%flux(0,nb)
    endif
    PAR = PAR*Wm22Em2*hour_sec !convertion to einsten/(m**2*hour), the units of PAR_prod_max_warm
    !phi_l = PAR/PAR_prod_max_warm*exp(1. - PAR/PAR_prod_max_warm)
    phi_l = PAR*(1. + 2.*sqrt(k1_c1/k2_c2)) / (PAR + k1_c1 + PAR*PAR/k2_c2)
    phi_n = gas%phosph(i)/(halfsat_DIP_photosynt + gas%phosph(i)) !using half-saturation phosphorus for photosynthesis
    !>@todo{use here phytoplankton instead of POCL}
    !if (phi_l > 0.5) then
    !  print*, phi_l, PAR, PAR_prod_max_warm, i
    !endif
    Chla_to_POCL_phyto = C1_Chla_to_POCLphyto + C2_Chla_to_POCLphyto*phi_n*(1.-phi_l)
    !Assuming stoichiometry of POCL that of sugar
    Chl_a(i) = gas%POCL(i)*molmass_ch2o*Chla_to_POCL_phyto
    !> @todo{add to extinction CDOC, detritus, pure water value}
    !RadWater%extinct(i,nb) = 0.5 + Chla_to_extinct*Chl_a(i)*m3_to_l !1. -- extinction at zero chlorophyll-a
  enddo
endselect

if (firstcall) firstcall = .false.
END SUBROUTINE CHLOROPHYLLA


!> Subroutine ADDOXPRODCONS adds to O_2 and CO_2 tendencies associated
!! with organics dynamics in a lake (photosynthesis, respiration,
!! BOD and SOD), and the same for DOC, POCL, POCD
SUBROUTINE ADDOXPRODCONS(M,i_maxN,H_mixed_layer,H_photic_zone, &
&                        prodox,resp,bod,sod,RadWater,dt,ls,gas)

use ARRAYS, only : gas_type
use ARRAYS_BATHYM, only : layers_type
use ARRAYS_GRID, only : gsp

use METH_OXYG_CONSTANTS, only : &
& alpha_POCL, tau_POCD, tau_DOC_auto, tau_DOC_allo, &
& beta_POCL, tau_Dh_epi, tau_Dh_hypo, &
& molmass_ch2o, &
& DOMauto_PtoC, DOMallo_PtoC, POM_PtoC, &
& sedoxid_PtoC, pH, &
& SUVA_305, AQY, UV_weight, PAR_weight

use DRIVING_PARAMS, only : &
& carbon_model

use PHYS_CONSTANTS, only : Kelvin0

use RADIATION, only : rad_type, nbands, bandbounds

implicit none

! Input variables
!> Mixed layer depth, m
real(kind=ireals), intent(in) :: H_mixed_layer 
!> Photic zone depth, m
real(kind=ireals), intent(in) :: H_photic_zone 
!> Production of O2 by photosynthesis, mol/(m**3*s)
real(kind=ireals), intent(inout) :: prodox(1:M+1) 
!> Sink due to respiration, mol/(m**3*s)
real(kind=ireals), intent(inout) :: resp(1:M+1)   
!> Biochemical oxygen demand, mol/(m**3*s)
real(kind=ireals), intent(inout) :: bod(1:M+1)    
!> Sediment oxygen demand (sink), mol/(m**3*s)
real(kind=ireals), intent(inout) :: sod(1:M+1)    
!> Radiation fluxes in water and optical properties
type(rad_type)   , intent(inout) :: RadWater
!> Time step, s
real(kind=ireals), intent(in) :: dt 
!> Number of model layers (cells)
integer(kind=iintegers), intent(in) :: M 
!> The computational level of maximal Brunt-Vaisala frequency
integer(kind=iintegers), intent(in) :: i_maxN
!> Physical layers' thicknesses
type(layers_type), intent(in) :: ls

!Input/ouput variables
!> Gas concentrations in water, carbon species content
type(gas_type), intent(inout) :: gas

! Local variables

real(kind=ireals) :: x, xx, hact, rad_CDOM, total_DOC
real(kind=ireals) :: E_POCL, D_DOC1, D_DOC2, P_POCL, R_POCL, Dh_POCL, D_POCD, tau_Dh, &
& Pd_DOC
real(kind=ireals) :: bodcorr(1:M)

integer(kind=iintegers) :: i, nb

do i = 1, M
  !Limitation of photosynthesis by DIC availability
  x  = min(gas%DIC(i,1)/CO2O2_prod        , dt*prodox(i)) 
  !Limitation of photosynthesis by DIP availability
  xx = min(gas%phosph(i)/(POM_PtoC*CO2O2_prod), dt*prodox(i)) 
  x  = min(x,xx)
  gas%DIC(i,1)    = gas%DIC(i,1)  - x*CO2O2_resp
  gas%oxyg(i,1)   = gas%oxyg(i,1) + x
  prodox(i) = x/dt

  x = min(gas%oxyg(i,1),dt*resp(i))
  !Limitation of respiration by available oxygen
  gas%oxyg(i,1)   = gas%oxyg(i,1) - x
  gas%DIC(i,1)    = gas%DIC(i,1)  + x*CO2O2_resp
  resp(i) = x/dt

  x = min(gas%oxyg(i,1),dt*bod(i))
  gas%oxyg(i,1)   = gas%oxyg(i,1) - x
  gas%DIC(i,1)    = gas%DIC(i,1)  + x*CO2O2_bod
  bodcorr(i) = x/(dt*(bod(i)+small_value))
  bod(i) = x/dt

  x = min(gas%oxyg(i,1),dt*sod(i))
  gas%oxyg(i,1)   = gas%oxyg(i,1) - x
  gas%DIC(i,1)    = gas%DIC(i,1)  + x*CO2O2_sod
  gas%phosph(i)   = gas%phosph(i) + x*sedoxid_PtoC
  sod(i) = x/dt
enddo

! Hanson et al. model tendencies for carbon species
if (carbon_model%par == 2) then
  hact = max(H_mixed_layer,H_photic_zone)
  do i = 1, M
    ! The death timescale assumed to be the same in hypolimion and during ice period
    !if (ls%l1 > 0 .or. i >= i_maxN) then
    if (ls%l1 > 0.) then
      tau_Dh = tau_Dh_hypo
    elseif (gsp%z_full(i) > hact) then
      x  = gsp%z_full(i) - hact
      xx = ls%h1 - hact
      tau_Dh = tau_Dh_epi - (tau_Dh_hypo - tau_Dh_epi)* &
      & ((x/xx)**2 - 2.*x/xx) !quadratic interpolation with zero derivative at bottom
    else
      tau_Dh = tau_Dh_epi
    endif

    !Photodissociation, assuming constant CDOM attenuation coefficient inside CDOC spectrum interval
    total_DOC = max(gas%DOC(i,1) + gas%DOC(i,2),0.)
    RadWater%extinct_CDOC(i) = SUVA_305*total_DOC
    if (nbands > 1) then
      ! Summing UV and PAR parts overlapping with CDOM absorbance spectrum,
      ! assuming linear spectrumn in UV and constant value in PAR regions
      rad_CDOM = RadWater%flux(i-1,1)*UV_weight + &
      &          RadWater%flux(i-1,2)*PAR_weight    
    else
      rad_CDOM = RadWater%flux(i-1,1)
    endif
    Pd_DOC = rad_CDOM * &
    & (1. - exp( - RadWater%extinct_CDOC(i)*gsp%ddz05(i-1)*ls%h1)) / &
    & gsp%ddz05(i-1)*ls%h1 * AQY
    if (Pd_DOC < 0.)then 
      print*, Pd_DOC, rad_CDOM, AQY, RadWater%extinct_CDOC(i)
      stop
    end if
    !Pd_DOC is subtracted from DOC but not taken into account in other
    !equations, as CO is the main product of photodegradation and there is
    !no enough evidence to assume it is fully added to DIC and not emitted to the atmosphere
    !(Zuo and Jones, Water Research, 1997)
    !print*, rad_CDOM, &
    !& (1. - exp( - RadWater%extinct_CDOC(i)*gsp%ddz05(i-1)*ls%h1))/ &
    !& gsp%ddz05(i-1)*ls%h1, AQY

    P_POCL = prodox(i) !*molmass_ch2o / kg_to_g
    R_POCL = resp(i)   !alpha_POCL*P_POCL
    E_POCL = beta_POCL*prodox(i) !*P_POCL
    !> @note{ using beta_POCL valid for mass concentrations is not strictly correct
    !! for molar concentrations, as E_POCL and P_POCL may have different C-to-mass ratios} 
    Dh_POCL = gas%POCL(i)/tau_Dh
    D_POCD = gas%POCD(i)/tau_POCD*bodcorr(i)
    D_DOC1 = gas%DOC(i,1)/tau_DOC_auto*bodcorr(i)
    D_DOC2 = gas%DOC(i,2)/tau_DOC_allo*bodcorr(i)

    !Adding tendencies to DOC
    gas%DOC(i,1) = gas%DOC(i,1) + dt*(E_POCL - D_DOC1)
    gas%DOC(i,2) = gas%DOC(i,2) + dt*(       - D_DOC2)
    !Extracting from DOM a number of C due to photodissociation
    gas%DOC(i,1) = gas%DOC(i,1) + dt*( - Pd_DOC*gas%DOC(i,1)/(total_DOC+small_value))
    gas%DOC(i,2) = gas%DOC(i,2) + dt*( - Pd_DOC*gas%DOC(i,2)/(total_DOC+small_value))
    !This limitation by zero is equivalent to correcting photodissociation, as the latter does not
    !entail correction in tendencies of other C quantities
    gas%DOC(i,1) = max(gas%DOC(i,1),0.)
    gas%DOC(i,2) = max(gas%DOC(i,2),0.)

    !Adding tendencies to POCL and POCD
    gas%POCL(i) = gas%POCL(i) + dt*(P_POCL - R_POCL - E_POCL - Dh_POCL)
    gas%POCD(i) = gas%POCD(i) + dt*(Dh_POCL - D_POCD)

    !Adding tendencis to phosphorus (DIP)
    gas%phosph(i) = gas%phosph(i) + dt*(D_DOC1*DOMauto_PtoC + D_DOC2*DOMallo_PtoC + &
    & ( - P_POCL + R_POCL + D_POCD)*POM_PtoC)
  enddo
endif

END SUBROUTINE ADDOXPRODCONS


SUBROUTINE OXYGEN_TOPSOIL(Tsoil,influx,h,dt,oxygsoil)

!A bulk model for oxygen in top of soil column,
!calculates oxygen concentration in an aerobic top
!soil layer at the next timestep

use METH_OXYG_CONSTANTS, only : &
k_O2_SOD, mubeta0, thetamu_SOD 

implicit none

!Input variables
real(kind=ireals), intent(in) :: Tsoil  !Temperature of aerobic layer, C
real(kind=ireals), intent(in) :: influx !Oxygen flux from water column, positive downwards
real(kind=ireals), intent(in) :: h      !Thickness of top aerobic soil layer, m
real(kind=ireals), intent(in) :: dt     !Timestep, s

!Input/output variables
real(kind=ireals), intent(inout) :: oxygsoil !Mean oxygen concentration in the top soil layer

!Local variables
real(kind=ireals) :: mubeta
logical, save :: firstcall = .true.

! Michaelis-Menthen (Monod) oxygen depletion according to Walker and Snodgrass, 1986, 
! neglecting "chemical sediment oxygen demand"
mubeta = mubeta0*thetamu_SOD**(Tsoil - 25.)/h
oxygsoil = (oxygsoil + influx/h*dt)/(1. + dt*mubeta/(k_O2_SOD + oxygsoil))

if (firstcall) firstcall = .false.
END SUBROUTINE OXYGEN_TOPSOIL


FUNCTION OXSURF(Tsoil,h,diff,oxmean)

!Calculates surface oxygen concentration from
!mean concentration in anaerobic layer. Mean is calculated
!from analytical solution of problem for oxygen concentration
!d^2 C/d^2 z - alpha*C=0 with Dirichlet b.c.s

use METH_OXYG_CONSTANTS, only : &
k_O2_SOD, mubeta0, thetamu_SOD 

implicit none

!Input variables
real(kind=ireals), intent(in) :: Tsoil  !Temperature of aerobic layer, C
real(kind=ireals), intent(in) :: h      !Thickness of anaerobic layer, m
real(kind=ireals), intent(in) :: diff   !Oxygen diffusivity in anaerobic layer
real(kind=ireals), intent(in) :: oxmean !Mean oxygen concentration in anaerobic layer

!Output variables
real(kind=ireals) :: OXSURF

!Local variables
real(kind=ireals) :: alpha, mubeta

mubeta = mubeta0*thetamu_SOD**(Tsoil - 25.)/h
alpha = sqrt(mubeta/(k_O2_SOD*diff))

OXSURF = oxmean*alpha*h*(exp(alpha*h) + 1)/(exp(alpha*h) - 1.)

END FUNCTION OXSURF


END MODULE OXYGEN_MOD
