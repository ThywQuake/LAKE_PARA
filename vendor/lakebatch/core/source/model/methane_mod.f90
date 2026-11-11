MODULE METHANE_LAKE_MOD

use NUMERICS, only : PROGONKA

use LAKE_DATATYPES, only : ireals, iintegers

contains
SUBROUTINE METHANE &
& (gas,pressure,wind10,ch4_pres_atm,zsoil,Tsoil,rosoil,&
& fbbleflx_ch4_sum,fbbleflx_ch4, &
& wl,wi,wa,Sals,rootss,rhogr,por,veg,qsoil_top,TgrAnn, methgenmh, &
& ddz,ddz05,wst,lammeth, h1, ls, dhw, dhw0, deepestsoil, ifflux, &
& lsm, bathymwater, &
& fplant, febul0, fdiff, ftot, fdiff_lake_surf, &
& febul_lake_bot, febul_lake_surf, &
& plant_sum,bull_sum,oxid_sum,rprod_sum, &
& anox,gs,gsp,dt,eps_surf,sodbot, &
& rprod_total_oldC,rprod_total_newC, &
& ice_meth_oxid_total, &
& h_talik,tot_ice_meth_bubbles,add_to_winter)

use DRIVING_PARAMS , only : &
& missing_value, &
& thermokarst_meth_prod, &  ! Switch for old organics methane production for thermokarst lakes
& tricemethhydr, &
& soil_meth_prod, & ! Switch for methane production from new organics
& soilswitch !switch for simulation of soil processes

use PHYS_CONSTANTS, only : &
& row0, g, row0, roa0, roi, &
& row0_d_roi, row0_d_roa0, &
& roa0_d_roi, roa0_d_row0, &
& Kelvin0, z0_bot

use METH_OXYG_CONSTANTS
use PHYS_PARAMETERS
use NUMERIC_PARAMS

use PHYS_FUNC, only : &
& WL_MAX, &
& HENRY_CONST, &
& DIFF_WATER_METHANE, &
& DIFF_AIR_METHANE, &
& GAS_WATATM_FLUX, &
& MELTPNT, &
& MELTINGPOINT, &
& LOGFLUX

use T_SOLVER_MOD, only : DIFF_COEF   

use BUBBLE_LAKE_MOD, only : BUBBLEFLUXAVER

use ARRAYS_SOIL, only : lsh_type
use ARRAYS_GRID, only : gridsize_type, gridspacing_type
use ARRAYS_BATHYM, only : bathym, layers_type
use ARRAYS_WATERSTATE, only : waterstate_type
use ARRAYS, only : gas_type, water_methane_indic

implicit none

!*  ===============================================================
!*  prod + oxid
!*	solution 1-D nonlineal equation
!*	describing production, oxidation
!*	and transport of methane from wetlands
!*  ===============================================================
!*	input parameters:
!*	=====================
!*    z(ml)    :	height of level in soil from soil surface (cm)
!*    Tsoil(ml)    :	the soil temperature (C degrees)
!*
!*	output parameters:
!*	=====================
!*    fplant   : the ch4 flux due to plant-mediated transport
!*    fdiff    : the diffusive ch4 flux at soil/water=atm boundary
!*    ftot	   : the total ch4 emission
!*  =============================================================
!*	parameters for production
!*	=============================
!*    rnpp     ! NPP (gC/(m**2*mo))
!*	  rnppmax  ! the maximum value of the NPP
!*    rnroot   ! the rooting depth (cm)
!*    q100     ! value lying within the range of observed
!*             ! q100 values ranging from 1.7 to 16
!*    r0       ! the constant rate factor (muM/s)
!*	=============================================================
!*	parameters for oxidation
!*	=============================
!*	  vmax	        ! Michaelis-Menten (muM/s)
!*	  rkm	        ! coefficients (muM)
!*	  q10	        ! the observed value for oxidation
!*	  TgrAnn(ml)	! the annual mean soil temperature (deg C)
!*	=============================================================
!*	parameters for ebullition
!*	=============================
!*    rke	!	a rate constant of the unit 1/s
!*    cmin	!	the concentration at which bubble
!*          !   formation occurs (muM)
!*    punveg	!	the percentage of unvegetated, bare soil
!*    cthresh	!	the threshold concentration
!*              !   for bubble formation
!*	=============================================================
!*	parameters for plant-mediated transport
!*	===========================================
!*     rkp	!	a rate constant of the unit 0.01/s
!*     tveg	!	a factor describing the quality of
!*          !   plant-mediated transport at a site
!*     pox	!	a certain fraction of methane oxidized in
!*          !   the oxic zone around the roots of plants
!*     rlmin   ! the parameter for the function of fgrow(t)
!*     rl      ! the parameter for the function of fgrow(t)
!*     rlmax   ! rlmax=rlmin+rl
!*	===============================================================

 type(gridsize_type), intent(in) :: gs
 type(gridspacing_type), intent(in) :: gsp
 type(layers_type), intent(in) :: ls

 real(kind=ireals), intent(in) :: pressure ! Atmospheric pressure, Pa
 real(kind=ireals), intent(in) :: wind10 ! Wind speed at 10 m above the surface    
 real(kind=ireals), intent(in) :: ch4_pres_atm ! Methane partial pressure in the atmosphere, Pa
 real(kind=ireals), intent(in) :: zsoil(1:gs%ns) ! numerical levels in soil, meters
 real(kind=ireals), intent(in) :: Tsoil(1:gs%ns) ! Celsius
 real(kind=ireals), intent(in) :: rosoil(1:gs%ns) ! soil density
 real(kind=ireals), intent(in) :: wl(1:gs%ns) ! liquid water content in a soil, kg/kg
 real(kind=ireals), intent(in) :: wi(1:gs%ns) ! ice content in a soil, kg/kg
 real(kind=ireals), intent(in) :: wa(1:gs%ns) ! air content in a soil, kg/kg
 real(kind=ireals), intent(in) :: Sals(1:gs%ns) ! Salinity, kg/kg
   
 real(kind=ireals), intent(in) :: rootss(1:gs%ns) ! meters
   
 real(kind=ireals), intent(in) :: rhogr(1:gs%ns) ! density of dry soil, kg/m**3
 real(kind=ireals), intent(in) :: por(1:gs%ns) ! soil porosity, n/d
 real(kind=ireals), intent(in) :: methgenmh(1:gs%ns) ! The rate of methane generation/consumption due to 
                                        ! methane hydrate dissociation/formation
 
 real(kind=ireals), intent(in) :: ddz(1:gs%M) ! the thickness of dzeta-layers in water, n/d
 real(kind=ireals), intent(in) :: ddz05(0:gs%M) ! the thickness of dzeta-layers in water, n/d

 type(waterstate_type), intent(in) :: wst !Water state variables

 real(kind=ireals), intent(inout) :: fbbleflx_ch4_sum(0:gs%M+1) ! the horizontally integrated bubble methane flux, mol/s
 real(kind=ireals), intent(in) :: fbbleflx_ch4(0:gs%M+1) ! the bubble methane flux, normalized by bottom value, n/d
 real(kind=ireals), intent(in) :: lammeth(1:gs%M) ! the heat turbulent diffusivity in water, m**2/s
 real(kind=ireals), intent(in) :: h1 ! the thickness (depth) of the water column, m
                                    ! ice and bottom ice
 real(kind=ireals), intent(in) :: dhw, dhw0   ! the time increment of water column thickness
                                    ! and those at its top, m
 logical, intent(in) :: deepestsoil ! .true. for the deepest soil column, .false. for those at bottom slope
 logical, intent(in) :: ifflux ! relevant when deepestsoil = .false.

 type(lsh_type), intent(in) :: lsm
 type(bathym),   intent(in) :: bathymwater(1:gs%M+1)

 real(kind=ireals), intent(in) :: dt ! Timestep, sec
 real(kind=ireals), intent(in) :: eps_surf ! TKE dissipation rate , m**2/s**3
 real(kind=ireals), intent(in) :: sodbot ! sedimentary oxygen demand, mol/(m**2*s)
  
 real(kind=ireals), intent(inout) :: veg
 
 ! Gases concentrations
 type(gas_type)   , intent(inout) :: gas

 real(kind=ireals), intent(in)    :: qsoil_top   ! Methane concentration at soil column top (if deepestsoil = .false.)
 real(kind=ireals), intent(inout) :: TgrAnn(1:gs%ns)
 real(kind=ireals), intent(out)   :: fplant, febul0, fdiff, ftot, fdiff_lake_surf, febul_lake_surf
 real(kind=ireals), intent(inout) :: febul_lake_bot
 real(kind=ireals), intent(inout) :: plant_sum, bull_sum,oxid_sum,rprod_sum
 real(kind=ireals), intent(inout) :: anox
 real(kind=ireals), intent(inout) :: tot_ice_meth_bubbles

 logical, intent(inout) :: add_to_winter
     
 real(kind=ireals), intent(out) :: rprod_total_oldC(1:gs%nsoilcols)
 real(kind=ireals), intent(out) :: rprod_total_newC(1:gs%nsoilcols)
 real(kind=ireals), intent(out) :: ice_meth_oxid_total ! Total amount of oxidized methane in ice cover during ice period, mol/m**2
 real(kind=ireals), intent(out) :: h_talik ! talik depth, m

   
 ! Local variables      

 integer(kind=iintegers), parameter :: bottom_bc = 1 ! 1 - Continuity of flux and temperature at the bottom
                                                     ! 2 - continuity of flux and gas exchange law across the bottom
     
 real(kind=ireals), parameter :: C_depth_age = 0.25 ! from 0.25 to 0.30, according to West and Plug, 2007
 real(kind=ireals), parameter :: C_talik_age = 0.5  ! from 0.5 to 0.7, according to West and Plug, 2007
 real(kind=ireals), parameter :: Tmelt_pnt = 0. ! melting point temperature, excluding salinity effect, degrees Celsius

 real(kind=ireals), parameter :: botflux_ch4 = missing_value !3.E-7  !bottom methane flux, positive downwards, mol/(m**2*sec) 
                                                                 !(for Neumann boundary condition, no coupling to soil) 

 real(kind=ireals), save :: fcoarse = 0.5 !the relative volume of the coarse pores
 real(kind=ireals), save :: scf = 16.04/(1./864.)! the scale factor  mgCH4/m**2/day

 real(kind=ireals) :: diff(gs%ns),forg(gs%ns),qplant(gs%ns),qebul(gs%ns),q_old(gs%ns),bull(gs%ns),ch4_crit(gs%ns)
 real(kind=ireals) :: oxid(gs%ns),dz_soil(gs%ns),z2(gs%ns),fcch4(gs%ns)
 real(kind=ireals) :: bp(gs%ns), ap(gs%ns)
 real(kind=ireals) :: rprod(gs%ns), rprod_new(gs%ns), rprod_old(gs%ns), plant(gs%ns)
 real(kind=ireals) :: n2(gs%ns)
 
 real(kind=ireals), allocatable :: a(:), b(:), c(:), f(:), y(:)

 real(kind=ireals) :: rnroot
 real(kind=ireals) :: froot(gs%ns), anox_crit(gs%ns)
 
 real(kind=ireals) :: pow_oldorg
 real(kind=ireals) :: lake_age
   
 real(kind=ireals) :: O2top, SO4top, NO3top
 real(kind=ireals) :: tmp_veg, t_grow, t_mature, fnpp, r0_meth
 real(kind=ireals) :: fin, ft, ft1, t50, fgrow, q_sum_old, fdiff1, q_sum_new
 real(kind=ireals) :: balance
 real(kind=ireals) :: Flux_atm, kexch
 real(kind=ireals) :: xx, yy, zz, uu, vv, dz_ ! working variables
 real(kind=ireals), save, pointer :: densi
 
 real(kind=ireals), allocatable :: z(:)
 real(kind=ireals), allocatable :: roots(:)
 real(kind=ireals), allocatable :: pressoil(:)
 
 real(kind=ireals), external :: DZETA, VARMEAN
   
 integer(kind=iintegers) :: i
 integer(kind=iintegers) :: i_talik

 real(kind=ireals), pointer :: qsoil(:), qwater(:,:), Twater(:), u(:), v(:)

 ! Constant of methane production in sediments is set differently for
 ! sediments in photic zone and underneath
 dz_ = ls%H_photic_zone - h1
 r0_meth = 0.5*( (1. - sign(1._ireals,dz_) )*r0_methprod_phot   (gs%isoilcol) + &
 &               (1. + sign(1._ireals,dz_) )*r0_methprod_nonphot(gs%isoilcol))

 qsoil  => gas%qsoil (:,gs%isoilcol)
 qwater => gas%qwater
 Twater => wst%Tw2
 u => wst%u1
 v => wst%v1

 allocate (z(1:gs%ns))
 allocate (roots(1:gs%ns))
 allocate (pressoil(1:gs%ns))
 allocate (a(1:nveclen),b(1:nveclen),c(1:nveclen), &
 &         f(1:nveclen),y(1:nveclen))
 a(:) = 0.; b(:) = 0.; c(:) = 0.; f(:) = 0.; y(:) = 0.

 if (soilswitch%par == 0 .and. botflux_ch4 == missing_value) then
   print*, 'Bottom flux of methane (botflux_ch4) must be set if soil is turned off: STOP'
   stop
 endif
 
 soilswitch_if : if (soilswitch%par == 1) then !Switch for soil processes

   z(:) = zsoil(:)
   roots(:) = rootss(:)

   pressoil(1) = pressure + row0*g*h1
   do i = 2, gs%ns
     pressoil(i) = pressoil(i-1) + (z(i) - z(i-1))*g*0.5*(rosoil(i) + rosoil(i-1))
   enddo

   ! TgrAnn(ns) = Tsoil(ns) !ms
   TgrAnn(gs%ns) = 0. ! excludes the factor of annual mean temperature
   
   if (tricemethhydr%par > 0.) then
     densi => densmh
   else
     densi => roi
   endif

   !Calculation of pore volume ratios for liquid water and air
   !They are needed for diffusivity calculations 
   ! do i = 1, ns
   !   water_porvolrat(i) = wl(i) / &
   !   & (por(i)*(row0/rhogr(i) + wl(i) + row0/densi*wi(i) + row0_d_roa0*wa(i)) )
   !   air_porvolrat(i) = wa(i) / &
   !   & (por(i)*(roa0/rhogr(i) + wa(i) + roa0/densi*wi(i) + roa0_d_row0*wl(i)) )
   ! enddo
   
   ! Specific treatment of the uppermost sediments' layer
   !Assuming exponential dependence of nitrogen concentration on depth
   n2(1) = 2.*n2_atm0*(1.-exp(-n2_exp_decay*0.5*z(2)) )/(z(2)*n2_exp_decay)
   n2(2:gs%ns) = 0. ! nitrogen concentration in the layers below the top one is
                    ! set to zero, since it decays rapidly with depth (Bazhin, 2001)

   !O2top - mean oxygen concentration in half of the top layer
   if (gs%isoilcol /= gs%nsoilcols) then !Lateral soil columns
     O2top = gas%oxyg(gsp%isoilcolc(gs%isoilcol),1)*por(1) ! bulk concentration
   else
     O2top = gas%oxyg(gs%M+1,1)*por(1) ! bulk concentration
   endif
   O2top = 2.*O2top*(1.-exp(-O2_exp_decay*0.5*z(2)) )/(z(2)*O2_exp_decay) !mean of the exponent
   !SO4top - mean sulfate concentration in half of the top layer
   SO4top = Sals(1)*kgkg_sal_to_molm3_so4*por(1) !bulk concentration
   SO4top = 2.*SO4top*(1.-exp(-SO4_exp_decay*0.5*z(2)) )/(z(2)*O2_exp_decay) !mean of the exponent
   !NO3top - mean nitrate concentration in half of the top layer
   NO3top = 0.
                  
   !Calculation of the talik depth and of the power in production term due to 
   !old talik organics decomposition 
   i_talik = 0
   h_talik = 0.    
   do i = 1, gs%ns-1
     if (Tsoil(i)   >  MELTINGPOINT(Sals(i)/wl(i),    pressoil(i),  tricemethhydr%par) .and. &
       & Tsoil(i+1) <= MELTINGPOINT(Sals(i+1)/wl(i+1),pressoil(i+1),tricemethhydr%par)) then
       h_talik = z(i) + (z(i+1) - z(i)) * &
       & (MELTINGPOINT((Sals(i) + Sals(i+1))/(wl(i) + wl(i+1)), 0.5*(pressoil(i) + pressoil(i+1)), &
       & tricemethhydr%par) - Tsoil(i))/(Tsoil(i+1) - Tsoil(i))
       h_talik = min( max(h_talik, z(i)), z(i+1) )
       if (h_talik < 0.5*(z(i) + z(i+1))) then
         i_talik = i
       else
         i_talik = i + 1
       endif
       exit
     endif
   enddo

   ! if (.not.allocated(mean_rprod_oldc0)) then
   !   allocate (mean_rprod_oldc0(1:gs%ns))
   !   do i = 2, ns-1
   !     mean_rprod_oldc0(i) = &
   !     & MEAN_RPROD_OLDC(0.5*(z(i-1) + z(i)), 0.5*(z(i+1) + z(i)), &
   !     & h_talik, C_talik_age)
   !   enddo
   !   mean_rprod_oldc0(1) = &
   !   & MEAN_RPROD_OLDC(z(1), 0.5*(z(2) + z(1)), &
   !   & h_talik, C_talik_age)
   !   mean_rprod_oldc0(ns) = &
   !   & MEAN_RPROD_OLDC(0.5*(z(ns-1) + z(ns)), z(ns), &
   !   & h_talik, C_talik_age)
   ! endif
    
   !Calculation of approximate THERMOKARST lake age, according to results by West and Plug, 2007
   ! lake_age = (h1/C_depth_age)**2 ! years
   ! lake_age = (h_talik/C_talik_age)**2 ! years
   ! pow_oldorg = 2.*alpha_old_org*lake_age
                 
   do i = 1, gs%ns
     xx = HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, Tsoil(i) + Kelvin0)
     yy = HENRY_CONST(Henry_const0_n2,  Henry_temp_dep_n2,  Henry_temp_ref, Tsoil(i) + Kelvin0)
     ch4_crit(i) = rel_conc_ebul_crit*por(i)*(pressoil(i)*xx - xx/yy*n2(i)) ! assuming hydrostatic pressure in soil h1 + z(i) 
   enddo
   
   do i = 1, gs%ns 
     anox_crit(i) = WL_MAX(por(i),rhogr(i),wi(i),tricemethhydr%par)*0.9
   end do

   !Calculation of rooting depth
   rnroot = 0.
   do i = 1, gs%ns 
     if (roots(i).gt.0.) rnroot = z(i)
   end do
      
   !*	============================================================
   !*	tgrow is the temperature at which plants start to grow
   !*	============================================================
   
   if (Tsoil(2) .lt. 5.) then ! 5 deg. Celsius
     t_grow = 2. ! deg. Celsius
   else
     t_grow = 7. ! deg. Celsius
   end if
   t_mature = t_grow + 10. ! the temperature at which plants reach maturity, deg. Celsius
       
   !*  =============================================================
   !*	parameters for diffusion
   !*	=============================
   
   anox = 0.
   
   if (rnroot > 0.) then
     froot(1:gs%ns) = 0.
     forall (i = 1:gs%ns, z(i) <= rnroot) froot(i) = 2.*(rnroot-z(i))/rnroot
   else
     froot(1:gs%ns) = 0.
   endif
   
   forg(1:gs%ns) = veg * 1.
   ! Vegetated soil
   forall (i = 1:gs%ns, z(i) > rnroot) forg(i) = veg * exp((rnroot - z(i))*10.)

   do i = 1, gs%ns                                                  ! for unvegetated
     forg(i) = forg(i) + (1.-veg) * forg0*exp(-z(i)*lambda_new_org) ! soils
   end do

   do i = 2, gs%ns 
     diff(i) = tortuosity_coef * DIFF_WATER_METHANE( 0.5*(Tsoil(i) + Tsoil(i-1)) )
   end do

   rprod_total_newC(gs%isoilcol) = 0.
   rprod_total_oldC(gs%isoilcol) = 0.
   
   do i = 1, gs%ns 
   
     if (wl(i) .ge. anox_crit(i)) then
       ft1 = 1. ! Note, methane generation through old organics decomposition
                ! is zero below talik automatically, see the code below
       if (Tsoil(i) .gt. MELTINGPOINT(Sals(i)/wl(i),pressoil(i),tricemethhydr%par)) then
         if (i /= i_talik .or. i == 1 .or. i == gs%ns) then
           ft = 1.
         else
           ft = max((h_talik - 0.5*(z(i-1) + z(i)) ) / &
           & (0.5*(z(i+1) - z(i-1))),0.e0_ireals)
         endif
       else
         if (i /= i_talik .or. i == 1 .or. i == gs%ns) then
           ft = 0.
         else
           if (i == gs%ns) then
             dz_ = 0.5*(z(i) - z(i-1))
           else
             dz_ = 0.5*(z(i+1) - z(i-1))
           endif
           ft = max((h_talik - 0.5*(z(i-1) + z(i)) ) / dz_, 0.e0_ireals)
         endif
       endif
     else
       ft = 0.
       if (i /= gs%ns .and. i /= 1) then
         if ((Tsoil(i)   > MELTINGPOINT(Sals(i)/wl(i),    pressoil(i),  tricemethhydr%par)   .and. &
         &    Tsoil(i+1) < MELTINGPOINT(Sals(i+1)/wl(i+1),pressoil(i+1),tricemethhydr%par) ) .or.  &
         &   (Tsoil(i)   < MELTINGPOINT(Sals(i)/wl(i),    pressoil(i),  tricemethhydr%par)   .and. &
         &    Tsoil(i-1) > MELTINGPOINT(Sals(i-1)/wl(i-1),pressoil(i-1),tricemethhydr%par)) ) then
         !Regularization since there is always minimum of soil moisture content
         !near the talik depth (due to diffusion from water saturated layers above
         !to frozen layers below)
         ft1 = 1. - (anox_crit(i) - wl(i))/anox_crit(i)
         ! ft1 = 1.
         ! ft = ft1
         ! if (i == i_talik) &
         ! & ft = ft * (h_talik - 0.5*(z(i-1) + z(i)) ) / &
         ! & (0.5*(z(i+1) - z(i-1)))
         else
           ft1 = 0.
         endif
       else
         ft1 = 0.
       endif
     endif

     !  Note that for the performance optimization MEAN_RPROD_OLDCs may
     !  be computed only once - at the first timestep
     
     !  Note that calculating MEAN_RPROD_OLDC the temperature step
     !  function weighting is already implemented, so ft multiplication
     !  is not needed

     if (i == 1) then
       xx = MEAN_RPROD_OLDC2(0.e0_ireals, 0.5*(z(2) + z(1)), h_talik, C_talik_age)
       rprod_old(i) = xx*q100**((Tsoil(1)-TgrAnn(gs%ns))*0.1)*ft1*thermokarst_meth_prod%par
     elseif (i == gs%ns) then
       xx = MEAN_RPROD_OLDC2(0.5*(z(gs%ns) + z(gs%ns-1)), z(gs%ns), h_talik, C_talik_age)
       rprod_old(i) = xx*q100**((Tsoil(gs%ns)-TgrAnn(gs%ns))*0.1)*ft1*thermokarst_meth_prod%par
     else
       xx = MEAN_RPROD_OLDC2(0.5*(z(i) + z(i-1)), 0.5*(z(i+1) + z(i)), h_talik, C_talik_age)
       rprod_old(i) = xx*q100**((Tsoil(i)-TgrAnn(gs%ns))*0.1)*ft1*thermokarst_meth_prod%par
     endif

     !   fnpp = rnpp             !function of the variation of NPP with t
     !   if (l1 /= 0.) then
     !     fnpp = 0.
     !   else
     !     fnpp = rnppmax
     !   endif
     fin = 1. !+ fnpp/rnppmax !variation of substrate availability with t
     
     rprod_new(i) = &
     & r0_meth*forg(i)*fin * & 
     & q100**((Tsoil(i)-TgrAnn(gs%ns))*0.1)*ft*soil_meth_prod%par

     ! Oxygen and sulfate inhibition in the top half of the uppermost soil layer
     if (i == 1) then
       if (SO4top > SO4inhib_conc) then
         ! CH_4 production = 0 if SO_4 concentration exceeds critical value
         rprod_new(i) = 0.
       else
         ! Inhibition of methane production by oxygen 
         rprod_new(i) = rprod_new(i)/(1. + O2inhib_const*O2top) 
       endif
     endif

     if (i == 1) then
       rprod_total_oldC(gs%isoilcol) = rprod_total_oldC(gs%isoilcol) + rprod_old(1)*0.5*(z(2)-z(1))
       rprod_total_newC(gs%isoilcol) = rprod_total_newC(gs%isoilcol) + rprod_new(1)*0.5*(z(2)-z(1))
     elseif (i == gs%ns) then
       rprod_total_oldC(gs%isoilcol) = rprod_total_oldC(gs%isoilcol) + rprod_old(gs%ns)*0.5*(z(gs%ns)-z(gs%ns-1))
       rprod_total_newC(gs%isoilcol) = rprod_total_newC(gs%isoilcol) + rprod_new(gs%ns)*0.5*(z(gs%ns)-z(gs%ns-1))
     else
       rprod_total_oldC(gs%isoilcol) = rprod_total_oldC(gs%isoilcol) + rprod_old(i)*0.5*(z(i+1)-z(i-1))
       rprod_total_newC(gs%isoilcol) = rprod_total_newC(gs%isoilcol) + rprod_new(i)*0.5*(z(i+1)-z(i-1))
     endif
     
     rprod(i) = rprod_new(i) + rprod_old(i)
     
     if (qsoil(i) .ge. ch4_crit(i) .and. wi(i) == 0.e0_ireals) then ! Formation of bubbles is not allowed in frozen soil
       fcch4(i) = 1.
     else
       fcch4(i) = 0.
     end if
     
   end do
   

   !    ===============================================================
   !    the function fgrow describes the growing state of the plants
   !    ===============================================================
   !    t50 - the soil temperature at 50cm depth below soil surface

   do i = 1, gs%ns !ms, ml
     if (z(i) .le. 0.5) then 
       t50 = Tsoil(i)
     else
       exit
     endif
   end do
   
   if (t50.lt.t_grow) fgrow = rlmin
   if (t50 .ge. t_grow .and. t50 .le. t_mature) then
     fgrow = rlmin + rl*(1.-((t_mature-t50)/(t_mature-t_grow))**2)
   endif
   if (t50.gt.t_mature) fgrow = rlmax
   
   !    =============================================================


   do i = 1, gs%ns !ms, ml
     ! oxid(i) - coefficient for oxidation
     ! methane oxidation occurs only in the unsaturated soil
     ! layers
     if (wl(i).lt.anox_crit(i)) then
     ! oxid(i) = vmax*q10**((Tsoil(i) - TgrAnn(ns))*0.1)/(rkm + qsoil(i)) !ml

     ! Currently oxidation is turned off in soil layers.
     ! It should be taken into account when oxygen concentration will
     ! be somehow extimated (calculated).
       oxid(i) = 0.
     else
       oxid(i) = 0.
     end if

     ! plant(i) - coefficient for plant-mediated transport
     ! plant(i) = rkp*tveg*froot(i)*fgrow*(1.-0.5)*veg

     plant(i) = rkp*tveg*froot(i)*fgrow*veg
     ! bull(i) - coefficient for ebullition

     if (wl(i) .ge. anox_crit(i)) then
       bull(i) = rke*fcch4(i)
     else
       bull(i) = 0.
     end if
     
   end do

   ! Oxidation in top half-layer
   oxid(1) = Vmax_soil*O2top/(k_o2_soil + O2top)/(k_ch4_soil + qsoil(1)) !ml

   oxid_sum = 0.
   rprod_sum = 0.
   bull_sum = 0.
   plant_sum = 0.
   q_sum_old = 0.
   do i = 2, gs%ns-1 !ms+1, ml-1
     q_sum_old = q_sum_old + qsoil(i)*0.5*(z(i+1)-z(i-1))
   end do

   q_sum_old = q_sum_old + qsoil(gs%ns)*0.5*(z(gs%ns)-z(gs%ns-1))
   q_sum_old = q_sum_old + qsoil(1)*(0.5*(z(2)-z(1))) ! + 0.5*ddz(M)*h1/por(1))

   ! Water methane content
   do i = 1, gs%M+1
     q_sum_old = q_sum_old + qwater(i,1)*h1*ddz05(i-1)
   enddo

   do i = 1, gs%ns !1, ml
     q_old(i) = qsoil(i)
   end do

   !=============================================================
   !    the methane flux due to ebullition (explicit scheme)
   !=================febul(t)====================================
   
   qebul(:) = 0.
   do i = 1, gs%ns !ms, ml
     if (wl(i) .gt. anox_crit(i)) then
       qebul(i) = bull(i)*(q_old(i) - ch4_crit(i)) ! cthresh -> ch4_crit(i), according to explicit scheme
     endif
   end do
    
   !Methane bubble flux at the bottom
   febul0 = 0.
   do i = 2, gs%ns-1 !ms+1, ml
      febul0 = febul0 + 0.5*(z(i+1)-z(i-1))*qebul(i)
   end do
   febul0 = febul0 + qebul(1)*0.5*(z(2)-z(1))
   febul0 = febul0 + qebul(gs%ns)*0.5*(z(gs%ns)-z(gs%ns-1))
   
   if (deepestsoil) then
     ! Adding bubble flux from the lowest soil column to horizontally averaged bubble flux
     if (gs%nsoilcols > 1) then
       call BUBBLEFLUXAVER(gs%ix,gs%iy,gs%nsoilcols)
     else
       fbbleflx_ch4_sum(0)      = febul0*fbbleflx_ch4(   0)  *bathymwater(1)     %area_int 
       fbbleflx_ch4_sum(1:gs%M) = febul0*fbbleflx_ch4(1:gs%M)*bathymwater(1:gs%M)%area_half
       fbbleflx_ch4_sum(gs%M+1) = febul0*fbbleflx_ch4(gs%M+1)*bathymwater(gs%M+1)%area_int 
       febul_lake_bot = febul0*bathymwater(1)     %area_int
     endif
     febul_lake_surf = fbbleflx_ch4_sum(0)/bathymwater(1)%area_int
     febul_lake_bot  = febul_lake_bot     /bathymwater(1)%area_int
   endif
 
 else

   !With soil deactivated, ebullition flux of methane is assumed zero
   febul0 = 0.
   fbbleflx_ch4_sum = 0.

 endif soilswitch_if 

!    =============================================================
!    set up the tridiagonal system of equations
!    =============================================================

if (deepestsoil) then
  water_exists_if : if (h1 > 0) then
    if (ls%l1 == 0) then
      ! c(1) = 1.
      ! b(1) = 0.
      ! f(1) = ch4_pres_atm * &
      ! & HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, Twater(1)+273.15) 
      ! ch4_atm0 ! atmospheric concentration
      xx = HENRY_CONST(Henry_const0_ch4, Henry_temp_dep_ch4, Henry_temp_ref, Twater(1)+Kelvin0)
      Flux_atm = GAS_WATATM_FLUX &
      & (Twater(1),wind10,qwater(1,1),ch4_pres_atm,xx,water_methane_indic,eps_surf)        
    else
      Flux_atm = 0.
    endif
    ! Top of water column (Crank - Nicolson scheme)
    xx = 0.5*( - bathymwater(1)%area_half/bathymwater(1)%area_int * &
    & lammeth(1)/(h1*ddz(1)) + dhw0/(2.d0*dt) )
    yy = ddz(1)*h1/(2.d0*dt)
    c(1)   = xx - yy
    b(1)   = xx
    f(1)   = - yy*qwater(1,1) + Flux_atm + xx*(qwater(2,1) - qwater(1,1)) - &
    & lsm%water(1)*yy*dt 
    call DIFF_COEF(nveclen,a,b,c,f,2,gs%M,2,gs%M,water_methane_indic,dt)
    water_bottom_bc_if : if (botflux_ch4 /= missing_value) then
      !Methane flux at the deepest point of a lake is set as b.c., coupling to soil is switched off
      xx = 0.5*( - bathymwater(gs%M)%area_half/bathymwater(gs%M+1)%area_int * &
      & lammeth(gs%M)/(ddz(gs%M)*h1) + (dhw - dhw0)/(2.d0*dt) )
      yy = ddz(gs%M)*h1/(2.d0*dt)
      c(gs%M+1) = xx - yy  ! bug corrected: - (dhw-dhw0)/(2.d0*dt) 
      a(gs%M+1) = xx 
      f(gs%M+1) = - yy*qwater(gs%M+1,1) + xx*(qwater(gs%M,1) - qwater(gs%M+1,1)) + botflux_ch4
      call PROGONKA (nveclen,a,b,c,f,y,1,gs%M+1)
      y(1:gs%M+1) = max(y(1:gs%M+1),0.e0_ireals)
      qwater(1:gs%M+1,2) = y(1:gs%M+1)
    else
      bottom_layer_if : if (ls%ls1 > 0) then
        ! Case water, deepice and soil; upper ice and snow are allowed       
        ! Methane diffusion in water       
        xx = 0.5*( - bathymwater(gs%M)%area_half/bathymwater(gs%M+1)%area_int * &
        & lammeth(gs%M)/(ddz(gs%M)*h1) + (dhw - dhw0)/(2.d0*dt) )
        yy = ddz(gs%M)*h1/(2.d0*dt)
        c(gs%M+1) = xx - yy  ! bug corrected: - (dhw-dhw0)/(2.d0*dt) 
        a(gs%M+1) = xx 
        f(gs%M+1) = - yy*qwater(gs%M+1,1) + xx*(qwater(gs%M,1) - qwater(gs%M+1,1)) ! + qflux1
        call PROGONKA (nveclen,a,b,c,f,y,1,gs%M+1)
        y(1:gs%M+1) = max(y(1:gs%M+1),0.e0_ireals)
        qwater(1:gs%M+1,2) = y(1:gs%M+1)
        ! Methane evolution in soil under bottom ice
        b(1) = dt*diff(2)/((z(2)-z(1))*(z(2)-z(1)))
        xx = b(1) + 0.5*dt*(plant(1) + oxid(1))
        c(1) = 1. + xx
        ! No diffusive flux is assumed at the ice-soil interface      
        f(1) = qsoil(1) + dt*(methgenmh(1) + rprod(1) + &
        & bull(1)*(ch4_crit(1) - qsoil(1))) + b(1)*qsoil(2) - xx*qsoil(1)
        do i = 2, gs%ns-1 !ms+1, ml-1
          ap(i) = dt/((z(i+1)-z(i-1))*(z(i+1)-z(i)))
          bp(i) = dt/((z(i+1)-z(i-1))*(z(i)-z(i-1)))
          a(i)  = bp(i) * diff(i)
          b(i)  = ap(i) * diff(i+1)
          xx    = ap(i) * diff(i+1) + bp(i)*diff(i) + 0.5*dt * (plant(i) + oxid(i))
          c(i)  = 1. + xx
          f(i)  = qsoil(i) + dt*(methgenmh(i) + rprod(i) + &
          & bull(i)*(ch4_crit(i) - qsoil(i))) + &
          & a(i)*qsoil(i-1) + b(i)*qsoil(i+1) - xx*qsoil(i)
        end do
        bp(gs%ns) = dt/( (z(gs%ns)-z(gs%ns-1))*(z(gs%ns)-z(gs%ns-1)) )
        a(gs%ns)  = bp(gs%ns) * diff(gs%ns) ! diff(ns-1)
        xx     = a(gs%ns) + 0.5*dt * (plant(gs%ns) + oxid(gs%ns))
        c(gs%ns) = 1. + xx
        f(gs%ns) = qsoil(gs%ns) + dt*(methgenmh(gs%ns) + rprod(gs%ns) + &
        & bull(gs%ns)*(ch4_crit(gs%ns) - qsoil(gs%ns)) ) + &
        & a(gs%ns)*qsoil(gs%ns-1) - xx*qsoil(gs%ns)
        call PROGONKA(nveclen,a, b, c, f, y, 1, gs%ns)
        y(1:gs%ns) = max(y(1:gs%ns),0.e0_ireals)
        qsoil(1:gs%ns) = y(1:gs%ns)     
      else
        if (bottom_bc == 1) then ! Continuity of concentration and flux
          ! Case water, soil; upper ice and snow are allowed       
          ! ------------------WATER-SOIL INTERFACE------------------
          ! Porosity is added since concentration in water equals to pore(!) concentration in ground
          uu = - (bathymwater(gs%M+1)%area_int - bathymwater(gs%M)%area_half) / &
          &      (bathymwater(gs%M+1)%area_int*bathymwater(gs%M+1)%area_int) * &
          &       fbbleflx_ch4_sum(gs%M+1)
          zz = bathymwater(gs%M+1)%area_int/bathymwater(gs%M)%area_half
          a(gs%M+1) = 0.5*( lammeth(gs%M)/(ddz(gs%M)*h1) - zz*(dhw-dhw0)/(2.*dt) )
          b(gs%M+1) = 0.5*diff(2)/( (z(2)-z(1)) )
          xx     = por(1)*(z(2)-z(1))/(2.*dt) + zz*ddz(gs%M)*h1/(2.*dt)
          yy     = 0.25*(z(2)-z(1))*por(1)*(plant(1) + oxid(1)) + a(gs%M+1) + por(1)*b(gs%M+1)
          c(gs%M+1) = xx + yy ! porosity added
          f(gs%M+1) = qwater(gs%M+1,1)*xx + 0.5*(z(2)-z(1))*(methgenmh(1) + rprod(1) + &
          & bull(1)*(ch4_crit(1) - qsoil(1))) - methox2sod*sodbot + &
          & a(gs%M+1)*qwater(gs%M,1) + b(gs%M+1)*qsoil(2) - yy*qwater(gs%M+1,1) + &
          & uu*zz
          ! --------------------------------------------------------
          do i = 2, gs%ns-1 !ms+1, ml-1
            ap(i)  = dt/((z(i+1)-z(i-1))*(z(i+1)-z(i)))
            bp(i)  = dt/((z(i+1)-z(i-1))*(z(i)-z(i-1)))
            a(i+gs%M) = bp(i) * diff(i)
            b(i+gs%M) = ap(i) * diff(i+1)
            xx     = ap(i)*diff(i+1) + bp(i)*diff(i) + 0.5 * dt * (plant(i) + oxid(i))
            c(i+gs%M) = 1. + xx 
            f(i+gs%M) = qsoil(i) + dt*(methgenmh(i) + rprod(i) + &
            & bull(i)*(ch4_crit(i) - qsoil(i))) + &
            & a(i+gs%M)*qsoil(i-1) + b(i+gs%M)*qsoil(i+1) - xx*qsoil(i)
            if (i == 2) a(i+gs%M) = a(i+gs%M)*por(1)
          end do
          bp(gs%ns)  = dt/( (z(gs%ns)-z(gs%ns-1))*(z(gs%ns)-z(gs%ns-1)) )
          a(gs%M+gs%ns) = bp(gs%ns) * diff(gs%ns)
          xx      = a(gs%M+gs%ns) + 0.5 * dt * (plant(gs%ns) + oxid(gs%ns))
          c(gs%M+gs%ns) = 1. + xx
          f(gs%M+gs%ns) = qsoil(gs%ns) + dt*(methgenmh(gs%ns) + rprod(gs%ns) + &
          & bull(gs%ns)*(ch4_crit(gs%ns) - qsoil(gs%ns))) + &
          & a(gs%M+gs%ns)*qsoil(gs%ns-1) - xx*qsoil(gs%ns)
          call PROGONKA (nveclen,a,b,c,f,y,1,gs%M+gs%ns)
          y(1:gs%M+gs%ns) = max(y(1:gs%M+gs%ns),0.e0_ireals)
          qwater(1:gs%M+1,2) = y(1:gs%M+1)
          qsoil(2:gs%ns) = y(gs%M+2:gs%M+gs%ns)
          qsoil(1) = y(gs%M+1)*por(1) ! converting from pore concentration to bulk concentration
          ! y(1:M+1) - methane concentration in water
          ! y(M+2:M+ns) - methane concentration in soil
        elseif (bottom_bc == 2) then ! Continuity of flux and the law for diffusive gas exchange accross the bottom
          ! Case water, soil; upper ice and snow are allowed       
          !  ------------------WATER-SOIL INTERFACE------------------
          ! Porosity is added since concentration in water equals to pore(!) concentration in ground
          ! Water side
          uu = - (bathymwater(gs%M+1)%area_int - bathymwater(gs%M)%area_half) / &
          &      (bathymwater(gs%M+1)%area_int*bathymwater(gs%M+1)%area_int) * &
          &       fbbleflx_ch4_sum(gs%M+1)
          zz = bathymwater(gs%M+1)%area_int/bathymwater(gs%M)%area_half
          xx = zz*ddz(gs%M)*h1/(2.*dt)
          ! Accounting for top methane-inhibited layer
          if (CH4_exp_growth /= 0.) then
            vv = CH4_exp_growth*0.5*(z(2) - z(1)) / &
            & ( (exp(CH4_exp_growth*0.5*(z(2) - z(1))) - 1.) * por(1) )
          else
            vv = 1./por(1)
          endif
          ! Exchange coefficient for bottom logarithmic layer
          kexch = LOGFLUX(sqrt(u(gs%M+1)*u(gs%M+1) + v(gs%M+1)*v(gs%M+1)), 1._ireals, &
          & 0.25*ddz(gs%M)*h1, z0_bot, z0_bot, 1._ireals, 2)
          a(gs%M+1) = 0.5*( & !bathymwater(gs%M+1)%area_int/bathymwater(gs%M)%area_int * &
          & lammeth(gs%M)/(ddz(gs%M)*h1) - zz*(dhw-dhw0)/(2.*dt) )
          b(gs%M+1) = 0.5*kexch*vv
          yy     = a(gs%M+1) + b(gs%M+1)/vv
          c(gs%M+1) = xx + yy 
          f(gs%M+1) = qwater(gs%M+1,1)*xx + &
          & a(gs%M+1)*qwater(gs%M,1) + b(gs%M+1)*qsoil(1) - yy*qwater(gs%M+1,1) + &
          & uu*zz
 
          ! Soil side
          xx     = (z(2)-z(1))/(2.*dt) 
          a(gs%M+2) = 0.5*kexch
          b(gs%M+2) = 0.5*diff(2)/( (z(2)-z(1)) )
          yy     = 0.25*(z(2)-z(1))*(plant(1) + oxid(1)) + a(gs%M+2)*vv + b(gs%M+2)
          c(gs%M+2) = xx + yy 
          f(gs%M+2) = qsoil(1)*(xx - yy) + & !b(gs%M+2) - a(gs%M+2)*vv) + &
          & 0.5*(z(2)-z(1))*(methgenmh(1) + rprod(1) + bull(1)*(ch4_crit(1) - qsoil(1))) - &
          & methox2sod*sodbot + a(gs%M+2)*qwater(gs%M+1,1) + b(gs%M+2)*qsoil(2)
          do i = 2, gs%ns-1 !ms+1, ml-1
            ap(i)  = dt/((z(i+1)-z(i-1))*(z(i+1)-z(i)))
            bp(i)  = dt/((z(i+1)-z(i-1))*(z(i)-z(i-1)))
            a(i+gs%M+1) = bp(i) * diff(i)
            b(i+gs%M+1) = ap(i) * diff(i+1)
            xx     = ap(i)*diff(i+1) + bp(i)*diff(i) + 0.5 * dt * (plant(i) + oxid(i))
            c(i+gs%M+1) = 1. + xx 
            f(i+gs%M+1) = qsoil(i) + dt*(methgenmh(i) + rprod(i) + &
            & bull(i)*(ch4_crit(i) - qsoil(i))) + &
            & a(i+gs%M+1)*qsoil(i-1) + b(i+gs%M+1)*qsoil(i+1) - xx*qsoil(i)
          end do
          bp(gs%ns)  = dt/( (z(gs%ns)-z(gs%ns-1))*(z(gs%ns)-z(gs%ns-1)) )
          a(gs%M+1+gs%ns) = bp(gs%ns) * diff(gs%ns)
          xx      = a(gs%M+1+gs%ns) + 0.5 * dt * (plant(gs%ns) + oxid(gs%ns))
          c(gs%M+1+gs%ns) = 1. + xx
          f(gs%M+1+gs%ns) = qsoil(gs%ns) + dt*(methgenmh(gs%ns) + rprod(gs%ns) + &
          & bull(gs%ns)*(ch4_crit(gs%ns) - qsoil(gs%ns))) + &
          & a(gs%M+1+gs%ns)*qsoil(gs%ns-1) - xx*qsoil(gs%ns)
          call PROGONKA (nveclen,a,b,c,f,y,1,gs%M+1+gs%ns)
          y(1:gs%M+1+gs%ns) = max(y(1:gs%M+1+gs%ns),0.e0_ireals)
          qwater(1:gs%M+1,2) = y(1:gs%M+1)
          qsoil(1:gs%ns) = y(gs%M+2:gs%M+1+gs%ns)
          ! y(1:M+1) - methane concentration in water
          ! y(M+2:M+ns+1) - methane concentration in soil
          ! Flux from logarithmic layer theory, positive downwards

          fdiff = - kexch*0.5*( vv*(qsoil(1) + q_old(1)) - &
          & (qwater(gs%M+1,1) + qwater(gs%M+1,2)) )
        endif
      endif bottom_layer_if
    endif water_bottom_bc_if
  elseif (ls%l1 > 0._ireals) then
    ! Case of the soil and the ice above     
    ! Methane evolution in soil under the ice
    b(1) = dt*diff(2)/((z(2)-z(1))*(z(2)-z(1)))
    xx   = b(1) + 0.5*dt*(plant(1) + oxid(1))
    c(1) = 1. + xx
    ! No diffusive flux is assumed at the ice-soil interface
    f(1) = qsoil(1) + dt*(methgenmh(1) + rprod(1) + &
    & bull(1)*(ch4_crit(1) - qsoil(1))) + &
    & b(1)*qsoil(2) - xx*qsoil(1)
    do i = 2, gs%ns-1 !ms+1, ml-1
      ap(i) = dt/((z(i+1)-z(i-1))*(z(i+1)-z(i)))
      bp(i) = dt/((z(i+1)-z(i-1))*(z(i)-z(i-1)))
      a(i)  = bp(i) * diff(i)
      b(i)  = ap(i) * diff(i+1)
      xx    = ap(i)*diff(i+1) + bp(i)*diff(i) + 0.5*dt * (plant(i) + oxid(i))
      c(i)  = 1. + xx
      f(i)  = qsoil(i) + dt*(methgenmh(i) + rprod(i) + &
      & bull(i)*(ch4_crit(i) - qsoil(i))) + &
      & a(i)*qsoil(i-1) + b(i)*qsoil(i+1) - xx*qsoil(i)
    end do
    bp(gs%ns) = dt/( (z(gs%ns)-z(gs%ns-1))*(z(gs%ns)-z(gs%ns-1)) )
    a(gs%ns)  = bp(gs%ns) * diff(gs%ns)
    xx     = a(gs%ns) + 0.5 * dt * (plant(gs%ns) + oxid(gs%ns))
    c(gs%ns)  = 1. + xx
    f(gs%ns) = qsoil(gs%ns) + dt*(methgenmh(gs%ns) + rprod(gs%ns) + &
    & bull(gs%ns)*(ch4_crit(gs%ns) - qsoil(gs%ns))) + &
    & a(gs%ns)*qsoil(gs%ns-1) - xx*qsoil(gs%ns)
    call PROGONKA(nveclen,a, b, c, f, y, 1, gs%ns)
    y(1:gs%ns) = max(y(1:gs%ns),0.e0_ireals)
    qsoil(1:gs%ns) = y(1:gs%ns)     
  else ! bare soil
    print*, 'Bare soil in methane module is not functional: STOP'
    STOP
  endif water_exists_if
  if (.not. (bottom_bc == 2 .and. h1 > 0. .and. ls%ls1 == 0.)) then 
    ! otherwise the flux is calculated from logarithmic layer theory
    fdiff = - 0.5*lammeth(gs%M)*(qwater(gs%M+1,2) + qwater(gs%M+1,1) - &
    & qwater(gs%M,2) - qwater(gs%M,1))/(h1*ddz(gs%M)) ! Estimate, downwards
  endif
else
  ! Case of the soil column at the bottom slope
  if (ifflux) then !in this case qsoil_top is a downward methane flux
    b(1) = dt*diff(2)/((z(2)-z(1))*(z(2)-z(1)))
    xx   = b(1) + 0.5*dt*(plant(1) + oxid(1))
    c(1) = 1. + xx
    f(1) = qsoil(1) + dt*(methgenmh(1) + rprod(1) + &
    & bull(1)*(ch4_crit(1) - qsoil(1))) + &
    & b(1)*qsoil(2) - xx*qsoil(1) + &
    & 2.*qsoil_top*dt/(z(2)-z(1))
  else
    b(1) = 0._ireals
    c(1) = 1._ireals
    f(1) = qsoil_top ! qsoil must be prescribed outside
  endif
  do i = 2, gs%ns-1 !ms+1, ml-1
    ap(i) = dt/((z(i+1)-z(i-1))*(z(i+1)-z(i)))
    bp(i) = dt/((z(i+1)-z(i-1))*(z(i)-z(i-1)))
    a(i)  = bp(i) * diff(i)
    b(i)  = ap(i) * diff(i+1)
    xx    = ap(i)*diff(i+1) + bp(i)*diff(i) + 0.5*dt * (plant(i) + oxid(i))
    c(i)  = 1. + xx
    f(i)  = qsoil(i) + dt*(methgenmh(i) + rprod(i) + &
    & bull(i)*(ch4_crit(i) - qsoil(i))) + &
    & a(i)*qsoil(i-1) + b(i)*qsoil(i+1) - xx*qsoil(i)
  end do
  bp(gs%ns) = dt/( (z(gs%ns)-z(gs%ns-1))*(z(gs%ns)-z(gs%ns-1)) )
  a(gs%ns)  = bp(gs%ns) * diff(gs%ns)
  xx     = a(gs%ns) + 0.5 * dt * (plant(gs%ns) + oxid(gs%ns))
  c(gs%ns)  = 1. + xx
  f(gs%ns) = qsoil(gs%ns) + dt*(methgenmh(gs%ns) + rprod(gs%ns) + &
  & bull(gs%ns)*(ch4_crit(gs%ns) - qsoil(gs%ns))) + &
  & a(gs%ns)*qsoil(gs%ns-1) - xx*qsoil(gs%ns)
  call PROGONKA(nveclen,a, b, c, f, y, 1, gs%ns)
  y(1:gs%ns) = max(y(1:gs%ns),0.e0_ireals)
  qsoil(1:gs%ns) = y(1:gs%ns)
  if (.not.ifflux) then
    fdiff = 0.5*(qsoil(1) - q_old(1))/dt*( z(2) - z(1) ) - &
    & 0.5*diff(2)*(qsoil(2) + q_old(2) - qsoil(1) - q_old(1))/( z(2) - z(1) ) - &
    & 0.5*( z(2) - z(1) ) * ( methgenmh(1) + rprod(1) + &
    & bull(1)*(ch4_crit(1) - qsoil(1)) - &
    & 0.5*(qsoil(1) + q_old(1))*(plant(i) + oxid(i)) ) 
  endif
endif

!    ============================================================
!    the methane flux due to plant-mediated transport
!    =================fplant(t)==================================
!   do i = ms,ml
!    	qplant(i) = rkp*tveg*froot(i)*fgrow*
!  *                 (1.-0.5)*qsoil(i)
!   end do
fplant = 0.
if (soilswitch%par == 1) then
  do i = 2, gs%ns-1 !ms+1, ml-1
    fplant = fplant + 0.5*(z(i+1)-z(i-1))*plant(i)*0.5*(qsoil(i) + q_old(i)) * (1 - pox) !* veg
  end do
endif

ifdeep : if (deepestsoil) then

  !! Adding bubble flux from the lowest soil column to horizontally averaged bubble flux
  !if (gs%nsoilcols > 1) then
  !  call BUBBLEFLUXAVER(gs%ix,gs%iy,gs%nsoilcols)
  !else
  !  fbbleflx_ch4_sum(0)      = febul0*fbbleflx_ch4(   0)  *bathymwater(1)     %area_int 
  !  fbbleflx_ch4_sum(1:gs%M) = febul0*fbbleflx_ch4(1:gs%M)*bathymwater(1:gs%M)%area_half
  !  fbbleflx_ch4_sum(gs%M+1) = febul0*fbbleflx_ch4(gs%M+1)*bathymwater(gs%M+1)%area_int 
  !endif
 
  !The second step of splitting-up scheme - bubble dissolution source in water column
  if (h1 > 0.) then
    qwater(1,2) = qwater(1,2) + 2.d0*dt/(h1*ddz(1))*(fbbleflx_ch4_sum(1) - fbbleflx_ch4_sum(0)) / &
    & bathymwater(1)%area_int
    qwater(1,2) = max(qwater(1,2),small_value)
    do i = 2, gs%M
      qwater(i,2) = qwater(i,2) + dt/(h1*ddz05(i-1))*(fbbleflx_ch4_sum(i) - fbbleflx_ch4_sum(i-1)) / &
      & bathymwater(i)%area_int
      qwater(i,2) = max(qwater(i,2),small_value)
    enddo
    if (bottom_bc == 1) then
      qwater(gs%M+1,2) = qwater(gs%M+1,2) + 2.d0*dt/(h1*ddz(gs%M) + por(1)*(z(2) - z(1))) & ! porosity added
      & *(fbbleflx_ch4_sum(gs%M+1) - fbbleflx_ch4_sum(gs%M)) / bathymwater(gs%M+1)%area_int
      qwater(gs%M+1,2) = max(qwater(gs%M+1,2),small_value) !small_value
      qsoil(1) = por(1)*qwater(gs%M+1,2) !porosity added
    elseif (bottom_bc == 2) then
      qwater(gs%M+1,2) = qwater(gs%M+1,2) + 2.d0*dt/(h1*ddz(gs%M)) & 
      & *(fbbleflx_ch4_sum(gs%M+1) - fbbleflx_ch4_sum(gs%M)) / bathymwater(gs%M+1)%area_int
      qwater(gs%M+1,2) = max(qwater(gs%M+1,2),small_value) !small_value
    endif
  endif

  if (ls%l1 /= 0.) then
    ! A portion of methane ebullition flux (meth_ebul_ice_intercept) 
    ! is intercepted by the ice cover
    xx = fbbleflx_ch4_sum(0)/bathymwater(1)%area_int
    fbbleflx_ch4_sum(0) = fbbleflx_ch4_sum(0)*(1. - meth_ebul_ice_intercept)
    tot_ice_meth_bubbles = tot_ice_meth_bubbles + &
    & xx*ice_trap_bubbl_meth_decr*meth_ebul_ice_intercept*dt
  endif
  
  ice_meth_oxid_total = 0.
  add_to_winter = .false.
  if (ls%l1 == 0. .and. tot_ice_meth_bubbles /= 0.) then
    !When ice cover disappears, all ice-trapped bubbles are
    !released to the atmosphere instantaneously
    fbbleflx_ch4_sum(0) = fbbleflx_ch4_sum(0) + tot_ice_meth_bubbles/dt*bathymwater(1)%area_int
    ice_meth_oxid_total = tot_ice_meth_bubbles * &
    & (1.-ice_trap_bubbl_meth_decr)/ice_trap_bubbl_meth_decr
    tot_ice_meth_bubbles = 0.
    add_to_winter = .true.
  endif
 
endif ifdeep

!    ===============================================================
!  the diffusive flux at the soil/water boundary, downwards
!    =================fdiff(t, dzeta=1)==================================

! fdiff = - (diff(1) + diff(2))*0.5*(qsoil(1)-qsoil(2))/ & ! ms
! &         (z(2)-z(1))

!  fdiff = - 0.5*h1*ddz(M)/dt*(qwater(M+1,2)-qwater(M+1,1)) - &
!  & lammeth(M)*(qwater(M+1,2)-qwater(M,2))/(h1*ddz(M)) + &
!  & (dhw/dt - dhw0/dt)*0.5*(qwater(M+1,2)-qwater(M,2))

! xx = 0.5*(z(2)-z(1))
! fdiff = - xx*(qsoil(1) - q_old(1))/dt - &
! & diff(2)*(qsoil(2) - qsoil(1))*2./(z(2)-z(1)) + &
! & xx*( - qebul(1) - plant(1)*qsoil(1) + rprod(1) - oxid(1)*qsoil(1))

 if (soilswitch%par == 1) then    
   fdiff1 = - 0.5*diff(gs%ns)*(qsoil(gs%ns) + q_old(gs%ns) - qsoil(gs%ns-1) - q_old(gs%ns-1))/ & 
   &          (z(gs%ns) - z(gs%ns-1))
 else
   fdiff1 = 0.
 endif
 
!===============================================================
!    total methane emission ftot
!==============================================================
    
 if (deepestsoil) then 
   fdiff_lake_surf = Flux_atm !- lammeth(1)/(h1*ddz(1))*(qwater(2,2)-qwater(1,2))
   ftot = fdiff_lake_surf + fbbleflx_ch4_sum(0)/bathymwater(1)%area_int + fplant
 endif

 ! Water methane content
 q_sum_new = 0.
 do i = 1, gs%M+1
   q_sum_new = q_sum_new + qwater(i,2)*h1*ddz05(i-1)
 enddo

 if (soilswitch%par == 1) then
   !Diagnostic calculations of methane balance in sediments
   q_sum_new = 0.
   do i = 2, gs%ns-1 !ms+1, ml-1
     q_sum_new = q_sum_new + qsoil(i)*0.5*(z(i+1)-z(i-1))
   end do
   q_sum_new = q_sum_new + qsoil(gs%ns)*0.5*(z(gs%ns)-z(gs%ns-1)) 
   q_sum_new = q_sum_new + qsoil(1)*(0.5*(z(2)-z(1))) ! + 0.5*ddz(M)*h1/por(1))

   do i = 2, gs%ns-1 !ms+1, ml-1
     rprod_sum = rprod_sum + rprod(i)*dt*0.5*(z(i+1)-z(i-1))
     oxid_sum = oxid_sum + oxid(i)*(qsoil(i) + q_old(i))*dt*0.25*(z(i+1)-z(i-1))
     bull_sum = bull_sum + bull(i)*(q_old(i)-ch4_crit(i))*dt*0.5*(z(i+1)-z(i-1)) ! explicit scheme
     plant_sum = plant_sum + plant(i)*(qsoil(i) + q_old(i))*dt*0.25*(z(i+1)-z(i-1))
   end do
   
   rprod_sum = rprod_sum + rprod(gs%ns)*dt*0.5*(z(gs%ns)-z(gs%ns-1)) ! ml
   oxid_sum = oxid_sum + oxid(gs%ns)*(qsoil(gs%ns) + q_old(gs%ns))*dt*0.25*(z(gs%ns)-z(gs%ns-1))
   bull_sum = bull_sum + bull(gs%ns)*(q_old(gs%ns)-ch4_crit(gs%ns))*dt*0.5*(z(gs%ns)-z(gs%ns-1))
   plant_sum = plant_sum + plant(gs%ns)*(qsoil(gs%ns) + q_old(gs%ns))*dt*0.25*(z(gs%ns)-z(gs%ns-1))
   
   rprod_sum = rprod_sum + rprod(1)*dt*0.5*(z(2)-z(1))
   oxid_sum = oxid_sum + oxid(1)*(qsoil(1) + q_old(1))*dt*0.25*(z(2)-z(1))
   bull_sum = bull_sum + bull(1)*(q_old(1)-ch4_crit(1))*dt*0.5*(z(2)-z(1))
   plant_sum = plant_sum + plant(1)*(qsoil(1) + q_old(1))*dt*0.25*(z(2)-z(1))

 endif
 
 !if( (q_sum_new - q_sum_old).ne.0. .and. gs%isoilcol == gs%nsoilcols) then 
 !  xx = 0.
 !  do i = 1, gs%M+1
 !    xx = xx + 0.5*(qwater(i,1) + qwater(i,2))*ddz05(i-1)
 !  enddo
 !  yy = dhw*(0.5*(qwater(gs%M+1,2)+qwater(gs%M+1,1)) - xx) - &
 !  & dhw0*0.5*((qwater(gs%M+1,2)+qwater(gs%M+1,1)) - &
 !  &           (qwater(1     ,2)+qwater(1     ,1)))
 !  !Check the conservation of the scheme in soil
 !  !balance = (rprod_sum - oxid_sum - plant_sum - &
 !  !&  bull_sum + fdiff*dt - methox2sod*sodbot*dt - & 
 !  !&  (q_sum_new-q_sum_old)) !/(q_sum_new-q_sum_old)*100. ! percents
 !  balance = (rprod_sum + yy - oxid_sum - plant_sum - &
 !  &  bull_sum - Flux_atm*dt - methox2sod*sodbot*dt - & 
 !  &  (q_sum_new-q_sum_old)) !/(q_sum_new-q_sum_old)*100. ! percents
 !  write(*,*) 'balance = ', balance, &
 !  & 'rprod=', rprod_sum, &
 !  & 'oxid=' , -oxid_sum, &
 !  & 'plant=', -plant_sum, &
 !  & 'ebull=', -bull_sum, &
 !  & 'febul0=', -febul0*dt, &
 !  & 'fdiff=',  fdiff*dt, &
 !  & 'Flux_atm=',  -Flux_atm*dt, &
 !  & 'storage=', - (q_sum_new - q_sum_old)
 !  read*
 !endif


! fdiff = ( - rprod_sum + oxid_sum + plant_sum + &
! &  bull_sum + q_sum_new - q_sum_old)/dt


 deallocate (z)
 deallocate (roots)
 deallocate (pressoil)
 deallocate (a,b,c,f,y)
 
 nullify(qsoil,qwater,Twater,u,v)
 
 return
 END SUBROUTINE METHANE
 
 !> Subroutine METHANE_OXIC_PRODUCTION adds a tendency to methane water concentration 
 !! due to oxic (aerobic) methane production (OMP) following (Ordonez et al., Nature Comm., 2023).
 !! This parameterization provides OMP minus aerobic methane oxidation.
 SUBROUTINE METHANE_OXIC_PRODUCTION(dt,H_mixed_layer,extwat,Chl_a,methoxid,prodox,gs,gsp,gas)
 use UNITS, only : m3_to_l, day_to_sec
 use PHYS_CONSTANTS, only : k_PA
 use ARRAYS_GRID, only : gridsize_type, gridspacing_type
 use ARRAYS, only : gas_type
 use NUMERIC_PARAMS, only : small_value
 implicit none
 !Input variables
 real(kind=ireals)  , intent(in   ) :: dt              !< Time step, s
 real(kind=ireals)  , intent(in   ) :: H_mixed_layer   !< Mixed-layer depth, m
 real(kind=ireals)  , intent(in   ) :: extwat          !< Extinction coefficient for shortwave radiation, m**(-1)
 type(gridsize_type   ), intent(in) :: gs              !< Grid size structure
 type(gridspacing_type), intent(in) :: gsp             !< Grid spacing structure
 real(kind=ireals)  , intent(in   ) :: Chl_a(1:gs%M+1) !< Chlorophyll-a concentration, m
 real(kind=ireals)  , intent(in   ) :: methoxid(1:gs%M+1) !< Aerobic methane oxidation rate, mol/(m**3*s)
 real(kind=ireals)  , intent(in   ) :: prodox  (1:gs%M+1) !< Photosynthesis rate, mol/(m**3*s)
 !Input/output variables
 type(gas_type     ), intent(inout) :: gas             !< Structure with gas concentrations
 !Local variables
 real   (kind=ireals   ), parameter :: Clc = 2.5 !Multiplier in the light climate definition, n/d
 real   (kind=ireals   ), parameter :: slope = 0.01, offset = -0.14 !Linear regression parameters for OMP model
 real   (kind=ireals   ) :: light_climate !light climate
 real   (kind=ireals   ) :: Secchi_depth !Secchi disk depth, m
 real   (kind=ireals   ) :: tend !methane tendency
 real   (kind=ireals   ) :: x1 = 0., x2 = 0. !ancillary variables
 integer(kind=iintegers) :: i !loop index

 Secchi_depth = k_PA/extwat !Poole and Atkins formula
 light_climate = Clc*Secchi_depth/H_mixed_layer
 do i = 2, gs%M
   !OMP source is added only in the internal mesh points
   tend = (slope*light_climate*Chl_a(i)*m3_to_l*Secchi_depth + offset)*gas%qwater(i,2)/day_to_sec
   !print*, tend*day_to_sec/gas%qwater(i,2)
   if (Chl_a(i) > small_value) then
     gas%qwater(i,2) = gas%qwater(i,2) + dt*(tend - methoxid(i))
     x1 = x1 + (tend - methoxid(i))*gsp%ddz05(i-1)
     x2 = x2 + prodox(i)           *gsp%ddz05(i-1)
   endif
 enddo
 !print*, x2/x1
 
 END SUBROUTINE METHANE_OXIC_PRODUCTION
 
 SUBROUTINE METHANE_OXIDATION &
 & (M, i_maxN, dt, ddz, h1, bathymwater, oxyg, qwater, DIC, Tw, &
 & methoxid, qwateroxidtot, qwateroxidML)
 
 use METH_OXYG_CONSTANTS, only : Vmaxw, k_ch4, k_ch40, k_o2, koxyg
 use ARRAYS_BATHYM, only : bathym
 use PHYS_FUNC, only : &
 & REACPOT_ARRHEN 
     
 implicit none
 
!Input/output variables    
 integer(kind=iintegers), intent(in) :: M, i_maxN
 
 real(kind=ireals), intent(in) :: dt
 real(kind=ireals), intent(in) :: ddz(1:M)
 real(kind=ireals), intent(in) :: h1 ! Water depth, m

 type(bathym), intent(in) :: bathymwater(1:M+1)

 real(kind=ireals), intent(inout) :: oxyg(1:M+1)
 real(kind=ireals), intent(inout) :: qwater(1:M+1)
 real(kind=ireals), intent(inout) :: DIC(1:M+1)
 real(kind=ireals), intent(in) :: Tw(1:M+1)
 real(kind=ireals), intent(out) :: methoxid(1:M+1)
 real(kind=ireals), intent(out) :: qwateroxidtot, qwateroxidML
 
!Local variables and constants

 real(kind=ireals), parameter :: Kelvin0 = 273.15
 real(kind=ireals), parameter :: small_number = 1.d-20
 integer(kind=iintegers), parameter :: nox = 1 ! 1 - Michaelis-Menthen oxidation
                                  ! 2 - oxygen-saturated Michaelis-Menthen oxidation
                                  ! 3 - 1-st order kinetics

 integer(kind=iintegers) :: i
 real(kind=ireals) :: x, xo2, xch4, exc, D, temp_K ! Work variables
 real(kind=ireals) :: qwater_old, oxyg_old
 real(kind=ireals), allocatable :: qwaterold(:)
 logical, save :: firstcall = .true.

 allocate(qwaterold(1:M+1)) 
 qwaterold(:) = qwater(:)
 
 do i = 2, M

   select case (nox)
   case (1)
     ! Full Michaelis-Menten
     temp_K = Tw(i) + Kelvin0
     x = dt*Vmaxw & !REACPOT_ARRHEN(delta_Eq,temp_K,temp0,Vmaxw) & 
     & /(k_O2 + oxyg(i))/(k_CH4 + qwater(i))
     D = 1. + 2.*x*(2.*qwater(i) + oxyg(i)) + &
     & x*x*(2.*qwater(i) - oxyg(i))*(2.*qwater(i) - oxyg(i))
     oxyg_old = oxyg(i)
     oxyg(i) = ( - 1. - x*(2.*qwater(i) - oxyg(i))  + sqrt(D) ) * 0.5 / x
     qwater(i) = qwater(i)/(1. + x*oxyg_old)
     DIC(i) = DIC(i) - 0.5d0*(oxyg(i) - oxyg_old)
   case(2)
     ! Michaelis-Menten with methane concentrations only (oxygen-independent or oxygen-saturated)
     ! (implicit scheme)
     x = (k_ch40 + qwater(i))*(k_ch40 + qwater(i)) + &
     & 2.*(k_ch40 - qwater(i))*Vmaxw*dt + Vmaxw*dt*Vmaxw*dt
     qwater(i) = 0.5*( (qwater(i) - Vmaxw*dt - k_ch40) + sqrt(x) )
     x = dt*Vmaxw*qwater(i)/(k_ch40 + qwater(i))
     oxyg(i)   = oxyg(i)   - 2.*x
     DIC(i) = DIC(i) +    x
   case(3)
     ! Implicit scheme for the 1-st order kinetics
     qwater(i) = qwater(i)/(1. + dt*koxyg)
     oxyg(i) = oxyg(i) - 2.*dt*koxyg*qwater(i)
     DIC(i) = DIC(i) + dt*koxyg*qwater(i)
   end select
   
!  Ensuring postiveness of reactants (ch4 and 02)
   xo2 = 0.; xch4 = 0.
   if (oxyg(i) < 0.) xo2 = - oxyg(i)
   if (qwater(i) < 0.) xch4 = - qwater(i)
   if (xo2 > 0. .or. xch4 > 0.) then
     if (xo2/2. > xch4) then 
       exc = xo2/2.
     else
       exc = xch4
     endif
     oxyg(i) = oxyg(i) + exc*2.
     qwater(i) = qwater(i) + exc
     DIC(i) = DIC(i) - exc
   endif
   
 enddo  

 forall (i=2:M) methoxid(i) = (qwater(i) - qwaterold(i))/dt

 ! Total methane oxidation in a lake, normalized by surface area
 qwateroxidtot = 0.e0_ireals
 qwateroxidtot = qwateroxidtot + bathymwater(1)%area_int*0.5*ddz(1)*h1*methoxid(1)
 qwateroxidtot = qwateroxidtot + bathymwater(M+1)%area_int*0.5*ddz(M)*h1*methoxid(M+1)
 do i = 2, M
   qwateroxidtot = qwateroxidtot + bathymwater(i)%area_int*0.5*(ddz(i-1)+ddz(i))*h1*methoxid(i)
 enddo
 qwateroxidtot = qwateroxidtot/bathymwater(1)%area_int

 ! Total methane oxidation in mixed layer, normalized by surface area
 qwateroxidML = 0.e0_ireals
 qwateroxidML = qwateroxidML + bathymwater(1)%area_int*0.5*ddz(1)*h1*methoxid(1)
 if (i_maxN >= 2) then
   do i = 2, i_maxN
     qwateroxidML = qwateroxidML + bathymwater(i)%area_int*0.5*(ddz(i-1)+ddz(i))*h1*methoxid(i)
   enddo
 endif
 qwateroxidML = qwateroxidML/bathymwater(1)%area_int

 deallocate(qwaterold)

 if (firstcall) firstcall = .false. 
 END SUBROUTINE METHANE_OXIDATION


 
 FUNCTION MEAN_RPROD_OLDC(z1, z2, h_talik, C_talik_age)

!Function MEAN_RPROD_OLDC calculates mean of methane source due
!to "old" organics decomposition in the given interval [z1, z2], 
!using trapezium method for numerical integration
!The formulation is based on an assumption that organics
!from beneath the talik, decomposes according to
!first-order kinetics

 use METH_OXYG_CONSTANTS, only : &
 & alpha_old_org, r0_oldorg, C0_oldorg, r0_oldorg_star

 implicit none
 
 real(kind=ireals) :: MEAN_RPROD_OLDC
 
!Input/output variables

 real(kind=ireals), intent(in) :: z1, z2
 real(kind=ireals), intent(in) :: h_talik, C_talik_age
 
!Local parameters and variables
 integer(kind=iintegers), parameter :: N = 100
 
 real(kind=ireals) :: zz1, zz2, dz, dzeff
 
 integer(kind=iintegers) :: i ! Loop index
 
 MEAN_RPROD_OLDC = 0.
 zz1 = z1
 dz = (z2 - z1)/real(N)
 do i = 1, N
!  Trapezium method for integral calculation
   if (zz1 < h_talik) then
     zz2 = min(zz1 + dz, h_talik)
     dzeff = zz2 - zz1
     MEAN_RPROD_OLDC = MEAN_RPROD_OLDC + &  
     & (r0_oldorg_star*C0_oldorg*exp(-alpha_old_org*C_talik_age**(-2) * &
     & (h_talik**2 - zz1**2)) + &  
     &  r0_oldorg_star*C0_oldorg*exp(-alpha_old_org*C_talik_age**(-2) * &
     & (h_talik**2 - zz2**2)) ) * 0.5 * dzeff
!     MEAN_RPROD_OLDC = MEAN_RPROD_OLDC + &  
!     & (r0_oldorg*exp(alpha_old_org*C_talik_age**(-2) * &
!     & min(zz1,h_talik)**2) + &  
!     & r0_oldorg*exp(alpha_old_org*C_talik_age**(-2) * &
!     & min(zz2,h_talik)**2) ) * 0.5 * dz
   endif
   zz1 = zz1 + dz
 enddo
 
 MEAN_RPROD_OLDC = MEAN_RPROD_OLDC/(z2 - z1)
 
 END FUNCTION MEAN_RPROD_OLDC
 
 
 FUNCTION MEAN_RPROD_OLDC2(z1, z2, h_talik, C_talik_age)

!Function MEAN_RPROD_OLDC calculates mean of methane source due
!to "old" organics decomposition in the given interval [z1, z2], 
!using trapezium method for numerical integration.
!The formulation is based on an assumption that organics
!from beneath the talik, decomposes according to
!Michaelis-Menten kinetics. Approximate analytical solution is used
!(Stepanenko et al., 2011)

 use METH_OXYG_CONSTANTS, only : &
 & k_oldorg, V_oldorg, C0_oldorg, r0_oldorg2, r0_oldorg2_star

 implicit none
 
 real(kind=ireals) :: MEAN_RPROD_OLDC2
 
!Input/output variables

 real(kind=ireals), intent(in) :: z1, z2
 real(kind=ireals), intent(in) :: h_talik, C_talik_age
 
!Local parameters and variables
 integer(kind=iintegers), parameter :: N = 20

 real(kind=ireals), save :: lambda
 real(kind=ireals), save :: gamma
 real(kind=ireals), save :: MEAN_RPROD_OLDC2_min
     
 real(kind=ireals) :: zz1, zz2, dz, dzeff, x1, x2
 real(kind=ireals) :: gammaC
 
 integer(kind=iintegers) :: i ! Loop index

 logical, save :: firstcall = .true.

 if (firstcall) then
   lambda = C0_oldorg/k_oldorg
   gamma = V_oldorg/k_oldorg
   MEAN_RPROD_OLDC2_min = 1.d-15
 endif
 
 gammaC = 2.*gamma*C_talik_age**(-2)   
 MEAN_RPROD_OLDC2 = 0.
 zz1 = z1
 dz = (z2 - z1)/real(N)
 do i = 1, N
!  Trapezium method for integral calculation
   if (zz1 < h_talik) then
     zz2 = min(zz1 + dz, h_talik)
     dzeff = zz2 - zz1
     x1 = &
     & max(r0_oldorg2_star*C0_oldorg* &
     & (2. + lambda - sqrt( (1. + lambda)**2 + gammaC*(h_talik**2 - zz1**2)) ), &
     & MEAN_RPROD_OLDC2_min)
     x2 = &
     & max(r0_oldorg2_star*C0_oldorg* &
     & (2. + lambda - sqrt( (1. + lambda)**2 + gammaC*(h_talik**2 - zz2**2)) ), &
     & MEAN_RPROD_OLDC2_min)
     MEAN_RPROD_OLDC2 = MEAN_RPROD_OLDC2 + (x1 + x2) * 0.5 * dzeff
!     MEAN_RPROD_OLDC = MEAN_RPROD_OLDC + &  
!     & (r0_oldorg*exp(alpha_old_org*C_talik_age**(-2) * &
!     & min(zz1,h_talik)**2) + &  
!     & r0_oldorg*exp(alpha_old_org*C_talik_age**(-2) * &
!     & min(zz2,h_talik)**2) ) * 0.5 * dz
   endif
   zz1 = zz1 + dz
 enddo
 
 MEAN_RPROD_OLDC2 = MEAN_RPROD_OLDC2/(z2 - z1)
 
 if (firstcall) firstcall = .false.
 END FUNCTION MEAN_RPROD_OLDC2    


 FUNCTION MEAN_RPROD_NUM_OLDC2(z1, z2, h_talik, C_talik_age)

!Function MEAN_RPROD_OLDC calculates mean of methane source due
!to "old" organics decomposition in the given interval [z1, z2], 
!using trapezium method for numerical integration.
!The formulation is based on an assumption that organics
!from beneath the talik, decomposes according to
!Michaelis-Menten kinetics. Numerical solution is used


 use METH_OXYG_CONSTANTS, only : &
 & k_oldorg, V_oldorg, C0_oldorg, r0_oldorg2, r0_oldorg2_star

 implicit none
 
 real(kind=ireals) :: MEAN_RPROD_NUM_OLDC2
 
!Input/output variables

 real(kind=ireals), intent(in) :: z1, z2
 real(kind=ireals), intent(in) :: h_talik, C_talik_age
 
!Local parameters and variables
 integer(kind=iintegers), parameter :: N = 100

 real(kind=ireals), save :: lambda
 real(kind=ireals), save :: gamma
 real(kind=ireals), save :: MEAN_RPROD_NUM_OLDC2_min
     
 real(kind=ireals) :: zz1, zz2, dz, dzeff, x1, x2
 real(kind=ireals) :: gammaC
 
 integer(kind=iintegers) :: i ! Loop index

 logical, save :: firstcall = .true.

 if (firstcall) then
   lambda = C0_oldorg/k_oldorg
   gamma = V_oldorg/k_oldorg
   MEAN_RPROD_NUM_OLDC2_min = 1.d-15
 endif

 gammaC = 2.*gamma*C_talik_age**(-2)   
 MEAN_RPROD_NUM_OLDC2 = 0.
 zz1 = z1
 dz = (z2 - z1)/real(N)
 do i = 1, N
!  Trapezium method for integral calculation
   if (zz1 < h_talik) then
     zz2 = min(zz1 + dz, h_talik)
     dzeff = zz2 - zz1
     x1 = max(r0_oldorg2_star*RPROD_NUM_OLDC2(zz1), MEAN_RPROD_NUM_OLDC2_min)
     x2 = max(r0_oldorg2_star*RPROD_NUM_OLDC2(zz2), MEAN_RPROD_NUM_OLDC2_min)
     MEAN_RPROD_NUM_OLDC2 = MEAN_RPROD_NUM_OLDC2 + (x1 + x2) * 0.5 * dzeff
!     MEAN_RPROD_OLDC = MEAN_RPROD_OLDC + &  
!     & (r0_oldorg*exp(alpha_old_org*C_talik_age**(-2) * &
!     & min(zz1,h_talik)**2) + &  
!     & r0_oldorg*exp(alpha_old_org*C_talik_age**(-2) * &
!     & min(zz2,h_talik)**2) ) * 0.5 * dz
   endif
   zz1 = zz1 + dz
 enddo
 
 MEAN_RPROD_NUM_OLDC2 = MEAN_RPROD_NUM_OLDC2/(z2 - z1)
 
 
 if (firstcall) firstcall = .false.
 contains
 FUNCTION RPROD_NUM_OLDC2(z)
 
!Uses chord method to derive numerical solution of
!transcendental equation for concentration of 
!organics, arising from integration of Michaelis-Menten equation
 
 implicit none
 real(kind=ireals), intent(in) :: z
 real(kind=ireals), parameter :: conc_min = 1.d-15
 real(kind=ireals) :: conc_max
 real(kind=ireals), parameter :: resid_min = 1.d-13
 real(kind=ireals) :: RPROD_NUM_OLDC2
 real(kind=ireals) :: conc1, conc2, conc3
 real(kind=ireals) :: resid1, resid2, resid3
 integer :: iter

 conc_max = C0_oldorg
        
 conc1 = conc_min
 conc2 = conc_max
 resid3 = 1.
 iter = 0
 
 do while (abs(resid3) > resid_min)
   iter = iter + 1
   resid1 = k_oldorg*log(conc1/C0_oldorg) + conc1 - C0_oldorg + &
   & V_oldorg*C_talik_age**(-2)*(h_talik**2 - z**2)
   resid2 = k_oldorg*log(conc2/C0_oldorg) + conc2 - C0_oldorg + &
   & V_oldorg*C_talik_age**(-2)*(h_talik**2 - z**2)
   conc3 = (conc1*resid2 - conc2*resid1)/(resid2 - resid1)
   resid3 = k_oldorg*log(conc3/C0_oldorg) + conc3 - C0_oldorg + &
   & V_oldorg*C_talik_age**(-2)*(h_talik**2 - z**2)
   if (resid1 == 0.e0_ireals) then
     conc3 = conc1
     exit
   elseif (resid2 == 0.e0_ireals) then
     conc3 = conc2
     exit
   endif
   if (resid1*resid3 < 0.) then
     conc2 = conc3
   elseif (resid2*resid3 < 0.) then
     conc1 = conc3
   endif
 enddo
 
 RPROD_NUM_OLDC2 = conc3
 
 END FUNCTION RPROD_NUM_OLDC2
     
 END FUNCTION MEAN_RPROD_NUM_OLDC2    


 END MODULE METHANE_LAKE_MOD
