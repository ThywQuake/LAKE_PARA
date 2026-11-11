MODULE TURB_LAKE_MOD

use NUMERICS, only : PROGONKA, IND_STAB_FACT_DB
use NUMERIC_PARAMS, only : nveclen, small_value
use INOUT, only : CHECK_UNIT
use LAKE_DATATYPES, only : ireals, iintegers
use DRIVING_PARAMS, only : missing_value

contains
SUBROUTINE K_EPSILON(ix, iy, nx, ny, year, month, day, &
& hour, kor, a_veg, c_veg, h_veg, dt, &
& b0, tau_air, tau_wav, tau_i, tau_gr, roughness, fetch)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals
      
! KT_eq calculates eddy diffusivity (ED) in water coloumn following parameterizations:

! 1) empirical profile of ED;
! 2) Semi-empirical formulations for ED;
! 2) k-epsilon parameterization for ED, including Kolmogorov relation.

use COMPARAMS, only: &
& num_ompthr

use DRIVING_PARAMS , only : &
& nstep_keps, &
& turb_out, &
& Turbpar, &
& path, &
& stabfunc, &
& kepsbc, &
& kwe, &
& M, &
& omp, &
& eos, lindens, &
& zserout

use ATMOS, only : &
& uwind, vwind

use ARRAYS_TURB , only : &
& S_integr_positive, &
& S_integr_negative, &
& Gen_integr, &
& Seps_integr_positive, &
& Seps_integr_negative, &
& Geneps_integr, &
& epseps_integr, &
& eps_integr, &
& E_integr, &
& TKE_balance, &
& eps_balance, &
& Seps_integr_positive, &
& Seps_integr_negative, &
& Geneps_integr, &
& epseps_integr, &
& TKE_turb_trans_integr, &
& veg_sw, &
& E_it1, E_it2, &
& E1, E2, &
& eps_it1, eps_it2, &
& eps1, eps2, &
& k2_mid, k2, k2t, &
& E12, eps12, &
& row, row2, &
& k5_mid, k5, &
& L, &
& TF, &
& k3_mid, &
& WU_, WV_, &
& GAMT, GAMU, GAMV, &
& Gen, Gen_seiches, &
& S, &
& F, &
& KT, &
& TKE_turb_trans, &
& Re, &
& C1aup , &
& Ri , Ri_bulk, &
& k_turb_T_flux, &
& H_entrainment, &
& Buoyancy0, signwaveheight, &
& knum, &
& Eseiches, &
& TKE_budget_terms

use ARRAYS_GRID, only : &
& ddz, ddz2, ddz05, ddz052, ddz054, &
& dzeta_int, dzeta_05int, z_half

use ARRAYS_BATHYM, only : &
& dhw, dhw0, &
& area_int, area_half, l1, h1, &
& bathymwater, vol, botar, ls

use ARRAYS_WATERSTATE, only : &
& Tw1, Tw2, &
& Sal1, Sal2, lamw, lamsal

use ARRAYS, only : &
& um, u2, u1, &
& vm, v2, v1, &
& nstep, &
& time, &
& uv

use PHYS_CONSTANTS, only: &
& row0, roa0, &
& lamw0, &
& cw, &
& g, &
& cw_m_row0, &
& roughness0, niu_wat, &
& alsal, &
& Racr

use TURB_CONST

use PHYS_FUNC, only : &
& H13

use WATER_DENSITY, only: &
& water_dens_t_unesco, &
& water_dens_ts, &
& DDENS_DTEMP0, DDENS_DSAL0, &
& SET_DDENS_DTEMP, SET_DDENS_DSAL

use INOUT_PARAMETERS, only : &
& lake_subr_unit_min, &
& lake_subr_unit_max

use NUMERICS, only : STEP

use SEICHES_PARAM, only: SEICHE_ENERGY, C_Deff, cnorm

use LB_SAVED_LOCALS, only : init_keps => keps_init_keps ! lakebatch restart
implicit none

! Input variables

! Reals
real(kind=ireals), intent(in) :: dt
real(kind=ireals), intent(in) :: kor
real(kind=ireals), intent(in) :: h_veg, a_veg, c_veg
real(kind=ireals), intent(in) :: hour
real(kind=ireals), intent(in) :: b0
real(kind=ireals), intent(in) :: tau_air, tau_wav, tau_i, tau_gr
real(kind=ireals), intent(in) :: roughness
real(kind=ireals), intent(in) :: fetch


! Integers
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: nx, ny
integer(kind=iintegers), intent(in) :: year, month, day
        
! Local variables     
! Reals
real(kind=ireals) :: C_eps1(M), C_eps2(M), C_eps3(M)
real(kind=ireals) :: CE(M), CEt(M)
real(kind=ireals) :: lam_E(M+1), lam_eps(M+1)

!real(kind=ireals) :: a(nveclen)
!real(kind=ireals) :: b(nveclen)
!real(kind=ireals) :: c(nveclen)
!real(kind=ireals) :: d(nveclen)

real(kind=ireals), allocatable :: a(:), b(:), c(:), d(:) 

real(kind=ireals) :: AG(5)

real(kind=ireals) :: dt05
real(kind=ireals) :: wr

real(kind=ireals) :: dhw_keps, dhw0_keps
real(kind=ireals) :: pi
real(kind=ireals) :: ufr
real(kind=ireals) :: dist_surf
real(kind=ireals) :: ACC, ACC2, ACCk
real(kind=ireals) :: dE_it, deps_it
real(kind=ireals) :: xx(1:2), yy(1:2), zz
real(kind=ireals) :: lm
real(kind=ireals) :: ext_lamw
real(kind=ireals) :: month_sec, day_sec, hour_sec
real(kind=ireals) :: al_it
real(kind=ireals) :: maxTKEinput, maxTKEinput_i, maxTKEinput_gr
real(kind=ireals) :: FS, FTKES, TKES, DISS_TKES
real(kind=ireals) :: FB, FTKEB, TKEB, DISS_TKEB
real(kind=ireals) :: al, alft
real(kind=ireals) :: GAMUN
real(kind=ireals) :: urels, vrels
real(kind=ireals) :: ksurf1(1:2), ksurf2(1:2)
real(kind=ireals) :: Esurf1(1:2), Esurf2(1:2)
real(kind=ireals) :: ceps3
real(kind=ireals) :: vdamp
real(kind=ireals) :: dE_it_max, deps_it_max
real(kind=ireals) :: E_min, E_min_lamw0, E_min_Racr, eps_min, eps_min0
real(kind=ireals) :: Buoyancy1, dBdz0, z0, bvf2
real(kind=ireals) :: Eseiches_diss

real(kind=ireals) :: tf_integr
     
real(kind=ireals) :: GRADU, GRADV, GRADT, GAMVN
real(kind=ireals) :: TFR
real(kind=ireals) :: COGRTN, COGRUN, COGRVN
real(kind=ireals) :: DTDZH

real(kind=ireals), allocatable :: work(:,:)
real(kind=ireals), allocatable :: rhotemp(:), rhosal(:)

! Integers
integer(kind=iintegers), parameter :: maxiter = 35
integer(kind=iintegers) :: iter, iter_t
integer(kind=iintegers) :: nl
integer(kind=iintegers) :: i, j, k
integer(kind=iintegers) :: keps_coef
! integer(kind=iintegers) :: stabfunc
integer(kind=iintegers) :: time_deriv
integer(kind=iintegers) :: i_entrain, i_entrain_old
integer(kind=iintegers) :: nunit_ = lake_subr_unit_min

integer(kind=iintegers) :: grav_wave_Gill = 0
integer(kind=iintegers) :: seiches_Goudsmit = 0

! Logicals
! init_keps now lives in LB_SAVED_LOCALS (lakebatch restart)
logical :: indstab
logical :: ind_bound
logical :: iterat
logical :: firstcall
logical :: cycle_keps
logical :: next_tstep_keps
logical :: smooth
logical :: perdam
logical :: semi_implicit_scheme
logical :: contrgrad

! Characters
character :: tp_name*10
character :: numt*1


data firstcall /.true./

! Externals
real(kind=ireals), external :: DZETA

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_K_EPSILON)
include 'lb_ctx_k_epsilon.inc'


firstcall_if : if (firstcall) then

  if (turb_out%par == 1) then
    nunit_ = lbc_open_unit(path(1:len_trim(path))//'results/debug/err_keps_iter.dat')
    if (nunit_ < 0) then ! not yet opened by another point of this process
      nunit_ = lake_subr_unit_min ! CHECK_UNIT searches from here, as on a fresh start
      call CHECK_UNIT(lake_subr_unit_min,lake_subr_unit_max,nunit_)
      open (nunit_,file=path(1:len_trim(path))//'results/debug/err_keps_iter.dat', &
      & status = 'unknown')
      write(unit=nunit_,fmt=*) 'Errors of iterative process in k-epsilon solver'
    endif
  endif
  
  if (.not. allocated(init_keps)) then ! may already be restored (lakebatch restart)
    allocate (init_keps(1:nx, 1:ny) )
    init_keps(:,:) = .false.
  endif
       
  month_sec   = 30.*24.*60.*60.
  day_sec     = 24*60.*60.
  hour_sec    = 60.*60.

  pi          = 4.*atan(1.e0_ireals)

  AL    = g/row0
  ALFT  = 1.d0 
  
  ! Parameters of numerical scheme
  semi_implicit_scheme = .true. ! No iterations in k-epsilon, semi-implicit scheme
  al_it   = 0.2 !0.8
  ACC     = 1.d-20 !1.d-20
  ACC2    = 1.d-20 !1.d-20
  ACCk    = 1.d-20 !1.d-20
  knum    = 0.

  z0 = 1.d-2
  
  eps_min0    = 1.d-12 ! The range for dissipation rate values 
                       ! 10**(-11) - 10**(-6) m**2/s**3 (Wuest and Lorke, 2003)
  E_min_lamw0 = sqrt(eps_min0*1.E-5*lamw0*(cw_m_row0*CEt0)**(-1) ) ! This expression ensures that
                                                                   ! eddy diffusivity in the decaying
                                                                   ! turbulence (E -> Emin, eps -> eps_min)
                                                                   ! is much less than the molecular diffusivity
  dE_it_max = 1.d-11 !1.d-9
  deps_it_max = 1.d-15 !1.d-12

  smooth = .false.
  perdam = .false.
  vdamp  = 1.d-3 !1.d-3
 
  contrgrad = .false.
  ! AG(1)   = 0._ireals
  ! AG(2)   = 0._ireals
  ! AG(3)   = 0._ireals
  ! AG(4)=1. !0.
  ! AG(5)=1. !0.

  keps_coef = 1

  ! keps_coef = 1 - standard empirical coefficients in epsilon-equation
  ! keps_coef = 2 - the coefficient by (Aupoix et al., 1989) is used
    
  ! stabfunc = 1  - stability functions CE and CEt are constants
  ! stabfunc = 2  - stability functions CE and CEt are dependent 
  ! on shear and buoyancy (Canuto et al., 2001)
  ! stabfunc = 3  - stability functions CE and CEt are dependent
  ! only on buoyancy (Galperin et al., 1988)
         
  ! FRICTION: VEGETATION

  if (h_veg>0) then
    print*, 'Vegetation friction parameterization &
    & is not operational currently: STOP'
    STOP
  endif

  veg_sw = 0.
  do i = 1, M+1
    if (h1-dzeta_int(i)*h1 < h_veg) veg_sw(i) = 1.
  enddo

endif firstcall_if

allocate(a(1:nveclen), b(1:nveclen), &
&        c(1:nveclen), d(1:nveclen))

if (l1 > 0._ireals) then
  !The minimal values for E and epsilon are determined from two conditions:
  !1.corresponding turbulent diffusion coefficient is much less than molecular coefficient
  !2.free convection occurs in quiscent state under Ra>Racr
  eps_min = 1.E-5*(lamw0/cw_m_row0)**2*Racr*niu_wat/(maxval(ddz)*h1)**4
  E_min_Racr = eps_min*(maxval(ddz)*h1)**2*(CEt0*Racr*niu_wat*lamw0/cw_m_row0)**(-0.5)
  !E_min = max(E_min_lamw0,E_min_Racr)
  E_min = E_min_Racr
else
  eps_min = eps_min0
  E_min   = E_min_lamw0
endif

dt05 = 0.5d0*dt

allocate (work(1:M+1,1:2))
allocate (rhotemp(1:M), rhosal(1:M))

! Implementation of the Crank-Nicolson numerical scheme
! for k-epsilon parameterization using simple iterations to
! solve the set of nonlinear finite-difference equations

!  k_turb_T_flux used here from the previous time step

  ! Mixed layer defined from the gradient of heat turbulent flux
  !i_entrain = M
  !do i = M-1, 1, -1
  !  if (abs(k_turb_T_flux(i+1)-k_turb_T_flux(i))/abs(k_turb_T_flux(i+1) + 1.d-20) &
  !  & > 0.2 ) then ! 0.2 - quite arbitrary value 
  !    i_entrain = i
  !    exit
  !  endif
  !enddo

  ! Seiche energy evolution, TKE production due to seiche dissipation
  if (seiches_Goudsmit == 1) then
    call SEICHE_ENERGY &
    & (M,bathymwater,ls,tau_air-tau_wav,sqrt(uwind*uwind+vwind*vwind), &
    & vol,dt,Eseiches,Eseiches_diss)
    !A part of seiche energy dissipation feeding TKE (Goudsmit et al. 2002, eg. (24))
    do i = 1, M
      bvf2 = max(AL*(row(i+1) - row(i))/(h1*ddz(i)),0._ireals)
      Gen_seiches(i) = (1. - 10.*sqrt(C_Deff))*bvf2/(row0*cnorm*botar)*&
      & (area_int(i) - area_int(i+1))/area_half(i)*Eseiches_diss
    enddo
  else
    Gen_seiches(1:M) = 0.
  endif

  ! Mixed layer defined from TKE gradient
  i_entrain = minloc(E1(1:M),1)
  !do i = 1, M
  !  if (E1(i) < 1.E-6) then
  !    i_entrain = i
  !    exit
  !  endif
  !enddo

  !do i = i_entrain, 1, -1
  !  if (E1(i)/E1(i_entrain) > 5.E+0_ireals) then
  !    i_entrain = i
  !    exit
  !  endif
  !enddo

  H_entrainment = dzeta_05int(i_entrain) * h1

  ! Bulk Richardson number for mixed layer
  Ri_bulk = g/row0*(row(i_entrain)-row(1))*(dzeta_int(i_entrain)*h1)/ &
  & ( (u1(i_entrain) - u1(1))**2 + (v1(i_entrain) - v1(1))**2 + small_value)

  ! Significant wave height
  if (kepsbc%par == 2 .or. kepsbc%par == 3) then
    signwaveheight = H13(tau_air,fetch)
  endif

  dhw_keps  = dhw  / real(nstep_keps%par)
  dhw0_keps = dhw0 / real(nstep_keps%par)

  ! This time interpolation ensures correspondence between mean energy
  ! viscous dissipation in Crank-Nickolson scheme for momentum equation and TKE shear production
  ! when semi-implicit scheme is used (semi_implicit_scheme = .true.)
  do i = 1, M+1
    Um(i) = 0.5d0 * (u2(i) + u1(i))
    Vm(i) = 0.5d0 * (v2(i) + v1(i))
  enddo

  ! The TKE injection from waves breaking is limited
  ! by the wind kinetic energy input to waves development
  maxTKEinput = tau_wav/row0*sqrt(uwind*uwind + vwind*vwind)
  ! maxTKEinput_i = tau_i/row0*sqrt(Um(1)*Um(1)+Vm(1)*Vm(1))
  ! maxTKEinput_gr = tau_gr/row0*sqrt(Um(M+1)*Um(M+1)+Vm(M+1)*Vm(M+1))
      
  ! DTDZH=(Tw1(2)-Tw1(1))/(h1*ddz(1))
  
  next_tstep_keps = .false.
      
  keps_mode: do while (.not.next_tstep_keps)


  if (.not.init_keps(ix,iy)) then
    ! Initialization mode of k-epsilon parameterization: time derivatives are neglected  
    time_deriv = 0
  else
    ! Evolution mode of k-epsilon parameterization: full equations are solved
    time_deriv = 1
  endif
  
  do i = 1, M+1
    E_it1(i)   = E1(i)
    eps_it1(i) = eps1(i)
    k2_mid(i)  = 0._ireals
    k2(i)      = 0._ireals
    k2t(i)     = 0._ireals
  enddo

  iter_t = 0

  cycle_keps = .true.
  
  if (semi_implicit_scheme) then
    iterat = .false.
  else
    iterat = .true.
  endif  

  ! Setting density derivatives on temperature and salinity
  do i = 1, M
    rhotemp(i) = SET_DDENS_DTEMP(eos%par,lindens%par,0.5*(Tw1(i)+Tw1(i+1)), &
                                                     0.5*(Sal1(i)+Sal1(i+1)))
    rhosal(i)  = SET_DDENS_DSAL (eos%par,lindens%par,0.5*(Tw1(i)+Tw1(i+1)), &
                                                     0.5*(Sal1(i)+Sal1(i+1)))
  enddo

  iterat_keps: do while (cycle_keps)

    ! The cycle implements iterations to
    ! solve the set of nonlinear finite-difference equations
    ! of k-epsilon parameterization

    do i = 1, M+1
      E12(i)   = 0.5d0*(E1(i)   + E_it1(i))
      eps12(i) = 0.5d0*(eps1(i) + eps_it1(i))
    enddo
 
    ! E12   =  E_it1
    ! eps12 =  eps_it1
    
    ! Calculating the stability functions
    do i = 1, M
      if (stabfunc%par == 1) then
        lam_eps(i) = lam_eps0
        lam_E  (i) = lam_E0
        CE     (i) = CE0
        CEt    (i) = CEt0
      else
        work(i,1) = E12(i)*E12(i)/((eps12(i) + ACC2)*(eps12(i) + ACC2))* & 
        &        (rhotemp(i) * & ! Assuming Pr = Sc
        &        ( Tw1(i+1) - Tw1(i) ) + &
        &         alsal*rhosal(i) * &
        &        ( Sal1(i+1) - Sal1(i)) ) / &
        &        (h1*ddz(i)) *AL ! ~ bouyancy production of TKE
        work(i,2) = E12(i)*E12(i)/((eps12(i) + ACC2)*(eps12(i) + ACC2))* &
        & ( (U1(i+1) - U1(i))*(U1(i+1) - U1(i)) + &
        &   (V1(i+1) - V1(i))*(V1(i+1) - V1(i)) ) / &
        &   (h1*h1*ddz(i)*ddz(i)) ! ~ shear production of TKE
        if (stabfunc%par == 2) then
          CE(i)  = CE_CANUTO (work(i,2), work(i,1))
          CEt(i) = CEt_CANUTO(work(i,2), work(i,1))
        elseif (stabfunc%par == 3) then
          CE(i)  = sqrt(2.d0)*CL_K_KL_MODEL*SMOMENT_GALPERIN(work(i,1))
          CEt(i) = sqrt(2.d0)*CL_K_KL_MODEL*SHEAT_GALPERIN(work(i,1))
        endif
        lam_eps(i) = CE(i)/sigmaeps
        lam_E  (i) = CE(i)/sigmaE
      endif
    enddo

    ! Calculating eddy diffusivities for kinetic energy and its dissipation
    ! in the middle between half levels
    do i = 2, M
      ! E_mid=0.5d0*(E12(i-1)+E12(i))
      ! eps_mid=0.5d0*(eps12(i-1)+eps12(i))
      work(i,1) = INTERPOLATE_TO_INTEGER_POINT(E12(i-1),E12(i),ddz(i-1),ddz(i)) ! interpolated TKE
      work(i,2) = INTERPOLATE_TO_INTEGER_POINT(eps12(i-1),eps12(i),ddz(i-1),ddz(i)) ! interpolated dissipation
      xx(1) = INTERPOLATE_TO_INTEGER_POINT(lam_eps(i-1),lam_eps(i),ddz(i-1),ddz(i))
      xx(2) = INTERPOLATE_TO_INTEGER_POINT(lam_E(i-1),lam_E(i),ddz(i-1),ddz(i))
      if (iter_t == 0) then
        k2_mid(i) = work(i,1)*work(i,1)/(work(i,2) + ACCk)
      else
        k2_mid(i) = k2_mid(i)*al_it + work(i,1)*work(i,1)/(work(i,2) + ACCk)*(1-al_it)
      endif     
      k5_mid(i) = niu_wat + max(k2_mid(i)*xx(1), min_visc/sigmaeps)
      k3_mid(i) = niu_wat + max(k2_mid(i)*xx(2), min_visc/sigmaE)
    enddo
    do i = 1, M
      if (iter_t == 0) then
        k2(i)  = max(E12(i)*E12(i)/(eps12(i) + ACCk), min_visc/CE(i))
        k2t(i) = max(E12(i)*E12(i)/(eps12(i) + ACCk), min_diff/CEt(i))
      else
        k2(i)  = k2(i) *al_it + &
        & max(E12(i)*E12(i)/(eps12(i) + ACCk), min_visc/CE(i))*(1-al_it)
        k2t(i) = k2t(i)*al_it + &
        & max(E12(i)*E12(i)/(eps12(i) + ACCk), min_diff/CEt(i))*(1-al_it)
      endif
      k5(i) = niu_wat + k2(i)*lam_eps(i)
      l(i) = E12(i)**(1.5d0)/(eps12(i) + ACC)
      TF(i) = sqrt(E12(i))/(l(i) + ACC)
    enddo
    do i = 2, M-1
      WU_(i) = (E_it1(i+1)-E_it1(i-1))*(U2(i)-U2(i-1))/ &
      & (h1*h1*ddz052(i)*ddz(i-1) )
      WV_(i) = (E_it1(i+1)-E_it1(i-1))*(V2(i)-V2(i-1))/ &
      & (h1*h1*ddz052(i)*ddz(i-1) )
    enddo
 
    if (contrgrad) then
      do i = 2, M
        GRADT = (Tw1(i+1)-Tw1(i-1))/(h1*ddz2(i-1))
        GRADU = (U2(i)-U2(i-1))/(h1*ddz(i-1) )
        GRADV = (V2(i)-V2(i-1))/(h1*ddz(i-1) )
        GAMUN = - CONUV*(WU_(i+1)-WU_(i-1))/(h1*ddz2(i-1))
        GAMVN = - CONUV*(WV_(i+1)-WV_(i-1))/(h1*ddz2(i-1))
        COGRTN = DTDZH * 0.1
        TFR=E_it1(i)/(l(i)*l(i)+ACC) + acc
        COGRUN = GAMUN/TFR
        COGRVN = GAMVN/TFR 
        if(gradT < 0.) then
          GAMT(i) = - AG(3)*cogrtn
          GAMU(i) = - AG(1)*cogrun
          GAMV(i) = - AG(2)*cogrvn
        else
          GAMT(i) = - AG(3)*(AL*GRADT**2+CON0*TFR*COGRTN) / &
          &            (AL*GRADT+CON0*TFR)
          GAMU(i) = - AG(1)*(AL*(CON2*GRADT+(CON2-1.)*GAMT(i))*GRADU + &
          &            CON1*TFR*COGRUN)/(AL*GRADT+CON1*TFR)
          GAMV(i) = - AG(2)*(AL*(CON2*GRADT+(CON2-1.)*GAMT(i))*GRADV + &
          &            CON1*TFR*COGRVN)/(AL*GRADT+CON1*TFR)
        endif
      enddo
      GAMT(1)=0.
      GAMU(1)=0.
      GAMV(1)=0.
    else
      do i = 1, M+1
        GAMT(i) = 0.
        GAMU(i) = 0.
        GAMV(i) = 0.
      enddo
    endif

    do i = 1, M
      Gen(i) = ((Um(i+1) - Um(i))*(Um(i+1) - Um(i) + h1*ddz(i)*GAMU(i)) + &
      &        (Vm(i+1) - Vm(i))*(Vm(i+1) - Vm(i) + h1*ddz(i)*GAMV(i))) / &
      &        (h1**2*ddz(i)**2)*CE(i)
      ! S(i) = - ( 0.5d0*(row(i+1) + row2(i+1) - row(i) - row2(i)) / &
      ! &       (h1*ddz(i)) + GAMT(i))*ALFT*AL*CEt(i)
      S(i) = - ( 0.5d0*( rhotemp(i) * & ! Assuming Pr = Sc
      &        (Tw1(i+1) + Tw2(i+1) - Tw1(i) - Tw2(i)) + &
      &                  alsal*rhosal(i) * &
      &        (Sal1(i+1) + Sal2(i+1) - Sal1(i) - Sal2(i)) ) / &
      &        (h1*ddz(i)) + GAMT(i) )*ALFT*AL*CEt(i)
      !S(i) = max(S(i),0._ireals)
      F(i) = Gen(i) + S(i)
      ! Adding shear production of turbulence by gravitational waves in stable stratification
      Gen(i) = Gen(i) - grav_wave_Gill*grav_wave_Gill_const*STEP(-S(i))*S(i)*CE(i)/CEt(i) 
    enddo
    
    if (iter_t == 0) then
      Buoyancy1 = S(1)*k2t(1)
      dBdz0 = (Buoyancy1 - Buoyancy0)/(0.5*h1*ddz(1))
    endif
      
    ! Solution of the equation for TKE (turbulent kinetic energy)

    if_tkebc : if (kepsbc%par <= 4) then
      !Neumann boundary conditions for TKE
      !Boundary condition at the top
      if (l1 == 0._ireals) then
        !Open water
        if (kepsbc%par < 4) then
          FTKES = TKE_FLUX_SHEAR(tau_air,kwe%par,maxTKEinput,kepsbc%par)
        elseif (kepsbc%par == 4) then ! The new boundary condition, 
                                      ! implemented currently only for open water case
          FTKES = TKE_FLUX_BUOY(Buoyancy0,roughness0,H_entrainment)
        endif
      elseif (l1 > 0._ireals .and. l1 <= L0) then
        !Fractional ice
        FTKES =        b0*TKE_FLUX_SHEAR(tau_i,0._ireals,maxTKEinput,1) + &
        &    (1.d0-b0)*TKE_FLUX_SHEAR(tau_air,kwe%par,maxTKEinput,kepsbc%par)
      elseif (l1 > L0) then
        !Complete ice cover
        FTKES = TKE_FLUX_SHEAR(tau_i,0._ireals,maxTKEinput,1)
      endif
      xx(1) = 0.5*(area_int(2)/area_half(1)*k3_mid(2)*dt / &
      &  (h1**2*ddz(1)*ddz05(1)) + &
      &  time_deriv*dzeta_05int(1)*dhw_keps/(h1*ddz05(1) ) - &
      &  time_deriv*dhw0_keps/(h1*ddz05(1) ) )
      yy(1) = dt/(h1*ddz(1))
      b(1) = - xx(1)
      c(1) = - (xx(1) + time_deriv*1.d0) - (abs(S(1)) - S(1))/(TF(1) + ACC)*dt05 - TF(1)*dt
      d(1) = - time_deriv*E1(1) - xx(1)*(E1(2) - E1(1)) - &
      & area_int(1)/area_half(1)*yy(1)*FTKES - &
      & k2t(1)*(S(1) + abs(S(1)))*dt05 - dt*(k2(1)*Gen(1) + Gen_seiches(1))!+eps_it1(i)*dt
      ! Boundary condition at the bottom
      FTKEB = - TKE_FLUX_SHEAR(tau_gr,0._ireals,maxTKEinput,1)
      xx(2) = 0.5*(area_int(M)/area_half(M)*k3_mid(M)*dt / &
      & (h1**2*ddz(M)*ddz05(M-1) ) - &
      & time_deriv*dzeta_05int(M)*dhw_keps/(h1*ddz05(M-1) ) + &
      & time_deriv*dhw0_keps/(h1*ddz05(M-1) ) )
      yy(2) = dt/(h1*ddz(M))
      a(M) = - xx(2)
      c(M) = - (xx(2) + time_deriv*1.d0) - (abs(S(M)) - S(M))/(TF(M) + ACC)*dt05 - &
      & TF(M)*dt
      d(M) = - time_deriv*E1(M) - xx(2)*(E1(M-1) - E1(M)) + &
      & area_int(M+1)/area_half(M)*yy(2)*FTKEB - &
      & k2t(M)*(S(M) + abs(S(M)))*dt05 - dt*(k2(M)*Gen(M) + Gen_seiches(M)) !+eps_it1(i)*dt
    elseif (kepsbc%par == 5) then
      !Dirichlet boundary condition for TKE
      !Boundary condition at the top
      if (l1 == 0._ireals) then
        !Open water
        TKES = TKE_WALL_SHEAR(tau_air)
      elseif (l1 > 0._ireals .and. l1 <= L0) then
        !Fractional ice
        TKES = b0*TKE_WALL_SHEAR(tau_i) + (1.-b0)*TKE_WALL_SHEAR(tau_air)
      elseif (l1 > L0) then
        !Complete ice cover
        TKES = TKE_WALL_SHEAR(tau_i)
      endif
      b(1) = 0.
      c(1) = 1.
      d(1) = TKES
      ! Boundary condition at the bottom
      TKEB = TKE_WALL_SHEAR(tau_gr)
      a(M) = 0.
      c(M) = 1.
      d(M) = TKEB
    endif if_tkebc


    ! The coefficients of linear algebraic equations in the internal points of the mesh
    do i = 2, M-1
      a(i) = time_deriv*dzeta_05int(i)*dhw_keps/(ddz054(i)*h1) - &
      & time_deriv*dhw0_keps/(ddz054(i)*h1) - &
      & area_int(i)/area_half(i)*k3_mid(i)*dt/(h1**2*ddz(i)*ddz2(i-1) )

      b(i) = -time_deriv*dzeta_05int(i)*dhw_keps/(ddz054(i)*h1) + &
      & time_deriv*dhw0_keps/(ddz054(i)*h1) - &
      & area_int(i+1)/area_half(i)*k3_mid(i+1)*dt/(h1**2*ddz(i)*ddz2(i) )

      c(i) = a(i) + b(i) - time_deriv*1.d0-(abs(S(i))-S(i))/(TF(i)+ACC)*dt05 - TF(i)*dt

      d(i) = -time_deriv*E1(i)-k2t(i)*(S(i)+abs(S(i)))*dt05 - dt*(k2(i)*Gen(i) + Gen_seiches(i))
      d(i) = d(i)-dt*(h1**2*ddz(i)*ddz2(i))**(-1)* &
      & (area_int(i+1)/area_half(i)*k3_mid(i+1)*(E1(i+1)-E1(i)))   
      d(i) = d(i)+dt*(h1**2*ddz(i)*ddz2(i-1))**(-1)* &
      & (area_int(i)/area_half(i)*k3_mid(i)*(E1(i)-E1(i-1)))
      d(i) = d(i)-time_deriv*dzeta_05int(i)*dhw_keps* &
      & (ddz054(i)*h1)**(-1)*(E1(i+1)-E1(i-1)) 
      d(i) = d(i)+time_deriv*dhw0_keps* &
      & (ddz054(i)*h1)**(-1)*(E1(i+1)-E1(i-1)) 
    enddo
    ind_bound = .false.
    call IND_STAB_FACT_DB(nveclen,a,b,c,1,M,indstab,ind_bound)
    if (indstab .eqv. .false.) then
      do i = 2, M-1
        a(i) = - k3_mid(i)*dt/(h1**2*ddz(i)*ddz2(i-1) )
        b(i) = - k3_mid(i+1)*dt/(h1**2*ddz(i)*ddz2(i) )
      enddo
    endif
     
    call PROGONKA (nveclen,a,b,c,d,E_it2,1,M)
    E_it2(M+1) = 0.
                
    call CHECK_MIN_VALUE(E_it2, M, E_min) 

    dE_it = maxval(abs(E_it1-E_it2))


    ! C1 is given following Satyanarayana et al., 1999; Aupoix et al., 1989

    if (keps_coef == 1) then
      do i = 1, M
        yy(1) = 0.5*(1. + sign(1._ireals,S(i)))
        zz    = 0.5*(1. - sign(1._ireals,S(i)))
        ceps3 = ceps3_stable*zz + ceps3_unstable*yy(1)
        ! ceps3 = 1.14d0
        ! ceps3 < -0.4 should be used for stable stratification (Burchard, 2002)
        ! ceps3 = 1. ! following (Kochergin and Sklyar, 1992)
        ! ceps3 = 1.14d0 ! Baum and Capony (1992)
        ! Other values for ceps3:
        ! 0.<ceps3<0.29 for stable and ceps3 = 1.44 for unstable conditions (Rodi, 1987);
        ! ceps3 >= 1 (Kochergin and Sklyar, 1992)
        C_eps1(i) = ceps1
        C_eps2(i) = ceps2
        C_eps3(i) = ceps3
      enddo
    elseif (keps_coef == 2) then
      do i = 1, M 
        Re(i) = ACC + (2*E_it1(i)/3.)**2/(eps_it1(i)*niu_wat + ACC)
        C1aup(i) = Co/(1.d0 + 0.69d0*(2.d0 - Co)/sqrt(Re(i)))
        C_eps1(i) = C1aup(i)
        C_eps2(i) = C1aup(i)
        C_eps3(i) = C1aup(i)
      enddo  
    endif

    ! The solution of the equation for dissipation rate
    
    ! dist_surf is the distance from the surface, 
    ! where the b.c. for epsilon are formulated
    
    ! dist_surf = 0._ireals !0.5d0*ddz(1)*h1
    if_dissbc: if (kepsbc%par <= 4) then
      !Neumann boundary conditions for TKE dissipation
      !Boundary condition at the top
      if (l1 == 0._ireals) then
        if (kepsbc%par < 4) then
          ! FS = CE0**(0.75d0)*k5(1)*E_it1(1)**(1.5d0)/kar* &
          ! & (roughness0 + dist_surf)**(-2)
          ! FS = DISSIPATION_FLUX_SHEAR(tau_air,roughness0)

          ! Extrapolation of the turbulent diffusivity to the boundary level, assuming
          ! logarithmic profile
          ksurf1(1) = max(niu_wat, CE(1)*k2(1)-kar*sqrt(tau_air/row0)*0.5*ddz(1)*h1)
          
          ! ksurf1(1) = kar*sqrt(tau_air/row0)*z0 !roughness0
          ! ksurf1(1) = CE(1)*k2(1)
          Esurf1(1) = E_it1(1) + 0.5*FTKES*sigmaE*h1*ddz(1)/ksurf1(1)
          FS = DISSIPATION_FLUX_SHEAR(tau_air,ksurf1(1),Esurf1(1),z0, &
          & kwe%par,fetch,maxTKEinput,kepsbc%par,signwaveheight)
        elseif (kepsbc%par == 4) then ! The new boundary condition, 
                                      ! implemented currently only for open water case
          Esurf1(1) = E_it1(1)
          FS = DISSIPATION_FLUX_BUOY(Buoyancy0,roughness0,H_entrainment,dBdz0,Esurf1(1))
        endif
      elseif (l1 > 0._ireals .and. l1 <= L0) then
        ! FS = CE0**(0.75d0)*k5(1)*E_it1(1)**(1.5d0)/kar* &
        ! & (0.0._ireals + dist_surf)**(-2) !0.01 m - roughness of ice 
        ! FS =     b0*DISSIPATION_FLUX_SHEAR(tau_i,1.d-3) + & ! 1.d-3 roughness of ice
        ! & (1.d0-b0)*DISSIPATION_FLUX_SHEAR(tau_air,roughness0)

        ! Extrapolation of the turbulent diffusivity to the boundary level, assuming
        ! logarithmic profile
        ksurf1(1) = max(niu_wat, CE(1)*k2(1) - kar*sqrt(tau_air/row0)*0.5*ddz(1)*h1)
        ksurf2(1) = max(niu_wat, CE(1)*k2(1) - kar*sqrt(tau_i/row0)*0.5*ddz(1)*h1)
        
        ! ksurf1(1) = kar*sqrt(tau_air/row0)*z0 !*roughness0
        ! ksurf2(1) = kar*sqrt(tau_i/row0)*z0 !1.d-3 is the roughness of ice bottom
        ! ksurf1(1) = CE(1)*k2(1)
        ! ksurf2(1) = CE(1)*k2(1)
        Esurf1(1) = E_it1(1) + 0.5*FTKES*sigmaE*h1*ddz(1)/ksurf1(1)
        Esurf2(1) = E_it1(1) + 0.5*FTKES*sigmaE*h1*ddz(1)/ksurf2(1)
        FS =     b0*DISSIPATION_FLUX_SHEAR(tau_i,ksurf2(1),Esurf2(1),z0, &
        & 0._ireals,fetch,maxTKEinput,1,signwaveheight) + &
        & (1.d0-b0)*DISSIPATION_FLUX_SHEAR(tau_air,ksurf1(1),Esurf1(1),z0, &
        & kwe%par,fetch,maxTKEinput,kepsbc%par,signwaveheight)
      elseif (l1 > L0) then
        ! FS = DISSIPATION_FLUX_SHEAR(tau_i,1.d-3)

        ! Extrapolation of the turbulent diffusivity to the boundary level, assuming
        ! logarithmic profile
        ksurf2(1) = max(niu_wat, CE(1)*k2(1) - kar*sqrt(tau_i/row0)*0.5*ddz(1)*h1)
        
        ! ksurf2(1) = kar*sqrt(tau_i/row0)*z0 !1.d-3 is the roughness of ice bottom
        ! ksurf2(1) = CE(1)*k2(1)
        Esurf2(1) = E_it1(1) + 0.5*FTKES*sigmaE*h1*ddz(1)/ksurf2(1)
        FS = DISSIPATION_FLUX_SHEAR(tau_i,ksurf2(1),Esurf2(1),z0, &
        & 0._ireals,fetch,maxTKEinput,1,signwaveheight)
      endif 
      xx(1) = 0.5*(area_int(2)/area_half(1)*k5_mid(2)*dt / &
      & (h1**2*ddz(1)*ddz05(1)) + &
      & time_deriv*dzeta_05int(1)*dhw_keps/(h1*ddz05(1) ) - &
      & time_deriv*dhw0_keps/(h1*ddz05(1) ) )
      yy(1) = dt/(h1*ddz(1))
      b(1) = - xx(1)
      c(1) = - (xx(1) + time_deriv*1.d0) - C_eps2(1)*TF(1)*dt + &
      & C_eps3(1)*(- abs(S(1)) + S(1))/(TF(1) + ACC)*dt05
      d(1) = - time_deriv*eps1(1) - xx(1)*(eps1(2) - eps1(1)) - &
      & area_int(1)/area_half(1)*yy(1)*FS - C_eps3(1) * &
      & (abs(S(1)) + S(1))*TF(1)*k2t(1)*dt05 - &
      & C_eps1(1)*TF(1)*dt*(k2(1)*Gen(1) + Gen_seiches(1)) !+eps_it1(i)*dt
 
      ! Boundary condition at the bottom
      ! dist_surf = 0._ireals ! ddz(M)/2*h1
      
      ! FB = - CE0**(0.75d0)*k5(M)* &
      ! & E_it1(M)**(1.5d0)/kar*(0.0._ireals + dist_surf)**(-2) 
      ! FB = - DISSIPATION_FLUX_SHEAR(tau_gr,1.d-3) ! 1.d-3 m - rougnness of bottom
      
      ! Extrapolation of the turbulent diffusivity to the boundary level, assuming
      ! logarithmic profile
      ksurf2(2) = max(niu_wat, CE(M)*k2(M) - kar*sqrt(tau_gr/row0)*0.5*ddz(M)*h1)
 
      ! ksurf2(2) = kar*sqrt(tau_gr/row0)*z0 !1.d-3 is the roughness of bottom
      ! ksurf2(2) = CE(M)*k2(M)
      Esurf2(2) = E_it1(M) - 0.5*FTKEB*sigmaE*h1*ddz(M)/ksurf2(2)
      FB = - DISSIPATION_FLUX_SHEAR(tau_gr,ksurf2(2),Esurf2(2),z0, &
      & 0._ireals,fetch,maxTKEinput,1,signwaveheight)
      xx(2) = 0.5*(area_int(M)/area_half(M)*k5_mid(M)*dt / &
      & (h1**2*ddz(M)*ddz05(M-1) ) - &
      & time_deriv*dzeta_05int(M)*dhw_keps/(h1*ddz05(M-1) ) + &
      & time_deriv*dhw0_keps/(h1*ddz05(M-1) ))
      yy(2) = dt/(h1*ddz(M))
      a(M) = - xx(2)
      c(M) = - (xx(2) + time_deriv*1.d0) - C_eps2(M)*TF(M)*dt + &
      & C_eps3(M)*( - abs(S(M)) + S(M))/(TF(M) + ACC)*dt05
      d(M) = - time_deriv*eps1(M) - xx(2)*(eps1(M-1) - eps1(M)) + &
      & area_int(M+1)/area_half(M)*yy(2)*FB - C_eps3(M) * &
      & (abs(S(M)) + S(M))*TF(M)*k2t(M)*dt05 - &
      & C_eps1(M)*TF(M)*dt*(k2(M)*Gen(M) + Gen_seiches(M))
    elseif (kepsbc%par == 5) then
      !Dirichlet boundary conditions for TKE dissipation
      !Boundary condition at the top
      dist_surf = 0.5*ddz(1)*h1
      if (l1 == 0._ireals) then
        !Open water
        DISS_TKES = DISSIPATION_WALL_SHEAR(tau_air,Buoyancy0,dist_surf)
      elseif (l1 > 0._ireals .and. l1 <= L0) then
        !Fractional ice
        DISS_TKES = b0    *DISSIPATION_WALL_SHEAR(tau_i  ,Buoyancy0,dist_surf) + &
        &          (1.-b0)*DISSIPATION_WALL_SHEAR(tau_air,Buoyancy0,dist_surf)
      elseif (l1 > L0) then
        !Complete ice cover
        DISS_TKES = DISSIPATION_WALL_SHEAR(tau_air,Buoyancy0,dist_surf)
      endif
      b(1) = 0.
      c(1) = 1.
      d(1) = DISS_TKES
      !Boundary condition at the bottom
      dist_surf = 0.5*ddz(M)*h1
      DISS_TKEB = DISSIPATION_WALL_SHEAR(tau_gr,0._ireals,dist_surf)
      a(M) = 0.
      c(M) = 1.
      d(M) = DISS_TKEB
    endif if_dissbc

    do i = 2, M-1
      a(i) = time_deriv*dzeta_05int(i)*dhw_keps/(ddz054(i)*h1) - &
      & time_deriv*dhw0_keps/(ddz054(i)*h1) - &
      & area_int(i)/area_half(i)*k5_mid(i)*dt/(h1**2*ddz(i)*ddz2(i-1) )

      b(i) = -time_deriv*dzeta_05int(i)*dhw_keps/(ddz054(i)*h1) + &
      & time_deriv*dhw0_keps/(ddz054(i)*h1) - &
      & area_int(i+1)/area_half(i)*k5_mid(i+1)*dt/(h1**2*ddz(i)*ddz2(i) )

      c(i) = a(i) + b(i) - time_deriv*1.d0 - C_eps2(i)*TF(i)*dt + &
      & C_eps3(i)*(-abs(S(i))+S(i))/(TF(i)+ACC)*dt05

      d(i) = - time_deriv*eps1(i) - C_eps3(i)*(abs(S(i)) + S(i))*TF(i)*k2t(i)*dt05 - &
      & C_eps1(i)*TF(i)*dt*(k2(i)*Gen(i) + Gen_seiches(i))
      d(i) = d(i)-dt*(h1**2*ddz(i)*ddz2(i))**(-1)* &
      & (area_int(i+1)/area_half(i)*k5_mid(i+1)*(eps1(i+1)-eps1(i)))
      d(i) = d(i)+dt*(h1**2*ddz(i)*ddz2(i-1))**(-1)* &
      & (area_int(i)/area_half(i)*k5_mid(i)*(eps1(i)-eps1(i-1)))
      d(i) = d(i)-time_deriv*dzeta_05int(i)*dhw_keps* &
      & (ddz054(i)*h1)**(-1)*(eps1(i+1)-eps1(i-1))
      d(i) = d(i)+time_deriv*dhw0_keps* &
      & (ddz054(i)*h1)**(-1)*(eps1(i+1)-eps1(i-1))
    enddo
    ind_bound = .false.
    call IND_STAB_FACT_DB(nveclen,a,b,c,1,M,indstab,ind_bound)
    if (indstab .eqv. .false.) then
      do i = 2, M-1
        a(i) = - k5_mid(i)*dt/(h1**2*ddz(i)*ddz2(i-1))
        b(i) = - k5_mid(i+1)*dt/(h1**2*ddz(i)*ddz2(i))
      enddo
    endif
      
    call PROGONKA (nveclen,a,b,c,d,eps_it2,1,M)
    eps_it2(M+1) = 0.
 
    call CHECK_MIN_VALUE(eps_it2, M, eps_min)

    deps_it = maxval(abs(eps_it1-eps_it2))

    if (iterat) then
      if ((dE_it > dE_it_max .or. deps_it > deps_it_max) .and. iter_t < maxiter) then
        do i = 1, M+1
          E_it1(i)   = E_it1(i)   * al_it + E_it2(i)   * (1-al_it)
          eps_it1(i) = eps_it1(i) * al_it + eps_it2(i) * (1-al_it)
        enddo
        iter = iter + 1
        iter_t = iter_t + 1
        if (iter == 8) iter = 0
      else
        if (dE_it < 1.E-6 .and. deps_it < 1.E-7) then
          do i = 1, M+1
            E2(i) = E_it2(i)
            eps2(i) = eps_it2(i)
          enddo
          iter = 0
          iter_t = 0
          nl = 2
          cycle_keps = .false.
        else
          do i = 1, M+1
            eps_it1(i) = eps1(i)
            E_it1(i) = E1(i)
          enddo
          iterat = .false.
          iter_t = 0
          nl = 1
        endif
      endif
    else
      do i = 1, M+1
        E2(i) = E_it2(i)
        eps2(i) = eps_it2(i)
      enddo
      ! E2   = E_it1   * al_it + E_it2   * (1-al_it)
      ! eps2 = eps_it1 * al_it + eps_it2 * (1-al_it)
      iter = 0
      iter_t = 0
      cycle_keps = .false.
    endif
    ! if (dE_it<1.E-10.or.deps_it<1.E-13) nl=3
              
  enddo iterat_keps

  
  if (.not.init_keps(ix,iy)) then
    ! K-epsilon solver is implemented in initialization mode, i.e.
    ! to obtain initial profiles of E1 and eps1. 
    ! Time derivatives in k-epslilon equations are neglected in this case.
    do i = 1, M+1
      E1(i) = E2(i)
      eps1(i) = eps2(i)
    enddo
    init_keps(ix,iy) = .true.
  else
    ! The case when k-epsilon solver is implemented in evolution mode, i.e.
    ! E2 and epsilon2 are obtained at the next time step.
    next_tstep_keps = .true.
  endif

  enddo keps_mode

    
  ! Smoothing of the k and epsilon profiles
  if (smooth) then
    call VSMOP_LAKE(E2,E2,lbound(E2,1),ubound(E2,1),1,M,vdamp,perdam)
    call CHECK_MIN_VALUE(E2, M, E_min)
    call VSMOP_LAKE(eps2,eps2,lbound(eps2,1),ubound(eps2,1),1,M,vdamp,perdam)
    call CHECK_MIN_VALUE(eps2, M, eps_min)
  endif

  do i = 1, M+1
    E12(i) = 0.5d0*(E1(i) + E2(i))
    eps12(i) = 0.5d0*(eps1(i) + eps2(i))
  enddo

  if (stabfunc%par /= 1) then
    do i = 1, M
        work(i,1) = E2(i)*E2(i)/((eps2(i) + ACC2)*(eps2(i) + ACC2))* & 
        &        (rhotemp(i) * & ! Assuming Pr = Sc
        &        ( Tw2(i+1) - Tw2(i) ) + &
        &         alsal*rhosal(i) * &
        &        ( Sal2(i+1) - Sal2(i)) ) / &
        &        (h1*ddz(i)) *AL ! ~ bouyancy production of TKE
        work(i,2) = E2(i)*E2(i)/((eps2(i) + ACC2)*(eps2(i) + ACC2))* &
        & ( (U2(i+1) - U2(i))*(U2(i+1) - U2(i)) + &
        &   (V2(i+1) - V2(i))*(V2(i+1) - V2(i)) ) / &
        &   (h1*h1*ddz(i)*ddz(i)) ! ~ shear production of TKE
        if (stabfunc%par == 2) then
          CEt(i) = CEt_CANUTO(work(i,2), work(i,1))
        elseif (stabfunc%par == 3) then
          CEt(i) = sqrt(2.d0)*CL_K_KL_MODEL*SHEAT_GALPERIN(work(i,1))
        endif
    enddo
  endif

  do i = 1, M
    KT(i) = max(CEt(i)*E2(i)**2/(eps2(i) + ACCk),min_diff)*cw_m_row0 !E2, eps2
  enddo

  ! Diagnostic calculation
  !Turbulent transport of TKE
  do i = 2, M-1
    TKE_turb_trans(i) = 1.d0/(area_half(i)*h1*h1*ddz(i))* &
    & (area_int(i+1)*k3_mid(i+1)*0.5*(E2(i+1) + E1(i+1) - E2(i  ) - E1(i) ) / ddz05(i)  - &
    & area_int (i)  *k3_mid(i)  *0.5*(E2(i)   + E1(i)   - E2(i-1) - E1(i-1)) / ddz05(i-1) )
  enddo

  TKE_turb_trans(1) = 1.d0/(area_half(1)*h1*h1*ddz(1)) * &
  & (area_int(2)*k3_mid(2)*0.5*(E2(2) + E1(2) - E2(1) - E1(1)) / ddz05(1) + &
  &  area_int(1)*FTKES*h1 )
  TKE_turb_trans(M) = 1.d0/(area_half(M)*h1*h1*ddz(M))* &
  & ( - area_int(M+1)*FTKEB*h1  - &
  & area_int (M)  *k3_mid(M)*0.5 *(E2(M) + E1(M) - E2(M-1) - E1(M-1)) / ddz05(M-1) )

  ! Diagnostic calculations
  do i = 1, M
    S  (i) = 0.5*(S(i)+abs(S(i)))*k2t(i) + 0.5*(-abs(S(i))+S(i))/(TF(i)+ACC)*E2(i)
    Gen(i) = Gen(i)*k2(i)
  enddo

  ! Components of TKE budget for output
  if (zserout%par /= missing_value) then
    !i = minloc(abs(z_half(:) - zserout%par),1)
    i = M
    do 
      if (z_half(i) - zserout%par < 0.) exit
      i = i - 1
    enddo
    zz = (zserout%par - z_half(i))/(z_half(i+1) - z_half(i))
    TKE_budget_terms(1) = zz*( E2(i+1) - E1(i+1) )/dt + (1.-zz)*( E2(i) - E1(i) )/dt
    TKE_budget_terms(2) = zz*TKE_turb_trans(i+1) + (1.-zz)*TKE_turb_trans(i)
    TKE_budget_terms(3) = zz*Gen           (i+1) + (1.-zz)*Gen           (i)
    TKE_budget_terms(4) = zz*S             (i+1) + (1.-zz)*S             (i)
    TKE_budget_terms(5) = zz*E2(i+1)*TF(i+1) + (1.-zz)*E2(i)*TF(i)
    !print*, 'resid', TKE_budget_terms(1) - sum(TKE_budget_terms(2:4)) + TKE_budget_terms(5)
  endif
  
  S_integr_positive = 0._ireals
  S_integr_negative = 0._ireals
  Gen_integr = 0._ireals
  TKE_turb_trans_integr = 0._ireals
  
  Seps_integr_positive = 0._ireals
  Seps_integr_negative = 0._ireals
  Geneps_integr = 0._ireals
  epseps_integr = 0._ireals
  
  eps_integr = 0._ireals
  E_integr = 0._ireals
  TF_integr = 0._ireals

  do i = 1, i_entrain !M
    xx(1) = ddz(i)*h1
    yy(1) = 0.5*(1. + sign(1._ireals,S(i)))
    zz = 0.5*(1. - sign(1._ireals,S(i)))

    S_integr_positive = S_integr_positive + yy(1)*S(i)*xx(1)
    Seps_integr_positive = Seps_integr_positive + &
    & yy(1)*S(i)*C_eps3(i)*eps2(i)/E2(i)*xx(1)

    S_integr_negative = S_integr_negative + zz*S(i)*xx(1)
    Seps_integr_negative = Seps_integr_negative + &
    & zz*S(i)*C_eps3(i)*eps2(i)/E2(i)*xx(1)

    Gen_integr = Gen_integr + Gen(i)*xx(1)

    TKE_turb_trans_integr = TKE_turb_trans_integr + TKE_turb_trans(i)*xx(1)

    Geneps_integr = Geneps_integr + Gen(i)*C_eps1(i)*eps2(i)/E2(i)*xx(1)
    epseps_integr = epseps_integr + eps2(i)*C_eps2(i)*eps2(i)/E2(i)*xx(1)
    eps_integr = eps_integr + TF(i)*E2(i)*xx(1)
    E_integr = E_integr + E2(i)*xx(1)
    TF_integr = TF_integr + TF(i)*xx(1)
  enddo
  TKE_balance = Gen_integr + S_integr_positive + S_integr_negative - eps_integr
  eps_balance = Geneps_integr + Seps_integr_positive + Seps_integr_negative - epseps_integr
  
  ! Averaging
  S_integr_positive = S_integr_positive/H_entrainment
  S_integr_negative = S_integr_negative/H_entrainment
  Gen_integr = Gen_integr/H_entrainment
  TKE_turb_trans_integr = TKE_turb_trans_integr/H_entrainment

  Seps_integr_positive = Seps_integr_positive/H_entrainment 
  Seps_integr_negative = Seps_integr_negative/H_entrainment 
  Geneps_integr = Geneps_integr/H_entrainment 
  epseps_integr = epseps_integr/H_entrainment 
  
  eps_integr = eps_integr/H_entrainment 
  E_integr = E_integr/H_entrainment 
  TF_integr = TF_integr/H_entrainment 

  TKE_balance = TKE_balance/H_entrainment 
  eps_balance = eps_balance/H_entrainment   

  ! Debug output
  if (firstcall) then
    i_entrain_old = i_entrain
  else
    !if (i_entrain /= i_entrain_old) then
    !  write (1,'(i10,f12.2,6e15.5)') nstep, H_entrainment, eps_integr, TF_integr, FS, &
    !  & Seps_integr_positive, Seps_integr_negative, epseps_integr
    !  i_entrain_old = i_entrain
    !endif
  endif
  ! End debug output
  
  ! do i = 2, M-1
  !   work(i,1) = dhw_keps/(dt*h1)*dzeta_05int(i)*(E2(i+1)-E2(i-1))/ &
  !   & (0.5d0*ddz(i-1)+ddz(i)+0.5d0*ddz(i+1))/Buoyancy0
  !   work(i,2) = dhw0_keps/(dt*h1)*(E2(i+1)-E2(i-1))/ &
  !   & (0.5d0*ddz(i-1)+ddz(i)+0.5d0*ddz(i+1))/Buoyancy0
  ! enddo

  ! Output
  if (turb_out%par == 1) then
    if ((dE_it > 1.d-7 .or. deps_it > 1.d-9).and.(.not.semi_implicit_scheme)) then
      write (nunit_, *) nstep, nl, dE_it, deps_it
    endif
  endif

7 format (f10.4,f8.2,2f12.1,3f8.2)
8 format (f10.4,2f12.1,f8.2)      

  deallocate (work)
  deallocate (rhotemp, rhosal)
  deallocate (a, b, c, d)
    
  do i = 1, M+1     
    E1(i) = E2(i)
    eps1(i) = eps2(i)
  enddo
 
  !Recalculation for output
  do i = 1, M
    k2(i) = k2(i)*CE(i)
  enddo
     
firstcall = .false.

END SUBROUTINE K_EPSILON


SUBROUTINE CHECK_MIN_VALUE(E,N,threshold)
implicit none

! Input variables
! Integers
integer(kind=iintegers), intent(in) :: N
! Reals
real(kind=ireals), intent(in) :: threshold

! Input/output variables
real(kind=ireals), intent(inout) :: E(1:N)

integer(kind=iintegers) :: i ! Loop index

do i = 1, N 
  E(i) = max(E(i),threshold)
enddo
END SUBROUTINE CHECK_MIN_VALUE


real(kind=ireals) FUNCTION CE_CANUTO(lambda_M, lambda_N)

! The function CE_CANUTO calculates stability function for momentum
! according to Canuto et al., 2001

use TURB_CONST, only: &
& CEcoef01, &
& CEcoef02, &
& CEcoef03, &

& CEcoef11, &
& CEcoef12, &
& CEcoef13, &
& CEcoef14, &
& CEcoef15, &
& CEcoef16

implicit none

! Upper limit for stability function for momentum (Galperin et al., 1988)
real(kind=ireals), parameter :: upper_limit = 0.46d0 
real(kind=ireals), parameter :: small_number = 1.d-20 
real(kind=ireals), parameter :: lambda_N_low_limit = - 5.9d0

real(kind=ireals), intent(in) :: lambda_M 
real(kind=ireals), intent(in) :: lambda_N

real(kind=ireals) :: lambda_N1

! Putting lower limit for lambda_N, that is defined from the condition
! d(CE_CANUTO)/d(lambda_N)<0

lambda_N1 = max(lambda_N_low_limit,lambda_N)

CE_CANUTO = &
& (CEcoef01 + CEcoef02*lambda_N1 - CEcoef03*lambda_M)/ &
& (CEcoef11 + CEcoef12*lambda_N1 + CEcoef13*lambda_M + &
&  CEcoef14*lambda_N1*lambda_N1 + CEcoef15*lambda_N1*lambda_M - & 
&  CEcoef16*lambda_M*lambda_M)

CE_CANUTO = max(small_number,CE_CANUTO)
CE_CANUTO = min(upper_limit,CE_CANUTO)

END FUNCTION CE_CANUTO



real(kind=ireals) FUNCTION CEt_CANUTO(lambda_M, lambda_N)

! The function CEt_CANUTO calculates stability function for scalars
! according to Canuto et al., 2001

use TURB_CONST , only: &
& CEtcoef01, &
& CEtcoef02, &
& CEtcoef03, &

& CEcoef11, &
& CEcoef12, &
& CEcoef13, &
& CEcoef14, &
& CEcoef15, &
& CEcoef16

implicit none
! Upper limit for stability function for scalars (Galperin et al., 1988)
real(kind=ireals), parameter:: upper_limit = 0.61d0
real(kind=ireals), parameter:: small_number = 1.d-20 
real(kind=ireals), parameter:: lambda_N_low_limit = - 5.9d0   

real(kind=ireals), intent(in):: lambda_M
real(kind=ireals), intent(in):: lambda_N
 
real(kind=ireals) lambda_N1
 
! Putting lower limit for lambda_N, that is defined from the condition
! d(CEt_CANUTO)/d(lambda_N)<0 
lambda_N1 = max(lambda_N,lambda_N_low_limit)

CEt_CANUTO = &
& (CEtcoef01 + CEtcoef02*lambda_N1 + CEtcoef03*lambda_M) / &
& (CEcoef11 + CEcoef12*lambda_N1 + CEcoef13*lambda_M + &
& CEcoef14*lambda_N1**2 + CEcoef15*lambda_N1*lambda_M - &
& CEcoef16*lambda_M**2)

CEt_CANUTO = max(small_number,CEt_CANUTO)
CEt_CANUTO = min(upper_limit,CEt_CANUTO)
 
END FUNCTION CEt_CANUTO
      
      
real(kind=ireals) FUNCTION SMOMENT_GALPERIN(lambda_N)
 
! The function SMOMENT_GALPERIN calculates the equilibrium stability function
! for momentum according to Galperin et al., 1988
 
use TURB_CONST!, only: &
!& A1, &
!& B1, &
!& C1, &
!& A2, &
!& B2, &
!& CL_K_KL_MODEL 

implicit none

! Upper limit for stability function (Galperin et al., 1988) 
! real(kind=ireals), parameter:: upper_limit = 0.46d0 ! This value is in terms of CE
real(kind=ireals), parameter:: small_number = 1.d-2
! Below this limit for lambda_N stability function has negative values    
!real(kind=ireals), parameter:: lambda_N_lower_limit = - 1.8d0 
real(kind=ireals), parameter:: GH_MAX = (A2*(12.d0*A1 + B1 + 3.d0*B2))**(-1) ! Galperin et al., 1988
real(kind=ireals), parameter:: GH_MIN = - 0.28d0 ! Galperin et al., 1988
 
real(kind=ireals), intent(in):: lambda_N

real(kind=ireals) :: GH
 
GH = - lambda_N ! According to Mellor and Yamada, 1982 GH has the opposite sign to lambda_N
! The scaling of lambda_N according to Mellor and Yamada, 1982 
GH = CL_K_KL_MODEL**2*0.5d0*GH
! Implementing limitations Galperin et al., 1988
GH = min(GH,GH_MAX)
GH = max(GH,GH_MIN)
 
SMOMENT_GALPERIN = 1.d0 - 3.d0 * C1 - 6.d0 * A1 / B1 - &
& 3.d0 * A2 * GH * ( (B2 - 3.d0*A2) * (1.d0 - 6.d0*A1 / B1) - &
& 3.d0 * C1 * (B2 + 6.d0 * A1) )
SMOMENT_GALPERIN = SMOMENT_GALPERIN / &
& ( (1.d0 - 3.d0 * A2 * GH * (6.d0 * A1 + B2) ) * &
&   (1.d0 - 9.d0 * A1 * A2 * GH) )
SMOMENT_GALPERIN = SMOMENT_GALPERIN * A1
 
SMOMENT_GALPERIN = max(small_number,SMOMENT_GALPERIN)
!SMOMENT_GALPERIN = min(upper_limit, SMOMENT_GALPERIN)

END FUNCTION SMOMENT_GALPERIN
      
      
real(kind=ireals) FUNCTION SHEAT_GALPERIN(lambda_N)
 
! The function SHEAT_GALPERIN calculates the equilibrium stability function
! for heat according to Galperin et al., 1988
 
use TURB_CONST!, only: &
!& A1, &
!& B1, &
!& C1, &
!& A2, &
!& B2, &
!& CL_K_KL_MODEL

implicit none

! Upper limit for stability function (Galperin et al., 1988)
! real(kind=ireals), parameter:: upper_limit = 0.61d0 ! This value is in terms of CEt
real(kind=ireals), parameter:: small_number = 1.d-2 
! Below this limit for lambda_N stability function has negative values
! real(kind=ireals), parameter:: lambda_N_lower_limit = - 1.8d0 
real(kind=ireals), parameter:: GH_MAX = (A2*(12.d0*A1 + B1 + 3.d0*B2))**(-1) ! Galperin et al., 1988
real(kind=ireals), parameter:: GH_MIN = - 0.28d0 ! Galperin et al., 1988

real(kind=ireals), intent(in):: lambda_N
 
real(kind=ireals) :: GH
 
! lambda_N1 = - max(lambda_N,lambda_N_lower_limit)
! The scaling of lambda_N according to Mellor and Yamada, 1982
GH = - lambda_N ! According to Mellor and Yamada, 1982 GH has the opposite sign to lambda_N
! The scaling of lambda_N according to Mellor and Yamada, 1982 
GH = CL_K_KL_MODEL**2*0.5d0*GH
! Implementing limitations Galperin et al., 1988
GH = min(GH,GH_MAX)
GH = max(GH,GH_MIN)
 
SHEAT_GALPERIN = 1.d0 - 6.d0 * A1 / B1
SHEAT_GALPERIN = SHEAT_GALPERIN / &
& (1.d0 - 3.d0 * A2 * GH * (6.d0 * A1 + B2) )
SHEAT_GALPERIN = SHEAT_GALPERIN * A2
 
SHEAT_GALPERIN = max(small_number,SHEAT_GALPERIN)
! SHEAT_GALPERIN = min(upper_limit, SHEAT_GALPERIN)
 
END FUNCTION SHEAT_GALPERIN


SUBROUTINE VSMOP_LAKE(f,fs,nmin,nmax,k0,k1,gama,perdam)

! only the perturbation is smoothed (if perdam is .true.)
!
! 4th-order vertical smoothing. 2nd order for grid-points adjacent
! to the boundary. no smoothing for boundary points.
!
! the second order coeficient is calculated from the fourth order one,
! imposing similar damping for 2 grid-length waves
!
! fourth order damping: response function
! r(lx,ly=)1-16*gama*sin(pi*dx/lx)**4

implicit none

! Input variables
! Integers
integer(kind=iintegers), intent(in) :: nmin
integer(kind=iintegers), intent(in) :: nmax
integer(kind=iintegers), intent(in) :: k0
integer(kind=iintegers), intent(in) :: k1

! Reals
real(kind=ireals), intent(in) :: fs(nmin:nmax)
real(kind=ireals), intent(in) :: gama

! Logicals
logical, intent(in) :: perdam

! Input/output variables
! Reals
real(kind=ireals), intent(inout) :: f(nmin:nmax)

! Local variables
! Reals
real(kind=ireals) :: work(nmin:nmax)
real(kind=ireals) :: um6g
real(kind=ireals) :: gasc
real(kind=ireals) :: gasc05
real(kind=ireals) :: umgasc

! Integers
integer(kind=iintegers) :: i ! Loop index

um6g = 1. - 6.*gama
gasc = 8.*gama
gasc05 = 0.5*gasc
umgasc = 1. - gasc

if (perdam) then
  do i = k0, k1
    f(i) = f(i) - fs(i)
  enddo
endif

do i = k0+2, k1-2
  work(i) = um6g*f(i) - gama*(f(i+2) + f(i-2) - 4.*(f(i+1) + f(i-1)) )
enddo

do i = k0+1, k1-1, k1-k0-2
  work(i) = umgasc*f(i) + gasc05*(f(i-1) + f(i+1))
enddo

do i = k0+1, k1-1
  f(i) = work(i)
enddo

if (perdam) then
  do i = k0, k1
    f(i) = f(i) + fs(i)
  enddo
endif

RETURN
END SUBROUTINE VSMOP_LAKE


FUNCTION INTERPOLATE_TO_INTEGER_POINT(fim05,fip05,ddzim1,ddzi)

! Interpolates linearly the grid function 
! from two half-integer points to integer point between them

implicit none

real(kind=ireals) :: INTERPOLATE_TO_INTEGER_POINT

! Input variables
real(kind=ireals), intent(in) :: fim05  ! The function at (i-0.5)-th level
real(kind=ireals), intent(in) :: fip05  ! The function at (i+0.5)-th level
real(kind=ireals), intent(in) :: ddzim1 ! The thickness of (i-1)-th layer
real(kind=ireals), intent(in) :: ddzi   ! The thickness of i-th layer

INTERPOLATE_TO_INTEGER_POINT = fim05 + (fip05 - fim05)*ddzim1/(ddzim1 + ddzi)

END FUNCTION INTERPOLATE_TO_INTEGER_POINT


FUNCTION TKE_FLUX_BUOY(Buoyancy_flux,roughness_length,H_entrainment)

! The function TKE_FLUX_BUOY calculates the flux of turbulent kinetic energy
! at the surface in the case of free convection, i.e. no wind stress but buoyancy flux
! given at the surface (Stepanenko and Lykosov, 2009).

use TURB_CONST

real(kind=ireals) :: TKE_FLUX_BUOY

! Input variables
real(kind=ireals), intent(in) :: Buoyancy_flux
real(kind=ireals), intent(in) :: roughness_length
real(kind=ireals), intent(in) :: H_entrainment

! Local variables
real(kind=ireals), save :: const

logical, save :: firstcall = .true.

if (firstcall) then
  const = 2.d0/3.d0*CE0*kar*kar/(sigmaE*CL*CL)
endif

!TKE_FLUX_BUOY = - const*Buoyancy_flux*roughness_length * &
!&               (1.d0-2.d0*roughness_length/H_entrainment)

TKE_FLUX_BUOY = 0._ireals

if (firstcall) firstcall = .false.
END FUNCTION TKE_FLUX_BUOY


FUNCTION DISSIPATION_FLUX_BUOY(Buoyancy_flux,roughness_length,H_entrainment, dBdz0, TKE0)

! The function DISSIPATION_FLUX_BUOY calculates the flux of turbulent kinetic energy
! dissipation rate at the surface in the case of free convection, 
! i.e. no wind stress but buoyancy flux given at the surface (Stepanenko and Lykosov, 2009).

use TURB_CONST

real(kind=ireals) :: DISSIPATION_FLUX_BUOY

! Input variables
real(kind=ireals), intent(in) :: Buoyancy_flux
real(kind=ireals), intent(in) :: roughness_length
real(kind=ireals), intent(in) :: H_entrainment
real(kind=ireals), intent(in) :: dBdz0 ! The vertical gradient of Buoyancy flux at the water surface
real(kind=ireals), intent(in) :: TKE0 ! The turbulent kinetic energy at the water surface

! Local variables
real(kind=ireals), save :: const
real(kind=ireals) :: wstar

logical, save :: firstcall = .true.

if (firstcall) then
  const = CE0*(kar/CL)**(4.d0/3.d0)/sigmaeps
endif

! Flux obtained neglecting diffusion term in TKE equation
! DISSIPATION_FLUX_BUOY = const*(Buoyancy_flux*roughness_length)**(4.d0/3.d0) / &
! &                       H_entrainment

! Flux obtained including diffusion term in TKE equation
!wstar = (Buoyancy_flux*H_entrainment)**(1.d0/3.d0)
!DISSIPATION_FLUX_BUOY = CE0*C_wstar*C_wstar*Buoyancy_flux*wstar/sigmaeps

DISSIPATION_FLUX_BUOY = - CE0*TKE0*TKE0*dBdz0/((Buoyancy_flux+small_value)*sigmaeps)

if (firstcall) firstcall = .false.
END FUNCTION DISSIPATION_FLUX_BUOY


FUNCTION TKE_FLUX_SHEAR(momentum_flux,kwe,maxTKEinput,nbc)

! The function TKE_FLUX_SHEAR calculates the flux of turbulent kinetic energy
! at the surface in the case of shear lorarithmic flow, i.e. no buoyancy flux but wind stress
! given at the surface (Burchard and Petersen, 1999).

use PHYS_CONSTANTS, only : &
& row0

use TURB_CONST, alpham_=>alpham!, only : &
!& kar, sigmaE, &
!& CE0, kar_tilde

implicit none

real(kind=ireals) :: TKE_FLUX_SHEAR

! Input variables
real(kind=ireals), intent(in) :: momentum_flux
real(kind=ireals), intent(in) :: kwe
real(kind=ireals), intent(in) :: maxTKEinput

integer(kind=iintegers), intent(in) :: nbc

!Local variables
real(kind=ireals), parameter :: alpham = (1.5*CE0**0.5*sigmaE*kar_tilde**(-2))**(0.5)
real(kind=ireals), parameter :: const0 = (1.5*sigmaE)**0.5*CE0**0.25
real(kind=ireals), parameter :: const = 2./3.*CE0**(-0.5)*kar*const0*alpham/sigmaE

select case (nbc)
  case(1)
    TKE_FLUX_SHEAR = 0._ireals ! for logarithmic profile (shear only)  
  case(2)
    ! TKE_FLUX_SHEAR = min(kwe*(momentum_flux/row0)**1.5d0,maxTKEinput) ! for breaking waves case 
                                                                      ! Craig and Banner (1994)
    TKE_FLUX_SHEAR = kwe*(momentum_flux/row0)**1.5d0
  case(3)
    ! Generalized b.c. for sheared flow with wave breaking (Burchard, 2002)
    TKE_FLUX_SHEAR = min(const*kwe*(momentum_flux/row0)**1.5d0,maxTKEinput)
end select

END FUNCTION TKE_FLUX_SHEAR


FUNCTION DISSIPATION_FLUX_SHEAR(momentum_flux,kturb,Esurf_in,z0_in, &
& kwe,fetch,maxTKEinput,nbc,signwaveheight)

! The function DISSIPATION_FLUX_SHEAR calculates the flux of turbulent kinetic energy
! dissipation rate at the surface in the case of shear logarithmic flow, 
! i.e. no buoyancy flux but wind stress given at the surface (Burchard, 2000).

! Note, that turbulent kinetic energy (TKE) and turbulent diffusivity have to be taken
! at the surface. The TKE may be taken at the first level below the surface, assuming
! d(TKE)/dz = 0, that is satisfied for logarithmic surface layer. 
! The appropriate extrapolation of turbulent diffusivity (kturb) from the first level below
! the surface to the surface itself is using the linear profile of kturb:

! kturb ~ karman_constant*friction_velocity*z

! that is exact one for logarithmic profile.

use PHYS_CONSTANTS, only : &
& row0
use TURB_CONST !, only : sigmaeps, CE0, kar, sigmaE

use PHYS_FUNC, only : &
& H13

implicit none

real(kind=ireals) :: DISSIPATION_FLUX_SHEAR

! Input variables
real(kind=ireals), intent(in) :: momentum_flux
real(kind=ireals), intent(in) :: kturb
real(kind=ireals), intent(in) :: Esurf_in
real(kind=ireals), intent(in) :: z0_in
real(kind=ireals), intent(in) :: kwe
real(kind=ireals), intent(in) :: fetch
real(kind=ireals), intent(in) :: maxTKEinput
real(kind=ireals), intent(in) :: signwaveheight

integer(kind=iintegers), intent(in) :: nbc ! switch for b.c.s

! Local variables
real(kind=ireals), parameter :: small_number = 1.d-20
real(kind=ireals), save :: const
real(kind=ireals), save :: const1, const2, const3, a
real(kind=ireals) :: const4, const5
real(kind=ireals) :: z0s, xx, Esurf

logical, save :: firstcall = .true.

if (firstcall) then
  const = 1.d0/sigmaeps
  const1 = CE0**(0.75d0)/(sigmaeps*kar)
  const2 = const1/kar
  const3 = 1.5*sigmaE*CE0**(0.75d0)/CE0 ! must be CE instead of CE0 in denominator
  a = 1. ! Switch for shear (0 or 1)
endif

z0s = 1.d-2 !signwaveheight*0.85 ! 0.85 - empirical constant

!DISSIPATION_FLUX_SHEAR = (momentum_flux/row0)*(momentum_flux/row0) * &
!& const/z0_in

select case (nbc)
  case(1)
    ! From logarithmic profile (sheared flow without waves breaking)
    Esurf = Esurf_in !momentum_flux/row0/CE0**0.5 ! TKE at the surface
    DISSIPATION_FLUX_SHEAR = const1*kturb*Esurf**(1.5d0) / &
    & (z0_in*z0_in + small_number) ! roughness_length -> z0s?
  case(2)
    ! Dissipation flux in flow with breaking waves (Burchard, 2001)
    const4 = (1.5*sigmaE)**0.5*CE0**0.25*kwe
    ! xx = min(kwe*(momentum_flux/row0)**1.5,maxTKEinput)
    Esurf = momentum_flux/row0/(CE0**0.5)*const4**(2./3.) !Esurf_in TKE at the surface
    xx = kwe*(momentum_flux/row0)**1.5
    DISSIPATION_FLUX_SHEAR = const2*(momentum_flux/row0)**0.5* &
    & kar*z0s*const4**(1./3.)* &
    & (const3*xx + kar*Esurf**1.5) / (z0s*z0s + small_number)
  case(3)
    ! Generalized b.c. for sheared flow with wave breaking (Burchard, 2002)
    const4 = (1.5*sigmaE)**0.5*CE0**0.25*kwe
    const5 = sigmaeps**(-1)*(a + const4)**(1./3.)*((a + const4)/ &
    & (z0s + small_number) + alpham*const4)
    DISSIPATION_FLUX_SHEAR = const5*(momentum_flux/row0)**2
end select

if (firstcall) firstcall = .false.
END FUNCTION DISSIPATION_FLUX_SHEAR

!> The function TKE_WALL_SHEAR calculates the turbulent kinetic energy
!! at the surface in the case of shear lorarithmic flow, i.e. no buoyancy flux but wind stress
!! given at the surface.
FUNCTION TKE_WALL_SHEAR(momentum_flux)

use PHYS_CONSTANTS, only : &
& row0
use TURB_CONST, only : CE0

implicit none

real(kind=ireals) :: TKE_WALL_SHEAR

! Input variables
real(kind=ireals), intent(in) :: momentum_flux !> Momentum flux at the wall (surface), N/m**2

!Local variables

TKE_WALL_SHEAR = (momentum_flux/row0)/sqrt(CE0)

END FUNCTION TKE_WALL_SHEAR


!> The function DISSIPATION_WALL_SHEAR calculates the turbulent kinetic energy
!! dissipation rate at the surface in the case of shear logarithmic flow, 
!! i.e. no buoyancy flux but wind stress given at the surface.
FUNCTION DISSIPATION_WALL_SHEAR(momentum_flux,buoyancy_flux,dist)

use PHYS_CONSTANTS, only : &
& row0, kappa

implicit none

real(kind=ireals) :: DISSIPATION_WALL_SHEAR

! Input variables
real(kind=ireals), intent(in) :: momentum_flux !> Momentum flux at the wall (surface), N/m**2
real(kind=ireals), intent(in) :: buoyancy_flux !> Bouyancy flux at the wall (surface), m**2/s**3
real(kind=ireals), intent(in) :: dist          !> Distance from the wall (surface), m

! Local variables
!logical, save :: firstcall = .true.

!if (firstcall) then
!endif

DISSIPATION_WALL_SHEAR = (momentum_flux/row0)**1.5/(kappa*dist) + buoyancy_flux

!if (firstcall) firstcall = .false.
END FUNCTION DISSIPATION_WALL_SHEAR


!> Subroutine ED_TEMP_HOSTETLER calculates the eddy diffusivity for temperature
!! in a lake according to Technical Description of the Community Land Model (2004), p. 160
SUBROUTINE ED_TEMP_HOSTETLER(wind,frict_vel,zref,phi180,levels,row,ed_temp,N_levels)

use PHYS_CONSTANTS, only : &
& kappa, & ! Karman constant, n/d
& g ! Acceleration due to gravity, m/s**2

implicit none

! Input variables
real(kind=ireals), intent(in) :: wind !> Module of the wind speed, m/s
real(kind=ireals), intent(in) :: frict_vel !> Friction velocity, m/s
real(kind=ireals), intent(in) :: zref !> The height in the atmosphere, where the wind is measured (computed), m
real(kind=ireals), intent(in) :: phi180 !> The latitude, degrees

integer(kind=iintegers), intent(in) :: N_levels !> The number of levels in water column
real(kind=ireals), intent(in) :: levels(1:N_levels+1) !> The level depths, m
real(kind=ireals), intent(in) :: row(1:N_levels+1) !> Water density, kg/m**3

! Output variables
real(kind=ireals), intent(out) :: ed_temp(1:N_levels) !> The eddy diffusivity for temperature, m**2/s

! Local variables
real(kind=ireals), parameter :: w_star_const = 1.2d-3
real(kind=ireals), parameter :: k_star_const = 6.6, k_star_power_const = -1.84
real(kind=ireals), parameter :: PrandtL0 = 1.
real(kind=ireals), parameter :: Ri_const1 = 37., Ri_const2 = 20.
real(kind=ireals), parameter :: Brunt_Vaisala2_const = 40.
real(kind=ireals), parameter :: z0_water = 1.d-3 ! Water surface roughness, m
real(kind=ireals), parameter :: wind2_min = 1. ! Minimal wind speed at 2 meters
real(kind=ireals), parameter :: pi = 3.14159265 ! pi constant
real(kind=ireals), parameter :: small_value = 1.e-20

real(kind=ireals) :: phi
real(kind=ireals) :: wind2
real(kind=ireals) :: k_star, w_star
real(kind=ireals) :: Brunt_Vaisala2, Richardson
real(kind=ireals) :: level

integer(kind=iintegers) :: i

phi = phi180/180.*pi
wind2 = max(frict_vel/kappa*log(2./z0_water),wind2_min)
w_star = w_star_const*wind2
k_star = k_star_const*wind2**(k_star_power_const)*sqrt(abs(sin(phi)))

do i = 1, N_levels
  level = 0.5*(levels(i) + levels(i+1))
  Brunt_Vaisala2 = 2.*g/(row(i)+row(i+1))*(row(i+1)-row(i))/(levels(i+1)-levels(i))
  Richardson = &
  & (- 1. + sqrt(1. + &
  & max(Brunt_Vaisala2_const*Brunt_Vaisala2*kappa**2*level**2/ &
  & (w_star**2*exp( - 2.*k_star*level) + small_value), -1.d0) &
  & ))/Ri_const2
  ed_temp(i) = kappa*w_star*level*exp(-k_star*level)/ &
  & (PrandtL0*(1.+Ri_const1*Richardson**2) )
enddo

END SUBROUTINE ED_TEMP_HOSTETLER

!> Subroutine ED_TEMP_HOSTETLER calculates the eddy diffusivity for temperature
!! in a shallow lake (pond), where bottom and top boundary layers intersect
!! in the middle of a water column. Hendersson-Sellers formulations are applied
!! for bottom and top boundary layers, excluding Ekman terms.
SUBROUTINE ED_TEMP_HOSTETLER2(wind,frict_vel,zref,phi180,levels,row,ed_temp,N_levels)

use PHYS_CONSTANTS, only : &
& kappa, & ! Karman constant, n/d
& g ! Acceleration due to gravity, m/s**2

implicit none

! Input variables
real(kind=ireals), intent(in) :: wind !> Module of the wind speed, m/s
real(kind=ireals), intent(in) :: frict_vel !> Friction velocity, m/s
real(kind=ireals), intent(in) :: zref !> The height in the atmosphere, where the wind is measured (computed), m
real(kind=ireals), intent(in) :: phi180 !> The latitude, degrees

integer(kind=iintegers), intent(in) :: N_levels !> The number of levels in water column
real(kind=ireals), intent(in) :: levels(1:N_levels+1) !> The level depths, m
real(kind=ireals), intent(in) :: row(1:N_levels+1) !> Water density, kg/m**3

! Output variables
real(kind=ireals), intent(out) :: ed_temp(1:N_levels) !> The eddy diffusivity for temperature, m**2/s

! Local variables
real(kind=ireals), parameter :: w_star_const = 1.2d-3
real(kind=ireals), parameter :: k_star_const = 6.6, k_star_power_const = -1.84
real(kind=ireals), parameter :: PrandtL0 = 1.
real(kind=ireals), parameter :: Ri_const1 = 37., Ri_const2 = 20.
real(kind=ireals), parameter :: Brunt_Vaisala2_const = 40.
real(kind=ireals), parameter :: z0_water = 1.d-3 ! Water surface roughness, m
real(kind=ireals), parameter :: wind2_min = 1. ! Minimal wind speed at 2 meters
real(kind=ireals), parameter :: pi = 3.14159265 ! pi constant
real(kind=ireals), parameter :: small_value = 1.e-20

real(kind=ireals) :: phi
real(kind=ireals) :: wind2
real(kind=ireals) :: k_star, w_star
real(kind=ireals) :: Brunt_Vaisala2, Richardson
real(kind=ireals) :: level

integer(kind=iintegers) :: i

phi = phi180/180.*pi
wind2 = max(frict_vel/kappa*log(2./z0_water),wind2_min)
w_star = w_star_const*wind2
!k_star = k_star_const*wind2**(k_star_power_const)*sqrt(abs(sin(phi)))

do i = 1, N_levels
  level = 0.5*(levels(i) + levels(i+1))
  Brunt_Vaisala2 = 2.*g/(row(i)+row(i+1))*(row(i+1)-row(i))/(levels(i+1)-levels(i))
  if ( level <= 0.5*levels(N_levels+1) ) then
    Richardson = &
    & (- 1. + sqrt(1. + &
    & max(Brunt_Vaisala2_const*Brunt_Vaisala2*kappa**2*level**2/ &
    & (w_star**2 + small_value), -1._ireals) ))/Ri_const2
    ed_temp(i) = kappa*w_star*level/(PrandtL0*(1.+Ri_const1*Richardson**2) )
  else
    Richardson = &
    & (- 1. + sqrt(1. + &
    & max(Brunt_Vaisala2_const*Brunt_Vaisala2*kappa**2* &
    & (levels(N_levels+1) - level)**2/ &
    & (w_star**2 + small_value), -1._ireals) ))/Ri_const2
    ed_temp(i) = kappa*w_star*(levels(N_levels+1) - level)/ &
    & (PrandtL0*(1.+Ri_const1*Richardson**2) )
  endif 
enddo

END SUBROUTINE ED_TEMP_HOSTETLER2


!> Subroutine EDDY_DIFF_MOST_RICH returns eddy viscosity and eddy diffusivity for scalars
!! basing on length scale and local Monin-Obukhov flux-profile relationships
SUBROUTINE EDDY_DIFF_MOST_RICH(M,Rich,ind,u,v,dzeta_05int,ddz,h,k2,KT)
use NUMERIC_PARAMS, only : pi
use PHYS_CONSTANTS, only : kappa
implicit none
!Input varaibles
integer(kind=iintegers), intent(in) :: M    !> Number of layers in water
real(kind=ireals), intent(in ) :: Rich(1:M) !> Richardson number, n/d
integer(kind=iintegers), intent(in) :: ind  !> Switch: 1 - compute viscosity, 
                                            !! 2 - compute diffusivity, 3 - compute both
real(kind=ireals), intent(in ) :: u(1:M+1) !> x-component of velocity, m/s
real(kind=ireals), intent(in ) :: v(1:M+1) !> y-component of velocity, m/s
real(kind=ireals), intent(in ) :: dzeta_05int(1:M) !> non-dim depth of cells' centers, n/d
real(kind=ireals), intent(in ) :: ddz       (1:M) !> non-dim thickness of cells, n/d
real(kind=ireals), intent(in ) :: h        !> maximal depth of a water body, m
!Output varaibles
real(kind=ireals), intent(out) :: k2(1:M) !> eddy viscosity   (for momentum), m**2/s
real(kind=ireals), intent(out) :: KT(1:M) !> eddy diffusivity (for scalars) , m**2/s
!Local variables
real(kind=ireals) :: phi_m(1:M), phi_s(1:M)
real(kind=ireals) :: z_Obukhov(1:M)
integer(kind=iintegers) :: ind1, i

ind1 = ind
if (ind == 2) ind1 = 3 !for computing diffusivity both MOST functions are needed
call MOST_FUNCTIONS_RICH(M,Rich,ind1,phi_m,phi_s)

forall (i=1:M) z_Obukhov(i) = sin(pi*dzeta_05int(i))*h/pi !Obukhov length scale

if (ind == 1 .or. ind == 3) then
  do i = 1, M
    k2(i) = (kappa*z_Obukhov(i))**2*( (u(i+1)-u(i))**2 + (v(i+1)-v(i))**2 )**0.5 / &
    &       (ddz(i)*h) / (phi_m(i)*phi_m(i)+small_value)
  enddo
endif

if (ind == 2 .or. ind == 3) then
  do i = 1, M
    KT(i) = (kappa*z_Obukhov(i))**2*( (u(i+1)-u(i))**2 + (v(i+1)-v(i))**2 )**0.5 / &
    &       (ddz(i)*h) / (phi_m(i)*phi_s(i)+small_value)
  enddo
endif

END SUBROUTINE EDDY_DIFF_MOST_RICH


!> Subroutine MOST_FUNCTIONS_RICH returns the vertical profiles of Monin-Obukhov functions
!! of the Richardson number for momentum and scalar, following Alexander Soloviev et al., J.Geophys.Res., 2001
SUBROUTINE MOST_FUNCTIONS_RICH(M,Rich,ind,phi_m,phi_s)
implicit none
!Input variables
integer(kind=iintegers), intent(in) :: M    !> Number of layers in water
real(kind=ireals), intent(in ) :: Rich(1:M) !> Richardson number, n/d
integer(kind=iintegers), intent(in) :: ind  !> Switch: 1 - compute phi_m, 2 - compute phi_s, 3 - compute both
!Output variables
real(kind=ireals), intent(out) :: phi_m(1:M) !> Monin-Obukhov function for momentum, n/d
real(kind=ireals), intent(out) :: phi_s(1:M) !> Monin-Obukhov function for scalar, n/d
!Local parameters
real(kind=ireals), parameter :: a_m = 1.26, c_m = 8.38
real(kind=ireals), parameter :: a_s = -28.86, c_s = 98.96
real(kind=ireals), parameter :: alpha = 16.
real(kind=ireals), parameter :: Rich_cr = 0.25, Rich_m = -0.2, Rich_s = -1.
integer(kind=iintegers) :: i

if (ind == 1 .or. ind == 3) then
  forall(i=1:M, Rich(i)> Rich_cr                  ) phi_m(i) = Rich_cr/(small_value)
  forall(i=1:M, Rich(i)> 0. .and. Rich(i)<=Rich_cr) phi_m(i) = Rich_cr/(Rich_cr - Rich(i)+small_value)
  forall(i=1:M, Rich(i)<=0. .and. Rich(i)> Rich_m ) phi_m(i) = (max(1. - alpha*Rich(i),small_value))**(-0.25)
  forall(i=1:M, Rich(i)<=Rich_m                   ) phi_m(i) = (max(a_m  - c_m*Rich(i),small_value))**(-1./3.)
endif

if (ind == 2 .or. ind == 3) then
  forall(i=1:M, Rich(i)> Rich_cr                  ) phi_s(i) = Rich_cr/(small_value)
  forall(i=1:M, Rich(i)> 0. .and. Rich(i)<=Rich_cr) phi_s(i) = Rich_cr/(Rich_cr - Rich(i)+small_value)
  forall(i=1:M, Rich(i)<=0. .and. Rich(i)> Rich_s ) phi_s(i) = (max(1. - alpha*Rich(i),small_value))**(-0.50)
  forall(i=1:M, Rich(i)<=Rich_s                   ) phi_s(i) = (max(a_s -  c_s*Rich(i),small_value))**(-1./3.)
endif

END SUBROUTINE MOST_FUNCTIONS_RICH


!> An analytical model for neutrally stratified river flow using Obukhov coordinate
SUBROUTINE ZIRIVMODEL(M,h1,u_star_surf,gradp,z_half,z_full,eps_ziriv,viscturb_ziriv,speed_ziriv)
use PHYS_CONSTANTS, only : kappa
use NUMERIC_PARAMS, only : pi
implicit none
!Input/output variables
integer(kind=iintegers), intent(in) :: M      !> Number of layers
real(kind=ireals), intent(in ) :: h1          !> Riverflow depth, m
real(kind=ireals), intent(in ) :: u_star_surf !> Friction velocity in water at the surface, m/s
real(kind=ireals), intent(in ) :: gradp       !> Longitudinal pressure gradient, Pa/m
real(kind=ireals), intent(in ) :: z_half(1:M) !> A set of depths where dissipation and viscosity 
                                              !! are to be evaluated
real(kind=ireals), intent(in ) :: z_full(1:M+1) !> A set of depths where speed is to be evaluated
real(kind=ireals), intent(out) :: eps_ziriv(1:M) !> TKE dissipation rate, m**2/s**3
real(kind=ireals), intent(out) :: viscturb_ziriv(1:M) !> Coefficient of turbulent viscosity, m**2/s
real(kind=ireals), intent(out) :: speed_ziriv(1:M+1) !> Longitudinal speed, m/s
!Local variables
integer(kind=iintegers) :: i !Loop index
real(kind=ireals) :: u_star, z_Obukhov, h_max

if (gradp*u_star_surf >= 0.) then 
  !Wind forcing along the pressure gradient, 
  !monotonic decrease of flow speed with depth
  speed_ziriv(M+1) = 0.
  do i = M, 1, -1
    z_Obukhov = sin(pi*z_half(i)/h1)*h1/pi
    u_star = sqrt(abs(gradp)*z_half(i) + u_star_surf**2)
    eps_ziriv(i) = u_star**3/(kappa*z_Obukhov)
    viscturb_ziriv(i) = kappa*u_star*z_Obukhov
    speed_ziriv(i) = speed_ziriv(i+1) + &
    & u_star/(kappa*z_Obukhov)*(z_full(i+1) - z_full(i))
  enddo
  speed_ziriv(:) = speed_ziriv(:)*sign(1._ireals,u_star_surf)
else
  !Wind forcing opposite to the pressure gradient, 
  !a maxumum of flow speed locates between top and bottom boundary
  h_max = u_star_surf**2/abs(gradp)
  do i = M, 1, -1
    z_Obukhov = sin(pi*z_half(i)/h1)*h1/pi
    u_star = sqrt(abs(abs(gradp)*z_half(i) - u_star_surf**2))
    eps_ziriv(i) = u_star**3/(kappa*z_Obukhov)
    viscturb_ziriv(i) = kappa*u_star*z_Obukhov
    speed_ziriv(i) = speed_ziriv(i+1) + &
    & u_star/(kappa*z_Obukhov)*(z_full(i+1) - z_full(i)) * &
    & sign(1._ireals,z_half(i)-h_max)
  enddo
  speed_ziriv(:) = speed_ziriv(:)*sign(1._ireals, - u_star_surf)
endif

END SUBROUTINE ZIRIVMODEL

END MODULE TURB_LAKE_MOD
