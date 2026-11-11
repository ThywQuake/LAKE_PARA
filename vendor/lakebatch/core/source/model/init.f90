 MODULE GRID_OPERATIONS_LAKE

 contains
 SUBROUTINE GRID_CREATE

 use LAKE_DATATYPES, only : ireals, iintegers

 use ARRAYS_GRID
 use DRIVING_PARAMS
 use DZETA_MOD, only : DZETA, DZETAI

 implicit none

 integer(kind=iintegers) :: i
 real(kind=ireals) :: dzeta1, dzeta2

!The grid according to Burchard, 2002, pp. 97-98
 do i = 1, M
   dzeta1 = &
   & (tanh(d_surf%par)+tanh((d_surf%par+d_bot%par)*real(i-1)/real(M)-d_surf%par))/ &
   & (tanh(d_surf%par)+tanh(d_bot%par))
   dzeta2 = &
   & (tanh(d_surf%par)+tanh((d_surf%par+d_bot%par)*real(i)/real(M)-d_surf%par))/ &
   & (tanh(d_surf%par)+tanh(d_bot%par))
   ddz(i) = dzeta2 - dzeta1
 enddo
 
 ddz05(0) = 0.5d0*ddz(1)
 do i = 1, M-1
   ddz2(i) = ddz(i) + ddz(i+1)
   ddz05(i) = 0.5d0*(ddz(i) + ddz(i+1))   
 enddo
 ddz05(M) = 0.5d0*ddz(M)
 
 do i = 2, M-1
   ddz052(i) = 0.5d0*ddz(i-1) + ddz(i) + 0.5d0*ddz(i+1)
   ddz054(i) = 2.d0*ddz052(i)
 enddo
 
 do i = 1, M+1 
   dzeta_int(i) = DZETA(real(i))
 enddo
 
 do i = 1, M
   dzeta_05int(i) = DZETA(real(i)+0.5)
 enddo
 
 ddzi(:) = 1.d0/float(Mice)

 ddzi05(0) = 0.5d0*ddzi(1)
 do i = 1, Mice-1
   ddzi05(i) = 0.5d0*(ddzi(i) + ddzi(i+1))
 enddo
 ddzi05(Mice) = 0.5d0*ddzi(Mice)

 do i = 1, Mice+1 
   dzetai_int(i) = DZETAI(real(i))
 enddo
 
 do i = 1, Mice
   dzetai_05int(i) = DZETAI(real(i)+0.5)
 enddo

 
 END SUBROUTINE GRID_CREATE


SUBROUTINE LININTERPOL (z1,f1,N1,z2,f2,N2,flag,trextr)

use LAKE_DATATYPES, only : ireals, iintegers
use DRIVING_PARAMS, only : missing_value

 implicit none
! Input variables
! Integers
integer(kind=iintegers), intent(in) :: N1, N2
! Reals
real(kind=ireals), intent(in) :: z1(1:N1), z2(1:N2)
real(kind=ireals), intent(in) :: f1(1:N1)
logical, intent(in), optional :: trextr

! Output variables
! Reals
real(kind=ireals), intent(out) :: f2(1:N2)
logical, intent(out) :: flag

! Local variables
integer(kind=iintegers) :: i, j ! Loop indices
logical :: extrap

extrap = .true.
if (present(trextr)) extrap = trextr

flag = .true.

if (extrap) then
  forall(i = 1 : N2, z2(i) < z1(1)) f2(i) = f1(1) + &
  & (f1(2) - f1(1)) / (z1(2)-z1(1)) * (z2(i) - z1(1))
  forall(i = 1 : N2, z2(i) >= z1(N1)) f2(i) = f1(N1)
else
  forall(i = 1 : N2, z2(i) <  z1(1 )) f2(i) = missing_value
  forall(i = 1 : N2, z2(i) >= z1(N1)) f2(i) = missing_value
endif

do i = 1, N2
  if (z2(i) >= z1(1) .and. z2(i) < z1(N1)) then
    do j = 1, N1
      if (z2(i) >= z1(j) .and. z2(i) < z1(j+1)) then
        f2(i) = f1(j) + ( f1(j+1) - f1(j) ) * &
        & ( z2(i) - z1(j) )/( z1(j+1) - z1(j) )
        exit
      endif
    enddo
  endif  
enddo 

END SUBROUTINE LININTERPOL


SUBROUTINE PIECEWISECONSTINT (z1,f1,N1,z2,f2,N2,flag)

 use LAKE_DATATYPES, only : ireals, iintegers

 implicit none
! Input variables
! Integers
integer(kind=iintegers), intent(in) :: N1, N2

! Reals
real(kind=ireals), intent(in) :: z1(1:N1), z2(1:N2)

real(kind=ireals), intent(in) :: f1(1:N1)

! Output variables
! Reals
real(kind=ireals), intent(out) :: f2(1:N2)

integer(kind=iintegers), intent(in) :: flag

! Local variables
integer(kind=iintegers) :: i, j, k ! Loop indices


select case (flag)
case(1_iintegers)
  k = 1
  cyc: do i = 2, N1
    do while (z2(k) <= z1(i))
      f2(k) = f1(i-1)
      k = k + 1
      if (k > N2) exit cyc
    enddo
  enddo cyc 
case(2_iintegers)
  k = 1
  cyc1: do i = 2, N1
    do while (z2(k) <= 0.5*(z1(i) + z1(i-1)) )
      f2(k) = f1(i-1)
      k = k + 1
      if (k > N2) exit cyc1
    enddo
  enddo cyc1
  if (k <= N2) then
    f2(k:N2) = f1(N1)
  endif
end select

END SUBROUTINE PIECEWISECONSTINT
END MODULE GRID_OPERATIONS_LAKE

 MODULE ALLOCARRAYS_MOD
 contains


 !> Subroutine ALLOCARRAYS allocates arrays, sets arrays to reference values, 
 !! and assigns pointers
 SUBROUTINE ALLOCARRAYS(nx,ny)

 use LAKE_DATATYPES, only : ireals, iintegers

 use DRIVING_PARAMS
 use ARRAYS
 use ARRAYS_GRID
 use ARRAYS_BATHYM
 use ARRAYS_SOIL
 use ARRAYS_WATERSTATE
 use ARRAYS_METHANE
 use ARRAYS_OXYGEN
 use ARRAYS_TURB
 use NUMERIC_PARAMS
 use METH_OXYG_CONSTANTS, only : &
 & ngasb
 use RADIATION, only : RadWater, RadIce, RadDeepIce, &
 & fracbands, nbands, RAD_POINTERS
 use GRID_OPERATIONS_LAKE, only : GRID_CREATE

 implicit none

 real(kind=ireals), parameter :: small_number = 1.d-20

 !> nx, ny -- sizes in x and y, respectively, of a mesh of lakes
 integer(kind=iintegers), target :: nx, ny
 integer(kind=iintegers) :: i, j, k !Loop index

 SAVE

 nsoilcols = nsoilcols_%par

 allocate (dz_full(1:M), z_full(1:M+1), z_half (1:M), &
 & zsoilcols(1:nsoilcols+1,1:nx,1:ny))
 zsoilcols(:,:,:) = missing_value
 allocate (z_full_ice(1:Mice+1), z_half_ice(1:Mice))
 allocate (isoilcolc(1:nsoilcols),ksoilcol(1:M+1))
 allocate (ddz(1:M), ddz2(1:M-1), ddz05(0:M), ddz052(2:M-1), ddz054(2:M-1))
 allocate (dzeta_int(1:M+1), dzeta_05int(1:M))
 allocate (ddzi(1:Mice), ddzi05(0:Mice))
 allocate (dzetai_int(1:M+1), dzetai_05int(1:M)) 
 allocate (Tw1(1:M+1),Tw2(1:M+1),Ti1(1:Mice+1),Ti2(1:Mice+1), &
 & Tis1(1:Mice+1),Tis2(1:Mice+1),RS(1:M+1),lamw(1:M),lamw_back(1:M), &
 & Sal1(1:M+1),Sal2(1:M+1),lamsal(1:M),preswat(1:M+1))
 allocate (porice(1:Mice+1),salice(1:Mice+1))
 allocate (ci_m_roi_v(1:Mice+1), lami_v(1:Mice))
 allocate ( Tskin(1:2) )

 !Time group
 time_group%time  => time
 time_group%nstep => nstep

 ! Radiation group
 !allocate (fracbands(1:nbands))
 call RAD_POINTERS()
 call RadWater  %RAD_INIT(nbands=nbands,nlevels=M+1,extCDOC=(carbon_model%par==2))
 call RadIce    %RAD_INIT(nbands=nbands,nlevels=Mice+1,extCDOC=.false.)
 call RadDeepIce%RAD_INIT(nbands=nbands,nlevels=Mice+1,extCDOC=.false.)

 if (dyn_pgrad%par == 3 .or. dyn_pgrad%par == 4) &
 & allocate(hx1ml(1:M+1),hx2ml(1:M+1),hy1ml(1:M+1),hy2ml(1:M+1))

 if (dyn_pgrad%par == 4) then
   allocate ( gradpx_tend(1:M+1), gradpy_tend(1:M+1) )
   allocate ( itherm(1:M+2) )
 endif

 ! Soil group
 allocate (Tsoil1(1:ns,1:nsoilcols), Tsoil2(1:ns), &
 &         Tsoil3(1:ns,1:nsoilcols), &
 &         wl1(1:ns,1:nsoilcols),    wl2(1:ns), &
 &         wl3(1:ns),                wl4(1:ns,1:nsoilcols), &
 &         wi1(1:ns,1:nsoilcols),    wi2(1:ns,1:nsoilcols), &
 &         rosoil(1:ns),             csoil(1:ns), &
 &         lamsoil(1:ns),            dzs(1:ns),   &
 &         dzss(1:ns),               zsoil(1:ns) ) 
 allocate (z_watersoil(1:M+ns))
 allocate (lammoist(1:ns), rosdry(1:ns), &
 & filtr(1:ns), wsoil(1:ns), &
 & Sals1(1:ns,1:nsoilcols), &
 & Sals2(1:ns,1:nsoilcols), wa(1:ns) )
 allocate (soilflux(1:nsoilcols,1:nx,1:ny)) 
 allocate (methsoilflux(1:nsoilcols),co2soilflux(1:nsoilcols),o2soilflux(1:nsoilcols))
 allocate (lsh%water(1:M+1),lsh%ice(1:Mice+1),lsh%dice(1:Mice+1))
 allocate (WLM0(1:ns),WLM7(1:ns),bH(1:ns),PSIMAX(1:ns), &
 & POR(1:ns),FLWMAX(1:ns),DLMAX(1:ns))

 ! Bathymetry group
 allocate ( area_int(1:M+1), area_half(1:M+1), Lx(1:M+1), Ly(1:M+1) )
 allocate ( bathymwater(1:M+1), bathymice(1:Mice+1) ) 
 allocate ( bathymdice(1:Mice+1), bathymsoil(1:nsoilcols+1,1:nx,1:ny) )

 !! Horizontal cross-section areas of soil columns
 !do j = 1, ny
 !  do i = 1, nx
 !    do k = 1, nsoilcols+1
 !      allocate(bathymsoil(k,i,j)%sarea_int (1:ns))
 !      allocate(bathymsoil(k,i,j)%sarea_half(1:ns))
 !    enddo
 !  enddo
 !enddo


 ! Methane group
 allocate (rootss(1:ns), qsoil(1:ns,1:nsoilcols), &
 &         qwater(1:M+1,1:2), qmethane(1:M+ns), TgrAnn(1:ns))
 allocate (fbbleflx_ch4(0:M+1,1:nsoilcols), febul0(1:nsoilcols), &
 & fbbleflx_ch4_sum(0:M+1)); febul0 = 0.
 allocate (fdiffbot(1:nsoilcols))
 allocate (fracb0(1:ngasb))
 allocate (methgenmh(1:ns))
 allocate (lsm%water(1:M+1))
 allocate (lammeth(1:M))
 allocate (methoxid(1:M+1))
 allocate (rprod_total_oldC(1:nsoilcols), rprod_total_newC(1:nsoilcols))
 
 ! Oxygen group
 allocate (oxyg(1:M+1,1:2), lamoxyg(1:M))
 allocate (fbbleflx_o2(0:M+1,1:nsoilcols), fbbleflx_o2_sum(0:M+1))
 allocate (Chl_a(1:M+1), prodox(1:M+1), resp(1:M+1), bod(1:M+1), sod(1:M+1)) 
 allocate (lso%water(1:M+1))
 allocate (oxygsoil(1:nsoilcols))
       
 ! Carbon dioxide group
 allocate (DIC(1:M+1,1:2), lamcarbdi(1:M), lampart(1:M))
 allocate (fbbleflx_co2(0:M+1,1:nsoilcols), fbbleflx_co2_sum(0:M+1))
 allocate (lsc%water(1:M+1))
 allocate(DOC(1:M+1,1:2),POCL(1:M+1),POCD(1:M+1))
 
 ! Phosphorus group
 allocate (lsp%water(1:M+1))
 allocate (phosph(1:M+1))
 phosph(1:M+1) = 0.
 
 ! Turbulence and dynamics group
 allocate (lsu%water(1:M+1),lsv%water(1:M+1))
 allocate (S(1:M),Gen(1:M),F(1:M),TKE_turb_trans(1:M),Gen_seiches(1:M))
 allocate (KT(1:M+1),k2(1:M+1),u1(1:M+1),v1(1:M+1), &
 & E1(1:M+1), eps1(1:M+1), wArea(0:M+1), w(0:M+1))
 !if (dyn_pgrad%par == 5 .or. dyn_pgrad%par == 6) then 
 allocate (eps_ziriv     (1:M  )); eps_ziriv      (:) = missing_value
 allocate (viscturb_ziriv(1:M  )); viscturb_ziriv (:) = missing_value
 allocate (speed_ziriv   (1:M+1)); speed_ziriv    (:) = missing_value
 !endif
 allocate (WU_(1:M+1),WV_(1:M+1),GAMT(1:M+1),GAMU(1:M+1), &
 &  GAMV(1:M+1),TF(1:M+1),KLengu(1:M+1)) 
 allocate (num(1:M+1)) 
 allocate ( E2(1:M+1),eps2(1:M+1), &
 & u2(1:M+1),v2(1:M+1), &
 & k1(1:M+1),k3(1:M+1),k4(1:M+1),k5(1:M+1), &
 & u3(1:M+1),v3(1:M+1),C1aup(1:M+1),Re(1:M+1),row(1:M+1),row2(1:M+1), rowc(1:M+1), &
 & Ri(1:M),uv(1:M+1),E_it1(1:M+1),dres2dE(1:M+1),dres2deps(1:M+1), & 
 & E_it2(1:M+1),C_num(1:M+1),E_it3(1:M+1),Feps(1:M+1), &
 & E_it21(1:M+1),eps_it21(1:M+1),eps_it1(1:M+1), &
 & eps_it2(1:M+1),l(1:M+1),k2_mid(1:M+1),k3_mid(1:M+1), &
 & k4_mid(1:M+1),k5_mid(1:M+1),Um(1:M+1),Vm(1:M+1),E12(1:M+1), &
 & eps12(1:M+1),knum(1:M+1),k2t(1:M+1),veg_sw(1:M+1) )
 allocate ( Eeps(1:M+1),Ecent(1:M+1), &
 & Epscent(1:M+1),res_E(1:M+1),res_eps(1:M+1), &
 & dresd(2,2,1:M+1,1:M+1) ) 
 allocate (KC(1:M+1),KLengT(1:M+1),RSR(1:M+1),dep_2d(0:nx-1,0:ny-1))
 allocate ( init(nx,ny) )
 allocate ( PEMF    (1:M)   , PDENS_DOWN (1:M+1) , &
 & PT_DOWN (1:M+1) , PSAL_DOWN  (1:M+1) , &
 & pt_down_f (1:M)  )
 allocate ( k_turb_T_flux(1:M), T_massflux(1:M) )
 allocate (trb%Rp(1:M), trb%Rpdens(1:M))

 Tw1= 0. ; Tw2    = 0.
 Ti1= 0. ; Ti2    = 0.
 Tis1    = 0. ; Tis2   = 0.
 Sal1    = 0. ; Sal2   = 0.
 wArea   = 0. ; w = 0.
 RS = 0. 
 SR_botsnow = 0.d0
 lamw    = 0. ; lamsal = 0. ; lamsoil = 0. 
 Tsoil1  = 0. ; Tsoil2 = 0.
 rosoil  = 0. ; csoil  = 0. 
 lsh%water(:) = 0.; lsh%ice(:) = 0.; lsh%dice(:) = 0.
 soilflux(:,:,:) = 0.
 methsoilflux(:) = 0.
 co2soilflux (:) = 0.
 o2soilflux  (:) = 0.
 dzs= 0. ; dzss   = 0. 
 wi1= 0. ; 
 wl1= 0. ; 
 S  = 0. ; Gen    = 0  ; F  = 0. ; TKE_turb_trans  = 0.
 KT = 0. ; k2= 0.
 u1 = 0. ; v1= 0.
 u2 = 0. ; v2= 0.
 E1 = 0. ; eps1   = 0.
 WLM0    = 0. ; WLM7   = 0. 
 bH = 0. ; POR    = 0.
 PSIMAX  = 0. ; FLWMAX = 0. ; DLMAX   = 0.
 KC = 0. ; KLengT = 0.
 dhw= 0. ; dhw0   = 0. 
 dhi= 0. ; dhi0   = 0.
 dls= 0. ; dls0   = 0.
 
 PEMF    = 0.d0 ; PDENS_DOWN = 0.d0
 PT_DOWN = 0.d0 ; PSAL_DOWN  = 0.d0
 pt_down_f = 0.d0
 zinv    = 0.d0
 
 k_turb_T_flux = 0.d0
 T_massflux    = 0.d0
 
 febultot(1:2) = 0.d0
 febulbottot(1:2) = 0.d0
 fdifftot(1:2) = 0.d0
 fdiff_lake_surftot(1:2) = 0.d0
 methoxidwat(1:2) = 0.d0
 ice_meth_oxid = 0.d0
 meth_cont_wat_total0 = 0.d0
! febultot2(1:2,1:2) = 0.d0 ! two-meth
! fdifftot2(1:2,1:2) = 0.d0 ! two-meth
! fdiff_lake_surftot2(1:2,1:2) = 0.d0 ! two-meth
 rprod_total_newC_integr(1:2) = 0.d0
 rprod_total_oldC_integr(1:2) = 0.d0

! Relative bubble gas fluxes corresponding to 
! case of zero interaction between bubble and water solution
 fbbleflx_ch4(:,:) = 1.
 fbbleflx_o2 (:,:) = 1.
 fbbleflx_co2(:,:) = 1.

 fbbleflx_ch4_sum(:) = 0.
 fbbleflx_o2_sum (:) = 0.
 fbbleflx_co2_sum(:) = 0.

 fracb0(1) = 1.-2.*small_number ! The molar fraction of methane in bubble at the bottom
 fracb0(2) = small_number ! The molar fraction of carbon dioxide in bubble at the bottom
 fracb0(3) = small_number ! The molar fraction of oxygen in bubble at the bottom

 lsm%water(:) = 0.
 lsc%water(:) = 0.
 lso%water(:) = 0.

 init    = 0
  
 !ddz = 1.d0/float(M)

! Specification of dzeta-grid (liquid water layers)
! and of the dzetai-grid (deep ice and upper ice layers)
 call GRID_CREATE

 ! Associating pointers to groups of variables
 ! Layers' thicknesses
 ls%h1 => h1; ls%hx1 => hx1; ls%hx2 => hx2; ls%hy1 => hy1; ls%hy2 => hy2
 ls%hx1t => hx1t; ls%hx2t => hx2t; ls%hy1t => hy1t; ls%hy2t => hy2t
 ls%hx1ml => hx1ml; ls%hx2ml => hx2ml; ls%hy1ml => hy1ml; ls%hy2ml => hy2ml
 ls%l1 => l1; ls%hs1 => hs1; ls%ls1 => ls1; ls%vol => vol; ls%botar => botar
 ls%dhw=>dhw; ls%dhw0=>dhw0; ls%dhi => dhi; ls%dhi0=>dhi0
 ls%dls=>dls; ls%dls0=>dls0; ls%dhiimp => dhiimp; ls%snmeltice => snmeltice
 ls%H_photic_zone => H_photic_zone
 ls%dhwhigh => dhwhigh; ls%dhwlow => dhwlow
 ls%dhwp => dhwp; ls%dhwe => dhwe
 ls%dhwtrib => dhwtrib
 ls%dhwls => dhwls; ls%dhwsnmelt => dhwsnmelt
 ls%dhihigh => dhihigh; ls%dhilow => dhilow
 ls%dhip => dhip; ls%dhis => dhis
 ls%dhif => dhif

 !Gridsize group
 gs%M => M; gs%Mice => Mice; gs%ns => ns; gs%nsoilcols => nsoilcols
 gs%ms => ms_; gs%ml => ml_
 ! Note that gs%nx, gs%ny is a grid size in relative coordinates on a current processor
 gs%nx => nx; gs%ny => ny

 !Grid spacing group
 gsp%dz_full => dz_full; gsp%z_full => z_full; gsp%z_half => z_half
 gsp%z_full_ice => z_full_ice; gsp%z_half_ice => z_half_ice
 gsp%zsoilcols => zsoilcols; gsp%isoilcolc => isoilcolc; gsp%ksoilcol => ksoilcol
 gsp%ddz    => ddz;    gsp%ddz2   => ddz2; gsp%ddz05 => ddz05
 gsp%ddz052 => ddz052; gsp%ddz054 => ddz054
 gsp%ddzi   => ddzi;   gsp%ddzi05 => ddzi05
 gsp%dzeta_int  => dzeta_int;  gsp%dzeta_05int  => dzeta_05int
 gsp%dzetai_int => dzetai_int; gsp%dzetai_05int => dzetai_05int

 !Water state group
 wst%Tw1 => Tw1; wst%Tw2 => Tw2; wst%row => row;
 wst%Ti1 => Ti1; wst%Ti2 => Ti2; 
 wst%Tis1 => Tis1; wst%Tis2 => Tis2
 wst%RS => RS !; wst%extwatarr => extwatarr
 wst%lamw => lamw; wst%lamw_back => lamw_back; wst%lamsal => lamsal 
 wst%Sal1 => Sal1; wst%Sal2 => Sal2
 wst%salice => salice
 wst%porice => porice
 wst%ci_m_roi_v => ci_m_roi_v
 wst%lami_v => lami_v
 wst%preswat => preswat
 wst%u1 => u1; wst%v1 => v1
 wst%u2 => u2; wst%v2 => v2
 wst%wArea => wArea ; wst%w => w

 !Gases concentration group
 gas%qmethane => qmethane
 gas%qwater   => qwater
 gas%qsoil    => qsoil
 gas%oxyg     => oxyg
 gas%DIC      => DIC
 gas%DOC      => DOC
 gas%POCL     => POCL
 gas%POCD     => POCD
 gas%phosph   => phosph

 END SUBROUTINE ALLOCARRAYS
 END MODULE ALLOCARRAYS_MOD


