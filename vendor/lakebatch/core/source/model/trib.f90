MODULE TRIBUTARIES

use LAKE_DATATYPES, only : ireals, iintegers

use ARRAYS_GRID, only : gridspacing_type
use ARRAYS_WATERSTATE, only : waterstate_type

use TIMEVAR, only : day_sec
use NUMERIC_PARAMS, only : small_value

use GRID_OPERATIONS_LAKE, only : PIECEWISECONSTINT
use LB_POINT_CTX ! lakebatch multi-point: per-point module state (LBC_TRIBUTARIES)

private

integer(kind=iintegers), save :: itriblev

integer(kind=iintegers), parameter, public :: nvars   = 11
integer(kind=iintegers), parameter, public :: nheat   = 1
integer(kind=iintegers), parameter, public :: nsal    = 2
integer(kind=iintegers), parameter, public :: nDIC    = 3
integer(kind=iintegers), parameter, public :: nDOC    = 4
integer(kind=iintegers), parameter, public :: nPOCL   = 5
integer(kind=iintegers), parameter, public :: nPOCD   = 6
integer(kind=iintegers), parameter, public :: nmeth   = 7
integer(kind=iintegers), parameter, public :: noxyg   = 8
integer(kind=iintegers), parameter, public :: nphosph = 9
integer(kind=iintegers), parameter, public :: nu      = 10
integer(kind=iintegers), parameter, public :: nv      = 11

!Type for fluxes of physical and biogeochemical properties in and out of the lake
!due to tributaries and effluents
type, public :: influx_outflux
  real(kind=ireals) :: fluxes_in (1:nvars) = 0.
  real(kind=ireals) :: fluxes_out(1:nvars) = 0.
end type influx_outflux

type(influx_outflux), public :: inout_fluxes

public :: ABSTR, TRIBTEMP, LBC_TRIBUTARIES

real(kind=ireals), allocatable, save :: trib_level_2d(:,:)
integer(kind=iintegers), allocatable, save :: trib_unit1_2d(:,:), trib_unit2_2d(:,:)
logical, allocatable, save :: trib_open_2d(:,:), trib_first_read_2d(:,:)
logical, save :: trib_allocated = .false.

contains

!>Subroutine TRIBTEMP computes the change of the
!!horizontally average lake temperature, salinity, constituents and velocities
!!due to tributaries (inflows and outflows) advection
SUBROUTINE TRIBTEMP(time,dt,h1,h10,dhwtrib,trib_inflow, &
&                   z_full,area_int,area_half,gsp,gas,wst,spinup_done,ix,iy,nx,ny)

 use DRIVING_PARAMS , only : &
 & N_tribin, & 
 & N_triblev, & 
 & N_tribout, &
 & U_tribin, &
 & U_tribout, &
 & T_tribin, &
 & Sal_tribin, &
 & DIC_tribin, &
 & DOC_tribin, &
 & POC_tribin, &
 & meth_tribin, &
 & oxyg_tribin, &
 & Ux_tribin, &
 & Uy_tribin, &
 & width_tribin, &
 & width_tribout, &
 & disch_tribin, &
 & disch_tribout, &
 & M, &
 & dttribupdate, &
 & missing_value, &
 & outflpar, &
 & tribheat, &
 & carbon_model, &
 & nvartribext

 use ARRAYS, only : gas_type
 use INOUT_PARAMETERS, only : lake_misc_unit_min

 use ARRAYS_METHANE, only : foutflow, finflow

 implicit none

 real(kind=ireals), intent(in)    :: time, dt, h1, h10
 real(kind=ireals), intent(inout) :: dhwtrib
 real(kind=ireals), intent(in)    :: trib_inflow(1:nvartribext)
 real(kind=ireals), intent(in)    :: z_full(1:M+1)
 real(kind=ireals), intent(in)    :: area_int(1:M+1), area_half(1:M)

 type(gridspacing_type), intent(in)   :: gsp
 type(gas_type)        , intent(inout):: gas
 type(waterstate_type) , intent(inout):: wst

 logical, intent(in) :: spinup_done
 integer(kind=iintegers), intent(in) :: ix, iy, nx, ny

!Local variables

 real(kind=ireals) :: heatinflow, salinflow, uinflow, vinflow, &
 & methinflow, oxyginflow, DICinflow, POCinflow, DOCinflow
 real(kind=ireals) :: meanTin, meansalin, meanuin, meanvin, &
 & meanmethin, meanoxygin, meanDICin, meanDOCin, meanPOCin
 real(kind=ireals) :: outflow, inflow, work 
 real(kind=ireals) :: invdt

 real(kind=ireals), allocatable :: U_tribin_(:,:), T_tribin_(:,:), Sal_tribin_(:,:), &
 & Ux_tribin_(:,:), Uy_tribin_(:,:), U_tribout_(:,:), &
 & meth_tribin_(:,:), oxyg_tribin_(:,:), DIC_tribin_(:,:),DOC_tribin_(:,:), &
 & POC_tribin_(:,:), width_tribin_(:,:), width_tribout_(:,:), z_(:)

 integer(kind=iintegers) :: i, j, k

 logical :: flag

 if (.not. allocated(trib_level_2d)) then
   allocate(trib_level_2d(nx,ny), trib_unit1_2d(nx,ny), trib_unit2_2d(nx,ny))
   allocate(trib_open_2d(nx,ny), trib_first_read_2d(nx,ny))
   trib_level_2d = missing_value
   trib_unit1_2d = lake_misc_unit_min
   trib_unit2_2d = lake_misc_unit_min + 1
   trib_open_2d = .false.
   trib_first_read_2d = .true.
 endif
 if (.not. trib_allocated) then
   if ((tribheat%par == 2 .or. tribheat%par == 3) &
     & .and. dttribupdate == missing_value) then
     write(*,*) 'Missing tributaries update interval: STOP'
     STOP
   endif
   ! Allocation of inflow and outflow profiles
   if (tribheat%par == 1 .or. tribheat%par == 2) then
     itriblev = N_triblev%par
   elseif (tribheat%par == 3 .or. tribheat%par == 4) then
     itriblev = M+1
   endif
   allocate (width_tribin  (1:N_tribin%par ,1:itriblev) )
   allocate (U_tribin      (1:N_tribin%par ,1:itriblev) )
   allocate (T_tribin      (1:N_tribin%par ,1:itriblev) )
   allocate (Sal_tribin    (1:N_tribin%par ,1:itriblev) )
   allocate (meth_tribin   (1:N_tribin%par ,1:itriblev) )
   allocate (oxyg_tribin   (1:N_tribin%par ,1:itriblev) )
   allocate (DIC_tribin    (1:N_tribin%par ,1:itriblev) )
   allocate (DOC_tribin    (1:N_tribin%par ,1:itriblev) )
   allocate (POC_tribin    (1:N_tribin%par ,1:itriblev) )
   allocate (Ux_tribin     (1:N_tribin%par ,1:itriblev) )
   allocate (Uy_tribin     (1:N_tribin%par ,1:itriblev) )
   allocate (disch_tribin  (1:N_tribin%par ,1:M+1) )
   allocate (disch_tribout (1:N_tribout    ,1:M+1) )
   allocate (width_tribout (1:N_tribout    ,1:itriblev) )
   allocate (U_tribout     (1:N_tribout    ,1:itriblev) ) 
   trib_allocated = .true.
 endif 

 ! If tribheat == 1, tributaries' and effluent's data are read once, 
 ! and kept constant throughout model integration time
 if (tribheat%par == 1 .and. trib_first_read_2d(ix,iy)) &
 &  call TRIBUPDATE(spinup_done,gsp,wst,area_int,h1,dhwtrib,trib_inflow,dt,trib_level_2d(ix,iy),ix,iy)

 if (tribheat%par == 2 .or. tribheat%par == 3) then 
   work = mod(time - dt, dttribupdate*day_sec) ! time is for next time step, thus lateral inflow
                                               ! values are updated to the current time step
   if ((work >= 0.d0 .and. work < dt) .or. trib_first_read_2d(ix,iy) .or. spinup_done) then
     ! Updating tributary data
     call TRIBUPDATE(spinup_done,gsp,wst,area_int,h1,dhwtrib,trib_inflow,dt,trib_level_2d(ix,iy),ix,iy)
   endif
 endif

 if (tribheat%par == 4) &
 &  call TRIBUPDATE(spinup_done,gsp,wst,area_int,h1,dhwtrib,trib_inflow,dt,trib_level_2d(ix,iy),ix,iy)



 allocate( U_tribin_(1:N_tribin%par,1:M+1), T_tribin_(1:N_tribin%par,1:M+1), &
 & Sal_tribin_(1:N_tribin%par,1:M+1), Ux_tribin_(1:N_tribin%par,1:M+1), &
 & Uy_tribin_(1:N_tribin%par,1:M+1), U_tribout_(1:N_tribout,1:M+1), &
 & meth_tribin_(1:N_tribin%par,1:M+1), oxyg_tribin_(1:N_tribin%par,1:M+1), &
 & DOC_tribin_(1:N_tribin%par,1:M+1), &
 & DIC_tribin_(1:N_tribin%par,1:M+1), POC_tribin_(1:N_tribin%par,1:M+1), &
 & width_tribin_(1:N_tribin%par,1:M+1), width_tribout_(1:N_tribout,1:M+1))

 !Interpolation of tributaries' data on the model grid
 if (itriblev /= M+1) then 
   allocate(z_(1:itriblev))
   forall (i=1:itriblev) z_(i) = (i-1)*h1/real(itriblev-1)
   do k = 1, N_tribin%par
     !call LININTERPOL(z_,U_tribin(k,1:itriblev),itriblev,z_full,U_tribin_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,T_tribin(k,1:itriblev),itriblev,z_full,T_tribin_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,Sal_tribin(k,1:itriblev),itriblev,z_full,Sal_tribin_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,Ux_tribin(k,1:itriblev),itriblev,z_full,Ux_tribin_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,Uy_tribin(k,1:itriblev),itriblev,z_full,Uy_tribin_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,U_tribout(k,1:itriblev),itriblev,z_full,U_tribout_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,width_tribin(k,1:itriblev),itriblev,z_full,width_tribin_(k,1:M+1),M+1,flag)
     !call LININTERPOL(z_,width_tribout(k,1:itriblev),itriblev,z_full,width_tribout_(k,1:M+1),M+1,flag)
     call PIECEWISECONSTINT(z_    , U_tribin (k,1:itriblev),itriblev, &
     &                      z_full, U_tribin_(k,1:M+1     ),M+1     , 2_iintegers)
     call PIECEWISECONSTINT(z_    ,T_tribin (k,1:itriblev),itriblev, &
     &                      z_full,T_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,Sal_tribin (k,1:itriblev),itriblev, &
     &                      z_full,Sal_tribin_(k,1:M+1      ),M+1    ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,meth_tribin (k,1:itriblev),itriblev, &
     &                      z_full,meth_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,oxyg_tribin (k,1:itriblev),itriblev, &
     &                      z_full,oxyg_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,DIC_tribin (k,1:itriblev),itriblev, &
     &                      z_full,DIC_tribin_(k,1:M+1)     ,M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,DOC_tribin (k,1:itriblev),itriblev, &
     &                      z_full,DOC_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,POC_tribin (k,1:itriblev),itriblev, &
     &                      z_full,POC_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,Ux_tribin (k,1:itriblev),itriblev, &
     &                      z_full,Ux_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,Uy_tribin (k,1:itriblev),itriblev, &
     &                      z_full,Uy_tribin_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,width_tribin (k,1:itriblev),itriblev, &
     &                      z_full,width_tribin_(k,1:M+1)     ,M+1     ,2_iintegers)
   enddo
   do k = 1, N_tribout
     call PIECEWISECONSTINT(z_    ,U_tribout (k,1:itriblev),itriblev, &
     &                      z_full,U_tribout_(k,1:M+1     ),M+1     ,2_iintegers)
     call PIECEWISECONSTINT(z_    ,width_tribout (k,1:itriblev),itriblev, &
     &                      z_full,width_tribout_(k,1:M+1     ),M+1     ,2_iintegers)
   enddo 
   deallocate(z_)
 else
   U_tribin_ = U_tribin
   T_tribin_ = T_tribin
   Sal_tribin_ = Sal_tribin
   meth_tribin_ = meth_tribin
   oxyg_tribin_ = oxyg_tribin
   DOC_tribin_ = DOC_tribin
   DIC_tribin_ = DIC_tribin
   POC_tribin_ = POC_tribin
   Ux_tribin_ = Ux_tribin
   Uy_tribin_ = Uy_tribin
   U_tribout_ = U_tribout
   width_tribin_ = width_tribin
   width_tribout_ = width_tribout
 endif

 ! Calculating discharges of all tributaries and an effluent
 forall (i=1:M+1, j=1:N_tribin%par) disch_tribin(j,i) = &
 & width_tribin_(j,i)*U_tribin_(j,i)*gsp%ddz05(i-1)*h1
 !disch_tribout = 0. ! Assuming N_tribout == 1
 forall (i=1:M+1, j=1:N_tribout)   disch_tribout(j,i) = &
 & width_tribout_(j,i)*U_tribout_(j,i)*gsp%ddz05(i-1)*h1

 invdt = 1./dt

 select case(outflpar%par)

 case(2)
!  Values at outflow are calculated before following "averaged Lagrangian" approach
!  Explicit scheme

   print*, 'Case outflpar%par = ', outflpar%par, 'is deprecated: STOP'
   STOP
 
 case default ! outflpar%par == 0 or 1
 
   foutflow = 0. !Methane flux through outflow
   finflow  = 0. !Methane flux by tributaries
   inout_fluxes%fluxes_in (:) = 0.
   inout_fluxes%fluxes_out(:) = 0.

   wst%wArea(M+1) = 0. !Lower boundary condition for vertical speed
   do i = M+1, 1, -1 !1, M+1

     heatinflow = 0.
     salinflow  = 0.
     DICinflow  = 0.
     DOCinflow  = 0.
     POCinflow  = 0.
     methinflow = 0.
     oxyginflow = 0.
     uinflow    = 0.
     vinflow    = 0.
     inflow     = 0.
     do k = 1, N_tribin%par
       heatinflow = heatinflow + U_tribin_(k,i)*T_tribin_(k,i)   *width_tribin_(k,i)
       salinflow  = salinflow  + U_tribin_(k,i)*Sal_tribin_(k,i) *width_tribin_(k,i)
       methinflow = methinflow + U_tribin_(k,i)*meth_tribin_(k,i)*width_tribin_(k,i)
       oxyginflow = oxyginflow + U_tribin_(k,i)*oxyg_tribin_(k,i)*width_tribin_(k,i)
       DICinflow  = DICinflow  + U_tribin_(k,i)*DIC_tribin_(k,i) *width_tribin_(k,i)
       DOCinflow  = DOCinflow  + U_tribin_(k,i)*DOC_tribin_(k,i) *width_tribin_(k,i)
       POCinflow  = POCinflow  + U_tribin_(k,i)*POC_tribin_(k,i) *width_tribin_(k,i)
       uinflow    = uinflow    + U_tribin_(k,i)*Ux_tribin_(k,i)  *width_tribin_(k,i)
       vinflow    = vinflow    + U_tribin_(k,i)*Uy_tribin_(k,i)  *width_tribin_(k,i)
       inflow     = inflow     + U_tribin_(k,i)                  *width_tribin_(k,i)
     enddo
     heatinflow = heatinflow/area_int(i)
     salinflow  = salinflow /area_int(i)
     methinflow = methinflow/area_int(i)
     oxyginflow = oxyginflow/area_int(i)
     DICinflow  = DICinflow /area_int(i)
     DOCinflow  = DOCinflow /area_int(i)
     POCinflow  = POCinflow /area_int(i)
     uinflow    = uinflow   /area_int(i)
     vinflow    = vinflow   /area_int(i)

     outflow = 0.
     do k = 1, N_tribout
       outflow = outflow + U_tribout_(k,i)*width_tribout_(k,i)
     enddo
     !Solution of horizontally averaged continuity equation
     wst%wArea(i-1) = wst%wArea(i) - gsp%ddz05(i-1)*h1*(inflow - outflow) 
     outflow = outflow/area_int(i)
     outflow = max(outflow,0.) !Ensuring positivity of outflow discharge, used for advection only

     meanTin = 0.
     meansalin = 0.
     meanmethin = 0.
     meanoxygin = 0.
     meanDICin = 0.
     meanDOCin = 0.
     meanPOCin = 0.
     meanuin = 0.
     meanvin = 0.
     if (outflpar%par == 1) then
       do k = 1, N_tribin%par
         meanTin = meanTin + T_tribin_(k,i)*width_tribin_(k,i)
         meansalin = meansalin + Sal_tribin_(k,i)*width_tribin_(k,i)
         meanmethin = meanmethin + meth_tribin_(k,i)*width_tribin_(k,i)
         meanoxygin = meanoxygin + oxyg_tribin_(k,i)*width_tribin_(k,i)
         meanDICin = meanDICin + DIC_tribin_(k,i)*width_tribin_(k,i)
         meanDOCin = meanDOCin + DOC_tribin_(k,i)*width_tribin_(k,i)
         meanPOCin = meanPOCin + POC_tribin_(k,i)*width_tribin_(k,i)
         meanuin = meanuin + Ux_tribin_(k,i)*width_tribin_(k,i)
         meanvin = meanvin + Uy_tribin_(k,i)*width_tribin_(k,i)
       enddo
       work = sum(width_tribin(1:N_tribin%par,i))
       meanTin = meanTin/work
       meansalin = meansalin/work
       meanmethin = meanmethin/work
       meanoxygin = meanoxygin/work
       meanDICin = meanDICin/work
       meanDOCin = meanDOCin/work
       meanPOCin = meanPOCin/work
       meanuin = meanuin/work
       meanvin = meanvin/work
     endif

     !Implicit scheme for the outflow term
     !Check: if for any of tributaries the data on a variable is missing,
     !this variable is not updated with inlet/outlet sources
     work = invdt + outflow*(1. + outflpar%par)

     k = nheat
     if (.not. any(T_tribin_(:,i) == missing_value)) then 
       wst%Tw1(i) = (invdt*wst%Tw1(i) + heatinflow + outflpar%par*outflow*meanTin)/work
       call ADD2FLUX(heatinflow,meanTin,wst%Tw1(i),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       wst%Tw1(i) = (invdt*wst%Tw1(i) )/work
       call ADD2FLUX(0._ireals,0._ireals,wst%Tw1(i),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

     k = nsal
     if (.not. any(Sal_tribin_(:,i) == missing_value)) then
       wst%Sal1(i) = (invdt*wst%Sal1(i) + salinflow + outflpar%par*outflow*meansalin)/work
       call ADD2FLUX(salinflow,meansalin,wst%Sal1(i),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       wst%Sal1(i) = (invdt*wst%Sal1(i) )/work
       call ADD2FLUX(0._ireals,0._ireals,wst%Sal1(i),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

     k = noxyg
     if (.not. any(oxyg_tribin_(:,i) == missing_value)) then
       gas%oxyg(i,1) = (invdt*gas%oxyg(i,1) + oxyginflow + outflpar%par*outflow*meanoxygin)/work
       call ADD2FLUX(oxyginflow,meanoxygin,gas%oxyg(i,1),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       gas%oxyg(i,1) = (invdt*gas%oxyg(i,1) )/work
       call ADD2FLUX(0._ireals,0._ireals,gas%oxyg(i,1),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

     k = nmeth
     if (.not. any(meth_tribin_(:,i) == missing_value)) then 
       gas%qwater(i,1) = (invdt*gas%qwater(i,1) + methinflow + outflpar%par*outflow*meanmethin)/work
       call ADD2FLUX(methinflow,meanmethin,gas%qwater(i,1),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       gas%qwater(i,1) = (invdt*gas%qwater(i,1) )/work
       call ADD2FLUX(0._ireals,0._ireals,gas%qwater(i,1),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

     if (carbon_model%par == 2) then
       ! Inlet DOC is added to the allochtonous DOC pool
       k = nDOC
       if (.not. any(DOC_tribin_(:,i) == missing_value)) then
         gas%DOC(i,2) = (invdt*gas%DOC(i,2) + DOCinflow + outflpar%par*outflow*meanDOCin)/work
         call ADD2FLUX(DOCinflow,meanDOCin,gas%DOC(i,2),i,outflpar%par, &
         &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
       else
         gas%DOC(i,2) = (invdt*gas%DOC(i,2) )/work
         call ADD2FLUX(0._ireals,0._ireals,gas%DOC(i,2),i,0_iintegers, &
         &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
       endif
       gas%DOC(i,1) = (invdt*gas%DOC(i,1) )/work
       call ADD2FLUX(0._ireals,0._ireals,gas%DOC(i,1),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
       k = nPOCL
       if (.not. any(POC_tribin_(:,i) == missing_value)) then
         gas%POCL(i) = (invdt*gas%POCL(i) + POCinflow + outflpar%par*outflow*meanPOCin)/work
         call ADD2FLUX(POCinflow,meanPOCin,gas%POCL(i),i,outflpar%par, &
         &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
       else
         gas%POCL(i) = (invdt*gas%POCL(i) )/work
         call ADD2FLUX(0._ireals,0._ireals,gas%POCL(i),i,0_iintegers, &
         &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
       endif
       k = nPOCD
       gas%POCD(i) = (invdt*gas%POCD(i) )/work !> %todo: include influx of POCD by tributaries
       call ADD2FLUX(0._ireals,0._ireals,gas%POCD(i),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
       k = nphosph
       gas%phosph(i) = (invdt*gas%phosph(i) )/work !> %todo: include influx of DIP by tributaries
       call ADD2FLUX(0._ireals,0._ireals,gas%phosph(i),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

     k = nDIC
     if (.not. any(DIC_tribin_(:,i) == missing_value)) then
       gas%DIC(i,1) = (invdt*gas%DIC(i,1) + DICinflow + outflpar%par*outflow*meanDICin)/work
       call ADD2FLUX(DICinflow,meanDICin,gas%DIC(i,1),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       gas%DIC(i,1) = (invdt*gas%DIC(i,1) )/work
       call ADD2FLUX(0._ireals,0._ireals,gas%DIC(i,1),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

     k = nu
     if (.not. any(Ux_tribin_(:,i) == missing_value)) then
       wst%u1(i) = (invdt*wst%u1(i) + uinflow + outflpar%par*outflow*meanuin)/work
       call ADD2FLUX(uinflow,meanuin,wst%u1(i),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       wst%u1(i) = (invdt*wst%u1(i) )/work
       call ADD2FLUX(0._ireals,0._ireals,wst%u1(i),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif
     k = nv
     if (.not. any(Uy_tribin_(:,i) == missing_value)) then
       wst%v1(i) = (invdt*wst%v1(i) + vinflow + outflpar%par*outflow*meanvin)/work
       call ADD2FLUX(vinflow,meanvin,wst%v1(i),i,outflpar%par, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     else
       wst%v1(i) = (invdt*wst%v1(i) )/work
       call ADD2FLUX(0._ireals,0._ireals,wst%v1(i),i,0_iintegers, &
       &                inout_fluxes%fluxes_in(k),inout_fluxes%fluxes_out(k))
     endif

   enddo

 end select 
 !Input/outputs by inlets and outlets are normalized by surface area
 inout_fluxes%fluxes_in (:) = inout_fluxes%fluxes_in (:)/area_int(1)
 inout_fluxes%fluxes_out(:) = inout_fluxes%fluxes_out(:)/area_int(1)
 k = nmeth
 finflow  = inout_fluxes%fluxes_in (k) !Methane flux by inlets
 foutflow = inout_fluxes%fluxes_out(k) !Methane flux by outlets

 !Diagnosing horizontally averaged vertical speed
 wst%w(0) = wst%wArea(0)/area_int(1)
 wst%w(M+1) = wst%wArea(M+1)/area_int(M+1)
 forall(i = 1:M) wst%w(i) = wst%wArea(i)/area_half(i)

 !Correction of dhwtrib (level change due to tributaries)
 if (trib_level_2d(ix,iy) /= missing_value) then
   dhwtrib = (sum(disch_tribin(:,:)) - sum(disch_tribout(:,:)))* &
   & dt/area_int(1)
 endif
 !Currently, tribheat = 4 includes nudging water level to initial value (constant)
 if (tribheat%par == 4) dhwtrib = (h10 - h1)/10.

 deallocate(U_tribin_, T_tribin_, Sal_tribin_, Ux_tribin_, Uy_tribin_, U_tribout_, &
 & width_tribin_, width_tribout_, DIC_tribin_, DOC_tribin_, &
 & POC_tribin_, meth_tribin_, oxyg_tribin_)

 trib_first_read_2d(ix,iy) = .false.
 RETURN

 contains

 !> Subroutine ADD2FLUX adds to the total flux by inlets and outlets the flux at i-th level
 SUBROUTINE ADD2FLUX(xinflow,meanxin,xi,i,noutflpar,flux_in,flux_out)
 implicit none
 real   (kind=ireals   ), intent(in   ) :: xinflow   !< Inflow by tributaries of variable x at i-th level
 real   (kind=ireals   ), intent(in   ) :: meanxin   !< Mean of variable x in tributaries at i-th level
 real   (kind=ireals   ), intent(in   ) :: xi        !< Variable x at i-th level in lake
 integer(kind=iintegers), intent(in   ) :: i         !< Level index
 integer(kind=iintegers), intent(in   ) :: noutflpar !< Switch for outflow parameterization
 real   (kind=ireals   ), intent(inout) :: flux_in   !< Total flux of x by inlets
 real   (kind=ireals   ), intent(inout) :: flux_out  !< Total flux of x by outlets 
 flux_in  = flux_in  + xinflow*area_int(i)*gsp%ddz05(i-1)*h1
 flux_out = flux_out + &
 & outflow*area_int(i)*(xi*(1. + noutflpar) - noutflpar*meanxin)*gsp%ddz05(i-1)*h1
 END SUBROUTINE ADD2FLUX

 END SUBROUTINE TRIBTEMP


!> Subroutine TRIBUPDATE updates tributary data
SUBROUTINE TRIBUPDATE(spinup_done,gsp,wst,area_int,h1,dhwtrib,trib_inflow,dt,level,ix,iy)

use INOUT_PARAMETERS, only : &
& lake_misc_unit_min, &
& lake_misc_unit_max

use DRIVING_PARAMS, only : &
& N_tribin, & 
& N_triblev, & 
& N_tribout, &
& U_tribin, &
& U_tribout, &
& T_tribin, &
& Sal_tribin, &
& DIC_tribin, &
& DOC_tribin, &
& POC_tribin, &
& meth_tribin, &
& oxyg_tribin, &
& Ux_tribin, &
& Uy_tribin, &
& width_tribin, &
& width_tribout, &
& disch_tribin, &
& disch_tribout, &
& M, &
& eos,lindens, &
& missing_value, &
& tribheat, &
& fileinflow, &
& fileoutflow, &
& dttribupdate, &
& nvartribext, &
& READPROFILE

use INOUT, only : CHECK_UNIT
use PHYS_CONSTANTS, only : pref
use WATER_DENSITY, only : DENSITY_W

implicit none

! Input variables
logical, intent(in) :: spinup_done

type(gridspacing_type), intent(in) :: gsp
type(waterstate_type) , intent(in) :: wst

real(kind=ireals), intent(in) :: area_int(1:M+1)
real(kind=ireals), intent(in) :: h1 
real(kind=ireals), intent(inout) :: dhwtrib
real(kind=ireals), intent(in) :: trib_inflow(1:nvartribext)
real(kind=ireals), intent(in) :: dt
!Output variables
real(kind=ireals), intent(inout) :: level
integer(kind=iintegers), intent(in) :: ix, iy

! Local variables
integer(kind=iintegers), parameter :: mixtype = 0 !> Indicates the type of mixing of tributaries water at the
                                                  !! entrance to the lake, relevant it tribheat$par == 3

real(kind=ireals), allocatable :: work(:,:)
real(kind=ireals) :: disch, depth0, depth, width, temp, sal, &
& dirprojX, dirprojY, DOC, POC, DIC, CH_4, O_2, xx, rowtrib, disch_inflow

integer(kind=iintegers) :: ncol, i, j, idep, idep0

character(len=20) :: workchar

if (.not. trib_open_2d(ix,iy)) then
  call CHECK_UNIT(lake_misc_unit_min,lake_misc_unit_max,trib_unit1_2d(ix,iy))
  trib_unit2_2d(ix,iy) = trib_unit1_2d(ix,iy) + 1
  call CHECK_UNIT(lake_misc_unit_min,lake_misc_unit_max,trib_unit2_2d(ix,iy))
  open (unit=trib_unit1_2d(ix,iy),file='./meteo/'//fileinflow(1:len_trim(fileinflow)),status='old')
  open (unit=trib_unit2_2d(ix,iy),file='./meteo/'//fileoutflow(1:len_trim(fileoutflow)),status='old')
  trib_open_2d(ix,iy) = .true.
endif

if (spinup_done) then
  rewind trib_unit1_2d(ix,iy)
  rewind trib_unit2_2d(ix,iy)
endif

if_tribform : if (tribheat%par == 2) then
  !Reading inflow profiles from file
  ncol = 12
  allocate (work(itriblev,ncol))
  do i = 1, N_tribin%par
    call READPROFILE(trib_unit1_2d(ix,iy),itriblev,ncol,work) 
    width_tribin(i,:) = work (:,2)
    U_tribin    (i,:) = work (:,3)
    T_tribin    (i,:) = work (:,4)
    Sal_tribin  (i,:) = work (:,5)
    Ux_tribin   (i,:) = work (:,6)
    Uy_tribin   (i,:) = work (:,7)
    DOC_tribin  (i,:) = work (:,8)
    POC_tribin  (i,:) = work (:,9)
    DIC_tribin  (i,:) = work (:,10)
    meth_tribin (i,:) = work (:,11)
    oxyg_tribin (i,:) = work (:,12)
  enddo
  deallocate(work)

  ncol = 3
  allocate (work(itriblev,ncol))
  call READPROFILE(trib_unit2_2d(ix,iy),itriblev,ncol,work) 
  ! In current version there is only ONE outflow 
  width_tribout(1,:) = work (:,2)
  U_tribout    (1,:) = work (:,3)
  deallocate(work)

else if (tribheat%par == 3) then

  ! Reading bulk characteristics of inlets and an outlet.
  ! The data for each inlet in an input file should be given in rows containing:
  ! timestring, discharge, depth, width, temperature, salinity, riverflow direction
  ! projection on X axis (-1,+1), riverflow direction projection on Y axis (-1,+1),
  ! DOC, POC, DIC, CH_4

  disch_inflow = 0.
  do i = 1, N_tribin%par

    read(trib_unit1_2d(ix,iy),*) xx, disch, depth, width, temp, sal, &
    & dirprojX, dirprojY, DOC, POC, DIC, CH_4, O_2

    if (mixtype == 1 .and. temp /= missing_value) then
      ! Assuming that inlet properties are advected at levels from surface to the lake depth 
      ! where the water density equals to that of an inlet
      depth = DEPTH_TRIB_MIX()
    endif

    idep = 1
    do while (gsp%z_half(idep)/depth < 1.) 
      idep = idep + 1
      if (idep == M+1) exit
    enddo
    if (idep /= 1_iintegers) idep = idep - 1
    depth = gsp%z_half(idep)

    width_tribin(i,1:idep) = width                          
    U_tribin    (i,1:idep) = disch/(depth*width)            
    T_tribin    (i,1:idep) = temp                           
    Sal_tribin  (i,1:idep) = sal                            
    if (dirprojX /= missing_value) then
      Ux_tribin   (i,1:idep) = disch/(depth*width)*dirprojX 
    else
      Ux_tribin   (i,1:idep) = missing_value
    endif
    if (dirprojY /= missing_value) then
      Uy_tribin   (i,1:idep) = disch/(depth*width)*dirprojY
    else
      Uy_tribin   (i,1:idep) = missing_value
    endif
    DOC_tribin  (i,1:idep) = DOC                            
    POC_tribin  (i,1:idep) = POC                            
    DIC_tribin  (i,1:idep) = DIC                            
    meth_tribin (i,1:idep) = CH_4                               
    oxyg_tribin (i,1:idep) = O_2                                
    if (idep < M+1) then
      width_tribin(i,idep+1:M+1) = 0.
      U_tribin    (i,idep+1:M+1) = 0.
      T_tribin    (i,idep+1:M+1) = 0.
      Sal_tribin  (i,idep+1:M+1) = 0.
      Ux_tribin   (i,idep+1:M+1) = 0.
      Uy_tribin   (i,idep+1:M+1) = 0.
      DOC_tribin  (i,idep+1:M+1) = 0.
      POC_tribin  (i,idep+1:M+1) = 0.
      DIC_tribin  (i,idep+1:M+1) = 0.
      meth_tribin (i,idep+1:M+1) = 0.
      oxyg_tribin (i,idep+1:M+1) = 0.
    endif

    disch_inflow = disch_inflow + disch
  enddo

  ! In current version there is only ONE outflow
  ! "level" is the input maximal lake depth, if available
  read(trib_unit2_2d(ix,iy),*) workchar, disch, depth0, depth, width, level
  !disch = 0.
  !Depth of upper bound of outlet stream
  idep0 = 1
  do while (gsp%z_half(idep0)/(depth0+small_value) < 1.) 
    idep0 = idep0 + 1
    if (idep0 == M+1) exit
  enddo
  if (idep0 /= 1_iintegers) idep0 = idep0 - 1
  if (idep0 == 1) then
    depth0 = 0.
  else
    depth0 = gsp%z_half(idep0-1)
  endif
  !Depth of lower bound of outlet stream
  idep = 1
  do while (gsp%z_half(idep)/depth < 1.) 
    idep = idep + 1
    if (idep == M+1) exit
  enddo
  if (idep /= 1_iintegers) idep = idep - 1
  depth = gsp%z_half(idep)

  !If data on the water level is available, outlet discharge is chosen to keep
  !this level (i.e. as a residual of water balance equation)
  if (level /= missing_value) then
    disch = - area_int(1)*(level-h1)/(dttribupdate*day_sec)+ disch_inflow
  endif

  width_tribout(1,idep0:idep) = width  
  U_tribout    (1,idep0:idep) = disch/((depth-depth0)*width)
  if (idep0 > 1) then
    width_tribout(1,1:idep0-1) = 0.
    U_tribout    (1,1:idep0-1) = 0.
  endif
  if (idep < M+1) then
    width_tribout(1,idep+1:M+1) = 0.
    U_tribout    (1,idep+1:M+1) = 0.
  endif

else if (tribheat%par == 4) then

  ! Taking bulk characteristics of a single inlet from trib_inflow, provided by the LAKE model driver.
  ! The data for the inlet is discharge, depth, width, temperature, ... (possibly other scalars)

  if (N_tribin%par /= 1_iintegers) print*, 'N_tribin must be 1 under tribheat = 4'

  disch = trib_inflow(1)
  depth = trib_inflow(2)
  width = trib_inflow(3)
  temp  = trib_inflow(4)
  dirprojX = 1. !assuming the inflow directs eastward
  dirprojY = 0.
  !Other scalars, which can be added:
  !sal, dirprojX, dirprojY, DOC, POC, DIC, CH_4, O_2
  sal = missing_value
  DOC = missing_value
  POC = missing_value
  DIC = missing_value
  CH_4 = missing_value
  O_2 = missing_value

  if (mixtype == 1 .and. temp /= missing_value) then
    ! Assuming that inlet properties are advected at levels from surface to the lake depth 
    ! where the water density equals to that of an inlet
    depth = DEPTH_TRIB_MIX()
  endif

  idep = 1
  do while (gsp%z_half(idep)/depth < 1.) 
    idep = idep + 1
    if (idep == M+1) exit
  enddo
  if (idep /= 1_iintegers) idep = idep - 1
  depth = gsp%z_half(idep)

  i = 1
  width_tribin(i,1:idep) = width                          
  U_tribin    (i,1:idep) = disch/(depth*width)            
  T_tribin    (i,1:idep) = temp                           
  Sal_tribin  (i,1:idep) = sal                            
  if (dirprojX /= missing_value) then
    Ux_tribin   (i,1:idep) = disch/(depth*width)*dirprojX 
  else
    Ux_tribin   (i,1:idep) = missing_value
  endif
  if (dirprojY /= missing_value) then
    Uy_tribin   (i,1:idep) = disch/(depth*width)*dirprojY
  else
    Uy_tribin   (i,1:idep) = missing_value
  endif
  DOC_tribin  (i,1:idep) = DOC                            
  POC_tribin  (i,1:idep) = POC                            
  DIC_tribin  (i,1:idep) = DIC                            
  meth_tribin (i,1:idep) = CH_4                               
  oxyg_tribin (i,1:idep) = O_2                                
  if (idep < M+1) then
    width_tribin(i,idep+1:M+1) = 0.
    U_tribin    (i,idep+1:M+1) = 0.
    T_tribin    (i,idep+1:M+1) = 0.
    Sal_tribin  (i,idep+1:M+1) = 0.
    Ux_tribin   (i,idep+1:M+1) = 0.
    Uy_tribin   (i,idep+1:M+1) = 0.
    DOC_tribin  (i,idep+1:M+1) = 0.
    POC_tribin  (i,idep+1:M+1) = 0.
    DIC_tribin  (i,idep+1:M+1) = 0.
    meth_tribin (i,idep+1:M+1) = 0.
    oxyg_tribin (i,idep+1:M+1) = 0.
  endif
  disch_inflow = disch

  depth0 = 0.
  disch = disch_inflow ! The inflow and outflow are equal, ensuring constant level
  !Depth of upper bound of outlet stream
  idep0 = 1
  !do while (gsp%z_half(idep0)/(depth0 + small_value) < 1.) 
  !  idep0 = idep0 + 1
  !  if (idep0 == M+1) exit
  !enddo
  !if (idep0 /= 1_iintegers) idep0 = idep0 - 1
  !if (idep0 == 1) then
  !  depth0 = 0.
  !else
  !  depth0 = gsp%z_half(idep0-1)
  !endif
  !Depth of lower bound of outlet stream
  !idep = 1
  !do while (gsp%z_half(idep)/depth < 1.) 
  !  idep = idep + 1
  !  if (idep == M+1) exit
  !enddo
  !if (idep /= 1_iintegers) idep = idep - 1
  !depth = gsp%z_half(idep)

  width_tribout(1,idep0:idep) = width  
  U_tribout    (1,idep0:idep) = disch/((depth-depth0)*width)
  if (idep0 > 1) then
    width_tribout(1,1:idep0-1) = 0.
    U_tribout    (1,1:idep0-1) = 0.
  endif
  if (idep < M+1) then
    width_tribout(1,idep+1:M+1) = 0.
    U_tribout    (1,idep+1:M+1) = 0.
  endif

endif if_tribform


contains 

!> Computing the lake depth 
!! where the water density equals to that of an inlet
FUNCTION DEPTH_TRIB_MIX() result(depth_mix)
implicit none
real(kind=ireals) :: depth_mix
allocate(work(1:M+1,1))
xx = sal
if (sal == missing_value) xx = 0.
call DENSITY_W(0_iintegers,eos%par,lindens%par,(/temp/),(/xx/),(/pref/),work(1,1))
rowtrib = work(1,1)
work(1:M+1,1) = abs(wst%row(1:M+1) - rowtrib)
j = minloc(work(1:M+1,1),1)
depth_mix = gsp%z_half(j)
deallocate(work)
END FUNCTION DEPTH_TRIB_MIX

END SUBROUTINE TRIBUPDATE 

!> Calculates artificial abstraction rate (m**3/s) 
!! for the new year dependent
!! on reseroir volume (m**3) at the end of previous year.
FUNCTION ABSTR(vol) 

implicit none

! Input variables
real(kind=ireals), intent(in) :: vol

! Output variables
real(kind=ireals) :: ABSTR

!Local variables
real(kind=ireals), parameter :: coef0 = 0.2770399597
real(kind=ireals), parameter :: coef1 = -0.0344416893
real(kind=ireals), parameter :: coef2 = 0.0153926477

ABSTR = coef2*vol*vol + coef1*vol + coef0

END FUNCTION ABSTR

!> lakebatch multi-point: swap this module's variables for the current point;
!! called by LAKE next to LBC_MODULES, at entry.
SUBROUTINE LBC_TRIBUTARIES()
call lbc_begin(LBC_SCOPE_TRIBUTARIES)
include 'lb_ctx_tributaries.inc'
END SUBROUTINE LBC_TRIBUTARIES

END MODULE TRIBUTARIES
