MODULE VERTADV

use LAKE_DATATYPES, only : ireals, iintegers
use NUMERIC_PARAMS, only : small_value
use NUMERICS, only : BOUND_FROM_BELOW

contains
!> Subroutine VERTADV_UPDATE performs update  
!! of horizontal momentum and scalars by vertical advection
SUBROUTINE VERTADV_UPDATE(M,dt,h1,gsp,bathymw,wst,gas)
use ARRAYS,            only : gas_type
use ARRAYS_GRID,       only : gridspacing_type
use ARRAYS_BATHYM,     only : bathym
use ARRAYS_WATERSTATE, only : waterstate_type
use NUMERICS,          only : SMOOTHER_CONSERV
use DZETA_MOD,         only : VARMEAN
use METH_OXYG_CONSTANTS, only : molm3tomgl_o2
use DRIVING_PARAMS, only : carbon_model

implicit none
!Input/output variables
integer(kind=iintegers), intent(in   ) :: M !> Number of layers
real   (kind=ireals)   , intent(in   ) :: h1 !> Water depth
real   (kind=ireals)   , intent(in   ) :: dt !> Timestep
type(gridspacing_type) , intent(in   ) :: gsp !> Grid spacing group
type(bathym          ) , intent(in   ) :: bathymw(1:M+1) !> Bathymetry group
type(waterstate_type ) , intent(inout) :: wst !> Water state group
type(gas_type        ) , intent(inout) :: gas !> Biogeochemical properties

!Local variables
integer(kind=iintegers) :: i
real   (kind=ireals)    :: dz(0:M+1), x, xx
logical :: log1, log2


! Updating variables due to vertical advection
call VERTICAL_ADVECTION(wst%Tw1 )
call VERTICAL_ADVECTION(wst%Sal1)
call VERTICAL_ADVECTION(wst%u1  )
call VERTICAL_ADVECTION(wst%v1  )
call VERTICAL_ADVECTION(gas%oxyg  (:,1))
call VERTICAL_ADVECTION(gas%qwater(:,1))
if (carbon_model%par == 2) then
  call VERTICAL_ADVECTION(gas%DIC   (:,1)) 
  call VERTICAL_ADVECTION(gas%DOC   (:,1))!autochtonous DOC
  call VERTICAL_ADVECTION(gas%DOC   (:,2))!allochtonous DOC
  call VERTICAL_ADVECTION(gas%POCL       )
  call VERTICAL_ADVECTION(gas%POCD       )
endif
call VERTICAL_ADVECTION(gas%phosph     )

!! Conservative filtering
!dz(0) = 0.
!dz(M+1) = 0.
!forall(i = 1:M) dz(i) = gsp%ddz(i)*h1
!call SMOOTHER_CONSERV(wst%Tw1,dz,M+1)
!call SMOOTHER_CONSERV(wst%u1 ,dz,M+1)
!call SMOOTHER_CONSERV(wst%v1 ,dz,M+1)
!call SMOOTHER_CONSERV(gas%oxyg  (:,1),dz,M+1)
!call SMOOTHER_CONSERV(gas%qwater(:,1),dz,M+1)
!call SMOOTHER_CONSERV(gas%DIC   (:,1),dz,M+1)
!call SMOOTHER_CONSERV(gas%phosph,dz,M+1)


contains
!> Subroutine VERTICAL_ADVECTION performs vertical advection 
!! of a given variable
SUBROUTINE VERTICAL_ADVECTION(var)
implicit none
real(kind=ireals), intent(inout) :: var(1:M+1)
!Local variables
real   (kind=ireals), allocatable :: varint(:), fluxes(:), work(:)
integer(kind=iintegers) :: i 

!!Second-order non-TVD central difference scheme
!allocate(varint(0:M+1))
!varint(0) = var(1)
!do i = 1, M
!  varint(i) = 0.5*(var(i) + var(i+1))
!enddo
!varint(M+1) = var(M+1)
!do i = 1, M+1
!  var(i) = var(i) - dt / bathymw(i)%area_int * &
!  & (wst%wArea(i)*varint(i) - wst%wArea(i-1)*varint(i-1)) / (h1*gsp%ddz05(i-1))
!enddo
!deallocate(varint)

!MUSCL scheme
allocate(work(1:M+1),fluxes(0:M+1))
forall(i=1:M+1) work(i) = var(i)*bathymw(i)%area_int
call MUSCL_FLUXES(M,work,wst%w,fluxes)
do i = 1, M+1
  var(i) = var(i) - dt / bathymw(i)%area_int * &
  & (fluxes(i) - fluxes(i-1)) / (h1*gsp%ddz05(i-1))
enddo
deallocate(work,fluxes)

END SUBROUTINE VERTICAL_ADVECTION

!!> Subroutine MUSCL_FLUXES computes flux following MUSCL scheme
!!! assuming equidistant grid
!SUBROUTINE MUSCL_FLUXES(var,flux)
!implicit none
!real(kind=ireals), intent(in ) :: var (1:M+1)
!real(kind=ireals), intent(out) :: flux(0:M+1)
!real(kind=ireals) :: r, varL, varR
!integer(kind=iintegers) :: i 
!
!flux(0  ) = wst%w(0  )*var(1  )
!flux(M+1) = wst%w(M+1)*var(M+1)
!
!!Interpolation scheme below assumes the speed is positive
!do i = 2, M-1
!  r = (var(i) - var(i-1) + small_value)/(var(i+1) - var(i) + small_value)
!  varL = var(i) + 0.5*FLUX_LIMITER(r)*(var(i+1) - var(i))
!  r = (var(i+1) - var(i) + small_value)/(var(i+2) - var(i+1) + small_value)
!  varR = var(i+1) - 0.5*FLUX_LIMITER(r)*(var(i+2) - var(i+1))
!  if (varR*varL < 0.) then
!    flux(i) = wst%w(i)*0.5*(varR + varL) !sqrt(varR*varL)
!  else
!    flux(i) = wst%w(i)*sqrt(varR*varL)*sign(1._ireals,varR)
!  endif
!enddo
!
!i = 1
!r = (var(i+1) - var(i) + small_value)/(var(i+2) - var(i+1) + small_value)
!varR = var(i+1) - 0.5*FLUX_LIMITER(r)*(var(i+2) - var(i+1))
!flux(i) = wst%w(i)*varR
!
!i = M
!r = (var(i) - var(i-1) + small_value)/(var(i+1) - var(i) + small_value)
!varL = var(i) + 0.5*FLUX_LIMITER(r)*(var(i+1) - var(i))
!flux(i) = wst%w(i)*varL
!
!END SUBROUTINE MUSCL_FLUXES

END SUBROUTINE VERTADV_UPDATE

!> Subroutine MUSCL_FLUXES computes flux following MUSCL scheme
!! assuming equidistant grid
SUBROUTINE MUSCL_FLUXES(M,var,w,flux)
implicit none
integer(kind=iintegers), intent(in   ) :: M !> Number of layers
real(kind=ireals), intent(in ) :: var (1:M+1) !>Variable to be advected
real(kind=ireals), intent(in ) :: w   (0:M+1) !>Vertical speed
real(kind=ireals), intent(out) :: flux(0:M+1) !>Variable flux due to advection
real(kind=ireals) :: r, varL, varR, dv1, dv2
integer(kind=iintegers) :: i 


!Interpolation scheme below assumes the speed is positive
do i = 2, M-1
  dv1 = BOUND_FROM_BELOW(var(i  ) - var(i-1),small_value)
  dv2 = BOUND_FROM_BELOW(var(i+1) - var(i  ),small_value)
  r = dv1/dv2
  varL = var(i  ) + 0.5*FLUX_LIMITER(r)*(var(i+1) - var(i  ))
  !print*, i, w(i), r, varL, FLUX_LIMITER(r)
  dv1 = BOUND_FROM_BELOW(var(i+1) - var(i  ),small_value)
  dv2 = BOUND_FROM_BELOW(var(i+2) - var(i+1),small_value)
  r = dv1/dv2
  varR = var(i+1) - 0.5*FLUX_LIMITER(r)*(var(i+2) - var(i+1))
  !if (varR*varL < 0.) then
  !  flux(i) = w(i)*0.5*(varR + varL) !sqrt(varR*varL)
  !else
  !  flux(i) = w(i)*sqrt(varR*varL)*sign(1._ireals,varR)
  !endif
  ! Interpolation following Kurganov and Tadmor, 2000, J. Comp. Phys.
  if (w(i) > 0.) then
    flux(i) = w(i)*varL
  else
    flux(i) = w(i)*varR
  endif
enddo

i = 0
flux(i) = w(i)*var(i+1)

i = 1
if (w(i) > 0.) then
  flux(i) = w(i)*0.5*(var(i) + var(i+1))
else
  dv1 = BOUND_FROM_BELOW(var(i+1) - var(i  ),small_value)
  dv2 = BOUND_FROM_BELOW(var(i+2) - var(i+1),small_value)
  r = dv1/dv2
  varR = var(i+1) - 0.5*FLUX_LIMITER(r)*(var(i+2) - var(i+1))
  flux(i) = w(i)*varR
endif

i = M
if (w(i) > 0.) then
  dv1 = BOUND_FROM_BELOW(var(i  ) - var(i-1),small_value)
  dv2 = BOUND_FROM_BELOW(var(i+1) - var(i  ),small_value)
  r = dv1/dv2
  varL = var(i) + 0.5*FLUX_LIMITER(r)*(var(i+1) - var(i))
  flux(i) = w(i)*varL
else
  flux(i) = w(i)*0.5*(var(i) + var(i+1))
endif

i = M+1
flux(i) = w(i)*var(i)

END SUBROUTINE MUSCL_FLUXES

!> Flux limiter for TVD scheme
FUNCTION FLUX_LIMITER(r) result(fl)
implicit none
real(kind=ireals), intent(in) :: r
real(kind=ireals) :: fl

!minmod limiter (Roe, 1986)
fl = max(0.,min(1.,r))

END FUNCTION FLUX_LIMITER


END MODULE VERTADV
