MODULE DIFFUSION_LAKE_MOD

use LAKE_DATATYPES, only : ireals, iintegers
use NUMERIC_PARAMS, only : nveclen

implicit none

contains

!> Subroutine DIFFUSION performs one timestep in diffusion
!! using Crank-Nicolson scheme
SUBROUTINE DIFFUSION(N,bctype,bcs,dt,h,dh,dh0, &
& ddz,ddz05,dzeta_int,var1,var2,diff,lsh,bath)
! Modules 
use T_SOLVER_MOD, only : DIFF_COEF
use ARRAYS_BATHYM,  only : bathym
use NUMERICS, only : PROGONKA, STEP
use NUMERIC_PARAMS, only : nveclen
implicit none
!Input variables
integer(kind=iintegers), intent(in) :: N           !> Number of levels of finite-difference grid
integer(kind=iintegers), intent(in) :: bctype(1:2) !> Kind of boundary conditions
real(kind=ireals)      , intent(in) :: dt          !> Timestep
real(kind=ireals)      , intent(in) :: h           !> Physical layer thickness size
real(kind=ireals)      , intent(in) :: dh          !> Increment of layer thickness
real(kind=ireals)      , intent(in) :: dh0         !> Increment of the layer at the top
real(kind=ireals)      , intent(in) :: bcs(1:2)    !> Value of variable or flux at boundaries
real(kind=ireals)      , intent(in) :: diff(1:N-1) !> Diffusion coefficient
real(kind=ireals)      , intent(in) :: ddz(1:N-1)  !> Grid spacing, n/d
real(kind=ireals)      , intent(in) :: ddz05(0:N-1) !> Distance between cells centers, n/dKE
real(kind=ireals)      , intent(in) :: dzeta_int(1:N) !> Non-dimensional coordinate of cell interfaces
real(kind=ireals)      , intent(in) :: lsh(1:N)    !> Exchange with sediments
type(bathym)           , intent(in) :: bath(1:N)   !> Bathymetry
real(kind=ireals)      , intent(in) :: var1(1:N) !> Variable to be diffused at the previous timestep
!Input/output variables
!Output variables
real(kind=ireals)      , intent(out) :: var2(1:N) !> Variable to be diffused at the next timestep:w

!Local variables
real(kind=ireals) :: dt_inv, dt_inv05, x, xx

real(kind=ireals), allocatable :: a(:), b(:), c(:), f(:), y(:)

integer(kind=iintegers) :: i !Loop index

allocate (a(1:nveclen),b(1:nveclen),c(1:nveclen), &
&         f(1:nveclen),y(1:nveclen))
a(:) = 0.; b(:) = 0.; c(:) = 0.; f(:) = 0.; y(:) = 0.

dt_inv = 1./dt
dt_inv05 = 0.5*dt_inv

!Top boundary conditions
if     (bctype(1) == 1) then !Dirichlet condition
  c(1) = 1.
  b(1) = 0.
  f(1) = bcs(1)
elseif (bctype(1) == 2) then !Neumann condition
  xx = 0.5* ( - bath(1)%area_half/bath(1)%area_int * &
  & diff(1)/(h*ddz(1)) + dh0*dt_inv05)
  x  = ddz(1)*h*dt_inv05
  c(1)   = xx - x
  b(1)   = xx
  f(1)   = - x*var1(1) - bcs(1) + b(1)*var1(2) - xx*var1(1) - &
  & x*dt*lsh(1)
endif

!Coefficients for domain interior
do i = 2, N-1
  a(i) = 0.5d0*( - bath(i-1)%area_half*diff(i-1)/(ddz(i-1)*h*bath(i)%area_int) + &
  & dh*dzeta_int(i)*dt_inv05 - dh0*dt_inv05)
  b(i) = 0.5d0*( - bath(i)  %area_half*diff(i)  /(ddz(i)  *h*bath(i)%area_int) - &
  & dh*dzeta_int(i)*dt_inv05 + dh0*dt_inv05)
  c(i) = a(i) + b(i) - dt_inv*ddz05(i-1)*h  
  f(i) = - var1(i)*dt_inv*ddz05(i-1)*h + &
  & a(i)*var1(i-1) + b(i)*var1(i+1) - &
  & (a(i) + b(i))*var1(i) - &
  & lsh(i)*ddz05(i-1)*h
enddo

!Bottom boundary conditions
if     (bctype(2) == 1) then !Dirichlet condition
  c(N) = 1.
  a(N) = 0.
  f(N) = bcs(2) 
elseif (bctype(2) == 2) then !Neumann condition
  xx = 0.5*( - bath(N-1)%area_half/bath(N)%area_int * &
  & diff(N-1)/(ddz(N-1)*h) + (dh - dh0)*dt_inv05)
  x  = ddz(N-1)*h*dt_inv05
  a(N) = xx
  c(N) = xx - x
  f(N) = - var1(N)*x + bcs(2)*bath(N-1)%area_half/bath(N)%area_int + &
  & a(N)*var1(N-1) - xx*var1(N)
endif

call PROGONKA (nveclen,a,b,c,f,var2,1_iintegers,N)

deallocate (a,b,c,f,y)
END SUBROUTINE DIFFUSION

END MODULE DIFFUSION_LAKE_MOD
