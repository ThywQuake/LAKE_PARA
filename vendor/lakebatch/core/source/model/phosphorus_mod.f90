!> Module containing data and procedures
!! dealing with phosphorus dynamics
MODULE PHOSPHORUS_MOD

use LAKE_DATATYPES, only : ireals, iintegers

implicit none

contains
!> Subroutine performs a time step for phosphorus 
SUBROUTINE PHOSPHORUS(M,dt,ls,gsp,bath,gas,lsp,diff)

use DIFFUSION_LAKE_MOD, only : DIFFUSION
use ARRAYS       , only : gas_type
use ARRAYS_BATHYM, only : bathym, layers_type
use ARRAYS_GRID  , only : gridspacing_type
use ARRAYS_SOIL  , only : lsh_type

implicit none

!Input/output variables
integer(kind=iintegers), intent(in) :: M !> Number of numerical layers
real(kind=ireals), intent(in)       :: dt !> Timestep
type(layers_type), intent(in)       :: ls !> Thicknesses of physical layers
type(gridspacing_type), intent(in)  :: gsp !> Numerical grid parameters
type(bathym)     , intent(in)       :: bath(1:M+1) !> Bathymetry data
type(gas_type)   , intent(inout)    :: gas !> Gases and biogeochemical species
type(lsh_type)   , intent(inout)    :: lsp !> Phosphorus flux through bottom 
real(kind=ireals), intent(in)       :: diff(1:M) !> Diffusivity profile

!Local variables
integer(kind=iintegers) :: bctype(2)
real   (kind=ireals   ) :: bcs(2)
real   (kind=ireals   ), allocatable :: work(:)

allocate(work(1:M+1))
bctype(1) = 2; bcs(1) = 0. !Zero surface flux
bctype(2) = 2; bcs(2) = 0. !Zero flux at maximal lake depth
lsp%water(:) = 0. !DIP flux from sediments degradation is added in ADDOXPRODCONS
call DIFFUSION(M+1,bctype,bcs,dt,ls%h1,ls%dhw,ls%dhw0, &
& gsp%ddz,gsp%ddz05,gsp%dzeta_int,gas%phosph,work,diff,lsp%water,bath)
gas%phosph(1:M+1) = work(1:M+1)

deallocate(work)
END SUBROUTINE PHOSPHORUS

END MODULE PHOSPHORUS_MOD
