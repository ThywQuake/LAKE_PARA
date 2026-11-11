MODULE LAKE_INTERACTION
contains
SUBROUTINE LAKETRIBINT(larea,h,hbot,hbottrib,width,length,nlakes, &
& dhdt)

! Subroutine calculates the intensity of lakes water
! exchange through a connecting channel, expressed in
! corresponding water level change rate, m/yr.
! Currently operational only for 2 lakes.

use DRIVER_DATATYPES, only : &
& iintegers, ireals

implicit none

! Input variables
real(kind=ireals), intent(in) :: larea(1:nlakes) ! Lake surface area, m**2
real(kind=ireals), intent(in) :: h(1:nlakes) ! Lake depth (water layer thickness), m
real(kind=ireals), intent(in) :: hbot(1:nlakes) ! The height of lake bottom a.s.l., m
real(kind=ireals), intent(in) :: hbottrib(1:nlakes) ! The relative altitude of channel bottom above lake bottom, m
real(kind=ireals), intent(in) :: width ! Channel width, m
real(kind=ireals), intent(in) :: length ! Channel length, m 

integer(kind=iintegers), intent(in) :: nlakes

! Output variables
real(kind=ireals), intent(out) :: dhdt(1:nlakes) ! Water level change rate due to 
                                                 ! water exchange through a channel, m/yr

! Local variables
real(kind=ireals), parameter :: year_sec = 365.*86400. ! seconds
real(kind=ireals), parameter :: roughness_height = 1.d-2 ! Bottom roughness parameter, m
real(kind=ireals) :: work, chdepth, HydrRad, CoefShezy, dhdx, uspeed, Scross
real(kind=ireals) :: sgn(1:nlakes)

integer(kind=iintegers) :: botfric = 1 ! Switch between Shezy (1) and Mannings (2) bottom friction
integer(kind=iintegers) :: i ! Loop index

! It is assumed that the channel flow is directed from the edge (i.e. lake)
! with higher channel bottom and has uniform depth and width
select case (hbot(1) + hbottrib(1) > hbot(2) + hbottrib(2))
case (.true.)
  if (h(1) > hbottrib(1)) then
    work = max(h(2) - hbottrib(2),0.d0)
    chdepth = 0.5d0*(h(1) - hbottrib(1) + work) ! Approximation for uniform channel depth
!    dhdx = (hbot(1) + h(1) - (hbot(2) + hbottrib(1) + work))/length
    sgn(1) = -1.
    sgn(2) = 1.
  endif
case (.false.)
  if (h(2) > hbottrib(2)) then
    work = max(h(1) - hbottrib(1),0.d0)
    chdepth = 0.5d0*(h(2) - hbottrib(2) + work) ! Approximation for uniform channel depth
!    dhdx = (hbot(1) + h(1) - (hbot(2) + hbottrib(1) + work))/length
    sgn(1) = 1.
    sgn(2) = -1.
  endif
end select

HydrRad = HYDRAULIC_RADIUS(width,chdepth)
CoefShezy = COEF_SHEZY(HydrRad,roughness_height,botfric)
dhdx = abs(hbot(1) + hbottrib(1) - hbot(2) - hbottrib(2))/length
uspeed = CoefShezy*sqrt(dhdx*HydrRad) ! Homogeneous flow
Scross = SCROSS_AREA(width,chdepth)
do i = 1, nlakes
  dhdt(i) = sgn(i)*Scross*uspeed/larea(i)*year_sec
enddo

END SUBROUTINE LAKETRIBINT


FUNCTION OUTFLOW_DISCHARGE(ndatamax,coefs,h)

! Subroutine calculates the discharge of outflow, assuming it to be dependent
! on water level in a form of polynom
use DRIVER_DATATYPES, only : &
& iintegers, ireals

implicit none

! Input variables
integer(kind=iintegers), intent(in) :: ndatamax

real(kind=ireals), intent(in) :: coefs(1:ndatamax)
real(kind=ireals), intent(in) :: h ! Lake depth, m

! Output variables
real(kind=ireals) :: OUTFLOW_DISCHARGE ! m**3/s

! Local variables
integer(kind=iintegers) :: i, j
real(kind=ireals) :: hout

i = 1
do while (coefs(i) /= -999.)
  i = i + 1 
enddo

if (i > 2) then
  OUTFLOW_DISCHARGE = 0.
  hout = max(h - coefs(i-1), 0.d0) ! coefs(i-1) - relative altitude of outflow bottom above the lake bottom, m
  do j = 1, i-2
    OUTFLOW_DISCHARGE = OUTFLOW_DISCHARGE + coefs(j)*hout**(j-1)
  enddo
else
  OUTFLOW_DISCHARGE = 0.
endif

END FUNCTION OUTFLOW_DISCHARGE


FUNCTION COEF_SHEZY(HydrRad,roughness_height,ind)

! Computes Shezy coefficient according to Manning formula

use DRIVER_DATATYPES, only : &
& iintegers, ireals

use PHYS_CONSTANTS, only : &
& g

real(kind=ireals) :: COEF_SHEZY

! Input variables
real(kind=ireals), intent(in) :: HydrRad ! Hydraulic radius, m
real(kind=ireals), intent(in) :: roughness_height ! The height of roughness elements, m

integer(kind=iintegers), intent(in) :: ind

! Local variables
real(kind=ireals), parameter :: coef1 = 7.7

if (ind == 1) then
  COEF_SHEZY = coef1*sqrt(g)*(HydrRad/roughness_height)**(1./6.)
elseif (ind == 2) then
  COEF_SHEZY = HydrRad**(-2./3.)*roughness_height**(-2)
endif

END FUNCTION COEF_SHEZY


FUNCTION SCROSS_AREA(width,waterstage)

! Computes cross-section area of a river flow from its width and water stage

use DRIVER_DATATYPES, only : &
& iintegers, ireals

real(kind=ireals) :: SCROSS_AREA

! Input variables
real(kind=ireals), intent(in) :: width, waterstage

SCROSS_AREA = width*waterstage

END FUNCTION SCROSS_AREA
 

FUNCTION HYDRAULIC_RADIUS(width,waterstage)

! Computes hydraulic radius of a river channel assuming rectangular cross-section

use DRIVER_DATATYPES, only : &
& iintegers, ireals

real(kind=ireals) :: HYDRAULIC_RADIUS

! Input variables
real(kind=ireals), intent(in) :: width, waterstage

! Local variables
real(kind=ireals), parameter :: small_number = 1.d-10

HYDRAULIC_RADIUS = width*waterstage / (width + 2.*waterstage)
HYDRAULIC_RADIUS = max(HYDRAULIC_RADIUS,small_number)

!HYDRAULIC_RADIUS = 1.d+20 ! Turns bottom friction to zero

END FUNCTION HYDRAULIC_RADIUS

END MODULE LAKE_INTERACTION
