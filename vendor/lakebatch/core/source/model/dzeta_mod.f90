MODULE DZETA_MOD

implicit none

contains
FUNCTION DZETA(nl)
use LAKE_DATATYPES, only : ireals, iintegers
use ARRAYS_GRID, only : ddz
use DRIVING_PARAMS, only : M

!The function DZETA
!returns the dzeta coordinate [0..1]
!(non-dimensional coordinate in water layer)
!of the level nl [1..M+1].
!nl might be fractional number, however the function is opimized for
!integer or half-integer nl.

implicit none

real(kind=ireals) :: DZETA

!Input variable
real(4), intent(in) :: nl

!local variables
real(kind=ireals), allocatable, save :: dzeta_int(:)
real(kind=ireals), allocatable, save :: dzeta_05int(:)

integer(kind=iintegers) :: nl_int
integer(kind=iintegers) :: i !Loop index

logical :: firstcall
data firstcall /.true./

if (firstcall) then
  allocate (dzeta_05int(1:M))
  dzeta_05int(1) = 0.5d0*ddz(1)
  do i = 2, M
    dzeta_05int(i) = dzeta_05int(i-1) + 0.5d0*(ddz(i-1) + ddz(i))
  enddo
  allocate (dzeta_int(1:M+1))
  dzeta_int(1) = 0.e0_ireals
  do i = 2, M+1
    dzeta_int(i) = dzeta_int(i-1) + ddz(i-1)
  enddo
endif

if (mod(nl,1._4) == 0._4) then
  DZETA = dzeta_int(int(nl))
elseif (mod(2._4*nl,1._4) == 0._4) then
  DZETA = dzeta_05int(int(nl))
else
  nl_int = int(nl)
  if (nl_int > 1) then
    DZETA = sum(ddz(1:nl_int-1))
  else
    DZETA = 0.
  endif
  if (nl<M+1) DZETA = DZETA + (nl-real(nl_int))*ddz(nl_int)
endif

if (firstcall) firstcall = .false.
END FUNCTION DZETA


FUNCTION DZETAI(nl)
use LAKE_DATATYPES, only : ireals, iintegers
use ARRAYS_GRID, only : ddzi
use DRIVING_PARAMS, only : Mice

!The function DZETAI
!returns the dzetai coordinate [0..1]
!(non-dimensional vertical coordinate in ice layers)
!of the level nl [1..M+1].
!nl might be fractional number, however the function is optimized
!for integer or half-integer nl.

implicit none

real(kind=ireals) :: DZETAI

real(4), intent(in) :: nl

real(kind=ireals), allocatable, save :: dzetai_int(:)
real(kind=ireals), allocatable, save :: dzetai_05int(:)

integer(kind=iintegers) :: nl_int
integer(kind=iintegers) :: i !Loop index

logical :: firstcall
data firstcall /.true./

if (firstcall) then
  allocate (dzetai_05int(1:Mice))
  dzetai_05int(1) = 0.5d0*ddzi(1)
  do i = 2, Mice
    dzetai_05int(i) = dzetai_05int(i-1) + 0.5d0*(ddzi(i-1) + ddzi(i))
  enddo
  allocate (dzetai_int(1:Mice+1))
  dzetai_int(1) = 0.e0_ireals
  do i = 2, Mice + 1
    dzetai_int(i) = dzetai_int(i-1) + ddzi(i-1)
  enddo
endif

if (mod(nl,1._4) == 0._4) then
  DZETAI = dzetai_int(int(nl))
elseif (mod(2._4*nl,1._4) == 0._4) then
  DZETAI = dzetai_05int(int(nl))
else
  nl_int = int(nl)
  if (nl_int>1) then
    DZETAI = sum(ddzi(1:nl_int-1))
  else
    DZETAI = 0.
  endif
  if (nl<Mice+1) DZETAI = DZETAI + (nl-real(nl_int))*ddzi(nl_int)
endif

if (firstcall) firstcall = .false.
END FUNCTION DZETAI


FUNCTION VARMEAN(var,bathym_,grid)

use LAKE_DATATYPES, only : ireals, iintegers
use ARRAYS_GRID, only : ddz, ddz05
use DRIVING_PARAMS
use ARRAYS_BATHYM, only : bathym

!The function VARMEAN computes
!the vertical average of the
!variable var.
!If grid = 1, than var is averaged as 
!if it has been placed in cell interfaces
!If grid = 2, than var is averaged as 
!if it has been placed in cell centers

implicit none

real(kind=ireals) :: VARMEAN(2)

!Input variables
integer(kind=iintegers), intent(in) :: grid
real(kind=ireals), intent(in) :: var(1:M+1)

type(bathym), intent(in) :: bathym_(1:M+1)

!Local variables
integer(kind=iintegers) :: i
real(kind=ireals) :: vol, x, varmean_

varmean_ = 0.
vol = 0.
if(grid == 1 .or. grid == 11) then
  x = 0.5*ddz(1)*bathym_(1)%area_int
  varmean_ = varmean_ + var(1)*x; vol = vol + x
  do i = 2, M
    x = ddz05(i-1)*bathym_(i)%area_int
    varmean_ = varmean_ + var(i)*x; vol = vol + x
  enddo
  if (grid == 1) then
    x = 0.5*ddz(M)*bathym_(M+1)%area_int
    varmean_ = varmean_ + var(M+1)*x; vol = vol + x
  endif
elseif (grid==2) then
  do i = 1, M
    x = ddz(i)*bathym_(i)%area_half
    varmean_ = varmean_ + var(i)*x; vol = vol + x
  enddo
else
  print*, 'Illegal identifier GRID in the function varmean: STOP'
  STOP
endif
VARMEAN = [varmean_, varmean_/vol]

END FUNCTION VARMEAN

END MODULE DZETA_MOD

