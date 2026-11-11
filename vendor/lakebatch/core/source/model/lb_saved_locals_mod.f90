!> State that LAKE keeps in SAVE locals of individual routines, moved here so
!! the generated restart (LB_RESTART_IO) saves it.  Each routine accesses its
!! variables under their original names via use-renaming; the routine bodies
!! are unchanged.
MODULE LB_SAVED_LOCALS

use LAKE_DATATYPES, only : ireals

implicit none

! SUBROUTINE OXYGEN (oxygen_mod.f90)
real(kind=ireals) :: oxygen_Foxyg1 = 0.d0 !< bottom oxygen flux, mol/(m**2*s)
real(kind=ireals) :: oxygen_botO2 = 0.    !< bottom oxygen concentration, mol/m**3

! SUBROUTINE K_EPSILON (turb_mod.f90): .false. until a lake's first k-epsilon
! step, which neglects time derivatives (initialization mode)
logical, allocatable :: keps_init_keps(:,:)

! SUBROUTINE T_SOLVER (t_solver_mod.f90): previous surface temperature and
! surface type bracket the next surface heat balance iteration
real(kind=ireals), allocatable :: tsolver_Tsurf_prev(:,:)
integer, allocatable :: tsolver_surftyp_prev(:,:)
real(kind=ireals) :: tsolver_total_Bal = 0.

END MODULE LB_SAVED_LOCALS
