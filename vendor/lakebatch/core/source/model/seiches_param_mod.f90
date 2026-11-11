MODULE SEICHES_PARAM

use LAKE_DATATYPES, only : ireals, iintegers

implicit none

real(kind=ireals), parameter :: C_Deff = 2.e-3 !(Goudsmit et al., 2002; Eliott, 1984)
real(kind=ireals), parameter :: alpha = 2.e-3 ! (Goudsmit et al., 2002. eq. (19))
real(kind=ireals), parameter :: cnorm = 1. !normalization constant

contains
SUBROUTINE SEICHE_ENERGY(M,bathymwater,ls,tau,wind,vol,dt,Eseiches,Eseiches_diss)
!Subroutine calculates the integral seiche energy in a lake
!at the next time step according to (Goudsmit et al., 2002)

use ARRAYS_BATHYM, only : bathym, layers_type

use PHYS_CONSTANTS, only : row0, roa0

implicit none

!Input variables
integer(kind=iintegers), intent(in) :: M

type(bathym), intent(in) :: bathymwater(1:M+1)
type(layers_type), intent(in) :: ls

real(kind=ireals), intent(in) :: tau, wind 
real(kind=ireals), intent(in) :: vol !Lake volume
real(kind=ireals), intent(in) :: dt

!Input/output variables
real(kind=ireals), intent(inout) :: Eseiches ! Integral seiche energy
real(kind=ireals), intent(out) :: Eseiches_diss ! Integral seiche energy dissipation

real(kind=ireals) :: beta, gamma_, Eseiches_old

!dEdt = alpha*bathymwater(1)%area_int*tau*wind - & !generation by wind work
!&      C_Deff*bathymwater(1)%area_int*vol**(-1.5)*row0**(-0.5)*Eseiches**1.5 !dissipation

Eseiches_old = Eseiches

gamma_ = C_Deff*bathymwater(1)%area_int*vol**(-1.5)*row0**(-0.5)

!Semi-implicit scheme
!beta = 0.5*dt*sqrt(Eseiches_old)*gamma_
!Eseiches = (Eseiches_old*(1. - beta) + dt*alpha*bathymwater(1)%area_int*tau*wind)/(1. + beta)
!Eseiches_diss = gamma_*0.5*(Eseiches + Eseiches_old)*sqrt(Eseiches_old)

!Implicit Patankar scheme (1980)
beta = dt*sqrt(Eseiches_old)*gamma_
Eseiches = (Eseiches_old + dt*alpha*bathymwater(1)%area_int*tau*wind)/(1. + beta)
Eseiches_diss = gamma_*Eseiches*sqrt(Eseiches_old)



END SUBROUTINE SEICHE_ENERGY



END MODULE SEICHES_PARAM
