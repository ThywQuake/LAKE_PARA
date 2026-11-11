MODULE CONVECTPAR_LAKE_MOD

!INTERFACE
!SUBROUTINE WaterMix(M,flag, T1, T2, ro1)
!
!use LAKE_DATATYPES, only : ireals, iintegers
!! Input variables
!integer(kind=iintegers), intent(in)    ::    flag , M
!! Output variables
!real(kind=ireals)   , intent(inout) ::    T1(:), T2(:), ro1(:) 
!
!END SUBROUTINE WaterMix
!END INTERFACE

contains
SUBROUTINE WaterMix(M,flag, T1, T2, ro1)

use water_density, only: &
& water_dens_ts, water_dens_ts_kivu

use LAKE_DATATYPES, only : ireals, iintegers
implicit none

! Input variables
integer(kind=iintegers), intent(in)    ::    flag , M
! Output variables
real(kind=ireals)   , intent(inout) ::    T1(:), T2(:), ro1(:)

real(kind=ireals)                   ::    ro2(1:SIZE(ro1))

integer(kind=iintegers) n, i, k, p
logical    firstcall

data firstcall /.true./

!SAVE
                            
do i = 1, M+1
  ro1(i) = WATER_DENS_TS_KIVU(T1(i),0.e0_ireals)
  ro2(i) = ro1(i)
end do    
                       
do i = 1, M+1
  T2(i) = T1(i)
end do        
                   
!if (flag==0) then                
! do n = 1, M-1 
!  k = 1
!  do i = 1, M+1-n
!   if (ro1(i) > ro1(k)) then
!    k = i 
!   end if
!  end do
!  T2(M+1-n) = T1(k)
!  T2(k) = T1(M+1-n)
!  ro2(M+1-n) = ro1(k)
!  ro2(k) = ro1(M+1-n) 
!  do p = 1, M+1-n
!  ro1(p) = ro2(p)
!   T1(p) = T2(p)
!  end do
! end do
!else       

do n = 1, M-2 
  k = 2
    do i = 2, M+1-n
      if (ro1(i) > ro1(k)) then
        k = i 
      end if
    end do
  T2(M+1-n) = T1(k)
  T2(k) = T1(M+1-n)
  ro2(M+1-n) = ro1(k)
  ro2(k) = ro1(M+1-n) 
  do p = 1, M+1-n
    ro1(p) = ro2(p)
    T1(p) = T2(p)
  end do
end do      

!endif
      
if (flag==0) then
  ro1(1) = WATER_DENS_TS_KIVU(0.5*(T1(1)+T1(2)),0.e0_ireals)
!800.969d-7+588.194d-7*(T1(1)+T1(2))/2.- 
!& 811.465d-8*((T1(1)+T1(2))**2)/4.+476.600d-10*
!& ((T1(1)+T1(2))**3)/8.
  ro2(1) = ro1(1)
   c10:do i=M,2,-1
       if (ro2(1)>ro2(i)) then
         T2(i)=(T1(1)+T1(2))/2
         ro2(i)=ro1(1)
         do k=2,i
           T2(k-1)=T1(k)
           ro2(k-1)=ro1(k)
         enddo
         T1=T2
         ro1=ro2
         exit c10
       endif
     enddo c10   
endif

if (firstcall) firstcall=.false.                                       
END SUBROUTINE WaterMix 


!> Subroutine UnDenGrMix performs convective adjustment of unstably stratified regions of density profiles
SUBROUTINE UnDenGrMix(M,eos,lindens,dz,dz_s,c_s,ro_s,wst,gas) 

use LAKE_DATATYPES, only : ireals, iintegers

use PHYS_CONSTANTS, only : &
& row0, cw, cw_m_row0

use WATER_DENSITY, only : &
& DENSITY_W

use ARRAYS, only : gas_type
use ARRAYS_WATERSTATE, only : waterstate_type

implicit none

! Input variables
integer(kind=iintegers), intent(in) :: M
integer(kind=iintegers), intent(in) :: eos
integer(kind=iintegers), intent(in) :: lindens

real(kind=ireals), intent(in), dimension(M) :: dz
real(kind=ireals), intent(in) :: dz_s
real(kind=ireals), intent(in) :: c_s, ro_s

! Input/output variables
type(waterstate_type), intent(inout) :: wst
type(gas_type       ), intent(inout) :: gas

!Local variables
integer(kind=iintegers), parameter :: nscalars = 9
integer(kind=iintegers) :: i, j, k, l, mm, Nr
real(kind=ireals) :: scalars(1:nscalars,1:M+1) !Scalars to be mixed along with temperature
real(kind=ireals) :: T(1:M+1)
real(kind=ireals) :: Sc_Av(1:nscalars), dz_Sc_sum(1:nscalars)
real(kind=ireals) :: dz_T_sum,dz_sum,T_Av,ro_D,ro_H,x,T_sum,eps,dz_sum_cro
real(kind=ireals), allocatable :: work1d(:), work1d_1(:), work1d_2(:)
logical :: mix 

T      (  1:M+1) = wst%Tw2 (1:M+1)
scalars(1,1:M+1) = wst%Sal2(1:M+1)
scalars(2,1:M+1) = gas%qwater(1:M+1,2)
scalars(3,1:M+1) = gas%oxyg  (1:M+1,2)
scalars(4,1:M+1) = gas%DIC   (1:M+1,2)
scalars(5,1:M+1) = gas%DOC   (1:M+1,1)
scalars(6,1:M+1) = gas%DOC   (1:M+1,2)
scalars(7,1:M+1) = gas%POCL  (1:M+1)
scalars(8,1:M+1) = gas%POCD  (1:M+1)
scalars(9,1:M+1) = gas%phosph(1:M+1)

T_Av = 0.0
Nr = 0
mix = .true.
eps = 0.001 !density threshold increment, kg/m**3
mixing_loop : do while (mix)
  T_sum = 0.0
  do i = 2, M
    T_sum = T_sum + cw_m_row0*T(i) * (dz(i-1) + dz(i))*0.5
  enddo
  T_sum = T_sum + cw_m_row0 * T(1) * dz(1)*0.5 + &
  &       T(M+1) * (cw_m_row0 * dz(M)*0.5 + c_s * ro_s * dz_s*0.5)
  do i = 1, M
    j = 0
    k = 1
    dz_T_sum              = 0.
    dz_Sc_sum(1:nscalars) = 0.
    dz_sum_cro            = 0.
    dz_sum                = 0.
    T_Av                  = 0.
    Sc_Av    (1:nscalars) = 0.
    if (wst%row(i+1)-wst%row(i) < -eps) then
      j = i + 1
      if (i == 1) then
        dz_T_sum  = (T(1)*dz(1)*0.5 + T(2)*(dz(1) + dz(2))*0.5) * cw_m_row0
        forall (mm = 1:nscalars) &
        & dz_Sc_sum(mm) = (scalars(mm,1)*dz(1)*0.5 + scalars(mm,2)*(dz(1) + dz(2))*0.5)
        dz_sum_cro    = (dz(1)*0.5 + (dz(1) + dz(2))*0.5) * cw_m_row0
        dz_sum        = (dz(1)*0.5 + (dz(1) + dz(2))*0.5)
      else if (i == M) then
        T_Av     = (T(M) * cw_m_row0*(dz(M-1) + dz(M))*0.5 + &
        &           T(M+1) * (cw_m_row0*dz(M)*0.5 + c_s*ro_s*dz_s*0.5))/ &
        &           (cw_m_row0 * (dz(M-1)*0.5 + dz(M)) + c_s*ro_s * dz_s*0.5)
        forall (mm = 1:nscalars) &
        & Sc_Av(mm) = ( scalars(mm,M) * (dz(M-1) + dz(M))*0.5 + scalars(mm,M+1) * dz(M)*0.5 ) / &
        &             (dz(M-1)*0.5 + dz(M))
      else
        dz_T_sum = (T(i)*(dz(i-1) + dz(i))*0.5 + T(i+1)*(dz(i) + dz(i+1))*0.5) * &
        &           cw_m_row0
        forall (mm = 1:nscalars) &
        & dz_Sc_sum(mm) = scalars(mm,i)*(dz(i-1) + dz(i))*0.5 + scalars(mm,i+1)*(dz(i) + dz(i+1))*0.5
        dz_sum_cro   = ((dz(i-1) + dz(i))*0.5 + (dz(i) + dz(i+1))*0.5) * cw_m_row0
        dz_sum       = ((dz(i-1) + dz(i))*0.5 + (dz(i) + dz(i+1))*0.5)
      endif
      if (i < M) then
        ro_H = wst%row(i)
        ro_D = wst%row(i+2)
        do while (ro_D-ro_H < -eps)
          k = k+1
          j = i+k
          if (j < M+1) then
            dz_T_sum = dz_T_sum + T(j)*(dz(j-1) + dz(j))*0.5 * cw_m_row0
            forall (mm = 1:nscalars) dz_Sc_sum(mm) = dz_Sc_sum(mm) + scalars(mm,j)*(dz(j-1) + dz(j))*0.5
            dz_sum_cro   = dz_sum_cro + (dz(j-1) + dz(j))*0.5 * cw_m_row0
            dz_sum       = dz_sum     + (dz(j-1) + dz(j))*0.5
            ro_D = wst%row(j+1)
          else
            dz_T_sum = dz_T_sum +  T(M+1) * (cw_m_row0*dz(M)*0.5 + c_s*ro_s*dz_s*0.5)
            forall (mm = 1:nscalars) dz_Sc_sum(mm) = dz_Sc_sum(mm) +  scalars(mm,M+1) * dz(M)*0.5
            dz_sum_cro   = dz_sum_cro + dz(M)*0.5 * cw_m_row0 + dz_s*0.5 * c_s * ro_s
            dz_sum       = dz_sum     + dz(M)*0.5
            exit 
          endif
        enddo
        T_Av = dz_T_sum / dz_sum_cro
        forall (mm = 1:nscalars) Sc_Av(mm) = dz_Sc_sum(mm) / dz_sum
      endif

      T(i:j) = T_Av
      forall (mm = 1:nscalars) scalars(mm,i:j) = Sc_Av(mm)
      allocate(work1d(i:j),work1d_1(i:j),work1d_2(i:j))
      work1d  (i:j) = scalars  (1,i:j) !salinity
      work1d_1(i:j) = wst%preswat(i:j) 
      call DENSITY_W(j-i,eos,lindens,T(i),work1d,work1d_1,work1d_2)
      wst%row (i:j) = work1d_2   (i:j)
      deallocate(work1d,work1d_1,work1d_2)

    endif
  enddo
  do i = 1, M
    mix = .false.
    if (wst%row(i+1)-wst%row(i) < -eps) then
      mix = .true.
      Nr = Nr + 1
      exit
    endif
  enddo
enddo mixing_loop

wst%Tw2 (1:M+1) = T      (  1:M+1)
wst%Sal2(1:M+1) = scalars(1,1:M+1)
gas%qwater(1:M+1,2) = scalars(2,1:M+1)
gas%oxyg  (1:M+1,2) = scalars(3,1:M+1)
gas%DIC   (1:M+1,2) = scalars(4,1:M+1)
gas%DOC   (1:M+1,1) = scalars(5,1:M+1)
gas%DOC   (1:M+1,2) = scalars(6,1:M+1)
gas%POCL  (1:M+1)   = scalars(7,1:M+1)
gas%POCD  (1:M+1)   = scalars(8,1:M+1)
gas%phosph(1:M+1)   = scalars(9,1:M+1)

END SUBROUTINE UnDenGrMix


END MODULE CONVECTPAR_LAKE_MOD
