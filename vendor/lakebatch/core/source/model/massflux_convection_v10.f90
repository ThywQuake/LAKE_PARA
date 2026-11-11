!-----------------------------------------------------------------
!    #################################
     MODULE MODI_MASSFLUX_CONVECTION
!    #################################
!

INTERFACE
!
      SUBROUTINE MASSFLUX_CONVECTION(KTCOUNT,PTSTEP,PZZ,ZZM,DZF,       &
                 PSDENS,PSFT,PSLE,PSFSAL,                              &
                 PSFU,PSFV,                                            &
                 PUM,PVM,PWM,PTKEM,                                    &
                 PUT,PVT,PWT,PDENSM,PTM, PSALM,                        &
                 PEMF,PDENS_DOWN,PT_DOWN,PSAL_DOWN,ZINV,IKB,IKE         )
!
use LAKE_DATATYPES, only : ireals, iintegers
!
INTEGER,               INTENT(IN)   ::  KTCOUNT      ! number of the current timestep
real(kind=ireals),               INTENT(IN)   ::  PTSTEP       ! timestep
real(kind=ireals), DIMENSION(:), INTENT(IN)   ::  PZZ          ! depths of flux levels (w points) (1:M)
real(kind=ireals), DIMENSION(:), INTENT(IN)   ::  ZZM          ! depths of mass levels (T)        (1:M+1)
real(kind=ireals), DIMENSION(:), INTENT(IN)   ::  DZF          ! distance between full levels     (1:M)
real(kind=ireals),               INTENT(IN)   ::  PSDENS       ! surface density flux
real(kind=ireals),               INTENT(IN)   ::  PSFT,PSLE,PSFSAL  ! surface sensible heat flux, latent heat flux and salinity flux
real(kind=ireals),               INTENT(IN)   ::  PSFU,PSFV    ! normal surface fluxes of (u,v)

!    prognostic variables at t- deltat
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PUM,PVM,PWM ! current components (1:M+1)
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PTKEM       ! TKE                (1:M)
!
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PUT,PVT,PWT ! Wind  at t         (1:M+1)
!
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PDENSM       ! density at time t-1     (1:M+1)
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PTM          ! temperature at time t-1 (1:M+1)
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PSALM        ! salinity at time t-1    (1:M+1)

!
real(kind=ireals), DIMENSION(:), INTENT(OUT)::  PEMF,    &    ! Mass Flux (M)          (1:M)
                                   PDENS_DOWN, &    ! DOWNdraft density      (1:M+1)
                                   PT_DOWN,    &    ! DOWNdraft temperature  (1:M+1)
                                   PSAL_DOWN        ! DOWNdraft salinity     (1:M+1)
real(kind=ireals), INTENT(INOUT)            ::  ZINV
! Beginning and the End of the physical domain for the flux points
INTEGER, INTENT(IN)            ::  IKB,IKE
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE MASSFLUX_CONVECTION
!
END INTERFACE

END MODULE MODI_MASSFLUX_CONVECTION
!
!
!     #################################################################
      SUBROUTINE MASSFLUX_CONVECTION(KTCOUNT,PTSTEP,PZZ,ZZM,DZF,       &
                 PSDENS,PSFT,PSLE,PSFSAL,                              &
                 PSFU,PSFV,                                            &
                 PUM,PVM,PWM,PTKEM,                                    &
                 PUT,PVT,PWT,PDENSM,PTM, PSALM,                        &
                 PEMF,PDENS_DOWN,PT_DOWN,PSAL_DOWN,ZINV,IKB,IKE        )
!     #################################################################
!!
!!
!!    PURPOSE
!!    -------
!!****  The purpose of this routine is to formulate the mass-flux term
!!      for turbulent transpor in the mixed lake layer.
!
!!    REFERENCE
!! --------------------------------------------------------------------------
!
!*      0. DECLARATIONS
!          ------------
!
! USE STATEMENTS
!
use LAKE_DATATYPES, only : ireals, iintegers
USE WATER_DENSITY
USE PHYS_CONSTANTS, ONLY: &
 G

IMPLICIT NONE
!
!
!*      0.1  declarations of arguments
!
!
!
INTEGER,            INTENT(IN)   ::  KTCOUNT      ! number of the current timestep
real(kind=ireals),               INTENT(IN)   ::  PTSTEP       ! timestep
real(kind=ireals), DIMENSION(:), INTENT(IN)   ::  PZZ          ! height of flux levels (w points)
real(kind=ireals), DIMENSION(:), INTENT(IN)   ::  ZZM          ! height of mass levels (T)
real(kind=ireals), DIMENSION(:), INTENT(IN)   ::  DZF          ! distance between full levels
real(kind=ireals),               INTENT(IN)   ::  PSDENS       ! Densisty surface flux
real(kind=ireals),               INTENT(IN)   ::  PSFT,PSLE,PSFSAL  ! surface sensible heat fluxe, latent heat flux and salinity flux
real(kind=ireals),               INTENT(IN)   ::  PSFU,PSFV    ! normal surface fluxes of (u,v) parallel to the orography

!    prognostic variables at t- deltat
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PUM,PVM,PWM ! wind components
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PTKEM       ! TKE
!
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PUT,PVT,PWT ! Wind  at t
!
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PDENSM       ! density at time t-1
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PTM          ! temperature at time t-1
real(kind=ireals), DIMENSION(:), INTENT(IN) ::  PSALM        ! salinity at time t-1

!
real(kind=ireals), DIMENSION(:), INTENT(OUT)::  PEMF,       &    ! Mass Flux
                                   PDENS_DOWN, &    ! DOWNdraft density
                                   PT_DOWN,    &    ! DOWNdraft temperature
                                   PSAL_DOWN        ! DOWNdraft salinity
real(kind=ireals), INTENT(INOUT)     ::  ZINV
INTEGER, INTENT(IN)        ::  IKE,IKB
!
!-------------------------------------------------------------------------------
!
!       0.2  declaration of local variables
!
real(kind=ireals), DIMENSION(SIZE(PTKEM,1)) ::     &
                   ZDENSM_F,             &
                   ZW_DOWN,              &
                   ZW_DOWN2,             &
                   ZEPSILON,             &
                   ZBUO_F,               &
                   ZMIX,                 &
                   ZBUO,                 &
                   ZSAL_F,               &
                   ZT_F,                 &
                   ZTKEM_F,              &
                   ZVARW_F,              &
                   ZWORK4
!
INTEGER:: K,JK        ! loop counters
!

real(kind=ireals), parameter :: ZTKEMIN = 1.D-10

real(kind=ireals) :: XWTHVSURF,ZDT,ZDDENS,ZDSAL
real(kind=ireals) :: XCSTMF
real(kind=ireals) :: ALPHA
real(kind=ireals) :: XEPSCST
real(kind=ireals) :: XABUO
real(kind=ireals) :: XBEPS
real(kind=ireals) :: XZS
real(kind=ireals) :: ZVARW_F_MIN ! V. M. Stepanenko, 01.04.08
INTEGER :: ISTART
!----------------------------------------------------------------------------
!
!*      PRELIMINARIES
!       -------------
!
!        GRID DEFINITION
!
!IKB=1    ! first vertical level
!IKE=     ! last vertical level
!
!
ALPHA   = 0.3   ! constant for the properties parcel excess
XEPSCST = 0.6   ! 0.5 ! prescribed prefactor for downdraft entrainment, estimated 0.4-0.5 in atmospheric LES
XCSTMF  = 0.1   ! 0.1 ! prescribed prefactor for mass flux coeficient,
                ! meaning that 10% of the grid point is occupied by downdrafts
XABUO   = 2.    ! prefactor for bouyancy term in vertical velocity equation, related to the neglecting of the pressure fluctuations term
XBEPS   = 1.   !1. ! prefactor for the entrainment in the vertical velocity equation
ISTART  = 1     ! vertical level where the downdraft starts

ZVARW_F_MIN = 1.d-8 ! The minimum value for vertical velocity variance at the top level
                    ! a value of 1.d-4 has been used for atmospheric applications
                    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!!     Lake edmf - Soares 20080218
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!
!-------------------------------------------------------------------------------
!      1. COMPUTE CONSERVATIVE VARIABLES AND RELATED QUANTITIES
!-------------------------------------------------------------------------------
!
!-------------------------------------------------------------------------------
!      2. reset arrays
!-------------------------------------------------------------------------------
PDENS_DOWN (:) = 0.
PT_DOWN    (:) = 0.
PSAL_DOWN  (:) = 0.
ZDENSM_F   (:) = 0.
ZTKEM_F    (:) = 0.
ZVARW_F    (:) = 0.
ZBUO_F     (:) = 0.
ZBUO       (:) = 0.
ZW_DOWN    (:) = 0.
ZW_DOWN2   (:) = 0.
ZEPSILON   (:) = 0.
PEMF       (:) = 0.
ZWORK4     (:) = 0.
ZMIX       (:) = 0.
ZDSAL          = 0.
ZDT            = 0.
ZDDENS         = 0.
XZS            = 0.

!
!------------------------------------------------------------------------------------
!     3.  linear interpolation to flux levels
!------------------------------------------------------------------------------------
!
!     1.2 linear interpolation to half levels
!=====================================================
!
! here we need the distance between the full levels (DZF)
!
!ZVARW(:) = (2./3.)*PTKEM(:)

! Note, that TKE in Lake model is calculated at the half (flux) layers,
! so that the interpolation is not needed

ZVARW_F(:) = (2./3.)*PTKEM(:)
ZTKEM_F(:) =         PTKEM(:)

!print*, 'ZVARW_F(ISTART) = ', ZVARW_F(ISTART)

!
DO k = 1,IKE
!       ZDENSM_F(k) = (DZF(k-1)*PDENSM(k) + DZF(k)*PDENSM(k-1))  &
!       /(DZF(k-1)+DZF(k))
!       ZSAL_F (k) = (DZF(k-1)*PSALM (k) + DZF(k)*PSALM (k-1))  &
!       /(DZF(k-1)+DZF(k))
       ZDENSM_F(k) = 0.5*( PDENSM(k) + PDENSM(k+1) )
       ZSAL_F  (k) = 0.5*( PSALM (k) + PSALM (k+1) )
       ZT_F    (k) = 0.5*( PTM   (k) + PTM   (k+1) )
!       ZTKEM_F(k) = (DZF(k-1)*PTKEM(k) + DZF(k)*PTKEM(k-1))  &
!       /(DZF(k-1)+DZF(k))
!       ZVARW_F (k) = (DZF(k-1)*ZVARW (k) + DZF(k)*ZVARW (k-1))  &
!       /(DZF(k-1)+DZF(k))
!       ZT_F (k) = (DZF(k-1)*PTM (k) + DZF(k)*PTM (k-1))  &
!       /(DZF(k-1)+DZF(k))
ENDDO
!
!--------------------------------------------------------------------------
!     4. Inital values and Excess calculations for w, density, temperature and salinity
!--------------------------------------------------------------------------------------
if (KTCOUNT.eq.1) then
   ZINV = 20.         ! initial inversion depth
endif
!---------------------------------------------------
!    4.1. Surface buoyancy flux
!
! here we have the 1st problem, which mechanism induces the downdrafts (sensible flux, momentum flux?????)
! 1st approach is to take in account only the sensible heat flux
!
XWTHVSURF = PSDENS
!
IF(XWTHVSURF > 0.) THEN ! we will have a parcel only when the density flux is positive
!
!-------------------------------------------------
!   4.2 Parcel Excess
!
!-----Soares formulation for the parcel initialization with the sqrt(TKE)
!
     ZDDENS = (PSDENS/SQRT(max(ZTKEM_F(ISTART),ZTKEMIN) ))*ALPHA
     ZDT    = (PSFT  /SQRT(max(ZTKEM_F(ISTART),ZTKEMIN) ))*ALPHA
     ZDSAL  = (PSFSAL/SQRT(max(ZTKEM_F(ISTART),ZTKEMIN) ))*ALPHA
!
!-------------------------------------------------
!   4.3 Parcel Initialization
!
 PDENS_DOWN(ISTART) = ZDENSM_F(ISTART) + ZDDENS
 PT_DOWN   (ISTART) = ZT_F(ISTART) + ZDT
 PSAL_DOWN (ISTART) = ZSAL_F  (ISTART) + ZDSAL
 ZBUO_F(ISTART)     = (PDENS_DOWN(ISTART) - ZDENSM_F(ISTART))*G/ZDENSM_F(ISTART)
 ZW_DOWN (ISTART)   = SQRT(MAX(ZVARW_F_MIN,ZVARW_F(ISTART)))    ! attention we need the vertical velocity variance
 ZW_DOWN2(ISTART)   = ZW_DOWN(ISTART)**2
!
!------------------------------------------------------------------------------------------------
!        5.0 Descent
!------------------------------------------------------------------------------------------------
!------------------------------------------------------------------------
!        5.1 Descent with a first guess of entrainment rate
!------------------------------------------------------------------------
!
   DO JK=ISTART,IKE-1
      IF (ZZM(JK) < ZINV) THEN
         ZEPSILON(JK) = MAX(0.d0,MIN(XEPSCST/(PZZ(JK+1)-PZZ(JK)),                  &
                            XEPSCST*(1./((ZZM(JK)-XZS)) +                                &
                            1./(ZINV-(ZZM(JK)-XZS)))))
      ELSE
         ZEPSILON(JK) = XEPSCST*0.1
      ENDIF

      ZMIX(JK)         = 0.5*(PZZ(JK+1)-PZZ(JK))*ZEPSILON(JK)

!     PDENS_DOWN(JK+1) = (PDENS_DOWN(JK)*(1-ZMIX(JK)) + PDENSM(JK)*2*ZMIX(JK))  &
!                         /(1+ZMIX(JK))
      PT_DOWN(JK+1)   = (PT_DOWN(JK)*(1-ZMIX(JK)) + PTM(JK)*2*ZMIX(JK))  &
                         /(1+ZMIX(JK))
      PSAL_DOWN(JK+1) = (PSAL_DOWN(JK)*(1-ZMIX(JK)) + PSALM(JK)*2*ZMIX(JK))  &
                         /(1+ZMIX(JK))

!     Note, that the temperature, passing to WATER_DENS_TS, must be in Celsius degrees
      PDENS_DOWN(JK+1) = WATER_DENS_TS_KIVU(PT_DOWN(JK+1),0._ireals)

      ZBUO_F(JK+1)     = XABUO*(PDENS_DOWN(JK+1) - ZDENSM_F(JK+1))*(G/ZDENSM_F(JK+1))
      ZW_DOWN2(JK+1)   = ZW_DOWN2(JK)*(1-XBEPS*ZMIX(JK))/(1+XBEPS*ZMIX(JK)) +     &
                         ZBUO_F(JK)*(PZZ(JK+1)-PZZ(JK))/(1+XBEPS*ZMIX(JK))
!      ZWORK4(JK+1) = ZVARW_F(JK+1)
      IF ( ZW_DOWN2 (JK+1)>0.0) THEN
         ZW_DOWN (JK+1)  = sqrt(ZW_DOWN2(JK+1))
      ELSE
         ZW_DOWN(JK+1) = 0.0
         PDENS_DOWN(JK+1) = 0.0
      ENDIF
      IF (ZW_DOWN2 (JK+1) < 0.0) THEN
         ZINV = PZZ(JK) + ZW_DOWN2(JK)/(ZW_DOWN2(JK)-ZW_DOWN2(JK+1))*  &
                    (PZZ(JK+1)-PZZ(JK))
         EXIT
      ENDIF
  ENDDO
!
!-------------------------------------------------------------------
!    5.2 Mass-flux profile
!-------------------------------------------------------------------
 DO JK=ISTART,IKE
   IF (ZINV > ZZM(JK)) THEN
!      PEMF(JK) = XCSTMF*SQRT(MAX(0.000000001,ZVARW_F(JK)))
      PEMF(JK) = XCSTMF*ZW_DOWN(JK)
   ELSE
      PEMF(JK) = 0.
   ENDIF
 ENDDO
!
ELSE
  ZEPSILON   (:) = 0.
  PEMF       (:) = 0.
  ZW_DOWN    (:) = 0.
  PDENS_DOWN (:) = 0.
  PSAL_DOWN  (:) = 0.
  PT_DOWN    (:) = 0.
  ZMIX       (:) = 0.
  ZBUO_F     (:) = 0.
ENDIF
!
!-------------------------------------------------------------------------------
!   6. Write some output
!------------------------------------------------------------------------------
!
END SUBROUTINE MASSFLUX_CONVECTION

