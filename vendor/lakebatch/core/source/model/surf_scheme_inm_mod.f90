      MODULE SURF_SCHEME_INM

      contains
      SUBROUTINE PBLDAT_INM

!
      !USE PRMT
      !USE GRIDS
      !USE DRAIN
      IMPLICIT REAL (A-H,O-Z)
!C
!C*=====================================================================
!C*       INITIALIZATION OF PBL COMMON BLOCKS (FBL,DRAG,STRM)          =
!C*    FBL(1)     :  FOR LINEAR EXTRAPOLATION OF WIND ONTO SURFACE     =
!C*    FBL(2)     :  ---------------------------------------------     =
!C*    FBL(3)     :  PRESCRIBED MINIMAL VALUE OF SURFACE WIND MODULE   =
!C*    FBL(4)     :  HIGHT OF CONSTANT FLUX LAYER ( M )                =
!C*    FBL(5)     :  CRITICAL VALUE OF RELATIVE HUMIDITY IN            =
!C*               :  CALCULATION OF EQUIVALENT POTENTIAL TEMPERATURE   =
!C*    FBL(6)     :  SEA WATER TEMPERATURE UNDER ICE ( DEG. K )        =
!C*    FBL(7)     :  PRECRIBED MINIMAL VALUE OF SEA ICE SURFACE        =
!C*                  TEMPERATURE ( DEG. K )                            =
!C*    FBL(8)     :  RESERVED                                          =
!C*    FBL(9)     :  PBL WIND TURNING ANGLE OVER OCEAN ( DEG )         =
!C*    FBL(10)    :  --------------------------- ICE ----------------- =
!C*    FBL(11)    :  --------------------------- SNOW COVERED SOIL --- =
!C*    FBL(12)    :  --------------------------- BARED SOIL ---------- =
!C*    FBL(13)    :  --------------------------- IN TROPICS ---------- =
!C*    FBL(14)    :  BOUNDARIES OF TROPICS ( RADIANS )                 =
!C*    FBL(15)    :  RESERVED
!C*    FBL(16)    :  (SNOW COVERED SOIL HEAT CONDUCTIVITY)/DEPTH ----- =
!C*    FBL(17)    :  (SEA ICE HEAT CONDUCTIVITY)/(SEA ICE DEPTH) ----- =
!C*    FBL(18)    :  (SOIL HEAT CONDUCTIVITY)/DEPTH, CAL/(DEG.*M**2*S) =
!C*    FBL(19)    :  BULK SOIL HEAT CAPACITY, CAL/(DEG.*M**2)          =
!C*    FBL(20)    :  (SOIL MOISTURE CONDUCTIVITY)/DEPTH,SEC**-1        =
!C*=====================================================================
      COMMON/DRAG/ AL1,AL2,A2,A3,R0    ,B4,D1,Y10,X10, &
      &            P0,P1,AKA,H1,AN,X8,AN1,AN2,G,G10,G4,A0
      COMMON/STRM/ TAX(4),AI30(4),A3K(4),B3K(4),C3K(4), &
      &             AI40(4),A4K(4),B4K(4),C4K(4),A1K,B1K,DMK,R2K,AI1, &
     &             AI2,TAU0,DTAU
      COMMON /FBL/ FBL(20)
      COMMON /HYDR/ SNCR,WLMMX,CEFF,CA,DZL,DZG,BMIN,HR,ZRM,ZRMM,TRM, &
     & FLXMIN,TOMIN
      COMMON /VEGSW/ CCT,TBEST,FFTMIN,FFQMIN,CK,SWW
      !FBL(1)=SIG(NLEV)
      !FBL(2)=1.0E0
      !FBL(3)=1.0E0
      !FBL(4)=70.0E0
!C     !WILL BE PRESCRIBED AS A HEIGHT OF THE NEAREST TO THE SURFACE
!C     !SIGMA-LEVEL
      !FBL(5)=1.0E0
      !FBL(6)=271.5E0
      !FBL(7)=143.2E0
      !FBL(8)=0.0E0
      !FBL(9)=20.0E0
      !FBL(10)=10.0E0
      !FBL(11)=30.0E0
      !FBL(12)=30.0E0
      !FBL(13)=0.0E0
      !FBL(14)=20.0E0*3.14159E0/180.0E0
!C     !FBL(8)...FBL(14) WILL BE NOT USED IN THE MULTILEVEL PBL
      !FBL(15)=0.0E0
      !FBL(16)=0.04E0/2.0E0
      !FBL(17)=0.5E0/3.0E0
      !FBL(18)=0.6E0/2.0E0
      !FBL(19)=4.0E+04
      !FBL(20)=1.0E-6/5.0E0
!C     FBL(20)=0.0E0
!C-------  SUBROUTINE '' STORM '' CONSTANTS---------------

      TAX(1)=-5.0E0
                TAX(2)=5.0E0
                         TAX(3)=15.0E0
                                   TAX(4)=30.0E0
      AI30(1)=1.485E-01
                AI30(2)=1.1214E-01
                              AI30(3)=8.16E-02
                                    AI30(4)=4.775E-02
      A3K(1)=-3.911E-03
                A3K(2)=-3.353E-03
                          A3K(3)=-2.747E-03
                                    A3K(4)=0.0E0
      B3K(1)=2.6706E-05
               B3K(2)=2.9100E-05
                        B3K(3)=3.1494E-05
                                 B3K(4)=0.0E0
      C3K(1)=7.9810E-08
               C3K(2)=7.9810E-08
                        C3K(3)=7.9810E-08
                                 C3K(4)=0.0E0
      AI40(1)=2.0978E-01
                AI40(2)=1.6097E-01
                          AI40(3)=1.1863E-01
                                    AI40(4)=7.0226E-01
      A4K(1)=-5.1592E-03
                A4K(2)=-4.5802E-03
                          A4K(3)=-3.8652E-03
                                    A4K(4)=0.0E0
      B4K(1)=0.2555E-04
               B4K(2)=0.3235E-04
                        B4K(3)=0.3915E-04
                                 B4K(4)=0.0E0
      C4K(1)=0.2267E-06
               C4K(2)=0.2267E-06
                        C4K(3)=0.2267E-06
                                 C4K(4)=0.0E0
      A1K=17.57E0
               B1K=241.9E0
                        DMK=1.5E-02
      KK=1
          N1K=2
               I1K=(N1K+4)/KK
      AMU=((1.0E0*FLOAT(N1K))/FLOAT(KK))**I1K
           WMK=300.0E0
                G6K=120.0E0
                     RWK=1.0E0
                       R2K=12.0E0*FLOAT(KK)*AMU*WMK/(G6K*RWK*DMK**2)
      AI1=0.317909E0
                  AI2=0.2E0*0.416873E0
      TAU0=0.168518E0
                   DTAU=1.8290E-05
                                  UST=1500.0E0

!C--------- SUBROUTINE DRAG3(CONSTANT FLUX LAYER) CONSTANTS-------------


      AMI=0.0E0
                                       AKA=.40E0
                                               G=9.81E0
      A0=1.15E0
              A6=3.5E0
                     G4=16.0E0
                            G10=16.0E0
                                      B4=4.7E0
                                               ALFAM=.0144E0
      BETAM=.111E0
                AN=.000015E0
                            P4=.71E0
                                     H1=10.0E0
                                               X8=16.3E0
      AN1=5.0E0/6.0E0
                AN2=.45E0
                         AL1=AKA*P4
                                        AL2=.14E0*(30.0E0**AN2)
      AL2=AL2*(P4**.8E0)
                        A2=G/ALFAM
                                        A3=BETAM*AN*A2
      A2=ALOG(H1*A2)
                        R0=.9E0/B4
                                        AN5=(A6/A0)**4
      D1=(2.0E0*G10-AN5*G4-SQRT((AN5*G4)**2+4.0E0*AN5*G10*&
      &   (G10-G4)))/(2.0E0*G10**2)
      Y10=(1.0E0-G4*D1)**.25E0
                             X10=(1.0E0-G10*D1)**.5E0
      P1=2.0E0*ATAN(Y10)+ALOG((Y10-1.0E0)/(Y10+1.0E0))
      P0=ALOG((X10-1.0E0)/(X10+1.0E0))
!C-------------- OTHER CONSTANTS ---------------------------
      SNCR=0.4E0      ! SM. OF WATER, FROM WICH SNOW COVERED ALL THE CELL
      WLMMX=5.0E-2    ! SM. OF WATER. MINIMAL SKIN LAYER CAPACITY
      CCT=0.0016E0    ! 1/K. DEPEND. CANOPY RESIST. FROM TEMPERATURE
      TBEST=298.15E0  ! K. LOWEST CANOPY RESIST. TEMPERATURE
      FFTMIN=0.01E0   ! 1 MINIMAL RESIST. COEFF. FOR TEMPERATURE
      FFQMIN=0.01E0   ! 1 MINIMAL RESIST. COEFF. FOR HUMIDITY
      CK=0.9E0        ! 1 COEFF. FOR CANOPY RESIST.
      CEFF=0.5E0      ! 1 PART OF INTERCEPTED BY SKIN LAYER PRECIP.
      CA=1.0E0        ! 1 PART OF THE CELL, COVERED BY PRECIP.
      DZL=100.0E0     ! M. MINIMAL ST. DEV. OF HEIGHT FOR RUNOFF CALC.
      DZG=1000.0E0    ! M. MAXIMAL ST. DEV. OF HEIGHT FOR RUNOFF CALC.
      BMIN=0.01E0     ! 1 MINIMAL B FOR RUNOFF CALC.
      WSDL=0.05E0     ! 1 SOIL WETNESS FROM WICH DRAINAGE BEGIN
      WSDG=0.75E0     ! 1 SOIL WETNESS FROM WICH FAST DRAINAGE BEGIN
      DMIN=2.8E-10    ! SM/S. COEFF. FOR SLOW DRAINAGE
      DMAX=2.8E-8     ! SM/S. COEFF. FOR FAST DRAINAGE
      D=1.5E0         ! 1 COEFF. FOR FAST DRAINAGE
      WIINF=0.30E0    ! 1 SOIL ICE, FROM WICH INFILTRATION BEGIN
      HR=8.0E0        ! SM. OF SOIL FROM WICH EVAPORATION EXISTS
      ZRM=10.0E0      ! SM. OF SOIL WITH LARGE AMOUNT OF ROOTS
      ZRMM=20.0E0     ! SM. OF SOIL FOR CALCULATION OF LAI AND VEG
      TRM=5.0E0       ! C. CRITICAL SOIL TEMPERATURE FOR LAI AND VEG
      TOMIN=0.1       ! C. CRITICAL TEMPERATURE FOR WATER ACCESS FOR CANOPY
      FLXMIN=0.25     ! CAL/(M**2*S) ACCURACY OF BALANCE AT THE SURFACE

      RETURN
      END SUBROUTINE PBLDAT_INM


      SUBROUTINE DRAG3_LAKE(AR1,IT,AR2)
!C
!      USE PRMT
      IMPLICIT REAL (A-H,O-Z)
!C
!C*====================================================================
!C*     .....DEFENITION OF DRAG AND HEAT EXCHANGE COEFFICIENTS......  =
!C*     DETAILS OF ALGORITM ARE GIVEN IN:                             =
!C*     A.L.KAZAKOV,V.N.LYKOSSOV,"TRUDY ZAP.SIB.NII",1982,N.55,3-20   =
!C*                    INPUT DATA:
!C*     AR1(1) - ABS(WIND VELOCITY) AT CONSTANT FLUX LAYER            =
!C*                                 (CFL) HIGHT (M/S)                 =
!C*     AR1(2) - DIFFERENCE BETWEEN POTENTIAL TEMPERATURE AT CFL HIGHT=
!C*                             AND AT SURFACE  ( DEG. K)             =
!C*     AR1(3) - SEMI-SUM OF POTENTIAL TEMPERATURE AT CFL HIGHT AND   =
!C*                             AND AT SURFACE  ( DEG. K)             =
!C*     AR1(4) - DIFFERENCE BETWEEN HUMIDITY AT CFL HIGHT             =
!C*                             AND A SURFACE   ( GR/GR )             =
!C*     AR1(5) - CFL HIGHT ( M )                                      =
!C*     AR1(6) - ROUGHNESS OF SURFACE ( M ); FOR SEA SURFACE PUT -1   =
!C*      IT    - NUMBER OF ITERATIONS                                 =
!C*                     OUTPUT DATA:
!C*     AR2(1) - NON-DIMENSIONAL CFL HIGHT                            =
!C*     AR2(2) - RICHARDSON NUMBER                                    =
!C*     AR2(3) - REYNODS NUMBER                                       =
!C*     AR2(4) - LN(ZU/ZT)                                            =
!C*     AR2(5) - DYNAMICAL ROUGHNESS ZU (M)                           =
!C*     AR2(6) - THERMAL   ROUGHNESS ZT (M)                           =
!C*     AR2(7) - CRITICAL RICHARDSON NUMBER                           =
!C*     AR2(8) - TRANSFER COEFFICIENT FOR MOMENTUM                    =
!C*     AR2(9) - TRANSFER COEFFICIENT FR HEAT                         =
!C*     AR2(10)- COEFFICIENT OF TURBULENCE (KM) AT CFL HIGHT (M**2/S) =
!C*     AR2(11)- ALFT=KT/KM ( KT-COEFFICIENT OF TURBULENCE FOR HEAT)  =
!C*     COMMENT: DRAG COEFFICIENT =          AR2(8)*AR2(8)            =
!C*              HEAT EXCHANGE COEFFICIENT = AR2(8)*AR2(9)            =
!C*====================================================================
      DIMENSION AR1(6),AR2(11)
      COMMON/DRAG/ AL1,AL2,A2,A3,R0,B4,D1,Y10,X10,&
     &             P0,P1,AP0,H1,AN,X8,AN1,AN2,G,G0,G4,A0
      D3=0.0E0
      D0MAX=2.0E0
      U=AR1(1)
      T4=AR1(2)
      Q4=AR1(4)
      H=AR1(5)
      Z0=AR1(6)
      IF(Z0.LT.0.0E0) D0MAX=8.0E0
      IF(Z0.LT.0.0E0) THEN
!C*     ......DEFINITION Z0 OF SEA SURFACE......
      U1=U
      A1=0.0E0
      Y1=25.0E0
      C1MIN=ALOG(H1/1.0E0)/AP0
      DO 630 I=1,IT
      F=A2-2.0E0*ALOG(U1)
      DO 570 J=1,IT
      C1=(F+2.0E0*ALOG(Y1))/AP0
      IF(U.LE.8.0E0) A1=ALOG(1.0E0+A3*((Y1/U1)**3))/AP0
      C1=C1-A1
      C1=AMAX1(C1,C1MIN)
      Y1=C1
  570 CONTINUE
      Z0=H1*EXP(-C1*AP0)
      Z0=AMAX1(Z0,0.000015E0)
      U2=U*ALOG(H1/Z0)/(ALOG(H/Z0))
      U1=U2
  630 CONTINUE
      J=1
      H0=H/Z0
      U3=U1/C1
      ELSE
!C*     ......PARAMETERS FROM VISCOSITY SUBLAYER......
      J=0
      H0=H/Z0
      U3=U*AP0/ALOG(H0)
      END IF
      X7=U3*Z0/AN
      IF(X7.LE.X8) THEN
      D0=AN1*ALOG(AL1*X7)+AN2
      ELSE
      D0=AL2*(X7**0.45E0)
      END IF
!C*     ......HUMIDITY STRATIFICATION AND RI-NUMBER......
      T1=AR1(3)
      AL=G/T1
      D0=AMIN1(D0,D0MAX)
      R6=AL*H*(T4+0.61E0*T1*Q4)/U**2
      D00=D0
      ZT=Z0/EXP(D00)
      H00=H/ZT
      FT0=ALOG(H00)
!C*     ......DEFINITION OF R-PRIM......
      AN4=D1/H0
      AN5=D1/H00
!C     IF(D0.EQ.0.0E0) AN5=AN4
      IF (ABS(D0).LT.1.0E-10) AN5=AN4
      AN5=SQRT(1.0E0-G0*AN5)
      AN4=(1.0E0-G4*AN4)**0.25E0
      F0=ALOG((X10-1.0E0)*(AN5+1.0E0)/((X10+1.0E0)*(AN5-1.0E0)))/A0
      F4=2.0E0*(ATAN(Y10)-ATAN(AN4))+ALOG((Y10-1.0E0)*(AN4+1.0E0)/&
      &((Y10+1.0E0)*(AN4-1.0E0)))
      R1=D1*F0/(F4*F4)
!C*     ......DEFINITION OF DZ,TA,FU,FO,FIU,FIO......
      IF(R6.GT.0.0E0) GO TO 1460
      IF(R6.LT.R1) GO TO 1305
      IF(R6.GT.-0.001E0) THEN
!C*     ......NEARLY NEUTRAL......
      F4=ALOG(H0)
      F0=FT0/A0
!C     IF(D0.EQ.0.0E0) F0=F4/A0
      IF (ABS(D0).LT.1.0E-10) F0=F4/A0
      AM=1.0E0
      O=1.0E0/A0
      GO TO 1570
      ELSE
!C*     ......WEEK AND SEMISTRONG INSTABILITY......
      F1=ALOG(H0)
!C     IF(D0.EQ.0.0E0)FT0=F1
      IF (ABS(D0).LT.1.0E-10) FT0=F1
      D3=R6*A0*F1**2/FT0
      M=1
 1245 DO 1300 I=1,IT
      D=D3/H0
      DD=D3/H00
!C     IF(D0.EQ.0.0E0)DD=D
      IF (ABS(D0).LT.1.0E-10) DD=D
      Y1=(1.0E0-G4*D3)**0.25E0
      X1=SQRT(1.0E0-G0*D3)
      Y0=(1.0E0-G4*D)**0.25E0
      X0=SQRT(1.0E0-G0*DD)
      Y0=AMAX1(Y0,1.000001E0)
      X0=AMAX1(X0,1.000001E0)
      F4=ALOG((Y1-1.0E0)*(Y0+1.0E0)/((Y1+1.0E0)*(Y0-1.0E0)))+&
      &2.0E0*(ATAN(Y1)-ATAN(Y0))
      F0=ALOG((X1-1.0E0)*(X0+1.0E0)/((X1+1.0E0)*(X0-1.0E0)))/A0
      IF(M.NE.1) GO TO 1350
      Z3=R6*F4**2/F0
      D3=Z3
 1300 CONTINUE
      M=2
      GO TO 1245
 1350 AM=(1.0E0-G4*D3)**(-0.25E0)
      O=1.0E0/(A0*SQRT(1.0E0-G0*D3))
      GO TO 1570
      END IF
!C*     ......STRONG INSTABILITY.....
 1305 CONTINUE
      D3=D1
      M=1
 1355 DO 1410 I=1,IT
      D=D3/H0
      DD=D3/H00
!C     IF(D0.EQ.0.0E0)DD=D
      IF (ABS(D0).LT.1.0E-10) DD=D
      A1=(D1/D3)**(1.0E0/3.0E0)
      X0=SQRT(1.0E0-G0*DD)
      Y0=(1.0E0-G4*D)**0.25E0
      C=ALOG((X0+1.0E0)/(X0-1.0E0))
      B1=-2.0E0*ATAN(Y0)+ALOG((Y0+1.0E0)/(Y0-1.0E0))
      F=3.0E0*(1.0E0-A1)
      F4=F/Y10+P1+B1
      F0=(F/X10+P0+C)/A0
      IF(M.NE.1) GO TO 1430
      Z3=R6*F4**2/F0
      D3=Z3
 1410 CONTINUE
      M=2
      GO TO 1355
 1430 AM=A1/Y10
      O=A1/(A0*X10)
      GO TO 1570
!C*     ......STABLE STRATIFICATION......
 1460 CONTINUE
      R6=AMIN1(R6,R0)
      F=ALOG(H0)
      F1=D0/F
      A1=B4*R6
      A2CH=(F1+1.0E0)/A0-2.0E0*A1
      D3=F*(SQRT(A2CH**2+4.0E0*A1*(1.0E0-A1))-A2CH)/(2.0E0*B4*&
      &(1.0E0-A1))
      F1=B4*D3
      F4=F+F1
      F0=(F+D0)/A0+F1
      O=1.0E0/A0+F1
      AM=1.0E0+F1
 1570 CONTINUE
!C*     ......COMPUTATION OF CU,CO,K(H),ALFT
      C4=AP0/F4
      C0=AP0/F0
      AN4=AP0*C4*U*H/AM
      AN0=AM/O
!C*     ......EXIT......
  140 CONTINUE
      AR2(1)=D3
      AR2(2)=R6
      AR2(3)=X7
      AR2(4)=D00
      AR2(5)=Z0
      AR2(6)=ZT
      AR2(7)=R1
      AR2(8)=C4
      AR2(9)=C0
      AR2(10)=AN4
      AR2(11)=AN0
      RETURN
      END SUBROUTINE DRAG3_LAKE

      END MODULE SURF_SCHEME_INM
