        SUBROUTINE FLUXES_REPINA(TA,TS,V,TWB,P,HT,HW,N,M_,HSENS,HLAT)
        CHARACTER*20 INF1,INF2
        DIMENSION TA(M_), TS(M_), V(M_), P(M_), TWB(M_),
     *  ICH(M_), PMO(M_), EZ(M_), ES(M_), QA(M_), QS(M_),
     *  UDIN(M_), TDIN(M_), alon(M_), alat(M_),
     *  QDIN(M_), TAU(M_), DZITA(M_)
        real, intent(out) :: HSENS(M_), HLAT(M_)
c        WRITE(*,1)
c1      FORMAT(/2X,'TEMPERATURE SENSOR HEIGHT..............'\)
c        READ(*,*)HT
c        HT=2.5
c       WRITE(*,4)
c4      FORMAT(2X,'WIND SENSOR HEIGHT.....................'\)
c       READ(*,*)HW
c        HW=2.5
        ELE=24.68*10.**5
C ***** ELE- LATENT HEAT OF EVAPORATION ( J/KG ) ***********
        CP=1004.8
C ***** CP- SPECIFIC HEAT CAPICITY  ( J/KG*K ) *************
c       WRITE(*,2)
c2      FORMAT(/2X,'INPUT DATA FROM THE KEYBOARD...........1'/
c     * 2X,'INPUT DATA FROM DISK FILE..............2'/
c     * 2X,'WHATs INPUT............................'\)
c       READ(*,*)K1
        k1=2
c       WRITE(*,6)
c6      FORMAT(/2X,'RESULTS TO THE SCREEN ONLY.............1'/
c     * 2X,'RESULTS TO THE DISK FILE...............2'/
c     * 2X,'RESULTS TO THE SCREEN AND FILE.........3'/
c     * 2X,'RESULTS TO THE PRINTER ONLY............4'/
c     * 2X,'WHAT DO YOU WANT.......................'\)
c       READ(*,*)K2
        k2=2
c       WRITE(*,120)
c120    FORMAT(/2X,'HOW MANY EVENTS........................'\)
c       READ(*,*)N
!       n=1686
        IF(K1.EQ.2)GO TO 999
C
C  INPUT DATA FROM THE KEYBOARD
C
c       WRITE(*,'(A\)')'  YEAR...................................'
c       READ(*,*)NG
c       DO 100 I=1,N
c       WRITE(*,'(A\)')'  MONTH..................................'
c       READ(*,*)NM(I)
c       WRITE(*,'(A\)')'  DAY....................................'
c       READ(*,*)Nm(I)
c       WRITE(*,'(A\)')'  HOUR...................................'
c       READ(*,*)Ns(I)
c       WRITE(*,7)
7       FORMAT(/2X,'AIR TEMPERATURE........................'/)
        READ(*,*)TA(I)
        WRITE(*,8)
8       FORMAT(2X,'SURFACE TEMPERATURE....................'/)
        READ(*,*)TS(I)
        WRITE(*,9)
9       FORMAT(/2X,'WET-BULB TEMPERATURE ON WATER..........1'/
     *  2X,'WET-BULB TEMPERATURE ON ICE............2'/
     *  2X,'RELATIVE HUMIDITY......................3'/
     *  2X,'IF NONE, PRESS ........................0'/
     *  2X,'WHAT DO YOU HAVE.......................'/)
        READ(*,*)ICH(I)
        IF(ICH(I).EQ.0)GO TO 16
        GO TO (10,11,12),ICH(I)
10      WRITE(*,13)
13      FORMAT(/2X,'WET-BULB TEMPERATURE ON WATER..........'/)
        READ(*,*)TWB(I)
        GO TO 16
11      WRITE(*,14)
14      FORMAT(/2X,'WET-BULB TEMPERATURE ON ICE............'/)
        READ(*,*)TWB(I)
        GO TO 16
12      WRITE(*,15)
15      FORMAT(/2X,'RELATIVE HUMIDITY......................'/)
        READ(*,*)TWB(I) 
16      CONTINUE
        WRITE(*,18)
18      FORMAT(/2X,'WIND VELOSITY..........................'/)
        READ(*,*)V(I)
        WRITE(*,19)
19      FORMAT(2X,'PRESSURE...............................'/)
        READ(*,*)P(I)
100     CONTINUE
C
C   COMPLITION OF INPUT DATA FROM THE KEYBOARD

        GO TO 50
C
C   INPUT DATA FROM DISK FILE
C
999     CONTINUE
C
c       WRITE(*,61)
c61     FORMAT(/2X,'INPUT DATA FROM DISK FILE..............'\)
c        READ(*,'(A20)')INF2
        !OPEN(2,FILE='potokin.txt',STATUS='OLD',
        !* ACCESS='SEQUENTIAL')
c       DO 155 I=1,8
c       READ(2,*)
c155    CONTINUE
        DO 62 I=1,N
!       READ(2,*)TA(I),
!     * TS(I),V(I), TWB(I),P(I),a
        ich(i)=3
62      CONTINUE
!       CLOSE(2)
C
50      CONTINUE
C
C   CALCULATIONS OF HEAT FLUXES
C
c       WRITE(*,21)
c21     FORMAT(/2X,'ROUGHNESS....( CM )....................'\)
c       READ(*,*)Z0
        Z0=0.0001
C
C  BEGINING OF CALCULATIONS
C
        DO 51 I=1,N
        IF(TS(I).LT.-99.0)GO TO 51
        RO=P(I)/(2.8703*(273.+TA(I)))
        HH=HW
        IF(HT.GT.HH)HH=HT
        AZ0=ALOG(Z0)
        AZT=ALOG(HT)
        AZW=ALOG(HW)
        KP=ICH(I)-1
        IF(KP.LT.0)GO TO 23
C
C   CALCULATION OF SPECIFIC HUMIDITY
C
        CALL EA(TA(I),KP,TWB(I),P(I),QW)
        EZ(I)=QW
        IF(TS(I).GT.-2.)QW=7.63*TS(I)/(241.9+TS(I))
        IF(TS(I).LE.-2.)QW=9.5*TS(I)/(265.5+TS(I))
        ES(I)=6.1*10.**QW
        QA(I)=0.622*EZ(I)/P(I)
        QS(I)=0.622*ES(I)/P(I)
23      CONTINUE
        TSR=(TA(I)+TS(I))/2.+273.
        DT=TA(I)-TS(I)
        RR=ABS(DT)
        BO=0.
        IF(RR.GT.0.0001)BO=CP*DT/(ELE*(QA(I)-QS(I)))
        EM=0.61*CP*TSR/ELE
        B4=0.
        IF(RR.GT.0.0001)B4=EM/BO
C
C   DIVISION OF REGIMES
C
        IF(DT.LE.0.)GO TO 24
******* TO UNSTABLE AND NEUTRAL STRATIFICATIONS ************
C
        IF(DT.GT.0.)GO TO 25
******* TO STABLE STRATIFICATION ***************************
C
25      CONTINUE
C
C  STABLE STRATIFICATION
C
C       A=(V(I)*V(I)*TSR/(9.8*DT))
C       Q=A*(1.+B4)*HT/(HW*HW)
C       IF(Q.GT.10.)BETA=10.
C       IF(Q.LE.10..AND.Q.GT.5.)BETA=5.
C       IF(Q.LE.5.)GO TO 29
C26     CONTINUE
C       EL=10000.
C74     CONTINUE
C       QL=((AZT-AZ0+BETA*HT/EL)/((AZW-AZ0+BETA*HW/EL)**2))*A*(1.+B4)
C       DD=ABS(EL-QL)
C       IF(DD.LT.1.)GO TO 73
C       EL=QL
C       GO TO 74
C73     CONTINUE
C       DDD=ABS(HH/EL-HH/QL)
C       IF(DDD.LT.0.2)GO TO 75
C       EL=QL
C       GO TO 74
C75     CONTINUE
C       DZETA=HH/QL
C       IF(BETA.GT.9..AND.DZETA.GT.0.4)GO TO 27
C       IF(BETA.LT.6..AND.DZETA.GT.1.)GO TO 29
C       PMO(I)=QL
C       TDIN(I)=DT/(AZT-AZ0+BETA*HT/QL)
C       UDIN(I)=0.4*V(I)/(AZW-AZ0+BETA*HW/QL)
C       DZITA(I)=HH/QL
C       GO TO 30
C27     CONTINUE
C       BETA=5.
C       GO TO 26
C29     CONTINUE
C       UDIN(I)=V(I)*0.4/(6.*(AZW-AZ0))
C       TDIN(I)=DT/(6.*(AZT-AZ0))
C       PMO(I)=UDIN(I)*UDIN(I)*(1.+B4)/(0.4*0.4*9.8*TDIN(I)/TSR)
C       IF(PMO(I).LT.0.0001)DZITA(I)=100.
C       IF(PMO(I).GE.0.0001)DZITA(I)=HH/PMO(I)
C30     CONTINUE
C
        A=(V(I)*V(I)*TSR/(9.8*DT))
26      CONTINUE
        EL=10000.
        JJJJJ=0
274     CONTINUE
        DZW=HW/EL
        UNIVFW=0.7*DZW+0.75*(DZW-5./0.35)*EXP(-0.35*DZW)
        DZT=HT/EL
        UNIVFT=0.7*DZT+0.75*(DZT-5./0.35)*EXP(-0.35*DZT)
        DZ0=Z0/EL
        UNIVFZ0=0.7*DZ0+0.75*(DZ0-5./0.35)*EXP(-0.35*DZ0)
        QL=((AZT-AZ0+UNIVFT-UNIVFZ0)/((AZW-AZ0+UNIVFW-UNIVFZ0)**2))*
     *  A*(1.+B4)
C
C
        DD=ABS(EL-QL)                                 
        JJJJJ=JJJJJ+1                                
        IF(JJJJJ.GT.25)GO TO 201                      
        IF(DD.LT.5.)GO TO 206                         
        IF(JJJJJ.EQ.1)EL=QL                          
        IF(JJJJJ.GT.1)EL=(EL+QL)/2.                   
        GO TO 274                                    
C
206     CONTINUE
        IF(ABS(EL).GT.HH.AND.ABS(QL).GT.HH)GO TO 205
        IF(ABS(EL).LT.HH.AND.ABS(QL).LT.HH)GO TO 207
C
        QQA=ABS((EL+QL)/2.-HH)                       
        IF(QQA.LT.0.2)GO TO 201                     
        EL=QL                                         
        GO TO 274                                    
C                                                    
C
207     CONTINUE                                      
        QQA=ABS(EL-QL)                               
        IF(QQA.LT.0.2)GO TO 201                      
        EL=QL                                        
        GO TO 274                                    
C
205     CONTINUE
        IF(ABS(EL-QL).LT.1.)GO TO 201                 
        Q1=HH/EL                                     
        Q2=HH/QL                                     
        IF(ABS(Q1-Q2).LT.0.2)GO TO 201                
        EL=(QL+EL)/2.                               
        GO TO 274                                    
C
C
201   CONTINUE
        PMO(I)=QL
        DZW=HW/QL
        UNIVFW=0.7*DZW+0.75*(DZW-5./0.35)*EXP(-0.35*DZW)
        DZT=HT/QL
        UNIVFT=0.7*DZT+0.75*(DZT-5./0.35)*EXP(-0.35*DZT)
        DZ0=Z0/QL
        UNIVFZ0=0.7*DZ0+0.75*(DZ0-5./0.35)*EXP(-0.35*DZ0)
        TDIN(I)=DT/(AZT-AZ0+UNIVFT-UNIVFZ0)
        UDIN(I)=0.4*V(I)/(AZW-AZ0+UNIVFW-UNIVFZ0)
        DZITA(I)=HH/QL
        TAU(I)=RO*UDIN(I)*UDIN(I)
        HSENS(I)=-RO*CP*0.4*TDIN(I)*UDIN(I)
        HLAT(I)=-999.999
        IF(ICH(I).GT.0)HLAT(I)=-HSENS(I)*ELE*(QS(I)-QA(I))/(CP*DT)      
        QDIN(I)=-999.999
        IF(UDIN(I).LT.0.00001)GO TO 51
        IF(ICH(I).GT.0)QDIN(I)=-HLAT(I)/(0.4*RO*ELE*UDIN(I))
        GO TO 51
C
C   COMPLETION OF CALCULATIONS ( STABLE STRATIFICATION )
C
24      CONTINUE
C

C   UNSTABLE AND NEUTRAL STRATIFICATIONS
C
        IF(RR.LT.0.00001)GO TO 667
        A=V(I)*V(I)*TSR/(9.8*DT)
        C=((AZT-AZ0)/(AZW-AZ0)**2)*(1.+B4)
        HNEUTRAL=HH/(A*C)
        ZH=-0.07*A*C
        IF(HNEUTRAL.LT.-0.07)GO TO 31
******* TRANSITION TO UNSTABLE STRATIFICATION***************
C
35      CONTINUE
C
C   NEUTRAL STRATIFICATION
C
        UDIN(I)=0.4*V(I)/(AZW-AZ0)
        TDIN(I)=DT/(AZT-AZ0)
        TAU(I)=RO*UDIN(I)*UDIN(I)
        HSENS(I)=-RO*CP*0.4*TDIN(I)*UDIN(I)
        IF(RR.LT.0.00001)GO TO 667
        PMO(I)=UDIN(I)*UDIN(I)*(1.+B4)/(0.4*0.4*9.8*TDIN(I)/TSR)
        HLAT(I)=-999.999
        IF(ICH(I).GT.0)HLAT(I)=-HSENS(I)*ELE*(QS(I)-QA(I))/(CP*DT)
        QDIN(I)=-999.999
        IF(ICH(I).GT.0)QDIN(I)=-HLAT(I)/(0.4*RO*ELE*UDIN(I))
        DZITA(I)=HH/PMO(I)
        GO TO 51
667     PMO(I)=-9999.999
        UDIN(I)=0.4*V(I)/(AZW-AZ0)
        TDIN(I)=DT/(AZT-AZ0)
        TAU(I)=RO*UDIN(I)*UDIN(I)
        HLAT(I)=0.
        QDIN(I)=0.
        DZITA(I)=0.
        GO TO 51
C
C   UNSTABLE STRATIFICATION
C
31      CONTINUE
        ZH=-0.07*A*C
        PP=1./3.
        B=ALOG(0.07)
105     CONTINUE
        IF(ZH.LT.HT.AND.ZH.LT.HW)GO TO 70
C
        IF(ZH.GT.HT.AND.ZH.LT.HW)GO TO 71
C
        IF(ZH.LT.HT.AND.ZH.GT.HW)GO TO 72
C
        IF(ZH.GE.HT.AND.ZH.GE.HW)GO TO 35
C
70      CONTINUE
******* ZH < HT,HW ****************************
C
        EL=-10000.
33      CONTINUE
        QQQ=ABS(EL)
        AEL=ALOG(QQQ)
        BT=1.2*(1./(HT**PP))*(QQQ**PP)
        BU=1.2*(1./(HW**PP))*(QQQ**PP)
        QL=((B-AZ0+AEL-BT+2.912)/(B-AZ0+AEL-BU+2.912)**2)*A*(1.+B4)
        QQ=ABS(EL-QL)
        IF(QQ.LE.1.)GO TO 32
        EL=QL
        GO TO 33
32      CONTINUE
        DD=ABS(HH/EL-HH/QL)
        IF(DD.LT.0.2)GO TO 34
        EL=QL
        GO TO 33
34      CONTINUE
        PMO(I)=QL
        UDIN(I)=0.4*V(I)/(B-AZ0+AEL-BU+2.912)
        TDIN(I)=DT/(B-AZ0+AEL-BT+2.912)
        ZH=-0.07*QL
        IF(ZH.GE.HT.AND.ZH.GE.HW)GO TO 35
        IF(ZH.GT.HT.AND.ZH.LT.HW)GO TO 71
        IF(ZH.GT.HW.AND.ZH.LT.HT)GO TO 72
        GO TO 110
C
71      CONTINUE
****** ZH < HW      AND        ZH > HT ********************
C
        EL=-10000.
76      CONTINUE
        QQQ=ABS(EL)
        AEL=ALOG(QQQ)
        BU=1.2*(1./(HW**PP))*(QQQ**PP)
        QL=((AZT-AZ0)/((B-AZ0+AEL-BU+2.912)**2))*A*(1.+B4)      
        DD=ABS(EL-QL)
        IF(DD.LT.1.)GO TO 77
        EL=QL
        GO TO 76
77      CONTINUE
        DDD=ABS(HH/EL-HH/QL)
        IF(DDD.LT.0.2)GO TO 78
        EL=QL
        GO TO 76
78      CONTINUE
        PMO(I)=QL
        TDIN(I)=DT/(AZT-AZ0)
        UDIN(I)=0.4*V(I)/(B-AZ0+AEL-BU+2.912)
        ZH=-0.07*QL
        IF(ZH.GT.HW)GO TO 35
        IF(ZH.LT.HT)GO TO 70
        GO TO 110
C
72      CONTINUE
******* ZH < HT    AND    ZH> HW **************************
C
        EL=-10000.
79      CONTINUE
        QQQ=ABS(EL)
        AEL=ALOG(QQQ)
        BT=1.2*(1./(HT**PP))*(QQQ**PP)
        QL=((B-AZ0+AEL-BT+2.912)/(AZW-AZ0)**2)*A*(1.+B4)
        DD=ABS(EL-QL)
        IF(DD.LT.1.)GO TO 80
        EL=QL
        GO TO 79
80      CONTINUE
        DDD=ABS(HH/EL-HH/QL)
        IF(DDD.LT.0.2)GO TO 81
        EL=QL
        GO TO 79
81      CONTINUE
        PMO(I)=QL
        UDIN(I)=0.4*V(I)/(AZW-AZ0)
        TDIN(I)=DT/(B-AZ0+AEL-BT+2.912)
        ZH=-0.07*QL
        IF(ZH.GT.HT)GO TO 35
        IF(ZH.LT.HW)GO TO 70
110     CONTINUE
        TAU(I)=RO*UDIN(I)*UDIN(I)
        HSENS(I)=-RO*CP*0.4*TDIN(I)*UDIN(I)
        HLAT(I)=-999.999
        DZITA(I)=HH/PMO(I)
        IF(ICH(I).GT.0)HLAT(I)=-HSENS(I)*ELE*(QS(I)-QA(I))/(CP*DT)
        QDIN(I)=-999.999
        IF(UDIN(I).LT.0.00001)GO TO 51
        IF(ICH(I).GT.0)QDIN(I)=-HLAT(I)/(0.4*RO*ELE*UDIN(I))    
51      CONTINUE        
C
******* OUTPUT OF RESULTS **********************************
C
        GO TO (37,39,37,37),K2
C
37      CONTINUE
******* OUTPUT  OF RESULTS TO THE SCREEN OR PRINTER ********
C
        IF(K2.EQ.1.OR.K2.EQ.3)OPEN(2,FILE='CON')
        IF(K2.EQ.4)OPEN(2,FILE='PRN')
        DO 41 I=1,N
        MT=IFIX(HT)
        MW=IFIX(HW)
        WRITE(2,42)HT,TA(I),TS(I),HW,V(I),TWB(I),ICH(I)
42      FORMAT(/2X,'TA(',F4.1,')=',F7.2,'  TS=',F7.2,'  
     *  V(',F4.1,')=',F5.2,'  TWB=',F6.2,'  K=',I2)
        WRITE(2,43)HT,EZ(I),ES(I),HT,QA(I),QS(I)        
43      FORMAT(2X,'EA(',F4.1,')=',F5.2,'  ES=',F5.2,
     *  '  QA(',F4.1,')=',F8.5,'  QS=',F8.5)
        WRITE(2,44)TAU(I),HSENS(I),HLAT(I)
44      FORMAT(2X,'TAU=',F6.3,'  HSENS=',F8.2,'  HLAT=',F7.2)
        WRITE(2,45)UDIN(I),TDIN(I),PMO(I)
45      FORMAT(2X,'UDIN=',F7.3,'  TDIN=',F7.3,'  L=',F8.2)
41      CONTINUE
        IF(K2.EQ.1.OR.K2.EQ.4)GO TO 111
C
39      CONTINUE
******* OUTPUT OF RESULTS TO THE DISK FILE*******************
C
c       WRITE(*,47)
c47     FORMAT(/2X,'DISK FILE...............................'\)
c       READ(*,'(A20)')INF1
        OPEN(3,FILE='potOUT.dat',STATUS='unknown')
        I=0
52      CONTINUE
        READ(3,*,ERR=111,END=152)
        I=I+1
        GO TO 52
152     CONTINUE
        BACKSPACE 3
        IF(I.EQ.0)WRITE(3,153)
153     FORMAT(4X,'TAU',4X,'HSENS',4X,'HLAT',2X,'UDIN',
     *  3X,'TDIN',7X,'L',4X,'DZITA' )
        IF(I.EQ.0)WRITE(3,*)
        DO 54 I=1,N
        WRITE(3,55) 
     *  TAU(I),HSENS(I),HLAT(I),UDIN(I),TDIN(I),PMO(I),DZITA(I)
     *     
54      CONTINUE
55      FORMAT(F9.3,F9.3,F10.3,F10.3,F10.3,F10.3,F10.3,F9.4)
111     CONTINUE
        CLOSE(2)
        CLOSE(3)
c        STOP
        END SUBROUTINE FLUXES_REPINA

C 
C 
        SUBROUTINE EA (T,K,TE,P,E)
        IF(K.EQ.0)GO TO 1 
        IF(K.EQ.1)GO TO 2 
        IF(T.LT.-10.)GO TO 3
        B=EMAXW(T)
        E=B*TE/100. 
        GO TO 4 
3       CONTINUE
        B=EMAXI(T)
        E=B*TE/100. 
        GO TO 4 
1       CONTINUE
        B=EMAXW(TE) 
        E=B-7.947*P*(T-TE)*(1.+0.00115*TE)/10000. 
        DE=7.947*(T-TE)*(1000.-P)/10000.
        E=E+DE
        GO TO 4 
2       CONTINUE
        B=EMAXI(TE) 
        E=B-0.88229*7.947*P*(T-TE)/10000. 
        DE=0.88229*7.947*(T-TE)*(1000.-P)/10000.
        E=E+DE
4       CONTINUE
        RETURN
        END 
C 
C 
        FUNCTION EMAXI (T)
        T1=273.16 
        T2=T+273.15 
        A=T1/T2 
        B1=9.09685*(A-1.) 
        B2=3.56654*ALOG(A)*0.43429
        B3=0.87682*(1.-1./A)
        B=-B1-B2+B3+0.78614 
        EMAXI=EXP(B*ALOG(10.))
        RETURN
        END 

C 
C 
        FUNCTION EMAXW (T)
        T1=273.16 
        T2=T+273.15 
        A=T1/T2 
        B1=10.79574*(1.-A)
        B2=5.028*ALOG(1./A)*0.43429 
        Q=-8.2969*(1./A-1.) 
        B3=1.50475*(1.-10.**Q)/10000. 
        Q=4.76955*(1.-A)
        B4=0.42873*(10.**Q-1.)/1000.
        B=B1-B2+B3+B4+0.78614 
        EMAXW=EXP(B*ALOG(10.))
        RETURN
        END 

