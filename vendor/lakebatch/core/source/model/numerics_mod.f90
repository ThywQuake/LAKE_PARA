MODULE NUMERICS

use LAKE_DATATYPES, only : ireals, iintegers
use NUMERIC_PARAMS, only : nveclen
use IEEE_ARITHMETIC, only : IEEE_IS_NAN, IEEE_IS_FINITE

contains

!> Function BOUND_FROM_BELOW limits the value by module from below
FUNCTION BOUND_FROM_BELOW(x,xmin) result (xlim)
implicit none
!Input/output variables
real(kind=ireals), intent(in) :: x     !< Value to be bounded
real(kind=ireals), intent(in) :: xmin  !< Minimal value of the abs(x)
real(kind=ireals) :: xlim !< Bounded value of x
!Local variables
real(kind=ireals) :: xsign
xsign = sign(1._ireals,x)
xlim = max(abs(x),xmin)*xsign
END FUNCTION BOUND_FROM_BELOW

!> Function VALUE_IS_FINITE checks, whether the value is not infinite nor NaN
FUNCTION VALUE_IS_FINITE(val)
implicit none
real(kind=ireals), intent(in) :: val !>Real value to be checked
logical :: VALUE_IS_FINITE
VALUE_IS_FINITE = (.not.(IEEE_IS_NAN(val)) .and. IEEE_IS_FINITE(val))
END FUNCTION VALUE_IS_FINITE

SUBROUTINE MATRIXSUM(nn,a,b,c,k)
 implicit none
 
!MATRIXES: C=A+B
 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals), dimension (nn,2,2) :: a,b,c
 integer(kind=iintegers) :: k,j,i

 do j=1,2
  do i=1,2
   c(k,i,j)=a(k,i,j)+b(k,i,j)
  enddo
 enddo
 
 END SUBROUTINE MATRIXSUM


 SUBROUTINE MATRIXMULT(nn,a,b,c,k)
 implicit none

!MATRIXES: C=A*B      

 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals), dimension (nn,2,2) :: a,b,c
 integer(kind=iintegers) :: k

 c(k,1,1)=a(k,1,1)*b(k,1,1)+a(k,1,2)*b(k,2,1)
 c(k,1,2)=a(k,1,1)*b(k,1,2)+a(k,1,2)*b(k,2,2)
 c(k,2,1)=a(k,2,1)*b(k,1,1)+a(k,2,2)*b(k,2,1)
 c(k,2,2)=a(k,2,1)*b(k,1,2)+a(k,2,2)*b(k,2,2)

 END SUBROUTINE MATRIXMULT


 SUBROUTINE MATRIXMULTVECTOR(nn,a,g,f,k)
 implicit none

!MATRIX A, VECTORS g, f: Ag=f

 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals) :: a(nn,2,2),f(nn,2), g(nn,2)
 integer(kind=iintegers) :: k

 f(k,1)=a(k,1,1)*g(k,1)+a(k,1,2)*g(k,2)
 f(k,2)=a(k,2,1)*g(k,1)+a(k,2,2)*g(k,2)

 return
 END SUBROUTINE MATRIXMULTVECTOR


 SUBROUTINE VECTORSUM(nn,a,b,c,k)
 implicit none

!VECTORS: C=A+B
 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals), dimension(nn,2) :: a,b,c
 integer(kind=iintegers) k
 c(k,1)=a(k,1)+b(k,1)
 c(k,2)=a(k,2)+b(k,2)
 END SUBROUTINE VECTORSUM


 SUBROUTINE INVERSMATRIX(nn,a,a1,k)
 implicit none 
 
!MATRIXES: A1=A*(-1)
 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals), dimension(nn,2,2) :: a,a1
 integer(kind=iintegers) k
 
 a1(k,1,1)=a(k,2,2)/(a(k,1,1)*a(k,2,2)-a(k,1,2)*a(k,2,1)) 
 a1(k,1,2)=-a(k,1,2)/(a(k,1,1)*a(k,2,2)-a(k,1,2)*a(k,2,1))
 a1(k,2,1)=-a(k,2,1)/(a(k,1,1)*a(k,2,2)-a(k,1,2)*a(k,2,1))
 a1(k,2,2)=a(k,1,1)/(a(k,1,1)*a(k,2,2)-a(k,1,2)*a(k,2,1))  

 END SUBROUTINE INVERSMATRIX

 
 SUBROUTINE MATRIXPROGONKA(nn,a,b,c,d,y,N)
 
!MATRIXPROGONKA solves the set of MATRIX three-point diference equations 
 
 implicit none

 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals), dimension(nn,2,2) :: a,b,c,x3,x32,x31,x4,alpha
 real(kind=ireals), dimension(nn,2) :: y,d,x2,beta,x21
 integer(kind=iintegers) N,i,j,k

 call INVERSMATRIX(nn,c,x32,1)
 call MATRIXMULT(nn,x32,b,x3,1)
 do j = 1, 2
   do i = 1, 2
     alpha(2,i,j) = x3(1,i,j)
   enddo
 enddo
 call MATRIXMULTVECTOR(nn,x32,d,x2,1)
 do i = 1, 2
   beta(2,i) = x2(1,i)
 enddo

 do k = 3, N
  CALL MATRIXMULT(nn,A,ALPHA,X3,k-1)
  X4(k-1,1:2,1:2) = - X3(k-1,1:2,1:2)
  CALL MATRIXSUM(nn,C,X4,X31,k-1)
  CALL INVERSMATRIX(nn,X31,X32,k-1)
  CALL MATRIXMULT(nn,X32,B,X3,k-1)
  do j = 1, 2
    do i = 1, 2
      alpha(k,i,j) = X3(k-1,i,j)
    enddo
  enddo
  !call matrixmult(x3,x31,x33,k-1)
  !call matrixsum(-b,x33,x3,k-1)
  CALL MATRIXMULTVECTOR(nn,A,BETA,X2,K-1)
  CALL VECTORSUM(nn,D,X2,X21,K-1)
  CALL MATRIXMULTVECTOR(nn,X32,X21,X2,K-1)
  do i = 1, 2
    beta(k,i) = X2(k-1,i)
  enddo
 enddo

 CALL MATRIXMULT(nn,A,ALPHA,X3,N)
 X4(N,1:2,1:2) = - X3(N,1:2,1:2)
 CALL MATRIXSUM(nn,C,X4,X31,N)
 CALL INVERSMATRIX(nn,X31,X32,N)
 CALL MATRIXMULTVECTOR(nn,A,BETA,X2,N)
 CALL VECTORSUM(nn,D,X2,X21,N)
 CALL MATRIXMULTVECTOR(nn,X32,X21,X2,N)
 do i = 1, 2
   Y(N,i) = X2(N,i)
 enddo

 do k = N-1, 1, -1
  CALL MATRIXMULTVECTOR(nn,ALPHA,Y,X2,K+1)
  CALL VECTORSUM(nn,X2,BETA,X21,K+1)
  Y(K,1) = X21(K+1,1)
  Y(K,2) = X21(K+1,2)
 enddo

 return

 END SUBROUTINE MATRIXPROGONKA


 real(kind=ireals) FUNCTION KRON(i,j)
 implicit none
 integer(kind=iintegers) i,j
 kron=0.
 if(i==j) kron=1.
 END FUNCTION 


 SUBROUTINE IND_STAB_FACT_DB (nn,a,b,c,N,M,ind_stab,ind_bound)

 implicit none

 integer(kind=iintegers), intent(in) :: nn
 real(kind=ireals), dimension(1:nn) :: a,b,c
 integer(kind=iintegers) M,i,N
 logical ind_stab, ind_bound 

 SAVE

 ind_stab=.true.
 if (ind_bound .eqv. .true.) then 
  if (abs(b(N))>=abs(c(N)).or.abs(a(M))>=abs(c(M))) then
   ind_stab=.false.
   RETURN
  endif
 endif
 do i=N+1,M-1
  if (abs(a(i))+abs(b(i))>=abs(c(i))) then
   ind_stab=.false.
   RETURN
  endif
 enddo
 
 END SUBROUTINE IND_STAB_FACT_DB


 LOGICAL FUNCTION CHECK_PROGONKA(N,a,b,c,d,y)

!Function CHECK_PROGONKA checks the accuracy
!of tridiagonal matrix system solution

 implicit none

 integer(kind=iintegers), intent(in):: N
 real(kind=ireals), intent(in):: a(1:N)
 real(kind=ireals), intent(in):: b(1:N)
 real(kind=ireals), intent(in):: c(1:N)
 real(kind=ireals), intent(in):: d(1:N)
 real(kind=ireals), intent(in):: y(1:N)

 real(kind=ireals), parameter:: del0 = 1.0d-13
 real(kind=ireals) del

 integer(kind=iintegers) i

 del = 0.d0
 del = max(c(1)*y(1)-b(1)*y(2)-d(1),del)
 del = max(c(N)*y(N)-a(N)*y(N-1)-d(N),del)
 do i = 2, N-1
   del = max(-a(i)*y(i-1)+c(i)*y(i)-b(i)*y(i+1)-d(i),del)
 enddo

 CHECK_PROGONKA = del < del0

 END FUNCTION CHECK_PROGONKA



 SUBROUTINE PROGONKA(nn, a, b, c, f, y, K, N)
 implicit none
!FACTORIZATION METHOD FOR THE FOLLOWING SYSTEM OF LINEAR EQUATIONS:
!-a(i)*y(i-1)+c(i)*y(i)-b(i)*y(i+1)=f(i) i=K+1,N-1
! c(K)*y(K)-b(K)*y(K+1)=f(K)
!-a(N)*y(N-1)+c(N)*y(N)=f(N)
!

 integer(kind=iintegers), intent(in) :: nn
 integer(kind=iintegers), intent(in) :: K, N
 real(kind=ireals), intent(in) :: a(nn), b(nn), c(nn), f(nn) 
 real(kind=ireals), intent(out) :: y(nn)

 real(kind=ireals) :: alpha(nn+2), beta(nn+2) 
 integer(kind=iintegers) :: i

 SAVE
             
 alpha(K+1) = b(K)/c(K)
 beta(K+1) = f(K)/c(K)
 do i = K+2, N+1 
   alpha(i) = b(i-1)/(c(i-1)-a(i-1)*alpha(i-1))
   beta(i) = (f(i-1)+a(i-1)*beta(i-1))/ &
   & (c(i-1)-a(i-1)*alpha(i-1))
 end do
 y(N) = beta(N+1)
 do i = N-1, K, -1
   y(i) = alpha(i+1)*y(i+1)+beta(i+1)
 end do
  
 END SUBROUTINE PROGONKA


FUNCTION STEP(x)
! Heavyside (step) function of x
implicit none

real(kind=ireals) :: STEP

real(kind=ireals), intent(in) :: x

STEP = 0.5*(sign(1._ireals,x) + 1.)

END FUNCTION STEP


!>Function ACCUMSUM updates an accumulated mean
FUNCTION ACCUMM(n,sum_nm1,xn)

implicit none

real(kind=ireals), intent(in) :: sum_nm1 !> mean over n-1 values
real(kind=ireals), intent(in) :: xn !> n-th value of time series
integer(kind=iintegers), intent(in) :: n 

real(kind=ireals) :: ACCUMM

ACCUMM = ((n-1)*sum_nm1 + xn)/real(n)

END FUNCTION ACCUMM

REAL FUNCTION FindDet(matrix, n)
    IMPLICIT NONE
    REAL, DIMENSION(n,n) :: matrix
    INTEGER, INTENT(IN) :: n
    REAL :: m, temp
    INTEGER :: i, j, k, l
    LOGICAL :: DetExists = .TRUE.
    l = 1
    !Convert to upper triangular form
    DO k = 1, n-1
        IF (matrix(k,k) == 0) THEN
            DetExists = .FALSE.
            DO i = k+1, n
                IF (matrix(i,k) /= 0) THEN
                    DO j = 1, n
                        temp = matrix(i,j)
                        matrix(i,j)= matrix(k,j)
                        matrix(k,j) = temp
                    END DO
                    DetExists = .TRUE.
                    l=-l
                    EXIT
                ENDIF
            END DO
            IF (DetExists .EQV. .FALSE.) THEN
                FindDet = 0
                return
            END IF
        ENDIF
        DO j = k+1, n
            m = matrix(j,k)/matrix(k,k)
            DO i = k+1, n
                matrix(j,i) = matrix(j,i) - m*matrix(k,i)
            END DO
        END DO
    END DO
   
    !Calculate determinant by finding product of diagonal elements
    FindDet = l
    DO i = 1, n
        FindDet = FindDet * matrix(i,i)
    END DO
   
END FUNCTION FindDet

!> Conservative second-order smoothing after Vreman, 2004
SUBROUTINE SMOOTHER_CONSERV(y,dx,N)
implicit none
!Input/output variables
integer, intent(in) :: N
real(kind=8), intent(inout) :: y(1:N)
real(kind=8), intent(in) :: dx(0:N)
!Local variables
integer :: i
real(kind=8), parameter :: gamma = 0.99
real(kind=8), allocatable :: x(:)
real(kind=8) :: dxi, dxii, xx

allocate(x(1:N))
dxi = 0.5*(dx(0) + dx(1))
dxii = 0.5*(dx(1) + dx(2))
xx = 0.5*(1-gamma)/dxi
x(1) = xx*dx(1)*y(2) + (1. - xx*dx(1))*y(1)
do i = 2, N-1
  dxi = 0.5*(dx(i-1) + dx(i))
  xx = 0.5*(1-gamma)/dxi
  x(i) = xx*dx(i-1)*y(i-1) + gamma*y(i) + xx*dx(i)*y(i+1)
enddo
dxi = 0.5*(dx(N-1) + dx(N))
dxii = 0.5*(dx(N-2) + dx(N-1))
xx = 0.5*(1-gamma)/dxi
x(N) = xx*dx(N-1)*y(N-1) + (1.-xx*dx(N-1))*y(N)
y(1:N) = x(1:N)
deallocate(x)
END SUBROUTINE SMOOTHER_CONSERV

!************************************************************************
!*                                                                      *
!*    Program to calculate the first kind modified Bessel function of   *
!*    integer order N, for any REAL X, using the function BESSI(N,X).   *
!*                                                                      *
!* -------------------------------------------------------------------- *
!*                                                                      *
!*    SAMPLE RUN:                                                       *
!*                                                                      *
!*    (Calculate Bessel function for N=2, X=0.75).                      *
!*                                                                      *
!*    Bessel function of order  2 for X =  0.7500:                      *
!*                                                                      *
!*         Y =  0.73666878E-01                                          *
!*                                                                      *
!* -------------------------------------------------------------------- *
!*   Reference: From Numath Library By Tuan Dang Trong in Fortran 77.   *
!*                                                                      *
!*                               F90 Release 1.2 By J-P Moreau, Paris.  *
!*                                        (www.jpmoreau.fr)             *
!*                                                                      *
!*   Version 1.1: corected value of P4 in BESSIO (P4=1.2067492 and not  *
!*                1.2067429) Aug. 2011.                                 *
!*   Version 2: all variables are declared.                             *
!************************************************************************
!PROGRAM TBESSI
!
!  IMPLICIT NONE
!  REAL*8  BESSI, X, Y
!  INTEGER N
!
!  N=2
!  X=0.75D0
!
!  Y = BESSI(N,X)
!
!  write(*,10) N, X
!  write(*,20) Y
!
!  stop
!
!10 format (/' Bessel Function of order ',I2,' for X=',F8.4,':')
!20 format(/'      Y = ',E15.8/)
!
!END

! ----------------------------------------------------------------------
! Auxiliary Bessel functions for N=0, N=1
      FUNCTION BESSI0(X)
      IMPLICIT NONE
      REAL *8 X,BESSI0,Y,P1,P2,P3,P4,P5,P6,P7,  &
      & Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX
      DATA P1,P2,P3,P4,P5,P6,P7/1.D0,3.5156229D0,3.0899424D0,1.2067492D0,  &
      & 0.2659732D0,0.360768D-1,0.45813D-2/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,0.1328592D-1, &
      & 0.225319D-2,-0.157565D-2,0.916281D-2,-0.2057706D-1,  &
      & 0.2635537D-1,-0.1647633D-1,0.392377D-2/
      IF(ABS(X).LT.3.75D0) THEN
      Y=(X/3.75D0)**2
      BESSI0=P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7)))))
      ELSE
      AX=ABS(X)
      Y=3.75D0/AX
      BX=EXP(AX)/SQRT(AX)
      AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))))
      BESSI0=AX*BX
      ENDIF
      RETURN
      END
! ----------------------------------------------------------------------
      FUNCTION BESSI1(X)
      IMPLICIT NONE
      REAL *8 X,BESSI1,Y,P1,P2,P3,P4,P5,P6,P7,  &
      & Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX
      DATA P1,P2,P3,P4,P5,P6,P7/0.5D0,0.87890594D0,0.51498869D0,  &
      & 0.15084934D0,0.2658733D-1,0.301532D-2,0.32411D-3/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,-0.3988024D-1, &
      & -0.362018D-2,0.163801D-2,-0.1031555D-1,0.2282967D-1, &
      & -0.2895312D-1,0.1787654D-1,-0.420059D-2/
      IF(ABS(X).LT.3.75D0) THEN
      Y=(X/3.75D0)**2
      BESSI1=X*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
      AX=ABS(X)
      Y=3.75D0/AX
      BX=EXP(AX)/SQRT(AX)
      AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))))
      BESSI1=AX*BX
      ENDIF
      RETURN
      END

! ----------------------------------------------------------------------

! ----------------------------------------------------------------------
      FUNCTION BESSI(N,X)
!
!     This subroutine calculates the first kind modified Bessel function
!     of integer order N, for any REAL X. We use here the classical
!     recursion formula, when X > N. For X < N, the Miller's algorithm
!     is used to avoid overflows. 
!     REFERENCE:
!     C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
!     MATHEMATICAL TABLES, VOL.5, 1962.
!
      IMPLICIT NONE
      INTEGER, PARAMETER :: IACC = 40
      REAL*8, PARAMETER :: BIGNO = 1.D10, BIGNI = 1.D-10
      INTEGER N, M, J
      REAL *8 X,BESSI,TOX,BIM,BI,BIP !,BESSI0,BESSI1
      IF (N.EQ.0) THEN
      BESSI = BESSI0(X)
      RETURN
      ENDIF
      IF (N.EQ.1) THEN
      BESSI = BESSI1(X)
      RETURN
      ENDIF
      IF(X.EQ.0.D0) THEN
      BESSI=0.D0
      RETURN
      ENDIF
      TOX = 2.D0/X
      BIP = 0.D0
      BI  = 1.D0
      BESSI = 0.D0
      M = 2*((N+INT(SQRT(FLOAT(IACC*N)))))
      DO 12 J = M,1,-1
      BIM = BIP+DFLOAT(J)*TOX*BI
      BIP = BI
      BI  = BIM
      IF (ABS(BI).GT.BIGNO) THEN
      BI  = BI*BIGNI
      BIP = BIP*BIGNI
      BESSI = BESSI*BIGNI
      ENDIF
      IF (J.EQ.N) BESSI = BIP
12    CONTINUE
      BESSI = BESSI*BESSI0(X)/BI
      RETURN
      END


END MODULE NUMERICS
