MODULE INOUT

use LAKE_DATATYPES, only : ireals, iintegers, ibyterecl
use COMPARAMS, only : parallel_params

type, public :: wrfild_2d_coords
  sequence
  integer(kind=iintegers) :: i0, i1, j0, j1, i00, i11, j00, j11, &
  & i0_write, i1_write, j0_write, j1_write, isx, isy, lrec, &
  & recn
endtype wrfild_2d_coords

type, public :: wrfild_3d_coords
  sequence
  integer(kind=iintegers) :: i0, i1, j0, j1, k0, k1, &
  & i00, i11, j00, j11, k00, k11, &
  & i0_write, i1_write, j0_write, j1_write, k0_write, k1_write, &
  & isx, isy, iss, lrec
endtype wrfild_3d_coords


contains
SUBROUTINE CHECK_UNIT(unit_min,unit_max,nunit)

! The subroutine CHECK_UNIT checks if the output unit is already occupied,
! and if yes, it returns the number of the free unit
implicit none

! Input variables
integer(kind=iintegers), intent(in) :: unit_min
integer(kind=iintegers), intent(in) :: unit_max

! Input/output variables
integer(kind=iintegers), intent(inout) :: nunit

! Local variables
logical :: unit_opened

inquire (unit=nunit,opened=unit_opened)

if (unit_opened) then
  do while (unit_opened)
!    write(*,*) 'The unit ', nunit, 'is attempted &
!    & to be connected to a file, while already connected: incrementing unit'
    nunit = nunit + 1
    inquire (unit=nunit,opened=unit_opened)
  enddo
!  STOP
endif

if (nunit < unit_min .or. nunit > unit_max) then
  write(*,*) 'Error on LAKE model: the bounds of permitted input/ouput &
  &unit numbers exceeded: STOP', unit_min, unit_max
  STOP
endif

END SUBROUTINE CHECK_UNIT


FUNCTION GETVARVAL(n1,n2,line,name)
implicit none

real(kind=ireals) :: GETVARVAL

! Input variables
integer,       intent(in):: n2,n1
character*200, intent(in):: line
character(len=*),  intent(in):: name

! Local variables
real(kind=ireals) work

read (line((n2+1):100),*) work
!print*, name//' = ', work

GETVARVAL = work      

RETURN
END FUNCTION GETVARVAL


FUNCTION IGETVARVAL(n1,n2,line,name)
implicit none

integer(kind=iintegers) :: IGETVARVAL

! Input variables
integer(kind=iintegers),    intent(in):: n2,n1
character*200, intent(in):: line
character(len=*),  intent(in):: name

! Local variables
integer(kind=iintegers) iwork

read (line((n2+1):100),*) iwork
!print*, name//' = ', iwork

IGETVARVAL = iwork      

RETURN
END FUNCTION IGETVARVAL

function fext2(fname,ext)
character*80 fname,fext2
character*3 ext
parameter(n=80)

fext2=fname

do 10 i=30,1,-1
  if(fext2(i:i).eq.'.') then
    fext2(i:n)=char(0)
    go to 11
 endif
10 continue
11 continue

do 20 i=30,1,-1
  if(fext2(i:i).ne.' ' .and. fext2(i:i).ne.char(0)) then
    fext2(i+1:i+4)='.'//ext
    go to 21
  endif
20 continue
   write(*,*) 'erro em fext'

21 continue
   return
end function fext2
                 
subroutine readgrd_lake(nunit,var,i0,i1,j0,j1)
!
! reads 2d array from a grd file
!
implicit real(kind=ireals)(a-h,o-z)
dimension var(i0:i1,j0:j1)
character*4 dsaa

read(nunit,'(a4)') dsaa
read(nunit,*) nx,ny
read(nunit,*) xmin,xmax
read(nunit,*) ymin,ymax
read(nunit,*) zmin,zmax
      
do j=j0,j1
  read(nunit,*) (var(i,j),i=i0,i1)
enddo
      
return
end subroutine readgrd_lake
      
subroutine wrigrd_lake(nunit,z,dx,dy,i0,i1,j0,j1)

implicit none
integer i0,i1,j0,j1,nunit,i,j
real(kind=ireals) z(i0:i1,j0:j1)
real(kind=ireals) xmin,xmax,ymin,ymax,zmin,zmax,dx,dy
      
zmin=z(i0,j0)
zmax=z(i0,j0)

do j=j0,j1
  do i=i0,i1
    zmin=min(zmin,z(i,j))
    zmax=max(zmax,z(i,j))
  enddo
enddo

xmin=float(i0)*dx
xmax=float(i1)*dx
ymin=float(j0)*dy
ymax=float(j1)*dy
      
write(nunit,'(a4)') 'DSAA'
write(nunit,*) (i1-i0+1),(j1-j0+1)
write(nunit,*) xmin,xmax
write(nunit,*) ymin,ymax
write(nunit,*) zmin,zmax

do j=j0,j1
  write(nunit,*) (z(i,j),i=i0,i1)
enddo
     
return
end subroutine wrigrd_lake


 SUBROUTINE WRFILD_2D_LAKE &
 & (xa, coords, unitout, iotsepftstep, iotsinftstep, parparams, filen)

 ! Writes distributed 2D array in a file

 use INOUT_PARAMETERS, only : lake_misc_unit_min, lake_misc_unit_max

 implicit none

#ifdef mpi
 include 'mpif.h'
#endif

!Input/output variables
 type(wrfild_2d_coords), intent(inout) :: coords
 type(parallel_params),  intent(in)    :: parparams

 integer(kind=iintegers), intent(inout) :: unitout
 logical, intent(in) :: iotsepftstep
 logical, intent(in) :: iotsinftstep

 real(kind=ireals), intent(in) :: xa(coords%i0 : coords%i1, coords%j0 : coords%j1)

 character(len=*), intent(in) :: filen
 
!Local variables
 real(kind=ireals), allocatable :: work(:,:), work2(:,:)

 integer(kind=iintegers) :: ierr
 integer(kind=iintegers) :: recll, recnn
 integer(kind=iintegers) :: i, j

 character(len=4) :: chwork

 allocate(work (coords%i0_write : coords%i1_write, coords%j0_write : coords%j1_write))
 allocate(work2(coords%i0_write : coords%i1_write, coords%j0_write : coords%j1_write))

 if (coords%isx /= 1_iintegers .or. coords%isy /= 1_iintegers) then
   write(*,*) 'isx/=1 and isy/=1 does not work in current &
   & version of subroutine WRFILD_2D: STOP'
   STOP
 endif

 work = 0.
 !if (parparams%coords(3) == 0) then ! for land surface arrays
   work(max(coords%i00,coords%i0_write) : min(coords%i11,coords%i1_write), &
   &    max(coords%j00,coords%j0_write) : min(coords%j11,coords%j1_write)) = &
   & xa(max(coords%i00,coords%i0_write) : min(coords%i11,coords%i1_write), &
   &    max(coords%j00,coords%j0_write) : min(coords%j11,coords%j1_write))
 !endif

#ifdef mpi
 if (parparams%parallel) then
   if (ireals == 4) then
     i = MPI_REAL
   elseif (ireals == 8) then
     i = MPI_DOUBLE_PRECISION
   endif  
   call MPI_REDUCE(work, work2, &
   &              (coords%i1_write - coords%i0_write + 1) * &
   &              (coords%j1_write - coords%j0_write + 1), &
   &              i, MPI_SUM, 0, parparams%comm3d, ierr)
   work = work2
 endif
#endif

 deallocate(work2)

 if (parparams%rank_comm3d == 0) then
   recll = coords%lrec*(coords%i1_write - coords%i0_write + 1)* &
   &                   (coords%j1_write - coords%j0_write + 1)

   if(iotsinftstep) then
     call CHECK_UNIT(lake_misc_unit_min, lake_misc_unit_max, unitout)
     open(unitout,file=filen,status='unknown',form='unformatted', &
     &  access='direct',recl=recll)
     recnn = coords%recn
     call WRITEARR
     coords%recn = coords%recn + 1
     close(unitout)
   endif

   if (iotsepftstep) then
     call CHECK_UNIT(lake_misc_unit_min, lake_misc_unit_max, unitout)
     write(chwork,'(i4.4)') coords%recn
     open(unitout,file=filen//chwork,status='unknown', &
     & form='unformatted',access='direct',recl=recll)
     recnn = 1
     call WRITEARR
     coords%recn = coords%recn + 1
     close(unitout)
   endif

 endif

 deallocate (work)

 contains
 SUBROUTINE WRITEARR

 !The case, when 4-byte reals are to be written with 8-byte accuracy, is not considered
 if (coords%lrec == ibyterecl .and. ireals == 8) then 
   write(unitout,rec=recnn) ( (sngl(work(i,j)), &
   &                                 i = coords%i0_write, coords%i1_write, coords%isx), &
   &                                 j = coords%j0_write, coords%j1_write, coords%isy ) 
 else
   write(unitout,rec=recnn) ( (work(i,j), &
   &                                 i = coords%i0_write, coords%i1_write, coords%isx), &
   &                                 j = coords%j0_write, coords%j1_write, coords%isy)
 endif

 END SUBROUTINE WRITEARR


 END SUBROUTINE WRFILD_2D_LAKE


 SUBROUTINE WRFILD_3D_LAKE &
 & (xa, coords, recn, unitout, iotsepftstep, iotsinftstep, parparams, filen)

 ! Writes distributed 3D array in a file

 use INOUT_PARAMETERS, only : lake_misc_unit_min, lake_misc_unit_max

 implicit none

#ifdef mpi
   include 'mpif.h'
#endif

!Input/output variables
 type(wrfild_3d_coords), intent(in) :: coords
 type(parallel_params),  intent(in) :: parparams

 integer(kind=iintegers), intent(inout) :: recn
 integer(kind=iintegers), intent(inout) :: unitout
 logical, intent(in) :: iotsepftstep
 logical, intent(in) :: iotsinftstep

 real(kind=ireals), intent(in) :: &
 & xa(coords%i0 : coords%i1, coords%j0 : coords%j1, coords%k0 : coords%k1)

 character(len=*), intent(in) :: filen

!Local variables
 real(kind = ireals), allocatable :: work(:,:), work2(:,:)

 integer(kind=iintegers) :: ierr
 integer(kind=iintegers) :: recll, recnn, recnn1
 integer(kind=iintegers) :: i, j, k

 character(len=4) :: chwork

 allocate(work (coords%i0_write : coords%i1_write, coords%j0_write : coords%j1_write)) 
 allocate(work2(coords%i0_write : coords%i1_write, coords%j0_write : coords%j1_write)) 

 if (coords%isx /= 1_iintegers .or. &
   & coords%isy /= 1_iintegers .or. coords%iss /= 1_iintegers) then
   write(*,*) 'isx/=1 and isy/=1 and iss/=1 do not work in current &
   & version of subroutine WRFILD_3D: STOP'
   STOP
 endif

 if (parparams%rank_comm3d == 0) then
   recll = coords%lrec*(coords%i1_write - coords%i0_write + 1)* &
   &                   (coords%j1_write - coords%j0_write + 1)
   if (iotsinftstep) then 
     call CHECK_UNIT(lake_misc_unit_min, lake_misc_unit_max, unitout)
     open(unitout,file=filen,status='unknown', &
     & form='unformatted', access='direct',recl=recll)
     recnn = recn
   endif
 endif
 
 kcycle : do k = coords%k1_write, coords%k0_write, - coords%iss

   work = 0.
   if (k >= coords%k00 .and. k <= coords%k11) then
     work(max(coords%i00,coords%i0_write):min(coords%i11,coords%i1_write), &
     &    max(coords%j00,coords%j0_write):min(coords%j11,coords%j1_write)) = &
     & xa(max(coords%i00,coords%i0_write):min(coords%i11,coords%i1_write), &
     &    max(coords%j00,coords%j0_write):min(coords%j11,coords%j1_write),k)
   endif

#ifdef mpi
   if (parparams%parallel) then
     if (ireals == 4) then
       i = MPI_REAL
     elseif (ireals == 8) then
       i = MPI_DOUBLE_PRECISION
     endif  
     call MPI_REDUCE(work, work2, &
     &              (coords%i1_write - coords%i0_write + 1)* &
     &              (coords%j1_write - coords%j0_write + 1), &
     &               i, MPI_SUM, 0, parparams%comm3d, ierr)
     work = work2
   endif
#endif

   if (parparams%rank_comm3d == 0) then
!    lrec = 8 if double precision
     if (iotsinftstep) then
       recnn = recnn + 1
       call WRITEARR(recnn,unitout)
     endif
   endif
 
 enddo kcycle



 if (parparams%rank_comm3d == 0) then
   recn = recnn
   if (iotsinftstep) then
     close(unitout)
   endif
 endif

 deallocate (work,work2)

 contains
 SUBROUTINE WRITEARR(recnn_,unitout_)

 implicit none

 integer(kind=iintegers), intent(in) :: recnn_, unitout_

 if (coords%lrec == ibyterecl .and. ireals == 8) then
   write(unitout_,rec=recnn_)((sngl(work(i,j)), i = coords%i0_write, coords%i1_write, coords%isx) &
   &                                          , j = coords%j0_write, coords%j1_write, coords%isy)
 else
   write(unitout_,rec=recnn_)((      work(i,j), i = coords%i0_write, coords%i1_write, coords%isx) &
   &                                          , j = coords%j0_write, coords%j1_write, coords%isy)
 endif


 END SUBROUTINE WRITEARR

 END SUBROUTINE WRFILD_3D_LAKE


 SUBROUTINE RDFILD_3D_LAKE &
 & (coords, recn, unitout, iotsepftstep, iotsinftstep, parparams, filen, xa)

 ! Writes distributed 3D array in a file

 use INOUT_PARAMETERS, only : lake_misc_unit_min, lake_misc_unit_max

 implicit none

#ifdef mpi
   include 'mpif.h'
#endif

!Input/output variables
 type(wrfild_3d_coords), intent(in) :: coords
 type(parallel_params),  intent(in) :: parparams

 integer(kind=iintegers), intent(inout) :: recn
 integer(kind=iintegers), intent(inout) :: unitout
 logical, intent(in) :: iotsepftstep
 logical, intent(in) :: iotsinftstep

 character(len=*), intent(in) :: filen

 real(kind=ireals), intent(out) :: &
 & xa(coords%i0 : coords%i1, coords%j0 : coords%j1, coords%k0 : coords%k1)

!Local variables
 real(kind = ireals), allocatable :: work(:,:), work2(:,:)

 integer(kind=iintegers) :: ierr
 integer(kind=iintegers) :: recll, recnn, recnn1
 integer(kind=iintegers) :: i, j, k

 character(len=4) :: chwork

 allocate(work (coords%i0_write : coords%i1_write, coords%j0_write : coords%j1_write)) !, k0_write:k1_write))
 allocate(work2(coords%i0_write : coords%i1_write, coords%j0_write : coords%j1_write)) !, k0_write:k1_write))


 if (coords%isx /= 1_iintegers .or. &
   & coords%isy /= 1_iintegers .or. coords%iss /= 1_iintegers) then
   write(*,*) 'isx/=1 and isy/=1 and iss/=1 do not work in current &
   & version of subroutine RDFILD_3D: STOP'
   STOP
 endif

 if (parparams%rank_comm3d == 0) then
   recll = coords%lrec*(coords%i1_write - coords%i0_write + 1)* &
   &                   (coords%j1_write - coords%j0_write + 1)
   if (iotsinftstep) then 
     call CHECK_UNIT(lake_misc_unit_min, lake_misc_unit_max, unitout)
     open(unitout,file=filen,status='old', &
     & form='unformatted', access='direct',recl=recll)
     recnn = recn
   !lrec = 8 if double precision
   endif
   !if (iotsepftstep) then
   !  call CHECK_UNIT(lake_misc_unit_min, lake_misc_unit_max, unitout)
   !  write(chwork,'(i4.4)') recn
   !  open(unitout+1,file=filen//chwork,status='unknown', &
   !  & form='unformatted',access='direct',recl=recll)
   !  recnn1 = 0
   !endif

   !if (iotsepftstep) then
   !  recnn1 = recnn1 + 1
   !  call WRITEARR(recnn1,unitout+1)
   !endif
 endif
 
 kcycle : do k = coords%k1_write, coords%k0_write, - coords%iss

   work = 0.
   if (iotsinftstep .and. parparams%rank_comm3d == 0) then 
     recnn = recnn + 1
     !print*, 'Read lake CP, recnn = ', recnn, k, maxval(work), minval(work)
     call READARR(recnn,unitout)
   endif
   !call MPI_BARRIER(parparams%comm3d,ierr)
   !if (parparams%rank_comm3d == 0) print*, 'Read passed'

   
#ifdef mpi
   if (parparams%parallel) then
     if (ireals == 4) then
       i = MPI_REAL
     elseif (ireals == 8) then
       i = MPI_DOUBLE_PRECISION
     endif  
     call MPI_ALLREDUCE(work, work2, &
     &                 (coords%i1_write - coords%i0_write + 1)* &
     &                 (coords%j1_write - coords%j0_write + 1), &
     &                 i, MPI_SUM, parparams%comm3d, ierr)
     work = work2
   endif
#endif
   !call MPI_BARRIER(parparams%comm3d,ierr)
   !if (parparams%rank_comm3d == 0) print*, 'Exchange passed'

   if (k >= coords%k00 .and. k <= coords%k11) then
     xa    (max(coords%i00,coords%i0_write):min(coords%i11,coords%i1_write), &
     &      max(coords%j00,coords%j0_write):min(coords%j11,coords%j1_write),k) = &
     & work(max(coords%i00,coords%i0_write):min(coords%i11,coords%i1_write), &
     &      max(coords%j00,coords%j0_write):min(coords%j11,coords%j1_write))
   endif
   !call MPI_BARRIER(parparams%comm3d,ierr)
   !if (parparams%rank_comm3d == 0) print*, 'Assignment passed'

 enddo kcycle



 if (parparams%rank_comm3d == 0) then
   recn = recnn
   if (iotsinftstep) then
     close(unitout)
   endif
   !if (iotsepftstep) then
   !  close(unitout+1)
   !endif
 endif

 deallocate (work,work2)

 contains
 SUBROUTINE READARR(recnn_,unitout_)

 implicit none

 integer(kind=iintegers), intent(in) :: recnn_, unitout_
 real(kind=4), allocatable :: work1(:,:)

 if (coords%lrec == ibyterecl .and. ireals == 8) then
   allocate (work1(coords%i0_write:coords%i1_write, coords%j0_write:coords%j1_write))
   read(unitout_,rec=recnn_)((work1(i,j), i = coords%i0_write, coords%i1_write, coords%isx) &
   &                                    , j = coords%j0_write, coords%j1_write, coords%isy)
   work   (coords%i0_write:coords%i1_write, coords%j0_write:coords%j1_write) = &
   & work1(coords%i0_write:coords%i1_write, coords%j0_write:coords%j1_write)
   deallocate(work1)
 else
   read(unitout_ ,rec=recnn_)((work(i,j), i = coords%i0_write, coords%i1_write, coords%isx) &
   &                                    , j = coords%j0_write, coords%j1_write, coords%isy)
 endif

 END SUBROUTINE READARR

 END SUBROUTINE RDFILD_3D_LAKE

END MODULE INOUT
