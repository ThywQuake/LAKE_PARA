!> lakebatch: write LAKE's time-series products straight to NetCDF at full precision.
!!
!! Controlled by environment variables read once at the first call:
!!   LB_NC_OUT=<path>   write every time-series product to this NetCDF-4 file
!!   LB_ASCII_OUT=0     do not write the ASCII time-series files (default: write them)
!! With neither set the model behaves exactly as LAKE 3.2.
!!
!! Layout of the file: one group per product (the ASCII file name up to the
!! grid-point suffix, e.g. "T_fluxes").  Each group holds
!!   data(record, column)  double   the ASCII row, column k = "Col. k"
!!   attribute "header"            the ASCII header lines, newline separated
!!   attribute "file"              the ASCII file name
!! so a product carries exactly the values of its ASCII file, unrounded.
!!
!! Products are identified by the Fortran unit their ASCII file is opened on;
!! header lines and rows are routed here through LB_HEADER / LB_ROW / LB_PROFILE_ROW.
!! This module holds no model state, so it is not part of a restart.
MODULE LB_NCOUT

use, intrinsic :: iso_c_binding, only : c_int, c_size_t, c_double, c_char, c_null_char, c_ptr, &
& c_f_pointer
use LAKE_DATATYPES, only : ireals, iintegers

implicit none
private

public :: LB_ASCII_OUT, LB_HEADER, LB_ASCII_HEADER, LB_ROW, LB_PROFILE_ROW, LB_NC_CLOSE

integer(c_int), parameter :: NC_NETCDF4 = 4096, NC_CLOBBER = 0, NC_DOUBLE = 6, NC_GLOBAL = -1, &
& NC_CHUNKED = 0
integer(c_size_t), parameter :: NC_UNLIMITED = 0
integer, parameter :: maxprod = 200, chunk_rows = 512

type product_t
  integer :: unit = -1
  integer(c_int) :: grp = -1, var = -1
  integer :: ncol = -1
  integer(c_size_t) :: nrec = 0
  character(len=:), allocatable :: header
end type product_t

logical, save :: initialised = .false.
logical, save :: ascii_on = .true., nc_on = .false.
integer(c_int), save :: ncid = -1
integer, save :: nprod = 0
type(product_t), save :: prods(maxprod)

interface
  integer(c_int) function nc_create(path, cmode, ncidp) bind(c, name='nc_create')
    import :: c_int, c_char
    character(kind=c_char), intent(in) :: path(*)
    integer(c_int), value :: cmode
    integer(c_int), intent(out) :: ncidp
  end function
  integer(c_int) function nc_def_grp(parent, name, grpid) bind(c, name='nc_def_grp')
    import :: c_int, c_char
    integer(c_int), value :: parent
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int), intent(out) :: grpid
  end function
  integer(c_int) function nc_def_dim(id, name, len, dimid) bind(c, name='nc_def_dim')
    import :: c_int, c_char, c_size_t
    integer(c_int), value :: id
    character(kind=c_char), intent(in) :: name(*)
    integer(c_size_t), value :: len
    integer(c_int), intent(out) :: dimid
  end function
  integer(c_int) function nc_def_var(id, name, xtype, ndims, dimids, varid) bind(c, name='nc_def_var')
    import :: c_int, c_char
    integer(c_int), value :: id, xtype, ndims
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int), intent(in) :: dimids(*)
    integer(c_int), intent(out) :: varid
  end function
  integer(c_int) function nc_def_var_chunking(id, varid, storage, chunks) bind(c, name='nc_def_var_chunking')
    import :: c_int, c_size_t
    integer(c_int), value :: id, varid, storage
    integer(c_size_t), intent(in) :: chunks(*)
  end function
  integer(c_int) function nc_put_att_text(id, varid, name, len, op) bind(c, name='nc_put_att_text')
    import :: c_int, c_char, c_size_t
    integer(c_int), value :: id, varid
    character(kind=c_char), intent(in) :: name(*)
    integer(c_size_t), value :: len
    character(kind=c_char), intent(in) :: op(*)
  end function
  integer(c_int) function nc_put_vara_double(id, varid, start, count, op) bind(c, name='nc_put_vara_double')
    import :: c_int, c_size_t, c_double
    integer(c_int), value :: id, varid
    integer(c_size_t), intent(in) :: start(*), count(*)
    real(c_double), intent(in) :: op(*)
  end function
  integer(c_int) function nc_close(id) bind(c, name='nc_close')
    import :: c_int
    integer(c_int), value :: id
  end function
  type(c_ptr) function nc_strerror(status) bind(c, name='nc_strerror')
    import :: c_int, c_ptr
    integer(c_int), value :: status
  end function
end interface

contains

SUBROUTINE INIT
  character(len=1024) :: path
  integer :: n, st
  if (initialised) return
  initialised = .true.
  call get_environment_variable('LB_ASCII_OUT', path, n, st)
  if (st == 0) ascii_on = (trim(path) /= '0')
  call get_environment_variable('LB_NC_OUT', path, n, st)
  if (st == 0 .and. n > 0) then
    call check(nc_create(trim(path)//c_null_char, ior(NC_NETCDF4, NC_CLOBBER), ncid), 'create '//trim(path))
    call put_text(ncid, NC_GLOBAL, 'title', 'LAKE time-series products (lakebatch LB_NCOUT)')
    nc_on = .true.
  endif
END SUBROUTINE INIT

!> True when the ASCII time-series files are written.
LOGICAL FUNCTION LB_ASCII_OUT()
  call INIT
  LB_ASCII_OUT = ascii_on
END FUNCTION LB_ASCII_OUT

!> A header line of a product: written to its ASCII file and kept for the NetCDF group.
SUBROUTINE LB_HEADER(unit, text)
  integer(kind=iintegers), intent(in) :: unit
  character(len=*), intent(in) :: text
  integer :: k
  call INIT
  if (ascii_on) write (unit,*) text
  if (.not. nc_on) return
  k = find(int(unit), .true.)
  if (prods(k)%ncol >= 0) call fail('header line after the first row of a product')
  prods(k)%header = prods(k)%header//text//new_line('a')
END SUBROUTINE LB_HEADER

!> A header line that only belongs to the ASCII output (e.g. the long-format f2 files).
SUBROUTINE LB_ASCII_HEADER(unit, text)
  integer(kind=iintegers), intent(in) :: unit
  character(len=*), intent(in) :: text
  call INIT
  if (ascii_on) write (unit,*) text
END SUBROUTINE LB_ASCII_HEADER

!> One record of a scalar product: columns 1-5 then vals.
SUBROUTINE LB_ROW(unit, year, month, day, hour, time, vals)
  integer(kind=iintegers), intent(in) :: unit, year, month, day
  real(kind=ireals), intent(in) :: hour, time, vals(:)
  call INIT
  if (.not. nc_on) return
  call put_row(int(unit), [real(year, c_double), real(month, c_double), real(day, c_double), &
  &                        real(hour, c_double), real(time, c_double), real(vals, c_double)])
END SUBROUTINE LB_ROW

!> One record of a profile product: columns 1-5 then depth1, var1(1:nvars), depth2, ...
SUBROUTINE LB_PROFILE_ROW(unit, year, month, day, hour, time, z, prof)
  integer(kind=iintegers), intent(in) :: unit, year, month, day
  real(kind=ireals), intent(in) :: hour, time, z(:), prof(:,:)
  real(c_double), allocatable :: row(:)
  integer :: i, nv, j
  call INIT
  if (.not. nc_on) return
  nv = size(prof, 2)
  allocate (row(5 + size(z)*(1 + nv)))
  row(1:5) = [real(year, c_double), real(month, c_double), real(day, c_double), &
  &           real(hour, c_double), real(time, c_double)]
  j = 5
  do i = 1, size(z)
    row(j+1) = z(i)
    row(j+2:j+1+nv) = prof(i, 1:nv)
    j = j + 1 + nv
  enddo
  call put_row(int(unit), row)
END SUBROUTINE LB_PROFILE_ROW

!> Close the NetCDF file; called once at the end of the run.
SUBROUTINE LB_NC_CLOSE
  if (.not. nc_on) return
  call check(nc_close(ncid), 'close')
  nc_on = .false.
END SUBROUTINE LB_NC_CLOSE

! ---------------------------------------------------------------- internals

INTEGER FUNCTION find(unit, create)
  integer, intent(in) :: unit
  logical, intent(in) :: create
  integer :: k
  do k = 1, nprod
    if (prods(k)%unit == unit) then
      find = k
      return
    endif
  enddo
  if (.not. create) call fail('row written to a unit without header lines')
  if (nprod == maxprod) call fail('too many products')
  nprod = nprod + 1
  prods(nprod)%unit = unit
  prods(nprod)%header = ''
  find = nprod
END FUNCTION find

SUBROUTINE put_row(unit, row)
  integer, intent(in) :: unit
  real(c_double), intent(in) :: row(:)
  integer :: k
  integer(c_size_t) :: start(2), count(2)
  k = find(unit, .false.)
  if (prods(k)%ncol < 0) call define(k, size(row))
  if (size(row) /= prods(k)%ncol) call fail('row length changed in product '//group_name(unit))
  ! C order: data(record, column) -> start/count given slowest first
  start = [prods(k)%nrec, 0_c_size_t]
  count = [1_c_size_t, int(size(row), c_size_t)]
  call check(nc_put_vara_double(prods(k)%grp, prods(k)%var, start, count, row), 'write')
  prods(k)%nrec = prods(k)%nrec + 1
END SUBROUTINE put_row

SUBROUTINE define(k, ncol)
  integer, intent(in) :: k, ncol
  integer(c_int) :: drec, dcol
  integer(c_size_t) :: chunks(2)
  character(len=:), allocatable :: name
  name = group_name(prods(k)%unit)
  call check(nc_def_grp(ncid, name//c_null_char, prods(k)%grp), 'group '//name)
  call check(nc_def_dim(prods(k)%grp, 'record'//c_null_char, NC_UNLIMITED, drec), 'dim')
  call check(nc_def_dim(prods(k)%grp, 'column'//c_null_char, int(ncol, c_size_t), dcol), 'dim')
  call check(nc_def_var(prods(k)%grp, 'data'//c_null_char, NC_DOUBLE, 2_c_int, [drec, dcol], &
  &          prods(k)%var), 'var')
  chunks = [int(chunk_rows, c_size_t), int(ncol, c_size_t)]
  call check(nc_def_var_chunking(prods(k)%grp, prods(k)%var, NC_CHUNKED, chunks), 'chunking')
  call put_text(prods(k)%grp, NC_GLOBAL, 'header', prods(k)%header)
  call put_text(prods(k)%grp, NC_GLOBAL, 'file', file_name(prods(k)%unit))
  prods(k)%ncol = ncol
END SUBROUTINE define

!> Base name of the file connected to a unit.
FUNCTION file_name(unit) result(name)
  integer, intent(in) :: unit
  character(len=:), allocatable :: name
  character(len=1024) :: path
  integer :: i
  inquire (unit=unit, name=path)
  i = index(path, '/', back=.true.)
  name = trim(path(i+1:))
END FUNCTION file_name

!> Product name: the file name up to the grid-point suffix ("T_fluxes  1  1.dat" -> "T_fluxes").
FUNCTION group_name(unit) result(name)
  integer, intent(in) :: unit
  character(len=:), allocatable :: name
  integer :: i
  name = file_name(unit)
  i = index(name, '  ')
  if (i > 0) name = name(1:i-1)
  i = index(name, '.dat')
  if (i > 0) name = name(1:i-1)
END FUNCTION group_name

SUBROUTINE put_text(id, varid, name, text)
  integer(c_int), intent(in) :: id, varid
  character(len=*), intent(in) :: name, text
  call check(nc_put_att_text(id, varid, name//c_null_char, int(len(text), c_size_t), text), 'attribute '//name)
END SUBROUTINE put_text

SUBROUTINE check(status, what)
  integer(c_int), intent(in) :: status
  character(len=*), intent(in) :: what
  character(kind=c_char), pointer :: msg(:)
  integer :: n
  if (status == 0) return
  call c_f_pointer(nc_strerror(status), msg, [200])
  n = 0
  do while (n < 200)
    if (msg(n+1) == c_null_char) exit
    n = n + 1
  enddo
  call fail(what//': '//transfer(msg(1:n), repeat(' ', n)))
END SUBROUTINE check

SUBROUTINE fail(msg)
  character(len=*), intent(in) :: msg
  write (*,*) 'NETCDF OUTPUT ERROR: ', msg
  error stop 1
END SUBROUTINE fail

END MODULE LB_NCOUT
