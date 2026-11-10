!> Binary probe for the lakematrix reference comparison (M0.4).
!!
!! `LM_PROBE(tag, name, x)` appends one record to `probe/<nstep>/<tag>.bin`
!! when the current model step is listed in the environment variable
!! `LM_PROBE_STEPS` (comma separated).  A record is:
!!
!!   name       character(len=64)
!!   dtype      character(len=1)   f = real(8), i = integer(4), b = logical
!!   rank       integer(8)
!!   lower(1:rank), upper(1:rank)  integer(8) each
!!   data       rank-0: one value; rank>0: Fortran column-major order
!!
!! Data are little-endian (the target machines are little-endian).  The Python
!! reader is `lakematrix/tools/ref/probe.py`.
MODULE LM_PROBE_MOD

use, intrinsic :: iso_fortran_env, only : int8, int64, real64
use ARRAYS, only : nstep

implicit none
private
public :: LM_PROBE

integer, parameter :: MAX_STEPS = 128
integer(int64) :: steps(MAX_STEPS)
integer :: nsteps = 0
logical :: initialised = .false.

interface LM_PROBE
  module procedure p_r8_0, p_r8_1, p_r8_2, p_r8_3
  module procedure p_i4_0, p_i4_1, p_i4_2, p_i4_3
  module procedure p_l4_0, p_l4_1, p_l4_2, p_l4_3
end interface

contains

!> Parse LM_PROBE_STEPS once.
subroutine init_steps()
  character(len=1024) :: env
  integer :: st, pos, comma
  integer(int64) :: v

  if (initialised) return
  initialised = .true.
  call get_environment_variable('LM_PROBE_STEPS', env, status=st)
  if (st /= 0 .or. len_trim(env) == 0) return
  pos = 1
  do while (pos <= len_trim(env) .and. nsteps < MAX_STEPS)
    comma = index(env(pos:), ',')
    if (comma == 0) then
      read(env(pos:), *, iostat=st) v
      pos = len_trim(env) + 1
    else
      read(env(pos:pos + comma - 2), *, iostat=st) v
      pos = pos + comma
    endif
    if (st == 0) then
      nsteps = nsteps + 1
      steps(nsteps) = v
    endif
  enddo
end subroutine init_steps

logical function probe_step()
  integer :: i
  call init_steps()
  probe_step = .false.
  do i = 1, nsteps
    if (steps(i) == nstep) then
      probe_step = .true.
      return
    endif
  enddo
end function probe_step

!> Open (append) the file for this step and tag.
subroutine open_probe(tag, u)
  character(len=*), intent(in) :: tag
  integer, intent(out) :: u
  character(len=512) :: dir, path

  write(dir, '("probe/", i0)') nstep
  call execute_command_line('mkdir -p ' // trim(dir))
  path = trim(dir) // '/' // trim(tag) // '.bin'
  open(newunit=u, file=trim(path), access='stream', form='unformatted', &
  &    position='append', status='unknown', action='write')
end subroutine open_probe

subroutine write_header(u, name, dtype, rank, lb, ub)
  integer, intent(in) :: u
  character(len=*), intent(in) :: name, dtype
  integer, intent(in) :: rank
  integer(int64), intent(in) :: lb(:), ub(:)
  character(len=64) :: nm
  character(len=1) :: dt
  integer(int64) :: rk

  nm = name
  dt = dtype
  rk = int(rank, int64)
  write(u) nm
  write(u) dt
  write(u) rk
  if (rank > 0) write(u) lb(1:rank)
  if (rank > 0) write(u) ub(1:rank)
end subroutine write_header

! --- real(8) -------------------------------------------------------------

subroutine p_r8_0(tag, name, x)
  character(len=*), intent(in) :: tag, name
  real(real64), intent(in) :: x
  integer :: u
  integer(int64) :: lb(1), ub(1)
  if (.not. probe_step()) return
  call open_probe(tag, u)
  call write_header(u, name, 'f', 0, lb, ub)
  write(u) x
  close(u)
end subroutine p_r8_0

subroutine p_r8_1(tag, name, x)
  character(len=*), intent(in) :: tag, name
  real(real64), intent(in) :: x(:)
  integer :: u
  integer(int64) :: lb(1), ub(1)
  if (.not. probe_step()) return
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  call open_probe(tag, u)
  call write_header(u, name, 'f', 1, lb, ub)
  write(u) x
  close(u)
end subroutine p_r8_1

subroutine p_r8_2(tag, name, x)
  character(len=*), intent(in) :: tag, name
  real(real64), intent(in) :: x(:, :)
  integer :: u
  integer(int64) :: lb(2), ub(2)
  if (.not. probe_step()) return
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  lb(2) = lbound(x, 2); ub(2) = ubound(x, 2)
  call open_probe(tag, u)
  call write_header(u, name, 'f', 2, lb, ub)
  write(u) x
  close(u)
end subroutine p_r8_2

subroutine p_r8_3(tag, name, x)
  character(len=*), intent(in) :: tag, name
  real(real64), intent(in) :: x(:, :, :)
  integer :: u
  integer(int64) :: lb(3), ub(3)
  if (.not. probe_step()) return
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  lb(2) = lbound(x, 2); ub(2) = ubound(x, 2)
  lb(3) = lbound(x, 3); ub(3) = ubound(x, 3)
  call open_probe(tag, u)
  call write_header(u, name, 'f', 3, lb, ub)
  write(u) x
  close(u)
end subroutine p_r8_3

! --- integer(4) ----------------------------------------------------------

subroutine p_i4_0(tag, name, x)
  character(len=*), intent(in) :: tag, name
  integer, intent(in) :: x
  integer :: u
  integer(int64) :: lb(1), ub(1)
  if (.not. probe_step()) return
  call open_probe(tag, u)
  call write_header(u, name, 'i', 0, lb, ub)
  write(u) x
  close(u)
end subroutine p_i4_0

subroutine p_i4_1(tag, name, x)
  character(len=*), intent(in) :: tag, name
  integer, intent(in) :: x(:)
  integer :: u
  integer(int64) :: lb(1), ub(1)
  if (.not. probe_step()) return
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  call open_probe(tag, u)
  call write_header(u, name, 'i', 1, lb, ub)
  write(u) x
  close(u)
end subroutine p_i4_1

subroutine p_i4_2(tag, name, x)
  character(len=*), intent(in) :: tag, name
  integer, intent(in) :: x(:, :)
  integer :: u
  integer(int64) :: lb(2), ub(2)
  if (.not. probe_step()) return
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  lb(2) = lbound(x, 2); ub(2) = ubound(x, 2)
  call open_probe(tag, u)
  call write_header(u, name, 'i', 2, lb, ub)
  write(u) x
  close(u)
end subroutine p_i4_2

subroutine p_i4_3(tag, name, x)
  character(len=*), intent(in) :: tag, name
  integer, intent(in) :: x(:, :, :)
  integer :: u
  integer(int64) :: lb(3), ub(3)
  if (.not. probe_step()) return
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  lb(2) = lbound(x, 2); ub(2) = ubound(x, 2)
  lb(3) = lbound(x, 3); ub(3) = ubound(x, 3)
  call open_probe(tag, u)
  call write_header(u, name, 'i', 3, lb, ub)
  write(u) x
  close(u)
end subroutine p_i4_3

! --- logical (written as int8) -------------------------------------------

subroutine p_l4_0(tag, name, x)
  character(len=*), intent(in) :: tag, name
  logical, intent(in) :: x
  integer :: u
  integer(int64) :: lb(1), ub(1)
  integer(int8) :: v
  if (.not. probe_step()) return
  v = merge(1_int8, 0_int8, x)
  call open_probe(tag, u)
  call write_header(u, name, 'b', 0, lb, ub)
  write(u) v
  close(u)
end subroutine p_l4_0

subroutine p_l4_1(tag, name, x)
  character(len=*), intent(in) :: tag, name
  logical, intent(in) :: x(:)
  integer :: u
  integer(int64) :: lb(1), ub(1)
  integer(int8), allocatable :: v(:)
  if (.not. probe_step()) return
  allocate(v(size(x))); v = merge(1_int8, 0_int8, x)
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  call open_probe(tag, u)
  call write_header(u, name, 'b', 1, lb, ub)
  write(u) v
  close(u)
  deallocate(v)
end subroutine p_l4_1

subroutine p_l4_2(tag, name, x)
  character(len=*), intent(in) :: tag, name
  logical, intent(in) :: x(:, :)
  integer :: u
  integer(int64) :: lb(2), ub(2)
  integer(int8), allocatable :: v(:, :)
  if (.not. probe_step()) return
  allocate(v(size(x, 1), size(x, 2))); v = merge(1_int8, 0_int8, x)
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  lb(2) = lbound(x, 2); ub(2) = ubound(x, 2)
  call open_probe(tag, u)
  call write_header(u, name, 'b', 2, lb, ub)
  write(u) v
  close(u)
  deallocate(v)
end subroutine p_l4_2

subroutine p_l4_3(tag, name, x)
  character(len=*), intent(in) :: tag, name
  logical, intent(in) :: x(:, :, :)
  integer :: u
  integer(int64) :: lb(3), ub(3)
  integer(int8), allocatable :: v(:, :, :)
  if (.not. probe_step()) return
  allocate(v(size(x, 1), size(x, 2), size(x, 3))); v = merge(1_int8, 0_int8, x)
  lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
  lb(2) = lbound(x, 2); ub(2) = ubound(x, 2)
  lb(3) = lbound(x, 3); ub(3) = ubound(x, 3)
  call open_probe(tag, u)
  call write_header(u, name, 'b', 3, lb, ub)
  write(u) v
  close(u)
  deallocate(v)
end subroutine p_l4_3

END MODULE LM_PROBE_MOD
