!> Restart file handling for lakebatch (replaces CONTROL_POINT_IN/OUT).
!!
!! The restart holds everything one lake carries from one LAKE call to the next:
!! all variables of the state modules (generated LB_RESTART_IO) and all SAVE
!! locals of SUBROUTINE LAKE (generated lb_restart_lake.inc).  Both are produced
!! by core/tools/gen_restart_io.py from the declarations, so the list cannot
!! drift from the code.
!!
!! Activation needs no new keywords:
!!   restart_in.bin  present in the run directory -> state is restored at the first call
!!   restart_out/    directory present            -> state after the last step is
!!                                                   written to restart_out/state.bin
MODULE LAKEBATCH_RESTART

implicit none
private

public :: RESTART_IN_PRESENT, RESTART_OUT_REQUESTED, RESTART_OPEN, RESTART_CLOSE, RESTART_CLOCK_IO

character(len=*), parameter :: magic = 'LAKEBATCH-RESTART-3'
character(len=*), parameter :: file_in = 'restart_in.bin'
character(len=*), parameter :: file_out = 'restart_out/state.bin'

contains

logical function RESTART_IN_PRESENT()
  inquire(file=file_in, exist=RESTART_IN_PRESENT)
end function RESTART_IN_PRESENT

logical function RESTART_OUT_REQUESTED()
  inquire(file='restart_out/.', exist=RESTART_OUT_REQUESTED)
end function RESTART_OUT_REQUESTED

SUBROUTINE RESTART_OPEN(reading, u)
  logical, intent(in) :: reading
  integer, intent(out) :: u
  character(len=len(magic)) :: m
  integer :: ios
  if (reading) then
    open(newunit=u, file=file_in, access='stream', form='unformatted', status='old', &
    &    action='read', iostat=ios)
    if (ios /= 0) call fail('cannot open '//file_in)
    read(u, iostat=ios) m
    if (ios /= 0 .or. m /= magic) call fail(file_in//' is not a restart file of this version')
  else
    open(newunit=u, file=file_out, access='stream', form='unformatted', status='replace', &
    &    action='write', iostat=ios)
    if (ios /= 0) call fail('cannot create '//file_out)
    write(u) magic
  endif
END SUBROUTINE RESTART_OPEN

SUBROUTINE RESTART_CLOSE(reading, u)
  logical, intent(in) :: reading
  integer, intent(in) :: u
  character(len=len(magic)) :: m
  integer :: ios
  if (reading) then
    read(u, iostat=ios) m
    if (ios /= 0 .or. m /= magic) call fail('trailing marker missing in '//file_in)
  else
    write(u) magic
  endif
  close(u)
END SUBROUTINE RESTART_CLOSE

!> The driver's clock (accumulated in floating point every step) is part of the
!! state: restoring it exactly keeps forcing interpolation and dates bit-identical.
SUBROUTINE RESTART_CLOCK_IO(reading, year, month, day, hour, time)
  use LAKE_DATATYPES, only : ireals, iintegers
  logical, intent(in) :: reading
  integer(kind=iintegers), intent(inout) :: year, month, day
  real(kind=ireals), intent(inout) :: hour, time
  character(len=len(magic)) :: m
  integer :: u, ios
  if (reading) then
    open(newunit=u, file='restart_in_clock.bin', access='stream', form='unformatted', &
    &    status='old', action='read', iostat=ios)
    if (ios /= 0) call fail('restart_in.bin given without restart_in_clock.bin')
    read(u, iostat=ios) m, year, month, day, hour, time
    if (ios /= 0 .or. m /= magic) call fail('restart_in_clock.bin is not a clock file of this version')
  else
    open(newunit=u, file='restart_out/clock.bin', access='stream', form='unformatted', &
    &    status='replace', action='write', iostat=ios)
    if (ios /= 0) call fail('cannot create restart_out/clock.bin')
    write(u) magic, year, month, day, hour, time
  endif
  close(u)
END SUBROUTINE RESTART_CLOCK_IO

SUBROUTINE fail(msg)
  character(len=*), intent(in) :: msg
  write(*,*) 'RESTART ERROR: ', msg
  write(*,*) 'Terminating LAKE model ...'
  error stop 3
END SUBROUTINE fail

END MODULE LAKEBATCH_RESTART
