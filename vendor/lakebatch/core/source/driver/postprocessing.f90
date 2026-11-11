MODULE POSTPROCESSING_LAKE
contains
! Postprocessing procedures
SUBROUTINE SIMPLE_MOVING_AVERAGE(series_input, series_averaged, &
& series_length, average_window)

! Subroutine SIMPLE_MOVING_AVERAGE performs moving (running) average
! of time series

use DRIVER_DATATYPES, only : ireals, iintegers

implicit none

! Input variables
! Integers

integer(kind=iintegers), intent(in) :: series_length
integer(kind=iintegers), intent(in) :: average_window

! Reals
real(kind=ireals), intent(in) :: series_input(1:series_length)

! Output variables
! Reals
real(kind=ireals), intent(out) :: series_averaged(1:series_length)

! Local variables

integer(kind=iintegers) :: cw0
integer(kind=iintegers) :: cw
integer(kind=iintegers) :: cw21
integer(kind=iintegers) :: i


if (mod(average_window,2) == 0) then
  write(*,*) 'Average window must be an odd number: STOP'
endif

cw0 = (average_window - 1)/2
series_averaged(1) = series_input(1)
series_averaged(series_length) = series_input(series_length)
do i = 2, series_length-1
  cw = min(i-1, cw0)
  cw = min(series_length - i, cw)
  cw21 = 2*cw + 1
  series_averaged(i) = sum(series_input(i-cw:i+cw))/real(cw21)
enddo

END SUBROUTINE SIMPLE_MOVING_AVERAGE


SUBROUTINE MEAN_CYCLE_CALC(series_input, mean_cycle, &
& series_length, cycle_period)

! Subroutine calculates the mean cycle of period cycle_period

use DRIVER_DATATYPES, only : ireals, iintegers
implicit none

! Input variables
! Integers

integer(kind=iintegers), intent(in) :: series_length
integer(kind=iintegers), intent(in) :: cycle_period

! Reals
real(kind=ireals), intent(in) :: series_input(1:series_length)

! Output variables
! Reals
real(kind=ireals), intent(out) :: mean_cycle(1:cycle_period)

! Local variables

real(kind=ireals), allocatable :: work(:,:)

integer(kind=iintegers) :: number_periods
integer(kind=iintegers) :: i ! Loop index


number_periods = int(series_length/cycle_period)
allocate(work(1:number_periods, 1:cycle_period))

do i = 1, number_periods
  work(i,1:cycle_period) = series_input((i-1)*cycle_period+1:i*cycle_period)
enddo

mean_cycle = 0.d0
do i = 1, number_periods
  mean_cycle(1:cycle_period) = mean_cycle(1:cycle_period) + work(i,1:cycle_period)
enddo
mean_cycle = mean_cycle/real(number_periods)

END SUBROUTINE MEAN_CYCLE_CALC

END MODULE POSTPROCESSING_LAKE
