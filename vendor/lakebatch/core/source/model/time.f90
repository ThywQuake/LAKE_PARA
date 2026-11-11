MODULE TIME_LAKE_MOD

contains
SUBROUTINE JULIAN_DATE(year,month,day,hour,dt)

! The subroutine JULIAN_DATE updates the julian date
! after timestep dt

use LAKE_DATATYPES, only : ireals, iintegers

implicit none

real(kind=ireals), parameter:: hour_dur = 3600.d0 ! The duration of hour, s 

!Input variables
real(kind=ireals)   , intent(in)   :: dt  ! Timestep, s

!Input/output variables
integer(kind=iintegers), intent(inout):: year
integer(kind=iintegers), intent(inout):: month
integer(kind=iintegers), intent(inout):: day

real(kind=ireals)   , intent(inout):: hour

!Local variables
integer(kind=iintegers) ndaym(12)
data       ndaym /31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/

SAVE ndaym

if (mod(year,4  ) == 0 .and. .not. &
&  (mod(year,100) == 0 .and. mod(year,400) /= 0)) then ! Checking for leap-year
  ndaym(2) = 29 
else
  ndaym(2) = 28
endif

hour = hour + dt/hour_dur
if (hour > 24.d0) then
  hour = hour - 24.d0
  day  = day  + 1
  if (day > ndaym(month)) then
    day   = 1
    month = month + 1
    if (month > 12) then
      month = 1
      year  = year + 1
    endif 
  endif
endif

END SUBROUTINE JULIAN_DATE



integer(kind=iintegers) FUNCTION JULIAN_DAY(year,month,day)

use LAKE_DATATYPES, only : ireals, iintegers
implicit none

integer(kind=iintegers), intent(in):: year
integer(kind=iintegers), intent(in):: month
integer(kind=iintegers), intent(in):: day

integer(kind=iintegers) i

integer(kind=iintegers) ndaym(12)
data       ndaym /31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/

SAVE ndaym

if (mod(year,4  ) == 0 .and. .not. &
&  (mod(year,100) == 0 .and. mod(year,400) /= 0)) then ! Checking for leap-year
  ndaym(2) = 29
else
  ndaym(2) = 28
endif

if (month == 1) then
  JULIAN_DAY = day
else
  JULIAN_DAY = 0 
  do i = 1, month-1
    JULIAN_DAY = JULIAN_DAY + ndaym(i)
  enddo
  JULIAN_DAY = JULIAN_DAY + day
endif

END FUNCTION JULIAN_DAY



SUBROUTINE DATEMINUS(n,year,month,day,hour,year1,month1,day1,hour1)

use driving_params
use LAKE_DATATYPES, only : ireals, iintegers

implicit none
real(kind=ireals) hour
integer(kind=iintegers) year,month,day
integer(kind=iintegers) n,ndaym(12)
character year1*4,month1*2,day1*2,hour1*2
data ndaym /31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31/
      
SAVE
      
if (n == 1) then
! Minus 1 month      
  if (month == 1) then
    write (month1,'(i2)') 12
    write (year1, '(i4)') year - 1
  else
    write (month1,'(i2)') month - 1
    write (year1, '(i4)') year
  endif 
elseif (n == 2) then
! Minus 1 day      
  if (day == 1) then
    if (month == 1) then
      write (month1,'(i2)') 12
      write (year1, '(i4)') year - 1
      write (day1,  '(i2)') 31
    elseif (mod(year,4) == 0.and.month == 3) then !Checking for leap-year
      write (month1,'(i2)') month - 1
      write (year1, '(i4)') year
      write (day1,  '(i2)') ndaym(month-1) + 1 ! 29 days 
    else
      write (month1,'(i2)') month - 1
      write (year1, '(i4)') year
      write (day1,  '(i2)') ndaym(month-1) 
    endif
  else
    write (month1,'(i2)') month
    write (year1, '(i4)') year
    write (day1,  '(i2)') day - 1 
  endif
elseif (n == 3) then
! Minus 1 hour      
  if (int(hour) == 0) then
    if (day == 1) then
      if (month == 1) then
        write (month1,'(i2)') 12
        write (year1, '(i4)') year-1
        write (day1,  '(i2)') 31
        write (hour1, '(i2)') 23  !24-int(interval)
      elseif (mod(year,4) == 0.and.month == 3) then !Checking for leap-year
        write (month1,'(i2)') month-1
        write (year1, '(i4)') year
        write (day1,  '(i2)') ndaym(month-1) + 1 ! 29 days
        write (hour1, '(i2)') 23 !24-int(interval)
      else
        write (month1,'(i2)') month-1
        write (year1, '(i4)') year
        write (day1,  '(i2)') ndaym(month-1) 
        write (hour1, '(i2)') 23 !24-int(interval)
      endif
    else
      write (month1,'(i2)') month
      write (year1, '(i4)') year
      write (day1,  '(i2)') day-1
      write (hour1, '(i2)') 23 !24-int(interval)
    endif
  else
    write (month1,'(i2)') month
    write (year1, '(i4)') year
    write (day1,  '(i2)') day
    write (hour1, '(i2)') int(hour)-1 !int(interval)
  endif
else
  print*, 'Invalid meaning of control variable N in &
&  subroutine DATEMINUS: STOP'
  STOP
endif
      
END SUBROUTINE DATEMINUS
      
                  
SUBROUTINE TIMESTR(n,y,m,d,h,timestring)
use LAKE_DATATYPES, only : ireals, iintegers
implicit none
integer n,i
character y*4,m*2,d*2,h*2
character(len=n) timestring 
      
if (n==10) then
  write (timestring, '(a4,3a2)') y,m,d,h 
elseif (n==8) then
  write (timestring, '(a4,2a2)') y,m,d 
elseif (n==6) then
  write (timestring, '(a4,a2)') y,m 
endif 
      
do i = 1, n
  if (timestring(i:i)==' '.or.timestring(i:i)==char(0)) &
&  timestring(i:i)='0'
enddo
      
END SUBROUTINE TIMESTR


SUBROUTINE TIMEC(n)

use LAKE_DATATYPES, only : ireals, iintegers
use ARRAYS, only: &
& ncomp, comptime

implicit none

! Input variables
integer(kind=iintegers), intent(in) :: n

! Local variables
real(kind=ireals), save :: t(1:2)
integer(kind=iintegers), save :: nsw = 1

if (n == 0) then
! Initialization
  call CPU_TIME(t(nsw))
else
  call CPU_TIME(t(3-nsw))
  comptime(n) = comptime(n) + t(3-nsw) - t(nsw)
  nsw = 3 - nsw 
endif

END SUBROUTINE TIMEC

END MODULE TIME_LAKE_MOD
