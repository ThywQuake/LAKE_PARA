MODULE INOUT_PARAMETERS

use LAKE_DATATYPES, only : ireals, iintegers

integer(kind=iintegers), parameter :: lake_subr_unit_min = 11001
integer(kind=iintegers), parameter :: lake_subr_unit_max = 12000
integer(kind=iintegers), parameter :: lake_mon_out_unit_min = 12001
integer(kind=iintegers), parameter :: lake_mon_out_unit_max = 13000
integer(kind=iintegers), parameter :: lake_day_out_unit_min = 13001
integer(kind=iintegers), parameter :: lake_day_out_unit_max = 14000
integer(kind=iintegers), parameter :: lake_hour_out_unit_min = 14001
integer(kind=iintegers), parameter :: lake_hour_out_unit_max = 15000
integer(kind=iintegers), parameter :: lake_everystep_out_unit_min = 15001
integer(kind=iintegers), parameter :: lake_everystep_out_unit_max = 16000
integer(kind=iintegers), parameter :: lake_series_out_unit_min = 16001
integer(kind=iintegers), parameter :: lake_series_out_unit_max = 17000
integer(kind=iintegers), parameter :: lake_misc_unit_min = 17001
integer(kind=iintegers), parameter :: lake_misc_unit_max = 18000

END MODULE INOUT_PARAMETERS

