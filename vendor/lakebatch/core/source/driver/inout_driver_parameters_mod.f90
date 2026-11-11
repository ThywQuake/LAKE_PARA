 MODULE INOUT_DRIVER_PARAMETERS

 use DRIVER_DATATYPES, only : ireals, iintegers
 
 integer(kind=iintegers), parameter :: driver_subr_unit_min = 10001
 integer(kind=iintegers), parameter :: driver_subr_unit_max = 10500
 integer(kind=iintegers), parameter :: driver_file_unit_min = 10501
 integer(kind=iintegers), parameter :: driver_file_unit_max = 11000
 
 END MODULE INOUT_DRIVER_PARAMETERS

