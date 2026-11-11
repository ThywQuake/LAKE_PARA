MODULE LAKE_DATATYPES

! Data types for variables

integer, parameter :: iintegers = 4
integer, parameter :: ireals = 8

!Compilation options
#if FCOMPILER==1
integer, parameter :: ibyterecl = 1 !> the length of one record in binary files: 1 for ifort
#elif FCOMPILER==2
integer, parameter :: ibyterecl = 4 !> the length of one record in binary files: 4 for gfortran
#endif

END MODULE LAKE_DATATYPES

