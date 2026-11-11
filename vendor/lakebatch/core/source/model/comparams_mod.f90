MODULE COMPARAMS

use LAKE_DATATYPES, only : ireals, iintegers

integer(kind=iintegers) :: num_ompthr

type, public :: parallel_params
  sequence
  integer(kind=iintegers) :: comm3d, rank_comm3d, coords(1:3) 
  logical :: parallel
endtype parallel_params
type(parallel_params) :: parparams

END MODULE COMPARAMS

