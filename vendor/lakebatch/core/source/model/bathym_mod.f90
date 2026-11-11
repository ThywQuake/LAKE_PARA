MODULE BATHYMSUBR

use GRID_OPERATIONS_LAKE, only : LININTERPOL, PIECEWISECONSTINT

contains
!> Subroutine BATHYMETRY calculates lake bathymetry characteristics
SUBROUTINE BATHYMETRY(M,Mice,ns,ix,iy,ndatamax,month,day,lakeform,hour,dt, &
                     & depth_area,area_lake,cellipt, &
                     & multsoil,trib_inflow,dhwtrib,vol,botar)

! Subroutine updates bathymetry variables

use LAKE_DATATYPES, only : &
& ireals, iintegers

use DRIVING_PARAMS, only : soilcolconjtype, nvartribext

use PHYS_CONSTANTS, only : g, row0

use ARRAYS, only : init

use ARRAYS_WATERSTATE, only: preswat

use ARRAYS_BATHYM, only : &
& area_int, area_half, &
& bathymwater, bathymice, &
& bathymdice, bathymsoil, &
& h1,l1,ls1,Lx,Ly, &
& form_ellipse, form_rectangle

use ARRAYS_GRID, only : &
& ddzi, nsoilcols, zsoilcols, &
& isoilcolc, ksoilcol, &
& z_full, z_half, &
& z_full_ice, z_half_ice, &
& dzeta_int, dzeta_05int, ddz05, &
& dzetai_int, dzetai_05int

use ARRAYS_SOIL, only : dzs

use NUMERIC_PARAMS, only : &
& pi, small_value

use ATMOS, only : pressure

use TIMEVAR, only : hour_sec

use TRIBUTARIES, only : ABSTR

implicit none

! Input variables
integer(kind=iintegers), intent(in) :: M, Mice, ns
integer(kind=iintegers), intent(in) :: ix, iy
integer(kind=iintegers), intent(in) :: ndatamax
integer(kind=iintegers), intent(in) :: month, day
integer(kind=iintegers), intent(in) :: lakeform

real(kind=ireals),       intent(in) :: hour

real(kind=ireals),       intent(in) :: dt

real(kind=ireals),       intent(in) :: depth_area(1:ndatamax,1:2)
real(kind=ireals),       intent(in) :: area_lake, cellipt

logical, intent(in) :: multsoil

real(kind=ireals), intent(in) :: trib_inflow(1:nvartribext)
real(kind=ireals), intent(inout) :: dhwtrib

real(kind=ireals), intent(out) :: vol, botar

! Local variables
real(kind=ireals), allocatable :: work1(:), work2(:), work3(:), work4(:), work5(:), work6(:)

integer(kind=iintegers) :: i, j

real(kind=ireals) :: aellipt, bellipt, z, dz, dadz, da

real(kind=ireals) :: watabstr = 0. ! Water abstraction, currently implemented for single lake 

logical, save :: water_abstraction = .false.
logical :: flag

!real(kind=ireals), external :: ABSTR

do i = 1, M+1
  z_full(i) = dzeta_int(i)*h1
  preswat(i) = pressure + row0*g*z_full(i)
enddo
do i = 1, M    
  z_half(i) = dzeta_05int(i)*h1
enddo  
forall (i = 1:Mice+1) z_full_ice(i) = dzetai_int  (i)*l1
forall (i = 1:Mice  ) z_half_ice(i) = dzetai_05int(i)*l1

! Constant cross-section
if (depth_area(1,2) < 0.) then ! No data on morphometry
    area_int            (1:M+1)    = area_lake
    area_half           (1:M)      = area_lake
    bathymice (1:Mice+1)%area_int  = area_lake
    bathymice (1:Mice)  %area_half = area_lake
    bathymdice(1:Mice+1)%area_int  = area_lake
    bathymdice(1:Mice)  %area_half = area_lake
    !if (init(ix,iy) == 0 .and. multsoil) then
    !  bathymsoil(1:nsoilcols+1,ix,iy)%area_int  = area_lake
    !  bathymsoil(1:nsoilcols,  ix,iy)%area_half = area_lake     
    !endif
else
  ! Multiple soil columns bathymetry (constant in time)
  if_multsoil : if (multsoil) then
    allocate(work3(1:nsoilcols) )
    do i = 1, nsoilcols
      work3(i) = 0.5*( zsoilcols(i,ix,iy) + zsoilcols(i+1,ix,iy) )
    enddo
    if (init(ix,iy) == 0) then
      allocate(work5(1:nsoilcols+1), work6(1:nsoilcols))
      call LININTERPOL (depth_area(1,1),depth_area(1,2),ndatamax, &
                       & zsoilcols(1,ix,iy),work5,nsoilcols+1,flag,.true.)
      call LININTERPOL (depth_area(1,1),depth_area(1,2),ndatamax, &
                       & work3,work6,nsoilcols,flag,.true.) 
      bathymsoil(1:nsoilcols+1,ix,iy)%area_int  = work5(1:nsoilcols+1)
      bathymsoil(1:nsoilcols,  ix,iy)%area_half = work6(1:nsoilcols)
      deallocate (work5, work6)
      !! Horizontal cross-section area of soil columns at soil numerical grid levels
      !do i = 1, nsoilcols-1
      !  bathymsoil(i,ix,iy)%sarea_int(1) = small_value
      !  z = 0.; j = 1
      !  dz = zsoilcols(i+1,ix,iy) - zsoilcols(i,ix,iy)
      !  da = bathymsoil(i,ix,iy)%area_int - bathymsoil(i+1,ix,iy)%area_int
      !  dadz = da / dz 
      !  do while (z < dz)
      !    j = j + 1
      !    z = z + dzs(j-1)
      !    bathymsoil(i,ix,iy)%sarea_int(j) = bathymsoil(i,ix,iy)%sarea_int(j-1) + dadz*dzs(j-1)
      !    bathymsoil(i,ix,iy)%sarea_half(j-1) = &
      !    & 0.5*(bathymsoil(i,ix,iy)%sarea_int(j) + bathymsoil(i,ix,iy)%sarea_int(j-1))
      !  enddo
      !  bathymsoil(i,ix,iy)%sarea_int(j)    = min(bathymsoil(i,ix,iy)%sarea_int(j)   ,da)
      !  bathymsoil(i,ix,iy)%sarea_half(j-1) = min(bathymsoil(i,ix,iy)%sarea_half(j-1),da)
      !  bathymsoil(i,ix,iy)%sarea_int(j+1:ns)  = bathymsoil(i,ix,iy)%sarea_int(j)
      !  bathymsoil(i,ix,iy)%sarea_half(j:ns-1) = bathymsoil(i,ix,iy)%sarea_half(j-1)
      !enddo
      !!Lowest soil column
      !i = nsoilcols
      !bathymsoil(i,ix,iy)%sarea_int(1:ns)    = bathymsoil(nsoilcols,ix,iy)%area_int
      !bathymsoil(i,ix,iy)%sarea_half(1:ns-1) = bathymsoil(nsoilcols,ix,iy)%area_int
    endif
    dz = - maxval(depth_area(:,1)) + h1 + ls1 ! Relating coordinate systems before interpolation
    do i = 2, nsoilcols
      do j = M, 1, -1
        if (zsoilcols(i,ix,iy) + dz >  z_full(j) .and. &
            zsoilcols(i,ix,iy) + dz <= z_full(j+1)) then
          bathymsoil(i,ix,iy)%itop = j
          bathymsoil(i,ix,iy)%dzSL = work3(i) + dz - z_full(j)
        endif
      enddo
    enddo
    bathymsoil(1,ix,iy)%itop = 1
    bathymsoil(1,ix,iy)%dzSL = work3(1) + dz - z_full(1)
    do i = 2, nsoilcols
      do j = 1, M
        if (zsoilcols(i,ix,iy) + dz >  z_full(j) .and. &
            zsoilcols(i,ix,iy) + dz <= z_full(j+1)) then
          bathymsoil(i-1,ix,iy)%ibot = j
        endif
      enddo
    enddo
    bathymsoil(nsoilcols,ix,iy)%ibot = M+1
    bathymsoil(:,ix,iy)%icent = 1
    do i = 1, nsoilcols-1
      do j = 1, M-1
        if (work3(i) + dz >  z_half(j) .and. &
            work3(i) + dz <= z_half(j+1)) then
          bathymsoil(i,ix,iy)%icent = j+1
          bathymsoil(i,ix,iy)%dzSLc = work3(i) + dz - z_half(j)
        endif
      enddo
    enddo
    deallocate(work3)
  else
    !i = nsoilcols
    !bathymsoil(i,ix,iy)%sarea_int(1:ns)    = area_lake
    !bathymsoil(i,ix,iy)%sarea_half(1:ns-1) = area_lake
  endif if_multsoil


  ! Interpolation of the predefined area-depth dependence to the current grid...
  ! ...in water layer
  if_waterexist : if (h1 > small_value) then
    if     (soilcolconjtype == 1) then
      allocate(work1(1:nsoilcols+1),work2(1:nsoilcols+1))
      work2(:) = bathymsoil(:,ix,iy)%area_int
      work1(:) = zsoilcols(:,ix,iy) - maxval(zsoilcols(:,ix,iy)) + &
      & h1 + ls1 ! Relating coordinate systems before interpolation
      call PIECEWISECONSTINT (work1,work2,nsoilcols+1,z_full,area_int,M+1,1_iintegers)
      call PIECEWISECONSTINT (work1,work2,nsoilcols+1,z_half,area_half,M,1_iintegers)
      area_int(M+1) = bathymsoil(nsoilcols+1,ix,iy)%area_int
    elseif (soilcolconjtype == 2) then
      allocate(work1(1:ndatamax),work2(1:ndatamax))
      work2(:) = depth_area(:,2)
      work1(:) = depth_area(:,1) - maxval(depth_area(:,1)) + &
      & h1 + ls1 ! Relating coordinate systems before interpolation
      call LININTERPOL (work1,work2,ndatamax,z_full,area_int,M+1,flag,.true.)
      call LININTERPOL (work1,work2,ndatamax,z_half,area_half,M,flag,.true.)
    endif
  else
    print*, 'Warining in BATHYM: water does not exist'
  endif if_waterexist
  !... in ice layer
  if (l1 > small_value) then
    allocate (work3(1:Mice+1), work4(1:Mice))
    work3(1) = 0.
    do i = 2, Mice+1
      work3(i) = work3(i-1) + ddzi(i-1)*l1
    enddo
    work4(1) = 0.5*ddzi(1)*l1
    do i = 2, Mice
      work4(i) = work4(i-1) + 0.5*(ddzi(i-1) + ddzi(i))*l1
    enddo
    allocate(work5(1:Mice+1), work6(1:Mice))
    if     (soilcolconjtype == 1) then
      work1(:) = zsoilcols(:,ix,iy) - maxval(zsoilcols(:,ix,iy)) + h1 + l1 + ls1 
      call PIECEWISECONSTINT (work1,work2,nsoilcols,work3,work5,Mice+1,1_iintegers)
      call PIECEWISECONSTINT (work1,work2,nsoilcols,work4,work6,Mice,1_iintegers)
    elseif (soilcolconjtype == 2) then
      work1(:) = depth_area(:,1) - maxval(depth_area(:,1)) + &
      & h1 + l1 + ls1 
      call LININTERPOL (work1,work2,ndatamax,work3,work5,Mice+1,flag,.true.)
      call LININTERPOL (work1,work2,ndatamax,work4,work6,Mice,flag,.true.) 
    endif
    bathymice(1:Mice+1)%area_int  = work5(1:Mice+1)
    bathymice(1:Mice)  %area_half = work6(1:Mice)
    deallocate(work3, work4, work5, work6)
  endif
  ! ... in deep ice layer
  if (ls1 > small_value) then
    allocate (work3(1:Mice+1), work4(1:Mice))
    work3(1) = 0.
    do i = 2, Mice+1
      work3(i) = work3(i-1) + ddzi(i-1)*ls1
    enddo
    work4(1) = 0.5*ddzi(1)*ls1
    do i = 2, Mice
      work4(i) = work4(i-1) + 0.5*(ddzi(i-1) + ddzi(i))*ls1
    enddo
    allocate(work5(1:Mice+1), work6(1:Mice))
    if     (soilcolconjtype == 1) then
      work1(:) = zsoilcols(:,ix,iy) - maxval(zsoilcols(:,ix,iy)) + ls1 
      call PIECEWISECONSTINT (work1,work2,nsoilcols,work3,work5,Mice+1,1_iintegers)
      call PIECEWISECONSTINT (work1,work2,nsoilcols,work4,work6,Mice,1_iintegers)
      work5(Mice+1) = bathymsoil(nsoilcols+1,ix,iy)%area_int
    elseif (soilcolconjtype == 2) then
      work1(:) = depth_area(:,1) - maxval(depth_area(:,1)) + ls1 
      call LININTERPOL (work1,work2,ndatamax,work3,work5,Mice+1,flag,.true.)
      call LININTERPOL (work1,work2,ndatamax,work4,work6,Mice,flag,.true.) 
    endif
    bathymdice(1:Mice+1)%area_int  = work5(1:Mice+1)
    bathymdice(1:Mice)  %area_half = work6(1:Mice)
    deallocate(work3, work4, work5, work6)
  endif

  if (allocated(work1)) deallocate (work1)
  if (allocated(work2)) deallocate (work2)
endif


! Cross-section parameters for water
if (h1 > small_value) then
  do i = 1, M+1
    ! Assuming horizontal cross-section to be an ellipse at every level
    call LXLY(area_int(i),Lx(i),Ly(i))
  enddo
  do i = 1, M
    call LXLY(area_half(i),bathymwater(i)%Lx_half,bathymwater(i)%Ly_half)
  enddo
  bathymwater(1:M+1)%area_int  = area_int (1:M+1)
  bathymwater(1:M  )%area_half = area_half(1:M  )
  bathymwater(1:M+1)%Lx        = Lx       (1:M+1)
  bathymwater(1:M+1)%Ly        = Ly       (1:M+1)
  bathymwater(1:M+1)%rad_int  =  sqrt(area_int (1:M+1)/pi)
endif
! Cross-section parameters for ice
if (l1 > small_value) then
  do i = 1, Mice+1
    ! Assuming horizontal cross-section to be an ellipse at every level
    call LXLY(bathymice(i)%area_int,bathymice(i)%Lx,bathymice(i)%Ly)
  enddo
  do i = 1, Mice
    call LXLY(bathymice(i)%area_half,bathymice(i)%Lx_half,bathymice(i)%Ly_half)
  enddo
endif
! Cross-section parameters for deep ice
if (ls1 > small_value) then
  do i = 1, Mice+1
    call LXLY(bathymdice(i)%area_int,bathymdice(i)%Lx,bathymdice(i)%Ly)
  enddo
  do i = 1, Mice
    call LXLY(bathymdice(i)%area_half,bathymdice(i)%Lx_half,bathymdice(i)%Ly_half)
  enddo
endif
! Cross-section parameters for multiple soil columns
if (init(ix,iy) == 0 .and. multsoil) then
  ! Assuming horizontal cross-section to be an ellipse at every level
  do i = 1, nsoilcols+1
    call LXLY(bathymsoil(i,ix,iy)%area_int,bathymsoil(i,ix,iy)%Lx,bathymsoil(i,ix,iy)%Ly)
  enddo
endif

if (h1 > small_value .and. multsoil) then
  allocate(work1(1:nsoilcols+1))
  work1(:) = zsoilcols(:,ix,iy) - maxval(zsoilcols(:,ix,iy)) + &
  & h1 + ls1 ! Relating coordinate systems before interpolation
  ksoilcol(:) = nsoilcols
  do i = 1, nsoilcols-1
    z = 0.5*(work1(i) + work1(i+1))
    do j = 1, M
      if (z_full(j) <= z .and. z_full(j+1) > z) then
        isoilcolc(i) = j
        exit
      endif
    enddo
    do j = 1, M
      if (z_full(j) >= work1(i)   .and. &
        & z_full(j) <  work1(i+1)) then
        ksoilcol(j) = i
      endif
    enddo
  enddo
  deallocate(work1)
endif


! Lake volume and bottom area
botar = area_int(M+1)
do i = M, 1, -1
  !vol = vol + 0.5*(area_int(i+1) + area_int(i))* &
  !& (z_full(i+1) - z_full(i))
  botar = botar + area_int(i) - area_int(i+1)
enddo
vol = 0.
do i = 1, M+1
  vol = vol + area_int(i)*h1*ddz05(i-1)
enddo

!TRIBUTARY INFLOW to a lake
if (water_abstraction .and. month == 12 .and. day == 31 &
& .and. (24. - hour)*hour_sec <= dt) then
! Water abstraction for the next year, m**3/s
  watabstr = ABSTR(vol)
  if (trib_inflow(1) /= -9999.) then
    dhwtrib = dhwtrib - watabstr*dt/area_int(1)
  endif
endif

contains

!> Subroutine LXLY calculates the length and width of the lake
!! given its cross-section area and form
SUBROUTINE LXLY(area,Lx_,Ly_)

implicit none

real(kind=ireals), intent(in) :: area
real(kind=ireals), intent(out) :: Lx_, Ly_

if (lakeform == form_ellipse) then
  bellipt = sqrt(area/(pi*cellipt))
  aellipt = bellipt*cellipt
  Lx_ = 2.*aellipt
  Ly_ = 2.*bellipt
elseif (lakeform == form_rectangle) then
  Ly_ = sqrt(area/cellipt)
  Lx_ = cellipt*Ly_
else
  print*, 'Incorrect LAKEFORM identifier:', lakeform, ', STOP'
  STOP
endif

END SUBROUTINE LXLY

END SUBROUTINE BATHYMETRY

END MODULE BATHYMSUBR
