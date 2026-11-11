MODULE SNOW_LAKE

contains

SUBROUTINE SNOW_CALC(ix,iy,t2,Erad_sc,ts,snow_sc,iyear,imonth, &
& iday,CCT,pmelt,pheat,dt)
use LB_POINT_CTX ! lakebatch multi-point: per-point saved locals

use LAKE_DATATYPES, only : ireals, iintegers

use PHYS_CONSTANTS, only : &
& g, Lwi, Lwv, Liv, &
& row0, roi, &
& csnow, rosn_dry_ref
use PHYS_PARAMETERS
use NUMERIC_PARAMS
use ATMOS
use DRIVING_PARAMS
use ARRAYS
use PHYS_FUNC, only : &
& EXTINCT_SNOW
use BL_MOD_LAKE
use SNOWSOIL_MOD

implicit none

!C CCCCCCCCCCCCC input parameters:
!C      e, W/m2, --- surface heat balance: e = Radiation-SIGMA*(TS**4)-Latent-Sensible-InSoil
!C      Elatent, W/m2, --- latent heat flux.
!C      ts, deg C, --- surface temperature.
!C      precip, m --- precipitation during dt.
!C
!C CCCCCCC  some other parameters: CCCCCCCCCCC
!C      densny, kg/m3       --- density of fresh snow, depends on temperature at 2m
!C      dznorm, m          --- initial thickness of snow layers
!C      dzmin, m           --- minimal thickness of snow layers
!C      dt, sec             --- timestep
!C      ms                  --- max. number of layers in snowpack
!C      cwat, J/(kg K)    --- the water specific heat content
!C      csnow, J/(kg K)   --- the snow specific heat content
!C      rhow, kg/m3        --- the liquid water density
!C      PLI,  J/kg         --- latent heat for freezing/melting
!C      PL,  J/kg          --- latent heat for evapor./condens.
!C      whc                 --- hydraulic constant
!C      hcond, m/sec       --- hydraulic conductivity
!C
!C CCCCCCCCC  values to be calculate after every timestep:
!C      wl(i), m       --- moisture content in snow layer No.i
!C      t(i), deg.C     --- mean temperature of snow layer No.i
!C      dz(i), m       --- hickness of snow layer No.i
!C      dens(i), kg/m3  --- mean density of snow layer No.i
!C      hsnow, m       --- snow depth
!C      swe, m         --- water equivalent snow depth
!C      snmelt, m/sec  --- mean snow melting speed during dt
!C CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
integer(kind=iintegers), intent(in) :: ix, iy, iyear, imonth, iday
integer(kind=iintegers) :: i, j, k

real(kind=ireals) dt
real(kind=ireals) Erad_sc,ts,snow_sc, &
& evap,erain,countrad,qbase,dmass,densev,counter,qin,rf, &
& smelti,ein,dzi,ziz,porosityOfSnow,w0,fukt,dztop,smelt,eout,p1,q0, &
& dzmasi,swe,densny,Erad1,eflux1,Phase,Ei,sigma,Cond,Elatent1
real(kind=ireals) EInRad,T2,rho_dry,CCT(ML),a2,eta,P,dens_old,ti(ml)
real(kind=ireals) pheat,pmelt

SAVE

! lakebatch multi-point: this point's copy of the saved locals
call lbc_begin(LBC_SCOPE_SNOW_CALC)
include 'lb_ctx_snow_calc.inc'
 
precip = precip*dt
Elatent1 = Elatent

!hsnow = 0.
!do i = itop, ms-1
!  hsnow = hsnow + dz(i)
!end do
!eFlux = eFlux - Erad

 eflux1 = eflux !/2 
 Erad1 =  Erad_sc !/2

!eflux1 = eflux1*(1.-exp(-0.02*snow))  ! not -0.02*snow*10.                  
 do i=itop,ms-1
!  wl(i) = wl(i)*dz(i)*dens(i)/row0    
   ti(i) = t(i)
 end do
 
 snmelt = 0.
 swe = 0.0
 Phase = 0.
! if(t2 .le. -10) then
!     densny = 0.05
! elseif(t2 .le. -2) then
!     densny = 5./800.*(t2+18.)
!     densny = 0.1
! else
!     densny = (t2+4.)/20.
! end if
 densny = (67.9 + 51.3 * exp(t2/2.6))
 if(ts .gt. 0.00001) then
     Elatent1 = 0
 elseif(snow_sc .lt. 0.000001 .and. Elatent .gt. 0.0) then    
     Elatent1 = 0
 end if

 dz(ms-1) = dz(ms-1) + dz(ms)
 evap = Elatent1/Liv 
!totalevap=totalevap+evap/row0*dt
 hsnow=0

!extinct(itop)=0
 do i = itop,ms-1
!  extinct(i) = (0.9 - 0.4/0.45*(dens(i) - 0.05))*100.
!  extinct(i) = exp(-(0.13*dens(i)+3.4))
   extinct(i) = EXTINCT_SNOW(dens(i))
!  extinct(i) = 0.
 end do

 if(itop.eq.ms) then

 else
!    --- first top snow layer ---
     i=itop
     dztop=dz(i)
     Erain = pheat                           
     if(extinct(itop).eq.0.0) then
         Eflux1 = Eflux1 + Erad1 + Erain                            
     else    
         if(itop.eq.ms-1) then
            Eflux1 = Eflux1 + Erain + Erad1            
         else
            Eflux1 = Eflux1 + Erain + Erad1*(1 - extinct(i)**dz(i))      
         end if
         countrad = Erad1*(1 - extinct(i)**dz(i))
     end if               
!rain   heat from rain is added as pheat
!rain   the rain water itself is assumed to freeze giving heat to top layer
!rain   then this heat will melt snow
     if(Eflux1.gt.0.0)then
         if(itop.eq.ms-1) then
             if(Ts.ge.-0.0001)then
               if(Eflux1/Lwi*dt .le. DENS(i)*DZ(i)-WL(i)*row0) then
                     smelt = Eflux1/Lwi/row0                          
                     Eout=0.
                 else
                     smelt = (DZ(i)*DENS(i)/row0 - WL(i))/dt
                     Eout = Eflux1 - smelt*row0*Lwi    
                 end if
             else
                 smelt = 0.0
                 Eout = 0.
             end if
         else
             smelt = 0.0
             Eout = Eflux1
         end if
     else
         smelt = 0.
         Eout = 0.
     end if

 
!
!PERCOLATION
!
     smelt = smelt + wl(i)/dt
     wl(i) = 0.
     qbase = smelt

!
!TOP  LAYER THICKNESS AND DENSITY
!
     dmass=dens(i)*dztop-(evap+smelt*row0)*dt     
     if(evap.gt.0.0)then
         densev=dens(i)
     else
         densev=roi
     endif
     dztop=dztop-(smelt*row0/dens(i)+evap/densev)*dt   
   if(dztop.eq.0.) dztop = 0.000001  
     dens(i)= max(dmass/dztop,0.d0)
     !if (dens(i) <= 0.) print*, 'dens!1', 'i=',i, dmass, dztop
     if(dens(i).gt.roi) then
         dztop = dztop*dens(i)/roi
         dens(i)=roi
         dmass=dens(i)*dztop
     endif
     dz(i)=dztop
     if(itop .eq.ms-1) snmelt=qbase
 end if 


 counter = dz(itop)
 if(itop.lt.ms-1) then
!
!  INTERNAL LAYERS
!
     do i=itop+1,ms-1
         !qin = qbase                                     
         rf = 0.0                                        
         smelti = 0.0                                    
         Ein=Eout                                        
         T(i) = T(i)*dz(i)*dens(i)/(qbase*row0*dt+dz(i)*dens(i))  
         dzi=dz(i)                                       
         wl(i) = wl(i) + qbase*dt
         if(wl(i).lt.0.) wl(i) = 0.
         w0=wl(i)
         if(extinct(i).eq.0.0) then
             Ei = Ein                
         else
             if(i.eq.ms-1) then
                 Ei=Ein + (Erad1 - countrad)
             else
                 Ei=Ein + Erad1*(extinct(i)**(counter) - &
                 & extinct(i)**(counter+dz(i)) )
                 EInRad = Erad1*(extinct(i)**(counter) - &
                 & extinct(i)**(counter+dz(i)) )
             end if                            
             countrad = countrad + Erad1*(extinct(i)**(counter) - &
             & extinct(i)**(counter+dz(i)) )
             counter = counter+dz(i)
         end if

         dz(i)=dz(i)+qbase*dt  
         ziz=dzi/dz(i)
         dens(i)=ziz*dens(i)+(1.0-ziz)*row0
         !if (dens(i) <= 0.) print*, 'dens!', 'i=',i, ziz, dzi, dz(i), qbase

       
         if(Ei.gt.0.0) then
             if(T(i).ge.0.0001) then  ! 0.0001
                 Ei = Ei + csnow*dens(i)*T(i)*dz(i)/dt
                 T(i) = 0.0
                 if(Ei/Lwi*dt .le. (DZ(i)*DENS(i)-WL(i)*row0)) then
                     smelti = Ei/Lwi/row0 
                     Eout = 0.
                 else
                     smelti = (DZ(i)*DENS(i)/row0 - WL(i))/dt
                     Eout = Ei - smelti*row0*Lwi 
                     dzi = 0.
                 end if
                 wl(i) = wl(i) + smelti*dt
             else
!    T<0
                 T(i) = T(i) + Ei*dt/(csnow*dens(i)*dz(i))
                 if(T(i).lt.-0.0001) then   !0.0001
                 if(wl(i) .gt. -T(i)*csnow*dz(i)*dens(i)/Lwi/row0) then
                       rf = -T(i)*csnow*dz(i)*dens(i)/Lwi/dt/row0 
                         T(i) = 0.0
                         wl(i) = wl(i) - rf*dt
                     else
                         rf = wl(i)/dt
                         T(i) = T(i) + rf*row0*dt*Lwi/(csnow*dz(i)*dens(i))    
                         wl(i) = 0.
                     end if
                     Eout = 0.
                 end if
                 if(T(i).ge.0.0001)then  !0.0001
                     if(csnow*dens(i)*T(i)*dz(i)/Lwi .le.dz(i)*dens(i)) then
                         smelti = csnow*dens(i)*T(i)*dz(i)/dt/Lwi/row0 
                         Eout = 0.
                     else
                         smelti = dz(i)*dens(i)/dt/row0  
                         Eout = csnow*dens(i)*T(i)*dz(i)/dt - smelti*Lwi*row0 
                         dzi = 0.
                     end if
                     T(i) = 0.0
                     wl(i) = wl(i) + smelti*dt
                 endif
             endif
         else

!E = 0 (E<0 is impossible case)
             if(T(i).ge.0.0001) then     !0.0001
                 if(csnow*dens(i)*T(i)*dz(i)/Lwi .le. dz(i)*dens(i)) then
                     smelti = csnow*dens(i)*T(i)*dz(i)/dt/Lwi/row0  
                     T(i) = 0.0
                     Eout = 0.
                     wl(i) = wl(i) + smelti*dt
                 else
                     smelti = dz(i)*dens(i)/dt/row0  
                     Eout = csnow*dens(i)*T(i)*dz(i)/dt - smelti*Lwi*row0  
                     T(i) = 0.0
                     dzi = 0.
                     wl(i) = wl(i) + smelti*dt
                 end if
             else
!T<0
             if(wl(i) .gt. -csnow*dens(i)*T(i)*dz(i)/Lwi/row0) then  
                     rf = -csnow*dens(i)*T(i)*dz(i)/dt/Lwi/row0   
                     T(i) = 0.0
                     wl(i) = wl(i) - rf*dt
                 else
                     rf = wl(i)/dt
                     wl(i) = 0.
                 T(i) = T(i) + rf*row0*dt*Lwi/(csnow*dz(i)*dens(i))
                 end if
                 Eout = 0.
                 if(T(i).ge.0.0001) then   !0.0001
                     if(csnow*dens(i)*T(i)*dz(i)/Lwi .le. dz(i)*dens(i)) then
                         smelti = csnow*dens(i)*T(i)*dz(i)/dt/Lwi/row0 
                         Eout = 0.0
                     else
                         smelti = dz(i)*dens(i)/dt/row0
                         Eout = csnow*dens(i)*T(i)*dz(i)/dt - smelti*Lwi*row0 
                         dzi = 0.
                     end if
                     T(i) = 0.0
                     wl(i) = wl(i) + smelti*dt
                 endif
             endif
         endif
 
!
!
!internal q
!
         rho_dry = (dens(i)-w0/(dz(i))*row0)/(1-w0/(dz(i)))
         if(rho_dry.lt.0.) then
           WRITE(*,*) 'rho_dry = ', rho_dry, i, dens(i), w0, dz(i)
           !converting excess water to ice
           !> @todo{add heat release}
           if (dens(i) - rosn_dry_ref > 0.) then
             wl(i) = dz(i)*(dens(i) - rosn_dry_ref)/(row0 - rosn_dry_ref)
             rho_dry = rosn_dry_ref
           else
             wl(i) = 0.
             rho_dry = dens(i)
           endif
           print*, 'ix=',ix,'iy=',iy,'wl(i)=',wl(i),'itop=',itop
         endif

!PorosityOfSnow = 1 - rho_dry/roi - wl(i)/dz(i)*(1-row0/roi) wl(i)/row0/dz(i)
        PorosityOfSnow = 1 - rho_dry/roi - wl(i)/dz(i)
         PorosityOfSnow = max(PorosityOfSnow,whc + 0.1)
         p1 = PorosityOfSnow - whc   ! whc = water holding capacity
         fukt = wl(i)/dz(i)
         if(fukt.gt.whc)then
             fukt = (fukt-whc)/p1
             q0 = hcond*fukt**3.0
 
             qbase = min(q0,wl(i)/dt)
             wl(i) = wl(i) - qbase*dt
         else
             qbase = 0.0
             q0 = 0.0
         end if
         if(dzi .le. 0.000001) then   
             qbase = qbase + wl(i)/dt
             dz(i)=dzi
         else
             dzmasi=dz(i)*dens(i)-qbase*dt*row0   
             dzi=dz(i)+row0*(-qbase/row0-rf/row0+rf/roi -  &
             & smelti/rho_dry+smelti/row0)*dt     
             dens(i)=max(dzmasi/dzi,0.d0)
             dz(i)=dzi
             if(dens(i).gt.roi)then
                 dz(i)=dz(i)*dens(i)/roi
                 dens(i)=roi
             end if
         end if
         Phase = Phase + (t(i)-ti(i))*dens(i)*dz(i)/snow_sc/dt
 
         if(i.eq.ms-1) snmelt=qbase
     end do
 end if
 if (itop==ms-1.and.dz(itop)<0.000001) then
   dz(itop)=0
   dens(itop)=0
   goto 123
 endif
 if (itop .lt. ms .and. dz(itop) .le. 0.000001) then 
   dz(itop)=0.
   dens(itop)=0.
   itop=itop+1
 end if
     
222   do i=ms-1,itop,-1
     if(dz(i).le.0.000001) then   
         do k=i-1,itop,-1
             if(dz(k).gt.0.000001) goto 111   
         end do
 
111           do j=i,itop,-1
             dz(j)=dz(k+j-i)
             dens(j)=dens(k+j-i)
             t(j)=t(k+j-i)
             wl(j)=wl(k+j-i)
         end do
         itop=itop-k+i
         goto 222
     end if
 end do
123   continue
       
333   do i=itop,ms-1,1
     if(dz(i).lt.dzmin.and.itop.lt.ms-1) then
     if(i.lt.ms-1) then
         dens(i+1)=dens(i+1)*dz(i+1)/(dz(i)+dz(i+1)) + &
         &         dens(i)*dz(i)/(dz(i)+dz(i+1))
         t(i+1)=t(i+1)*dz(i+1)/(dz(i)+dz(i+1)) + &
         &      t(i)*dz(i)/(dz(i)+dz(i+1))
         wl(i+1)=wl(i+1)+wl(i)
         dz(i+1)=dz(i+1)+dz(i)
         do j=i,itop+1,-1
             dens(j)=dens(j-1)
             t(j)=t(j-1)
             wl(j)=wl(j-1)
             dz(j)=dz(j-1)
         end do
         itop=itop + 1
         goto 333
     end if
     end if
 end do
 
 swe = 0.
 hsnow = 0.
 do j=ms-1,itop,-1
     swe = swe + dz(j)*dens(j)
     hsnow = hsnow + dz(j)
 end do
  
!ccccccccccccc snow densification due to gravity and metamorphism cccccccccccccc
 a2 = 0.00000066
 sigma = 75.                             ! metamorphic processes, Pa
 do j = itop+1,ms-1
     if(dens(j).lt.3*densny) then
         dens_old = dens(j)
         eta = &                          ! compactive viscosity of snow 
         &   a2*exp(19.3*dens(j)/roi) * &
         &   exp(67300/8.314/(t(j)+273.15))
         P = 0.0                         ! gravity, Pa
         do i = j,itop,-1
             !P = P + dens(i)/1000.*g*(dz(i))*10000.
             P = P + dens(i)*g*dz(i)
         end do
         dens(j) = dens(j) + 0.001*(dt*dens(j)*1000.*(sigma+P)/eta)
                     dens(j) = min(dens(j),roi)
         if (dens(j) <= 0.) then
           print*, j, itop+1,ms-1
           print*, 'dz', dz(:)
           print*, 'dens', dens(:)
           print*, 'T', t(:)
           print*, 'E, s', evap, smelt*row0
         endif
         dz(j) = dz(j) * dens_old/dens(j)
     end if
 end do
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
 swe = 0.
 hsnow = 0.
 do j=ms-1,itop,-1
     swe = swe + dz(j)*dens(j)
     hsnow = hsnow + dz(j)
 end do
!do m = itop,ms-1
!  por(m) = 1 - dens(m)/roi
!end do
 
 snow_sc = swe

   !! dz(ms) !!
 if(itop.ne.ms-1) then
     dz(ms) = dz(ms-1)/3.
     dz(ms-1) = dz(ms-1) - dz(ms)
 else
     dz(ms) = dz(ms-1)/2.
     dz(ms-1) = dz(ms-1) - dz(ms)
 end if
 if(itop.eq.ms-1) then
     z(ms-1) = z(ms) - dz(ms-1) - dz(ms)
 else
     z(ms-1) = z(ms) - dz(ms-1)/2. - dz(ms)
 end if
 DO j = MS-2,itop+1,-1
     Z(j) = z(j+1)-dz(j)/2.-dz(j+1)/2.
 END DO
 if(itop.lt.ms-1) z(itop) = z(itop+1) - dz(itop+1)/2. - dz(itop)

!do i=itop,ms-1
!  wl(i) = wl(i)/dz(i)/dens(i)*row0
!end do
 
 do i = 1, itop-1
     dz(i) = 0.
 end do
 
 precip = precip/dt
 !eflux = eflux+Erad
 
 return
 END SUBROUTINE SNOW_CALC


 SUBROUTINE addPrecipToSnow(T2,ts,snow,iyear,imonth, &
 & iday,CCT,pmelt,pheat,dt)
 
 use LAKE_DATATYPES, only : ireals, iintegers
 use PHYS_CONSTANTS, only : &
 & row0, roi, &
 & cw
 use PHYS_PARAMETERS
 use NUMERIC_PARAMS
 use DRIVING_PARAMS
 use ATMOS
 use ARRAYS_BATHYM, only : h1,l1,hs1,ls1
 use BL_MOD_LAKE
 use SNOWSOIL_MOD

 
 implicit none
      
!CCCCCCCCCCCCCCC input parameters:
!C      e, W/(m2, --- surface heat balance: e = Radiation-SIGMA*(TS**4)-Latent-Sensible-InSoil
!C      Elatent, W/m2, --- latent heat flux.
!C      ts, deg C, --- surface temperature.
!C      precip, m --- precipitation during dt.
!C
!CCCCCCCCC  some other parameters: CCCCCCCCCCC
!C      densny, kg/m3       --- density of fresh snow, depends on surface temperature
!C      dznorm, m          --- initial thickness of snow layers
!C      dzmin, m           --- minimal thickness of snow layers
!C      dt, sec             --- timestep
!C      ms                  --- max. number of layers in snowpack
!C      cwat, J/(kg K)    --- the water specific heat content
!C      csnow, J/(kg K)   --- the snow specific heat content
!C      rhow, kg/m3        --- the liquid water density
!C      PLI,  J/kg          --- latent heat for freezing/melting
!C      PL,  J/kg          --- latent heat for evapor./condens.
!C      whc                 --- hydraulic constant
!C      hcond, m/sec       --- hydraulic conductivity
!C
!CCCCCCCCCCC  values to be calculate after every timestep:
!C      wl(i), m       --- moisture content in snow layer No.i
!C      t(i), deg.C     --- mean temperature of snow layer No.i
!C      dz(i), m       --- thickness of snow layer No.i
!C      dens(i), kg/m3  --- mean density of snow layer No.i
!C      hsnow, m       --- snow depth
!C      swe, m         --- water equivalent snow depth
!C      snmelt, m/sec  --- mean snow melting speed during dt
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

 real(kind=ireals) ts,snow,dz0,pmelt,pheat, &
 & z0z,dmass, swe,densny
 integer(kind=iintegers) j,i,k,mn,iyear,imonth,iday
 real(kind=ireals) T2,CCT(ML)
 real(kind=ireals) dt 

 SAVE
 
 precip = precip*dt
     
       
!if (snow > 0.) then
!  write (*, *) snow
!  STOP
!end if
!T2 = T2 - 273.15

! if(T2 .le. -10) then
!     densny = 0.05
! elseif(ts .le. -2) then
!     densny = 5./800.*(t2+18.)
!    densny = 0.1
! else
!     densny = (t2+4.)/20.
! end if

 densny = (67.9 + 51.3 * exp(t2/2.6))
 if(itop.lt.ms)  dz(ms-1) = dz(ms-1) + dz(ms) 
 
 snow = 0.
 do mn = itop,ms-1
     snow = snow + dz(mn)*dens(mn)
 end do
 
 
 j=itop
555   continue
 if(dz(itop).lt.dznorm+dzmin) goto 444
 dz(j-1) = dz(j) - dznorm
 dz(j) = dznorm
 z(j-1)= z(itop) 
 z(j)= z(j-1) + dz(j-1)*2.
 wl(j-1)=0.0
 dens(j-1)=dens(j)
 t(j-1)=t(j)
 j=j-1
 itop=j
 goto 555
444   continue
      
 if(T2.gt.0.) then
     pmelt = precip/dt                       
!    pmelt = 0.                       
     pheat = T2*precip*cw*row0/dt      
 else       
     pmelt=0.0
     pheat=0.0
     if(itop.eq.ms .and. precip.gt.0.0) then
         itop=itop-1
         z(itop) = z(itop+1)
         t(itop) = t(itop+1)
         dens(itop) = densny
         wl(itop) = 0
     end if
     if(itop.ne.ms) then
         dz0 = dz(itop)
         z(itop) = z(itop) - precip*row0/densny   
         dz(itop) = dz(itop) + precip*row0/densny   
         if(dz(itop) .lt. dznorm+dzmin) then
             z0z=dz0/dz(itop)
             dens(itop)=z0z*dens(itop)+(1.0-z0z)*densny
         end if
         j=itop
55       continue
         if(dz(itop) .lt. dznorm+dzmin) goto 54
         if (j == 1) print*, 'No enough layers to represent the snow cover: &
         & please, check precipitation data'
         dz(j-1) = dz(j) - dznorm
         dz(j) = dznorm
         z(j-1)= z(itop) 
         z(j)= z(itop) - dz(itop)*0.5
         z0z=dz0/(dz(itop)+dz(itop-1))
         dens(itop)=z0z*dens(itop)+(1.0-z0z)*densny
         wl(j-1)=0.0
         dz0 = 0.0
         dens(j-1)=densny
         t(j-1)=t(j)
         j=j-1
         itop=j
         goto 55
54       continue
     end if
 end if
     
 if(pheat.gt.0.0) then
     if(itop.eq.ms-1) then
         dmass = dens(itop)*dz(itop) + precip*row0    
         dz(itop) = dz(itop) + precip*row0/roi   
         if(dz(itop).eq.0.) dz(itop) = 0.000001   
         dens(itop)= max(dmass/dz(itop),0.d0)
         if(dens(itop).gt.roi) then
             dz(itop) = dz(itop)*dens(itop)/roi
             dens(itop)=roi
         end if
     else
         if(itop.lt.ms-1) then
             mn = itop+1
             dmass = dens(mn)*dz(mn) + precip*row0  
             dz(mn) = dz(mn) + precip*row0/roi 
             if(dz(mn).eq.0.) dz(mn) = 0.000001    
             dens(mn)= max(dmass/dz(mn),0.d0)
             if(dens(mn).gt.roi) then
                 dz(mn) = dz(mn)*dens(mn)/roi
                 dens(mn)=roi
             end if
             wl(mn) = wl(mn) + precip
             T(mn) = T(mn)*dz(mn)*dens(mn)/(precip*row0+dz(mn)*dens(mn)) 
         end if
     end if
 end if

   !!  dz(ms) !!!
 if(itop.ne.ms-1) then
     dz(ms) = dz(ms-1)/3.
     dz(ms-1) = dz(ms-1) - dz(ms)
 else
     dz(ms) = dz(ms-1)/2.
     dz(ms-1) = dz(ms-1) - dz(ms)
 end if
 if(itop.eq.ms-1) then
     z(ms-1) = z(ms) - dz(ms-1) - dz(ms)
 else 
     z(ms-1) = z(ms) - dz(ms-1)/2. - dz(ms)
 end if
 DO Mn = MS-2,itop+1,-1
     Z(Mn) = z(mn+1)-dz(mn)/2.-dz(mn+1)/2.
 END DO
 if(itop.lt.ms-1) z(itop) = z(itop+1) - dz(itop+1)/2. - dz(itop)

 !if (dz(itop).lt.dznorm+dzmin) goto 777 
 
 swe = 0.0
 do mn=ms-1,itop,-1
     swe=swe+dz(mn)*dens(mn)
 end do
 swe = swe + dz(ms)*dens(ms-1)
 snow = swe
 
 hsnow = 0.
 do i = itop, ms
   hsnow = hsnow + dz(i)
 end do
        
 precip = precip/dt

 if (itop < 0) then
   print*, 'addPrecipToSnow: itop<0', 'dz=',dz,'dens=',dens,&
   & 'wl=',wl,'T=',T,'swe=',swe
 endif
 
 !itop = max(itop,1)

 return
 END SUBROUTINE addPrecipToSnow

END MODULE SNOW_LAKE
