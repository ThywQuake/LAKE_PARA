MODULE RICHPBL_LAKE_MOD

contains
      SUBROUTINE RichPBL(h,le,tau,Tsurf1,humsurf,z0)
      
!     RichPBL calculates heat fluxes according to Louis parameterization (Louis, 1079)    
      use LAKE_DATATYPES, only : ireals, iintegers

      use ATMOS, only: &
      & tempair, &
      & humair, &
      & pressure, &
      & zref, &
      & urel, vrel
      implicit none
      
      real(kind=ireals),parameter:: gg = 9.80665
      real(kind=ireals),parameter:: ra = 287.05
      real(kind=ireals),parameter:: rv = 461.51
      real(kind=ireals),parameter:: cp = 1005.46
      real(kind=ireals),parameter:: cpv = 1869.46
      real(kind=ireals),parameter:: xl = 2500800.
      real(kind=ireals),parameter:: stefan = 5.6697d-08
      real(kind=ireals),parameter:: xpi = 3.141592654
      real(kind=ireals),parameter:: too = 273.16
      real(kind=ireals),parameter:: xpoo = 100000.
      real(kind=ireals),parameter:: vkarmn = .4
      real(kind=ireals),parameter:: xli = 2834500.
      real(kind=ireals),parameter:: tcdil = 86400.
      real(kind=ireals),parameter:: rascp = ra / cp
      real(kind=ireals),parameter:: xlsg = xl / gg
      real(kind=ireals),parameter:: xlscp = xl / cp
      real(kind=ireals),parameter:: cpsl = cp / xl
      real(kind=ireals),parameter:: cpsg = cp / gg
      real(kind=ireals),parameter:: gscp = gg / cp
      real(kind=ireals),parameter:: unsg = 1. / gg
      real(kind=ireals),parameter:: gsra = gg / ra
      real(kind=ireals),parameter:: rasl = ra / xl
      real(kind=ireals),parameter:: rascp2 = ra / (2.*cp)
      real(kind=ireals),parameter:: rasrv = ra / rv
      real(kind=ireals),parameter:: rvsra = rv / ra
      real(kind=ireals),parameter:: etv = rvsra - 1.
      real(kind=ireals),parameter:: ecph = cpv / cp - 1.
      real(kind=ireals),parameter:: xlsrv = xl / rv
      real(kind=ireals),parameter:: unscp = 1. / cp
      real(kind=ireals),parameter:: cpvmcp = cpv - cp
      real(kind=ireals),parameter:: etvq = 1. - rasrv
      real(kind=ireals),parameter:: xlf = xli - xl
      real(kind=ireals),parameter:: xliscp = xli / cp
      real(kind=ireals),parameter:: xlisg = xli / gg

      real(kind=ireals) z0hz0,Tsurf1,humsurf,z,xmu,xfh,ts,h,le,leg, &
     & ta,qa,ps,ua,z0h,z0,hu,veg,ztvi,rhoa,zdepl,zua,ztvis,Ri,&
     & valnon,zeps,zch,zsta,iyra,zdi,rra,zdim,cdm,zds,qs, &
     & cdh,zdsm,xhu,tau

      SAVE
      
      ta=tempair+273.15
      qa=humair
      ps=pressure
      ua=sqrt(urel**2+vrel**2)
      ts=Tsurf1+273.15
      qs=humsurf

      z0hz0=0.1
      z0h=z0*z0hz0

!
!**********************************************************************
!           delta calculatlon
!
!      delta=0.
!      if (veg.gt.0.) delta=(wr(li)/wrmax)**(2./3.)
!
!**********************************************************************
!                         hv calculation
!                         ______________
!
      ! hv = 1. - max(0.d0,dsign(1.d0,qsat(ts(li), ps) - qa)) * rs *
      !&      (1. - delta) / (rra + rs )

        hu=1. !air above water is saturated
        veg=0. !no vegetation on the lake
!
!?       rra a resistencia aerodinamica = 1/ChVa  -volta
!        a ser calculada frente (correctamente

!        Implicit resolution of the surface temperature equation
!        linearization of Ts

        ztvi = ta * (1. + etv * qa)
        rhoa = ps / (ra * ztvi)
        !qs = (hu * (1. - veg) + veg * hv) * qsat (ts(li),ps) +
     &  !     (1. - hv) * veg * qa
!
!**********************************************************************
!                CMWF, 1981/11/25-27, pp. 59-79)
!**********************************************************************
!
        zdepl = 0.
        !write(*,*) 'nhli9904zref',zref
        zua=zref + zdepl
        ztvis = ts * (1. + etv * qs)
!______________________________________________________________________
!       calculo do numero de richardson e da resistencia aerodinamica
!
      !write(*,*) 'zref',zref
      ua=max(1.d0,ua)
      ri=2.*gg*zua*zua*(ztvi-ztvis+gscp*zref)/(ztvi+ztvis)/(ua*ua)/zref
!
!       calculo do comprimento de Monin -Obukhov,de ustar,tstar
!
        valnon=-999.
!        call calclmon(z,zO,zOh,ua,ztvi,ztvls,xlmon,ustar,tstar,valnon)
        zeps= 1.e-6
        !write(* ,*) 'zua,z0',zua,z0
        zch=(vkarmn/log(abs(zua/z0)))**2
        !write(*,*) 'zch',zch
        zsta=ri*ua*ua
!
!  iyra escolhe entre a nova e a velha versao da rotina yra
!         de NP89. Quando iyra (ficheiro de entrada(?!)) e maior
!         do que 0 utiliza-se a nova versao que destingue entre
!         os comprimentos de rugosidade
!
        iyra=1
        if(iyra.eq.0) then
!
          if(ri.lt.0.) then
           zdi=1./(ua+75.*zch*sqrt(-zref*zsta/z0h))
           rra=zch*(ua-15.*zsta*zdi)
           zdim=1./(1.+75.*zch*sqrt(-zref*ri/z0))
           cdm=zch*(1.-10.*ri*zdim)*ua
          else
           zds=sqrt(ua * ua + 5. * zsta + zeps)
           rra = zch*ua/(1.+15.*zsta*zds/ua/ua/ua)
           zdim=(1.+5.*ri)**2.
           cdm=zch*ua/zdim
          endif
!
        else
!
! nova rotina yramng
          ! write(*,*) 'estez0h',z0h
          !write(*,*) 'z,zref',z,zref
           z=zref
           xmu=log(z0/z0h)
           xfh=log(z/z0)/log(z/z0h)
          ! write(*,*) '--------------------'
          ! write(*,*) 'z0,z0h',z0,z0h
           if(ri.lt.0.) then
                zdi=1./(ua+chstar2(xmu)*zch*15.*(z/z0h)** &
     &          ph2(xmu)*xfh &
     &              *sqrt(-zsta))
                rra=zch*(ua-15.*zsta*zdi)*xfh
!
! alteracoes para o calculo do coeficiente de dragg_______________
!        cdm=Cd*uvsuf
!
                cdh=rra
                zdim=1./(ua+cmstar2(xmu)*zch*10.*(z/z0)** &
     &          pm2(xmu) &
     &              *sqrt(-zsta))
                cdm=zch*(ua-10.*zsta*zdim)
!_______________________________________________________________________
           else
                zds = sqrt(ua * ua + 5. * zsta + zeps)
                rra = zch*ua/(1.+15.*zsta*zds/ua/ua/ua)*xfh
!
                cdh=rra
                zdsm= sqrt(ua * ua + 5. * zsta + zeps)
                cdm = zch*ua/(1.+10.*zsta/zdsm/ua)
      
           endif
!
        endif
!
        rra=1./rra
!----------------------------------------------------------------------
!
!      zrsra = rhoa / rra
!      if (iclay.ge.0) then
!         ct = 1. / ((1. - veg) / cso + veg / cv)
!         hv=1.-max(0.d0,dsign(1.d0,qsat(ts(li), ps) - qa)) * rs*
!     &    (1. - delta) / (rra + rs)
!
!         za=1. / dt + ct * (4. * emis * stefan * (ts(li)**3) +
!     &      zrsra*xl*dqsat(ts(li),ps)*(veg*hv+(1.-veg)*hu*xhu)
!     &      +zrsra*cp)+2.*xpi/tcdil
!         zb=1./dt+ ct*(3. * emis * stefan * (ts(li)** 3) +
!     &      zrsra*xl*dqsat(ts(li),ps)*(veg*hv+(1.-veg)*hu*xhu))
!         zc=2.*xpi*t2(li)/tcdil+ct*(zrsra*cp*ta+rg*
!     &      (1.-alb)+emis*rat-zrsra*xl*(veg*hv*(qsat(ts(li),
!     &      ps)-qa)+(1.-veg)*xhu*(hu*qsat(ts(li),ps)-qa)))
!
!         ts(lf) = (ts(li) * zb + zc) / za
!
!       resolution of t2 equation
!
!         t2(lf) = (t2(li) + dt * ts(lf) / tcdil) / (1. + dt / tcdil)
!      else
!          ts(lf)=tsfunc
!          t2(lf)=t2(li)
!      endif
!
!**********************************************************************
!
      xhu=1.
      !hqs2=hu*qsat(ts(lf),ps)
      !rn = rg * (1. - alb) + emis * (rat - stefan * (ts(lf)**4))
      h = rhoa * cp * (ts - ta) / rra
      leg = rhoa * xl*(1.-veg) *xhu*(qs - qa)/ rra 
      !lev = rhoa * xl * veg * hv * (qsat(ts(lf), ps) - qa) / rra
      !zzhv = max(0.d0,dsign(1.d0,qsat(ts(li), ps) - qa))
      !letr = zzhv * (1. - delta) * rhoa * xl * veg * (qsat(ts(lf), ps)
     &!       - qa) / (rra + rs)
      le = leg !+ lev
      tau = rhoa*cdm*ua
      !hw=h;xlew=le;cdmw=cdm

      return
      END SUBROUTINE RichPBL
          
      function chstar2(xmu)
      use LAKE_DATATYPES, only : ireals, iintegers
      implicit real(kind=ireals)(a-h,o-z)
      chstar2=3.2165+4.3431*xmu+.536*xmu*xmu-.0781*xmu*xmu*xmu
      return
      end

      function cmstar2(xmu)

      use LAKE_DATATYPES, only : ireals, iintegers
      implicit real(kind=ireals)(a-h,o-z)
      cmstar2=6.8741+2.6933*xmu+.3601*xmu*xmu-.0154*xmu*xmu*xmu
      return
      end

      function ph2(xmu)

      use LAKE_DATATYPES, only : ireals, iintegers
      implicit real(kind=ireals)(a-h,o-z)
      ph2    =0.5802-0.1571*xmu+.0327*xmu*xmu-.0026*xmu*xmu*xmu
      return
      end
!
!******************************************************************
      function pm2(xmu)
!****************************************************************
!

      use LAKE_DATATYPES, only : ireals, iintegers
      implicit real(kind=ireals)(a-h,o-z)
      pm2    =0.5233-0.0815*xmu+.0135*xmu*xmu-.001*xmu*xmu*xmu
      return
      end


END MODULE RICHPBL_LAKE_MOD
