MODULE SURF_SCHEME3_LAKE_MOD

contains
      SUBROUTINE SURF_SCHEME3(u2,t2,t1,q2,q1,z1,z2,ro,l1,   &
      & hflux, Elatent,tau)
      
          implicit none
          real*8 u1,u2,t1,t2,z1,z2,hum,pres,Tsr,pi,H
          real*8 jd,lat,lon,press,ws2_5,ws10,wd2_5,wd10,T2_5,T10,q2_5,q10,rhi2_5,rhi10,T_sfc
          real*8 dT,dU,L,g,k,dzita1,dzita2,ustar, Tstar,Lust,dL,ch,B_h,B_m,am,ah,bm,bh, &
          & unifU1,unifU2,unifT1,unifT2,x2,x1,yU2,yU1,yT2,yT1, xst1,xst2, &
          & kanzasU1,kanzasU2,kanzasT1,kanzasT2,convU1,convU2,convT1,convT2, &
          & Lit,ro,cp,b1,b0,b2,R,zt,dzita1T,nu,xx,Uneust, w,tau,Elatent,hflux,q1,q2, &
          & Le,Lsub,qstar,l1,dq,unifq1,unifq2,z0min
          integer strat, iter,paramzt,conv, param_uf
          character*30 input,output
          character*2 dd
          character*10 date
          character*5 time
          iter = 0
          !-----------------------------------------!
          k=0.4
          g=9.81
          pi=3.14159
          cp=1004.8
          nu=1.5/10**5
          Le=2.501e6
          Lsub=2.837e6
          z0min=1.5e-5
          !----------------SHEBA const---------------!
          am=5.
          ah=5.
          bm=am/6.5
          bh=5.
          B_m=((1-bm)/bm)**(1./3.)
          B_h=sqrt(5.)
          ch=3.
          !------------------------------------------!
          z1=max(z1,z0min)
          u1=0.
          !***************************************************************************!
          ! paramzt - parameterization of roughness parameter for heat flux
          ! 0 - z0t = z0m
          ! 1 - don't remember who
          ! 2 - Zilitinkevich
          ! 3 - Andreas - for sea ice and snow
          ! 4 - Beljaars
          !***************************************************************************!
          !param_uf - parameterization of universal functions for stable stratification
          ! 1 - Grachev (SHEBA) (2007)
          ! 2 - Beljaars and Holtslag (1991)
          ! 3 - Cheng and Brutsaert (2005)
          ! 4 - Log-linear
          ! 5 - Zilitinkevich and Esau (2006)
          !****************************************************************************!
          ! conv - adding to wind speed in unstable stratification due to convection
          ! 1 - on
          ! 2 - off
          !*****************************************************************************! 
          paramzt=3
          conv=1
          param_uf=5
          !-----------------------------------------------------------------------------!
          Tsr=(t1+t2)/2.
          dT=t2-t1
          dq=q2-q1
          dU=u2-u1
          !-------DEFINE STRATIFICATION-------------!
          ! 1 - stable
          ! 2 - unstable
          ! neutral is not necessary
          !-----------------------------------------!
          if(dT+0.61*dq*Tsr.gt.0.) strat=1           
          if(dT+0.61*dq*Tsr.lt.0.) strat=2     
      !-----------------------------------------!
          
          
          
          ustar=k*dU/log(z2/z1)
          R=ustar*z1/nu
          if (R.lt.0.135) then
          b0=1.25
          b1=0.
          b2=0.
          endif
          if(R.gt.0.135.and.R.lt.2.5) then
          b0=0.149
          b1=-0.55
          b2=0.
          endif
          if(R.gt.2.5) then
          b0=0.317
          b1=-0.565
          b2=-0.183
          endif
          if (paramzt.eq.0) zt=z1
          if (paramzt.eq.1) zt=z1/exp(6.5)
          if (paramzt.eq.2) zt=z1/exp(0.13*R**0.45)       
          if (paramzt.eq.3) zt=z1*exp(b0+b1*log(R)+b2*(log(R))**2)
          if (paramzt.eq.4) then
          if(R .le. 0.111) then
                        xx = - 2.43
                else
                        if(0.111 .lt. R .and. R .le. 16.3) then
                                xx = 0.83*log(R) - 0.6
                        else
                                xx = 0.49 * R**0.45
                        end if
                end if
                zt = z1*exp(-xx)
                endif
      
          L=dU**2/(g*dT*log(z2/z1)**2)*Tsr*log(z2/zt)           
      dzita2=z2/L
          dzita1=z1/L
      
          if (strat.eq.1) then   
          L=1000.    
20        iter=iter+1
      dzita2=z2/L
          dzita1=z1/L

                             ! STABLE STRATIFICATION
      if(param_uf==1) then
           xst1=(1+dzita1)**(1./3.)
           xst2=(1+dzita2)**(1./3.)
           unifU1=-3.*am/bm*(xst1-1)+am*B_m/(2.*bm)*(2*log((xst1+B_m)/(1+B_m))-&
           & log((xst1**2-xst1*B_m+B_m**2)/(1-B_m+B_m**2))+2*sqrt(3.)* &
           &(atan((2.*xst1-B_m)/(sqrt(3.)*B_m))-atan((2.-B_m)/(sqrt(3.)*B_m))))
           unifU2=-3.*am/bm*(xst2-1)+am*B_m/(2.*bm)*(2*log((xst2+B_m)/(1+B_m))-&
           &log((xst2**2-xst1*B_m+B_m**2)/(1-B_m+B_m**2))+2*sqrt(3.)*(atan((2.*xst2-B_m)/&
           &(sqrt(3.)*B_m))-atan((2.-B_m)/(sqrt(3.)*B_m))))
           else if(param_uf==2) then
           unifU1=-(1.*dzita1+0.667*(dzita1-5./0.35)*exp(-0.35*dzita1)+0.667*5./0.35)
           unifU2=-(1.*dzita2+0.667*(dzita2-5./0.35)*exp(-0.35*dzita2)+0.667*5./0.35)
           else if(param_uf==3) then
           unifU1=-6.1*log(dzita1+(1+dzita1**2.5)**(1./2.5))
           unifU2=-6.1*log(dzita2+(1+dzita2**2.5)**(1./2.5))
           else if(param_uf==4) then
           unifU1=-5.*dzita1
           unifU2=-5.*dzita2
           else if(param_uf==5) then
           unifU1=-(3.*dzita1**(5./6.))
           unifU2=-(3.*dzita2**(5./6.))
           endif
       ustar=k*dU/(log(z2/z1)-unifU2+unifU1)
          R=ustar*z1/nu
          if (R.lt.0.135) then
          b0=1.25
          b1=0.
          b2=0.
          endif
          if(R.gt.0.135.and.R.lt.2.5) then
          b0=0.149
          b1=-0.55
          b2=0.
          endif
          if(R.gt.2.5) then
          b0=0.317
          b1=-0.565
          b2=-0.183
          endif
          if (paramzt.eq.0) zt=z1
          if (paramzt.eq.1) zt=z1/exp(6.5)
          if (paramzt.eq.2) zt=z1/exp(0.13*R**0.45)
          if (paramzt.eq.3) zt=z1*exp(b0+b1*log(R)+b2*(log(R))**2)
          if (paramzt.eq.4) then
          if(R .le. 0.111) then
                        xx = - 2.43
                else
                        if(0.111 .lt. R .and. R .le. 16.3) then
                                xx = 0.83*log(R) - 0.6
                        else
                                xx = 0.49 * R**0.45
                        end if
                end if
                zt = z1*exp(-xx)
                endif
          dzita1T=zt/L
          if (param_uf==1) then
          unifT1=-bh/2.*log(1+ch*dzita1T+dzita1T**2)+&
          &(-ah/B_h+bh*ch/(2.*B_h))*(log((2*dzita1T+ch-B_h)/&
          &(2*dzita1T+ch+B_h))-log((ch-B_h)/(ch+B_h)))
          unifT2=-bh/2.*log(1+ch*dzita2+dzita2**2)+&
          &(-ah/B_h+bh*ch/(2.*B_h))*(log((2*dzita2+ch-B_h)/&
          &(2*dzita2+ch+B_h))-log((ch-B_h)/(ch+B_h)))
          else if(param_uf==2) then
          unifT1=-((1+2./3.*1.*dzita1T)**(3./2.)+0.667*(dzita1T-5./0.35)*&
          &exp(-0.35*dzita1T)+0.667*5./0.35-1.)
          unifT2=-((1+2./3.*1.*dzita2)**(3./2.)+0.667*(dzita2-5./0.35)*&
          &exp(-0.35*dzita2)+0.667*5./0.35-1.)
          else if (param_uf==3) then
          unifT1=-5.3*log(dzita1T+(1+dzita1T**1.1)**(1./1.1))
          unifT2=-5.3*log(dzita2+(1+dzita2**1.1)**(1./1.1))
          else if (param_uf==4) then
          unifT1=-5.*dzita1T
          unifT2=-5.*dzita2
          else if (param_uf==5) then
          unifT1=-(2.5*dzita1T**(4./5.))
          unifT2=-(2.5*dzita2**(4./5.))
          endif
          unifq1=unifT1
          unifq2=unifT2
          Tstar=(k)*dT/(log(z2/zt)-unifT2+unifT1)
          qstar=k*dq/(log(z2/zt)-unifq2+unifq1)
          Lit=ustar**2/(k*g*(Tstar+0.61*Tsr*qstar))*Tsr
          Lit = sign(1._8,Lit)*max(abs(Lit),1.e-3) 
          dL=abs(L-Lit)

          if (iter.gt.100) goto 33
          
          if (dL.lt.5) then
             if(L.gt.z2) then
                 if(dL.lt.1) goto 33
                 L=(Lit+L)/2.
                 else
                 if(dL.lt.0.2) goto 33
                 L=Lit
                 endif
      endif
          if(iter.gt.1) then
          L=(Lit+L)/2.
          else
          L=Lit
          endif
      goto 20
          endif

      
          if (strat.eq.2) then                   !UNSTABLE STRATIFICATION
          L=-10000.
          !-----------------------------------!
          if(conv.eq.1) then
          ustar=k*dU/log(z2/z1)
          tstar=k*dT/log(z2/zt)
          w=(-g/Tsr*tstar*ustar*1000.)**(1/3)
          Uneust=sqrt(u2**2+(w*1.)**2)
          dU=Uneust-u1
      endif

          !-----------------------------------!
          
 40   iter=iter+1
      dzita2=z2/L
          dzita1=z1/L     
          x2=(1-16*dzita2)**(1/4)
      x1=(1-16*dzita1)**(1/4)
          kanzasU2=2*log(0.5*(1+x2))+log(0.5*(1+x2**2))-2*atan(x2)+pi/2.
          kanzasU1=2*log(0.5*(1+x1))+log(0.5*(1+x1**2))-2*atan(x1)+pi/2.          
      yU2=(1-10.15*dzita2)**(1/3)
          yU1=(1-10.15*dzita1)**(1/3)     
          convU2=(3./2.)*log((1./3.)*(yU2**2+yU2+1))-sqrt(3.)*log((2*yU2+1)/sqrt(3.))+pi/sqrt(3.)
      convU1=(3./2.)*log((1./3.)*(yU1**2+yU1+1))-sqrt(3.)*log((2*yU1+1)/sqrt(3.))+pi/sqrt(3.)     
      unifU2=(kanzasU2+dzita2**2*convU2)/(1+dzita2**2)
      unifU1=(kanzasU1+dzita1**2*convU1)/(1+dzita1**2)      
          ustar=k*dU/(log(z2/z1)-unifU2+unifU1)
          R=ustar*z1/nu
          if (R.lt.0.135) then
          b0=1.25
          b1=0.
          b2=0.
          endif
          if(R.gt.0.135.and.R.lt.2.5) then
          b0=0.149
          b1=-0.55
          b2=0.
          endif
          if(R.gt.2.5) then
          b0=0.317
          b1=-0.565
          b2=-0.183
          endif
          if (paramzt.eq.0) zt=z1
          if (paramzt.eq.1) zt=z1/exp(6.5)
          if (paramzt.eq.2) zt=z1/exp(0.13*R**0.45)
          if (paramzt.eq.3) zt=z1*exp(b0+b1*log(R)+b2*(log(R))**2)
          if (paramzt.eq.4) then
          if(R .le. 0.111) then
                        xx = - 2.43
                else
                        if(0.111 .lt. R .and. R .le. 16.3) then
                                xx = 0.83*log(R) - 0.6
                        else
                                xx = 0.49 * R**0.45
                        end if
                end if
                zt = z1*exp(-xx)
                endif
      dzita1T=zt/L
          kanzasT2=2*log(0.5*(1+sqrt(1-16*dzita2)))
      kanzasT1=2*log(0.5*(1+sqrt(1-16*dzita1T)))
          yT2=(1-34.15*dzita2)**(1/3)
          yT1=(1-34.15*dzita1T)**(1/3)
          convT2=(3./2.)*log((1./3.)*(yT2**2+yT2+1))-sqrt(3.)*log((2*yT2+1)/sqrt(3.))+pi/sqrt(3.)
          convT1=(3./2.)*log((1./3.)*(yT1**2+yT1+1))-sqrt(3.)*log((2*yT1+1)/sqrt(3.))+pi/sqrt(3.)
          unifT2=(kanzasT2+dzita2**2*convT2)/(1+dzita2**2)
          unifT1=(kanzasT1+dzita1T**2*convT1)/(1+dzita1T**2)
          unifq2=unifT2
          unifq1=unifT1
          Tstar=(k)*dT/(log(z2/zt)-unifT2+unifT1)
          qstar=k*dq/(log(z2/zt)-unifq2+unifq1)
          Lit=ustar**2/(k*g*(Tstar+0.61*Tsr*qstar))*Tsr
          Lit = sign(1._8,Lit)*max(abs(Lit),1.e-3) 
      dL=abs(L-Lit)
          
           if (iter.gt.100) goto 35
          
          if (dL.lt.5) then
             if(L.gt.z2) then
                 if(dL.lt.1) goto 35
                 L=(Lit+L)/2.
                 else
                 if(dL.lt.0.2) goto 35
                 L=Lit
                 endif
      endif
          if(iter.gt.1) then
          L=(Lit+L)/2.
          else
          L=Lit
          endif
      goto 40
          
          endif
      
          if (strat.eq.3) then
      ustar=k*dU/log(z2/z1)
          tstar=k*dT/log(z2/zt)
          goto 30
          endif

33    L=Lit
       dzita2=z2/L
           dzita1=z1/L    
       if(param_uf==1) then
           xst1=(1+dzita1)**(1./3.)
           xst2=(1+dzita2)**(1./3.)
           unifU1=-3.*am/bm*(xst1-1)+am*B_m/(2.*bm)*(2*log((xst1+B_m)/&
           &(1+B_m))-log((xst1**2-xst1*B_m+B_m**2)/(1-B_m+B_m**2))+2*sqrt(3.)*&
           &(atan((2.*xst1-B_m)/(sqrt(3.)*B_m))-atan((2.-B_m)/(sqrt(3.)*B_m))))
           unifU2=-3.*am/bm*(xst2-1)+am*B_m/(2.*bm)*(2*log((xst2+B_m)/(1+B_m))-&
           &log((xst2**2-xst1*B_m+B_m**2)/(1-B_m+B_m**2))+2*sqrt(3.)*&
           (atan((2.*xst2-B_m)/(sqrt(3.)*B_m))-atan((2.-B_m)/(sqrt(3.)*B_m))))
           else if(param_uf==2) then
           unifU1=-(1.*dzita1+0.667*(dzita1-5./0.35)*exp(-0.35*dzita1)+0.667*5./0.35)
           unifU2=-(1.*dzita2+0.667*(dzita2-5./0.35)*exp(-0.35*dzita2)+0.667*5./0.35)
           else if(param_uf==3) then
           unifU1=-6.1*log(dzita1+(1+dzita1**2.5)**(1./2.5))
           unifU2=-6.1*log(dzita2+(1+dzita2**2.5)**(1./2.5))
           else if(param_uf==4) then
           unifU1=-5.*dzita1
           unifU2=-5.*dzita2
           else if(param_uf==5) then
           unifU1=-(3.*dzita1**(5./6.))
           unifU2=-(3.*dzita2**(5./6.))
           endif
          ustar=k*dU/(log(z2/z1)-unifU2+unifU1)
          R=ustar*z1/nu
          if (R.lt.0.135) then
          b0=1.25
          b1=0.
          b2=0.
          endif
          if(R.gt.0.135.and.R.lt.2.5) then
          b0=0.149
          b1=-0.55
          b2=0.
          endif
          if(R.gt.2.5) then
          b0=0.317
          b1=-0.565
          b2=-0.183
          endif
          if (paramzt.eq.0) zt=z1
          if (paramzt.eq.1) zt=z1/exp(6.5)
          if (paramzt.eq.2) zt=z1/exp(0.13*R**0.45)
          if (paramzt.eq.3) zt=z1*exp(b0+b1*log(R)+b2*(log(R))**2)
          if (paramzt.eq.4) then
          if(R .le. 0.111) then
                        xx = - 2.43
                else
                        if(0.111 .lt. R .and. R .le. 16.3) then
                                xx = 0.83*log(R) - 0.6
                        else
                                xx = 0.49 * R**0.45
                        end if
                end if
                zt = z1*exp(-xx)
                endif
                dzita1T=zt/L
          if (param_uf==1) then
          unifT1=-bh/2.*log(1+ch*dzita1T+dzita1T**2)+(-ah/B_h+bh*ch/(2.*B_h))*&
          &(log((2*dzita1T+ch-B_h)/(2*dzita1T+ch+B_h))-log((ch-B_h)/(ch+B_h)))
          unifT2=-bh/2.*log(1+ch*dzita2+dzita2**2)+(-ah/B_h+bh*ch/(2.*B_h))*&
          &(log((2*dzita2+ch-B_h)/(2*dzita2+ch+B_h))-log((ch-B_h)/(ch+B_h)))
          else if(param_uf==2) then
          unifT1=-((1+2./3.*1.*dzita1T)**(3./2.)+0.667*(dzita1T-5./0.35)*&
          &exp(-0.35*dzita1T)+0.667*5./0.35-1.)
          unifT2=-((1+2./3.*1.*dzita2)**(3./2.)+0.667*(dzita2-5./0.35)*&
          &exp(-0.35*dzita2)+0.667*5./0.35-1.)
          else if (param_uf==3) then
          unifT1=-5.3*log(dzita1T+(1+dzita1T**1.1)**(1./1.1))
          unifT2=-5.3*log(dzita2+(1+dzita2**1.1)**(1./1.1))
          else if (param_uf==4) then
          unifT1=-5.*dzita1T
          unifT2=-5.*dzita2
          else if (param_uf==5) then
          unifT1=-(2.5*dzita1T**(4./5.))
          unifT2=-(2.5*dzita2**(4./5.))
          endif
          unifq1=unifT1
          unifq2=unifT2
          Tstar=(k)*dT/(log(z2/zt)-unifT2+unifT1)
          qstar=k*dq/(log(z2/zt)-unifq2+unifq1)
          goto 30

35    L=Lit
      dzita2=z2/L
          dzita1=z1/L     
          x2=(1-16*dzita2)**(1/4)
      x1=(1-16*dzita1)**(1/4)
          kanzasU2=2*log(0.5*(1+x2))+log(0.5*(1+x2**2))-2*atan(x2)+pi/2.
          kanzasU1=2*log(0.5*(1+x1))+log(0.5*(1+x1**2))-2*atan(x1)+pi/2.          
      yU2=(1-10.15*dzita2)**(1/3)
          yU1=(1-10.15*dzita1)**(1/3)     
          convU2=(3./2.)*log((1./3.)*(yU2**2+yU2+1))-sqrt(3.)*log((2*yU2+1)/sqrt(3.))+pi/sqrt(3.)
      convU1=(3./2.)*log((1./3.)*(yU1**2+yU1+1))-sqrt(3.)*log((2*yU1+1)/sqrt(3.))+pi/sqrt(3.)     
      unifU2=(kanzasU2+dzita2**2*convU2)/(1+dzita2**2)
      unifU1=(kanzasU1+dzita1**2*convU1)/(1+dzita1**2)      
          ustar=k*dU/(log(z2/z1)-unifU2+unifU1)
          R=ustar*z1/nu
          if (R.lt.0.135) then
          b0=1.25
          b1=0.
          b2=0.
          endif
          if(R.gt.0.135.and.R.lt.2.5) then
          b0=0.149
          b1=-0.55
          b2=0.
          endif
          if(R.gt.2.5) then
          b0=0.317
          b1=-0.565
          b2=-0.183
          endif
          if (paramzt.eq.0) zt=z1
          if (paramzt.eq.1) zt=z1/exp(6.5)
          if (paramzt.eq.2) zt=z1/exp(0.13*R**0.45)
          if (paramzt.eq.3) zt=z1*exp(b0+b1*log(R)+b2*(log(R))**2)
          if (paramzt.eq.4) then
          if(R .le. 0.111) then
                        xx = - 2.43
                else
                        if(0.111 .lt. R .and. R .le. 16.3) then
                                xx = 0.83*log(R) - 0.6
                        else
                                xx = 0.49 * R**0.45
                        end if
                end if
                zt = z1*exp(-xx)
                endif
      dzita1T=zt/L
          kanzasT2=2*log(0.5*(1+sqrt(1-16*dzita2)))
      kanzasT1=2*log(0.5*(1+sqrt(1-16*dzita1T)))
          yT2=(1-34.15*dzita2)**(1/3)
          yT1=(1-34.15*dzita1T)**(1/3)
          convT2=(3./2.)*log((1./3.)*(yT2**2+yT2+1))-sqrt(3.)*log((2*yT2+1)/sqrt(3.))+pi/sqrt(3.)
          convT1=(3./2.)*log((1./3.)*(yT1**2+yT1+1))-sqrt(3.)*log((2*yT1+1)/sqrt(3.))+pi/sqrt(3.)
          unifT2=(kanzasT2+dzita2**2*convT2)/(1+dzita2**2)
          unifT1=(kanzasT1+dzita1T**2*convT1)/(1+dzita1T**2)
          unifq1=unifT1
          unifq2=unifT2
          Tstar=(k)*dT/(log(z2/zt)-unifT2+unifT1)
          qstar=k*dq/(log(z2/zt)-unifq2+unifq1)
          goto 30

30    continue

      
      hflux=-1.*ro*cp*ustar*Tstar
          if (l1==0) then
          Elatent=-1.*Le*ro*ustar*qstar
          else
          Elatent=-1.*Lsub*ro*ustar*qstar
          endif
          tau=ro*ustar**2
          iter=0
          return
          
          END SUBROUTINE SURF_SCHEME3

END MODULE SURF_SCHEME3_LAKE_MOD
