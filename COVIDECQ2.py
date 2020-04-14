# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:02:27 2020

@author: MONICA
"""


#INPUT:
#covidm('PROVINCE',initial susceptible,initial infected)
#EXAMPLE:
#covidm('PAMPANGA',1000000,15)
#EDIT FILE NAME AT LINE 199| FORMAT: ecq45province| Example: ecq45pampanga
#EDIT FILE NAME AT LINE 201| FORMAT: ecq45province.csv| Example: ecq45pampanga.csv
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas import DataFrame

def StateFunc(t,x,compartments,beta,rho,connection,protect,alpha,gammaA,isoA,gamma,\
                          iso,death,impor,gammaIs,deathIs,dt):
    #differential equations
    xprime=np.zeros(compartments)
    xprime[0]=-beta*(x[4]+x[2]*0.7+0.05*x[5]+0.02*x[3])*x[0]+rho*x[6]+connection+protect*x[1]
    xprime[1]=beta*(x[4]+x[2]*0.7+0.05*x[5]+0.02*x[3])*x[0]-alpha*0.8*x[1]-alpha*0.2*x[1]-protect*x[1]
    xprime[2]=alpha*0.8*x[1]-gammaA*x[2]-isoA
    xprime[3]=isoA-gammaA*x[3]
    xprime[4]=alpha*0.2*x[1]-gamma*x[4]-iso-death*x[4]+impor
    xprime[5]=iso-gammaIs*x[5]-deathIs*x[5]
    xprime[6]=gammaA*x[2]+gamma*x[4]+gammaA*x[3]-rho*x[6]+gammaIs*x[5]
    return xprime

def RK4ForwardState(t,x,compartments,R0,rho,tau,connection,protect,alpha,gammaA,isolateA,gamma,\
                          isolateIs,death,impor,bedsanddoctors,detectivekits,dt,N):
    for i in range(1,N):
        if i<45*100:
            beta=R0[0]/(tau*x[0,0])
        elif i>90*100:
            beta=R0[2]/(tau*x[0,0])
        else:
            beta=R0[1]/(tau*x[0,0])
        if beta<0:
            beta=0
        healthcarecapacity=bedsanddoctors-0.75*x[i-1,5]
        if healthcarecapacity<0:
            healthcarecapacity=0
        deathIs=0.15/14*(1-(0.5*healthcarecapacity/(1+healthcarecapacity)))
        if deathIs<0:
            deathIs=0
        elif deathIs>1:
            deathIs=1
        gammaIs=1/14-deathIs
        if gammaIs<0:
            gammaIs=0
        elif gammaIs>1:
            gammaIs=1
        isoA=detectivekits*isolateA
        iso=detectivekits*isolateIs
        if iso>x[i-1,4]:
            iso=x[i-1,4]
        if isoA>x[i-1,2]:
            isoA=x[i-1,2]
        k1=StateFunc(t[i-1],x[i-1],compartments,beta,rho,connection,protect,alpha,gammaA,isoA,gamma,\
                          iso,death,impor,gammaIs,deathIs,dt)
        k2=StateFunc(t[i-1]+(dt/2),x[i-1]+(dt/2)*k1,compartments,beta,rho,connection,protect,alpha,gammaA,isoA,gamma,\
                          iso,death,impor,gammaIs,deathIs,dt)
        k3=StateFunc(t[i-1]+(dt/2),x[i-1]+(dt/2)*k2,compartments,beta,rho,connection,protect,alpha,gammaA,isoA,gamma,\
                          iso,death,impor,gammaIs,deathIs,dt)
        k4=StateFunc(t[i-1]+dt,x[i-1]+dt*k3,compartments,beta,rho,connection,protect,alpha,gammaA,isoA,gamma,\
                          iso,death,impor,gammaIs,deathIs,dt)
        x[i]=x[i-1]+(dt/6)*(k1+2*k2+2*k3+k4)
        for j in range(compartments):
            if x[i,j]<0:
                x[i,j]=0
    return x
    
def covidm(place,S0,I0):
    T=400
    T1=150
    tol=0.00001
    N=100*T+1
    compartments=7 #no. of compartments
    t=np.linspace(0,T,N)
    t1=np.linspace(0,T1,100*T1+1)
    x=np.zeros((N,compartments))
    dt=T/N
#==============================================================================    
    #INITIALIZE POPULATION
    x[0,0]=S0  #Susceptible
    x[0,1]=0        #Exposed
    x[0,2]=0       #Asymptomatic
    x[0,3]=0        #Isolated Asymptomatic 
    x[0,4]=I0       #Symptomatic
    x[0,5]=0     #Isolated Symptomatic 
    x[0,6]=0      #Recovered
#==============================================================================
    bedsanddoctors=50000
    detectivekits=100
    #PARAMETERS
    connection=.001*x[0,0]
    protect=0
    R01=[3,1.5,3]
    R02=[3,0.9,1.5]
    R03=[2,0.9,1.2]
    R04=[3,3,3]
    impor=0.1
    tau=5.5
    isolateIs=0.999
    isolateA=1-isolateIs
    alpha=11/14
    death=0.1/14
    gamma=1/14-death
    gammaA=1.5/14
    rho=0.1/30
    x1=np.zeros((N,4))
    y1=np.zeros((N,4))
    x11=np.zeros((T1*100+1,4))
    y11=np.zeros((T1*100+1,4))
    for k in range(4):
        if k==0:
            R0=R01
        elif k==1:
            R0=R02
        elif k==2:
            R0=R03
        else:
            R0=R04
        test=-1
        while (test<0):
            oldx=x
            x=RK4ForwardState(t,x,compartments,R0,rho,tau,connection,protect,alpha,gammaA,isolateA,gamma,\
                          isolateIs,death,impor,bedsanddoctors,detectivekits,dt,N)
            temp2=tol*np.sum(np.abs(x))-np.sum(np.abs(oldx-x))        
            test=temp2
        
        x1[:,k]=x[:,4]+x[:,5]+x[:,2]+x[:,3]
        y1[:,k]=x[:,4]+x[:,5]+x[:,3]
        for h in range(T1*100+1):
            x11[h,k]=x[h,4]+x[h,5]+x[h,2]+x[h,3]
            y11[h,k]=x[h,4]+x[h,5]+x[h,3]
    plt.figure(1)
    gs = gridspec.GridSpec(2, 3)
    plt.suptitle('%s \n $S_0=%d,I_0=%d$' %(place,x[0,0],x[0,4]),fontsize=12,x=0.25,y=0.97)
    plt.subplots_adjust(top=0.8,hspace=0.2, wspace=0.5,left=0.15,right=0.95)
    plt.rc('ytick',labelsize=6)
    plt.rc('xtick',labelsize=6)

    plt.subplot(gs[:, 0]) # row 0, col 0
    plt.rc('ytick',labelsize=6)
    plt.rc('xtick',labelsize=6)
    plt.plot(t,x1[:,3],linestyle="-",color='lightblue',linewidth=2,label='$R_0=3$')
    plt.plot(t,x1[:,0],linestyle="-.",color='yellowgreen',linewidth=2,label='$R_0=[3,1.5,3]$')
    plt.plot(t,x1[:,1],linestyle="--",color='lime',linewidth=2,label='$R_0=[3,0.9,1.5]$')
    plt.plot(t,x1[:,2],linestyle="-",color='green',linewidth=2,label='$R_0=[2,0.9,1.2]$')
    plt.figlegend(loc="upper right",ncol=2,bbox_to_anchor=(0.97,1))
    plt.ylabel('Active Cases',fontsize=9)
    ax=plt.gca()
    ax.set_xlim(0,T)
#    plt.xlabel('Days',fontsize=12)

       
    plt.subplot(gs[:, 1])
    plt.rc('ytick',labelsize=6)
    plt.rc('xtick',labelsize=6)
    plt.plot(t,y1[:,3],linestyle="-",color='lightblue',linewidth=2)
    plt.plot(t,y1[:,0],linestyle="-.",color='yellowgreen',linewidth=2)
    plt.plot(t,y1[:,1],linestyle="--",color='lime',linewidth=2)
    plt.plot(t,y1[:,2],linestyle="-",color='green',linewidth=2)
    plt.ylabel('Detected Active Cases',fontsize=9)
    ax=plt.gca()
    ax.set_xlim(0,T)
    plt.xlabel('Days',fontsize=10)
    
    plt.subplot(gs[0, 2]) # row 0, col 0
    plt.rc('ytick',labelsize=6)
    plt.rc('xtick',labelsize=6)
    plt.plot(t1,x11[:,3],linestyle="-",color='lightblue',linewidth=2)
    plt.plot(t1,x11[:,0],linestyle="-.",color='yellowgreen',linewidth=2)
    plt.plot(t1,x11[:,1],linestyle="--",color='lime',linewidth=2)
    plt.plot(t1,x11[:,2],linestyle="-",color='green',linewidth=2)
    plt.gca().set_title('ZOOMED IN',fontsize=10)
    plt.ylabel('Active Cases',fontsize=7)
    ax=plt.gca()
    ax.set_xlim(0,T1)
#    plt.xlabel('Days',fontsize=12)
    
    plt.subplot(gs[1, 2]) # row 0, col 0
    plt.rc('ytick',labelsize=6)
    plt.rc('xtick',labelsize=6)
    plt.plot(t1,y11[:,3],linestyle="-",color='lightblue',linewidth=2)
    plt.plot(t1,y11[:,0],linestyle="-.",color='yellowgreen',linewidth=2)
    plt.plot(t1,y11[:,1],linestyle="--",color='lime',linewidth=2)
    plt.plot(t1,y11[:,2],linestyle="-",color='green',linewidth=2)
    plt.ylabel('Detected Active Cases',fontsize=7)
    ax=plt.gca()
    ax.set_xlim(0,T1)
    plt.show
    plt.savefig('ecq45NCR', dpi = 300)
    df=DataFrame({'time':t,'3,1.5,3':x1[:,0],'3,0.9,1.5':x1[:,1],'2,0.9,1.2':x1[:,2],'3':x1[:,3],'3,1.5,3DAC':y1[:,0],'3,0.9,1.5DAC':y1[:,1],'2,0.9,1.2DAC':y1[:,2],'3DAC':y1[:,3]})
    df.to_csv('ecq45NCR.csv')