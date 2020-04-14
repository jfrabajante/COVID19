# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:18:08 2020
@author: MONICA
"""
#INPUT:
#covidm('PROVINCE',initial susceptible,initial infected)
#EXAMPLE:
#covidm('PAMPANGA',1000000,15)
#EDIT FILE NAME AT LINE 150| FORMAT: R0province| Example: R0pampanga
#EDIT FILE NAME AT LINE 152| FORMAT: R0province.csv| Example: R0pampanga.csv
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

def RK4ForwardState(t,x,compartments,beta,rho,connection,protect,alpha,gammaA,isolateA,gamma,\
                          isolateIs,death,impor,bedsanddoctors,detectivekits,dt,N):
    for i in range(1,N):
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
    tol=0.00001
    N=100*T
    compartments=7 #no. of compartments
    t=np.linspace(0,T,N)
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
    R0=[1.5,2,2.5,3,3.5,4]
    impor=0.1
    tau=5.5
    isolateIs=0.999
    isolateA=1-isolateIs
    alpha=11/14
    death=0.1/14
    gamma=1/14-death
    gammaA=1.5/14
    rho=0.1/30
    x1=np.zeros((N,6))
    y1=np.zeros((N,6))
    
    for k in range(6):
        beta=R0[k]/(tau*x[0,0])
        if beta<0:
            beta=0
        test=-1
        while (test<0):
            oldx=x
            x=RK4ForwardState(t,x,compartments,beta,rho,connection,protect,alpha,gammaA,isolateA,gamma,\
                          isolateIs,death,impor,bedsanddoctors,detectivekits,dt,N)
            temp2=tol*np.sum(np.abs(x))-np.sum(np.abs(oldx-x))        
            test=temp2
        
        x1[:,k]=x[:,4]+x[:,5]+x[:,2]+x[:,3]
        y1[:,k]=x[:,4]+x[:,5]+x[:,3]
    
    plt.figure(1)
    gs = gridspec.GridSpec(1, 2)
    plt.suptitle('%s \n $S_0=%d,I_0=%d$' %(place,x[0,0],x[0,4]),fontsize=12,x=0.3,y=0.94)
    plt.subplots_adjust(top=0.8,hspace=0.2, wspace=0.4,left=0.15,right=0.95)


    plt.subplot(gs[0, 0]) # row 0, col 0
    plt.plot(t,x1[:,0],linestyle="-.",color='yellowgreen',linewidth=2,label='$R_0=1.5$')
    plt.plot(t,x1[:,1],linestyle="--",color='lime',linewidth=2,label='$R_0=2$')
    plt.plot(t,x1[:,2],linestyle="-",color='green',linewidth=2,label='$R_0=2.5$')
    plt.plot(t,x1[:,3],linestyle="-.",color='burlywood',linewidth=2,label='$R_0=3$')
    plt.plot(t,x1[:,4],linestyle="--",color='red',linewidth=2,label='$R_0=3.5$')
    plt.plot(t,x1[:,5],linestyle="-",color='maroon',linewidth=2,label='$R_0=4$')
    plt.figlegend(loc="upper right",ncol=2,bbox_to_anchor=(0.95,1))
    plt.ylabel('Active Cases',fontsize=10)
    ax=plt.gca()
    ax.set_xlim(0,T)
    plt.xlabel('Days',fontsize=12)

       
    plt.subplot(gs[0, 1])
    plt.plot(t,y1[:,0],linestyle="-.",color='yellowgreen',linewidth=2)
    plt.plot(t,y1[:,1],linestyle="--",color='lime',linewidth=2)
    plt.plot(t,y1[:,2],linestyle="-",color='green',linewidth=2)
    plt.plot(t,y1[:,3],linestyle="-.",color='burlywood',linewidth=2)
    plt.plot(t,y1[:,4],linestyle="--",color='red',linewidth=2)
    plt.plot(t,y1[:,5],linestyle="-",color='maroon',linewidth=2)
    plt.ylabel('Detected Active Cases',fontsize=10)
    ax=plt.gca()
    ax.set_xlim(0,T)
    plt.xlabel('Days',fontsize=12)
    
    plt.show
    plt.savefig('R0NCR', dpi = 300)
    df=DataFrame({'time':t,'1.5AC':x1[:,0],'2AC':x1[:,1],'2.5AC':x1[:,2],'3AC':x1[:,3],'3.5AC':x1[:,4],'4AC':x1[:,5],'1.5DAC':y1[:,0],'2DAC':y1[:,1],'2.5DAC':y1[:,2],'3DAC':y1[:,3],'3.5DAC':y1[:,4],'4DAC':y1[:,5]})
    df.to_csv('R0NCR.csv')