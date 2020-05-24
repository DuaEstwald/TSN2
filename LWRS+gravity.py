
# Author: Elena Arjona Galvez
# Project: Jeans mass and gravitational collapse 

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


time_start = time.clock()

# =========================================================================
# ======================== FUNCTIONS ======================================
# =========================================================================



def centered(f,x,h):
    F = np.empty(len(x))
    for i in range(1,len(x)-1):
        F[i] = (f[i+1]-f[i-1])/(2.*h)  # EN ESTE CASO SE PUEDE CALCULAR DESDE i = 1 HASTA i = len-1 PORQUE NUNCA SABEMOS EL VAMOR len+1

    return F # usarlo con el array de x[1:len-1]



def LWRS1(u,f,x,ht,hx):
    U = np.empty(len(x))
    for i in range(0,len(x)-1):
        U[i] = 0.5*(u[i]+u[i+1])-0.5*(ht/hx)*(f[i+1]-f[i])
    return U



def rhodef(Arho,rho0,x,t,k,w,phi):
    return rho0*(1.0+Arho*np.cos(k*x-w*t+phi))


def Pdef(AP,P0,x,t,k,w,phi):
    return P0*(1.0+AP*np.cos(k*x-w*t+phi))


def vdef(Av,x,t,k,w,phi):
    return Av*np.cos(k*x-w*t+phi)

def csdef(gamma,rho,P):
    return np.sqrt(gamma*P/rho)

def edef(P,rho,gamma,v):
    return (P/(rho*(gamma-1.)))+0.5*v**2
# =========================================================================
# ========================== PARAMETERS ===================================
# =========================================================================
import sys

G = 1.


xi = 0.
xf = 2.*np.pi
N = 500
dx = (xf-xi)/N

x = np.arange(xi-dx/2,xf+3.*dx/2.,dx) # xE(0,2pi) & N = 50 puntos

Arho = 1e-4

phi = 0.
k = 1.
gamma = 5./3.
#P00 = float(sys.argv[1])
P00 = 0.6
rho00 = 1.0
#rho00 = float(sys.argv[1])
#rho00 = 1.*np.sqrt(gamma*P00/(4.*np.pi*G))*k
#rho00 = np.sqrt(gamma*P00)*k

# =========================================================================
# ================= SYSTEM OF EQUATIONS ===================================
# =========================================================================

# Queremos resolver el siguiente sistema de ecuaciones

# drho/dt + Lambda[rhov] = 0
# d(rhov)/dt + Lambda[rhov x v + PI] = rhog
# d(rhoe)/dt + Lambda[(rhoe + P)v] = rhovg


# Donde P = (e-0.5v**2)rho(gamma-1)
#       e = u + 0.5v**2


# Por ahora, lo que vamos a hacer es utilizar el modo positivo de omega
cs0 = np.sqrt(gamma*P00/rho00)
AP = gamma*Arho
Av = cs0*Arho
w0 = -cs0*k

# ========================================================================
# ====================== SPATIAL DERIVATIVE ==============================
# ========================================================================

ht_show = 2.
t = 0.
next_show = ht_show + t


CFL = 0.9
# ========================================================================
rho = rhodef(Arho,rho00,x,t,k,w0,phi)
P = Pdef(AP,P00,x,t,k,w0,phi)
v = vdef(Av,x,t,k,w0,phi)


e = edef(P,rho,gamma,v)

rhov = rho*v
rhoe = rho*e



# =======================================================================
# ================ ANALISIS LAGRANGIANO =================================
# =======================================================================

iprho = np.random.choice(np.arange(len(rho)),10)
xrho = x[iprho]
vrho = v[iprho]




# ========================================================================
# ========================== SUBPLOTS ====================================
# ========================================================================

fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(6,9))
line1, = ax1.plot([],[])
line1an, = ax1.plot([],[],'k:')
line2, = ax2.plot([],[])
line2an, = ax2.plot([],[],'k.')
line3, = ax3.plot([],[])
line3an, = ax3.plot([],[],'k.')

line = [line1, line1an, line2, line2an, line3, line3an]
ax1.set_xlim(xi,xf)
#ax1.set_ylim(0.,2.)
ax1.set_ylim(-10*Arho,10*Arho)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\rho_1$')
ax1.grid()


ax2.set_xlim(xi,xf)
ax2.set_ylim(3*Av,-3*Av)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$v_1$')
ax2.grid()


ax3.set_xlim(xi,xf)
ax3.set_ylim(0.,P00*1.5)
#ax3.set_ylim(0.,2.)
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$P_1$')
ax3.grid()


plt.tight_layout()

rhoan = rhodef(Arho,rho00,x,t,k,w0,phi)
Pan = Pdef(AP,P00,x,t,k,w0,phi)
van = vdef(Av,x,t,k,w0,phi)

phi = np.arange(0.,np.pi,dx)

#tfile = open('rhofix/rho'+str(rho00)+'P'+str(P00)+'.txt','w+')
#tfile.writelines('# rho00 = '+str(rho00)+' P00 = '+str(P00)+'\n')
#tfile.writelines('#max(rho)\t max(P)\ttime\n')
time_one = time.clock()
def animate(i):
    global t, next_show, rho, rhov, rhoe, P, rhoan, Pan, van, v, xrho, vrho
    if t <= 400.:
        while t < next_show:
            
#            tfile.writelines(str(max(rho))+'\t'+str(max(P))+'\t'+str(t)+'\n')
            rho0 = np.copy(rho)
            rhov0 = np.copy(rhov)
            rhoe0 = np.copy(rhoe)
            P0 = np.copy(P)
            cs = np.sqrt(gamma*P0/rho0)
            mmin = max(cs-rhov0/rho0)
            mmax = max(cs+rhov0/rho0)
            dt = CFL*dx/max(mmin,mmax)

        

# Para usar [:] las variables siempre tienen que estar definidas anteriormente
            rho1 = LWRS1(rho0,rhov0,x,dt,dx)
# CUIDADO!! Antes petaba y se desestabilizaba porque no se estaban copiando los valores y el rhov[:] cogia en realidad el rho[:] y como aun no habias establecido las condiciones iniciales, petaba. Por eso has puesto np.copy() antes
            rhov1 = LWRS1(rhov0,(rhov0**2/rho0)+P0,x,dt,dx)
            rhoe1 = LWRS1(rhoe0,(rhoe0+P0)*rhov0/rho0,x,dt,dx)


            v1 = rhov1/rho1
            P1 = (rhoe1 - 0.5*rhov1*v1)*(gamma-1.)
 

            frhov1 = (rhov1**2/rho1)+P1
            frhoe1 = (rhoe1+P1)*rhov1/rho1
            rho0[1:-1] = rho0[1:-1] - (dt/dx)*(rhov1[1:-1]-rhov1[0:-2])
            rhov0[1:-1] = rhov0[1:-1] - (dt/dx)*(frhov1[1:-1]-frhov1[0:-2])
            rhoe0[1:-1] = rhoe0[1:-1] - (dt/dx)*(frhoe1[1:-1]-frhoe1[0:-2])


  
# =========================================================================
# ============== SOLVING POISSON EQUATION =================================
# =========================================================================

            wf = np.fft.fftfreq(x[1:-1].shape[-1])/dx
            d2phif = np.fft.fft(rho0[1:-1])


            phif = d2phif/(2*np.pi*1j*wf)**2
            phif[wf == 0] = 0.

            phi_r = np.fft.ifft(phif).real
            phi_r = np.append(phi_r[-1],phi_r)
            phi_r = np.append(phi_r,phi_r[1])

# =========================================================================
# ============== DERIVATE PHI(X) TO OBTAIN g ==============================
# =========================================================================

            g = centered(-phi_r,x,dx)
            g[0] = g[-2]
            g[-1] = g[1]

            frhog = rho0[1:-1]*g[1:-1]
            frhovg = rhov0[1:-1]*g[1:-1]

# =========================================================================
# ============= INTEGRATION g VIA EULER ===================================
# =========================================================================

            rho = rho0
            rhov[1:-1] = rhov0[1:-1] + dt*frhog
            rhoe[1:-1] = rhoe0[1:-1] + dt*frhovg

            rho[0] = rho[-2]
            rho[-1] = rho[1]
            rhov[0] = rhov[-2]
            rhov[-1] = rhov[1]
            rhoe[0] = rhoe[-2]
            rhoe[-1] = rhoe[1]

            v = rhov/rho
            P = (rhoe - 0.5*rhov*v)*(gamma-1.)

# ====================================================
# ============ ANALISIS LAGRANGIANO ==================
# ====================================================
            xrho = xrho + vrho*dt
            vrho = np.interp(xrho,x,v)
        
            t += dt
    
        next_show = t+ht_show
        print(max(rho))
        line[0].set_data(x,(rho-rho00)/rho00)
        line[1].set_data(x,g)
#    line[1].set_data(xrho,rho[iprho])
#    line[1].set_data(x,(rhoan-rho00)/rho00)
#    line[2].set_data(x,v)
        line[2].set_data(x,v)
        line[3].set_data(xrho,vrho)
#    line[4].set_data(x,(P-P00)/P00)
        line[4].set_data(x,P)
        line[5].set_data(xrho,P00*np.ones(len(xrho)))
        return line

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True,repeat = False)
ani.save('lagrange.gif',writer='imagemagick',fps=100)
#plt.show()

#tfile.close()

