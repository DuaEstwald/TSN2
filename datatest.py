import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib as mpl


data = sorted(glob.glob('rhofix/rho*.txt'))

n = len(data)
norm = mpl.colors.Normalize(vmin=0.58, vmax=0.62)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap = mpl.cm.coolwarm)
cmap.set_array([])
c = np.arange(1,n+1)
P0=np.array([0.58,0.5850,0.59,0.5920,0.595,0.5970,0.598,0.599,0.5991,0.5992,0.5993,0.5994,0.5995,0.5996,0.5997,0.5998,0.5999,0.60,0.6001,0.6002,0.6003,0.6004,0.6005,0.6006,0.6007,0.6008,0.6009,0.601,0.603,0.605,0.607,0.608,0.609,0.61,0.615,0.619,0.62])

#rho0 = np.array([0.98,0.99,0.995,0.998,0.999,1.0,1.001,1.003,1.005,1.007,1.008,1.009,1.01,1.015,1.019,1.02])
#c = np.arange(0.98,1.2)
#fig, ax = plt.subplots(dpi=100)
#ax = plt.axes()
#ax.set_prop_cycle('color',[plt.cm.RdBu(i) for i in np.linspace(0, 1, n)])
#sm = plt.cm.ScalarMappable(cmap='RdBu', norm=1.0)
for i in range(n):
    rho,P,t = np.loadtxt(data[i],unpack=True)
    plt.plot(t,rho, c = cmap.to_rgba(P0[i]))
plt.colorbar(cmap)
plt.grid()


rho,P,t = np.loadtxt('rhofix/rho1.0P0.6000.txt',unpack=True)
plt.plot(t,rho,':k')
plt.ylabel(r'$max(\rho(t))$')
plt.xlabel('t')

plt.savefig('jeanspreassure.png')



