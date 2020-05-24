import numpy as np
import matplotlib.pyplot as plt


N = 50
xi = 0.
xf = 2.*np.pi
dx = (xf-xi)/N

A = 5.

x0 = np.arange(xi-dx/2.,xf+3.*dx/2.,dx) # Hay que ser consistente con los puntos en el array, tienen que ser periodicos y estar dentro del rango, por lo cual, no usaremos los dos puntos de los extremos de este array ya que estan fuera del rango.
x = x0[1:-1]

d2yr = 1.+A*np.cos(x)
#d2yr = np.sin(x)

yr = - A*np.cos(x)
#yr = -np.cos(x)

# Solving the poisson equation using fourier analysis. 

# Manual solve ways

j = complex(0.,1.)


w = np.fft.fftfreq(x.shape[-1])/dx  # Ten en cuenta que al calcular la frecuencia con fourier te devuelve un array sin dimensiones, es por esto que tienes que dividirlo entre el intervalo que estes utilizando. Esto puedes hacerlo tanto diviendo por dx como asignando el intervalo dentro de la funcion: np.fft.fftfreq(x.shape[-1],dx)


#d2yf = np.array([np.sum(d2yr*np.exp(-2*np.pi*j*np.arange(d2yr.shape[-1])*k/d2yr.shape[-1])) for k in range(d2yr.shape[-1])])

d2yf = np.fft.fft(d2yr)


yf = d2yf/(2*np.pi*j*w)**2
yf[w == 0] = 0.

#yr_f = np.array([np.sum(yf*np.exp(2*np.pi*j*np.arange(yf.shape[-1])*m/yf.shape[-1])) for m in range(yf.shape[-1])]).real

yr_f = np.fft.ifft(yf).real

plt.ion()

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(x,yr,'b',label='Solucion teorica')
ax1.plot(x,yr_f,'rx',label='Analisis de Fourier')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\phi(x)$')
ax1.grid()


ax2.plot(x,yr-yr_f,'gray')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\phi(x)_{theo} - \phi(x)_{Fourier}$')

ax2.grid()
plt.tight_layout()
plt.savefig('fourier.png')
