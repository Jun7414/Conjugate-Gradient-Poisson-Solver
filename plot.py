import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##################################################
# Change N_ln according to main.cpp for plotting #
##################################################

N_ln   = 128   # number of computing cell in 1D  


Lx   = 1.0   # x computational domain size
Ly   = 1.0   # y computational domain size
ghost = 1   # ghost zones
N = N_ln + 2*ghost #  total number of cells including ghost zones
D   = 1.0   # diffusion coefficient
u0  = 1.0   # background density
amp = 1.0   # sinusoidal amplitude
# derived constants
dx = Lx/N_ln                # spatial resolution
dy = Ly/N_ln
dt = dx*dy/(4*D)      # CLF stability

x   = np.linspace(dx/2, Lx-dx/2, N_ln)
y   = np.linspace(dy/2, Ly-dy/2, N_ln)
X , Y = np.meshgrid(x,y)

f = open('output.txt','r')
data = f.read().split()
f.close()
coor = np.loadtxt(data)
U = np.empty((N,N))
for i in range(N):
    U[i,:] = coor[N*i:N*(i+1)]
print(np.shape(U))

# plot
fig = plt.figure(figsize=(10,8), dpi=100)
ax = fig.add_subplot(111)
#ax.set_title('Initial Condition',fontsize=16)
plt.pcolormesh(X, Y, U[ghost : -ghost,ghost : -ghost], cmap='viridis', edgecolor='none')
plt.colorbar()
ax.set_xlabel( 'X',fontsize=14 )
ax.set_ylabel( 'Y',fontsize=14 )
plt.show()
