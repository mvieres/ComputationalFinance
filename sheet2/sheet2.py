import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

from functions import fd_forward_heat, fd_backward_heat, real_sol

Nx = 75
Nt = 150
T = 0.1
L = np.pi
t = np.linspace(0,T,Nt+1)
x = np.linspace(0,L,Nx+1)
u_disc = np.zeros(shape=(Nx,1))

for i in range(Nx):
    u_disc[i] = real_sol(0.1,np.pi*i/Nx)

u1 = fd_forward_heat(Nx= Nx, Nt= Nt, T = T, L = L, sigma= 1)
max_error =np.ndarray.max(abs(u_disc - u1[:,-1]))
print(max_error)

u2 = fd_backward_heat(Nx, Nt, T, L, sigma = 1)


y, z = np.meshgrid(x,t)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(y, z, u1, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation')
#plt.show()

t = np.linspace(0,T,Nt)
x = np.linspace(0,L,Nx)
y, z = np.meshgrid(t,x)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(z, y, u2, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('Implicit scheme for heat equation')
plt.show()
