import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

from functions import fd_forward_heat, fd_backward_heat, real_sol, get_max_error

Nx = 75
Nt = 108

T = 0.1
L = np.pi
deltax = (L-0)/Nx
t = np.linspace(0,T,Nt+1)
x = np.linspace(0,L,Nx+1)
u_disc = np.zeros(shape=(Nx,1))

for i in range(Nx):
    u_disc[i] = real_sol(0.1,np.pi*i/Nx)

u1 = fd_forward_heat(Nx= Nx, Nt= Nt, T = T, L = L, sigma= 1)
max_error = get_max_error(delta_x=deltax,u_approx= u1[:,-1],L=L)
print('Max error of explicit scheme (Nx = 108): ', max_error)

u2 = fd_backward_heat(Nx, Nt, T, L, sigma = 1)
max_error = get_max_error(delta_x=deltax,u_approx= u2[:,-1],L=L)
print('Max error of implicit scheme (Nx = 108): ', max_error)

figure, ax = plt.subplots(1,1)

y, z = np.meshgrid(x,t)

ax = plt.axes(projection='3d')
ax.contour3D(y, z, u1, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation')
plt.show()

figure, ax = plt.subplots(1,1)
t = np.linspace(0,T,Nt)
x = np.linspace(0,L,Nx)
y, z = np.meshgrid(t,x)

ax = plt.axes(projection='3d')
ax.contour3D(z, y, u2, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('Implicit scheme for heat equation')
plt.show()


Nx = 75
Nt = 150

t = np.linspace(0,T,Nt+1)
x = np.linspace(0,L,Nx+1)
u_disc = np.zeros(shape=(Nx,1))

for i in range(Nx):
    u_disc[i] = real_sol(0.1,np.pi*i/Nx)

u3 = fd_forward_heat(Nx= Nx, Nt= Nt, T = T, L = L, sigma= 1)
max_error =get_max_error(delta_x=deltax,u_approx= u3[:,-1],L=L)
print('Max error of explicit scheme (Nx = 150): ', max_error)

u4 = fd_backward_heat(Nx, Nt, T, L, sigma = 1)
max_error =get_max_error(delta_x=deltax,u_approx= u4[:,-1],L=L)
print('Max error of implicit scheme (Nx = 150): ', max_error)

figure, ax = plt.subplots(1,1)
y, z = np.meshgrid(x,t)

ax = plt.axes(projection='3d')
ax.contour3D(y, z, u3, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation')
plt.show()


figure, ax = plt.subplots(1,1)
t = np.linspace(0,T,Nt)
x = np.linspace(0,L,Nx)
y, z = np.meshgrid(t,x)

ax = plt.axes(projection='3d')
ax.contour3D(z, y, u4, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation')
plt.show()