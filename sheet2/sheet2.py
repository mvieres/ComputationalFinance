import numpy as np
import matplotlib.pyplot as plt


from functions import fd_forward_heat, fd_backward_heat, real_sol, get_max_error

Nx = 75
Nt = 108

# Paramteres
T = 0.1
L = np.pi
deltax = (L-0)/Nx

t = np.linspace(0,T,Nt+1)
x = np.linspace(0,L,Nx+1)
u_disc = np.zeros(shape=(Nx,1))

for i in range(Nx):
    u_disc[i] = real_sol(0.1,np.pi*i/Nx)

u1 = fd_forward_heat(Nx= Nx, Nt= Nt, T = T, L = L, sigma= 1)
max_error_u1 = get_max_error(delta_x=deltax,u_approx= u1[:,-1],L=L)
print('Max error of explicit scheme (Nt = 108): ', max_error_u1)

u2 = fd_backward_heat(Nx, Nt, T, L, sigma = 1)
max_error_u2 = get_max_error(delta_x=deltax,u_approx= u2[:,-1],L=L)
print('Max error of implicit scheme (Nt = 108): ',max_error_u2)

ax = plt.figure(1)

y, z = np.meshgrid(x,t)

ax = plt.axes(projection='3d')
ax.contour3D(y, z, u1, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation (Nt = 108); error: ' +  str(max_error_u1))
plt.show()

ax = plt.figure(2)
t = np.linspace(0,T,Nt)
x = np.linspace(0,L,Nx)
y, z = np.meshgrid(t,x)

ax = plt.axes(projection='3d')
ax.contour3D(z, y, u2, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('Implicit scheme for heat equation (Nt = 108); error:' +  str(max_error_u2))
plt.show()


Nx = 75
Nt = 150

t = np.linspace(0,T,Nt+1)
x = np.linspace(0,L,Nx+1)
u_disc = np.zeros(shape=(Nx,1))

for i in range(Nx):
    u_disc[i] = real_sol(0.1,np.pi*i/Nx)

u3 = fd_forward_heat(Nx= Nx, Nt= Nt, T = T, L = L, sigma= 1)
max_error_u3 = get_max_error(delta_x=deltax,u_approx= u3[:,-1],L=L)
print('Max error of explicit scheme (Nt = 150): ', max_error_u3)

u4 = fd_backward_heat(Nx, Nt, T, L, sigma = 1)
max_error_u4 = get_max_error(delta_x=deltax,u_approx= u4[:,-1],L=L)
print('Max error of implicit scheme (Nt = 150): ', max_error_u4)

ax = plt.figure(3)
y, z = np.meshgrid(x,t)

ax = plt.axes(projection='3d')
ax.contour3D(y, z, u3, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation (Nt = 150); error: '  +  str(max_error_u3))
plt.show()


ax = plt.figure(4)
t = np.linspace(0,T,Nt)
x = np.linspace(0,L,Nx)
y, z = np.meshgrid(t,x)

ax = plt.axes(projection='3d')
ax.contour3D(z, y, u4, 500)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u1')
ax.set_title('Explicit scheme for heat equation (Nt = 150); error: ' +  str(max_error_u1))
plt.show()