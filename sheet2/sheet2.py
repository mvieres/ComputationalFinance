import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

def boundary_cond(x,L):
    if x<=(L/2):
        y = 2*x
    else:
        y = 2*(L - x)
    return y

 

def fd_forward_heat(Nx, Nt, T, L, sigma):
    """Computes the forward finite differences approach for u_t = sigma^2 * u_xx

    Args:
        Nx (int): Discretization points for space
        Nt (int): Discretization points for time
        T (float): Time horizon
        L (float): Space horizon
        sigma (float): sigma coeff for heat equation
    """
    # Parameters
    delta_t = T / Nt
    delta_x = L / Nx
    nu = (sigma**2)*(delta_t/(delta_x**2))
    u = np.zeros((Nt+1, Nx+1))
    
    # Boundary conditions (E2)
    for j in range(Nx+1):
        #u[0,j] = boundary_cond(j*delta_x, L)
        u[0,j] = np.sin(j*delta_x)
       
    for i in range(Nt+1):
        u[i,0] = 0
        u[i,-1] = 0
        
    # Recursion for inner points
    for i in range(Nt):
        for j in range(1,Nx):
            u[i+1, j] = nu*u[i, j-1] + (1-2*nu)*u[i, j] + nu*u[i, j+1]
        
    return u

def fd_backward_heat(Nx, Nt, T, L, sigma):
    """performs finite difference scheme for heat eqaution u_t = sigma^2 * u_xx

    Args:
        Nx (int): Discretization points for space
        Nt (int): Discretization points for time
        T (float): Time horizon
        L (float): Space horizon
        sigma (float): sigma coeff for heat equation
    """
    delta_t = T / Nt
    delta_x = L / Nx
    nu = (sigma**2)*(delta_t/(delta_x**2))
    u = np.zeros((Nx, Nt))
    
    # Matrix B allocation
    B = np.zeros((Nx, Nx))
    for i in range(Nx):
        B[i,i] = 1-2*nu
        
    for i in range(Nx-1):
        B[i+1,i] = nu
        B[i,i+1] = nu
    
    # L U Decomposition, Jetzt "R-U Zerlegung", da L obere Grenze vom State space ist
    U, R = lu(B,permute_l=True)
    U_inv = np.linalg.inv(U)
    R_inv = np.linalg.inv(R)
    
    # 
    qi = nu*np.zeros(shape=(1,Nt))
    
    # Allocate U_0
    for j in range(1, Nx):
        u[0,j] = boundary_cond(j*delta_x,L)
        
    # Rekursion
    for i in range(Nt-1):  
        y = U_inv.dot(u[:,i])
        u[:,i+1] = R_inv.dot(y)
    return u
    
       


Nx = 14
Nt = 94
T = 3
L = np.pi
t = np.linspace(0,T,Nt+1)
x = np.linspace(0,L,Nx+1)
u1 = fd_forward_heat(Nx= Nx, Nt= Nt, T = T, L = L, sigma= 1)

u2 = fd_backward_heat(Nx, Nt, T, L, sigma = 1)


y, z = np.meshgrid(x,t)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(y, z, u1, 1000)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u1')
ax.set_title('Heat equation')
plt.show()


t = np.linspace(0,T,Nt)
x = np.linspace(0,L,Nx)
y, z = np.meshgrid(x,t)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(y, z, u2, 1000)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u2')
ax.set_title('Heat equation')
plt.show()