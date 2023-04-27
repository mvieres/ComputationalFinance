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
        u[0,j] = boundary_cond(j*delta_x, L)
        #u[0,j] = np.sin(j*delta_x)
       
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
    # Loesungsvektor u: Lenght  = N_x - 1
    u = np.zeros((Nx, Nt))
    
    # Matrix B allocation mit dim (N_x, N_x)
    B = np.zeros((Nx-1, Nx-1))
    for i in range(Nx-1):
        B[i,i] = 1-2*nu
        
    for i in range(Nx-2):
        B[i+1,i] = nu
        B[i,i+1] = nu
    
    # L U Decomposition, Jetzt "R-U Zerlegung", da L obere Grenze vom State space ist
    # Gleiche dim wie bei B
    U, R = lu(B,permute_l=True)
    U_inv = np.linalg.inv(U)
    R_inv = np.linalg.inv(R)
    
    # qi soll laenge N_x - 1 haben
   
    
    # Allocate U_0
    for j in range(1, Nx):
        u[j,0] = boundary_cond(j*delta_x,L)
    # Andere randbedingungen (fuer t) sind hier egal, da diese konstant 0 sind und mit 0 initialisiert wird
        
    # Rekursion
    for i in range(Nt-1):  
        u_vec = u[1:,i]
        y = U_inv.dot(u_vec)
        u[1:,i+1] = R_inv.dot(y)
    return u
    
       
