import numpy as np  
from scipy.linalg import lu

class Market():
    def __init__(self,n,N,sigma,mu,r,s0,T):
        self.n = n
        self.N = N
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.s0 = s0
        self.T = T
        
    def brownian_motion(self):
        """Computes #N Sample paths of brownian motion

        Args:
            n (_type_): _description_
            N (_type_): _description_
        """
        t = self.time_grid()
        delta_t = t[1] - t[0]
        dB = np.sqrt(delta_t) * np.random.normal(size=(self.N,self.n - 1))
        B0 = np.zeros(shape=(self.N,1))
        B = np.concatenate((B0,np.cumsum(dB,axis=1)),axis= 1)
        return B
    
    def black_scholes(self):
        t = self.time_grid()
        BB = self.brownian_motion()
        S = np.zeros(shape=(self.N,self.n))
        for j in range(self.N):
            for i in range(self.n):
                S[j,i] = self.s0*np.exp((self.mu - 0.5*(self.sigma**2))*t[i] + self.sigma*BB[j,i])
        return S
    
    def time_grid(self):
        time = np.linspace(0,self.T,self.n)
        return time


class European_Options():
    def __init__(self,n,N,K,Assetprice):
        self.K = K
        self.N = N
        self.n = n
        self.S = Assetprice
    
    def Arithmetic_asian_call(self):
    
        Value = np.zeros(shape = (self.N,))
        for j in range(self.N):
           Value[j] = np.max((1/self.n)* np.sum(self.S[j,:]) - self.K , 0) 
        return Value



# Monte Carlo Methods:
class Monte_Carlo(European_Options):
    def __init__(self, n, N, K, Assetprice):
        super().__init__(n, N, K, Assetprice)


def Standard_MC(N,Model,Option):
    market = Market()
    return



# Numerical Methods

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
        B[i,i] = 1+2*nu
        
    for i in range(Nx-2):
        B[i+1,i] = -nu
        B[i,i+1] = -nu

    # L U Decomposition, Jetzt "R-U Zerlegung", da L obere Grenze vom State space ist
    # Gleiche dim wie bei B
    U, R = lu(B,permute_l=True)
    U_inv = np.linalg.inv(U)
    R_inv = np.linalg.inv(R)
    

    
    # Allocate U_0
    for j in range(1, Nx):
        u[j,0] = boundary_cond(j*delta_x,L)
        #u[j,0] = np.sin(j*delta_x)
    # Randbedingungen (fuer t) sind hier egal, da diese konstant 0 sind und mit 0 initialisiert wird
        
    # Rekursion
    for i in range(Nt-1):  
        u_vec = u[1:,i]
        y = U_inv.dot(u_vec)
        u[1:,i+1] = R_inv.dot(y)
    return u
    
       
def real_sol(t, x):
    n = 10000
    u = 0

    for i in range(1,n+1):
        u = u + ((-1)**(i-1))*np.exp((-(2*i-1)**2)*t)*(np.sin((2*i-1)*x))*((2*i-1)**(-2))
    u = (8/np.pi)*u

    return u

def get_max_error(delta_x,u_approx,L):
    """
    get error at t = 0.1 for simple heat equation
    delta_t 
    """
    for j in range(len(u_approx)):
        u_t = real_sol(0.1,L*j/delta_x)
    diff = abs(u_t - u_approx)
    error = diff.max()
    return error
