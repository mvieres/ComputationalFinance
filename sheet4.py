import matplotlib.pyplot as plt
from functions import Market, European_Options, Monte_Carlo

s0 = 5
K = 6
r = 0.05
sigma = 0.3
mu = 0.4
T = 1
n = 10
N = 10**6

M = Market(T=T,N=N,n=n,s0=s0,r=r,sigma=sigma,mu=mu)
S = M.black_scholes()
options = European_Options(n=n,N=N,K=K,Assetprice=S)
Asian_call = options.Arithmetic_asian_call()

p_MC, var_mc, ki = Monte_Carlo(N=N,option_value=Asian_call,alpha=0.05,r=r,T=T).Standard_MC()
print(p_MC,var_mc,ki)
plt.plot(Asian_call)
plt.show()
