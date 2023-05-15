import matplotlib.pyplot as plt
from functions import Market, European_Options

s0 = 5
K = 6
r = 0.05
sigma = 0.3
mu = 0.4
T = 1
n = 10
N = 10^2

M = Market(T=T,N=N,n=n,s0=s0,r=r,sigma=sigma,mu=mu)
S = M.black_scholes()
options = European_Options(n=n,N=N,K=K,Assetprice=S)

Asian_call = options.Arithmetic_asian_call()

plt.plot(Asian_call)
plt.show()