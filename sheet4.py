import matplotlib.pyplot as plt
import time
import numpy as np
from functions import Market, European_Options, Monte_Carlo

s0 = 5
K = 6
r = 0.05
sigma = 0.3

T = 1
n = 10

Anzahl_j = 3

figure, axis = plt.subplots(3)

results = np.zeros(shape=(Anzahl_j,5))
for j in range(2,5):
    N = 10**j
    start = time.time()
    M = Market(T=T,N=N,n=n,s0=s0,r=r,sigma=sigma)
    S = M.black_scholes()
    options = European_Options(n=n,N=N,K=K,Assetprice=S)
    Asian_call = options.Arithmetic_asian_call()
    p_MC, var_mc, ki = Monte_Carlo(N=N,rv=Asian_call,alpha=0.05,r=r,T=T,K=K).Standard_MC()
    end = time.time()
    results[j-2,4] = end - start
    results[j-2,0] = p_MC
    results[j-2,1] = var_mc
    results[j-2,2] = ki[0]
    results[j-2,3] = ki[1]

print(results)
# Sieht falsch aus
axis[0].plot(results[:,4])



for j in range(2,5):
    N = 10**j
    start = time.time()
    M = Market(T=T,N=N,n=n,s0=s0,r=r,sigma=sigma)
    S = M.black_scholes()
    options = European_Options(n=n,N=N,K=K,Assetprice=S)
    Asian_call = options.Arithmetic_asian_call()
    p_MC, var_mc, ki = Monte_Carlo(N=N,rv=Asian_call,alpha=0.05,r=r,T=T,K=K).Anti_thetic_MC(env=M)
    end = time.time()
    results[j-2,4] = end - start
    results[j-2,0] = p_MC
    results[j-2,1] = var_mc
    results[j-2,2] = ki[0]
    results[j-2,3] = ki[1]



print(results)
axis[1].plot(results[:,4])



plt.show()
