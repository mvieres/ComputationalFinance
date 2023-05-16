import numpy as np
import matplotlib.pyplot as plt
from functions import Market

n = 1000
N = 5
r = 0.01
s0 = 1
T = 1
sigma = 0.5

M = Market(N=N, n=n, sigma=sigma, r=r, s0=s0, T=T)

S = M.black_scholes()
t = M.time_grid()
for j in range(N):
    plt.plot(t,S[j,:])

plt.show()