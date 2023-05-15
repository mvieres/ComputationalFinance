# TEsting Monte carlo class

import numpy as np

from functions import Monte_Carlo


N = 10000
X_rel = np.random.normal(loc = 4, scale = 10, size=(N,))
p_mc, var_mc, ki = Monte_Carlo(N=N,rv=X_rel,alpha=0.05,r=0.04,T=1).SMC()

print(p_mc)
print(var_mc)