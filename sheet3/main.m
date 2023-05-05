% Sheet 3 main
Nt = 100;
Ns = 100;
T = 1;
r = 0.03;
K = 110;
sigma = 0.2;
L = 200;
delta_t = T/Nt;
delta_s = L/Ns;

X = crank_nicolson(Nt,Ns,T,L,sigma,r,K);

[Y, Z] = meshgrid(0:delta_t:T, 0:delta_s:L);
surfc(Y,Z,X)
