% Sheet 3 main
Nt = 100;
Ns = 100;
T = 1;
r = 0.03;
K = 110;
sigma = 0.2;
L = 200;

X = crank_nicolson(Nt,Ns,T,L,sigma,r,K);