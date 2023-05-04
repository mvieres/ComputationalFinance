function X = crank_nicolson(Nt,Ns,T,L,sigma,r,K)
% SYNTAX: 
% Performs Crank Nicolson Scheme for Black Scholes PDE with constant sigma
% and r
%
% Input: 
%   Nt --> Number of discretization points in Time (Integer)
%   Ns --> Number of discretitation points in State space (Integer)
%   T --> Time Horizon (double)
%   L --> Space Horizon (double)
%   sigma --> Varianze from BS Model (double)
%   r --> Risk free interest rate of BS Model (double)
%   K --> Strike Price (double)
%
% Output:
%   X --> Nunerical Function values from Crank Nicolson (Matrix)


% European put
f = @(x) max(1,K-x);

% Discretization
delta_t = (T-0)/Nt;
delta_s = (L-0)/Ns;

% Preallocation for ALL points
X = zeros(Ns+1,Nt+1);
% Preallocation for INNER Points:
U = zeros(Ns-1,Nt+1);

% Boundary conditions
% for first Time Point:
X(2:Ns,1) = arrayfun(f,(1:(Ns-1))*delta_s);
% for boundaries at space = 0 and space = L

% 
U(:,1) = X(2:Ns,1);

% Matricies
xi = 1:(Ns-1);
a = -0.5*((sigma^2)*xi.^2 - r.*xi)*delta_t;
b = (r + (sigma^2)*xi.^2)*delta_t;
c = -0.5*((sigma^2)*xi.^2 + r*xi)*delta_t;

M = zeros(Ns-1,Ns-1);
for j = 1:(Ns-1)
    M(j,j) = b(j);
end
for j = 2:(Ns-1)
    M(j,j-1) = a(j);
    M(j-1,j) = c(j-1);
end

F = eye(Ns-1) - 0.5*M;
B = eye(Ns-1) + 0.5*M;
[L, R] = lu(B);

for i = 1:Nt
    r_i = 0.5*[-a(1)*(U(1,i)+U(1,i+1)) zeros(1, Ns-3) -c(Ns-1)*(U(end,i+1))];
    y = L\(F*U(:,i)+r_i');
    U(:,i+1) = R\y;
end
% TODO: Randbedingungen für s=0 und s = L;

% Saving inner Points
X(2:Ns,2:end) = U(:,2:end);
end
