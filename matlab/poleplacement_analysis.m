pkg load symbolic;
pkg load control;

A = [ 0.00000, 1.00000,  0.00000, 0.00000;
      0.00000, 0.00000, -0.71707, 0.00000;
      0.00000, 0.00000,  0.00000, 1.00000;
      0.00000, 0.00000, 15.77561, 0.00000;];

B = [ 0.00000;
      0.97561;
      0.00000;
     -1.46341;];

% C and D are default
C = [1 0 0 0];
D = 0;

%Create state space object
sys = ss(A,B,C,D);

%Check open loop eigenvalues
E = eig(A);
disp(E);
disp("------")

%Pole placement for lqr
k_lqr = [-3.1623, -12.8976, -113.5871, -43.193];
Acl_lqr = A -  B*k_lqr;
syslqr = ss(Acl_lqr, B, C, D);
disp(eig(Acl_lqr))

%Desired closed loop eigenvalues
P = [-30, -1+i, -1-i, -0.5];
K = place(A, B, P)

%K using pole placement
Acl = A -  B*K;
disp(eig(Acl));

%Create close loop system
syscl = ss(Acl, B, C, D, 0.02);

%Check step response
%t = 0:0.02:1
%step(syslqr, t); %!!! Hangs
