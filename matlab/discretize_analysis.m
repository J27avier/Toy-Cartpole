%pkg load symbolic;
%pkg load control;

clc 

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

tau = 0.02;

% According to http://eceweb1.rutgers.edu/~gajic/solmanual/slides/chapter8_DIS.pdf
A_d = exp(A.*tau);

%Fls_B = @(x) exp(A.*x);
ls_B = zeros(size(A));

step_tau = tau/100000;
for s=0:step_tau:tau
    ls_B = ls_B + exp(A.*s)*step_tau;
end%for

B_d = ls_B * B;

disp(A_d);
disp(B_d);

syscont = ss(A,B,C,D);
sysdisc = c2d(syscont, tau);
sysdisc.A
