pkg load symbolic;
pkg load control;

syms x x_dot theta theta_dot force;
% Kinematic equations to state space
accelerations = calc_acc(x, x_dot, theta, theta_dot, force);
f = [x_dot; accelerations(1); theta_dot; accelerations(2)];

%Calculates the partial derivatives
A_sym = jacobian(f, [x; x_dot; theta; theta_dot])
B_sym = jacobian(f, force)

%Linearize 
A = double(subs(A_sym, {x, x_dot, theta, theta_dot, force},{0,0,0,0, 0}));
B = double(subs(B_sym, {x, x_dot, theta, theta_dot, force},{0,0,0,0, 0}));

%A
disp("A");
disp(A);

%B
disp("---");
disp("B");
disp(B)

%Q
Q = [5 0  0 0; 
     0 1 0 0;
     0 0  10 0;
     0 0  0 1];

R = 1;

k = lqr(A, B, Q, R)
