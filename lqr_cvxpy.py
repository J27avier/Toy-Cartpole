import cvxpy as cvx
import numpy as np
# Where do I get the control law?
# Could be used for MPC?

tau = 0.02

A = np.matrix(
    [[0.00000, 1.00000,  0.00000, 0.00000],
     [0.00000, 0.00000, -0.71707, 0.00000],
     [0.00000, 0.00000,  0.00000, 1.00000],
     [0.00000, 0.00000, 15.77561, 0.00000],])

B = np.matrix(
    [[0.00000 ],
     [0.97561 ],
     [0.00000 ],
     [-1.46341],])

# C and D are default
C = np.matrix([[1, 0, 0, 0]]);
D = np.matrix([[0]]);

#Q
Q = np.matrix(
    [[5, 0,   0, 0], 
     [0, 1,   0, 0],
     [0, 0,  10, 0],
     [0, 0,   0, 1],])

R = np.matrix([[1]])

# Ref: https://stanfordasl.github.io/aa203/pdfs/lecture/recitation_1.pdf
n = 4
m = 1
T = 100
u_bound = 10

x_0 = np.random.normal(0, 0.05, (n,1))
X = {}
U = {}
K = cvx.Variable((1,n))
cost_terms = []
constraints = []

for t in range(T):
    X[t] = cvx.Variable((n,1))
    U[t] = cvx.Variable((m,1))
    cost_terms.append(cvx.quad_form(X[t], Q))
    cost_terms.append(cvx.quad_form(U[t], R))
    constraints.append(cvx.norm(U[t], "inf") <= u_bound)

    if (t == 0): # Initial condition
        constraints.append(X[t] == x_0) 
    else: # Dynamics constraint
        constraints.append((A @ X[t-1] + B @ U[t])*tau + X[t-1]== X[t])

objective = cvx.Minimize(cvx.sum(cost_terms))
prob = cvx.Problem(objective, constraints)
prob.solve()

print(prob.status)
print(prob.value)
print(U[0].value)

# Trying to get back the K, but failing
m_U = np.matrix([elem_U.value[:,0] for elem_U in U.values()])
m_X = np.matrix([elem_X.value[:,0] for elem_X in X.values()])
K = np.linalg.lstsq(m_X, m_U, rcond= None)[0];
print(K.T)
# Answer
#k = [-2.2361, -3.5987, -36.9680, -9.4652];
