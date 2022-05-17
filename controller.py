import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

class PID:
    def __init__(self, kp, ki, kd):
        self.tau = 0.02
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err_past = 0.0
        self.err_integral = 0.0
        self.err_der = 0.0
        self.u = 0

    def step(self, observation): #Assuming steady time step
        err = observation[2]
        self.err_der = (err - self.err_past)/ self.tau
        self.err_integral = self.err_integral + self.tau * err 

        self.u = self.kp * err + self.ki * self.err_integral + self.kd * self.err_der
        self.err_past = err
        return self.u


class PolePlacement:
    def __init__(self, k = [-3.1623, -12.8976, -113.5871, -43.193], ref = [0, 0, 0, 0]):
        self.tau = 0.02
        self.k = np.array([k])
        self.r = np.array([ref]).T # Reference position
        self.u = 0
    def step(self, observation):
        error = self.r - observation.reshape((4,1));
        self.u = np.dot(self.k, error);
        return self.u[0, 0];

class MPC: # Not really working
    def __init__(self, ref=[0,0,0,0]):
        self.tau = 0.02
        self.U_past = 0

        self.A = np.matrix(
            [[0.00000, 1.00000,  0.00000, 0.00000],
             [0.00000, 0.00000, -0.71707, 0.00000],
             [0.00000, 0.00000,  0.00000, 1.00000],
             [0.00000, 0.00000, 15.77561, 0.00000],])

        self.B = np.matrix(
            [[0.00000 ],
             [0.97561 ],
             [0.00000 ],
             [-1.46341],])
    
        self.A_d = np.matrix(
            [[1, 0.02, -0.0001435, -9.564e-07],
             [0,    1,   -0.01436, -0.0001435],
             [0,    0,      1.003,    0.02002],
             [0,    0,     0.3158,      1.003],])

        self.B_d = np.matrix(
            [[ 0.0001951],
             [   0.01951],
             [-0.0002928],
             [   -0.0293],])


        # C and D are default
        self.C = np.matrix([[1, 0, 0, 0]]);
        self.D = np.matrix([[0]]);

        #Q
        self.Q = np.matrix(
            [[5, 0,   0, 0], 
             [0, 1,   0, 0],
             [0, 0,  10, 0],
             [0, 0,   0, 1],])

        self.R = np.matrix([[0.1]])

        # Ref: https://stanfordasl.github.io/aa203/pdfs/lecture/recitation_1.pdf
        self.n = 4
        self.m = 1
        self.T = 40
        self.u_bound = 10

    def step(self, observation):
        x0 = np.asmatrix(observation.reshape((4,1)))
        X = {}
        U = {}
        #K = cvx.Variable((1,self.n))
        cost_terms = []
        constraints = []

        for t in range(self.T):
            X[t] = cvx.Variable((self.n,1))
            U[t] = cvx.Variable((self.m,1))
            cost_terms.append(cvx.quad_form(X[t], self.Q))
            cost_terms.append(cvx.quad_form(U[t], self.R))
            constraints.append(cvx.norm(U[t], "inf") <= self.u_bound)

            if (t == 0): # Initial condition
                constraints.append(X[t] == x0) 
            elif (0 < t < self.T-1): # Dynamics constraint
                constraints.append(self.A_d @ X[t-1] + self.B_d @ U[t-1]== X[t])

        objective = cvx.Minimize(cvx.sum(cost_terms))
        prob = cvx.Problem(objective, constraints)
        prob.solve()

        if U[0].value == None:
            print("U not available!!!")
            output = 0
        else:
            output = U[0].value[0,0]
            self.U_past = output
        return output 

class Model:
    def __init__(self, observation):
        self.A_d = np.matrix(
            [[1, 0.02, -0.0001435, -9.564e-07],
             [0,    1,   -0.01436, -0.0001435],
             [0,    0,      1.003,    0.02002],
             [0,    0,     0.3158,      1.003],])

        self.B_d = np.matrix(
            [[ 0.0001951],
             [   0.01951],
             [-0.0002928],
             [   -0.0293],])


        # C and D are default
        self.C = np.matrix([[1, 0, 0, 0]]);
        self.D = np.matrix([[0]]);
        self.x_past = np.matrix(observation.reshape(4,1))

    def predict(self, action):
        x = self.A_d @ self.x_past + self.B_d * action
        self.x_past = x
        return x 


def do_graph(t, l_action, l_observation, ref, force_mag, l_prediction):
    fig1 = plt.figure(figsize = (8, 6))
    ax1 = fig1.add_subplot(3,2,1)
    ax2 = fig1.add_subplot(3,2,2)
    ax3 = fig1.add_subplot(3,2,3)
    ax4 = fig1.add_subplot(3,2,4)
    ax5 = fig1.add_subplot(3,2,5)

    ax1.plot(t, l_observation[:,0])
    ax2.plot(t, l_observation[:,1])
    ax3.plot(t, l_observation[:,2])
    ax4.plot(t, l_observation[:,3])
    ax5.plot(t, l_action, color = 'red')

    if l_prediction is not None:
        ax1.plot(t, l_prediction[:,0])
        ax2.plot(t, l_prediction[:,1])
        ax3.plot(t, l_prediction[:,2])
        ax4.plot(t, l_prediction[:,3])

    ax1.axhline(ref[0], color = 'k', alpha = 0.5, linestyle='--')
    ax2.axhline(ref[1], color = 'k', alpha = 0.5, linestyle='--')
    ax3.axhline(ref[2], color = 'k', alpha = 0.5, linestyle='--')
    ax4.axhline(ref[3], color = 'k', alpha = 0.5, linestyle='--')
    ax5.axhline( force_mag, color = 'purple', alpha = 0.5, linestyle='--')
    ax5.axhline(-force_mag, color = 'purple', alpha = 0.5, linestyle='--')

    ax1.set_xlabel("t")
    ax2.set_xlabel("t")
    ax3.set_xlabel("t")
    ax4.set_xlabel("t")
    ax5.set_xlabel("t")
     
    ax1.set_ylabel("x")
    ax2.set_ylabel("x_dot")
    ax3.set_ylabel("theta")
    ax4.set_ylabel("theta_dot")
    ax5.set_ylabel("u")

    fig1.tight_layout()

    fig1.show()
