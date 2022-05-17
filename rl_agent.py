import numpy as np
import math

class ActorCritic:
    def __init__(self):
        self.phi = np.ones((1,1))
        self.w = np.zeros((1,1))
        self.alpha_phi = 0.01
        self.alpha_w = 0.01
        self.I = 1
        self.gamma = 0.9
        self.mu=0
        self.delta = 0
        self.sigma=0.01
        self.action = 0
        self.reward = 0

        #self.Q = np.matrix(
        #    [[5, 0,   0, 0], 
        #     [0, 1,   0, 0],
        #     [0, 0,  10, 0],
        #     [0, 0,   0, 1],])

        self.Q = np.matrix([[10]])
        self.R = np.matrix([[0.00000000000001]])

        #self.actions = np.array(range(-10,11,2))
        #self.pi = np.ones((len(self.actions, 1)))/len(self.actions)

        self.x_past = np.zeros((1,1))
        self.x = np.zeros((1,1))

    def _print_AC(self):
        print('---- I', self.I)
        print('---- delta', self.delta)
        print('---- action', self.action)
        print('---- reward', self.reward)
        print('---- x', self.x)
        print('---- x_past', self.x_past)
        print('---- phi', self.phi)
        print('---- w', self.w)
        
        

    def _perception(self, observation):
        #self.x = np.matrix(observation.reshape((4,1)))
        self.x = np.matrix(observation[2])

    def _policy(self):
        #a_res = np.array(range(5))
        #a_probs = np.exp(a_res) / np.sum(np.exp(a_res))
        #action = np.random.choice(a_res, p = a_probs)
        #print(a_res)
        #print(a_probs)
        #print(action)
        
        #h = self.phi.T * self.x
        #self.pi = np.exp(h) / np.sum(np.exp(x))
        #self.action = np.random.choice(self.action, p= pi)

        #self.mu = np.clip(self.phi_mu.T @ self.x, -10, 10)
        #self.sigma = np.clip(np.exp(self.phi_sigma.T @ self.x), 0.00001, 10)

        self.action = np.clip((self.phi.T @ self.x)[0,0], -10, 10)
        self.action = 15 * self.x[0,0]
        return self.action

    def _reward(self):
        self.reward = -(self.x.T @ self.Q @ self.x + np.multiply(self.action**2, self.R))
        return self.reward

    def step(self, observation):
        self._perception(observation)
        self.action = self._policy()       
        self.reward = self._reward()
        self.delta = (self.reward + self.gamma + self.w.T @ self.x - self.w.T @ self.x_past)[0,0]
        self.w = self.w + np.multiply(self.alpha_w * self.delta, self.x)
        #self.phi = self.phi + np.multiply(self.alpha_phi * self.I * self.delta, self.x)
        self.phi = self.phi + np.multiply(self.alpha_phi, self.x)
        self._print_AC()
        self.x_past = self.x.copy()
        #self.I = self.gamma * self.I

        return self.action 



        
