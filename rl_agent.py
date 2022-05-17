import numpy as np
import math
import refs.tiles3 as tc
import itertools

class CartPoleTileCoder:
    def __init__(self, iht_size = 4096, num_tilings = 32, num_tiles = 8):
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tc.IHT(iht_size)
    def get_tiles(self, observation):
        observation_scaled = np.zeros((4,)) 
        observation_scaled = np.divide(observation.reshape(4,), 0.25*np.array([4, 8, np.pi/2, np.pi])) + 1/2

        tiles = tc.tileswrap(self.iht, self.num_tilings, observation_scaled, wrapwidths = [False, False, False, False])
        return np.array(tiles)

def compute_softmax_prob(actor_w, tiles):
    state_action_preferences = []
    state_action_preferences = actor_w[:, tiles].sum(axis = 1)
    #print("State action preferences", state_action_preferences, np.shape(state_action_preferences))
    c = np.max(state_action_preferences)
    numerator = np.exp(state_action_preferences-c)
    denominator = np.sum(numerator)
    softmax_prob = np.divide(numerator, denominator)
    return softmax_prob

class ActorCritic:
    def __init__(self):
        self.rand_generator = None
        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None
        self.tc = None
        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None
        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_init(self):
        ######## 
        self.rand_generator = np.random.RandomState(1)

        iht_size = 2048
        self.num_tilings = 64
        self.num_tiles = 64

        self.tc = CartPoleTileCoder(iht_size=iht_size, num_tilings=self.num_tilings, num_tiles=self.num_tiles)

        self.actor_step_size = 1e-1/self.num_tiles
        self.critic_step_size = 1e-0/self.num_tiles
        self.avg_reward_step_size = 1e-2/self.num_tiles

        self.actions = [-10, -3, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 3, 10]
        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)
        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
        self.x = None


        self.Q = np.matrix(
            [[5, 0,   0, 0], 
             [0, 1,   0, 0],
             [0, 0,  10, 0],
             [0, 0,   0, 1],])

        self.R = np.matrix([[0]])

    def print_Agent(self):
        print("Reward", self.reward, np.shape(self.reward))
        print("Avg reward", self.avg_reward, np.shape(self.avg_reward))
        print("x", self.x, np.shape(self.x))
        print("Previous tiles", self.prev_tiles, np.shape(self.prev_tiles))
        print("Softmax prob", self.softmax_prob, np.shape(self.softmax_prob))
        print("Last action", self.last_action, np.shape(self.last_action))
        print("Actor_w", self.actor_w[:, self.prev_tiles].sum(axis=1))
        print("Critic_w", self.critic_w[self.prev_tiles].sum())


    def _perception(self, observation):
        self.x = np.matrix(observation.reshape((4,1)))

    def agent_policy(self, active_tiles):
        softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
        self.softmax_prob = softmax_prob
        return chosen_action

    def _reward(self):
        #self.reward = -(self.x.T @ self.Q @ self.x + np.multiply(self.last_action**2, self.R)).item() * 10
        self.reward = -self.x[2,0].item()**2 - 0.2*self.x[0,0].item()**2 + 0.2
        return self.reward

    def agent_start(self, observation):
        self._perception(observation)
        active_tiles = self.tc.get_tiles(np.asarray(self.x)[:,0])
        current_action = self.agent_policy(active_tiles)

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action

    def step(self, observation):
        self._perception(observation)
        active_tiles = self.tc.get_tiles(np.asarray(self.x)[:,0])
        reward = self._reward()
        #delta = reward + 0.99*self.critic_w[active_tiles].sum() - self.critic_w[self.prev_tiles].sum()
        delta = reward - self.avg_reward + \
                self.critic_w[active_tiles.reshape((1, self.num_tilings))].sum() - \
                self.critic_w[self.prev_tiles.reshape((1, self.num_tilings))].sum()

        self.avg_reward += self.avg_reward_step_size * delta
        
        self.critic_w[self.prev_tiles.reshape((1,self.num_tilings))] += self.critic_step_size * delta
        
        for i, a in enumerate(self.actions):
            if a == self.last_action:
                self.actor_w[i][self.prev_tiles.reshape((1,self.num_tilings))] += self.actor_step_size * delta * (1 - self.softmax_prob[i])
            else:
                self.actor_w[i][self.prev_tiles.reshape((1,self.num_tilings))] += self.actor_step_size * delta * (0 - self.softmax_prob[i])

        current_action = self.agent_policy(active_tiles)
        self.prev_tiles = active_tiles
        self.last_action = current_action
        #self.print_Agent()

        self.actor_w = np.clip(self.actor_w, -10_000, 10_000)
        self.critic_w = np.clip(self.critic_w, -10_000, 10_000)
        return self.last_action 

if __name__ == '__main__':
    observation_0 = np.linspace(-2, 2, num = 3)
    observation_1 = np.linspace(-4, 4, num = 3)
    observation_2 = np.linspace(-np.pi/4, np.pi/4, num = 3)
    observation_3 = np.linspace(-np.pi/2, np.pi/2, num = 3)

    test_obs = list(itertools.product(observation_0, observation_1, observation_2, observation_3))
    pctc = CartPoleTileCoder(iht_size=4096, num_tilings=8, num_tiles=2)
    result = []
    for obs in test_obs:
        tiles = pctc.get_tiles(obs)
        print(tiles)
