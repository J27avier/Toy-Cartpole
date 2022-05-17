import itertools
import gym
import numpy as np
from controller import PID, PolePlacement, MPC, do_graph, Model
from rl_agent import ActorCritic
import matplotlib.pyplot as plt

str_control = 'ss'
force_mag = 10.0
tau = 0.02
env = gym.make('CartPole-cont')
ref = [0,0,0,0]
#print("action_space", env.action_space)
#print("observation_space", env.observation_space)

control = ActorCritic()
control.agent_init()

l_ep_length = []

for i_episode in range(2_000):
    observation = env.reset()
    action = control.agent_start(observation)
    #model = Model(observation)

    l_observation = np.empty((0,4)) 
    l_prediction = np.empty((0,4))
    l_action = np.array([])
    i = 0
    #control.actor_step_size *= 0.999
    #control.critic_step_size *= 0.999
    #control.avg_reward_step_size *= 0.999

    for t in range(400):
        i+= 1
        if i_episode > 1995:
            env.render()
        #print(f"{i_episode}, {t}: {observation}")

        action = np.clip(control.step(observation), -force_mag, force_mag)

        l_observation = np.append(l_observation, np.array([observation]), axis = 0)
        #l_prediction = np.append(l_prediction, model.predict(action)[:,0].T, axis = 0)
        l_action = np.append(l_action, action)
        observation, reward, done, info = env.step(action)
        #break

        #input()
        if done:
            break
    print(f"Episode {i_episode} finished after {i} timesteps")
    l_ep_length.append(i)
env.close()

print(np.shape(action))
print(np.shape(observation))
print(np.shape(l_observation))


fig1 = plt.figure(figsize = (8,6))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(range(len(l_ep_length)), l_ep_length)
fig1.show()
input() 

t = np.multiply(np.array(range(len(l_action))),tau)
#do_graph(t, l_action, l_observation, ref, force_mag, l_prediction = l_prediction)
do_graph(t, l_action, l_observation, ref, force_mag, l_prediction = None)
input()

# TODO:
# * SWE To make just one control class
# * solve LQR with CVXPY
# * How is reward calculated?
# * Emulate LQR with reward
# * Do linear TD(0) to learn LQR K
# * Do RL on PID
