import gym
import numpy as np
from controller import PID, PolePlacement, MPC, do_graph, Model
import matplotlib.pyplot as plt

str_control = 'ss'
force_mag = 10.0
tau = 0.02
env = gym.make('CartPole-cont')
#print("action_space", env.action_space)
#print("observation_space", env.observation_space)

ref = [0,0,0,0]
if str_control == 'pid':
    control = PID(15, 0, 0) # Very good performance
    #pid = PID(1.5, 1.5, 0) # Worse performance, but integral works
    #pid = PID(1, 0, 1)
elif str_control == 'ss':
    ref = [-0.25, 0, 0, 0]
    # k = [-2.0918, -6.3452, -65.4748, -26.4386] # Experiments with pole placement
    k = [-2.2361, -3.5987, -36.9680, -9.4652]
    control = PolePlacement(k = k, ref = ref)
elif str_control == 'mpc':
    control = MPC()


for i_episode in range(1):
    observation = env.reset()
    model = Model(observation)

    l_observation = np.empty((0,4)) 
    l_prediction = np.empty((0,4))
    l_action = np.array([])
    for t in range(400):

        env.render()
        print(f"{i_episode}, {t}: {observation}")

        action = np.clip(control.step(observation), -force_mag, force_mag)

        l_observation = np.append(l_observation, np.array([observation]), axis = 0)
        l_prediction = np.append(l_prediction, model.predict(action)[:,0].T, axis = 0)
        l_action = np.append(l_action, action)
    
        observation, reward, done, info = env.step(action)
        #break

        if done and False:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

print(np.shape(action))
print(np.shape(observation))
print(np.shape(l_observation))

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
