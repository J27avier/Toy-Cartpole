import numpy as np

a_res = np.array(range(5))
a_probs = np.exp(a_res) / np.sum(np.exp(a_res))
action = np.random.choice(a_res, p = a_probs)
print(a_res)
print(a_probs)
print(action)
