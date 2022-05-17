# Toy Cart-Pole

Repository for the Cart-Pole problem.

![Example screen](/img/cartpole_screen.png)

## Implemented Classic Controllers:
* PID
* Pole-Placement in State-Space
* LQR
* MPC 

## To Do:
* Implement RL agent (discrete and continuous actions)

## To run:
1. Clone the [https://github.com/openai/gym](OpenAi Gym) repo
2. Add the `/gym/gym/envs/classic_control/cartpole_cont.py` file to the corresponding path
3. Add to `/gym/gym/envs/__init__.py`
```
register(
    id="CartPole-cont",
    entry_point="gym.envs.classic_control.cartpole_cont:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

```
4. Do `$pip install -e .`
