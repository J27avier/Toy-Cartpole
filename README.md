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
1. Create a venv
2. Clone the [OpenAi Gym](https://github.com/openai/gym) repo into project directory (`Toy-CartPole/`)
3. Add the `/gym/gym/envs/classic_control/cartpole_cont.py` file to the corresponding path
4. Add to `/gym/gym/envs/__init__.py`
```
register(
    id="CartPole-cont",
    entry_point="gym.envs.classic_control.cartpole_cont:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

```
5. Do `$pip install -e .` inside `/gym/`
