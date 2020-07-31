OpenaAI Gym Franka Panda robot environment with PyBullet

## Install

Install with `pip`:

    git clone https://github.com/mahyaret/gym-panda.git
    cd gym-panda
    pip install .

## Basic Usage

Running an environment:

```python
import gym
import gym_panda
env = gym.make('panda-v0')
env.reset()
for _ in range(100):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample()) # take a random action
env.close()
 ```
