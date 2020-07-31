OpenaAI Gym Franka Panda robot environment with PyBullet

Bsed on the awesome tutorial by Mahyar Abdeetedal --> https://www.etedal.net/

## Install

Install with `pip`:

    git clone https://github.com/mcarfagno/gym-panda-reach
    cd gym-panda-reach
    pip install .

## Basic Usage

Running an environment:

```python
import gym
import gym_panda_reach
env = gym.make('panda-reach-v0')
env.reset()
for _ in range(100):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample()) # take a random action
env.close()
```
