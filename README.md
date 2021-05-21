OpenaAI Gym Franka Panda robot environment with PyBullet

Bsed on the awesome tutorial by  --> 

## Install

Install with `pip`:

    git clone https://github.com/mcarfagno/gym-panda-reach
    cd gym-panda-reach
    pip install .

## Basic Usage 

Example running of the environment:

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
References and Special Thanks:
* [Mahyar Abdeetedal](https://github.com/mahyaret) -> awesome [tutorial](https://www.etedal.net/2020/04/pybullet-panda.html) and [inspiration](https://github.com/mahyaret/gym-panda)
* [OpenAI]() -> original [environment](https://github.com/openai/gym/tree/master/gym/envs/robotics/fetch) 
