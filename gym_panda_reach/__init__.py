from gym.envs.registration import register

register(
    id="panda-reach-v0",
    entry_point="gym_panda_reach.envs:PandaEnv",
)
