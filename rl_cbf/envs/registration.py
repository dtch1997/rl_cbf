from gym.envs.registration import register

register(
    'DiverseCartPole-v1',
    entry_point='ql_clbf.envs.safe_cartpole:DiverseCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)