from gym.envs.registration import register

register(
    'DiverseCartPole-v1',
    entry_point='rl_cbf.envs.diverse_cartpole:DiverseCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)