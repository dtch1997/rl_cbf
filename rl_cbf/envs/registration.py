from gym.envs.registration import register

register(
    'BaseCartPole-v1',
    entry_point='rl_cbf.envs.decomposed_cartpole:BaseCartPole',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'DiverseCartPole-v1',
    entry_point='rl_cbf.envs.diverse_cartpole:DiverseCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleA-v1',
    entry_point='rl_cbf.envs.decomposed_cartpole:CartPoleAEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleB-v1',
    entry_point='rl_cbf.envs.decomposed_cartpole:CartPoleBEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleC-v1',
    entry_point='rl_cbf.envs.decomposed_cartpole:CartPoleCEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    'CartPoleD-v1',
    entry_point='rl_cbf.envs.decomposed_cartpole:CartPoleDEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)