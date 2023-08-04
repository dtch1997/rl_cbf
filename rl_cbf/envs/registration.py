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
    'SafetyWalker2d-v3',
    entry_point='rl_cbf.envs.safety_env:SafetyWalker2dEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)