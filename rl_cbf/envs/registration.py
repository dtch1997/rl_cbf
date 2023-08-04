from gym.envs.registration import register

register(
    "BaseCartPole-v1",
    entry_point="rl_cbf.envs.decomposed_cartpole:BaseCartPole",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    "DiverseCartPole-v1",
    entry_point="rl_cbf.envs.diverse_cartpole:DiverseCartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    "Safety-Walker2d-v2",
    entry_point="rl_cbf.envs.safety_env:SafetyWalker2dEnv",
    kwargs={"env_id": "Walker2d-v2"},
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    "Safety-walker2d-medium-v2",
    entry_point="rl_cbf.envs.safety_env:SafetyWalker2dEnv",
    kwargs={"env_id": "walker2d-medium-v2"},
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    "Safety-ant-medium-v2",
    entry_point="rl_cbf.envs.safety_env:SafetyAntEnv",
    kwargs={"env_id": "ant-medium-v2"},
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    "Safety-hopper-medium-v2",
    entry_point="rl_cbf.envs.safety_env:SafetyHopperEnv",
    kwargs={"env_id": "hopper-medium-v2"},
    max_episode_steps=1000,
    reward_threshold=950.0,
)
