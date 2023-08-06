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
)

# Register d4rl envs

for env_type in ("ant", "halfcheetah", "hopper", "walker2d"):
    for dataset_type in (
        "random",
        "medium",
        "expert",
        "medium-replay",
        "medium-expert",
        "mixed",
    ):
        env_id = f"{env_type}-{dataset_type}-v2"
        env_class_name = "Safety" + env_type.capitalize() + "Env"
        base_env_id = env_id
        if dataset_type == "mixed":
            # The mixed dataset is constructed in the training script
            # So we set a dummy value for the base env id
            base_env_id = f"{env_type}-medium-v2"
        register(
            f"Safety-{env_id}",
            entry_point=f"rl_cbf.envs.safety_env:{env_class_name}",
            kwargs={"env_id": base_env_id},
            max_episode_steps=1000,
        )
