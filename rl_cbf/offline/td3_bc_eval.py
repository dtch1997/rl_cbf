""" Script to evaluate trained checkpoints """

from rl_cbf.offline.td3_bc import *


@torch.no_grad()
def eval_cbf(
    env: gym.Env, model: TD3_BC, device: str, n_episodes: int, seed: int
) -> Dict[str, np.ndarray]:
    # TODO: evaluate the CBF for off-policy safety constraints
    # Evaluate: Safety-constrained episode length
    # Evaluate: Safety success rate
    pass

    env.seed(seed)
    model.set_eval()
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            # Take random action
            action = env.action_space.sample()
            # Apply safety constraint
            state_th = torch.from_numpy(state).float().to(device).view(1, -1)
            action_th = torch.from_numpy(action).float().to(device).view(1, -1)
            next_q_value = model.get_q_value(state_th, action_th)
            if next_q_value < 0.5 / (1 - model.discount):
                # Unsafe; take action that maximizes Q-value
                action = model.actor.act(state, device)
            state, reward, done, info = env.step(action)
            env.render()
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    episode_safety_successes = np.asarray(episode_lengths) == 1000
    model.set_train()

    return {
        "episode_rewards": np.asarray(episode_rewards),
        "episode_lengths": np.asarray(episode_lengths),
        "episode_safety_successes": np.asarray(episode_safety_successes),
    }


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Dict[str, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = actor.act(state, device)
            state, reward, done, info = env.step(action)
            env.render()
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    actor.train()
    return {
        "episode_rewards": np.asarray(episode_rewards),
        "episode_lengths": np.asarray(episode_lengths),
    }


@pyrallis.wrap()
def eval(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    env_type = None
    env_types = ["ant", "halfcheetah", "hopper", "walker2d"]
    for env_type in env_types:
        if env_type in config.env:
            env_type = env_type
            break
    if env_type is None:
        raise ValueError(f"Unknown env type: {config.env}")

    state_mean = np.load(f"rl_cbf/data/{env_type}_state_mean.npy")
    state_std = np.load(f"rl_cbf/data/{env_type}_state_std.npy")
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Evaluating TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # Evaluate CBF
    eval_cbf_dict = eval_cbf(
        env,
        trainer,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    eval_episode_length = eval_cbf_dict["episode_lengths"].mean()
    eval_safety_success = eval_cbf_dict["episode_safety_successes"].mean()
    print("---------------------------------------")
    print(
        f"CBF evaluation over {config.n_episodes} episodes: "
        f"Episode length: {eval_episode_length:.3f}"
        f" , Safety success: {eval_safety_success:.3f}"
    )
    print("---------------------------------------")

    # Evaluate policy
    eval_dict = eval_actor(
        env,
        actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    eval_scores = eval_dict["episode_rewards"]
    eval_score = eval_scores.mean()
    eval_episode_lengths = eval_dict["episode_lengths"]
    eval_episode_length = eval_episode_lengths.mean()
    # TODO: avoid hardcoding this
    # Works for Ant, HalfCheetah, Hopper, Walker2d
    eval_safety_success = (eval_episode_lengths == 1000).mean()
    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
    print("---------------------------------------")
    print(
        f"Evaluation over {config.n_episodes} episodes: "
        f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
        f" , Episode length: {eval_episode_length:.3f}"
        f" , Safety success: {eval_safety_success:.3f}"
    )
    print("---------------------------------------")


if __name__ == "__main__":
    eval()
