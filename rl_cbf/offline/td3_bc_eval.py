""" Script to evaluate trained checkpoints """

from rl_cbf.offline.td3_bc import *


def parse_episode(
    state_buffer: np.ndarray, action_buffer: np.ndarray, H: int = 20
) -> np.ndarray:
    """
    Parse an episode into safe and unsafe states
    """

    terminal_state = state_buffer[-1].view(1, -1)
    # Here we make use of the finite irrecoverability assumption
    # which implies that any state at least H steps before terminal is safe
    assumed_safe_states = state_buffer[: -H - 1]
    return assumed_safe_states, terminal_state


def compute_metrics(
    env: gym.Env,
    model: TD3_BC,
    device: str,
    dataset: ReplayBuffer,
    n_samples: int = 1024,
    alpha: float = 0.1,
) -> Dict[str, float]:
    """
    Compute the validity of the CBF
    """
    safety_threshold = 0.5 / (1 - model.discount)
    batch = dataset.sample(n_samples)
    states, actions, rewards, next_states, dones, assumed_safety = batch

    # Q(s, pi(s)) is a variational approximation of max_a(Q(s, a))
    values = model.get_value(states)
    next_values = model.get_value(next_states)
    cbf_values = values - safety_threshold
    next_cbf_values = next_values - safety_threshold

    # Check validity of CBF
    is_unsafe = dones
    cbf_error_i = (is_unsafe == 1) & (cbf_values >= 0)
    cbf_error_ii = (cbf_values >= 0) & (next_cbf_values < (1 - alpha) * cbf_values)
    cbf_error = cbf_error_i | cbf_error_ii
    cbf_validity = 1.0 - cbf_error.float().mean().item()

    # Check coverage of CBF
    cbf_coverage = (cbf_values >= 0).float().mean().item()

    return {
        "cbf_validity": cbf_validity,
        "cbf_coverage": cbf_coverage,
        "cbf_error_i": cbf_error_i.float().mean().item(),
        "cbf_error_ii": cbf_error_ii.float().mean().item(),
    }


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
    safety_threshold = 0.5 / (1 - model.discount)
    episode_rewards = []
    episode_lengths = []
    values = []
    for _ in range(n_episodes):

        # Reset env
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0

        # Initialize episode buffer
        episode_state_buffer = np.zeros((1001, env.observation_space.shape[0]))
        episode_action_buffer = np.zeros((1000, env.action_space.shape[0]))
        episode_state_buffer[0] = state

        while not done:
            # Select random action
            action = env.action_space.sample()

            # Apply safety constraint;
            state_th = torch.from_numpy(state).float().to(device).view(1, -1)
            action_th = torch.from_numpy(action).float().to(device).view(1, -1)
            q_value_random = model.get_q_value(state_th, action_th)
            value = model.get_value(state_th)
            # print("q_value_random", q_value_random)
            # print("value", value)
            if q_value_random < safety_threshold:
                # Unsafe; take action that maximizes Q-value
                action = model.actor.act(state, device)

            # Step the environment
            state, reward, done, info = env.step(action)
            # env.render()

            # Record data
            episode_reward += reward
            episode_length += 1
            episode_state_buffer[episode_length] = state
            episode_action_buffer[episode_length - 1] = action
            values.append(value.item())

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_state_buffer = episode_state_buffer[: episode_length + 1]
        episode_action_buffer = episode_action_buffer[:episode_length]

    episode_safety_successes = np.asarray(episode_lengths) == 1000
    model.set_train()

    return {
        "episode_rewards": np.asarray(episode_rewards),
        "episode_lengths": np.asarray(episode_lengths),
        "episode_safety_successes": np.asarray(episode_safety_successes),
        "value_mean": np.mean(values),
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
        "bounded": config.bounded,
        "supervised": config.supervised,
        "detach_actor": config.detach_actor,
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
