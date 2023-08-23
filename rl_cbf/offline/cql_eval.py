from rl_cbf.offline.cql import *

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, render = False
) -> Tuple[np.ndarray, np.ndarray]:
    # Note: this is evaluation with the original reward
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            if render: 
                env.render()
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)

@torch.no_grad()
def eval_cbf(
    env: gym.Env, actor, critic1, critic2, device: str, n_episodes: int, seed: int, render = False, safety_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    # TODO: evaluate the CBF for off-policy safety constraints
    # Evaluate: Safety-constrained episode length
    # Evaluate: Safety success rate
    pass

    env.seed(seed)
    actor.eval()
    
    safety_threshold = safety_threshold / (1 - 0.99)
    episode_rewards = []
    episode_lengths = []
    values = []
    safety_pred_accuracies = []
    exploratory_action_counts = []
    for _ in range(n_episodes):

        # Reset env
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0
        n_exploratory_actions = 0
        safety_pred = 1

        # Initialize episode buffer
        episode_state_buffer = np.zeros((1001, env.observation_space.shape[0]))
        episode_action_buffer = np.zeros((1000, env.action_space.shape[0]))
        episode_state_buffer[0] = state

        while not done:
            # Select random action
            action = env.action_space.sample()

            # Apply safety constraint;
            state_th = torch.from_numpy(state).float().to(device).view(1, -1)
            random_action_th = torch.from_numpy(action).float().to(device).view(1, -1)
            learned_action_th = actor(state_th)[0]
            q_value_random_q1 = critic1(state_th, random_action_th)
            q_value_random_q2 = critic2(state_th, random_action_th)
            q_value_random = min(q_value_random_q1, q_value_random_q2)
            q_value_learned_q1 = critic1(state_th, learned_action_th)
            q_value_learned_q2 = critic2(state_th, learned_action_th)
            q_value_learned = min(q_value_learned_q1, q_value_learned_q2)
            # print("Q values: random {}, learned {}".format(q_value_random, q_value_learned))
            if q_value_random < safety_threshold:
                # Unsafe; take action that maximizes Q-value
                action = actor.act(state, device)
            else: 
                n_exploratory_actions += 1
            if q_value_learned < safety_threshold:
                # We have entered an unsafe state
                safety_pred = 0

            # Step the environment
            state, reward, done, info = env.step(action)
            if render: env.render()

            # Record data
            values.append(q_value_learned.item())
            episode_reward += reward
            episode_length += 1
            episode_state_buffer[episode_length] = state
            episode_action_buffer[episode_length - 1] = action

        if safety_pred == 1 and episode_length < 1000:
            # Predicted safe, but episode terminated early
            safety_pred_accuracies.append(0)
        elif safety_pred == 0 and episode_length == 1000:
            # Predicted unsafe, but episode did not terminate early
            safety_pred_accuracies.append(0)
        else:
            # Predicted correctly
            safety_pred_accuracies.append(1)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_state_buffer = episode_state_buffer[: episode_length + 1]
        episode_action_buffer = episode_action_buffer[:episode_length]
        exploratory_action_counts.append(n_exploratory_actions)

    episode_safety_successes = np.asarray(episode_lengths) == 1000
    actor.train()
    critic1.train()
    critic2.train()

    return {
        "episode_rewards": np.asarray(episode_rewards),
        "episode_lengths": np.asarray(episode_lengths),
        "episode_safety_successes": np.asarray(episode_safety_successes),
        "value_mean": np.mean(values),
        "safety_pred_accuracy": np.mean(safety_pred_accuracies),
        "exploratory_action_counts": np.asarray(exploratory_action_counts),
    }

@dataclass
class EvalConfig(TrainConfig):
    render: bool = False
    safety_threshold: float = 0.5

@pyrallis.wrap()
def eval(config: EvalConfig):
    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    if config.relabel == "identity":
        rewarder = safety_reward.IdentityRewarder(env)
    elif config.relabel == "zero_one":
        rewarder = safety_reward.ZeroOneRewarder(env)
    elif config.relabel == "constant_0.2":
        penalty = 0.2 * (env.ref_max_score - env.ref_min_score) / 1000
        rewarder = safety_reward.ConstantPenaltyRewarder(env, penalty=penalty)
    elif config.relabel == "constant_0.8":
        penalty = 0.8 * (env.ref_max_score - env.ref_min_score) / 1000
        rewarder = safety_reward.ConstantPenaltyRewarder(env, penalty=penalty)
    else:
        raise ValueError(f"Unknown relabeling method: {config.relabel}")

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = d4rl.qlearning_dataset(env)
    dataset["rewards"] = rewarder.modify_reward(
        dataset["observations"], dataset["rewards"]
    ).squeeze()

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim, action_dim, max_action, orthogonal_init=config.orthogonal_init
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    print("Evaluating CBF")
    eval_metrics = eval_cbf(
        eval_env,
        actor,
        critic_1,
        critic_2,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
        render=config.render,
        safety_threshold=config.safety_threshold,
    )
    print("Episode lengths: ", eval_metrics['episode_lengths'])
    print("Exploration counts: ", eval_metrics['exploratory_action_counts'])
    print("Mean exploration rate: ", np.sum(eval_metrics['exploratory_action_counts']) / np.sum(eval_metrics["episode_lengths"]))

    print("Evaluating actor")
    eval_scores, success_rate = eval_actor(
        eval_env,
        actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
        render=config.render,
    )
    eval_score = eval_scores.mean()
    eval_log = {}
    normalized = eval_env.get_normalized_score(np.mean(eval_scores))

if __name__ == "__main__":
    eval()