from rl_cbf.offline.cql import *
from rl_cbf.offline.cql_eval import *

@pyrallis.wrap()
def train(config: TrainConfig):
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

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

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

    wandb_init(asdict(config))

    evaluations = []
    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):
        if t == config.offline_iterations:
            print("Online tuning")
            trainer.cql_alpha = config.cql_alpha_online
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action, _ = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)

            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(
                    reward,
                    config.env,
                    reward_scale=config.reward_scale,
                    reward_bias=config.reward_bias,
                    **reward_mod_dict,
                )
            replay_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state

            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = (
                    normalized_return * 100.0
                )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            normalized_eval_score = normalized * 100.0
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")

            # Evaluate CBF
            eval_metrics = eval_cbf(
                eval_env, 
                trainer,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            for k, v in eval_metrics.items():
                eval_log[f"eval_cbf/{k}"] = v

            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(eval_log, step=trainer.total_it)


if __name__ == "__main__":
    train()