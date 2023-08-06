from td3_bc import *
from td3_bc_eval import eval_cbf, compute_metrics
from preprocess_dataset import parse_dataset_safety


def make_combined_dataset(env_type: str):
    combined_dataset = {}
    for env_id in [
        f"{env_type}-random-v2",
        f"{env_type}-medium-v2",
        f"{env_type}-expert-v2",
    ]:
        _env = gym.make(env_id)
        _dataset = d4rl.qlearning_dataset(_env)
        for key, value in _dataset.items():
            if key not in combined_dataset:
                combined_dataset[key] = value
            else:
                combined_dataset[key] = np.concatenate((combined_dataset[key], value))
    return combined_dataset


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.use_mixed_dataset:
        # Use a mixed dataset of random, medium, and expert
        env_type = config.env.split("-")[1]  # Safety-{env_type}-{etc}
        combined_dataset = make_combined_dataset(env_type)
        dataset = combined_dataset
    else:
        # Use the specified dataset
        dataset = d4rl.qlearning_dataset(env)

    # Add the assumed safety labels to dataset
    # TODO: avoid hardcoding horizon
    dataset = parse_dataset_safety(dataset, H=20)
    # Preprocess the safety reward
    dataset["rewards"] = rewarder.modify_reward(
        dataset["observations"], dataset["rewards"]
    ).squeeze()

    if config.normalize_reward:
        modify_reward(dataset, config.env)

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

        # Save state mean, state std
        env_type = config.env.split("-")[1]  # Safety-{env_type}-{etc}
        np.save(f"rl_cbf/data/{env_type}_state_mean.npy", state_mean)
        np.save(f"rl_cbf/data/{env_type}_state_std.npy", state_std)

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
        "safe_supervised": config.safe_supervised,
        "unsafe_supervised": config.unsafe_supervised,
        "detach_actor": config.detach_actor,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        # Optionally apply unsafe supervised loss
        if trainer.unsafe_supervised:
            sampled_states = env.sample_states(config.batch_size)
            sample_states = torchify(
                sampled_states, device=config.device, dtype=torch.float32
            )
            is_unsafe = env.is_unsafe_th(sample_states)
            batch_supervised = (sample_states, is_unsafe)
            supervised_log_dict = trainer.train_supervised(batch_supervised)
            log_dict.update(supervised_log_dict)

        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
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
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
                f" , Episode length: {eval_episode_length:.3f}"
                f" , Safety success: {eval_safety_success:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {
                    "eval/mean_d4rl_normalized_score": normalized_eval_score,
                    "eval/mean_episode_length": eval_episode_length,
                    "eval/mean_safety_success": eval_safety_success,
                },
                step=trainer.total_it,
            )

            # Evaluate the CBF as a safety constraint
            cbf_eval_dict = eval_cbf(
                env,
                trainer,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            constrained_episode_length = cbf_eval_dict["episode_lengths"].mean()
            constrained_safety_success = cbf_eval_dict[
                "episode_safety_successes"
            ].mean()
            print("---------------------------------------")
            print(
                f"CBF evaluation over {config.n_episodes} episodes: "
                f"Constrained episode length: {constrained_episode_length:.3f}"
                f" , Constrained safety success: {constrained_safety_success:.3f}"
            )
            print("---------------------------------------")

            wandb.log(
                {
                    "cbf/mean_constrained_episode_length": constrained_episode_length.mean(),
                    "cbf/mean_constrained_safety_success": constrained_safety_success.mean(),
                    "cbf/mean_value": cbf_eval_dict["value_mean"],
                    "cbf/mean_safety_pred_accuracy": cbf_eval_dict[
                        "safety_pred_accuracy"
                    ],
                },
                step=trainer.total_it,
            )

            # Evaluate the CBF metrics
            metrics = compute_metrics(
                env,
                trainer,
                device=config.device,
                dataset=replay_buffer,
            )
            print("---------------------------------------")
            print(f"CBF metrics: {metrics}")
            print("---------------------------------------")
            for key, value in metrics.items():
                wandb.log({f"cbf/{key}": value}, step=trainer.total_it)

    if config.checkpoints_path is not None:
        # save final model
        path = os.path.join(config.checkpoints_path, f"final.pt")
        torch.save(trainer.state_dict(), path)
        wandb.save(path, base_path=config.checkpoints_path)


if __name__ == "__main__":
    train()
