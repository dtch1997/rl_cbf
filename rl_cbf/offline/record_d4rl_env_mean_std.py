import gym
import numpy as np
import pyrallis
import d4rl

from td3_bc import TrainConfig, compute_mean_std


@pyrallis.wrap()
def main(config: TrainConfig):

    env = gym.make(config.env)
    dataset = d4rl.qlearning_dataset(env)
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

    # Save state mean, state std
    env_type = config.env.split("-")[1]  # Safety-{env_type}-{etc}
    np.save(f"rl_cbf/data/{env_type}_state_mean.npy", state_mean)
    np.save(f"rl_cbf/data/{env_type}_state_std.npy", state_std)


if __name__ == "__main__":
    main()
