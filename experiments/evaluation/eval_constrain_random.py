import gym
import torch
import torch.nn as nn
import numpy as np

import rl_cbf.envs
from rl_cbf.learning.env_utils import make_env
from rl_cbf.net.q_network import QNetwork
from typing import List


def evaluate(
    model: nn.Module,
    env: gym.Env,
    barrier_threshold: float = None,
    num_rollouts: int = 1,
    max_episode_length: int = 500,
):
    """Evaluate model on environment"""
    # Roll out model

    states = []
    ep_lengths = []

    for ep in range(num_rollouts):
        state = env.reset()
        states.append(state)
        done = False
        i = 0
        while not done and i < max_episode_length:
            i += 1
            action = env.action_space.sample()
            next_barrier = model.predict_next_barrier(
                state, action, barrier_threshold=barrier_threshold
            )

            if not next_barrier >= 0:
                # Choose the action with the highest Q-value
                action = model.predict_action(state).item()

            state, _, done, _ = env.step(action)
            states.append(state)
        ep_lengths.append(i)

    states = np.array(states)
    ep_lengths = np.array(ep_lengths, dtype=np.int64)
    return states, ep_lengths


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--env-id", type=str, default="DiverseCartPole-v1")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--barrier-threshold", type=float, default=None)
    parser.add_argument("--save-path", type=str, default="states.csv")
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-episode-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    env = gym.make("DiverseCartPole-v1")
    env.seed(args.seed)

    states, ep_lengths = evaluate(
        model,
        env,
        barrier_threshold=args.barrier_threshold,
        num_rollouts=args.num_rollouts,
        max_episode_length=args.max_episode_length,
    )

    np.savetxt(args.save_path + "_states.csv", states)
    np.savetxt(args.save_path + "_ep_lengths.csv", ep_lengths)
