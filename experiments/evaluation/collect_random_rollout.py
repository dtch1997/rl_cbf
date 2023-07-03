""" Rollout random policy in CartPole env and save trajectories """

import gym
import torch
import torch.nn as nn
import numpy as np

import rl_cbf.envs
from typing import List


def rollout_random_policy(env, num_rollouts: int = 1, max_episode_len: int = 500):

    states = np.zeros((num_rollouts, max_episode_len, env.observation_space.shape[0]))
    state_masks = -1 * np.ones((num_rollouts, max_episode_len))
    for i in range(num_rollouts):
        env.reset()
        for j in range(max_episode_len):
            state, _, done, _ = env.step(env.action_space.sample())
            states[i, j] = state
            state_masks[i, j] = 1
            if done:
                state_masks[i, j] = 0
                break
    return states, state_masks


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="DiverseCartPole-v1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--max-episode-len", type=int, default=500)
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env.seed(args.seed)

    states, terminals = rollout_random_policy(
        env, args.num_rollouts, args.max_episode_len
    )
    np.save("experiment/artifacts/data/states.npy", states)
    np.save("experiment/artifacts/data/terminals.npy", terminals)
