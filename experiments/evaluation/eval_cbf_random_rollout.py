import gym
import torch
import torch.nn as nn
import numpy as np

import rl_cbf.envs
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from typing import List

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--env-id", type=str, default="DiverseCartPole-v1")
    parser.add_argument("--data-path", type=str, default="states.npy")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--plot-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    from rl_cbf.envs.diverse_cartpole import DiverseCartPoleEnv

    env = DiverseCartPoleEnv()
    env.seed(args.seed)

    states = np.load("experiment/artifacts/data/states.npy")
    state_masks = np.load("experiment/artifacts/data/terminals.npy")
    barrier_values = model.predict_barrier(states)

    terminal_idx = np.where(state_masks == 0)
    terminal_preds = barrier_values[terminal_idx] < 0
    nonterminal_preds = barrier_values[state_masks == 1] >= 0
    print(np.mean(terminal_preds))
    print(np.mean(nonterminal_preds))

    breakpoint()
