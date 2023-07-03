import gym
import torch
import torch.nn as nn
import numpy as np

import rl_cbf.envs
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from typing import List


def evaluate_and_record_states(
    model: nn.Module,
    env: gym.Env,
    max_episode_length: int = 500,
):
    """Evaluate model on environment for 1 episode"""

    # Roll out model
    state = env.reset()
    done = False
    i = 0

    states = [state]
    imgs = []
    frame_indices = [
        int(max_episode_length * fraction) for fraction in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]

    while not done and i < max_episode_length:
        action = model.predict_action(state).item()

        next_state, _, done, _ = env.step(action)
        if i in frame_indices:
            img = env.render(mode="rgb_array")
            imgs.append(img)
        state = next_state
        i += 1
        states.append(state)

    return states, imgs


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--env-id", type=str, default="DiverseCartPole-v1")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--barrier-threshold", type=float, default=0)
    parser.add_argument("--plot-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="states.csv")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-constrain", action="store_true")
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    from rl_cbf.envs.diverse_cartpole import DiverseCartPoleEnv

    env = DiverseCartPoleEnv()
    env.seed(args.seed)

    states, _ = evaluate_and_record_states(model, env)
    np.savetxt(args.save_path, states)
