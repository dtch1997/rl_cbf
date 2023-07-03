import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.patches as patches

import rl_cbf.envs
from rl_cbf.learning.env_utils import make_env
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from typing import List


class NominalPolicy:
    """A nominal policy that tries to visit both extremes of the x-space"""

    def __init__(self, direction: str = "left", period: int = 400):
        self.direction = direction
        self.period = period

    def predict(self, state):
        """Predict action given state

        Action 0: apply left-pointing force
        Action 1: apply right-pointing force
        """
        if self.direction == "left":
            return 0
        else:
            return 1

    def update(self, state, timestep):
        period = self.period
        threshold = self.period // 2
        if timestep % period < threshold:
            self.direction = "left"
        if timestep % period > threshold:
            self.direction = "right"


def evaluate_and_record_states(
    model: nn.Module,
    env: gym.Env,
    barrier_threshold: float = None,
    constrain: bool = True,
    max_episode_length: int = 500,
):
    """Evaluate model on environment for 1 episode"""
    nominal_policy = NominalPolicy("right")

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
        nominal_policy.update(state, i)
        action = nominal_policy.predict(state)
        next_barrier_value = model.predict_next_barrier(state, action)
        print(next_barrier_value)
        if next_barrier_value < 0:
            # Choose the action with the highest Q-value
            action = model.predict_action(state).item()

        next_state, _, done, _ = env.step(action)
        if i in frame_indices:
            img = env.render(mode="rgb_array")
            imgs.append(img)
        if not constrain:
            done = False
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
    parser.add_argument("--barrier-threshold", type=float, default=None)
    parser.add_argument("--plot-path", type=str, default="")
    parser.add_argument("--save-path", type=str, default="states.csv")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-constrain", action="store_true")
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Avoid using gym.make.
    # We want to iterate past done, which gym.make does not allow.
    from rl_cbf.envs.diverse_cartpole import DiverseCartPoleEnv

    env = DiverseCartPoleEnv()
    env.seed(args.seed)

    states, imgs = evaluate_and_record_states(
        model,
        env,
        barrier_threshold=args.barrier_threshold,
        constrain=not args.no_constrain,
    )
    np.savetxt(args.save_path, states)

    # Now plot the frames
    frames = imgs
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # I don't know why but the rendered video has more than 500 frames.
    # So we need to manually adjust the frame indices.
    total_frames = 500
    frame_indices = [
        int(total_frames * fraction) for fraction in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]

    for i, frame in enumerate(frames):
        axs[i].imshow(frame)
        axs[i].set_title(f"Frame {frame_indices[i]}")
        axs[i].axis("off")

        # Add black border around the subplot
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor="black",
            facecolor="none",
            transform=axs[i].transAxes,
            clip_on=False,
        )
        axs[i].add_patch(rect)

    fig.tight_layout()
    fig.savefig(args.plot_path)
