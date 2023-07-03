import gym
import torch
import torch.nn as nn

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


def evaluate(
    model: nn.Module,
    env: gym.Env,
    barrier_threshold: float = 0,
    render: bool = False,
):
    """Evaluate model on environment for 1 episode"""
    nominal_policy = NominalPolicy("right")

    # Roll out model
    state = env.reset()
    done = False
    i = 0

    while not done:
        if render:
            env.render()
        nominal_policy.update(state, i)
        action = nominal_policy.predict(state)
        next_barrier_value = model.predict(state, apply_sigmoid=False)[action].item()

        if next_barrier_value < barrier_threshold:
            # Choose the action with the highest Q-value
            action = model.predict_action(state).item()

        next_state, _, done, _ = env.step(action)
        state = next_state
        i += 1

    return i


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--env-id", type=str, default="DiverseCartPole-v1")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--barrier-threshold", type=float, default=0)
    parser.add_argument("--video-path", type=str, default="")
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    env = gym.make(args.env_id)
    env = gym.wrappers.RecordVideo(env, args.video_path)
    metrics = evaluate(
        model, env, eval_episodes=1, barrier_threshold=args.barrier_threshold
    )
    print(metrics)
