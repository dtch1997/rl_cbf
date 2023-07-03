import gym
import numpy as np
import torch.nn as nn
import pandas as pd

import torch
import argparse
import rl_cbf.envs

from typing import Optional
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from rl_cbf.learning.dqn_cartpole_eval import DQNCartPoleEvaluator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--model-paths", nargs="+", type=str, required=True)
    parser.add_argument("--video-path", type=str, default="")
    parser.add_argument("--results-path", type=str, default="results.csv")
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])

    models = []
    for model_path in args.model_paths:
        model = QNetwork.from_argparse_args(_envs, args)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        models.append(model)
    model = QNetworkEnsemble(
        _envs,
        models,
        enable_bump_parametrization=args.enable_bump_parametrization,
        device=args.device,
    )

    capture_video = args.video_path != ""
    evaluator = DQNCartPoleEvaluator(
        capture_video=capture_video, video_path=args.video_path
    )
    df = evaluator.evaluate_rollout(model, num_rollouts=10)
    print(df["episode_length"])
    df.to_csv(args.results_path, index=False)
