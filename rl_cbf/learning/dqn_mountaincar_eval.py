import gym 
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import rl_cbf.envs
from typing import Optional
from rl_cbf.net.q_network import QNetwork
from rl_cbf.learning.dqn_cartpole_eval import DQNCartPoleEvaluator

class DQNCMountainCarEvaluator(DQNCartPoleEvaluator):

    def __init__(self,
                 capture_video: bool = False,
                 video_path: str = 'eval'):
        self.eval_env = gym.make('BaseMountainCar-v0')
        self.capture_video = capture_video
        if self.capture_video:
            self.eval_env = gym.wrappers.RecordVideo(self.eval_env, f'./videos/{video_path}')
    
    def sample_grid_points(self, n_grid_points: int = 1):
        """ Sample grid points from the state space """
        high = self.eval_env.observation_space.high
        n = self.eval_env.observation_space.shape[0]
        states = np.random.uniform(low=-high, high=high, size=(n_grid_points, n))
        return states
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--video-path', type=str, default='')
    parser.add_argument('--results-path', type=str, default='results.csv')
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make('BaseMountainCar-v0')])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    capture_video = (args.video_path != '')
    evaluator = DQNCMountainCarEvaluator(capture_video=capture_video, video_path=args.video_path)
    df = evaluator.evaluate_grid(model, n_grid_points=10000)
    df.to_csv(args.results_path, index=False)