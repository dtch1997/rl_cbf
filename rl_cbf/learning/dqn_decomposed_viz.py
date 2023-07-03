import argparse
import gym
import torch
import numpy as np

import rl_cbf.envs
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from rl_cbf.learning.dqn_cartpole_viz import DQNCartPoleVisualizer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--model-paths", nargs="+", type=str, required=True)
    args = parser.parse_args()

    envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])

    models = []
    for model_path in args.model_paths:
        model = QNetwork.from_argparse_args(envs, args)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        models.append(model)
    model = QNetworkEnsemble(
        envs,
        models,
        enable_bump_parametrization=args.enable_bump_parametrization,
        device=args.device,
    )

    visualizer = DQNCartPoleVisualizer()
    fig = visualizer.visualize(model)
    fig.show()
    input("Press enter to continue...")

    # Sanity check that model outputs are expected
    print(model.predict_value(np.array([0.0, 0.0, -0.3, 0.0])))
    print(model.predict_value(np.array([0.0, 0.0, 0.3, 0.0])))
    print(model.predict_value(np.array([0.0, 0.0, 0.0, -0.3])))
    print(model.predict_value(np.array([0.0, 0.0, 0.0, 0.3])))
    print()

    # Sanity check model values
    print(model.predict_value(np.array([-2.0, 0.0, 0.0, 0])))
    print(model.predict_value(np.array([2.0, 0.0, 0.0, 0])))
    print(model.predict_value(np.array([0.0, 2.0, 0.0, 0])))
    print(model.predict_value(np.array([0.0, -2.0, 0.0, 0])))
