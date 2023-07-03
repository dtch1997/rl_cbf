import argparse
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import rl_cbf.envs

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from typing import Callable, List
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from rl_cbf.learning.torch_utils import make_grid

COLORMAP = "coolwarm_r"


def get_default_spaces():
    """Get default spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    x_space = torch.linspace(-1.2, 0.6, steps=100)
    x_dot_space = torch.linspace(-0.07, 0.07, steps=100)
    return x_space, x_dot_space


def plot_heatmap_x(
    fig: plt.Figure,
    ax: plt.Axes,
    x_space: torch.Tensor,
    x_dot_space: torch.Tensor,
    values: torch.Tensor,
):

    X = x_space
    Y = x_dot_space
    Z = values
    # Transpose because contourf is weird
    Z = Z.T
    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap(COLORMAP)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    image = ax.contourf(X, Y, Z, cmap=cmap, norm=norm)
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("x_dot")

    return ax


class DQNMountainCarVisualizer:
    def visualize(self, model: QNetwork) -> plt.Figure:

        fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

        x_space, x_dot_space = get_default_spaces()
        states = make_grid([x_space, x_dot_space])
        values = model.predict_value(states)
        activations = model.predict_value(states, apply_sigmoid=False)

        ax[0] = plot_heatmap_x(fig, ax[0], x_space, x_dot_space, values)
        ax[0].set_title("Value Function")

        ax[1] = plot_heatmap_x(fig, ax[1], x_space, x_dot_space, activations)
        ax[1].set_title("Activation Function")

        fig.tight_layout()
        return fig


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    model_path = args.model_path
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("BaseMountainCar-v0")])
    model = QNetwork.from_argparse_args(envs, args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    visualizer = DQNMountainCarVisualizer()
    fig = visualizer.visualize(model)
    fig.show()
    input("Press enter to continue...")
