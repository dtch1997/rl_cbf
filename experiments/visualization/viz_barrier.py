import argparse
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import rl_cbf.envs
import pathlib

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from typing import Callable, List
from rl_cbf.net.q_network import QNetwork, QNetworkEnsemble
from rl_cbf.learning.torch_utils import make_grid

COLORMAP = "coolwarm_r"

exp_name_to_label = {
    "bump_supervised_base_2M": "NOEXP",
    "bump_supervised_2M": "SIGMOID_SUP",
    "bump_2M": "SIGMOID",
    "baseline_supervised_2M": "MLP_SUP",
    "baseline_2M": "MLP",
}


def get_zero_spaces():
    """Get zero spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    x_space = torch.zeros(1)
    x_dot_space = torch.zeros(1)
    theta_space = torch.zeros(1)
    theta_dot_space = torch.zeros(1)
    return x_space, x_dot_space, theta_space, theta_dot_space


def get_default_spaces():
    """Get default spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    theta_threshold = 12 * 2 * np.pi / 360
    x_space = torch.linspace(-4.8, 4.8, steps=100)
    x_dot_space = torch.linspace(-3.2, 3.2, steps=100)
    theta_space = torch.linspace(-2 * theta_threshold, 2 * theta_threshold, steps=100)
    theta_dot_space = torch.linspace(-3, 3, steps=100)
    return x_space, x_dot_space, theta_space, theta_dot_space


def plot_heatmap_theta(
    fig: plt.Figure,
    ax: plt.Axes,
    theta_space: torch.Tensor,
    theta_dot_space: torch.Tensor,
    values: torch.Tensor,
):
    """Plot heatmap of state values for theta and theta_dot

    Args:
        fig (matplotlib.figure.Figure): Figure to plot on
        ax (matplotlib.axes.Axes): Axes to plot on
        theta_values (torch.Tensor): Theta values
            Shape [N]
        theta_dot_values (torch.Tensor): Theta_dot values
            Shape [N]
        state_values (torch.Tensor): State values to plot
            Shape [1, 1, N, N]
    """
    X = theta_space
    Y = theta_dot_space
    Z = values[0, 0].detach().cpu().numpy()
    # Transpose because contourf is weird
    Z = Z.T
    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap(COLORMAP)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    image = ax.contourf(X, Y, Z, cmap=cmap, norm=norm)
    ax.contour(X, Y, Z, levels=[0], colors="k", linestyles=["dashed"], linewidths=[3])
    fig.colorbar(image, ax=ax)
    theta_threshold = 12 * 2 * np.pi / 360
    ax.axvline(-theta_threshold, color="k", linewidth=3)
    ax.axvline(theta_threshold, color="k", linewidth=3)
    ax.set_xlabel("theta")
    ax.set_ylabel("theta_dot")

    return ax


def plot_heatmap_x(
    fig: plt.Figure,
    ax: plt.Axes,
    x_space: torch.Tensor,
    x_dot_space: torch.Tensor,
    values: torch.Tensor,
):

    X = x_space
    Y = x_dot_space
    Z = values[:, :, 0, 0].detach().cpu().numpy()
    # Transpose because contourf is weird
    Z = Z.T
    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap(COLORMAP)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    image = ax.contourf(X, Y, Z, cmap=cmap, norm=norm)
    ax.contour(X, Y, Z, levels=[0], colors="k", linestyles=["dashed"], linewidths=[3])
    fig.colorbar(image, ax=ax)
    x_threshold = -2.4
    ax.axvline(-x_threshold, color="k", linewidth=3)
    ax.axvline(x_threshold, color="k", linewidth=3)
    ax.set_xlabel("x")
    ax.set_ylabel("x_dot")

    return ax


def get_initial_states(n_samples: int = 1):
    high = np.array([2.0, 0.1, 0.15, 0.1])
    low = -high
    return np.random.uniform(low=low, high=high, size=(n_samples, 4))


class DQNCartPoleVisualizer:
    def __init__(self):
        self.eval_env = gym.make("DiverseCartPole-v1")

    def visualize(
        self,
        model: QNetwork,
        x_ax: plt.Axes = None,
        theta_ax: plt.Axes = None,
    ) -> plt.Figure:
        # Evaluate model on theta and theta_dot
        _, _, theta_space, theta_dot_space = get_default_spaces()
        x_space, x_dot_space, _, _ = get_zero_spaces()
        states = make_grid([x_space, x_dot_space, theta_space, theta_dot_space])
        states = states.reshape(-1, 4)

        # Plot barrier function
        activations = model.predict_barrier(states)
        activations = activations.reshape(1, 1, 100, 100)
        activations = torch.Tensor(activations)
        plot_heatmap_theta(fig, theta_ax, theta_space, theta_dot_space, activations)

        # Evaluate model on x and x_dot
        x_space, x_dot_space, _, _ = get_default_spaces()
        _, _, theta_space, theta_dot_space = get_zero_spaces()
        states = make_grid([x_space, x_dot_space, theta_space, theta_dot_space])
        states = states.reshape(-1, 4)

        # Plot barrier function
        activations = model.predict_barrier(states)
        activations = activations.reshape(100, 100, 1, 1)
        activations = torch.Tensor(activations)
        plot_heatmap_x(fig, x_ax, x_space, x_dot_space, activations)

        return fig


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument("--model-paths", type=str, nargs="+", required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    num_models = len(args.model_paths)
    fig, axes = plt.subplots(2, num_models, figsize=(3.5 * num_models, 5.5))
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("DiverseCartPole-v1")])
    visualizer = DQNCartPoleVisualizer()

    for i, model_path in enumerate(args.model_paths):
        if "bump" in model_path:
            args.enable_bump_parametrization = True
        else:
            args.enable_bump_parametrization = False
        model = QNetwork.from_argparse_args(envs, args)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        x_ax = axes[0, i]
        theta_ax = axes[1, i]
        x_ax.set_title(exp_name_to_label[pathlib.Path(model_path).stem])
        fig = visualizer.visualize(model, x_ax, theta_ax)

    fig.tight_layout()
    fig.savefig(args.save_path)
