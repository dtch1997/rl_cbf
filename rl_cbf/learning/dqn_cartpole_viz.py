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
from rl_cbf.learning.env_utils import make_env

def get_zero_spaces():
    """ Get zero spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    x_space = torch.zeros(1)
    x_dot_space = torch.zeros(1)
    theta_space = torch.zeros(1)
    theta_dot_space = torch.zeros(1)
    return x_space, x_dot_space, theta_space, theta_dot_space

def get_default_spaces():
    """ Get default spaces for plotting

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of x and theta space
    """
    x_space = torch.linspace(-3.2, 3.2, steps=100)
    x_dot_space = torch.linspace(-3.2, 3.2, steps=100)
    theta_space = torch.linspace(-np.pi / 4, np.pi / 4, steps=100)
    theta_dot_space = torch.linspace(-3, 3, steps=100)
    return x_space, x_dot_space, theta_space, theta_dot_space

def plot_heatmap_theta(fig: plt.Figure, 
                        ax: plt.Axes, 
                        theta_space: torch.Tensor, 
                        theta_dot_space: torch.Tensor, 
                        barrier_values: torch.Tensor):
    """ Plot heatmap of state values for theta and theta_dot

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
    Z = barrier_values[0,0].detach().cpu().numpy()
    Z = Z.T
    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('coolwarm')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    image = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
    fig.colorbar(image, ax=ax)
    theta_threshold = 12 * 2 * np.pi / 360
    ax.axvline(-theta_threshold, color='k', linestyle='--', linewidth=3)
    ax.axvline(theta_threshold, color='k', linestyle='--', linewidth=3)
    ax.set_title('Barrier function value')        
    ax.set_xlabel('theta')
    ax.set_ylabel('theta_dot')
    
    return ax

def plot_heatmap_x(fig: plt.Figure, 
                    ax: plt.Axes, 
                    x_space: torch.Tensor, 
                    x_dot_space: torch.Tensor, 
                    barrier_values: torch.Tensor):

    X = x_space
    Y = x_dot_space
    Z = barrier_values[:,:,0,0].detach().cpu().numpy()
    Z = Z.T
    levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
    cmap = plt.get_cmap('coolwarm')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    image = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
    fig.colorbar(image, ax=ax)
    x_threshold = -2.4
    ax.axvline(-x_threshold, color='k', linestyle='--', linewidth=3)
    ax.axvline(x_threshold, color='k', linestyle='--', linewidth=3)
    ax.set_title('Barrier function value')        
    ax.set_xlabel('x')
    ax.set_ylabel('x_dot')
    
    return ax

def load_model(
    model_path: str,
    make_env: Callable,
    env_id: str,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, envs


def load_ensemble_model(
    envs: gym.vector.SyncVectorEnv,
    model_paths: List[str],
    device: torch.device = torch.device("cpu"),
):
    models = []
    for model_path in model_paths:
        _model = QNetwork(envs, True).to(device)
        _model.load_state_dict(torch.load(model_path, map_location=device))
        models.append(_model)
    model = QNetworkEnsemble(envs, models).to(device)
    model.eval()
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str)
    parser.add_argument('--model-paths', type=str, nargs='+')
    args = parser.parse_args()
    return args

def get_initial_states(n_samples: int = 1):
    high = np.array([2.0, 0.1, 0.15, 0.1])
    low = -high 
    return np.random.uniform(low=low, high=high, size=(n_samples, 4))

class DQNCartpoleVisualizer:

    def __init__(self):
        self.eval_env = gym.make('DiverseCartPole-v1')

    def visualize(self, model: QNetwork, barrier_threshold = 95, state_history: np.ndarray = None) -> plt.Figure:

        fig, ax = plt.subplots(ncols=2, figsize=(25, 10))

        # Evaluate model on theta and theta_dot
        _, _, theta_space, theta_dot_space = get_default_spaces()
        x_space, x_dot_space, _, _ = get_zero_spaces()
        grid_x, grid_x_dot, grid_theta, grid_theta_dot = torch.meshgrid(
            x_space, x_dot_space, theta_space, theta_dot_space, indexing='xy'
        )
        states = torch.stack([grid_x, grid_x_dot, grid_theta, grid_theta_dot], dim=-1)
        states = states.reshape(-1, 4)
        state_values = model.predict_value(states)
        state_values = state_values.reshape(1, 1, 100, 100)
        state_values = torch.Tensor(state_values)
        barrier_values = - state_values + barrier_threshold

        plot_heatmap_theta(fig, ax[0], theta_space, theta_dot_space, barrier_values)    
        if state_history is not None:
            ax[0].scatter(state_history[:,2], state_history[:,3], c='k', s=1)

        # Evaluate model on x and x_dot
        x_space, x_dot_space, _, _ = get_default_spaces()
        _, _, theta_space, theta_dot_space = get_zero_spaces()
        grid_x, grid_x_dot, grid_theta, grid_theta_dot = torch.meshgrid(
            x_space, x_dot_space, theta_space, theta_dot_space, indexing='xy'
        )
        states = torch.stack([grid_x, grid_x_dot, grid_theta, grid_theta_dot], dim=-1)
        states = states.reshape(-1, 4)
        state_values = model.predict_value(states)
        state_values = state_values.reshape(100, 100, 1, 1)
        state_values = torch.Tensor(state_values)
        barrier_values = - state_values + barrier_threshold
        
        plot_heatmap_x(fig, ax[1], x_space, x_dot_space, barrier_values)
        if state_history is not None:
            ax[1].scatter(state_history[:,0], state_history[:,1], c='k', s=1)

        return fig