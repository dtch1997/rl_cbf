import numpy as np

import torch
import torch.nn as nn

from typing import List

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(
        self,
        env,
        hidden_dim_1: int = 120,
        hidden_dim_2: int = 84,
        enable_bump_parametrization: bool = False,
        enable_siren_layer: bool = False,
        min_value: float = 0,
        max_value: float = 100,
        device: str = "cuda",
    ):
        super().__init__()
        self.enable_bump_parametrization = enable_bump_parametrization
        # Only used in case of bump parametrization
        self.max = max_value
        # Not implemented yet
        self.min = min_value
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ELU(),
            nn.Linear(120, 84),
            nn.ELU(),
            nn.Linear(84, env.single_action_space.n),
        )

        if enable_bump_parametrization:
            self.w = lambda x: (self.max - self.min) * torch.sigmoid(x) + self.min

            def w_inv(x):
                x = (x - self.min) / (self.max - self.min)
                return torch.log(x / (1 - x))

            self.w_inv = w_inv
        else:
            self.w = lambda x: x
            self.w_inv = lambda x: x

    @staticmethod
    def add_argparse_args(parser: "argparse.ArgumentParser"):
        parser.add_argument("--enable-bump-parametrization", action="store_true")
        parser.add_argument("--enable-siren-layer", action="store_true")
        parser.add_argument("--min-value", type=float, default=0)
        parser.add_argument("--max-value", type=float, default=100)
        parser.add_argument("--hidden-dim-1", type=int, default=120)
        parser.add_argument("--hidden-dim-2", type=int, default=84)
        parser.add_argument("--device", type=str, default="cuda")
        return parser

    @staticmethod
    def from_argparse_args(env, args, **kwargs):
        return QNetwork(
            env,
            enable_bump_parametrization=args.enable_bump_parametrization,
            enable_siren_layer=args.enable_siren_layer,
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2,
            min_value=args.min_value,
            max_value=args.max_value,
            device=args.device,
            **kwargs
        )

    @property
    def barrier_threshold(self):
        return (self.max - self.min) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(self.network(x))

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q_values(x).max(dim=-1)[0]

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q_values(x).max(dim=-1)[1]

    def get_barrier(
        self, x: torch.Tensor, barrier_threshold: float = None
    ) -> torch.Tensor:
        """Predict barrier values.

        Barrier values are positive if state is safe, negative if state is unsafe.
        """
        if barrier_threshold is None:
            barrier_threshold = self.barrier_threshold
        barrier_threshold = torch.tensor(
            barrier_threshold, dtype=torch.float32, device=self.device
        )

        if self.enable_bump_parametrization:
            barrier_values = self.network(x).max(dim=-1)[0]
            effective_threshold = self.w_inv(barrier_threshold)
            return barrier_values - effective_threshold
        else:
            barrier_values = self.get_values(x)
            return barrier_values - barrier_threshold

    def get_next_barrier(
        self, x: torch.Tensor, u: torch.Tensor, barrier_threshold: float = None
    ) -> torch.Tensor:
        """Predict safety of next state according to barrier threshold

        Input:
            x: current state [batch_size, state_dim]
            u: action [batch_size]
        """
        if barrier_threshold is None:
            barrier_threshold = self.barrier_threshold
        barrier_threshold = torch.tensor(
            barrier_threshold, dtype=torch.float32, device=self.device
        )

        if self.enable_bump_parametrization:
            barrier_values = self.network(x)[u]
            effective_threshold = self.w_inv(barrier_threshold)
            return barrier_values - effective_threshold
        else:
            barrier_values = self.get_q_values(x)[u]
            return barrier_values - barrier_threshold

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict q-values"""
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        q_values = self.get_q_values(x)
        return q_values.detach().cpu().numpy()

    def predict_value(self, x: np.ndarray) -> np.ndarray:
        """Predict value of current state"""
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        value = self.get_values(x)
        return value.detach().cpu().numpy()

    def predict_action(self, x: np.ndarray) -> np.ndarray:
        """Predict action that maximizes Q-value"""
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        action = self.get_action(x)
        return action.detach().cpu().numpy()

    def predict_barrier(
        self, x: torch.Tensor, barrier_threshold: float = None
    ) -> np.ndarray:
        """Predict barrier values"""
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        barrier = self.get_barrier(x, barrier_threshold=barrier_threshold)
        return barrier.detach().cpu().numpy()

    def predict_next_barrier(
        self, x: torch.Tensor, u: torch.Tensor, barrier_threshold: float = None
    ) -> np.ndarray:
        """Predict safety of next state according to barrier threshold

        Input:
            x: current state [batch_size, state_dim]
            u: action [batch_size]
        """
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        u = torch.tensor(u, dtype=torch.long, device=self.device)
        barrier = self.get_next_barrier(x, u, barrier_threshold=barrier_threshold)
        return barrier.detach().cpu().numpy()


class QNetworkEnsemble(QNetwork):
    def __init__(self, envs, models: List[nn.Module], **kwargs):
        super(QNetworkEnsemble, self).__init__(envs, **kwargs)
        self.envs = envs
        self.models = nn.ModuleList(models)

    def get_num_models(self):
        return len(self.models)

    def forward(self, x, reduction: str = "min"):
        assert reduction in ["min", "max", "mean"]
        q_values = torch.stack([model(x) for model in self.models], dim=0)
        if reduction == "min":
            return torch.min(q_values, dim=0)[0]
        elif reduction == "max":
            return torch.max(q_values, dim=0)[0]
        elif reduction == "mean":
            return torch.mean(q_values, dim=0)
