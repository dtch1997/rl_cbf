
import numpy as np

import torch 
import torch.nn as nn

from typing import List
from siren_pytorch import Siren

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self, env, 
                 hidden_dim_1: int = 120,
                 hidden_dim_2: int = 84,
                 enable_bump_parametrization: bool = False,
                 enable_siren_layer: bool = False,
                 min_value: float = 0,
                 max_value: float = 100,
                 device: str = 'cuda'):
        super().__init__()
        self.enable_bump_parametrization = enable_bump_parametrization
        # Only used in case of bump parametrization
        self.max = max_value
        # Not implemented yet
        self.min = min_value
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.device = device

        if enable_siren_layer:
            self.network = nn.Sequential(
                Siren(np.array(env.single_observation_space.shape).prod(), hidden_dim_1, c=6, w0=30.),
                nn.Linear(120, 84),
                nn.ELU(),
                nn.Linear(84, env.single_action_space.n),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
                nn.ELU(),
                nn.Linear(120, 84),
                nn.ELU(),
                nn.Linear(84, env.single_action_space.n),
            )

    @staticmethod
    def add_argparse_args(parser: 'argparse.ArgumentParser'):
        parser.add_argument('--enable-bump-parametrization', action='store_true')
        parser.add_argument('--enable-siren-layer', action='store_true')
        parser.add_argument('--min-value', type=float, default=0)
        parser.add_argument('--max-value', type=float, default=100)
        parser.add_argument('--hidden-dim-1', type=int, default=120)
        parser.add_argument('--hidden-dim-2', type=int, default=84)
        parser.add_argument('--device', type=str, default='cuda')
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

    def forward(self, x: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        if self.enable_bump_parametrization and apply_sigmoid:
            range = self.max - self.min
            return range * torch.sigmoid(self.network(x)) + self.min
        else:
            return self.network(x)
        
    def predict(self, states: np.ndarray, apply_sigmoid: bool = True) -> np.ndarray:
        states = np.array(states)
        states_th = torch.Tensor(states.astype(np.float32)).to(self.device)
        q_values = self.forward(states_th, apply_sigmoid).detach().cpu().numpy()
        return q_values
    
    def predict_value(self, states: np.ndarray, apply_sigmoid: bool = True) -> np.ndarray:
        q_values = self.predict(states, apply_sigmoid)
        return np.max(q_values, axis=-1)
    
    def predict_action(self, states: np.ndarray, apply_sigmoid: bool = True) -> np.ndarray:
        q_values = self.predict(states, apply_sigmoid)
        return np.argmax(q_values, axis=-1)

class QNetworkEnsemble(QNetwork):
    def __init__(self, envs, models: List[nn.Module], **kwargs):
        super(QNetworkEnsemble, self).__init__(envs, **kwargs)
        self.envs = envs
        self.models = nn.ModuleList(models)

    def get_num_models(self):
        return len(self.models)

    def forward(self, x,  apply_sigmoid: bool = True, reduction: str ='min'):
        assert reduction in ['min', 'max', 'mean']
        q_values = torch.stack([model(x, apply_sigmoid) for model in self.models], dim=0)
        if reduction == 'min':
            return torch.min(q_values, dim=0)[0]
        elif reduction == 'max':
            return torch.max(q_values, dim=0)[0]
        elif reduction == 'mean':
            return torch.mean(q_values, dim=0)