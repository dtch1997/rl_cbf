
import numpy as np

import torch 
import torch.nn as nn

from typing import List

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self, env, 
                 hidden_dim_1: int = 120,
                 hidden_dim_2: int = 84,
                 enable_bump_parametrization: bool = False,
                 device: str = 'cuda'):
        super().__init__()
        self.enable_bump_parametrization = enable_bump_parametrization
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.device = device

        if enable_bump_parametrization:
            self.network =  nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
                nn.ELU(),
                nn.Linear(120, 84),
                nn.ELU(),
                nn.Linear(84, env.single_action_space.n),
                nn.Sigmoid()
            )
            self.max = 100
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
        parser.add_argument('--hidden-dim-1', type=int, default=120)
        parser.add_argument('--hidden-dim-2', type=int, default=84)
        parser.add_argument('--device', type=str, default='cuda')
        return parser
    
    @staticmethod
    def from_argparse_args(env, args):
        return QNetwork(
            env, 
            enable_bump_parametrization=args.enable_bump_parametrization, 
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2, 
            device=args.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_bump_parametrization:
            return self.max * self.network(x)
        else:
            return self.network(x)
        
    def predict(self, states: np.ndarray) -> np.ndarray:
        states = np.array(states)
        states_th = torch.Tensor(states.astype(np.float32)).to(self.device)
        q_values = self.forward(states_th).detach().cpu().numpy()
        return q_values
    
    def predict_value(self, states: np.ndarray) -> np.ndarray:
        q_values = self.predict(states)
        return np.max(q_values, axis=-1)
    
    def predict_action(self, states: np.ndarray) -> np.ndarray:
        q_values = self.predict(states)
        return np.argmax(q_values, axis=-1)

class QNetworkEnsemble(QNetwork):
    def __init__(self, envs, models: List[nn.Module]):
        super(QNetworkEnsemble, self).__init__()
        self.envs = envs
        self.models = nn.ModuleList(models)

    def get_num_models(self):
        return len(self.models)

    def forward(self, x, reduction: str ='min'):
        assert reduction in ['min', 'max', 'mean']
        q_values = torch.stack([model(x) for model in self.models], dim=0)
        if reduction == 'min':
            return torch.min(q_values, dim=0)[0]
        elif reduction == 'max':
            return torch.max(q_values, dim=0)[0]
        elif reduction == 'mean':
            return torch.mean(q_values, dim=0)