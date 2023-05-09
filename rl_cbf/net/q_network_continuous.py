
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import List

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class QNetworkContinuousAction(nn.Module):

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

        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.array(env.single_action_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 120),
            nn.ELU(),
            nn.Linear(120, 84),
            nn.ELU(),
            nn.Linear(84, 1),
        )
        if self.enable_bump_parametrization:
            self.max = 100

    @staticmethod
    def add_argparse_args(parser: 'argparse.ArgumentParser'):
        parser.add_argument('--enable-bump-parametrization', action='store_true')
        parser.add_argument('--hidden-dim-1', type=int, default=120)
        parser.add_argument('--hidden-dim-2', type=int, default=84)
        parser.add_argument('--device', type=str, default='cuda')
        return parser
    
    @staticmethod
    def from_argparse_args(env, args):
        return QNetworkContinuousAction(
            env, 
            enable_bump_parametrization=args.enable_bump_parametrization, 
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2, 
            device=args.device
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        input = torch.cat([x, a], dim=-1)
        output = self.network(input)

        if self.enable_bump_parametrization and apply_sigmoid:
            return self.max * torch.sigmoid(output)
        else:
            return output
        
    def predict(self, states: np.ndarray, apply_sigmoid: bool = True) -> np.ndarray:
        states = np.array(states)
        states_th = torch.Tensor(states.astype(np.float32)).to(self.device)
        q_values = self.forward(states_th, apply_sigmoid).detach().cpu().numpy()
        return q_values
    
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean