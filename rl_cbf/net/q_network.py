
import numpy as np

import torch 
import torch.nn as nn

from typing import List

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, enable_bounded_parametrization: bool = False):
        super().__init__()
        self.enable_bounded_parametrization = enable_bounded_parametrization
        if enable_bounded_parametrization:
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

    def forward(self, x):
        if self.enable_bounded_parametrization:
            return self.max * self.network(x)
        else:
            return self.network(x)
    
    def predict(self, state: np.ndarray):
        # Get the optimal action
        state = np.array(state)
        state_th = torch.Tensor(state)
        return torch.argmax(self.forward(state_th), dim=-1).detach().numpy()
 
    def predict_value(self, state: np.ndarray, action: np.ndarray):
        # Get the state value
        state = np.array(state)
        state_th = torch.Tensor(state)
        return torch.max(self.forward(state_th), dim=-1)[0].detach().numpy()
    
class QNetworkEnsemble(nn.Module):
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
        
    def predict_q_values(self, state: np.ndarray, reduction:str ='min'):
        # Get the q-values for each action
        state = np.array(state)
        state_th = torch.Tensor(state)
        q_values = self.forward(state_th, reduction).detach().numpy()
        return q_values
    
    def predict_q_value(self, state: np.ndarray, action: np.ndarray, reduction:str ='min'):
        # Get the q-value for a given action
        q_values = self.predict_q_values(state, reduction)
        return q_values[..., action]

    def predict_action(self, state: np.ndarray, reduction:str ='min'):
        # Get the optimal action
        q_values = self.predict_q_values(state, reduction)
        return np.argmax(q_values, axis=-1)
 
    def predict_value(self, state: np.ndarray, reduction:str ='min'):
        # Get the state value
        q_values = self.predict_q_values(state, reduction)
        return np.max(q_values, axis=-1)