import gym 
import numpy as np
import torch.nn as nn
import pandas as pd

import rl_cbf.envs
from typing import Optional
from rl_cbf.net.q_network import QNetwork

class RolloutEvaluator:

    features = (
        'values',
        'states',
        'actions',
        'rewards',
        'dones',
        'td_errors',
        'episode_length',
        'episode_return',
    )

    def __init__(self):
        self.eval_env = gym.make('DiverseCartPole-v1')

    def get_default_initial_states(self, n_states: int = 1):
        states = np.zeros((n_states, self.eval_env.observation_space.shape[0]))
        for i in range(n_states):
            states[i] = self.eval_env.reset()
        return states
    
    def sample_grid_points(self, n_grid_points: int = 1):
        """ Sample grid points from the state space """
        states = np.zeros((n_grid_points, self.eval_env.observation_space.shape[0]))
        for i in range(n_grid_points):
            states[i] = self.eval_env.observation_space.sample()
        return states
    
    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['mean_values'] = df['values'].apply(lambda x: np.mean(x))
        df['mean_td_errors'] = df['td_errors'].apply(lambda x: np.mean(x))
        return df
    
    def evaluate(self, model: QNetwork, strategy: str) -> pd.DataFrame:
        if strategy == 'rollout':
            return self.evaluate_rollout(model)
        elif strategy == 'grid':
            return self.evaluate_grid(model)
        else:
            raise ValueError(f'Unknown strategy {strategy}')

    def evaluate_grid(self, model: QNetwork, n_grid_points: 10000) -> pd.DataFrame:
        grid_points = self.sample_grid_points(n_grid_points)
        return self.evaluate_rollout(model, grid_points, max_episode_length=1)

    def evaluate_rollout(self, model: QNetwork, 
                         initial_states: Optional[np.ndarray] = None, 
                         max_episode_length: int = 500
                        ) -> pd.DataFrame:
        """ Return pd.DataFrame of rollout data
        
        Each row is 1 episode 
        """
        rows = []
        if initial_states is None:
            initial_states = self.get_default_initial_states(10)

        for i, initial_state in enumerate(initial_states):
            initial_state = self.eval_env.reset_to(initial_state)
            done = False
            
            states = []
            actions = []
            values = []
            rewards = []
            dones = []
            td_errors = []

            state = initial_state
            t = 0
            while not done and t < max_episode_length:
                t += 1
                states.append(state)
                action = model.predict_action(state)
                actions.append(action)
                value = model.predict_value(state)
                values.append(value)
                next_state, reward, done, info = self.eval_env.step(action)
                rewards.append(reward)
                dones.append(done)
                
                next_value = model.predict_value(state)
                td_error = np.abs(value - reward - 0.99 * next_value)

                td_errors.append(td_error)
                state = next_state

            episode_length = len(states)
            # Convert to np.ndarray
            states = np.array(states)
            actions = np.array(actions)
            values = np.array(values)
            rewards = np.array(rewards)
            dones = np.array(dones)
            td_errors = np.array(td_errors)
            episode_return = np.sum(rewards)

            rows.append({
                'values': values,
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'td_errors': td_errors,
                'episode_length': episode_length,
                'episode_return': episode_return,
            })

        df = pd.DataFrame(rows)
        df = self.postprocess(df)
        return df