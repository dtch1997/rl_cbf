import gym 
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import rl_cbf.envs
from typing import Optional
from rl_cbf.net.q_network import QNetwork

class DQNCartPoleEvaluator:

    def __init__(self,
                 env_id: str = 'DiverseCartPole-v1',
                 capture_video: bool = False,
                 video_path: str = 'eval'):
        self.eval_env = gym.make(env_id)
        self.capture_video = capture_video
        if self.capture_video:
            self.eval_env = gym.wrappers.RecordVideo(self.eval_env, f'./videos/{video_path}')

    def get_default_initial_states(self, n_states: int = 1):
        states = np.zeros((n_states, self.eval_env.observation_space.shape[0]))
        for i in range(n_states):
            states[i] = self.eval_env.reset()
        return states
    
    def sample_grid_points(self, n_grid_points: int = 1):
        """ Sample grid points from the state space """
        high = self.eval_env.observation_space.high
        # Clip state space to 10
        high = np.clip(high, 0, 10)
        states = np.random.uniform(low=-high, high=high, size=(n_grid_points, 4))
        return states
    
    def compute_episode_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Compute episode statistics """
        episode_lengths = df.groupby('episode')['timestep'].max()
        episode_returns = df.groupby('episode')['reward'].sum()
        episode_statistics = pd.DataFrame({
            'episode_length': episode_lengths,
            'episode_return': episode_returns,
        })
        return episode_statistics
    
    def compute_overall_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Compute overall statistics """
        overall_statistics = pd.DataFrame({
            'mean_value': df['value'].mean(),
            'mean_td_error': df['td_error'].mean(),
            'max_td_error': df['td_error'].max(),
            '75th_percentile_td_error': df['td_error'].quantile(0.75),
        }, index=[0])
        return overall_statistics
    
    def evaluate(self, model: QNetwork, strategy: str) -> pd.DataFrame:
        if strategy == 'rollout':
            return self.evaluate_rollout(model)
        elif strategy == 'grid':
            return self.evaluate_grid(model)
        else:
            raise ValueError(f'Unknown strategy {strategy}')

    def evaluate_grid(self, model: QNetwork, n_grid_points: int = 10000) -> pd.DataFrame:
        grid_points = self.sample_grid_points(n_grid_points)
        return self.evaluate_rollout(model, grid_points, max_episode_length=1)

    def evaluate_rollout(self, model: QNetwork, 
                         initial_states: Optional[np.ndarray] = None, 
                         num_rollouts: int = 10,
                         max_episode_length: int = 500
                        ) -> pd.DataFrame:
        """ Return pd.DataFrame of rollout data
        
        Each row is 1 timestep of 1 rollout
        """
        rows = []
        if initial_states is None:
            initial_states = self.get_default_initial_states(num_rollouts)

        for episode_idx, initial_state in enumerate(initial_states):
            # Reset the time limit
            self.eval_env.reset()
            # Reset DiverseCartPole-v1 to appropriate initial state
            initial_state = self.eval_env.reset_to(initial_state)
            done = False

            state = initial_state
            timestep = 0
            while not done and timestep < max_episode_length:
                timestep += 1
                action = model.predict_action(state)
                value = model.predict_value(state)
                next_state, reward, done, info = self.eval_env.step(action)
                next_value = model.predict_value(next_state)
                td_error = np.abs(value - reward - 0.99 * next_value)

                rows.append({
                    'episode': episode_idx,
                    'timestep': timestep,
                    'value': value,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'next_state': next_state,
                    'td_error': td_error,
                })

                state = next_state

        df = pd.DataFrame(rows)
        return df
    
    def evaluate_barrier(self, model: QNetwork, 
                         initial_states: np.ndarray,
                        ) -> pd.DataFrame:
        """ Return pd.DataFrame of rollout data
        
        Each row is 1 timestep of 1 rollout
        """
        rows = []
        for episode_idx, initial_state in enumerate(initial_states):
            for action in range(self.eval_env.action_space.n):

                # Reset the time limit
                self.eval_env.reset()
                # Reset DiverseCartPole-v1 to appropriate initial state
                initial_state = self.eval_env.reset_to(initial_state)
                state = initial_state

                q_value = model.predict(state)[action]
                value = model.predict_value(state)
                barrier_value = model.predict_value(state, apply_sigmoid=False)
                next_state, reward, done, info = self.eval_env.step(action)
                next_value = model.predict_value(next_state)
                next_barrier_value = model.predict_value(next_state, apply_sigmoid=False)
                td_error = np.abs(q_value - reward - 0.99 * next_value)

                rows.append({
                    'episode': episode_idx, 
                    'q_value': q_value,
                    'value': value,
                    'barrier_value': barrier_value,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'next_state': next_state,
                    'td_error': td_error,
                    'next_value': next_value,
                    'next_barrier_value': next_barrier_value,
                })

                state = next_state

        df = pd.DataFrame(rows)
        return df
    
    def calculate_accuracy(self, df: pd.DataFrame) -> float:
        """ Determine whether barrier condition (i) is satisfied """
        accuracy = 0
        df = df[['episode', 'barrier_value', 'state']].drop_duplicates(['episode'])
        state = df['state']
        state = np.stack(state.to_numpy())

        is_unsafe = self.eval_env.is_done(state)
        barrier_values = df['barrier_value'].to_numpy()
        accuracy = (barrier_values[is_unsafe] < 0).mean()       
        
        return accuracy
    
    def calculate_valid_1(self, df):
        """ Check whether condition 1 is satisfied """
        df = df[['episode', 'barrier_value', 'state']].drop_duplicates(['episode'])
        state = df['state']
        state = np.stack(state.to_numpy())

        is_unsafe = self.eval_env.is_done(state)
        is_safe = ~is_unsafe
        barrier_values = df['barrier_value'].to_numpy()
        valid_1 = (is_safe | (barrier_values < 0))

        return valid_1
    
    def calculate_validity(self, df: pd.DataFrame, alphas: np.ndarray) -> np.ndarray:
        """ Determine whether barrier condition is satisfied for each point """
        df = df[['episode', 'barrier_value', 'next_barrier_value', 'state']]
        valid_1 = self.calculate_valid_1(df)

        barrier_values = df[['episode', 'barrier_value']].drop_duplicates(['episode'])
        sup_next_barrier_values = df[['episode', 'next_barrier_value']] \
            .groupby('episode').max()

        df_merged = pd.merge(barrier_values, sup_next_barrier_values, on='episode')  

        validities = np.zeros_like(alphas)
        for i, one_minus_alpha in enumerate(alphas):
            valid_2 = (
                (df_merged['barrier_value'] < 0) | 
                (df_merged['next_barrier_value'] >= one_minus_alpha * df_merged['barrier_value'])
            )
            validities[i] = (valid_1 * valid_2).mean()
        return validities
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = QNetwork.add_argparse_args(parser)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--video-path', type=str, default='')
    parser.add_argument('--results-path', type=str, default='results.csv')
    args = parser.parse_args()

    _envs = gym.vector.SyncVectorEnv([lambda: gym.make('DiverseCartPole-v1')])
    model = QNetwork.from_argparse_args(_envs, args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    capture_video = (args.video_path != '')
    evaluator = DQNCartPoleEvaluator(capture_video=capture_video, video_path=args.video_path)
    
    barrier_df = evaluator.evaluate_barrier(model, evaluator.sample_grid_points(10000))
    barrier_accuracy = evaluator.calculate_accuracy(barrier_df)
    barrier_validity = evaluator.calculate_validity(barrier_df, np.linspace(0, 1, 100))
    
    print("Barrier accuracy: ", barrier_accuracy)
    print("Barrier validity: ", barrier_validity)
    
    df = evaluator.evaluate_grid(model, n_grid_points=10000)
    df.to_csv(args.results_path, index=False)