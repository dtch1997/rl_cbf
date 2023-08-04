import numpy as np
import gym
import abc
import torch


class SafetyEnv(abc.ABC, gym.Wrapper):
    """TODO: write safety env interface"""

    def step(self, action: np.ndarray):
        state, reward, _, info = super().step(action)
        is_unsafe = self.is_unsafe_th(torch.from_numpy(state)).numpy().item()
        # TODO: overwrite the done condition to match is_unsafe_th
        done = is_unsafe
        return state, reward, done, info

    @staticmethod
    @abc.abstractmethod
    def is_unsafe_th(states: torch.Tensor) -> torch.Tensor:
        """Return boolean array indicating whether states are unsafe

        Args:
            states: (batch_size, state_dim) array of states

        Returns:
            is_unsafe: (batch_size, 1) float array indicating whether states are unsafe
        """
        raise NotImplementedError

    def sample_states(self, n_states: int) -> np.ndarray:
        """Sample n_states from the environment

        Args:
            n_states: number of states to sample

        Returns:
            states: (n_states, state_dim) array of sampled states
        """
        return np.random.uniform(
            low=-2.0, high=2.0, size=(n_states, self.observation_space.shape[0])
        )


class SafetyWalker2dEnv(SafetyEnv):
    def __init__(self, env_id: str = "Walker2d-v3"):
        env = gym.make(env_id)
        super().__init__(env)

    @staticmethod
    def is_unsafe_th(states: torch.Tensor):
        height = states[..., 0]
        angle = states[..., 1]

        height_ok = torch.logical_and(height > 0.8, height < 2.0)
        angle_ok = torch.logical_and(angle > -1.0, angle < 1.0)
        is_safe = torch.logical_and(height_ok, angle_ok)
        return (~is_safe).float().view(-1, 1)

    def reset_to(self, state: np.ndarray):
        new_state = self.reset()
        # TODO: implement this
        return new_state


class SafetyAntEnv(SafetyEnv):
    def __init__(self, env_id: str = "Ant-v3"):
        env = gym.make(env_id)
        super().__init__(env)

    @staticmethod
    def is_unsafe_th(states: torch.Tensor):
        height = states[..., 0]
        is_safe = torch.logical_and(height > 0.2, height < 1.0)
        return (~is_safe).float().view(-1, 1)


class SafetyHopperEnv(SafetyEnv):
    def __init__(self, env_id: str = "Hopper-v3"):
        env = gym.make(env_id)
        super().__init__(env)

    @staticmethod
    def is_unsafe_th(states: torch.Tensor):
        height = states[..., 0]
        ang = states[..., 1]
        remainder = states[..., 2:]
        is_safe = torch.logical_and(height > 0.7, torch.abs(ang) < 0.2)
        is_safe = torch.logical_and(is_safe, (torch.abs(remainder) < 100.0).all())
        return (~is_safe).float().view(-1, 1)
