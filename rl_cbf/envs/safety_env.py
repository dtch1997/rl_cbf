import numpy as np
import gym
import abc

class SafetyEnv(abc.ABC, gym.Env):
    """ TODO: write safety env interface """    
    
    @staticmethod
    @abc.abstractmethod
    def is_unsafe(states: np.ndarray) -> np.ndarray:
        """ Return boolean array indicating whether states are unsafe 
        
        Args:
            states: (batch_size, state_dim) array of states

        Returns:
            is_unsafe: (batch_size,) boolean array indicating whether states are unsafe
        """
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def sample_states(n_states: int) -> np.ndarray:
        """ Sample n_states from the environment 
        
        Args:
            n_states: number of states to sample

        Returns:
            states: (n_states, state_dim) array of sampled states
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def reset_to(self, state: np.ndarray):
        """ Reset the environment to a particular state 
        
        Args:
            state: (state_dim,) array of state to reset to
        """
        raise NotImplementedError
    

class SafetyWalker2dEnv(SafetyEnv, gym.Wrapper):

    def __init__(self):
        env = gym.make('Walker2d-v3')
        super().__init__(env)

    @staticmethod
    def is_unsafe(states: np.ndarray):
        height = states[..., 0]
        angle = states[..., 1]

        height_ok = np.logical_and(height > 0.8, height < 2.0)
        angle_ok = np.logical_and(angle > -1.0, angle < 1.0)
        is_safe = np.logical_and(height_ok, angle_ok)
        return ~is_safe
    
    @staticmethod
    def sample_states(n_states: int):
        states = np.random.uniform(
            low = -2.0,
            high = 2.0,
            size = ((n_states, 17))
        )
        return states
    
    def reset_to(self, state: np.ndarray):
        new_state = self.reset()
        # TODO: implement this
        return new_state