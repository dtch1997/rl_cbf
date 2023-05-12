import math
import numpy as np

from gym.envs.classic_control.mountain_car import MountainCarEnv
from rl_cbf.envs.wrappers import DiscretizeWrapper

class BaseMountainCarEnv(MountainCarEnv):
    """ MountainCarEnv with added functionality

    Define function for sampling states
    Termination condition is overridable 
    Define function for custom reset
    Modified reward to be correct 
    """
    
    def sample_states(self, n_states: int) -> np.ndarray:
        states = np.random.uniform(
            low = self.observation_space.low,
            high = self.observation_space.high,
            size = ((n_states, 2))
        )
        return states
    
    def reset_to(self, state: np.ndarray):
        self.reset()
        self.state = state
        return state
    
    def is_done(self, states):
        position, velocity = states[..., 0], states[..., 1]
        return np.logical_and(
            position >= self.goal_position,
            velocity >= self.goal_velocity
        )
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = 0.0 if done else -1.0
        return obs, reward, done, info

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        position, velocity = state[..., 0], state[..., 1]
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        idx = np.logical_and(position == self.min_position, velocity < 0)
        velocity[idx] = 0
        return np.stack([position, velocity], axis=-1)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Change here
        done = self.is_done(np.array(self.state)).item()
        reward = 0 if done else -1.0
        # End changes

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, done, {}

def make_discretized_mountaincar():
    env = BaseMountainCarEnv()
    env = DiscretizeWrapper(env, n_bins=20)
    return env