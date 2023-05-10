import math
import numpy as np

from typing import Optional 
from rl_cbf.envs.decomposed_cartpole import DecomposedCartPole

class DiverseCartPoleEnv(DecomposedCartPole):
    """ CartPoleEnv with more diverse initial states """
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        start_x = self.np_random.uniform(low=-2.0, high=2.0)
        self.state[0] = start_x        
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
        
    def reset_to(self, state: np.ndarray):
        self.reset()
        self.state = state
        return state
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done: reward = 0
        return obs, reward, done, info