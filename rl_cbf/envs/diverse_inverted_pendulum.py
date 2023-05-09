import math
import numpy as np

from typing import Optional 
from gym.envs.mujoco import inverted_pendulum

class DiverseInvertedPendulumEnv(inverted_pendulum.InvertedPendulumEnv):
    """ 
    
    Modified done condition to be same as base CartPole
    Modified initial states to be more diverse
    """

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        theta_threshold = 12 * 2 * math.pi / 360
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= theta_threshold) and (np.abs(ob[0]) <= 2.4)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.np_random.uniform(
            low = np.array([-2.0, -2.0]),
            high = np.array([-0.05, 0.05])
        )
        qvel = self.np_random.uniform(
            size=self.model.nv, low=-0.05, high=0.05
        )
        self.set_state(qpos, qvel)
        return self._get_obs()