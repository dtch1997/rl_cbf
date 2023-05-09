import math
import unittest
import numpy as np

from gym.envs.mujoco import inverted_pendulum
from rl_cbf.envs.diverse_inverted_pendulum import DiverseInvertedPendulumEnv

class TestDiverseInvertedPendulumEnv(unittest.TestCase):

    def test_diverse_inverted_pendulum_env(self):
        env = DiverseInvertedPendulumEnv()
        self.assertTrue(isinstance(env, inverted_pendulum.InvertedPendulumEnv))