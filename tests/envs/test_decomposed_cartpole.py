import numpy as np
import unittest
from gym.envs.classic_control import CartPoleEnv
from rl_cbf.envs.decomposed_cartpole import BaseCartPole

class TestDecomposedCartPole(unittest.TestCase):

    def test_base_cartpole(self):
        env = BaseCartPole()
        states = env.sample_states(100)
        self.assertEqual(states.shape, (100, 4))

if __name__ == '__main__':
    unittest.main()
