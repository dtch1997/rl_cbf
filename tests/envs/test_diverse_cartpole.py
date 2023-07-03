import unittest
import gym
import numpy as np

import rl_cbf.envs
from rl_cbf.envs.diverse_cartpole import DiverseCartPoleEnv


class TestDiverseCartPoleEnv(unittest.TestCase):
    def test_reset(self):
        env = DiverseCartPoleEnv()
        state = env.reset()
        self.assertEqual(state.shape, (4,))
        self.assertTrue(state[0] >= -2.0)
        self.assertTrue(state[0] <= 2.0)

    def test_reset_with_seed(self):
        env1 = DiverseCartPoleEnv()
        state1 = env1.reset(seed=42)

        env2 = DiverseCartPoleEnv()
        state2 = env2.reset(seed=42)

        self.assertTrue(np.all(state1 == state2))

    def test_step(self):
        env = DiverseCartPoleEnv()
        state = env.reset()
        next_state, reward, done, info = env.step(0)

        self.assertEqual(next_state.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_reset_to(self):
        env = DiverseCartPoleEnv()
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        env.reset_to(state)
        self.assertTrue(np.all(env.state == state))

    def test_gym_make(self):
        env = gym.make("DiverseCartPole-v1")
