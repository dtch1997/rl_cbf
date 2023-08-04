import gym
import numpy as np
import unittest

import rl_cbf.envs  # noqa: F401


class TestSafetyEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("SafetyWalker2d-v3")

    def test_is_unsafe(self):
        states = np.array(
            [
                [0.9, 0.0] + [0] * 15,  # safe state
                [0.7, 0.0] + [0] * 15,  # unsafe due to height
                [0.9, 1.1] + [0] * 15,  # unsafe due to angle
            ]
        )
        is_unsafe = self.env.is_unsafe(states)
        np.testing.assert_array_equal(is_unsafe, np.array([False, True, True]))

    def test_sample_states(self):
        n_states = 5
        states = self.env.sample_states(n_states)
        self.assertEqual(states.shape, (n_states, 17))
        self.assertTrue((states >= -2.0).all() and (states <= 2.0).all())

    def test_reset_to(self):
        state = self.env.sample_states(1)[0]
        new_state = self.env.reset_to(state)
        self.assertEqual(new_state.shape, (17,))
