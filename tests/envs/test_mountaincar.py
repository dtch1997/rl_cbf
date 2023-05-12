import unittest
import numpy as np
from rl_cbf.envs.mountaincar import BaseMountainCarEnv

class TestBaseMountainCarEnv(unittest.TestCase):

    def setUp(self):
        self.env = BaseMountainCarEnv()

    def test_sample_states(self):
        states = self.env.sample_states(10)
        self.assertEqual(states.shape, (10, 2))
        self.assertTrue(np.all(states >= self.env.observation_space.low))
        self.assertTrue(np.all(states <= self.env.observation_space.high))

    def test_reset_to(self):
        state = np.array([0.5, 0.1])
        self.env.reset_to(state)
        np.testing.assert_array_equal(self.env.state, state)

    def test_is_done(self):
        self.env.state = np.array([self.env.goal_position, self.env.goal_velocity])
        self.assertTrue(self.env.is_done(self.env.state))

        self.env.state = np.array([self.env.goal_position-0.1, self.env.goal_velocity-0.1])
        self.assertFalse(self.env.is_done(self.env.state))

    def test_step(self):
        self.env.reset()
        initial_state = self.env.state.copy()
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(state.shape, (2,))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)
        self.assertNotEqual(state.tolist(), initial_state.tolist())

        # Test done condition
        self.env.state = np.array([self.env.goal_position, self.env.goal_velocity])
        state, reward, done, _ = self.env.step(1)
        self.assertTrue(done)
        self.assertEqual(reward, 0)

if __name__ == '__main__':
    unittest.main()
