import numpy as np
import unittest
from gym.envs.classic_control import CartPoleEnv
from rl_cbf.envs.decomposed_cartpole import CartPoleAEnv, CartPoleBEnv, CartPoleCEnv, CartPoleDEnv, DecomposedCartPole

class TestDecomposedCartPole(unittest.TestCase):

    def test_decomposed_cartpole(self):
        env = DecomposedCartPole()
        states = env.sample_states(100)
        self.assertEqual(states.shape, (100, 4))

    def test_cart_pole_a_env(self):
        env = CartPoleAEnv()
        self.assertTrue(isinstance(env, CartPoleEnv))

        # Test a state where x is within the threshold
        state = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertFalse(env.is_done(state))

        # Test a state where x is outside the threshold
        state = np.array([env.x_threshold + 0.1, 0.0, 0.0, 0.0])
        self.assertTrue(env.is_done(state))

    def test_cart_pole_b_env(self):
        env = CartPoleBEnv()
        self.assertTrue(isinstance(env, CartPoleEnv))

        # Test a state where theta is within the threshold
        state = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertFalse(env.is_done(state))

        # Test a state where theta is outside the threshold
        state = np.array([0.0, 0.0, env.theta_threshold_radians + 0.1, 0.0])
        self.assertTrue(env.is_done(state))

    def test_cart_pole_c_env(self):
        env = CartPoleCEnv()
        self.assertTrue(isinstance(env, CartPoleEnv))

        # Test a state where x is within the threshold and theta is within the threshold
        state = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertFalse(env.is_done(state))

        state = np.array([env.x_threshold + 0.1, 0.0, 0.0, 0.0])
        self.assertFalse(env.is_done(state))
        state = np.array([-env.x_threshold - 0.1, 0.0, 0.0, 0.0])
        self.assertTrue(env.is_done(state))
        state = np.array([0.0, 0.0, env.theta_threshold_radians + 0.1, 0.0])
        self.assertFalse(env.is_done(state))
        state = np.array([0.0, 0.0, - env.theta_threshold_radians - 0.1, 0.0])
        self.assertTrue(env.is_done(state))

    def test_cart_pole_d_env(self):
        env = CartPoleDEnv()
        self.assertTrue(isinstance(env, CartPoleEnv))

        # Test a state where x is within the threshold and theta is within the threshold
        state = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertFalse(env.is_done(state))


        state = np.array([env.x_threshold + 0.1, 0.0, 0.0, 0.0])
        self.assertTrue(env.is_done(state))
        state = np.array([-env.x_threshold - 0.1, 0.0, 0.0, 0.0])
        self.assertFalse(env.is_done(state))
        state = np.array([0.0, 0.0, env.theta_threshold_radians + 0.1, 0.0])
        self.assertTrue(env.is_done(state))
        state = np.array([0.0, 0.0, - env.theta_threshold_radians - 0.1, 0.0])
        self.assertFalse(env.is_done(state))

if __name__ == '__main__':
    unittest.main()
