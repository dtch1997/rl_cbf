""" Collection of wrappers for safety reward. """

import gym
from rl_cbf.envs import safety_env


class ZeroOneWrapper(gym.RewardWrapper):
    """Wrapper that returns 1 if safe, 0 if unsafe."""

    def __init__(self, env: safety_env.SafetyEnv):
        super().__init__(env)

    def reward(self, reward):
        return 1.0 - self.env.is_unsafe(self.env.state)


class ConstantPenaltyWrapper(gym.RewardWrapper):
    """Wrapper that applies a constant penalty to unsafe states."""

    def __init__(self, env: safety_env.SafetyEnv, unsafe_penalty: float):
        super().__init__(env)
        self.unsafe_penalty = unsafe_penalty

    def reward(self, reward):
        return reward - self.unsafe_penalty * self.env.is_unsafe(self.env.state)
