""" Collection of wrappers for safety reward. """

import gym
from rl_cbf.envs import safety_env


class Rewarder:
    def __init__(self, env: safety_env.SafetyEnv):
        self.env = env

    def modify_reward(self, state, reward):
        raise NotImplementedError


class IdentityRewarder(Rewarder):
    def modify_reward(self, state, reward):
        return reward


class ZeroOneRewarder(Rewarder):
    def modify_reward(self, state, reward):
        """Return 1 if safe, 0 if unsafe."""
        return 1.0 - self.env.is_unsafe_th(state)


class ConstantPenaltyRewarder(Rewarder):
    def __init__(self, env: safety_env.SafetyEnv, penalty=1.0):
        super().__init__(env)
        self.penalty = penalty

    def modify_reward(self, state, reward):
        """Penalize unsafe states by unsafe_penalty."""
        return reward - self.penalty * self.env.is_unsafe_th(state)
