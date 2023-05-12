import numpy as np
import gym

class DiscretizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins):
        super().__init__(env)
        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins+1) for l, h in zip(env.observation_space.low, env.observation_space.high)]
        self.observation_space = gym.spaces.Discrete(n_bins * n_bins)

    def observation(self, obs):
        discretized = [np.digitize(x, val_bin) - 1 for x, val_bin in zip(obs, self.val_bins)]
        return discretized[0] * self.n_bins + discretized[1]

    def get_original_observation(self, disc_obs):
        row = disc_obs // self.n_bins
        col = disc_obs % self.n_bins
        return [self.val_bins[0][row], self.val_bins[1][col]]