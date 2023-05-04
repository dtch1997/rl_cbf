import gym 

import gym
import numpy as np

from rl_cbf.envs.wrappers import RecordObservationActionHistory

def make_env(
    env_id: str, 
    seed: int,
    idx: int, 
    capture_video: bool, 
    run_name: str,
    record_obs_act_hist: bool = False
):
    
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if record_obs_act_hist:
            env = RecordObservationActionHistory(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
