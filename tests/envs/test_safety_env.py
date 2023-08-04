import numpy as np
import torch

import rl_cbf.envs  # noqa: F401
from rl_cbf.envs.safety_env import SafetyWalker2dEnv, SafetyAntEnv, SafetyHopperEnv


def test_safety_walker_env():
    env = SafetyWalker2dEnv()

    states = torch.from_numpy(env.sample_states(100))
    unsafe_states = env.is_unsafe_th(states)

    assert states.shape == (100, 17), "States shape mismatch"
    assert unsafe_states.shape == (100, 1), "Unsafe states shape mismatch"

    height = states[..., 0]
    angle = states[..., 1]
    assert all(
        (height > 0.8) & (height < 2.0) & (angle > -1.0) & (angle < 1.0)
        == (~unsafe_states.bool()).squeeze()
    ), "Unsafe state mismatch"


def test_safety_ant_env():
    env = SafetyAntEnv()

    states = torch.from_numpy(np.random.uniform(low=-2.0, high=2.0, size=(100, 17)))
    unsafe_states = env.is_unsafe_th(states)

    assert states.shape == (100, 17), "States shape mismatch"
    assert unsafe_states.shape == (100, 1), "Unsafe states shape mismatch"

    height = states[..., 0]
    assert all(
        (height > 0.2) & (height < 1.0) == (~unsafe_states.bool()).squeeze()
    ), "Unsafe state mismatch"


def test_safety_hopper_env():
    env = SafetyHopperEnv()

    states = torch.from_numpy(np.random.uniform(low=-2.0, high=2.0, size=(100, 17)))
    unsafe_states = env.is_unsafe_th(states)

    assert states.shape == (100, 17), "States shape mismatch"
    assert unsafe_states.shape == (100, 1), "Unsafe states shape mismatch"

    height = states[..., 0]
    ang = states[..., 1]
    remainder = states[..., 2:]
    assert all(
        (height > 0.7) & (torch.abs(ang) < 0.2) & (torch.abs(remainder) < 100.0).all()
        == (~unsafe_states.bool()).squeeze()
    ), "Unsafe state mismatch"
