""" Script to preprocess D4RL dataset for evaluating coverage, validity """

import d4rl
import gym
import numpy as np
from typing import Dict, Tuple


def parse_dataset_safety(
    dataset: Dict[str, np.ndarray], H: int = 20
) -> Dict[str, np.ndarray]:
    """Parse a D4RL dataset into safe and unsafe states"""
    episode_end_idx = np.nonzero(dataset["terminals"])[0]
    episode_start_idx = np.concatenate([[0], episode_end_idx[:-1]])

    is_assumed_safe = np.zeros_like(dataset["terminals"], dtype=np.int64)
    for start, end in zip(episode_start_idx, episode_end_idx):
        start = start.item()
        end = end.item()
        if end - start < H:
            continue
        is_assumed_safe[start : end - H] = 1

    dataset["is_assumed_safe"] = is_assumed_safe
    return dataset


if __name__ == "__main__":
    env = gym.make("walker2d-medium-v2")
    dataset = d4rl.qlearning_dataset(env)
    breakpoint()
