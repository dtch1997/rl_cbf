import torch

from typing import List


def make_grid(xs: List[torch.Tensor]):
    """Make a grid of tensors"""
    grid = torch.meshgrid(*xs)
    grid = torch.stack(grid, dim=-1)
    return grid
