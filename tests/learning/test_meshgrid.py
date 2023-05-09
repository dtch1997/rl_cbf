import numpy as np
import torch
import itertools
from typing import List
from torch import Tensor 

from rl_cbf.learning.torch_utils import make_grid

# Test semantics of torch meshgrid
def is_grid_valid(grid) -> bool:
    grid_shape = grid.shape[:-1]  # Get the grid shape without the last dimension
    grid_coords = itertools.product(*[range(s) for s in grid_shape])  # Use the correct grid shape for generating coordinates

    for coord in grid_coords:
        coord_np = np.array(coord)
        grid_value_np = grid[coord].numpy()
        if not np.isclose(grid_value_np, coord_np).all():
            return False
    return True


def test_make_grid_2d():
    grid_2d = make_grid([torch.arange(2), torch.arange(2)])
    assert is_grid_valid(grid_2d)

def test_make_grid_3d():
    grid_3d = make_grid([torch.arange(2), torch.arange(2), torch.arange(2)])
    assert is_grid_valid(grid_3d)

def test_make_grid_4d():
    grid_4d = make_grid([torch.arange(2), torch.arange(2), torch.arange(2), torch.arange(2)])
    assert is_grid_valid(grid_4d)