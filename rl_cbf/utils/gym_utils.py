import numpy as np
from gym import spaces 

def get_flattened_dim(space: spaces.Space):
    if isinstance(space, spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, spaces.Discrete):
        return space.n
    else:
        raise NotImplementedError