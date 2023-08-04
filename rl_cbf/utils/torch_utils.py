import numpy as np
import torch


def torchify(x: np.ndarray, device, dtype=torch.float32):
    return torch.from_numpy(x).to(device).type(dtype)
