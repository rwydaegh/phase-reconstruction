import numpy as np


def normalized_rmse(a, b):
    range_a = np.max(a) - np.min(a)
    if range_a > 1e-10:
        return np.sqrt(np.mean((a - b) ** 2)) / range_a
    else:
        return np.inf
