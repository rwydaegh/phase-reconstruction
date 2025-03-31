import numpy as np


def normalized_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / (np.max(a) - np.min(a))
