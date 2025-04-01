import numpy as np


def normalized_correlation(a, b):
    a_norm = (a - np.mean(a)) / np.std(a)
    b_norm = (b - np.mean(b)) / np.std(b)
    return np.correlate(a_norm.flatten(), b_norm.flatten())[0] / len(a_norm.flatten())
