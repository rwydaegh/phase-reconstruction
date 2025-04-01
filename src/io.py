"""
measurement_data/
Module for handling data input/output operations.
"""

import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

def load_measurement_data(file_path):
    """
    Load measurement data from pickle file
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]
    frequency = data.get("frequency", 28e9)

    if isinstance(results, list):
        max_signal_length = max(len(signal) for signal in results)
        processed_results = []
        for signal in results:
            if len(signal) < max_signal_length:
                padded_signal = signal + [np.nan] * (max_signal_length - len(signal))
                processed_results.append(padded_signal)
            else:
                processed_results.append(signal)
        results = np.array(processed_results).astype(float)

    if isinstance(points_continuous, list):
        points_continuous = np.array(points_continuous)

    if isinstance(points_discrete, list):
        points_discrete = np.array(points_discrete)

    return {
        "results": results,
        "continuous_axis": continuous_axis,
        "discrete_axis": discrete_axis,
        "points_continuous": points_continuous,
        "points_discrete": points_discrete,
        "frequency": frequency,
    }
