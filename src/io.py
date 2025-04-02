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



def sample_measurement_data(data, target_resolution=50):
    """
    Sample measurement data to reduce dimensionality to the target resolution

    Creates approximately square output by sampling both axes to be close to the target resolution.
    """
    results = data["results"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]

    n_discrete = len(points_discrete)
    n_continuous = len(points_continuous)

    logger.info(f"Original data dimensions: {n_discrete}×{n_continuous}")

    assert target_resolution <= n_discrete, (
        f"Target resolution {target_resolution} must be less than or equal to "
        f"the original resolution of discrete axis {n_discrete}"
    )
    # Ensure target resolution does not exceed continuous axis either
    assert target_resolution <= n_continuous, (
        f"Target resolution {target_resolution} must be less than or equal to "
        f"the original resolution of continuous axis {n_continuous}"
    )

    target_n_discrete = min(target_resolution, n_discrete)
    target_n_continuous = min(target_resolution, n_continuous)

    logger.info(f"Target dimensions: {target_n_discrete}×{target_n_continuous}")

    discrete_indices = np.linspace(0, n_discrete - 1, target_n_discrete, dtype=int)
    continuous_indices = np.linspace(0, n_continuous - 1, target_n_continuous, dtype=int)

    sampled_points_discrete = points_discrete[discrete_indices]
    sampled_points_continuous = points_continuous[continuous_indices]
    # Use np.ix_ for safe indexing, especially if results is already a numpy array
    sampled_results = results[np.ix_(discrete_indices, continuous_indices)]

    logger.info(
        f"Sampled data dimensions: {len(sampled_points_discrete)}×{len(sampled_points_continuous)}"
    )

    sampled_data = data.copy()
    sampled_data["results"] = sampled_results
    sampled_data["points_discrete"] = sampled_points_discrete
    sampled_data["points_continuous"] = sampled_points_continuous

    return sampled_data
