"""
Module for handling data input/output operations.
"""

import logging
import pickle
import numpy as np
import os

logger = logging.getLogger(__name__)



def load_measurement_data(file_path):
    """
    Load measurement data from pickle file
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    logger.info(f"Loaded measurement data with keys: {list(data.keys())}")

    # Extract relevant data
    results = data["results"]
    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]
    frequency = data.get("frequency", 28e9)

    # Handle variable length lists by padding with np.nan
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

    # Convert points_discrete to numpy array if it's a list
    if isinstance(points_discrete, list):
        points_discrete = np.array(points_discrete)

    logger.info(f"Measurement data shape: {results.shape}")
    logger.info(
        f"Continuous axis ({continuous_axis}): {len(points_continuous)} points from "
        f"{np.min(points_continuous):.1f} to {np.max(points_continuous):.1f} mm"
    )
    logger.info(
        f"Discrete axis ({discrete_axis}): {len(points_discrete)} points from "
        f"{np.min(points_discrete):.1f} to {np.max(points_discrete):.1f} mm"
    )
    logger.info(
        f"Frequency: {frequency/1e9:.2f} GHz (wavelength: {299792458/(frequency/1e9)/1e6:.2f} mm)"
    )

    # Return relevant data
    return {
        "results": results,
        "continuous_axis": continuous_axis,
        "discrete_axis": discrete_axis,
        "points_continuous": points_continuous,
        "points_discrete": points_discrete,
        "frequency": frequency,
    }

# Placeholder for future functions