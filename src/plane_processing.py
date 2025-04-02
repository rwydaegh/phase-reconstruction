# src/plane_processing.py
import logging
import os
import pickle

import numpy as np
from omegaconf import DictConfig

# Assuming io.py and utils might contain relevant functions later
from src.io import load_measurement_data, sample_measurement_data # Need to ensure these exist/are correct
# from src.utils.geometry_utils import create_plane_grid # Example hypothetical function

logger = logging.getLogger(__name__)

def process_plane_definitions(plane_configs: list, base_dir: str = ".") -> list:
    """
    Iterates through plane definitions from the config, processing each one.

    Args:
        plane_configs: List of DictConfig objects, each defining a measurement plane.
        base_dir: The base directory of the project (e.g., hydra's original working dir)
                  to resolve relative paths in config.

    Returns:
        List of processed plane data dictionaries.
    """
    processed_planes = []
    if not plane_configs:
        logger.warning("No measurement planes defined in the configuration.")
        return processed_planes

    for i, plane_cfg in enumerate(plane_configs):
        logger.info(f"Processing plane definition {i+1}: Name='{plane_cfg.get('name', 'Unnamed')}'")
        try:
            if plane_cfg.get('is_real_plane', False):
                processed_data = load_real_plane(plane_cfg, base_dir)
            else:
                processed_data = generate_simulated_plane(plane_cfg)

            if processed_data:
                # Add common info back for reference
                processed_data['name'] = plane_cfg.get('name', f'plane_{i}')
                processed_data['is_real_plane'] = plane_cfg.get('is_real_plane', False)
                processed_data['use_train'] = plane_cfg.get('use_train', False)
                processed_data['use_test'] = plane_cfg.get('use_test', False)
                processed_planes.append(processed_data)
            else:
                 logger.warning(f"Processing failed for plane: {plane_cfg.get('name', 'Unnamed')}")

        except Exception as e:
            logger.error(f"Error processing plane '{plane_cfg.get('name', 'Unnamed')}': {e}", exc_info=True)

    return processed_planes


def load_real_plane(plane_cfg: DictConfig, base_dir: str) -> dict | None:
    """
    Loads, samples, and translates a real measurement plane based on config.

    Args:
        plane_cfg: DictConfig for a single real plane.
        base_dir: Base directory to resolve relative data paths.

    Returns:
        Dictionary containing processed plane data ('coordinates', 'measured_magnitude', etc.)
        or None if loading fails.
    """
    logger.info(f"  Type: Real Plane")
    data_path_rel = plane_cfg.get('measured_data_path')
    target_resolution = plane_cfg.get('target_resolution')
    translation = np.array(plane_cfg.get('translation', [0.0, 0.0, 0.0]))

    if not data_path_rel:
        logger.error("  'measured_data_path' is missing in real plane config.")
        return None
    if not target_resolution:
        logger.error("  'target_resolution' is missing in real plane config.")
        return None

    # Resolve absolute path
    data_path_abs = os.path.join(base_dir, data_path_rel)
    if not os.path.exists(data_path_abs):
         logger.error(f"  Measurement data file not found: {data_path_abs}")
         return None

    logger.info(f"  Loading data from: {data_path_abs}")
    try:
        measurement_data = load_measurement_data(data_path_abs)
    except Exception as e:
        logger.error(f"  Failed to load measurement data from {data_path_abs}: {e}", exc_info=True)
        return None

    logger.info(f"  Sampling data to target resolution: {target_resolution}")
    try:
        sampled_data = sample_measurement_data(measurement_data, target_resolution)
    except Exception as e:
        logger.error(f"  Failed to sample measurement data for {data_path_abs}: {e}", exc_info=True)
        return None


    logger.info(f"  Creating measurement plane coordinates...")
    # TODO: Refactor the plane creation logic (centering, coordinate generation)
    # from measured_data_reconstruction.create_measurement_plane into a reusable function,
    # potentially here or in utils. For now, keep the dummy logic.
    # --- Dummy Plane Creation (Keep for now until refactored) ---
    points_cont_m = sampled_data['points_continuous'] / 1000.0
    points_disc_m = sampled_data['points_discrete'] / 1000.0
    points_cont_m -= np.mean(points_cont_m)
    points_disc_m -= np.mean(points_disc_m)
    if sampled_data['continuous_axis'] == 'z' and sampled_data['discrete_axis'] == 'y':
        Z, Y = np.meshgrid(points_cont_m, points_disc_m)
        X = np.zeros_like(Z)
        measurement_plane_coords = np.stack([X, Y, Z], axis=-1)
    # --- Add handling for yz plane from y200_zx.pickle ---
    elif sampled_data['continuous_axis'] == 'x' and sampled_data['discrete_axis'] == 'z':
         # Assuming x is continuous, z is discrete for y-constant plane
         points_x_m = sampled_data['points_continuous'] / 1000.0
         points_z_m = sampled_data['points_discrete'] / 1000.0
         points_x_m -= np.mean(points_x_m)
         points_z_m -= np.mean(points_z_m)
         X, Z = np.meshgrid(points_x_m, points_z_m)
         Y = np.zeros_like(X) # Assuming centered Y=0 before translation
         measurement_plane_coords = np.stack([X, Y, Z], axis=-1)
    else: # Add other axis combinations if needed
        logger.error(f"Unsupported axis combination: {sampled_data.get('continuous_axis', 'N/A')}, {sampled_data.get('discrete_axis', 'N/A')}")
        return None
    # --- End Dummy Plane Creation ---

    logger.info(f"  Applying translation: {translation}")
    translated_coords = measurement_plane_coords.reshape(-1, 3) + translation

    # Extract measured magnitude
    measured_magnitude = np.abs(sampled_data["results"]).flatten()
    # Handle NaNs if necessary (e.g., replace with min non-NaN value)
    if np.isnan(measured_magnitude).any():
        min_val = np.nanmin(measured_magnitude)
        measured_magnitude = np.nan_to_num(measured_magnitude, nan=min_val)
        logger.info(f"  Replaced NaN values in magnitude with {min_val:.4e}")


    processed_data = {
        'coordinates': translated_coords, # Shape (N_m, 3)
        'measured_magnitude': measured_magnitude, # Shape (N_m,)
        'original_data_shape': sampled_data["results"].shape, # Store shape (res_y, res_z) or similar
        'frequency': sampled_data.get('frequency'),
        # Add other relevant info if needed, e.g., original axes names
        'continuous_axis': sampled_data['continuous_axis'],
        'discrete_axis': sampled_data['discrete_axis'],
    }
    logger.info(f"  Real plane processing complete. Coordinates shape: {processed_data['coordinates'].shape}")
    return processed_data


def generate_simulated_plane(plane_cfg: DictConfig) -> dict | None:
    """
    Generates coordinates for a simulated measurement plane based on config.

    Args:
        plane_cfg: DictConfig for a single simulated plane.

    Returns:
        Dictionary containing processed plane data ('coordinates')
        or None if generation fails.
    """
    logger.info(f"  Type: Simulated Plane")
    plane_type = plane_cfg.get('plane_type', 'xy')
    center = np.array(plane_cfg.get('center', [0.0, 0.0, 0.0]))
    size = np.array(plane_cfg.get('size', [1.0, 1.0]))
    resolution = plane_cfg.get('resolution', 50)
    translation = np.array(plane_cfg.get('translation', [0.0, 0.0, 0.0]))

    if size.shape != (2,):
        logger.error(f"  'size' must be a list/array of 2 numbers (width, height). Got: {size}")
        return None

    logger.info(f"  Generating {plane_type} plane. Center: {center}, Size: {size}, Resolution: {resolution}")

    # Placeholder: Use actual geometry generation function
    # coords_before_translation = create_plane_grid(plane_type, center, size, resolution)
    # --- Dummy Plane Generation ---
    w, h = size
    x = np.linspace(-w / 2, w / 2, resolution)
    y = np.linspace(-h / 2, h / 2, resolution)
    X, Y = np.meshgrid(x, y)

    if plane_type == 'xy':
        coords_centered = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    elif plane_type == 'yz':
        coords_centered = np.stack([np.zeros_like(X), X, Y], axis=-1) # Map x->y, y->z
    elif plane_type == 'xz':
        coords_centered = np.stack([X, np.zeros_like(X), Y], axis=-1) # Map y->z
    else:
        logger.error(f"  Unsupported plane_type: {plane_type}. Use 'xy', 'yz', or 'xz'.")
        return None

    coords_before_translation = coords_centered.reshape(-1, 3) + center # Apply center offset
    # --- End Dummy Plane Generation ---

    logger.info(f"  Applying translation: {translation}")
    translated_coords = coords_before_translation + translation

    processed_data = {
        'coordinates': translated_coords, # Shape (N_m, 3)
        'original_data_shape': (resolution, resolution), # Store shape
        # Simulated planes don't have inherent measured magnitude;
        # it will be calculated later during training (if use_train) or testing.
    }
    logger.info(f"  Simulated plane processing complete. Coordinates shape: {processed_data['coordinates'].shape}")
    return processed_data

# TODO: Refactor create_measurement_plane from measured_data_reconstruction.py here or into utils
# TODO: Ensure src.io.load_measurement_data and src.io.sample_measurement_data are implemented/correct