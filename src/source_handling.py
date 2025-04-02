# src/source_handling.py
import logging
import os
import pickle
import random

import numpy as np
from omegaconf import DictConfig

# Import necessary functions from other modules
from src.create_test_pointcloud import create_test_pointcloud
from src.utils.geometry_utils import get_cube_normals
from src.utils.preprocess_pointcloud import get_tangent_vectors

logger = logging.getLogger(__name__)

def generate_original_currents(points_true: np.ndarray, cfg: DictConfig, use_vector_model: bool) -> np.ndarray:
    """
    Generates the ground truth current distribution based on configuration.

    Args:
        points_true: The ground truth source point coordinates (N_c, 3).
        cfg: The source_pointcloud section of the main configuration.
             Requires 'num_sources' and 'amplitude_sigma'.
        use_vector_model: Boolean indicating if the vector model is used globally.

    Returns:
        np.ndarray: The generated original currents array (shape N_c or 2*N_c).
    """
    N_c = points_true.shape[0]
    if N_c == 0:
        logger.warning("Cannot generate currents for zero source points.")
        return np.array([], dtype=complex)

    # Determine number of coefficients based on model
    num_coeffs_total = 2 * N_c if use_vector_model else N_c
    currents = np.zeros(num_coeffs_total, dtype=complex)

    num_sources_to_activate = min(cfg.get('num_sources', 0), N_c)
    amplitude_sigma = cfg.get('amplitude_sigma', 1.0)

    if num_sources_to_activate <= 0:
        logger.warning("num_sources <= 0 in config. Generating zero currents.")
        return currents # Return zero currents

    # Ensure random state is consistent if called multiple times within same run
    # np.random.seed(...) should be set globally once at the start

    source_indices = random.sample(range(N_c), num_sources_to_activate) # Indices of points (0 to N_c-1)

    amplitudes = np.random.lognormal(mean=0, sigma=amplitude_sigma, size=num_sources_to_activate)
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources_to_activate)

    logger.info(f"Assigning random currents to {num_sources_to_activate} source points...")
    for i, point_idx in enumerate(source_indices):
        value = amplitudes[i] * np.exp(1j * phases[i])
        if use_vector_model:
            # Assign to the first component (index 2*point_idx) for vector model
            current_idx = 2 * point_idx
            currents[current_idx] = value
            # currents[current_idx + 1] = 0.0 # Keep second component zero (already initialized)
        else:
            # Assign directly to the point index for scalar model
            currents[point_idx] = value

    logger.info(f"Generated original currents shape: {currents.shape}")
    return currents


def get_source_pointcloud(cfg: DictConfig, global_cfg: DictConfig, base_dir: str = ".") -> tuple:
    """
    Loads or generates the source point cloud, tangents, and original currents.
    Applies optional perturbation to geometry.

    Args:
        cfg: The source_pointcloud section of the main configuration (DictConfig).
        global_cfg: The global_params section (needed for use_vector_model).
        base_dir: Base directory to resolve relative data paths.

    Returns:
        tuple: (points_true, tangents1_true, tangents2_true,
                points_perturbed, tangents1_perturbed, tangents2_perturbed,
                original_currents)
    """
    points_true = None
    tangents1_true = None
    tangents2_true = None

    if cfg.get('use_source_file', False):
        logger.info("Loading source point cloud from file...")
        file_path_rel = cfg.get('source_file_path')
        if not file_path_rel:
            raise ValueError("'source_file_path' must be provided when 'use_source_file' is true.")

        file_path_abs = os.path.join(base_dir, file_path_rel)
        if not os.path.exists(file_path_abs):
            raise FileNotFoundError(f"Source point cloud file not found: {file_path_abs}")

        logger.info(f"  Loading from: {file_path_abs}")
        with open(file_path_abs, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, np.ndarray):
             raise TypeError(f"Expected numpy array in source file, got {type(data)}")

        logger.info(f"  Loaded data shape: {data.shape}")

        # --- Data Format Handling (assuming preprocessed format or calculating tangents) ---
        if data.shape[1] == 13: # x,y,z, dist, nx,ny,nz, t1x,t1y,t1z, t2x,t2y,t2z
            logger.info("  Detected 13 columns (preprocessed format with tangents).")
            points_loaded = data[:, :3]
            # normals_loaded = data[:, 4:7] # Optional
            tangents1_loaded = data[:, 7:10]
            tangents2_loaded = data[:, 10:13]
        elif data.shape[1] == 7: # x,y,z, dist, nx,ny,nz
            logger.warning("  Detected 7 columns. Calculating tangents...")
            points_loaded = data[:, :3]
            normals_loaded = data[:, 4:7]
            tangents1_loaded, tangents2_loaded = get_tangent_vectors(normals_loaded)
            logger.info(f"  Calculated tangents. t1: {tangents1_loaded.shape}, t2: {tangents2_loaded.shape}")
            # Optional: Overwrite file with 13 columns? For now, just use calculated tangents.
        elif data.shape[1] >= 3: # Assume at least x,y,z
             logger.warning(f"  Detected {data.shape[1]} columns. Using first 3 as points. Tangents will be None.")
             points_loaded = data[:, :3]
             tangents1_loaded = None
             tangents2_loaded = None
        else:
            raise ValueError(f"Loaded point cloud data has unexpected shape: {data.shape}. Need at least 3 columns.")

        points_true = points_loaded
        tangents1_true = tangents1_loaded
        tangents2_true = tangents2_loaded

    else:
        logger.info("Generating test point cloud (cube)...")
        # Need to pass relevant sub-config to create_test_pointcloud if it expects DictConfig
        # Or extract parameters manually. Assuming manual extraction for now.
        gen_cfg_dict = {k: cfg.get(k) for k in ['wall_points', 'room_size']}
        gen_cfg = DictConfig(gen_cfg_dict) # Create a temporary DictConfig if needed by the function
        points_generated = create_test_pointcloud(gen_cfg)

        logger.info("  Calculating normals and tangents for generated cube...")
        # Assuming get_cube_normals takes points and room_size
        points_true, normals_true = get_cube_normals(points_generated, cfg.room_size)
        tangents1_true, tangents2_true = get_tangent_vectors(normals_true)
        logger.info(f"  Generated points: {points_true.shape}, Tangents: {tangents1_true.shape}")


    # --- Apply Downsampling ---
    downsample_factor = cfg.get('pointcloud_downsample', 1)
    if downsample_factor > 1 and points_true is not None:
        original_count = len(points_true)
        points_true = points_true[::downsample_factor]
        if tangents1_true is not None:
            tangents1_true = tangents1_true[::downsample_factor]
        if tangents2_true is not None:
            tangents2_true = tangents2_true[::downsample_factor]
        logger.info(f"  Downsampled point cloud from {original_count} to {len(points_true)} (factor: {downsample_factor})")

    # --- Apply Max Distance Filter ---
    max_distance = cfg.get('max_distance_from_origin', -1)
    if max_distance > 0 and points_true is not None:
        logger.info(f"  Applying max distance filter: {max_distance} m from origin")
        distances = np.sqrt(np.sum(points_true**2, axis=1))
        mask = distances <= max_distance
        original_count = len(points_true)
        points_true = points_true[mask]
        if tangents1_true is not None:
            tangents1_true = tangents1_true[mask]
        if tangents2_true is not None:
            tangents2_true = tangents2_true[mask]
        filtered_count = original_count - len(points_true)
        logger.info(f"  Filtered out {filtered_count} points. Retained {len(points_true)} points.")

    if points_true is None or len(points_true) == 0:
        raise ValueError("Source point cloud processing resulted in zero points.")

    # --- Apply Perturbation (to create points_perturbed) ---
    points_perturbed = points_true.copy()
    tangents1_perturbed = tangents1_true.copy() if tangents1_true is not None else None
    tangents2_perturbed = tangents2_true.copy() if tangents2_true is not None else None

    if cfg.get('perturb_points', False):
        perturbation_factor = cfg.get('perturbation_factor', 0.01)
        logger.info(f"  Perturbing points for reconstruction geometry (factor: {perturbation_factor})...")
        if perturbation_factor > 0:
            # Calculate distance from origin for each true point
            distances = np.sqrt(np.sum(points_true**2, axis=1))
            # Generate uniform random perturbations [-1, 1] for each coordinate
            random_perturbations = np.random.uniform(-1, 1, size=points_true.shape)
            # Scale perturbations by the factor and the distance
            scaled_perturbations = random_perturbations * perturbation_factor * distances[:, np.newaxis]
            # Apply perturbation
            points_perturbed = points_true + scaled_perturbations
            # For simplicity, assume tangents don't change significantly with small perturbations
            # If tangent recalculation is needed, it would go here.
            logger.info(f"  Perturbed point cloud shape: {points_perturbed.shape}")
        else:
             logger.info("  Perturbation factor is 0, points_perturbed is same as points_true.")
    else:
        logger.info("  Point perturbation disabled. Using true geometry for reconstruction.")

    # --- Generate Original Currents (using finalized true points) ---
    use_vector_model_flag = global_cfg.get('use_vector_model', False)
    original_currents = generate_original_currents(points_true, cfg, use_vector_model_flag)

    logger.info(f"Source handling complete. True points: {points_true.shape}, Perturbed points: {points_perturbed.shape}, Currents: {original_currents.shape}")

    return (points_true, tangents1_true, tangents2_true,
            points_perturbed, tangents1_perturbed, tangents2_perturbed,
            original_currents)

# TODO: Add unit tests for this module