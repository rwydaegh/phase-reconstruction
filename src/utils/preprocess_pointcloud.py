import argparse
import logging
import os
import pickle
import sys  # Added sys import for potential path manipulation in imports

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_tangent_vectors(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates two orthogonal tangent unit vectors (t1, t2) for each normal vector (n),
    forming a right-handed coordinate system (t1, t2, n). Aims for consistent
    orientation where t1 is generally horizontal relative to a global up vector.

    Args:
        normals: Array of normal vectors, shape (N, 3).

    Returns:
        Tuple containing t1 and t2, arrays of tangent vectors, each shape (N, 3).
    """
    num_normals = normals.shape[0]
    if num_normals == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # Ensure normals are unit vectors for accurate calculations
    norm_n = np.linalg.norm(normals, axis=1, keepdims=True)
    # Handle potential zero-norm normals to avoid NaN/errors
    valid_normals_mask = norm_n.flatten() > 1e-9
    if not np.all(valid_normals_mask):
        logger.warning(
            f"Found {np.sum(~valid_normals_mask)} zero-norm vectors in input normals. Tangents for these will be zero."
        )
        # Initialize tangents to zero, they will remain zero for invalid normals
        t1 = np.zeros_like(normals)
        t2 = np.zeros_like(normals)
        # Proceed only with valid normals
        normals_valid = normals[valid_normals_mask]
        norm_n_valid = norm_n[valid_normals_mask]
        if normals_valid.shape[0] == 0:  # All normals were invalid
            return t1, t2
    else:
        normals_valid = normals
        norm_n_valid = norm_n

    normals_unit = normals_valid / norm_n_valid

    # Define global axes
    global_up = np.array([0.0, 0.0, 1.0])
    global_alt_up = np.array([0.0, 1.0, 0.0])  # Use Y if normal is aligned with Z

    # Calculate first tangent (t1) - aiming for horizontal alignment
    # t1 = cross(up, normal)
    t1_valid = np.cross(global_up, normals_unit)
    norm_t1 = np.linalg.norm(t1_valid, axis=1)

    # Identify normals nearly parallel to the initial 'up' axis
    parallel_mask = norm_t1 < 1e-9

    # Use alternative 'up' axis for parallel cases
    if np.any(parallel_mask):
        t1_valid[parallel_mask] = np.cross(global_alt_up, normals_unit[parallel_mask])
        norm_t1[parallel_mask] = np.linalg.norm(t1_valid[parallel_mask], axis=1)

        # Handle cases where norm is still zero (normal is parallel to both up vectors)
        # This shouldn't happen with valid normals unless normal is zero, which is handled above.
        still_zero_mask = parallel_mask & (norm_t1 < 1e-9)
        if np.any(still_zero_mask):
            # This case implies the normal itself was likely zero or near-zero
            logger.warning(
                f"Tangent calculation resulted in zero vector for {np.sum(still_zero_mask)} normals (potentially zero normals). Assigning arbitrary tangents."
            )
            # Assign arbitrary orthogonal vectors if needed, though they should remain zero from init
            # For safety, explicitly set them, although this indicates an issue upstream
            t1_valid[still_zero_mask] = np.array([1.0, 0.0, 0.0])
            norm_t1[still_zero_mask] = 1.0

    # Normalize t1
    t1_valid /= np.maximum(norm_t1[:, np.newaxis], 1e-10)

    # Calculate second tangent vector (t2 = normal x t1) - ensures right-handed system
    t2_valid = np.cross(normals_unit, t1_valid)
    # t2 should be normalized if normals_unit and t1_valid are unit vectors and orthogonal.
    # Re-normalize for numerical robustness.
    norm_t2 = np.linalg.norm(t2_valid, axis=1, keepdims=True)
    t2_valid /= np.maximum(norm_t2, 1e-10)

    # Place calculated tangents back into the full arrays
    if not np.all(valid_normals_mask):
        t1[valid_normals_mask] = t1_valid
        t2[valid_normals_mask] = t2_valid
    else:
        t1 = t1_valid
        t2 = t2_valid

    return t1, t2


def preprocess_pointcloud(input_path: str, output_path: str):
    """
    Loads a point cloud from a .pkl file, calculates tangent vectors if missing,
    and saves the result (including tangents) to a new .pkl file.

    Expected input format: numpy array of shape (N, 7) with columns:
        x, y, z, distance, nx, ny, nz
    Output format: numpy array of shape (N, 13) with columns:
        x, y, z, distance, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z
    """
    logger.info(f"Loading point cloud data from: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load pickle file: {e}")
        return

    if not isinstance(data, np.ndarray):
        logger.error(f"Loaded data is not a numpy array (type: {type(data)}). Cannot preprocess.")
        return

    logger.info(f"Loaded data shape: {data.shape}")

    if data.ndim != 2:
        logger.error(f"Expected 2D array, but got {data.ndim}D array. Cannot preprocess.")
        return

    num_cols = data.shape[1]

    if num_cols == 13:
        logger.info(
            "Data already has 13 columns. Assuming tangents are precalculated. Saving as is."
        )
        output_data = data
    elif num_cols == 7:
        logger.info("Data has 7 columns. Calculating tangent vectors...")
        # Assuming columns 4, 5, 6 are nx, ny, nz
        normals = data[:, 4:7]
        t1, t2 = get_tangent_vectors(normals)
        logger.info(f"Calculated t1 shape: {t1.shape}, t2 shape: {t2.shape}")

        # Concatenate original data with tangent vectors
        output_data = np.hstack((data, t1, t2))
        logger.info(f"New data shape with tangents: {output_data.shape}")
    else:
        logger.error(f"Expected 7 or 13 columns, but got {num_cols}. Cannot preprocess.")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    logger.info(f"Saving preprocessed data to: {output_path}")
    try:
        with open(output_path, "wb") as f:
            pickle.dump(output_data, f)
        logger.info("Successfully saved preprocessed data.")
    except Exception as e:
        logger.error(f"Failed to save pickle file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess point cloud data by adding tangent vectors."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input .pkl file (expected N x 7 array)."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to save the output .pkl file (will be N x 13 array)."
    )
    args = parser.parse_args()

    preprocess_pointcloud(args.input_file, args.output_file)
