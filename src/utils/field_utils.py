from functools import lru_cache

import numpy as np

# from scipy.spatial.distance import cdist # No longer needed here
# Import the updated vectorized function
from src.create_channel_matrix import create_channel_matrix

# Removed the old scalar create_channel_matrix definition from this file.
# The vectorized version is now imported from src.create_channel_matrix


@lru_cache(maxsize=8)
def _create_channel_matrix_cached(
    # Arguments common to both models
    points_tuple: tuple,
    measurement_plane_shape: tuple,
    measurement_plane_bytes: bytes,
    k: float,
    use_vector_model: bool,
    # Vector model specific arguments (hashable) - use None if scalar
    tangents1_tuple: tuple | None,
    tangents2_tuple: tuple | None,
    measurement_direction_tuple: tuple | None,
):
    """Cached version of the vectorized channel matrix creation."""
    # Convert hashable types back to numpy arrays
    points = np.array(points_tuple)
    measurement_plane = np.frombuffer(measurement_plane_bytes, dtype=np.float64).reshape(
        measurement_plane_shape
    )
    # Convert vector-specific args only if needed
    tangents1 = np.array(tangents1_tuple) if tangents1_tuple is not None else None
    tangents2 = np.array(tangents2_tuple) if tangents2_tuple is not None else None
    measurement_direction = (
        np.array(measurement_direction_tuple) if measurement_direction_tuple is not None else None
    )

    # Call the imported vectorized function
    # Call the unified function with the flag and conditional arguments
    return create_channel_matrix(
        points=points,
        measurement_plane=measurement_plane,
        k=k,
        use_vector_model=use_vector_model,
        tangents1=tangents1,
        tangents2=tangents2,
        measurement_direction=measurement_direction,
    )


def compute_fields(
    # Common arguments
    points: np.ndarray,
    currents: np.ndarray,  # Shape depends on model: (N_c,) for scalar, (2*N_c,) for vector
    measurement_plane: np.ndarray,
    k: float,
    use_vector_model: bool,
    # Vector model specific arguments
    tangents1: np.ndarray | None = None,
    tangents2: np.ndarray | None = None,
    measurement_direction: np.ndarray | None = None,
    # Optional pre-computed matrix
    channel_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Compute measured field component from source currents/coefficients.

    Uses either the scalar or vector channel matrix H based on the flag.

    Args:
        points: Source points (clusters), shape (N_c, 3).
        currents: Complex currents/coefficients. Shape (N_c,) if scalar, (2*N_c,) if vector.
        measurement_plane: Measurement positions, shape (res, res, 3) or (N_m, 3).
        k: Wave number (2 * pi / lambda).
        use_vector_model: Flag to select model.
        tangents1: Required if use_vector_model is True. Shape (N_c, 3).
        tangents2: Required if use_vector_model is True. Shape (N_c, 3).
        measurement_direction: Required if use_vector_model is True. Shape (3,).
        channel_matrix: Optional pre-computed channel matrix. Shape depends on model.
                        If None, it will be computed.

    Returns:
        Complex field values at measurement points, shape (N_m,).
    """
    # Use provided channel matrix or compute it with optional caching
    if channel_matrix is None:
        try:
            # Prepare hashable arguments for caching
            # Prepare hashable arguments for caching
            points_tuple = tuple(map(tuple, points))
            measurement_plane_shape = measurement_plane.shape
            measurement_plane_bytes = measurement_plane.astype(np.float64, copy=False).tobytes()

            # Prepare vector-specific args for cache only if using vector model
            tangents1_tuple = (
                tuple(map(tuple, tangents1)) if use_vector_model and tangents1 is not None else None
            )
            tangents2_tuple = (
                tuple(map(tuple, tangents2)) if use_vector_model and tangents2 is not None else None
            )
            measurement_direction_tuple = (
                tuple(measurement_direction)
                if use_vector_model and measurement_direction is not None
                else None
            )

            H = _create_channel_matrix_cached(
                points_tuple=points_tuple,
                measurement_plane_shape=measurement_plane_shape,
                measurement_plane_bytes=measurement_plane_bytes,
                k=k,
                use_vector_model=use_vector_model,
                tangents1_tuple=tangents1_tuple,
                tangents2_tuple=tangents2_tuple,
                measurement_direction_tuple=measurement_direction_tuple,
            )
        except Exception as e:
            print(f"Cache lookup/creation failed: {e}. Falling back to direct computation.")
            # Fall back to direct computation
            H = create_channel_matrix(
                points=points,
                measurement_plane=measurement_plane,
                k=k,
                use_vector_model=use_vector_model,
                tangents1=tangents1,
                tangents2=tangents2,
                measurement_direction=measurement_direction,
            )
    else:
        H = channel_matrix

    # Calculate fields using matrix multiplication
    # Ensure currents vector has the correct shape based on the model
    expected_current_len = 2 * points.shape[0] if use_vector_model else points.shape[0]
    if currents.shape[0] != expected_current_len:
        raise ValueError(
            f"Currents shape {currents.shape} is incompatible with model (vector={use_vector_model}). Expected length {expected_current_len}."
        )
    if H.shape[1] != expected_current_len:
        raise ValueError(
            f"Channel matrix columns {H.shape[1]} incompatible with model (vector={use_vector_model}). Expected {expected_current_len}."
        )

    # Ensure fortran array layout for efficiency
    currents_fortran = np.asfortranarray(currents)
    # H is already Fortran contiguous from create_channel_matrix

    # Calculate fields: y = H * x
    return H @ currents_fortran


def reconstruct_field(channel_matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct field from source coefficients using the channel matrix.

    Args:
        channel_matrix: Matrix H relating sources to measurement points.
                        Shape (N_m, N_c) for scalar model, (N_m, 2*N_c) for vector model.
        coefficients: Coefficients of sources.
                      Shape (N_c,) for scalar model, (2*N_c,) for vector model.

    Returns:
        Reconstructed complex field at measurement points, shape (N_m,).
    """
    # Basic check for compatibility
    if channel_matrix.shape[1] != coefficients.shape[0]:
        raise ValueError(
            f"Shape mismatch: H columns {channel_matrix.shape[1]} != coefficients length {coefficients.shape[0]}"
        )

    # Ensure fortran array layout for more efficient matrix multiplication
    coeff_fortran = np.asfortranarray(coefficients)
    return channel_matrix @ coeff_fortran


def normalized_rmse(a, b):
    """Optimized normalized RMSE calculation.

    Args:
        a: First array
        b: Second array

    Returns:
        Normalized RMSE
    """
    diff = a - b
    return np.sqrt(np.mean(diff**2)) / (np.max(a) - np.min(a))


def normalized_correlation(a, b):
    """Optimized normalized correlation calculation.

    Args:
        a: First array
        b: Second array

    Returns:
        Normalized correlation
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    a_norm = (a - np.mean(a)) / np.std(a)
    b_norm = (b - np.mean(b)) / np.std(b)

    return np.correlate(a_norm, b_norm)[0] / len(a_norm)
