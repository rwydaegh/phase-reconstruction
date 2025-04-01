import numpy as np
from functools import lru_cache
# from scipy.spatial.distance import cdist # No longer needed here

# Import the updated vectorized function
from src.create_channel_matrix import create_channel_matrix
# Removed the old scalar create_channel_matrix definition from this file.
# The vectorized version is now imported from src.create_channel_matrix


@lru_cache(maxsize=8)
def _create_channel_matrix_cached(
    points_tuple: tuple,
    tangents1_tuple: tuple,
    tangents2_tuple: tuple,
    measurement_plane_shape: tuple,
    measurement_plane_bytes: bytes,
    measurement_direction_tuple: tuple,
    k: float,
):
    """Cached version of the vectorized channel matrix creation."""
    # Convert hashable types back to numpy arrays
    points = np.array(points_tuple)
    tangents1 = np.array(tangents1_tuple)
    tangents2 = np.array(tangents2_tuple)
    measurement_plane = np.frombuffer(measurement_plane_bytes, dtype=np.float64).reshape(
        measurement_plane_shape
    )
    measurement_direction = np.array(measurement_direction_tuple)

    # Call the imported vectorized function
    return create_channel_matrix(
        points, tangents1, tangents2, measurement_plane, measurement_direction, k
    )


def compute_fields(
    points: np.ndarray,
    tangents1: np.ndarray,
    tangents2: np.ndarray,
    currents: np.ndarray, # Should now be shape (2 * num_points,)
    measurement_plane: np.ndarray,
    measurement_direction: np.ndarray,
    k: float,
    channel_matrix: np.ndarray = None, # Will be shape (N_m, 2 * num_points)
) -> np.ndarray:
    """Compute measured E-field component from source current coefficients.

    Uses the vectorized channel matrix H.

    Args:
        points: Source points (clusters), shape (N_c, 3).
        tangents1: First set of unit tangent vectors at each source point, shape (N_c, 3).
        tangents2: Second set of unit tangent vectors (orthogonal to tangents1 and normal),
                   shape (N_c, 3).
        currents: Complex current coefficients, shape (2 * N_c,).
                  Order: [x_1_t1, x_1_t2, x_2_t1, x_2_t2, ...].
        measurement_plane: Measurement positions, shape (res, res, 3) or (N_m, 3).
        measurement_direction: Unit vector of the measured E-field component, shape (3,).
        k: Wave number (2 * pi / lambda).
        channel_matrix: Optional pre-computed channel matrix, shape (N_m, 2 * N_c).
                        If None, it will be computed.

    Returns:
        Complex field values (measured component) at measurement points, shape (N_m,).
    """
    # Use provided channel matrix or compute it with optional caching
    if channel_matrix is None:
        try:
            # Prepare hashable arguments for caching
            points_tuple = tuple(map(tuple, points))
            # Convert tangents to tuples for hashing
            tangents1_tuple = tuple(map(tuple, tangents1))
            tangents2_tuple = tuple(map(tuple, tangents2))
            measurement_plane_shape = measurement_plane.shape
            # Ensure consistent dtype for bytes conversion
            measurement_plane_bytes = measurement_plane.astype(np.float64, copy=False).tobytes()
            # Convert measurement_direction to tuple for hashing
            measurement_direction_tuple = tuple(measurement_direction)

            H = _create_channel_matrix_cached(
                points_tuple,
                tangents1_tuple,
                tangents2_tuple,
                measurement_plane_shape,
                measurement_plane_bytes,
                measurement_direction_tuple,
                k,
            )
        except Exception as e:
            print(f"Cache lookup/creation failed: {e}. Falling back to direct computation.")
            # Fall back to direct computation if caching fails
            H = create_channel_matrix(
                points, tangents1, tangents2, measurement_plane, measurement_direction, k
            )
    else:
        H = channel_matrix

    # Calculate fields using matrix multiplication
    # Ensure currents vector has the correct shape (2 * N_c)
    if currents.shape != (2 * points.shape[0],):
         raise ValueError(f"Currents shape {currents.shape} is incompatible with H shape {H.shape}. Expected ({2 * points.shape[0]},)")

    # Ensure fortran array layout for efficiency
    currents_fortran = np.asfortranarray(currents)
    # H is already Fortran contiguous from create_channel_matrix

    # Calculate fields: y = H * x
    return H @ currents_fortran


def reconstruct_field(channel_matrix: np.ndarray, cluster_coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct field from cluster coefficients.

    Args:
        channel_matrix: Matrix H relating clusters to measurement points, shape (N_m, 2*N_c).
        cluster_coefficients: Coefficients of clusters, shape (2*N_c,).
                              Order: [x_1_t1, x_1_t2, x_2_t1, x_2_t2, ...].

    Returns:
        Reconstructed complex field component at measurement points, shape (N_m,).
    """
    # Ensure fortran array layout for more efficient matrix multiplication
    coeff_fortran = np.asfortranarray(cluster_coefficients)
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
