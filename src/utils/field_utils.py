import numpy as np
from scipy.spatial.distance import cdist
from functools import lru_cache

def create_channel_matrix(
    points: np.ndarray,
    measurement_plane: np.ndarray,
    k: float
) -> np.ndarray:
    """Create channel matrix H relating source points to measurement points for scalar fields.
    
    Args:
        points: Source points (clusters), shape (num_points, 3)
        measurement_plane: Measurement positions, shape (resolution, resolution, 3)
        k: Wave number
        
    Returns:
        Channel matrix H with shape (num_measurements, num_points)
    """
    # Flatten measurement plane points for matrix operations
    measurement_points = measurement_plane.reshape(-1, 3)
    
    # Use scipy's cdist for fast distance calculations
    distances = cdist(measurement_points, points)
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-10)
    
    # Calculate exp(-jkR)/(4πR) directly
    H = np.exp(-1j * k * distances) / (4 * np.pi * distances)
    
    # Use fortran array layout for more efficient matrix multiplication
    H = np.asfortranarray(H)
    
    return H

@lru_cache(maxsize=8)
def _create_channel_matrix_cached(points_tuple, measurement_plane_shape, measurement_plane_bytes, k):
    """Cached version of channel matrix creation for repeated configurations."""
    # Convert back to numpy arrays
    points = np.array(points_tuple)
    measurement_plane = np.frombuffer(measurement_plane_bytes).reshape(measurement_plane_shape)
    
    return create_channel_matrix(points, measurement_plane, k)

def compute_fields(
    points: np.ndarray,
    currents: np.ndarray,
    measurement_plane: np.ndarray,
    k: float,
    channel_matrix: np.ndarray = None
) -> np.ndarray:
    """Compute fields from currents using the channel matrix.
    
    Args:
        points: Source points, shape (num_points, 3)
        currents: Current amplitudes at source points, shape (num_points,)
        measurement_plane: Measurement positions, shape (resolution, resolution, 3)
        k: Wave number
        channel_matrix: Optional pre-computed channel matrix. If None, it will be computed.
        
    Returns:
        Field values at measurement points
    """
    # Use provided channel matrix or compute it with optional caching
    if channel_matrix is None:
        try:
            # Try to use cached version for repeated configurations
            points_tuple = tuple(map(tuple, points))
            measurement_plane_shape = measurement_plane.shape
            measurement_plane_bytes = measurement_plane.tobytes()
            H = _create_channel_matrix_cached(points_tuple, measurement_plane_shape, 
                                              measurement_plane_bytes, k)
        except Exception:
            # Fall back to direct computation if caching fails
            H = create_channel_matrix(points, measurement_plane, k)
    else:
        H = channel_matrix
    
    # Calculate fields using matrix multiplication
    # Ensure fortran array layout for more efficient matrix multiplication
    currents_fortran = np.asfortranarray(currents)
    return H @ currents_fortran

def reconstruct_field(
    channel_matrix: np.ndarray,
    cluster_coefficients: np.ndarray
) -> np.ndarray:
    """Reconstruct field from cluster coefficients.
    
    Args:
        channel_matrix: Matrix H relating clusters to measurement points
        cluster_coefficients: Coefficients of clusters
    
    Returns:
        Reconstructed complex field
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
