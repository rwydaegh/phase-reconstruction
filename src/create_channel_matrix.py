import numpy as np
from scipy.spatial.distance import cdist

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
