import numpy as np
# from scipy.spatial.distance import cdist # No longer needed


# Removed _get_tangent_vectors function. Tangents are now precalculated.


def create_channel_matrix(
    points: np.ndarray,
    tangents1: np.ndarray,
    tangents2: np.ndarray,
    measurement_plane: np.ndarray,
    measurement_direction: np.ndarray,
    k: float,
) -> np.ndarray:
    """Create channel matrix H relating source currents to measured E-field component.

    Uses the far-field approximation of the dyadic Green's function and
    pre-calculated tangent vectors for the source points.
    Each source point corresponds to two columns in H, representing the two
    orthogonal tangent basis vectors.

    Args:
        points: Source points (clusters), shape (N_c, 3).
        tangents1: First set of unit tangent vectors at each source point, shape (N_c, 3).
        tangents2: Second set of unit tangent vectors (orthogonal to tangents1 and normal),
                   shape (N_c, 3).
        measurement_plane: Measurement positions, shape (res, res, 3) or (N_m, 3).
        measurement_direction: Unit vector of the measured E-field component, shape (3,).
        k: Wave number (2 * pi / lambda).

    Returns:
        Channel matrix H with shape (N_m, 2 * N_c).
    """
    measurement_points = measurement_plane.reshape(-1, 3)
    N_m = measurement_points.shape[0]
    N_c = points.shape[0]

    if tangents1.shape != (N_c, 3):
        raise ValueError(f"Shape mismatch for tangents1: expected ({N_c}, 3), got {tangents1.shape}")
    if tangents2.shape != (N_c, 3):
        raise ValueError(f"Shape mismatch for tangents2: expected ({N_c}, 3), got {tangents2.shape}")
    if measurement_direction.shape != (3,):
        raise ValueError("Measurement direction must be a 3-vector.")

    # Ensure measurement_direction is a unit vector
    norm_meas = np.linalg.norm(measurement_direction)
    if norm_meas < 1e-9:
        raise ValueError("Measurement direction cannot be a zero vector.")
    measurement_direction = measurement_direction / norm_meas
    # Assume tangents1 and tangents2 are already unit vectors from preprocessing


    # 1. Tangent vectors are now provided as input
    t_c1 = tangents1
    t_c2 = tangents2

    # 2. Calculate distances and direction vectors (vectorized)
    # R_vec shape: (N_m, N_c, 3)
    R_vec = measurement_points[:, None, :] - points[None, :, :]
    # R_dist shape: (N_m, N_c)
    R_dist = np.linalg.norm(R_vec, axis=2)
    # Add epsilon for numerical stability (avoid division by zero)
    R_dist = np.maximum(R_dist, 1e-10)
    # R_hat shape: (N_m, N_c, 3)
    R_hat = R_vec / R_dist[:, :, None]

    # 3. Calculate scalar Green's function part
    # G_scalar shape: (N_m, N_c)
    G_scalar = np.exp(-1j * k * R_dist) / (4 * np.pi * R_dist)

    # 4. Calculate vectorized projection terms
    # Dot products of R_hat with tangent vectors
    # dot_R_t1/2 shape: (N_m, N_c)
    dot_R_t1 = np.einsum('ijk,jk->ij', R_hat, t_c1)
    dot_R_t2 = np.einsum('ijk,jk->ij', R_hat, t_c2)

    # Calculate (I - R_hat R_hat) . t_c (projected tangents)
    # proj_t1/2 shape: (N_m, N_c, 3)
    proj_t1 = t_c1[None, :, :] - R_hat * dot_R_t1[:, :, None]
    proj_t2 = t_c2[None, :, :] - R_hat * dot_R_t2[:, :, None]

    # Dot product with measurement direction
    # dot_meas_proj1/2 shape: (N_m, N_c)
    dot_meas_proj1 = np.einsum('k,ijk->ij', measurement_direction, proj_t1)
    dot_meas_proj2 = np.einsum('k,ijk->ij', measurement_direction, proj_t2)

    # 5. Construct final matrix H
    H = np.zeros((N_m, 2 * N_c), dtype=np.complex128)

    # Assign columns using slicing and multiply by scalar Green's part
    H[:, 0::2] = G_scalar * dot_meas_proj1
    H[:, 1::2] = G_scalar * dot_meas_proj2

    # 6. Ensure Fortran-contiguous array for performance in subsequent linear algebra
    H = np.asfortranarray(H)

    return H
