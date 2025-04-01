import numpy as np
from scipy.spatial.distance import cdist  # Needed for scalar version


def create_channel_matrix(
    points: np.ndarray,
    measurement_plane: np.ndarray,
    k: float,
    use_vector_model: bool,
    # Vector model specific arguments (optional, only used if use_vector_model is True)
    tangents1: np.ndarray | None = None,
    tangents2: np.ndarray | None = None,
    measurement_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Create channel matrix H relating sources to measurement points.

    Switches between a scalar model and a vector model based on the flag.

    Args:
        points: Source points (clusters), shape (N_c, 3).
        measurement_plane: Measurement positions, shape (res, res, 3) or (N_m, 3).
        k: Wave number (2 * pi / lambda).
        use_vector_model: If True, use the vector model (requires tangents & direction).
                          If False, use the scalar model.
        tangents1: First tangent vectors (N_c, 3). Required if use_vector_model is True.
        tangents2: Second tangent vectors (N_c, 3). Required if use_vector_model is True.
        measurement_direction: Measurement direction vector (3,). Required if use_vector_model is True.

    Returns:
        Channel matrix H.
        Shape is (N_m, N_c) if use_vector_model is False.
        Shape is (N_m, 2 * N_c) if use_vector_model is True.
    """
    measurement_points = measurement_plane.reshape(-1, 3)
    N_m = measurement_points.shape[0]
    N_c = points.shape[0]

    if use_vector_model:
        # --- Vector Model Calculation ---
        if tangents1 is None or tangents2 is None or measurement_direction is None:
            raise ValueError(
                "tangents1, tangents2, and measurement_direction are required for vector model."
            )
        if tangents1.shape != (N_c, 3):
            raise ValueError(
                f"Shape mismatch for tangents1: expected ({N_c}, 3), got {tangents1.shape}"
            )
        if tangents2.shape != (N_c, 3):
            raise ValueError(
                f"Shape mismatch for tangents2: expected ({N_c}, 3), got {tangents2.shape}"
            )
        if measurement_direction.shape != (3,):
            raise ValueError("Measurement direction must be a 3-vector.")

        # Ensure measurement_direction is a unit vector
        norm_meas = np.linalg.norm(measurement_direction)
        if norm_meas < 1e-9:
            raise ValueError("Measurement direction cannot be a zero vector.")
        measurement_direction = measurement_direction / norm_meas
        # Assume tangents1 and tangents2 are already unit vectors from preprocessing

        t_c1 = tangents1
        t_c2 = tangents2

        # Calculate distances and direction vectors (vectorized)
        R_vec = measurement_points[:, None, :] - points[None, :, :]  # (N_m, N_c, 3)
        R_dist = np.linalg.norm(R_vec, axis=2)  # (N_m, N_c)
        R_dist = np.maximum(R_dist, 1e-10)  # Avoid division by zero
        R_hat = R_vec / R_dist[:, :, None]  # (N_m, N_c, 3)

        # Calculate scalar Green's function part
        G_scalar = np.exp(-1j * k * R_dist) / (4 * np.pi * R_dist)  # (N_m, N_c)

        # Calculate vectorized projection terms
        dot_R_t1 = np.einsum("ijk,jk->ij", R_hat, t_c1)  # (N_m, N_c)
        dot_R_t2 = np.einsum("ijk,jk->ij", R_hat, t_c2)  # (N_m, N_c)
        proj_t1 = t_c1[None, :, :] - R_hat * dot_R_t1[:, :, None]  # (N_m, N_c, 3)
        proj_t2 = t_c2[None, :, :] - R_hat * dot_R_t2[:, :, None]  # (N_m, N_c, 3)
        dot_meas_proj1 = np.einsum("k,ijk->ij", measurement_direction, proj_t1)  # (N_m, N_c)
        dot_meas_proj2 = np.einsum("k,ijk->ij", measurement_direction, proj_t2)  # (N_m, N_c)

        # Construct final matrix H
        H = np.zeros((N_m, 2 * N_c), dtype=np.complex128)
        H[:, 0::2] = G_scalar * dot_meas_proj1
        H[:, 1::2] = G_scalar * dot_meas_proj2

    else:
        # --- Scalar Model Calculation ---
        distances = cdist(measurement_points, points)  # (N_m, N_c)
        distances = np.maximum(distances, 1e-10)  # Avoid division by zero
        H = np.exp(-1j * k * distances) / (4 * np.pi * distances)  # (N_m, N_c)

    # Ensure Fortran-contiguous array for performance
    H = np.asfortranarray(H)
    return H
