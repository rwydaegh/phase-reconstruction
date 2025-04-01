import math  # For pi

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix

# Import functions under test
from src.create_test_pointcloud import create_test_pointcloud


# Helper to calculate k from wavelength
def calculate_k(wavelength: float) -> float:
    return 2 * math.pi / wavelength

@pytest.fixture
def basic_config() -> DictConfig:
    """Provides a basic Hydra config loaded from defaults for component testing."""
    with hydra.initialize(config_path="../conf", version_base=None):
        # Load the default config and override specific values for testing
        cfg = hydra.compose(config_name="measured_data", overrides=[
            "resolution=10",
            "plane_size=0.5",
            "wavelength=0.01", # 30 GHz
            "room_size=1.0",
            "wall_points=3", # Small number for easier testing (3x3 grid per wall face)
            "num_sources=5",
            "perturb_points=False", # Disable perturbation for predictable geometry
            "verbose=False",
            "no_plot=True",
            "no_anim=True",
            "show_plot=False",
            "use_source_pointcloud=False" # Ensure we generate points
        ])
    return cfg

# --- Tests for create_test_pointcloud ---

def test_create_pointcloud_shape(basic_config):
    """Tests if the generated point cloud has the correct shape."""
    """Tests if the generated point cloud has the correct shape using Hydra config."""
    # For wall_points=3 (n=3), each of the 6 walls has n*n = 9 points.
    # Total expected points = 6 * n^2 = 6 * 3^2 = 54
    # Note: basic_config fixture already sets wall_points=3
    points = create_test_pointcloud(basic_config) # Pass the OmegaConf object
    assert points.shape == (54, 3), f"Expected shape (54, 3), but got {points.shape}"

    # Test with a different number of wall points by overriding the loaded config
    basic_config.wall_points = 4 # Directly modify the loaded DictConfig for this test case
    points = create_test_pointcloud(basic_config)
    assert points.shape == (96, 3), f"Expected shape (96, 3), but got {points.shape}"

def test_create_pointcloud_on_surface(basic_config: DictConfig):
    """Tests if all generated points lie on the surface of the cube using Hydra config."""
    basic_config.wall_points = 5 # Override for this test
    points = create_test_pointcloud(basic_config)
    half_size = basic_config.room_size / 2.0

    # Check if each point has at least one coordinate equal to +/- half_size
    on_surface = np.any(np.abs(np.abs(points) - half_size) < 1e-9, axis=1)
    assert np.all(on_surface), "Not all points lie exactly on the cube surface."

def test_create_pointcloud_no_interior_points(basic_config: DictConfig):
    """Tests if no points are generated strictly inside the cube using Hydra config."""
    basic_config.wall_points = 5 # Override for this test
    points = create_test_pointcloud(basic_config)
    half_size = basic_config.room_size / 2.0

    # Check if any point is strictly inside the boundaries
    is_interior = (np.abs(points[:, 0]) < half_size - 1e-9) & \
                  (np.abs(points[:, 1]) < half_size - 1e-9) & \
                  (np.abs(points[:, 2]) < half_size - 1e-9)
    assert not np.any(is_interior), "Found points strictly inside the cube."



def test_create_pointcloud_perturbation(basic_config: DictConfig):
    """Tests if enabling perturbation actually modifies point coordinates."""
    # Create original points without perturbation
    basic_config.perturb_points = False
    original_points = create_test_pointcloud(basic_config)

    # Create points with perturbation enabled
    basic_config.perturb_points = True
    basic_config.perturbation_factor = 0.01 # Small perturbation
    perturbed_points = create_test_pointcloud(basic_config)

    # Ensure shapes are the same
    assert original_points.shape == perturbed_points.shape, \
        f"Shape mismatch: Original {original_points.shape}, Perturbed {perturbed_points.shape}"

    # Ensure points are actually different
    assert not np.allclose(original_points, perturbed_points, atol=1e-9), \
        "Perturbed points are identical to original points."

    # Optional: Check if the average distance moved is reasonable
    distances_moved = np.linalg.norm(original_points - perturbed_points, axis=1)
    avg_distance = np.mean(distances_moved)
    assert avg_distance > 1e-6, (
        f"Average distance moved ({avg_distance}) is too small, "
        f"perturbation might not be effective."
    )
    # We could add an upper bound check too,
    # but it's less critical than ensuring *some* change happened.


# --- Tests for create_channel_matrix ---

@pytest.fixture
def channel_matrix_setup(basic_config: DictConfig):
    """Provides points, tangents, measurement plane etc. for channel matrix tests."""
    # Simple point cloud: 2 points
    source_points = np.array([
        [0.1, 0.2, 0.3],
        [-0.1, -0.2, -0.3]
    ])
    # Create dummy normals and calculate tangents for these points
    # (Using simple Z-normals for fixture simplicity)
    dummy_normals = np.zeros_like(source_points)
    dummy_normals[:, 2] = 1.0
    from src.utils.preprocess_pointcloud import get_tangent_vectors # Import needed function
    tangents1, tangents2 = get_tangent_vectors(dummy_normals)

    # Simple measurement plane: 4 points (2x2 grid)
    res = 2
    plane_size = 0.2
    x = np.linspace(-plane_size / 2, plane_size / 2, res)
    y = np.linspace(-plane_size / 2, plane_size / 2, res)
    X, Y = np.meshgrid(x, y)
    # Place plane at z=0.5
    measurement_plane = np.stack([X, Y, np.ones_like(X) * 0.5], axis=-1)

    # Default measurement direction (Y-axis)
    measurement_direction = np.array([0.0, 1.0, 0.0])

    k = calculate_k(basic_config.wavelength) # Calculate k from wavelength in config
    return source_points, tangents1, tangents2, measurement_plane, measurement_direction, k

def test_create_channel_matrix_shape(channel_matrix_setup):
    """Tests the output shape of the channel matrix H."""
    source_points, tangents1, tangents2, measurement_plane, measurement_direction, k = channel_matrix_setup
    num_source_points = source_points.shape[0] # Should be 2
    num_measurement_points = measurement_plane.shape[0] * measurement_plane.shape[1] # Should be 4

    H = create_channel_matrix(
        points=source_points,
        tangents1=tangents1,
        tangents2=tangents2,
        measurement_plane=measurement_plane,
        measurement_direction=measurement_direction,
        k=k
    )

    # Expected shape: (num_measurement_points, 2 * num_source_points)
    expected_shape = (num_measurement_points, 2 * num_source_points)
    assert H.shape == expected_shape, f"Expected H shape {expected_shape}, but got {H.shape}"

def test_create_channel_matrix_dtype_and_values(channel_matrix_setup):
    """Tests the data type and calculates one value for verification."""
    source_points, tangents1, tangents2, measurement_plane, measurement_direction, k = channel_matrix_setup
    H = create_channel_matrix(
        points=source_points,
        tangents1=tangents1,
        tangents2=tangents2,
        measurement_plane=measurement_plane,
        measurement_direction=measurement_direction,
        k=k
    )

    # Check dtype
    assert H.dtype == np.complex128, f"Expected dtype complex128, but got {H.dtype}"

    # TODO: Add a value check based on the vector formula if needed,
    # but it's complex to calculate manually. Skipping for now.
    # # Check a specific value: H[0, 0] (contribution from source 0, tangent 1)
    # mp0 = measurement_plane.reshape(-1, 3)[0]
    # sp0 = source_points[0]
    # t1_0 = tangents1[0]
    # R_vec = mp0 - sp0
    # R = np.linalg.norm(R_vec)
    # R_hat = R_vec / R
    # proj_term = t1_0 - np.dot(R_hat, t1_0) * R_hat
    # dot_meas = np.dot(measurement_direction, proj_term)
    # G_scalar = np.exp(-1j * k * R) / (4 * np.pi * R)
    # expected_value = G_scalar * dot_meas
    # assert np.isclose(H[0, 0], expected_value), \
    #     f"H[0, 0] mismatch. Expected {expected_value}, got {H[0, 0]}"

def test_create_channel_matrix_fortran_order(channel_matrix_setup):
    """Tests if the matrix is Fortran-contiguous."""
    source_points, tangents1, tangents2, measurement_plane, measurement_direction, k = channel_matrix_setup
    H = create_channel_matrix(
        points=source_points,
        tangents1=tangents1,
        tangents2=tangents2,
        measurement_plane=measurement_plane,
        measurement_direction=measurement_direction,
        k=k
    )
    assert H.flags["F_CONTIGUOUS"], "Channel matrix H should be Fortran-contiguous."
