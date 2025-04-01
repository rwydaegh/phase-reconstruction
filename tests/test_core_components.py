import pytest
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import math # For pi

# Import functions under test
from src.create_test_pointcloud import create_test_pointcloud
from src.create_channel_matrix import create_channel_matrix
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval


# Helper to calculate k from wavelength
def calculate_k(wavelength: float) -> float:
    return 2 * math.pi / wavelength

@pytest.fixture
def basic_config() -> DictConfig:
    """Provides a basic Hydra config loaded from defaults for component testing."""
    with hydra.initialize(config_path="../conf", version_base=None):
        # Load the default config and override specific values for testing
        cfg = hydra.compose(config_name="config", overrides=[
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


# --- Tests for create_channel_matrix ---

@pytest.fixture
def channel_matrix_setup(basic_config: DictConfig):
    """Provides points and measurement plane for channel matrix tests."""
    # Simple point cloud: 2 points
    source_points = np.array([
        [0.1, 0.2, 0.3],
        [-0.1, -0.2, -0.3]
    ])
    # Simple measurement plane: 4 points (2x2 grid)
    res = 2
    plane_size = 0.2
    x = np.linspace(-plane_size / 2, plane_size / 2, res)
    y = np.linspace(-plane_size / 2, plane_size / 2, res)
    X, Y = np.meshgrid(x, y)
    # Place plane at z=0.5 (relative to room center if room_size=1.0)
    measurement_plane = np.stack([X, Y, np.ones_like(X) * basic_config.room_size / 2.0], axis=-1)
    # measurement_plane shape: (2, 2, 3)
    # measurement_points shape (flattened): (4, 3)

    k = calculate_k(basic_config.wavelength) # Calculate k from wavelength in config
    return source_points, measurement_plane, k

def test_create_channel_matrix_shape(channel_matrix_setup):
    """Tests the output shape of the channel matrix H."""
    source_points, measurement_plane, k = channel_matrix_setup
    num_source_points = source_points.shape[0] # Should be 2
    num_measurement_points = measurement_plane.shape[0] * measurement_plane.shape[1] # Should be 4

    H = create_channel_matrix(source_points, measurement_plane, k)

    # Expected shape: (num_measurement_points, num_source_points)
    expected_shape = (num_measurement_points, num_source_points)
    assert H.shape == expected_shape, f"Expected H shape {expected_shape}, but got {H.shape}"

def test_create_channel_matrix_dtype_and_values(channel_matrix_setup):
    """Tests the data type and calculates one value for verification."""
    source_points, measurement_plane, k = channel_matrix_setup
    H = create_channel_matrix(source_points, measurement_plane, k)

    # Check dtype
    assert H.dtype == np.complex128, f"Expected dtype complex128, but got {H.dtype}"

    # Check a specific value: H[0, 0]
    # Measurement point 0 (flattened): measurement_plane[0, 0, :] = [-0.1, -0.1, 0.5]
    # Source point 0: source_points[0, :] = [0.1, 0.2, 0.3]
    mp0 = measurement_plane.reshape(-1, 3)[0] # [-0.1, -0.1, 0.5]
    sp0 = source_points[0]                   # [ 0.1,  0.2, 0.3]
    distance = np.linalg.norm(mp0 - sp0)
    expected_value = np.exp(-1j * k * distance) / (4 * np.pi * distance)

    assert np.isclose(H[0, 0], expected_value), \
        f"H[0, 0] mismatch. Expected {expected_value}, got {H[0, 0]}"

def test_create_channel_matrix_fortran_order(channel_matrix_setup):
    """Tests if the matrix is Fortran-contiguous."""
    source_points, measurement_plane, k = channel_matrix_setup
    H = create_channel_matrix(source_points, measurement_plane, k)
    assert H.flags['F_CONTIGUOUS'], "Channel matrix H should be Fortran-contiguous."


# --- Tests for holographic_phase_retrieval (Gerchberg-Saxton) ---

@pytest.fixture
def gs_setup(basic_config: DictConfig): # Use basic_config to get wavelength etc.
    """Provides a simple setup for Gerchberg-Saxton tests using Hydra config."""
    # Simple setup: 2 sources, 4 measurement points
    num_sources = 2
    num_measurements = 4
    np.random.seed(42) # for reproducibility

    # Create a dummy channel matrix H (complex)
    H = np.random.rand(num_measurements, num_sources) + 1j * np.random.rand(num_measurements, num_sources)
    H = np.asfortranarray(H) # Ensure Fortran order as in the original code

    # Create known 'true' coefficients (complex)
    true_coefficients = np.random.rand(num_sources) + 1j * np.random.rand(num_sources)

    # Calculate the 'true' field based on H and true_coefficients
    true_field = H @ true_coefficients # Shape: (num_measurements,)

    # Calculate the 'measured' magnitude (what the algorithm receives)
    measured_magnitude = np.abs(true_field)

    # Load base config and override GS parameters for the test
    # We use basic_config fixture indirectly via dependency injection,
    # but create a specific config for GS tests here if needed, or just use basic_config.
    # For simplicity, let's assume basic_config has reasonable defaults for GS,
    # or we can override specific GS params if necessary.
    # Let's create a dedicated GS config by composing again.
    with hydra.initialize(config_path="../conf", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[
            # Inherit most from basic_config's overrides if possible,
            # or set specific GS test values here.
            # Let's assume basic_config is sufficient for now, but add GS specific overrides:
            "gs_iterations=50",
            "convergence_threshold=1e-5",
            "regularization=1e-4",
            "adaptive_regularization=False",
            "enable_perturbations=False", # Keep it simple for basic tests
            "verbose=False",
            "no_plot=True",
            "return_history=False" # Default for some tests
        ])

    # Ensure measured_magnitude is flat (num_measurements,)
    measured_magnitude = measured_magnitude.flatten()

    return H, measured_magnitude, true_field, cfg

def test_gs_basic_convergence(gs_setup): # Hydra config 'cfg' is now passed
    """Tests if the GS algorithm converges to a low error using Hydra config."""
    H, measured_magnitude, true_field, cfg = gs_setup

    # Run the algorithm using parameters from the Hydra config
    final_coefficients, stats = holographic_phase_retrieval(
        channel_matrix=H,
        measured_magnitude=measured_magnitude,
        num_iterations=cfg.gs_iterations,
        convergence_threshold=cfg.convergence_threshold,
        regularization=cfg.regularization,
        adaptive_regularization=cfg.adaptive_regularization,
        enable_perturbations=cfg.enable_perturbations,
        verbose=cfg.verbose,
        no_plot=cfg.no_plot,
        return_history=False # Explicitly false for this test
    )

    final_error = stats['final_error']
    initial_error = stats['errors'][0] # Get the first error calculated

    print(f"\nGS Convergence Test - Initial Error: {initial_error:.6f}, Final Error: {final_error:.6f}")

    # Assert that the final error is significantly lower than the initial error
    assert final_error < initial_error * 0.1, "Final error should be much lower than initial error."
    # Assert that the final error is below a reasonable threshold
    # This threshold might need adjustment based on the complexity of the test case
    assert final_error < 1e-3, f"Final error {final_error:.6f} did not reach expected threshold."

def test_gs_output_shape(gs_setup): # Hydra config 'cfg' is now passed
    """Tests the shape of the returned coefficients using Hydra config."""
    H, measured_magnitude, _, cfg = gs_setup
    num_sources = H.shape[1]

    final_coefficients, _ = holographic_phase_retrieval(
        channel_matrix=H,
        measured_magnitude=measured_magnitude,
        num_iterations=cfg.gs_iterations,
        convergence_threshold=cfg.convergence_threshold,
        regularization=cfg.regularization,
        adaptive_regularization=cfg.adaptive_regularization,
        enable_perturbations=cfg.enable_perturbations,
        verbose=cfg.verbose,
        no_plot=cfg.no_plot,
        return_history=False
    )

    assert final_coefficients.shape == (num_sources,), \
        f"Expected coefficient shape ({num_sources},), but got {final_coefficients.shape}"

def test_gs_return_history(gs_setup): # Hydra config 'cfg' is now passed
    """Tests if history arrays are returned correctly using Hydra config."""
    H, measured_magnitude, _, cfg = gs_setup
    num_sources = H.shape[1]
    num_measurements = H.shape[0]
    # Use num_iterations from the loaded config

    # Run with history enabled
    final_coeffs, coeff_history, field_history, stats = holographic_phase_retrieval(
        channel_matrix=H,
        measured_magnitude=measured_magnitude,
        num_iterations=cfg.gs_iterations,
        convergence_threshold=cfg.convergence_threshold,
        regularization=cfg.regularization,
        adaptive_regularization=cfg.adaptive_regularization,
        enable_perturbations=cfg.enable_perturbations,
        verbose=cfg.verbose,
        no_plot=cfg.no_plot,
        return_history=True # Explicitly true for this test
    )

    # Convergence might happen early, so check actual iterations run
    iterations_run = stats['iterations']

    assert coeff_history is not None, "Coefficient history should be returned."
    assert field_history is not None, "Field history should be returned."

    # Check shapes based on actual iterations run
    assert coeff_history.shape == (iterations_run, num_sources), \
        f"Expected coeff_history shape ({iterations_run}, {num_sources}), got {coeff_history.shape}"
    assert field_history.shape == (iterations_run, num_measurements), \
        f"Expected field_history shape ({iterations_run}, {num_measurements}), got {field_history.shape}"
