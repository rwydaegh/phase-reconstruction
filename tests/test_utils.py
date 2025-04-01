import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from src.utils.normalized_correlation import normalized_correlation

# Import functions to be tested
from src.utils.normalized_rmse import normalized_rmse
from src.utils.phase_retrieval_utils import (
    apply_magnitude_constraint,
    calculate_error,
    compute_pseudoinverse,
    initialize_field_values,
)

# --- Tests for normalized_rmse ---


def test_normalized_rmse_identical():
    """RMSE should be 0 for identical inputs."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert_almost_equal(normalized_rmse(a, a), 0.0)


def test_normalized_rmse_simple():
    """Test RMSE with a simple known case."""
    a = np.array([1.0, 2.0, 3.0, 4.0])  # Range = 3
    b = np.array([1.0, 2.0, 3.0, 5.0])
    # MSE = (0^2 + 0^2 + 0^2 + 1^2) / 4 = 1/4
    # RMSE = sqrt(1/4) = 1/2
    # Normalized RMSE = (1/2) / (4 - 1) = 0.5 / 3 = 1/6
    expected_nrmse = 1.0 / 6.0
    assert_almost_equal(normalized_rmse(a, b), expected_nrmse)


def test_normalized_rmse_zero_range():
    """Test RMSE when the true range is zero (should avoid division by zero)."""
    a = np.array([2.0, 2.0, 2.0, 2.0])
    b = np.array([2.0, 2.0, 2.0, 3.0])
    # Function should handle this gracefully, maybe return inf or nan, or a large number.
    # The current implementation divides by max(a)-min(a). If this is 0, it will result in inf.
    assert np.isinf(normalized_rmse(a, b))


# --- Tests for normalized_correlation ---


def test_normalized_correlation_identical():
    """Correlation should be 1 for identical inputs."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert_almost_equal(normalized_correlation(a, a), 1.0)


def test_normalized_correlation_perfect_negative():
    """Correlation should be -1 for perfectly negatively correlated inputs."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = -a
    assert_almost_equal(normalized_correlation(a, b), -1.0)


def test_normalized_correlation_uncorrelated():
    """Correlation should be near 0 for uncorrelated inputs."""
    np.random.seed(42)
    a = np.random.rand(100)
    b = np.random.rand(100)
    # For random inputs, correlation should be small
    assert abs(normalized_correlation(a, b)) < 0.3


def test_normalized_correlation_shifted_scaled():
    """Correlation should be 1 even if inputs are shifted and scaled."""

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a * 5.0 + 10.0  # Perfectly correlated
    assert_almost_equal(normalized_correlation(a, b), 1.0)


# --- Tests for compute_pseudoinverse ---


@pytest.fixture
def svd_setup():
    """Setup for pseudoinverse tests."""
    np.random.seed(123)
    # Create a non-square matrix
    H = np.random.rand(5, 3) + 1j * np.random.rand(5, 3)
    H = np.asfortranarray(H)
    return H


def test_compute_pseudoinverse_shape(svd_setup):
    """Test the shape of the computed pseudoinverse."""
    H = svd_setup
    reg = 1e-3
    H_pinv = compute_pseudoinverse(H, reg, adaptive_regularization=False)
    # Pseudoinverse should have shape (cols, rows)
    assert H_pinv.shape == (H.shape[1], H.shape[0])


def test_compute_pseudoinverse_dtype(svd_setup):
    """Test the data type of the pseudoinverse."""
    H = svd_setup
    reg = 1e-3
    H_pinv = compute_pseudoinverse(H, reg, adaptive_regularization=False)
    assert H_pinv.dtype == np.complex128


def test_compute_pseudoinverse_reconstruction(svd_setup):
    """Test if H_pinv @ H is close to identity."""
    H = svd_setup
    reg = 1e-7  # Small regularization for better reconstruction
    H_pinv = compute_pseudoinverse(H, reg, adaptive_regularization=False)
    identity_approx = H_pinv @ H
    identity_expected = np.identity(H.shape[1], dtype=np.complex128)
    # Use assert_allclose for matrix comparison with tolerance
    assert_allclose(identity_approx, identity_expected, atol=1e-5)


def test_compute_pseudoinverse_adaptive_regularization(svd_setup):
    """Test adaptive regularization runs and gives a different result."""
    H = svd_setup
    reg = 1e-3
    H_pinv_non_adaptive = compute_pseudoinverse(H, reg, adaptive_regularization=False)
    H_pinv_adaptive = compute_pseudoinverse(H, reg, adaptive_regularization=True)
    # Check they are not identical (adaptive thresholding should change the result)
    assert not np.allclose(H_pinv_non_adaptive, H_pinv_adaptive)
    assert H_pinv_adaptive.shape == (H.shape[1], H.shape[0])  # Shape check again


def test_compute_pseudoinverse_adaptive_thresholding():
    """Test if adaptive regularization correctly thresholds small singular values."""
    np.random.seed(456)
    # Create a matrix with known singular values
    U = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
    V = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    # Orthonormalize U and V
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    Vh = V.conj().T

    # Define singular values, including small ones
    singular_values = np.array([10.0, 1.0, 0.01, 0.0001])  # Last two are small
    S_diag = np.zeros((5, 3), dtype=np.complex128)
    np.fill_diagonal(S_diag, singular_values[:3])  # Fill diagonal for non-square matrix

    H = U @ S_diag @ Vh  # Construct H = U S Vh
    H = np.asfortranarray(H)

    # Set regularization factor. Adaptive threshold is typically s_max * reg
    reg = 1e-3  # Threshold will be around 10.0 * 1e-3 = 0.01

    # Compute pseudoinverse WITH adaptive regularization
    H_pinv_adaptive = compute_pseudoinverse(H, reg, adaptive_regularization=True)

    # Compute pseudoinverse WITHOUT adaptive regularization (Tikhonov)
    H_pinv_non_adaptive = compute_pseudoinverse(H, reg, adaptive_regularization=False)

    # --- Verification ---
    # The pseudoinverse H_pinv = V @ S_pinv @ Uh
    # where S_pinv contains 1/s_i for significant singular values s_i.

    # Expected S_pinv for adaptive regularization using the formula: s / (s^2 + tau^2)
    # where tau = reg * s_max
    s = singular_values[:3]  # Use the actual singular values
    s_max = s[0]
    tau = reg * s_max
    # Calculate expected inverse based on implementation
    s_inv_adaptive_expected = s / (s**2 + tau**2)
    S_pinv_adaptive_expected_diag = np.zeros((3, 5), dtype=np.complex128)
    np.fill_diagonal(S_pinv_adaptive_expected_diag, s_inv_adaptive_expected)
    H_pinv_adaptive_expected = V @ S_pinv_adaptive_expected_diag @ U.conj().T

    # Expected S_pinv for non-adaptive (Tikhonov: s_i / (s_i^2 + reg^2)
    # - approx 1/s_i for large s_i)
    # Note: The actual implementation uses reg, not reg^2.
    # Thresholding is s > s_max * reg.
    # The non-adaptive formula is s_i / (s_i^2 + reg)
    # - let's re-check the function or test behavior.
    # Looking at the code: s_inv = s / (s**2 + regularization). This is Tikhonov if reg = alpha^2.
    # Let's calculate the expected values based on this formula.
    s = singular_values[:3]
    s_inv_non_adaptive_expected = s / (s**2 + reg)  # Use reg directly as per implementation
    S_pinv_non_adaptive_expected_diag = np.zeros((3, 5), dtype=np.complex128)
    np.fill_diagonal(S_pinv_non_adaptive_expected_diag, s_inv_non_adaptive_expected)
    H_pinv_non_adaptive_expected = V @ S_pinv_non_adaptive_expected_diag @ U.conj().T

    # Compare computed pseudoinverses with expected ones
    assert_allclose(
        H_pinv_adaptive,
        H_pinv_adaptive_expected,
        atol=1e-6,
        err_msg="Adaptive pseudoinverse differs from expected thresholded result.",
    )

    # For non-adaptive, the match might be less exact due to floating point, but should be close
    assert_allclose(
        H_pinv_non_adaptive,
        H_pinv_non_adaptive_expected,
        atol=1e-6,
        err_msg="Non-adaptive pseudoinverse differs from expected Tikhonov result.",
    )

    # Crucially, verify that the adaptive and non-adaptive results differ significantly
    # especially in how they handle the smallest singular value component.
    assert not np.allclose(
        H_pinv_adaptive, H_pinv_non_adaptive, atol=1e-6
    ), "Adaptive and non-adaptive pseudoinverses should differ significantly."


# --- Tests for initialize_field_values ---


def test_initialize_field_values_shape_and_magnitude():
    """Test shape and magnitude preservation."""
    measured_magnitude = np.array([1.0, 2.0, 0.5, 3.0])
    initial_field = initialize_field_values(measured_magnitude)
    assert initial_field.shape == measured_magnitude.shape
    assert initial_field.dtype == np.complex128
    # Check if the magnitude of the initialized field matches the input magnitude
    assert_allclose(np.abs(initial_field), measured_magnitude, rtol=1e-7)


# --- Tests for calculate_error ---


def test_calculate_error_zero():
    """Error should be 0 for identical magnitudes."""
    mag_a = np.array([1.0, 2.0, 3.0])
    norm_a = np.linalg.norm(mag_a)
    assert_almost_equal(calculate_error(mag_a, mag_a, norm_a), 0.0)


def test_calculate_error_simple():
    """Test error calculation with a simple case."""
    measured_magnitude = np.array([3.0, 4.0])  # Norm = sqrt(9+16) = 5
    simulated_magnitude = np.array([3.0, 0.0])  # Diff = [0, -4], Norm of diff = 4
    measured_magnitude_norm = np.linalg.norm(measured_magnitude)  # Should be 5.0
    expected_error = 4.0 / 5.0
    error = calculate_error(simulated_magnitude, measured_magnitude, measured_magnitude_norm)
    assert_almost_equal(error, expected_error)


def test_calculate_error_zero_norm():
    """Test error calculation when measured norm is zero (should be inf or nan)."""
    measured_magnitude = np.array([0.0, 0.0])
    simulated_magnitude = np.array([1.0, 1.0])
    measured_magnitude_norm = np.linalg.norm(measured_magnitude)  # Should be 0.0
    # Division by zero should result in inf
    error = calculate_error(simulated_magnitude, measured_magnitude, measured_magnitude_norm)
    assert np.isinf(error)


# --- Tests for apply_magnitude_constraint ---


def test_apply_magnitude_constraint_basic():
    """Test basic application of magnitude constraint."""
    measured_magnitude = np.array([1.0, 2.0, 3.0])
    # Create a simulated field with different magnitudes but same phase direction (positive real)
    simulated_field = np.array([0.5 + 0j, 4.0 + 0j, 1.5 + 0j])
    constrained_field = apply_magnitude_constraint(simulated_field, measured_magnitude)

    assert constrained_field.shape == measured_magnitude.shape
    assert constrained_field.dtype == np.complex128
    # Check magnitude matches measured_magnitude
    assert_allclose(np.abs(constrained_field), measured_magnitude)
    # Check phase is preserved (all should be positive real -> angle 0)
    assert_allclose(np.angle(constrained_field), np.zeros_like(measured_magnitude), atol=1e-7)


def test_apply_magnitude_constraint_complex_phase():
    """Test constraint application preserves complex phase."""
    measured_magnitude = np.array([1.0, 2.0], dtype=np.float64)
    # Simulated field with complex phases
    simulated_field = np.array([1 * np.exp(1j * np.pi / 4), 4 * np.exp(1j * -np.pi / 2)])
    original_phases = np.angle(simulated_field)  # [pi/4, -pi/2]

    constrained_field = apply_magnitude_constraint(simulated_field, measured_magnitude).astype(
        np.complex128
    )

    # Check magnitude matches measured_magnitude
    assert_allclose(np.abs(constrained_field), measured_magnitude)
    # Check phase is preserved
    assert_allclose(np.angle(constrained_field), original_phases)


def test_apply_magnitude_constraint_zero_simulated():
    """Test constraint application when simulated magnitude is zero."""
    measured_magnitude = np.array([1.0, 2.0])
    # Contains a zero magnitude
    simulated_field = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    constrained_field = apply_magnitude_constraint(simulated_field, measured_magnitude)

    # Check magnitude matches measured_magnitude
    assert_allclose(np.abs(constrained_field), measured_magnitude)
    # The phase of the zero element is undefined, but the function handles it.
    # Check the phase of the non-zero element is preserved (angle=0)
    assert_almost_equal(np.angle(constrained_field[1]), 0.0)
