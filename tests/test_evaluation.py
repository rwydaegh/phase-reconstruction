# tests/test_evaluation.py
import numpy as np
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock, call

# Module to test
from src import evaluation
# Need to patch create_channel_matrix within evaluation module's scope
# from src.create_channel_matrix import create_channel_matrix

# --- Fixtures ---

@pytest.fixture
def default_global_config():
    """Scalar model config."""
    return OmegaConf.create({
        'wavelength': 0.01, # k = 2*pi/0.01 approx 628
        'use_vector_model': False,
        'measurement_direction': [1.0, 0.0, 0.0] # Irrelevant if scalar
    })

@pytest.fixture
def vector_global_config():
    """Vector model config."""
    return OmegaConf.create({
        'wavelength': 0.01,
        'use_vector_model': True,
        'measurement_direction': [1.0, 0.0, 0.0]
    })

@pytest.fixture
def dummy_geometry():
    """Provides consistent dummy geometry."""
    n_points = 10
    points_true = np.random.rand(n_points, 3)
    points_perturbed = points_true + np.random.normal(0, 0.01, size=(n_points, 3))
    # Dummy tangents (normalized)
    t1 = np.random.rand(n_points, 3); t1 /= np.linalg.norm(t1, axis=1)[:, np.newaxis]
    t2 = np.random.rand(n_points, 3); t2 /= np.linalg.norm(t2, axis=1)[:, np.newaxis]
    # Assume tangents don't change much for perturbation in this dummy data
    return {
        'points_true': points_true, 'tangents1_true': t1, 'tangents2_true': t2,
        'points_perturbed': points_perturbed, 'tangents1_perturbed': t1.copy(), 'tangents2_perturbed': t2.copy()
    }

@pytest.fixture
def dummy_coeffs_scalar(dummy_geometry):
    """Dummy coefficients for scalar model."""
    n_points = dummy_geometry['points_perturbed'].shape[0]
    return np.random.rand(n_points) + 1j * np.random.rand(n_points)

@pytest.fixture
def dummy_coeffs_vector(dummy_geometry):
    """Dummy coefficients for vector model."""
    n_points = dummy_geometry['points_perturbed'].shape[0]
    return np.random.rand(2 * n_points) + 1j * np.random.rand(2 * n_points)

@pytest.fixture
def dummy_currents_scalar(dummy_geometry):
    """Dummy original currents for scalar model."""
    n_points = dummy_geometry['points_true'].shape[0]
    return np.random.rand(n_points) + 1j * np.random.rand(n_points)

@pytest.fixture
def dummy_currents_vector(dummy_geometry):
    """Dummy original currents for vector model."""
    n_points = dummy_geometry['points_true'].shape[0]
    return np.random.rand(2 * n_points) + 1j * np.random.rand(2 * n_points)


@pytest.fixture
def dummy_test_planes():
    """Provides a list of dummy processed test plane data."""
    n_meas1 = 25 # 5x5
    n_meas2 = 36 # 6x6
    return [
        { # Real plane
            'name': 'real_test', 'is_real_plane': True, 'use_test': True,
            'coordinates': np.random.rand(n_meas1, 3),
            'measured_magnitude': np.random.rand(n_meas1) + 0.1, # Ensure positive
            'original_data_shape': (5, 5)
        },
        { # Simulated plane
            'name': 'sim_test', 'is_real_plane': False, 'use_test': True,
            'coordinates': np.random.rand(n_meas2, 3),
            'original_data_shape': (6, 6)
            # No measured_magnitude for simulated
        }
    ]

# --- Tests ---

def test_import():
    """Test module import."""
    assert evaluation is not None

@patch('src.evaluation.create_channel_matrix')
def test_evaluation_scalar(mock_create_H, default_global_config, dummy_geometry,
                           dummy_coeffs_scalar, dummy_currents_scalar, dummy_test_planes):
    """Test evaluation workflow with scalar model."""
    n_meas1 = dummy_test_planes[0]['coordinates'].shape[0]
    n_meas2 = dummy_test_planes[1]['coordinates'].shape[0]
    n_points = dummy_geometry['points_true'].shape[0]

    # Mock create_channel_matrix return values
    # H_true_sim, H_test_real, H_test_sim
    # Correct shapes: Rows = # measurement points, Cols = # coeffs (n_points for scalar)
    mock_H_true_sim = np.random.rand(n_meas2, n_points) + 1j * np.random.rand(n_meas2, n_points) # For sim_test ground truth (n_meas2 points)
    mock_H_test_real = np.random.rand(n_meas1, n_points) + 1j * np.random.rand(n_meas1, n_points) # For real_test reconstruction (n_meas1 points)
    mock_H_test_sim = np.random.rand(n_meas2, n_points) + 1j * np.random.rand(n_meas2, n_points)  # For sim_test reconstruction (n_meas2 points)
    # Corrected Call Order:
    # 1. H_test for real_test reconstruction (uses points_perturbed)
    # 2. H_true for sim_test ground truth (uses points_true)
    # 3. H_test for sim_test reconstruction (uses points_perturbed)
    mock_create_H.side_effect = [mock_H_test_real, mock_H_true_sim, mock_H_test_sim]

    results = evaluation.evaluate_on_test_planes(
        test_planes=dummy_test_planes,
        final_coefficients=dummy_coeffs_scalar,
        original_currents=dummy_currents_scalar,
        config=default_global_config,
        **dummy_geometry # Pass all geometry points/tangents
    )

    assert mock_create_H.call_count == 3
    # Check calls (simplified check, could be more specific about args)
    # Use np.allclose for comparing numpy arrays with potential float differences
    # Check points argument for each call based on the corrected order
    assert np.allclose(mock_create_H.call_args_list[0][1]['points'], dummy_geometry['points_perturbed']) # Call 1: H_test_real
    assert np.allclose(mock_create_H.call_args_list[1][1]['points'], dummy_geometry['points_true'])      # Call 2: H_true_sim
    assert np.allclose(mock_create_H.call_args_list[2][1]['points'], dummy_geometry['points_perturbed']) # Call 3: H_test_sim

    assert 'real_test' in results
    assert 'sim_test' in results
    assert 'rmse' in results['real_test']
    assert 'correlation' in results['real_test']
    assert 'rmse' in results['sim_test']
    assert 'correlation' in results['sim_test']
    # Check shapes based on the mocked H matrices and coefficients
    # Reconstructed = H_test @ coeffs
    # Ground Truth (Real) = measured_magnitude
    # Ground Truth (Sim) = H_true_sim @ original_currents
    assert results['real_test']['ground_truth_magnitude'].shape == (n_meas1,) # From dummy_test_planes
    assert results['real_test']['reconstructed_magnitude'].shape == (n_meas1,) # H_test_real is (n_meas1, n_points), coeffs is (n_points,)
    assert results['sim_test']['ground_truth_magnitude'].shape == (n_meas2,) # H_true_sim is (n_meas2, n_points), currents is (n_points,)
    assert results['sim_test']['reconstructed_magnitude'].shape == (n_meas2,) # H_test_sim is (n_meas2, n_points), coeffs is (n_points,)


@patch('src.evaluation.create_channel_matrix')
def test_evaluation_vector(mock_create_H, vector_global_config, dummy_geometry,
                           dummy_coeffs_vector, dummy_currents_vector, dummy_test_planes):
    """Test evaluation workflow with vector model."""
    n_meas1 = dummy_test_planes[0]['coordinates'].shape[0]
    n_meas2 = dummy_test_planes[1]['coordinates'].shape[0]
    n_points = dummy_geometry['points_true'].shape[0]

    # Mock create_channel_matrix return values (now expect 2*n_points columns)
    # Correct shapes: Rows = # measurement points, Cols = # coeffs (2*n_points for vector)
    mock_H_true_sim = np.random.rand(n_meas2, 2*n_points) + 1j * np.random.rand(n_meas2, 2*n_points) # For sim_test ground truth (n_meas2 points)
    mock_H_test_real = np.random.rand(n_meas1, 2*n_points) + 1j * np.random.rand(n_meas1, 2*n_points) # For real_test reconstruction (n_meas1 points)
    mock_H_test_sim = np.random.rand(n_meas2, 2*n_points) + 1j * np.random.rand(n_meas2, 2*n_points)  # For sim_test reconstruction (n_meas2 points)
    # Corrected Call Order: H_test_real, H_true_sim, H_test_sim
    mock_create_H.side_effect = [mock_H_test_real, mock_H_true_sim, mock_H_test_sim]

    results = evaluation.evaluate_on_test_planes(
        test_planes=dummy_test_planes,
        final_coefficients=dummy_coeffs_vector,
        original_currents=dummy_currents_vector,
        config=vector_global_config,
        **dummy_geometry
    )

    assert mock_create_H.call_count == 3
    # Check tangents were passed in calls
    # Use np.allclose for comparing numpy arrays with potential float differences
    # Check tangents argument for each call based on the corrected order
    assert np.allclose(mock_create_H.call_args_list[0][1]['tangents1'], dummy_geometry['tangents1_perturbed']) # Call 1: H_test_real
    assert np.allclose(mock_create_H.call_args_list[1][1]['tangents1'], dummy_geometry['tangents1_true'])      # Call 2: H_true_sim
    assert np.allclose(mock_create_H.call_args_list[2][1]['tangents1'], dummy_geometry['tangents1_perturbed']) # Call 3: H_test_sim

    assert 'real_test' in results
    assert 'sim_test' in results
    # Check shapes (Vector model: H is (N_m, 2*N_p), coeffs/currents are (2*N_p,))
    assert results['real_test']['ground_truth_magnitude'].shape == (n_meas1,)
    assert results['real_test']['reconstructed_magnitude'].shape == (n_meas1,)
    assert results['sim_test']['ground_truth_magnitude'].shape == (n_meas2,)
    assert results['sim_test']['reconstructed_magnitude'].shape == (n_meas2,)


def test_evaluation_no_test_planes(default_global_config, dummy_geometry,
                                   dummy_coeffs_scalar, dummy_currents_scalar):
    """Test evaluation with an empty list of test planes."""
    results = evaluation.evaluate_on_test_planes(
        test_planes=[], # Empty list
        final_coefficients=dummy_coeffs_scalar,
        original_currents=dummy_currents_scalar,
        config=default_global_config,
        **dummy_geometry
    )
    assert results == {}


@patch('src.evaluation.create_channel_matrix', side_effect=ValueError("H creation failed"))
def test_evaluation_h_creation_error(mock_create_H, default_global_config, dummy_geometry,
                                     dummy_coeffs_scalar, dummy_currents_scalar, dummy_test_planes):
    """Test handling when H matrix creation fails."""
    # create_channel_matrix will raise ValueError on the first call (H_true_sim_plane)
    results = evaluation.evaluate_on_test_planes(
        test_planes=dummy_test_planes,
        final_coefficients=dummy_coeffs_scalar,
        original_currents=dummy_currents_scalar,
        config=default_global_config,
        **dummy_geometry
    )
    # Should skip the plane where H creation failed, but might process others if error was later
    # In this setup, the first call fails (for sim_test ground truth), so sim_test is skipped.
    # The second call (for real_test H_test) would also fail here.
    assert 'sim_test' not in results
    assert 'real_test' not in results # Because H_test creation would also fail
    # The loop continues even if one plane fails. It tries to calculate
    # H_true for sim_test (fails), then H_test for real_test (fails).
    assert mock_create_H.call_count == 2


def test_evaluation_coeff_shape_mismatch(default_global_config, dummy_geometry,
                                         dummy_currents_scalar, dummy_test_planes):
    """Test error handling for coefficient shape mismatch."""
    wrong_coeffs = np.random.rand(5) + 1j*np.random.rand(5) # Clearly wrong size
    with pytest.raises(ValueError, match="Shape mismatch for evaluation"):
        evaluation.evaluate_on_test_planes(
            test_planes=dummy_test_planes,
            final_coefficients=wrong_coeffs,
            original_currents=dummy_currents_scalar,
            config=default_global_config,
            **dummy_geometry
        )

# TODO: Add tests for shape mismatches during ground truth calculation
# TODO: Add tests for shape mismatches between ground truth and reconstructed magnitudes