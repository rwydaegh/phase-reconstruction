# tests/test_plane_processing.py
import os
import pickle
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

# Module to test
from src import plane_processing

# Helper to create dummy measurement data file
@pytest.fixture(scope="function")
def dummy_pickle_file():
    shape = (20, 30) # Use non-square shape for testing
    data = {
        'results': np.random.rand(*shape) + 1j * np.random.rand(*shape),
        'continuous_axis': 'z',
        'discrete_axis': 'y',
        'points_continuous': np.linspace(-100, 100, shape[1]), # mm
        'points_discrete': np.linspace(-80, 80, shape[0]), # mm
        'frequency': 28e9
    }
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "dummy_measurement.pkl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        yield filepath # Provide the path to the test

# Test module import
def test_import():
    """Test if the module can be imported"""
    assert plane_processing is not None

# --- Tests for load_real_plane (using dummy data) ---

# Mock the io functions for initial tests as they might not be ready/refactored
@patch('src.plane_processing.load_measurement_data')
@patch('src.plane_processing.sample_measurement_data')
def test_load_real_plane_basic(mock_sample_data, mock_load_data, dummy_pickle_file):
    """Test basic execution of load_real_plane with mocked IO."""
    target_res = 15
    # Mock return values
    mock_load_data.return_value = {
        'results': np.random.rand(100, 100), 'continuous_axis': 'z', 'discrete_axis': 'y',
        'points_continuous': np.linspace(-1, 1, 100), 'points_discrete': np.linspace(-1, 1, 100),
        'frequency': 28e9
    }
    mock_sample_data.return_value = {
        'results': np.random.rand(target_res, target_res), 'continuous_axis': 'z', 'discrete_axis': 'y',
        'points_continuous': np.linspace(-1, 1, target_res), 'points_discrete': np.linspace(-1, 1, target_res),
        'frequency': 28e9
    }

    plane_cfg = OmegaConf.create({
        'name': 'test_real',
        'is_real_plane': True,
        'use_train': True, 'use_test': False,
        'translation': [0.1, 0.2, 0.3],
        'measured_data_path': os.path.basename(dummy_pickle_file), # Use relative path for config
        'target_resolution': target_res
    })

    # Use the directory containing the dummy file as base_dir
    base_dir = os.path.dirname(dummy_pickle_file)
    result = plane_processing.load_real_plane(plane_cfg, base_dir)

    assert result is not None
    assert 'coordinates' in result
    assert 'measured_magnitude' in result
    assert result['coordinates'].shape == (target_res * target_res, 3)
    assert result['measured_magnitude'].shape == (target_res * target_res,)
    assert result['original_data_shape'] == (target_res, target_res)
    # Check if translation was roughly applied (mean should be near translation)
    assert np.allclose(np.mean(result['coordinates'], axis=0), plane_cfg.translation, atol=1e-6)
    mock_load_data.assert_called_once_with(dummy_pickle_file)
    mock_sample_data.assert_called_once()


def test_load_real_plane_missing_config():
    """Test handling of missing essential config keys."""
    plane_cfg_no_path = OmegaConf.create({'target_resolution': 50})
    plane_cfg_no_res = OmegaConf.create({'measured_data_path': 'dummy.pkl'})
    assert plane_processing.load_real_plane(plane_cfg_no_path, ".") is None
    assert plane_processing.load_real_plane(plane_cfg_no_res, ".") is None

def test_load_real_plane_file_not_found():
    """Test handling when the data file doesn't exist."""
    plane_cfg = OmegaConf.create({
        'measured_data_path': 'non_existent_file.pkl',
        'target_resolution': 50
    })
    assert plane_processing.load_real_plane(plane_cfg, ".") is None

# --- Tests for generate_simulated_plane ---

@pytest.mark.parametrize("plane_type, expected_fixed_axis_index", [
    ('xy', 2), ('yz', 0), ('xz', 1)
])
def test_generate_simulated_plane_basic(plane_type, expected_fixed_axis_index):
    """Test basic generation for different plane types."""
    resolution = 10
    center = [1, 2, 3]
    size = [0.5, 0.6]
    translation = [0.1, 0.1, 0.1]
    plane_cfg = OmegaConf.create({
        'name': f'sim_{plane_type}',
        'is_real_plane': False,
        'use_train': False, 'use_test': True,
        'translation': translation,
        'plane_type': plane_type,
        'center': center,
        'size': size,
        'resolution': resolution
    })

    result = plane_processing.generate_simulated_plane(plane_cfg)

    assert result is not None
    assert 'coordinates' in result
    assert result['coordinates'].shape == (resolution * resolution, 3)
    assert 'measured_magnitude' not in result # Should not be generated here
    assert result['original_data_shape'] == (resolution, resolution)

    # Check if coordinates are centered around center+translation
    expected_center = np.array(center) + np.array(translation)
    assert np.allclose(np.mean(result['coordinates'], axis=0), expected_center, atol=1e-9)

    # Check if the correct axis is fixed (approximately) at the center value + translation
    fixed_axis_values = result['coordinates'][:, expected_fixed_axis_index]
    assert np.allclose(fixed_axis_values, expected_center[expected_fixed_axis_index])


def test_generate_simulated_plane_invalid_type():
    """Test handling of unsupported plane type."""
    plane_cfg = OmegaConf.create({
        'plane_type': 'abc', 'center': [0,0,0], 'size': [1,1], 'resolution': 10
    })
    assert plane_processing.generate_simulated_plane(plane_cfg) is None

def test_generate_simulated_plane_invalid_size():
    """Test handling of invalid size parameter."""
    plane_cfg = OmegaConf.create({
        'plane_type': 'xy', 'center': [0,0,0], 'size': [1], 'resolution': 10 # Invalid size
    })
    assert plane_processing.generate_simulated_plane(plane_cfg) is None


# --- Tests for process_plane_definitions ---

@patch('src.plane_processing.load_real_plane')
@patch('src.plane_processing.generate_simulated_plane')
def test_process_plane_definitions(mock_gen_sim, mock_load_real):
    """Test the main processing function orchestrates calls correctly."""
    # Mock return values - use side_effect for load_real to return distinct dicts
    mock_load_real.side_effect = [
        {'coordinates': np.random.rand(100, 3), 'measured_magnitude': np.random.rand(100)}, # Return value for 'real1' call
        {'coordinates': np.random.rand(64, 3), 'measured_magnitude': np.random.rand(64)}    # Return value for 'real2' call
    ]
    mock_gen_sim.return_value = {'coordinates': np.random.rand(25, 3)} # Only called once, return_value is fine

    plane_configs = [
        OmegaConf.create({'name': 'real1', 'is_real_plane': True, 'measured_data_path': 'path1', 'target_resolution': 10, 'use_train': True}),
        OmegaConf.create({'name': 'sim1', 'is_real_plane': False, 'plane_type': 'xy', 'resolution': 5, 'use_test': True}),
        OmegaConf.create({'name': 'real2', 'is_real_plane': True, 'measured_data_path': 'path2', 'target_resolution': 8, 'use_train': True, 'use_test': True}),
    ]

    results = plane_processing.process_plane_definitions(plane_configs, base_dir="/fake/dir")

    assert len(results) == 3
    # Check that the mocks were called with the correct config objects (or parts of them)
    # This is a bit more robust than just checking call count
    assert mock_load_real.call_args_list[0][0][0] is plane_configs[0] # First call with first config
    assert mock_gen_sim.call_args_list[0][0][0] is plane_configs[1] # Second call (first sim) with second config
    assert mock_load_real.call_args_list[1][0][0] is plane_configs[2] # Third call (second real) with third config
    assert mock_load_real.call_count == 2
    assert mock_gen_sim.call_count == 1

    # Check results (order might vary depending on dict iteration, so check by name)
    result_names = {r['name'] for r in results}
    assert result_names == {'real1', 'sim1', 'real2'}

    real1_res = next(r for r in results if r['name'] == 'real1')
    sim1_res = next(r for r in results if r['name'] == 'sim1')
    real2_res = next(r for r in results if r['name'] == 'real2')

    assert real1_res['is_real_plane'] is True
    assert real1_res['use_train'] is True
    assert real1_res['use_test'] is False # Default

    assert sim1_res['is_real_plane'] is False
    assert sim1_res['use_train'] is False # Default
    assert sim1_res['use_test'] is True

    assert real2_res['is_real_plane'] is True
    assert real2_res['use_train'] is True
    assert real2_res['use_test'] is True


@patch('src.plane_processing.load_real_plane', side_effect=Exception("Load failed"))
def test_process_plane_definitions_error_handling(mock_load_real):
    """Test that processing continues even if one plane fails."""
    plane_configs = [
        OmegaConf.create({'name': 'fail_real', 'is_real_plane': True, 'measured_data_path': 'fail', 'target_resolution': 10}),
        OmegaConf.create({'name': 'sim_ok', 'is_real_plane': False, 'plane_type': 'xy', 'resolution': 5}),
    ]
    # Need to patch generate_simulated_plane as well for the second config
    with patch('src.plane_processing.generate_simulated_plane') as mock_gen_sim:
         mock_gen_sim.return_value = {'coordinates': np.random.rand(25, 3)}
         results = plane_processing.process_plane_definitions(plane_configs, base_dir="/fake/dir")

    assert len(results) == 1 # Only the successful one
    assert results[0]['name'] == 'sim_ok'
    assert mock_load_real.call_count == 1
    assert mock_gen_sim.call_count == 1

def test_process_plane_definitions_empty_input():
    """Test with an empty list of plane definitions."""
    results = plane_processing.process_plane_definitions([], base_dir=".")
    assert results == []