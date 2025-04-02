# tests/test_source_handling.py
import os
import pickle
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

# Module to test
from src import source_handling


# Helper fixture to create dummy point cloud file in a temp directory
@pytest.fixture(scope="function")
def dummy_pointcloud_file_factory():
    temp_dirs = []

    def _create_file(
        filename="dummy_points.pkl", shape=(50, 3), add_tangents=False, add_normals=False
    ):
        data = np.random.rand(*shape) * 10  # Points up to ~10m away
        if add_normals or add_tangents:
            normals = np.random.rand(*shape)
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize
            # Add dummy distance column before normals
            dummy_dist = np.sqrt(np.sum(data**2, axis=1, keepdims=True))
            data = np.hstack((data, dummy_dist, normals))  # 7 cols: x,y,z, dist, nx,ny,nz
        if add_tangents:
            # Ensure normals exist before calculating tangents
            if data.shape[1] < 7:
                raise ValueError("Cannot add tangents without normals (need 7 columns first)")
            t1, t2 = source_handling.get_tangent_vectors(data[:, 4:7])  # Get tangents from normals
            data = np.hstack((data, t1, t2))  # Add tangents (13 cols)

        tmpdir = tempfile.mkdtemp()
        temp_dirs.append(tmpdir)  # Keep track for cleanup
        filepath = os.path.join(tmpdir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return filepath, tmpdir  # Return path and dir

    yield _create_file

    # Cleanup
    for tmpdir in temp_dirs:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


# Test module import
def test_import():
    """Test if the module can be imported"""
    assert source_handling is not None


# --- Tests for Loading from File ---


def test_load_from_file_basic_3col(dummy_pointcloud_file_factory):
    """Test loading a basic 3-column point cloud file."""
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(30, 3))
    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "perturb_points": False,
        }
    )
    dummy_global_cfg = OmegaConf.create({"use_vector_model": False})  # Add dummy global config
    pts_t, t1_t, t2_t, pts_p, t1_p, t2_p, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert pts_t.shape == (30, 3)
    assert pts_p.shape == (30, 3)
    assert np.array_equal(pts_t, pts_p)  # No perturbation
    assert t1_t is None
    assert t2_t is None
    assert t1_p is None
    assert t2_p is None


def test_load_from_file_7col(dummy_pointcloud_file_factory):
    """Test loading a 7-column file (calculates tangents)."""
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(40, 3), add_normals=True)
    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "perturb_points": False,
        }
    )
    dummy_global_cfg = OmegaConf.create({"use_vector_model": False})  # Add dummy global config
    pts_t, t1_t, t2_t, pts_p, t1_p, t2_p, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert pts_t.shape == (40, 3)
    assert pts_p.shape == (40, 3)
    assert np.array_equal(pts_t, pts_p)
    assert t1_t is not None
    assert t2_t is not None
    assert t1_p is not None
    assert t2_p is not None
    assert t1_t.shape == (40, 3)
    assert t2_t.shape == (40, 3)
    assert np.array_equal(t1_t, t1_p)
    assert np.array_equal(t2_t, t2_p)


def test_load_from_file_13col(dummy_pointcloud_file_factory):
    """Test loading a 13-column file (precalculated tangents)."""
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(50, 3), add_tangents=True)
    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "perturb_points": False,
        }
    )
    dummy_global_cfg = OmegaConf.create(
        {"use_vector_model": True}
    )  # Use vector model true since tangents are expected
    pts_t, t1_t, t2_t, pts_p, t1_p, t2_p, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert pts_t.shape == (50, 3)
    assert pts_p.shape == (50, 3)
    assert np.array_equal(pts_t, pts_p)
    assert t1_t is not None
    assert t2_t is not None
    assert t1_p is not None
    assert t2_p is not None
    assert t1_t.shape == (50, 3)
    assert t2_t.shape == (50, 3)
    assert np.array_equal(t1_t, t1_p)
    assert np.array_equal(t2_t, t2_p)


def test_load_file_not_found():
    """Test error handling for non-existent file."""
    cfg = OmegaConf.create({"use_source_file": True, "source_file_path": "non_existent.pkl"})
    with pytest.raises(FileNotFoundError):
        dummy_global_cfg = OmegaConf.create({"use_vector_model": False})
        source_handling.get_source_pointcloud(cfg, dummy_global_cfg, base_dir=".")


# --- Tests for Generating Test Cloud ---


@patch("src.source_handling.create_test_pointcloud")
@patch("src.source_handling.get_cube_normals")
@patch("src.source_handling.get_tangent_vectors")
def test_generate_cloud(mock_get_tangents, mock_get_normals, mock_create_cloud):
    """Test the generation path."""
    # Mock return values
    mock_create_cloud.return_value = np.random.rand(60, 3)  # 6 faces * 10 points
    mock_get_normals.return_value = (
        np.random.rand(60, 3),
        np.random.rand(60, 3),
    )  # points, normals
    mock_get_tangents.return_value = (np.random.rand(60, 3), np.random.rand(60, 3))  # t1, t2

    cfg = OmegaConf.create(
        {"use_source_file": False, "room_size": 2.0, "wall_points": 10, "perturb_points": False}
    )
    dummy_global_cfg = OmegaConf.create(
        {"use_vector_model": True}
    )  # Assume vector model for generated tangents
    pts_t, t1_t, t2_t, pts_p, t1_p, t2_p, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir="."
    )  # Add global_cfg and unpack currents

    mock_create_cloud.assert_called_once()
    mock_get_normals.assert_called_once()
    mock_get_tangents.assert_called_once()

    assert pts_t.shape == (60, 3)
    assert pts_p.shape == (60, 3)
    assert np.array_equal(pts_t, pts_p)
    assert t1_t is not None
    assert t2_t is not None
    assert t1_p is not None
    assert t2_p is not None
    assert np.array_equal(t1_t, t1_p)
    assert np.array_equal(t2_t, t2_p)


# --- Tests for Filters and Perturbation ---


def test_downsampling(dummy_pointcloud_file_factory):
    """Test point cloud downsampling."""
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(100, 3), add_tangents=True)
    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "pointcloud_downsample": 4,  # Keep every 4th point
            "perturb_points": False,
        }
    )
    dummy_global_cfg = OmegaConf.create({"use_vector_model": True})  # Tangents exist in file
    pts_t, t1_t, t2_t, pts_p, t1_p, t2_p, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert len(pts_t) == 100 // 4
    assert len(t1_t) == 100 // 4
    assert len(t2_t) == 100 // 4
    assert len(pts_p) == 100 // 4


def test_max_distance_filter(dummy_pointcloud_file_factory):
    """Test filtering points based on distance from origin."""
    # Create points where roughly half should be filtered
    points = np.vstack([np.random.rand(50, 3) * 5, np.random.rand(50, 3) * 15])  # 50 near, 50 far
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(100, 3))
    # Overwrite the dummy file with specific points
    with open(filepath, "wb") as f:
        pickle.dump(points, f)

    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "max_distance_from_origin": 10.0,  # Keep points within 10m
            "perturb_points": False,
        }
    )
    dummy_global_cfg = OmegaConf.create({"use_vector_model": False})
    pts_t, _, _, pts_p, _, _, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert len(pts_t) < 100
    assert len(pts_p) == len(pts_t)
    distances = np.sqrt(np.sum(pts_t**2, axis=1))
    assert np.all(distances <= 10.0)


def test_perturbation(dummy_pointcloud_file_factory):
    """Test if perturbation modifies points."""
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(50, 3))
    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "perturb_points": True,
            "perturbation_factor": 0.1,  # Non-zero factor
        }
    )
    dummy_global_cfg = OmegaConf.create({"use_vector_model": False})
    pts_t, _, _, pts_p, _, _, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert pts_t.shape == pts_p.shape
    # Check that perturbed points are NOT identical to true points
    assert not np.array_equal(pts_t, pts_p)
    # Check that the difference is within expected bounds (roughly)
    diff = np.linalg.norm(pts_t - pts_p, axis=1)
    distances = np.sqrt(np.sum(pts_t**2, axis=1))
    # Perturbation is scaled by distance, so relative difference might be more stable
    relative_diff = diff / distances
    # Allow for some variation due to random distribution
    assert np.mean(relative_diff) < 0.2  # Should be roughly around perturbation_factor


def test_perturbation_zero_factor(dummy_pointcloud_file_factory):
    """Test perturbation with zero factor."""
    filepath, tmpdir = dummy_pointcloud_file_factory(shape=(50, 3))
    cfg = OmegaConf.create(
        {
            "use_source_file": True,
            "source_file_path": os.path.basename(filepath),
            "perturb_points": True,
            "perturbation_factor": 0.0,  # Zero factor
        }
    )
    dummy_global_cfg = OmegaConf.create({"use_vector_model": False})
    pts_t, _, _, pts_p, _, _, _ = source_handling.get_source_pointcloud(
        cfg, dummy_global_cfg, base_dir=tmpdir
    )  # Add global_cfg and unpack currents

    assert pts_t.shape == pts_p.shape
    # Points should be identical if factor is zero
    assert np.array_equal(pts_t, pts_p)
