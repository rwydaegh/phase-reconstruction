import os
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use a non-interactive backend for testing to avoid showing plots
matplotlib.use("Agg")

from src.visualization.utils import visualize_field, visualize_point_cloud


@pytest.fixture
def vis_data():
    """Provides sample data for visualization tests."""
    np.random.seed(42)
    resolution = 4  # Small resolution for testing
    field_size = resolution * resolution
    field = np.random.rand(field_size) + 1j * np.random.rand(field_size)
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)

    num_points = 10
    points = np.random.rand(num_points, 3) * 2.0  # Room size 2.0
    currents = np.random.rand(num_points) + 1j * np.random.rand(num_points)
    highlight_indices = [1, 3, 5]

    # Simple XY measurement plane
    mp_res = 3
    mp_x = np.linspace(0.2, 0.8, mp_res)
    mp_y = np.linspace(0.2, 0.8, mp_res)
    MP_X, MP_Y = np.meshgrid(mp_x, mp_y)
    measurement_plane = np.stack([MP_X, MP_Y, np.ones_like(MP_X) * 1.0], axis=-1)  # Plane at z=1.0

    return {
        "field": field,
        "x": x,
        "y": y,
        "resolution": resolution,
        "points": points,
        "currents": currents,
        "highlight_indices": highlight_indices,
        "measurement_plane": measurement_plane,
        "room_size": 2.0,
    }


# Test functions will be added below

# --- Tests for visualize_field ---

# def test_visualize_field_runs(vis_data):
#     """Test if visualize_field executes without errors."""
#     try:
#         _ = visualize_field(
#             vis_data["field"],
#             vis_data["x"],
#             vis_data["y"],
#             title="Test Field",
#             show=False # Don't show plot during test
#         )
#         plt.close('all') # Ensure figures are closed
#     except Exception as e:
#         pytest.fail(f"visualize_field raised an exception: {e}")
#
# def test_visualize_field_saves_file(vis_data):
#     """Test if visualize_field saves a file when filename is provided."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         filename = Path(tmpdir) / "test_field.png"
#         _ = visualize_field(
#             vis_data["field"],
#             vis_data["x"],
#             vis_data["y"],
#             title="Test Save Field",
#             filename=str(filename),
#             show=False
#         )
#         assert filename.exists(), "Output file was not created."
#         assert filename.stat().st_size > 0, "Output file is empty."
#         plt.close('all')
#
# def test_visualize_field_returns_magnitude(vis_data):
#     """Test the return value of visualize_field."""
#     field_2d = visualize_field(
#         vis_data["field"],
#         vis_data["x"],
#         vis_data["y"],
#         title="Test Return",
#         show=False
#     )
#     assert isinstance(field_2d, np.ndarray), "Return value should be a numpy array."
#     expected_shape = (vis_data["resolution"], vis_data["resolution"])
#     assert field_2d.shape == expected_shape, \
#         f"Expected return shape {expected_shape}, got {field_2d.shape}"
#     assert np.allclose(field_2d, np.abs(vis_data["field"]).reshape(expected_shape)), \
#         "Returned array does not match the magnitude of the input field."
#     plt.close('all')
#
# # --- Tests for visualize_point_cloud ---
#
# def test_visualize_point_cloud_runs_basic(vis_data):
#     """Test if visualize_point_cloud runs without currents."""
#     try:
#         visualize_point_cloud(
#             vis_data["points"],
#             title="Test PC Basic",
#             show=False,
#             room_size=vis_data["room_size"]
#         )
#         plt.close('all')
#     except Exception as e:
#         pytest.fail(f"visualize_point_cloud (basic) raised an exception: {e}")
#
# def test_visualize_point_cloud_runs_with_currents(vis_data):
#     """Test if visualize_point_cloud runs with currents."""
#     try:
#         visualize_point_cloud(
#             vis_data["points"],
#             currents=vis_data["currents"],
#             title="Test PC Currents",
#             show=False,
#             room_size=vis_data["room_size"]
#         )
#         plt.close('all')
#     except Exception as e:
#         pytest.fail(f"visualize_point_cloud (with currents) raised an exception: {e}")
#
# def test_visualize_point_cloud_runs_with_highlight(vis_data):
#     """Test if visualize_point_cloud runs with highlighting."""
#     try:
#         visualize_point_cloud(
#             vis_data["points"],
#             currents=vis_data["currents"],
#             title="Test PC Highlight",
#             highlight_indices=vis_data["highlight_indices"],
#             show=False,
#             room_size=vis_data["room_size"]
#         )
#         plt.close('all')
#     except Exception as e:
#         pytest.fail(f"visualize_point_cloud (with highlight) raised an exception: {e}")
#
# def test_visualize_point_cloud_runs_with_measurement_plane(vis_data):
#     """Test if visualize_point_cloud runs with measurement plane."""
#     try:
#         visualize_point_cloud(
#             vis_data["points"],
#             title="Test PC Measurement Plane",
#             measurement_plane=vis_data["measurement_plane"],
#             show=False,
#             room_size=vis_data["room_size"]
#         )
#         plt.close('all')
#     except Exception as e:
#         pytest.fail(f"visualize_point_cloud (with measurement plane) raised an exception: {e}")
#
# def test_visualize_point_cloud_saves_file(vis_data):
#     """Test if visualize_point_cloud saves a file."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         filename = Path(tmpdir) / "test_pc.png"
#         visualize_point_cloud(
#             vis_data["points"],
#             currents=vis_data["currents"],
#             title="Test Save PC",
#             filename=str(filename),
#             show=False,
#             room_size=vis_data["room_size"]
#         )
#         assert filename.exists(), "Output file was not created."
#         assert filename.stat().st_size > 0, "Output file is empty."
#         plt.close('all')
