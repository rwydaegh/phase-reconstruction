import math

import numpy as np
import pytest

# Import necessary functions
from src.create_channel_matrix import create_channel_matrix
from src.utils.preprocess_pointcloud import get_tangent_vectors

# Define constants for the tests
WAVELENGTH = 10.7e-3
K = 2 * math.pi / WAVELENGTH
SOURCE_DIST = 0.5
CENTER_POINT = np.array([[0.0, 0.0, 0.0]])  # Measurement point as (1, 3) array
POINTS = np.array([[-SOURCE_DIST, 0.0, 0.0], [SOURCE_DIST, 0.0, 0.0]])
NORMALS = np.array(
    [
        [1.0, 0.0, 0.0],  # Inward normal for source 1
        [-1.0, 0.0, 0.0],  # Inward normal for source 2
    ]
)
TANGENTS1, TANGENTS2 = get_tangent_vectors(NORMALS)

# Define test cases: (currents_list, measurement_direction_list, expected_magnitude_is_zero, description)
test_cases = [
    ([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0], True, "t1 in phase, measure Ey"),
    ([1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0], False, "t1 out of phase, measure Ey"),
    ([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0], True, "t2 in phase, measure Ex"),
    ([0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0], True, "t2 out of phase, measure Ex (Expected Zero)"),
    # Add a case that should be non-zero for t2
    ([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0], False, "t2 in phase, measure Ez"),
]


@pytest.mark.parametrize(
    "currents_list, measurement_direction_list, expected_magnitude_is_zero, description", test_cases
)
def test_two_source_center_field(
    currents_list, measurement_direction_list, expected_magnitude_is_zero, description
):
    """
    Tests the calculated field at the center point (0,0,0) for various
    two-source configurations and measurement directions.
    """
    currents = np.array(currents_list, dtype=complex)
    measurement_direction = np.array(measurement_direction_list, dtype=float)

    # Calculate channel matrix for the single center point
    H = create_channel_matrix(
        points=POINTS,
        tangents1=TANGENTS1,
        tangents2=TANGENTS2,
        measurement_plane=CENTER_POINT,  # Pass the single point
        measurement_direction=measurement_direction,
        k=K,
    )
    assert H.shape == (1, 4), f"Test setup error: H shape is {H.shape}, expected (1, 4)"

    # Calculate field at the center point: y = H @ x
    field_value = (H @ currents)[0]  # Result is shape (1,), get the scalar

    # Assert based on expected outcome
    magnitude = np.abs(field_value)
    tolerance = 1e-9

    if expected_magnitude_is_zero:
        assert np.isclose(
            magnitude, 0.0, atol=tolerance
        ), f"Test Failed ({description}): Expected zero magnitude, got {magnitude:.3e}"
    else:
        assert (
            magnitude > tolerance
        ), f"Test Failed ({description}): Expected non-zero magnitude, got {magnitude:.3e}"

    print(
        f"Test Passed ({description}): Magnitude = {magnitude:.3e}"
    )  # Optional: print for confirmation
