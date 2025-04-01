import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.io import load_measurement_data


@pytest.fixture
def dummy_measurement_file():
    """Creates a temporary dummy pickle file with measurement data."""
    dummy_data = {
        "results": np.random.rand(10, 5).tolist(), # Simulate list format
        "continuous_axis": np.linspace(0, 1, 10),
        "discrete_axis": np.arange(5),
        "points_continuous": np.random.rand(10, 3).tolist(), # Simulate list format
        "points_discrete": np.random.rand(5, 3).tolist(), # Simulate list format
        "frequency": 29e9, # Example frequency
    }

    # Create a temporary file
    # Using NamedTemporaryFile with delete=False and manual cleanup
    # to ensure the file path is accessible after closing it.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    file_path = Path(temp_file.name)
    temp_file.close() # Close the file handle immediately

    # Write the dummy data using pickle
    with open(file_path, "wb") as f:
        pickle.dump(dummy_data, f)

    yield file_path # Provide the path to the test

    # Cleanup: Remove the temporary file after the test runs
    os.remove(file_path)

# Test functions will be added below

def test_load_measurement_data(dummy_measurement_file):
    """Tests loading data from a pickle file and checks structure/types."""
    file_path = dummy_measurement_file
    loaded_data = load_measurement_data(file_path)

    # Check if the output is a dictionary
    assert isinstance(loaded_data, dict), "Output should be a dictionary."

    # Check for expected keys
    expected_keys = [
        "results",
        "continuous_axis",
        "discrete_axis",
        "points_continuous",
        "points_discrete",
        "frequency",
    ]
    assert all(key in loaded_data for key in expected_keys), \
        f"Missing keys in loaded data. Expected {expected_keys}, got {list(loaded_data.keys())}"

    # Check data types and shapes after processing by load_measurement_data
    assert isinstance(loaded_data["results"], np.ndarray), "'results' should be a numpy array."
    assert loaded_data["results"].dtype == float, "'results' array should have float dtype."
    # Original dummy data was (10, 5)
    assert loaded_data["results"].shape == (10, 5), \
        f"Expected 'results' shape (10, 5), got {loaded_data['results'].shape}"

    assert isinstance(loaded_data["continuous_axis"], np.ndarray)
    assert loaded_data["continuous_axis"].shape == (10,)

    assert isinstance(loaded_data["discrete_axis"], np.ndarray)
    assert loaded_data["discrete_axis"].shape == (5,)

    assert isinstance(loaded_data["points_continuous"], np.ndarray)
    assert loaded_data["points_continuous"].shape == (10, 3)

    assert isinstance(loaded_data["points_discrete"], np.ndarray)
    assert loaded_data["points_discrete"].shape == (5, 3)

    # Check frequency value
    assert loaded_data["frequency"] == 29e9, \
        f"Expected frequency 29e9, got {loaded_data['frequency']}"


def test_load_measurement_data_missing_frequency(dummy_measurement_file):
    """Tests loading data where frequency key is missing, expecting default."""
    file_path = dummy_measurement_file

    # Modify the dummy file to remove the frequency key
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    del data["frequency"]
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    # Load the modified data
    loaded_data = load_measurement_data(file_path)

    # Check if the default frequency (28e9) is loaded
    assert loaded_data["frequency"] == 28e9, \
        f"Expected default frequency 28e9 when key is missing, got {loaded_data['frequency']}"


def test_load_measurement_data_uneven_results(dummy_measurement_file):
    """Tests loading data with uneven list lengths in 'results', expecting padding."""
    file_path = dummy_measurement_file

    # Modify the dummy file to have uneven results lists
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # Make results a list of lists with different lengths
    data["results"] = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]]
    max_len = 3
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    # Load the modified data
    loaded_data = load_measurement_data(file_path)

    # Check shape - should be padded to max length (3)
    assert loaded_data["results"].shape == (3, max_len), \
        f"Expected padded shape (3, {max_len}), got {loaded_data['results'].shape}"

    # Check if padding with NaN occurred
    assert np.isnan(loaded_data["results"][0, 2]), "Expected NaN padding in first row."
    assert np.isnan(loaded_data["results"][2, 1]), "Expected NaN padding in third row."
    assert np.isnan(loaded_data["results"][2, 2]), "Expected NaN padding in third row."
    assert not np.isnan(loaded_data["results"][1, 2]), "Expected no NaN padding in second row."
