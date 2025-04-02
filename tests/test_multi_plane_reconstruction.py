# tests/test_multi_plane_reconstruction.py
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig


# Basic test to ensure the script can be imported and the main function exists
def test_import():
    try:
        from multi_plane_reconstruction import main
    except ImportError as e:
        pytest.fail(f"Failed to import multi_plane_reconstruction: {e}")


# Test if Hydra configuration loads (using the new config)
# This requires the config file to exist at the expected path relative to this test file
# Adjust the relative path '../conf' if necessary based on your test execution context
def test_hydra_config_load():
    try:
        with initialize(config_path="../conf", version_base="1.2"):
            cfg = compose(config_name="multi_plane_config")
            assert cfg is not None
            assert isinstance(cfg, DictConfig)
            assert "global_params" in cfg
            assert "measurement_planes" in cfg
            assert "legacy_mode" in cfg
            assert cfg.legacy_mode is False  # Default check
    except Exception as e:
        pytest.fail(f"Hydra initialization or config composition failed: {e}")


# --- Integration Tests ---


@pytest.mark.integration  # Mark as integration test
def test_run_sim_train_test_config():
    """Runs the main script with the sim_train_test config."""
    try:
        # Use hydra.experimental.compose API for running within a test
        # It requires the config path relative to the original CWD where pytest is run
        # Assuming pytest runs from the project root:
        # Assuming pytest runs from the project root, '.' should be the correct base_dir
        # The job_name ensures hydra outputs go to a unique test directory if needed
        with initialize(config_path="../conf", version_base="1.2", job_name="test_sim_train_test"):
            cfg = compose(config_name="integration_test_sim_train_test")
            # Import the main function locally to avoid potential global state issues
            from multi_plane_reconstruction import (
                main_logic,  # Import the refactored logic function
            )

            # Run the main function with the composed config
            # This will execute the workflow defined in the config
            # Pass the composed config and the base directory relative to project root
            main_logic(cfg, base_dir_override=".")
            # If main(cfg) completes without raising an exception, the test passes
            # We could add checks here later, e.g., load the output pickle and check results
    except Exception as e:
        pytest.fail(
            f"Running multi_plane_reconstruction with integration_test_sim_train_test config failed: {e}"
        )


# Add more tests here as functionality is implemented
# e.g., test legacy mode triggering, test specific workflow steps with mock data/functions
