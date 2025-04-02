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
            assert cfg.legacy_mode is False # Default check
    except Exception as e:
        pytest.fail(f"Hydra initialization or config composition failed: {e}")

# Add more tests here as functionality is implemented
# e.g., test legacy mode triggering, test specific workflow steps with mock data/functions