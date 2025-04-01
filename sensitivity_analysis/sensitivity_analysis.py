# Standard library imports
import copy
import itertools
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
from omegaconf import DictConfig, OmegaConf # Import DictConfig and OmegaConf


# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports
from omegaconf import DictConfig # Import DictConfig for type hinting

# Import necessary functions from src
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_test_pointcloud import create_test_pointcloud
from src.utils.field_utils import create_channel_matrix, reconstruct_field
from src.utils.normalized_correlation import normalized_correlation
from src.utils.normalized_rmse import normalized_rmse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParameterRange:
    """Range of values for a parameter in the sensitivity analysis"""

    param_name: str
    start: float
    end: float
    num_steps: int = 5
    log_scale: bool = False

    def __post_init__(self):
        # Post-init processing for integer parameters
        if self.param_name in ["wall_points", "resolution", "num_sources", "gs_iterations"]:
            self.start = int(float(self.start))
            self.end = int(float(self.end))

    def get_values(self) -> np.ndarray:
        """Get array of parameter values based on range settings"""
        if self.log_scale:
            return np.logspace(np.log10(self.start), np.log10(self.end), self.num_steps)
        else:
            return np.linspace(self.start, self.end, self.num_steps)


@dataclass
class SensitivityAnalysisConfig:
    """Configuration for sensitivity analysis"""

    base_config: DictConfig
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    output_dir: str = "sensitivity_results"
    parallel: bool = True
    max_workers: int = 4

    def get_parameter_combinations(
        self, param1_idx: int, param2_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Get grid of parameter combinations for a pair of parameters"""
        param1 = self.parameter_ranges[param1_idx]
        param2 = self.parameter_ranges[param2_idx]

        values1 = param1.get_values()
        values2 = param2.get_values()

        # Create meshgrid for visualization
        X, Y = np.meshgrid(values1, values2)

        # Create list of parameter combinations
        configs = []
        for i, val1 in enumerate(values1):
            for j, val2 in enumerate(values2):
                # Create a copy and convert base config to dictionary
                config_dict = OmegaConf.to_container(self.base_config, resolve=True)

                # Update with the specific parameter values, ensuring type consistency if needed
                # Update with specific parameter values, ensuring type consistency
                param1_val = val1
                param2_val = val2
                if param1.param_name in ["wall_points", "resolution", "num_sources", "gs_iterations"]:
                    param1_val = int(val1)
                if param2.param_name in ["wall_points", "resolution", "num_sources", "gs_iterations"]:
                    param2_val = int(val2)

                config_dict[param1.param_name] = param1_val
                config_dict[param2.param_name] = param2_val

                # Add grid indices
                config_dict["grid_i"] = int(i)
                config_dict["grid_j"] = int(j)

                configs.append(config_dict)

        return X, Y, configs


def run_simulation(config_dict: Dict[Any, Any]) -> Dict[str, Any]:
    """Run a single simulation with the given configuration"""
    # Extract grid indices
    grid_i = config_dict.pop("grid_i")
    grid_j = config_dict.pop("grid_j")

    # Disable plot display for batch runs
    config_dict["show_plot"] = False

    # Pass the config dictionary directly, assuming the called function handles it.
    # If holographic_phase_retrieval strictly needs DictConfig, convert here.
    config = config_dict # Pass the dictionary directly for now

    try:
        start_time = time.time()

        # Run simulation
        rmse, corr = run_simulation_and_get_metrics(config)

        elapsed_time = time.time() - start_time

        return {
            "grid_i": grid_i,
            "grid_j": grid_j,
            "config": config_dict,
            "rmse": rmse,
            "corr": corr,
            "elapsed_time": elapsed_time,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise e


def run_simulation_and_get_metrics(config: Dict) -> Tuple[float, float]:
    """Run simulation and return metrics without visualization"""
    import random
    from omegaconf import OmegaConf # Needed to access DictConfig attributes easily

    # Convert dict to DictConfig for easier attribute access
    cfg = OmegaConf.create(config)

    # Create test environment
    points = create_test_pointcloud(cfg) # Pass DictConfig

    # Create currents (similar to main.py)
    currents = np.zeros(len(points), dtype=complex)
    num_sources = min(cfg.num_sources, len(currents))
    source_indices = random.sample(range(len(currents)), int(num_sources))

    # Use the same random seed for all simulations to ensure fair comparison
    np.random.seed(42)

    # Generate log-normal amplitudes
    amplitude_sigma = cfg.get("amplitude_sigma", 3.0)
    amplitudes = np.random.lognormal(mean=0, sigma=amplitude_sigma, size=int(num_sources))

    # Generate random phases between 0 and 2π
    phases = np.random.uniform(0, 2 * np.pi, size=int(num_sources))

    # Set complex currents with amplitude and phase
    for i, idx in enumerate(source_indices):
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])
    else:
        # Fallback: use first point
        currents[0] = 1.0

    # Create measurement plane (Ensure keys exist in cfg)
    resolution = int(cfg.resolution)
    room_size = cfg.room_size
    plane_size = cfg.plane_size
    x = np.linspace(
        room_size / 2 - plane_size / 2,
        room_size / 2 + plane_size / 2,
        resolution,
    )
    y = np.linspace(
        room_size / 2 - plane_size / 2,
        room_size / 2 + plane_size / 2,
        resolution,
    )
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.ones_like(X) * room_size / 2], axis=-1)

    # Create channel matrix
    k = cfg.get("k", 2 * np.pi / cfg.wavelength)
    H = create_channel_matrix(points, measurement_plane, k)

    # Calculate ground truth field on measurement plane
    true_field = H @ currents

    # Get field magnitude (what we would measure)
    measured_magnitude = np.abs(true_field)

    # Phase retrieval
    hpr_result = holographic_phase_retrieval(
        cfg,
        H,
        measured_magnitude,
    )

    # Handle return value (might be tuple)
    if isinstance(hpr_result, tuple):
        cluster_coefficients = hpr_result[0]
    else:
        cluster_coefficients = hpr_result

    # Reconstruct field using estimated coefficients
    reconstructed_field = reconstruct_field(H, cluster_coefficients)

    # Calculate metrics

    # Calculate reconstruction quality metrics
    rmse = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))

    return rmse, corr


def run_sensitivity_analysis(analysis_config: SensitivityAnalysisConfig):
    """Run sensitivity analysis for all parameter pairs"""
    # Ensure output directory exists
    os.makedirs(analysis_config.output_dir, exist_ok=True)

    # Get all parameter pair combinations
    param_pairs = list(itertools.combinations(range(len(analysis_config.parameter_ranges)), 2))

    logger.info(f"Running sensitivity analysis for {len(param_pairs)} parameter pairs")

    for pair_idx, (param1_idx, param2_idx) in enumerate(param_pairs):
        param1 = analysis_config.parameter_ranges[param1_idx]
        param2 = analysis_config.parameter_ranges[param2_idx]

        logger.info(
            f"Analyzing sensitivity to {param1.param_name} vs {param2.param_name} "
            f"({pair_idx+1}/{len(param_pairs)})"
        )

        # Get parameter combinations for this pair
        X, Y, config_dicts = analysis_config.get_parameter_combinations(param1_idx, param2_idx)

        # Run simulations
        results = []

        if analysis_config.parallel and len(config_dicts) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=analysis_config.max_workers) as executor:
                future_to_config = {
                    executor.submit(run_simulation, config_dict): config_dict
                    for config_dict in config_dicts
                }

                for i, future in enumerate(as_completed(future_to_config)):
                    result = future.result()
                    results.append(result)
                    # Optional: Log progress less verbosely or based on a flag
                    # logger.debug(f"Completed simulation {i+1}/{len(config_dicts)}")
        else:
            # Sequential execution
            for i, config_dict in enumerate(config_dicts):
                result = run_simulation(config_dict)
                results.append(result)
                # Optional: Log progress less verbosely or based on a flag
                # logger.debug(f"Completed simulation {i+1}/{len(config_dicts)}")

        # Process results
        rmse_grid = np.full_like(X, np.nan)
        corr_grid = np.full_like(X, np.nan)
        time_grid = np.full_like(X, np.nan)

        for result in results:
            if result["success"]:
                i, j = result["grid_i"], result["grid_j"]
                rmse_grid[j, i] = result["rmse"]
                corr_grid[j, i] = result["corr"]
                time_grid[j, i] = result["elapsed_time"]

        # Plotting is handled by the calling script (run_sensitivity_analysis.py)

        # Save raw data as numpy arrays for future analysis
        data_filename = os.path.join(
             analysis_config.output_dir,
             f"sensitivity_{param1.param_name}_vs_{param2.param_name}_data.npz"
        )
        np.savez(
            data_filename,
            X=X,
            Y=Y,
            rmse=rmse_grid,
            correlation=corr_grid,
            time=time_grid,
            param1_name=param1.param_name,
            param2_name=param2.param_name,
        )
        logger.info(f"Saved raw data to {data_filename}")


