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
from omegaconf import DictConfig, OmegaConf  # Import DictConfig and OmegaConf

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports

# Import necessary functions from src
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval

# Import the correct vectorized create_channel_matrix and the updated reconstruct_field wrapper
from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud
from src.utils.field_utils import reconstruct_field
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
                if param1.param_name in [
                    "wall_points",
                    "resolution",
                    "num_sources",
                    "gs_iterations",
                ]:
                    param1_val = int(val1)
                if param2.param_name in [
                    "wall_points",
                    "resolution",
                    "num_sources",
                    "gs_iterations",
                ]:
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
    config = config_dict  # Pass the dictionary directly for now

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

    from omegaconf import OmegaConf  # Needed to access DictConfig attributes easily

    # Convert dict to DictConfig for easier attribute access and default handling
    # Ensure use_vector_model exists, default to True if not specified in base config
    if "use_vector_model" not in config:
        config["use_vector_model"] = True  # Default to vector model for sensitivity
    cfg = OmegaConf.create(config)

    # Create test environment
    points = create_test_pointcloud(cfg)  # Pass DictConfig
    N_c = points.shape[0]

    # Initialize tangents and measurement direction
    tangents1 = None
    tangents2 = None
    measurement_direction = None

    if cfg.use_vector_model:
        # Calculate tangents for the test point cloud using geometry utils
        from src.utils.geometry_utils import get_cube_normals

        points, temp_normals = get_cube_normals(points, cfg.room_size)
        N_c = points.shape[0]  # Update N_c after potential filtering
        if N_c == 0:
            raise ValueError("No valid points after normal calculation in sensitivity run.")
        logger.debug(
            f"Calculating tangents for {N_c} generated cube points (using inward normals)."
        )
        from src.utils.preprocess_pointcloud import get_tangent_vectors

        tangents1, tangents2 = get_tangent_vectors(temp_normals)
        logger.debug(f"Generated tangents1 shape: {tangents1.shape}")
        logger.debug(f"Generated tangents2 shape: {tangents2.shape}")

        # Get measurement direction from config
        try:
            measurement_direction = np.array(cfg.measurement_direction, dtype=float)
            if measurement_direction.shape != (3,):
                raise ValueError("Shape must be (3,)")
            norm_meas = np.linalg.norm(measurement_direction)
            if norm_meas < 1e-9:
                raise ValueError("Norm cannot be zero.")
            measurement_direction /= norm_meas
            logger.debug(f"Using measurement direction: {measurement_direction}")
        except Exception as e:
            logger.warning(f"Using default measurement direction [0, 1, 0] due to error: {e}")
            measurement_direction = np.array([0.0, 1.0, 0.0])

    # Initialize currents based on model
    num_sources = min(cfg.num_sources, N_c)  # Number of points to activate
    source_indices = random.sample(range(N_c), int(num_sources))  # Indices of points (0 to N_c-1)
    if cfg.use_vector_model:
        currents = np.zeros(2 * N_c, dtype=complex)
    else:
        currents = np.zeros(N_c, dtype=complex)

    # Use the same random seed for all simulations to ensure fair comparison
    np.random.seed(42)

    # Generate log-normal amplitudes
    amplitude_sigma = cfg.get("amplitude_sigma", 3.0)
    amplitudes = np.random.lognormal(mean=0, sigma=amplitude_sigma, size=int(num_sources))

    # Generate random phases between 0 and 2π
    phases = np.random.uniform(0, 2 * np.pi, size=int(num_sources))

    # Set complex currents
    logger.debug(f"Assigning random currents to {num_sources} source points...")
    for i, point_idx in enumerate(source_indices):
        value = amplitudes[i] * np.exp(1j * phases[i])
        if cfg.use_vector_model:
            current_idx = 2 * point_idx  # Index for the first component
            currents[current_idx] = value
            # currents[current_idx + 1] = 0.0 # Second component remains zero
        else:
            currents[point_idx] = value

    # Ensure at least one source if num_sources was 0 or less
    if num_sources <= 0 and N_c > 0:
        logger.warning("num_sources <= 0, activating the first point/component.")
        currents[0] = 1.0  # Activate first point (scalar) or first component (vector)

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
    k = cfg.get("k", 2 * np.pi / cfg.wavelength)  # Use get for safety if k isn't guaranteed
    H = create_channel_matrix(
        points=points,
        measurement_plane=measurement_plane,
        k=k,
        use_vector_model=cfg.use_vector_model,
        tangents1=tangents1,  # Will be None if scalar
        tangents2=tangents2,  # Will be None if scalar
        measurement_direction=measurement_direction,  # Will be None if scalar
    )

    # Calculate ground truth field on measurement plane
    # Calculate ground truth field
    true_field = H @ currents

    # Get field magnitude (what we would measure)
    measured_magnitude = np.abs(true_field)

    # Phase retrieval
    # Phase retrieval - assume HPR handles H shape internally
    hpr_result = holographic_phase_retrieval(
        cfg=cfg,
        channel_matrix=H,
        measured_magnitude=measured_magnitude,
    )

    # Handle return value (might be tuple)
    if isinstance(hpr_result, tuple):
        cluster_coefficients = hpr_result[0]
    else:
        cluster_coefficients = hpr_result

    # Reconstruct field using estimated coefficients
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

                for _i, future in enumerate(as_completed(future_to_config)):
                    result = future.result()
                    results.append(result)
                    # Optional: Log progress less verbosely or based on a flag
                    # logger.debug(f"Completed simulation {i+1}/{len(config_dicts)}")
        else:
            # Sequential execution
            for _i, config_dict in enumerate(config_dicts):
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
            f"sensitivity_{param1.param_name}_vs_{param2.param_name}_data.npz",
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
