import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional, Callable
import sys
import os

# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulation_config_real_data import SimulationConfig
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define parameter variation ranges
from src.config_types import BaseConfig

@dataclass
class ParameterRange(BaseConfig):
    """Range of values for a parameter in the sensitivity analysis"""
    param_name: str
    start: float
    end: float
    num_steps: int = 5
    log_scale: bool = False

    def __post_init__(self):
        super().__post_init__()  # Call parent's type casting
        # Additional post-init processing for integer parameters
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
    base_config: SimulationConfig = field(default_factory=SimulationConfig)
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    output_dir: str = "sensitivity_results"
    parallel: bool = True
    max_workers: int = 4
    
    def get_parameter_combinations(self, param1_idx: int, param2_idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
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
                # Create a copy of the base config
                config_dict = asdict(self.base_config)
                
                # Update with the specific parameter values
                config_dict[param1.param_name] = val1
                config_dict[param2.param_name] = val2
                
                # Add indices for reshaping results later
                config_dict['grid_i'] = int(i)
                config_dict['grid_j'] = int(j)
                
                configs.append(config_dict)
        
        return X, Y, configs


def run_simulation(config_dict: Dict[Any, Any]) -> Dict[str, Any]:
    """Run a single simulation with the given configuration"""
    # Extract grid indices
    grid_i = config_dict.pop('grid_i')
    grid_j = config_dict.pop('grid_j')
    
    # Disable plot display for batch runs
    config_dict['show_plot'] = False
    
    # Create config dataclass
    config = SimulationConfig(**config_dict)
    
    try:
        # Capture start time
        start_time = time.time()
        
        # Run simulation - modified to return metrics
        rmse, corr = run_simulation_and_get_metrics(config)
        
        # Measure elapsed time
        elapsed_time = time.time() - start_time
        
        return {
            'grid_i': grid_i,
            'grid_j': grid_j,
            'config': config_dict,
            'rmse': rmse,
            'corr': corr,
            'elapsed_time': elapsed_time,
            'success': True
        }
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise e


def run_simulation_and_get_metrics(config: SimulationConfig) -> Tuple[float, float]:
    """Run simulation and return metrics without visualization"""
    # Import from parent directory
    from src.create_test_pointcloud import create_test_pointcloud
    from src.utils.field_utils import create_channel_matrix, compute_fields, reconstruct_field
    from src.holographic_phase_retrieval import holographic_phase_retrieval
    import random
    
    # Create test environment
    points = create_test_pointcloud(config)
    
    # Create currents (similar to main.py)
    currents = np.zeros(len(points), dtype=complex)
    num_sources = min(config.num_sources, len(currents))
    source_indices = random.sample(range(len(currents)), int(num_sources))
    
    # Use the same random seed for all simulations to ensure fair comparison
    np.random.seed(42)
    
    # Generate log-normal amplitudes
    amplitudes = np.random.lognormal(mean=0, sigma=3, size=int(num_sources))
    
    # Generate random phases between 0 and 2π
    phases = np.random.uniform(0, 2*np.pi, size=int(num_sources))
    
    # Set complex currents with amplitude and phase
    for i, idx in enumerate(source_indices):
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])
    else:
        # Fallback: use first point
        currents[0] = 1.0
    
    # Create measurement plane
    resolution = int(config.resolution)
    x = np.linspace(config.room_size/2 - config.plane_size/2, config.room_size/2 + config.plane_size/2, resolution)
    y = np.linspace(config.room_size/2 - config.plane_size/2, config.room_size/2 + config.plane_size/2, resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.ones_like(X) * config.room_size / 2], axis=-1)
    
    # Create channel matrix for scalar fields
    H = create_channel_matrix(points, measurement_plane, config.k)
    
    # Calculate ground truth field on measurement plane
    true_field = H @ currents
    
    # Get field magnitude (what we would measure)
    measured_magnitude = np.abs(true_field)
    
    # Phase retrieval
    cluster_coefficients = holographic_phase_retrieval(
        H, measured_magnitude,
        adaptive_regularization=config.adaptive_regularization,
        num_iterations=config.gs_iterations,
        convergence_threshold=config.convergence_threshold,
        regularization=1e-3,
        return_history=False,
        debug=False
    )
    
    # Reconstruct field using estimated coefficients
    reconstructed_field = reconstruct_field(H, cluster_coefficients)
    
    # Calculate metrics
    def normalized_rmse(a, b):
        return np.sqrt(np.mean((a - b)**2)) / (np.max(a) - np.min(a))
        
    def normalized_correlation(a, b):
        a_norm = (a - np.mean(a)) / np.std(a)
        b_norm = (b - np.mean(b)) / np.std(b)
        return np.correlate(a_norm.flatten(), b_norm.flatten())[0] / len(a_norm.flatten())
    
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
        
        logger.info(f"Analyzing sensitivity to {param1.param_name} vs {param2.param_name} ({pair_idx+1}/{len(param_pairs)})")
        
        # Get parameter combinations for this pair
        X, Y, config_dicts = analysis_config.get_parameter_combinations(param1_idx, param2_idx)
        
        # Run simulations
        results = []
        
        if analysis_config.parallel and len(config_dicts) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=analysis_config.max_workers) as executor:
                future_to_config = {executor.submit(run_simulation, config_dict): config_dict for config_dict in config_dicts}
                
                for i, future in enumerate(as_completed(future_to_config)):
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed simulation {i+1}/{len(config_dicts)}: RMSE={result.get('rmse', np.nan):.4f}, Time={result.get('elapsed_time', 0):.2f}s")
        else:
            # Sequential execution
            for i, config_dict in enumerate(config_dicts):
                result = run_simulation(config_dict)
                results.append(result)
                logger.info(f"Completed simulation {i+1}/{len(config_dicts)}: RMSE={result.get('rmse', np.nan):.4f}, Time={result.get('elapsed_time', 0):.2f}s")
        
        # Process results
        rmse_grid = np.full_like(X, np.nan)
        corr_grid = np.full_like(X, np.nan)
        time_grid = np.full_like(X, np.nan)
        
        for result in results:
            if result['success']:
                i, j = result['grid_i'], result['grid_j']
                rmse_grid[j, i] = result['rmse']  # Note: j, i order for correct orientation
                corr_grid[j, i] = result['corr']
                time_grid[j, i] = result['elapsed_time']
        
        # Plot RMSE heatmap
        plt.figure(figsize=(10, 8))
        
        # Use log scale for the colormap if values span multiple orders of magnitude
        vmin, vmax = np.nanmin(rmse_grid), np.nanmax(rmse_grid)
        if vmax / max(vmin, 1e-10) > 100:  # If range spans more than 2 orders of magnitude
            norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
            plt.pcolormesh(X, Y, rmse_grid, norm=norm, cmap='viridis')
        else:
            plt.pcolormesh(X, Y, rmse_grid, cmap='viridis')
        
        plt.colorbar(label='Normalized RMSE')
        plt.xlabel(param1.param_name)
        plt.ylabel(param2.param_name)
        plt.title(f'Sensitivity Analysis: Impact on Reconstruction Error')
        
        # Use log scale for axes if specified
        if param1.log_scale:
            plt.xscale('log')
        if param2.log_scale:
            plt.yscale('log')
        
        # Save figure
        plt.tight_layout()
        filename = f"{analysis_config.output_dir}/sensitivity_{param1.param_name}_vs_{param2.param_name}_rmse.png"
        plt.savefig(filename)
        plt.close()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, Y, corr_grid, cmap='plasma')
        plt.colorbar(label='Correlation')
        plt.xlabel(param1.param_name)
        plt.ylabel(param2.param_name)
        plt.title(f'Sensitivity Analysis: Impact on Reconstruction Correlation')
        
        # Use log scale for axes if specified
        if param1.log_scale:
            plt.xscale('log')
        if param2.log_scale:
            plt.yscale('log')
        
        # Save figure
        plt.tight_layout()
        filename = f"{analysis_config.output_dir}/sensitivity_{param1.param_name}_vs_{param2.param_name}_corr.png"
        plt.savefig(filename)
        plt.close()
        
        # Plot computation time heatmap
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, Y, time_grid, cmap='inferno')
        plt.colorbar(label='Computation Time (s)')
        plt.xlabel(param1.param_name)
        plt.ylabel(param2.param_name)
        plt.title(f'Sensitivity Analysis: Computation Time')
        
        # Use log scale for axes if specified
        if param1.log_scale:
            plt.xscale('log')
        if param2.log_scale:
            plt.yscale('log')
        
        # Save figure
        plt.tight_layout()
        filename = f"{analysis_config.output_dir}/sensitivity_{param1.param_name}_vs_{param2.param_name}_time.png"
        plt.savefig(filename)
        plt.close()
        
        # Save raw data as numpy arrays for future analysis
        np.savez(
            f"{analysis_config.output_dir}/sensitivity_{param1.param_name}_vs_{param2.param_name}_data.npz",
            X=X, Y=Y, 
            rmse=rmse_grid, 
            correlation=corr_grid, 
            time=time_grid,
            param1_name=param1.param_name,
            param2_name=param2.param_name
        )


if __name__ == "__main__":
    # Create default base configuration
    base_config = SimulationConfig(
        wavelength=10.7e-3,  # 28GHz wavelength in meters
        plane_size=1.0,      # 1m x 1m measurement plane
        resolution=30,       # Reduced for faster execution 
        room_size=2.0,       # 2m x 2m x 2m room
        wall_points=6,       # Points per wall edge
        num_sources=50,      # Number of sources to randomly select
        gs_iterations=100,   # Reduced for faster execution
        convergence_threshold=1e-3,
        show_plot=False,     # Disable plotting for batch runs
        return_history=False # Disable history for faster execution
    )
    
    # Define parameter ranges to analyze
    parameter_ranges = [
        # Wall discretization
        ParameterRange(
            param_name="wall_points",
            start=4,
            end=12,
            num_steps=5,
            log_scale=False
        ),
        # Number of sources
        ParameterRange(
            param_name="num_sources",
            start=10,
            end=200,
            num_steps=5,
            log_scale=True
        ),
        # Measurement plane resolution
        ParameterRange(
            param_name="resolution",
            start=10,
            end=60,
            num_steps=4,
            log_scale=False
        ),
        # GS iterations
        ParameterRange(
            param_name="gs_iterations",
            start=50,
            end=300,
            num_steps=4,
            log_scale=False
        ),
        # Convergence threshold
        ParameterRange(
            param_name="convergence_threshold",
            start=1e-4,
            end=1e-2,
            num_steps=4,
            log_scale=True
        )
    ]
    
    # Create sensitivity analysis configuration
    analysis_config = SensitivityAnalysisConfig(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        output_dir="sensitivity_results",
        parallel=True,
        max_workers=4  # Adjust based on available cores
    )
    
    # Run sensitivity analysis
    run_sensitivity_analysis(analysis_config)
