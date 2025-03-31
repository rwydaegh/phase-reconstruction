from dataclasses import dataclass

import numpy as np

from src.config_types import BaseConfig


@dataclass
class SimulationConfig(BaseConfig):
    """Configuration for field reconstruction simulation"""

    # Physical simulation parameters
    wavelength: float = 10.7e-3  # 28GHz wavelength in meters
    plane_size: float = 1.0  # 1m x 1m measurement plane
    resolution: int = 149  # 50x50 grid
    room_size: float = 2.0  # 2m x 2m x 2m room

    # Point cloud parameters
    use_source_pointcloud: bool = True  # Whether to use the source pointcloud file
    source_pointcloud_path: str = "source_pointcloud.pkl"  # Path to the source pointcloud file
    pointcloud_downsample: int = 4  # Downsample factor for point cloud (1 = no downsampling)
    max_distance_from_origin: float = (
        np.inf
    )  # Maximum distance from origin (in meters) to keep points
    wall_points: int = 10  # Points per wall edge (used if not using source pointcloud)
    num_sources: int = 100  # Number of sources to randomly select
    perturb_points: bool = True  # Enable point cloud perturbation
    perturbation_factor: float = 0.05  # Max perturbation as percentage of distance to origin
    amplitude_sigma: float = 3.0  # Sigma for log-normal amplitude distribution

    # Phase retrieval parameters
    gs_iterations: int = 200  # Maximum number of GS iterations
    convergence_threshold: float = 1e-3  # Convergence threshold
    regularization: float = 1e-3  # Regularization parameter for SVD
    adaptive_regularization: bool = True  # Enable adaptive regularization

    # Perturbation strategy parameters
    perturbation_mode: str = "none"  # "none", "basic", "momentum", or "archived"
    enable_perturbations: bool = (
        False  # Enable perturbation strategies (derived from perturbation_mode != "none")
    )
    stagnation_window: int = 30  # Window to detect stagnation
    stagnation_threshold: float = 1e-3  # Threshold for meaningful improvements
    perturbation_intensity: float = 0.8  # Intensity of perturbations
    constraint_skip_iterations: int = 3  # Skip constraint iterations after perturbation
    momentum_factor: float = 0.8  # For momentum-based perturbation
    temperature: float = 5.0  # For archived complex strategies

    # Execution control
    verbose: bool = False  # Enable verbose output
    return_history: bool = True  # Return the history of cluster coefficients and fields

    # Visualization parameters
    show_plot: bool = True  # Show plots interactively
    no_plot: bool = False  # Disable all plot generation
    no_anim: bool = False  # Disable animation generation
    output_file: str = "results.png"  # Output filename

    # Post-processing parameters
    enable_smoothing: bool = True  # Enable Gaussian smoothing as post-processing
    smoothing_radius_mm: float = 10.0  # Radius in mm for Gaussian smoothing

    @property
    def k(self) -> float:
        """Wave number"""
        return 2 * np.pi / self.wavelength
