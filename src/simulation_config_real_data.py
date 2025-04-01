from dataclasses import dataclass

import numpy as np

from src.config_types import BaseConfig


@dataclass
class SimulationConfig(BaseConfig):
    """Configuration for field reconstruction simulation"""

    wavelength: float = 10.7e-3
    plane_size: float = 1.0
    resolution: int = 149
    room_size: float = 2.0

    use_source_pointcloud: bool = True
    source_pointcloud_path: str = "measurement_data/source_pointcloud.pkl"
    pointcloud_downsample: int = 4
    max_distance_from_origin: float = (
        np.inf
    )
    wall_points: int = 10
    num_sources: int = 100
    perturb_points: bool = True
    perturbation_factor: float = 0.05
    amplitude_sigma: float = 3.0

    gs_iterations: int = 200
    convergence_threshold: float = 1e-3
    regularization: float = 1e-3
    adaptive_regularization: bool = True

    perturbation_mode: str = "none"
    enable_perturbations: bool = (
        False
    )
    stagnation_window: int = 30
    stagnation_threshold: float = 1e-3
    perturbation_intensity: float = 0.8
    constraint_skip_iterations: int = 3
    momentum_factor: float = 0.8
    temperature: float = 5.0

    verbose: bool = False
    return_history: bool = True

    show_plot: bool = True
    no_plot: bool = False
    no_anim: bool = False
    output_file: str = "results.png"

    enable_smoothing: bool = True
    smoothing_radius_mm: float = 10.0

    @property
    def k(self) -> float:
        """Wave number"""
        return 2 * np.pi / self.wavelength
