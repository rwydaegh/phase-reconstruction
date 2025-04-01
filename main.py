import hydra
from omegaconf import DictConfig, OmegaConf

# import argparse # Replaced by Hydra
import logging
import pickle
import random
# from dataclasses import asdict, dataclass, fields # No longer needed directly here

import numpy as np

from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
# from src.simulation_config_fake_data import SimulationConfig # Replaced by Hydra config
from src.utils.field_utils import compute_fields, reconstruct_field
# from src.visualization import visualize_current_and_field_history # Moved
from src.visualization.history_plots import visualize_current_and_field_history # Import moved function
from src.visualization.history_plots import visualize_iteration_history # Import moved function
from src.visualization.field_plots import visualize_fields

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="fake_data", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main demonstration of field reconstruction using holographic phase retrieval."""

    # Create or load point cloud
    # Log the configuration being used (Hydra handles saving it automatically)
    logger.info(OmegaConf.to_yaml(cfg))

    # Create or load point cloud
    if cfg.use_source_pointcloud:
        logger.info(f"Loading source point cloud from {cfg.source_pointcloud_path}...")
        # Ensure the path is resolved correctly relative to the original CWD if needed
        # For now, assume it's relative to the project root or absolute
        with open(cfg.source_pointcloud_path, "rb") as f:
            data = pickle.load(f)

        # Source is a (5328,4) array with last column being distance
        # Just take the first 3 columns (x, y, z coordinates)
        points = data[:, :3]
        logger.info(f"Loaded point cloud with {len(points)} points, shape: {points.shape}")
    else:
        logger.info("Creating test point cloud...")
        points = create_test_pointcloud(cfg) # Pass DictConfig, assuming compatibility

    logger.info(f"Using point cloud shape: {points.shape}")

    # Create simple current distribution (scalar amplitudes for each point)
    # For scalar field, we just need one value per point
    currents = np.zeros(len(points), dtype=complex)

    # Set random indices in the currents array to be active
    # with log-normal amplitude distribution and random phases
    num_sources = min(cfg.num_sources, len(currents))
    source_indices = random.sample(range(len(currents)), num_sources)

    # Generate log-normal amplitudes
    amplitudes = np.random.lognormal(mean=0, sigma=cfg.amplitude_sigma, size=num_sources)

    # Generate random phases between 0 and 2π
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    # Set complex currents with amplitude and phase
    for i, idx in enumerate(source_indices):
        # Convert amplitude and phase to complex number: A * e^(iθ)
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])
    else:
        # Fallback: use first point
        currents[0] = 1.0

    # Create measurement plane centered at (0,0,0)
    logger.info("Creating measurement plane...")
    x = np.linspace(-cfg.plane_size / 2, cfg.plane_size / 2, cfg.resolution)
    y = np.linspace(-cfg.plane_size / 2, cfg.plane_size / 2, cfg.resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    # Calculate k from wavelength
    k = 2 * np.pi / cfg.wavelength

    # Create channel matrix for scalar fields
    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, measurement_plane, k)

    # Calculate ground truth field on measurement plane
    logger.info("Calculating ground truth field...")
    true_field = compute_fields(points, currents, measurement_plane, k, H)

    # Reshape to 2D grid
    true_field_2d = true_field.reshape(cfg.resolution, cfg.resolution)

    # Get field magnitude (what we would measure)
    measured_magnitude = np.abs(true_field) # Use magnitude from synthetic true field
    measured_magnitude_2d = measured_magnitude.reshape(cfg.resolution, cfg.resolution)

    # Phase retrieval demonstration
    logger.info("Running holographic phase retrieval with perturbation strategies...")

    hpr_result = holographic_phase_retrieval(
        H,
        measured_magnitude,
        adaptive_regularization=cfg.adaptive_regularization,
        num_iterations=cfg.gs_iterations,
        convergence_threshold=cfg.convergence_threshold,
        regularization=cfg.regularization,
        return_history=cfg.return_history,
        # Use parameters from cfg
        enable_perturbations=cfg.enable_perturbations,
        stagnation_window=cfg.stagnation_window,
        stagnation_threshold=cfg.stagnation_threshold,
        perturbation_intensity=cfg.perturbation_intensity,
        perturbation_mode=cfg.perturbation_mode,
        constraint_skip_iterations=cfg.constraint_skip_iterations,
        momentum_factor=cfg.momentum_factor,
        temperature=cfg.temperature,
        verbose=cfg.verbose,
        no_plot=cfg.no_plot,
    )

    if cfg.return_history:
        # Handle the return signature with history and stats
        cluster_coefficients, coefficient_history, field_history, stats = hpr_result
        logger.info(f"Coefficient history shape: {coefficient_history.shape}")
        logger.info(f"Field history shape: {field_history.shape}")

        # Log the perturbation statistics
        logger.info("Perturbation statistics:")
        logger.info(f"  Total iterations: {stats['iterations']}")
        logger.info(f"  Final error: {stats['final_error']:.6f}")
        logger.info(f"  Best error: {stats['best_error']:.6f}")
        logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

        # Analyze perturbation effectiveness
        if stats["post_perturbation_tracking"]:
            successful_perturbations = [
                p for p in stats["post_perturbation_tracking"] if p["success"]
            ]
            success_rate = (
                len(successful_perturbations) / len(stats["post_perturbation_tracking"])
                if stats["post_perturbation_tracking"]
                else 0
            )
            avg_improvement = (
                np.mean([p["improvement"] for p in successful_perturbations])
                if successful_perturbations
                else 0
            )
            logger.info(f"  Perturbation success rate: {success_rate:.2f}")
            logger.info(
                f"  Average improvement from successful perturbations: {avg_improvement:.6f}"
            )

        if not cfg.no_anim:
            # Create enhanced 3-panel animation with field magnitude and both types of error
            logger.info("Creating GS iteration animation...")
            visualize_iteration_history(
                points,
                H,
                coefficient_history,
                field_history,
                cfg.resolution,
                measurement_plane,
                show_plot=(cfg.show_plot and not cfg.no_plot),
                output_file="gs_animation.gif",
                frame_skip=10,  # Increased frame skip for faster animation
                perturbation_iterations=stats.get("perturbation_iterations", []), # Get stats from HPR result
                restart_iterations=[],  # No restarts in vanilla implementation
                convergence_threshold=cfg.convergence_threshold,
                measured_magnitude=measured_magnitude,
                # Pass the measured magnitude for true error calculation
            )

            # Create enhanced animation with 3D current visualization
            # and 2D field history with true field and error
            logger.info("Creating enhanced 4-panel animation...")
            visualize_current_and_field_history(
                points,
                coefficient_history,
                field_history,
                true_field,  # Pass the synthetic true field for comparison
                cfg.resolution,
                measurement_plane,
                show_plot=(cfg.show_plot and not cfg.no_plot),
                output_file="current_field_animation.gif",
            )
        else:
            logger.info("Animation generation disabled via no_anim flag")
    else:
        # Handle the return signature with just stats
        cluster_coefficients, stats = hpr_result

        # Log the perturbation statistics
        logger.info("Perturbation statistics:")
        logger.info(f"  Total iterations: {stats['iterations']}")
        logger.info(f"  Final error: {stats['final_error']:.6f}")
        logger.info(f"  Best error: {stats['best_error']:.6f}")
        logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

    # Reconstruct field using estimated coefficients
    reconstructed_field = reconstruct_field(H, cluster_coefficients)
    reconstructed_field_2d = reconstructed_field.reshape(cfg.resolution, cfg.resolution)

    # Calculate metrics
    def normalized_rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2)) / (np.max(a) - np.min(a))

    def normalized_correlation(a, b):
        a_norm = (a - np.mean(a)) / np.std(a)
        b_norm = (b - np.mean(b)) / np.std(b)
        return np.correlate(a_norm.flatten(), b_norm.flatten())[0] / len(a_norm.flatten())

    # Calculate reconstruction quality metrics
    rmse = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))

    # Print metrics
    logger.info("Reconstruction quality metrics:")
    logger.info(f"  Normalized RMSE: {rmse:.4f}")
    logger.info(f"  Correlation: {corr:.4f}")

    # Generate final field comparison visualization if not disabled
    if not cfg.no_plot:
        logger.info("Generating final field comparison visualization...")
        visualize_fields(
            points,
            currents,
            measurement_plane,
            true_field_2d,
            measured_magnitude_2d,
            reconstructed_field_2d,
            rmse, # Calculated RMSE
            corr, # Calculated Correlation
            show_plot=cfg.show_plot,
            output_file="results.png",
        )
    else:
        logger.info("Plot generation disabled via no_plot flag")


if __name__ == "__main__":
    main() # Hydra automatically passes the config object (cfg)
