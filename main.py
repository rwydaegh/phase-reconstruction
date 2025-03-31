import argparse
import logging
import pickle
import random
from dataclasses import asdict, dataclass, fields

import numpy as np

from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.simulation_config_fake_data import SimulationConfig
from src.utils.field_utils import compute_fields, reconstruct_field
from src.visualization import visualize_current_and_field_history, visualize_iteration_history
from src.visualization.field_plots import visualize_fields

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    config: SimulationConfig,
) -> None:
    """Main demonstration of field reconstruction using holographic phase retrieval."""

    # Create or load point cloud
    if config.use_source_pointcloud:
        logger.info(f"Loading source point cloud from {config.source_pointcloud_path}...")
        with open(config.source_pointcloud_path, "rb") as f:
            data = pickle.load(f)

        # Source is a (5328,4) array with last column being distance
        # Just take the first 3 columns (x, y, z coordinates)
        points = data[:, :3]
        logger.info(f"Loaded point cloud with {len(points)} points, shape: {points.shape}")
    else:
        logger.info("Creating test point cloud...")
        points = create_test_pointcloud(config)

    logger.info(f"Final points shape: {points.shape}")

    # Create simple current distribution (scalar amplitudes for each point)
    # For scalar field, we just need one value per point
    currents = np.zeros(len(points), dtype=complex)

    # Set random indices in the currents array to be active
    # with log-normal amplitude distribution and random phases
    num_sources = min(config.num_sources, len(currents))
    source_indices = random.sample(range(len(currents)), num_sources)

    # Generate log-normal amplitudes
    amplitudes = np.random.lognormal(mean=0, sigma=config.amplitude_sigma, size=num_sources)

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
    x = np.linspace(-config.plane_size / 2, config.plane_size / 2, config.resolution)
    y = np.linspace(-config.plane_size / 2, config.plane_size / 2, config.resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    # Create channel matrix for scalar fields
    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, measurement_plane, config.k)

    # Calculate ground truth field on measurement plane
    logger.info("Calculating ground truth field...")
    true_field = compute_fields(points, currents, measurement_plane, config.k, H)

    # Reshape to 2D grid
    true_field_2d = true_field.reshape(config.resolution, config.resolution)

    # Get field magnitude (what we would measure)
    measured_magnitude = np.abs(true_field)
    measured_magnitude_2d = measured_magnitude.reshape(config.resolution, config.resolution)

    # Phase retrieval demonstration
    logger.info("Running holographic phase retrieval with perturbation strategies...")

    hpr_result = holographic_phase_retrieval(
        H,
        measured_magnitude,
        adaptive_regularization=config.adaptive_regularization,
        num_iterations=config.gs_iterations,
        convergence_threshold=config.convergence_threshold,
        regularization=config.regularization,
        return_history=config.return_history,
        # Use parameters from config
        enable_perturbations=config.enable_perturbations,
        stagnation_window=config.stagnation_window,
        stagnation_threshold=config.stagnation_threshold,
        perturbation_intensity=config.perturbation_intensity,
        perturbation_mode=config.perturbation_mode,
        constraint_skip_iterations=config.constraint_skip_iterations,
        momentum_factor=config.momentum_factor,
        temperature=config.temperature,
        verbose=config.verbose,
        no_plot=config.no_plot,
    )

    if config.return_history:
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

        if not config.no_anim:
            # Create enhanced 3-panel animation with field magnitude and both types of error
            logger.info("Creating GS iteration animation...")
            visualize_iteration_history(
                points,
                H,
                coefficient_history,
                field_history,
                config.resolution,
                measurement_plane,
                show_plot=(config.show_plot and not config.no_plot),
                output_file="gs_animation.gif",
                frame_skip=10,  # Increased frame skip for faster animation
                perturbation_iterations=stats.get("perturbation_iterations", []),
                restart_iterations=[],  # No restarts in vanilla implementation
                convergence_threshold=config.convergence_threshold,
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
                true_field,  # Pass the true field for comparison
                config.resolution,
                measurement_plane,
                show_plot=(config.show_plot and not config.no_plot),
                output_file="current_field_animation.gif",
            )
        else:
            logger.info("Animation generation disabled with no_anim flag")
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
    reconstructed_field_2d = reconstructed_field.reshape(config.resolution, config.resolution)

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
    if not config.no_plot:
        logger.info("Generating final field comparison visualization...")
        visualize_fields(
            points,
            currents,
            measurement_plane,
            true_field_2d,
            measured_magnitude_2d,
            reconstructed_field_2d,
            rmse,
            corr,
            show_plot=config.show_plot,
            output_file="results.png",
        )
    else:
        logger.info("Plot generation disabled with no_plot flag")


def validate_config(config: SimulationConfig) -> None:
    """Validate simulation configuration."""
    if config.resolution < 10 or config.resolution > 200:
        raise ValueError("Resolution must be between 10 and 200")
    if config.wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    if config.room_size <= 0:
        raise ValueError("Room size must be positive")
    if config.wall_points <= 0:
        raise ValueError("Wall points must be positive")
    if config.num_sources <= 0:
        raise ValueError("Number of sources must be positive")
    if config.gs_iterations <= 0:
        raise ValueError("GS iterations must be positive")
    if config.convergence_threshold <= 0:
        raise ValueError("Convergence threshold must be positive")


if __name__ == "__main__":
    try:
        # Create argument parser
        parser = argparse.ArgumentParser(
            description="Run field reconstruction simulation with refactored phase retrieval"
        )

        # Add arguments for each configuration parameter
        for field in fields(SimulationConfig):
            # Convert default value to string for help message
            default_value = getattr(SimulationConfig, field.name)
            help_text = f"{field.name} (default: {default_value})"

            # Handle boolean fields differently
            if field.type is bool:
                parser.add_argument(
                    f"--{field.name}",
                    type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
                    help=help_text,
                )
            else:
                # Add the argument with appropriate type
                parser.add_argument(f"--{field.name}", type=field.type, help=help_text)

        # Parse arguments
        args = parser.parse_args()

        # Create configuration with default values
        config = SimulationConfig()

        # Override with any command-line args provided
        for field in fields(SimulationConfig):
            arg_value = getattr(args, field.name)
            if arg_value is not None:
                setattr(config, field.name, arg_value)

        validate_config(config)

        logger.info("Starting field reconstruction simulation with phase retrieval")
        logger.info(f"Configuration: {config}")

        main(
            config=config,
        )

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise
