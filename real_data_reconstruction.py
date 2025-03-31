import logging
import os
import pickle
from dataclasses import asdict, fields

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud

# Import from src directory
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.simulation_config_real_data import SimulationConfig
from src.utils.field_utils import compute_fields, reconstruct_field
from src.visualization.field_plots import visualize_fields
from src.io import load_measurement_data


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function load_measurement_data moved to src/io.py


def create_measurement_plane(data):
    """
    Create measurement plane from continuous and discrete axes and convert to meters,
    ensuring the plane is centered at (0,0,0)
    """
    # Extract axes
    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]

    # Get coordinates (in mm) and convert to meters
    points_continuous = data["points_continuous"] / 1000.0  # Convert mm to m
    points_discrete = data["points_discrete"] / 1000.0  # Convert mm to m

    # For x400_zy.pickle: continuous_axis='z', discrete_axis='y'
    if continuous_axis == "z" and discrete_axis == "y":
        # Center the coordinate values around zero
        points_continuous_centered = points_continuous - np.mean(points_continuous)
        points_discrete_centered = points_discrete - np.mean(points_discrete)

        # Create meshgrid of z and y coordinates (in meters), centered at origin
        Z, Y = np.meshgrid(points_continuous_centered, points_discrete_centered)
        # Fixed x value at origin (0,0,0) - use exactly the same value for all points
        X = np.full_like(Z, 0.0)

        # Stack to create 3D coordinates
        measurement_plane = np.stack([X, Y, Z], axis=-1)

        logger.info(
            f"Created measurement plane centered at (0,0,0) with shape: {measurement_plane.shape}"
        )
        logger.info(f"Measurement plane type: YZ (fixed X at {X[0,0]})")
    else:
        raise ValueError(f"Unsupported axes: {continuous_axis} and {discrete_axis}")

    return measurement_plane


def sample_measurement_data(data, target_resolution=50):
    """
    Sample measurement data to reduce dimensionality to the target resolution

    Creates approximately square output by sampling both axes to be close to the target resolution.
    """
    results = data["results"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]

    # Current dimensions
    n_discrete = len(points_discrete)
    n_continuous = len(points_continuous)

    logger.info(f"Original data dimensions: {n_discrete}×{n_continuous}")

    # Limit both axes to target_resolution to create approximately square output
    assert target_resolution <= n_discrete, (
        f"Target resolution {target_resolution} must be less than or equal to "
        f"the original resolution of discrete axis {n_discrete}"
    )
    target_n_discrete = min(target_resolution, n_discrete)
    target_n_continuous = min(target_resolution, n_continuous)

    logger.info(f"Target dimensions: {target_n_discrete}×{target_n_continuous}")

    # Sample indices
    discrete_indices = np.linspace(0, n_discrete - 1, target_n_discrete, dtype=int)
    continuous_indices = np.linspace(0, n_continuous - 1, target_n_continuous, dtype=int)

    # Sample data
    sampled_points_discrete = points_discrete[discrete_indices]
    sampled_points_continuous = points_continuous[continuous_indices]
    sampled_results = results[np.ix_(discrete_indices, continuous_indices)]

    logger.info(
        f"Sampled data dimensions: {len(sampled_points_discrete)}×{len(sampled_points_continuous)}"
    )

    # Update data dictionary
    sampled_data = data.copy()
    sampled_data["results"] = sampled_results
    sampled_data["points_discrete"] = sampled_points_discrete
    sampled_data["points_continuous"] = sampled_points_continuous

    return sampled_data


def downsample_pointcloud(points, factor):
    """
    Downsample a point cloud by the specified factor.

    Args:
        points: Point cloud array with shape (N, 3) or more columns
        factor: Downsample factor (e.g., 2 means keep every 2nd point)

    Returns:
        Downsampled point cloud
    """
    if factor <= 1:
        return points  # No downsampling needed

    # Simple downsampling by taking every nth point
    return points[::factor]


def physical_to_pixel_radius(R_mm, points_continuous, points_discrete):
    """
    Convert physical radius (mm) to pixel radius

    Args:
        R_mm: Radius in mm
        points_continuous: Points along continuous axis (in mm)
        points_discrete: Points along discrete axis (in mm)

    Returns:
        R_pixels: Radius in pixels
    """
    # Calculate average spacing in continuous and discrete dimensions
    cont_spacing = (np.max(points_continuous) - np.min(points_continuous)) / (
        len(points_continuous) - 1
    )

    # Handle case where points_discrete might be a list
    if isinstance(points_discrete, list):
        disc_points = np.array(points_discrete)
    else:
        disc_points = points_discrete

    disc_spacing = (np.max(disc_points) - np.min(disc_points)) / (len(disc_points) - 1)

    # Average spacing
    avg_spacing = (cont_spacing + disc_spacing) / 2

    # Convert R in mm to pixels
    R_pixels = R_mm / avg_spacing

    return R_pixels


def apply_gaussian_smoothing(data, R_pixels):
    """
    Apply Gaussian convolution with local support R

    Args:
        data: 2D data to smooth
        R_pixels: Radius in pixels

    Returns:
        smoothed_data: Smoothed 2D data
    """
    # Create a copy of the data to avoid modifying the original
    smoothed_data = np.copy(data)

    # Calculate sigma based on R (standard choice: sigma = R/3)
    sigma = R_pixels / 3.0

    # Use scipy's gaussian_filter for efficient implementation
    smoothed_data = gaussian_filter(smoothed_data, sigma=sigma, mode="nearest")

    return smoothed_data


def main():
    # Load configuration
    config = SimulationConfig()

    # Create output directory
    output_dir = "cube_environment_results"
    os.makedirs(output_dir, exist_ok=True)

    # Determine whether to use test point cloud or real data
    if config.use_source_pointcloud:
        # Load source point cloud (real environment)
        logger.info(f"Loading source point cloud from {config.source_pointcloud_path}...")
        with open(config.source_pointcloud_path, "rb") as f:
            data = pickle.load(f)

        # Source point cloud processing
        if isinstance(data, np.ndarray):
            if data.shape[1] >= 3:
                # Downsample the point cloud
                full_points = downsample_pointcloud(data, config.pointcloud_downsample)

                # Assume it's a point cloud array with x,y,z coordinates in the first 3 columns
                points = full_points[:, :3]

                logger.info(
                    f"Downsampled point cloud from {len(data)} to {len(points)} points "
                    f"(factor: {config.pointcloud_downsample})"
                )

                # Apply translation by negative of specified coordinates
                translation = np.array([-5.35211, -6.34833, -1.28819])
                points = points + translation  # Adding negative values = subtraction
                logger.info(f"Translated point cloud by {translation} meters")

                # Filter points based on distance from origin if specified
                if hasattr(
                    config, "max_distance_from_origin"
                ) and config.max_distance_from_origin < float("inf"):
                    # Calculate distance from origin for each point
                    if full_points.shape[1] > 3:  # If there's a distance column
                        distances = full_points[:, 3]
                    else:
                        # Calculate Euclidean distance from origin
                        distances = np.sqrt(np.sum(points**2, axis=1))

                    # Filter points
                    orig_num_points = len(points)
                    points = points[distances <= config.max_distance_from_origin]

                    # Log how many points were removed
                    logger.info(
                        f"Filtered out {orig_num_points - len(points)} points "
                        f"more than {config.max_distance_from_origin}m from origin"
                    )
                    logger.info(
                        f"Retained {len(points)} points "
                        f"within {config.max_distance_from_origin}m from origin"
                    )
            else:
                raise ValueError(f"Source point cloud data has unexpected shape: {data.shape}")
        else:
            raise ValueError(f"Source point cloud data has unexpected type: {type(data)}")

        logger.info(f"Using real data point cloud with {len(points)} points, shape: {points.shape}")
    else:
        # Create test point cloud (centered cube)
        points = create_test_pointcloud(config)
        logger.info(
            f"Created cube environment with {len(points)} points, cube size: {config.room_size}m"
        )

    # Load and preprocess real measurement data
    measurement_file = "measurement_data/x400_zy.pickle"
    logger.info(f"Loading measurement data from {measurement_file}...")
    measurement_data = load_measurement_data(measurement_file)

    # Sample measurement data to reduce dimensionality
    sampled_data = sample_measurement_data(measurement_data, target_resolution=config.resolution)

    # Create measurement plane (already converted to meters)
    measurement_plane = create_measurement_plane(sampled_data)

    # Extract measured field magnitude (reshape to 1D array for phase retrieval)
    measured_magnitude = np.abs(sampled_data["results"]).flatten()

    # Set any NaN values to a small positive number
    measured_magnitude = np.nan_to_num(measured_magnitude, nan=np.nanmin(measured_magnitude))

    # Create wavelength in meters from frequency (for calculating wave number k)
    if (
        config.use_source_pointcloud
        and hasattr(config, "use_measurement_frequency")
        and config.use_measurement_frequency
    ):
        # Calculate k from the measurement frequency when using real data
        wavelength = 299792458 / (measurement_data["frequency"] / 1e9) / 1000  # meters
        k = 2 * np.pi / wavelength
        logger.info(f"Using measurement frequency: {measurement_data['frequency']/1e9:.2f} GHz")
        logger.info(f"Calculated wave number k: {k}, wavelength: {wavelength:.4f} m")
    else:
        # Use k from config
        k = config.k
        wavelength = config.wavelength
        logger.info(f"Using config wave number k: {k}, wavelength: {wavelength:.4f} m")

    # Create channel matrix for scalar fields
    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, measurement_plane, k)

    # If using uniform current initialization, prepare initial field values
    initial_field_values = None
    if hasattr(config, "uniform_current_init") and config.uniform_current_init:
        logger.info(
            f"Initializing currents with uniform amplitude {config.initial_current_amplitude} "
            f"and random phase"
        )
        # Create uniform amplitude random phase current densities
        n_points = points.shape[0]
        random_phases = np.random.uniform(0, 2 * np.pi, n_points)
        initial_currents = config.initial_current_amplitude * np.exp(1j * random_phases)

        # Compute pseudoinverse for initialization (using same regularization settings)
        from src.utils.phase_retrieval_utils import compute_pseudoinverse

        # H_pinv_init = compute_pseudoinverse( # Unused
        #     H, config.regularization, config.adaptive_regularization
        # )

        # Compute initial field values from these currents
        initial_field = H @ initial_currents

        # Apply measured magnitude constraint to get initial field values
        from src.utils.phase_retrieval_utils import apply_magnitude_constraint

        initial_field_values = apply_magnitude_constraint(initial_field, measured_magnitude)

        logger.info(
            f"Initialized field values from uniform currents "
            f"with shape {initial_field_values.shape}"
        )

    # Run holographic phase retrieval
    logger.info("Running holographic phase retrieval...")
    hpr_result = holographic_phase_retrieval(
        H,
        measured_magnitude,
        adaptive_regularization=config.adaptive_regularization,
        num_iterations=config.gs_iterations,
        convergence_threshold=config.convergence_threshold,
        regularization=config.regularization,
        return_history=config.return_history,
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
        initial_field_values=initial_field_values,  # Pass initial field values if provided
    )

    if config.return_history:
        cluster_coefficients, coefficient_history, field_history, stats = hpr_result
    else:
        cluster_coefficients, stats = hpr_result

    # Log the perturbation statistics
    logger.info("Perturbation statistics:")
    logger.info(f"  Total iterations: {stats['iterations']}")
    logger.info(f"  Final error: {stats['final_error']:.6f}")
    logger.info(f"  Best error: {stats['best_error']:.6f}")
    logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

    # Analyze perturbation effectiveness
    if stats["post_perturbation_tracking"]:
        successful_perturbations = [p for p in stats["post_perturbation_tracking"] if p["success"]]
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
        logger.info(f"  Average improvement from successful perturbations: {avg_improvement:.6f}")

    from src.utils.normalized_correlation import normalized_correlation
    from src.utils.normalized_rmse import normalized_rmse
    from src.visualization import (
        visualize_current_and_field_history,
        visualize_iteration_history,
    )

    # Reconstruct field using estimated coefficients
    reconstructed_field = reconstruct_field(H, cluster_coefficients)

    # Reshape for visualization
    resolution_y = len(sampled_data["points_discrete"])
    resolution_z = len(sampled_data["points_continuous"])
    measured_magnitude_2d = measured_magnitude.reshape(resolution_y, resolution_z)
    reconstructed_field_2d = reconstructed_field.reshape(resolution_y, resolution_z)

    # We don't have true field, so use measured magnitude as reference
    true_field = measured_magnitude.copy()
    true_field_2d = measured_magnitude_2d.copy()

    # Apply Gaussian smoothing as post-processing if enabled in config
    if config.enable_smoothing:
        logger.info(f"Applying Gaussian smoothing with radius {config.smoothing_radius_mm} mm")

        # Convert physical radius to pixel radius
        R_pixels = physical_to_pixel_radius(
            config.smoothing_radius_mm,
            sampled_data["points_continuous"],
            sampled_data["points_discrete"],
        )

        logger.info(
            f"Converted physical radius {config.smoothing_radius_mm} mm "
            f"to pixel radius {R_pixels:.2f}"
        )

        # Apply smoothing to true field (measured magnitude)
        smoothed_true_field_2d = apply_gaussian_smoothing(true_field_2d, R_pixels)

        # Apply smoothing to reconstructed field magnitude
        reconstructed_field_abs_2d = np.abs(reconstructed_field_2d)
        smoothed_reconstructed_field_abs_2d = apply_gaussian_smoothing(
            reconstructed_field_abs_2d, R_pixels
        )

        # Preserve phase but use smoothed magnitude for reconstructed field
        reconstructed_field_phase = np.angle(reconstructed_field_2d)
        smoothed_reconstructed_field_2d = smoothed_reconstructed_field_abs_2d * np.exp(
            1j * reconstructed_field_phase
        )

        # Update 2D fields with smoothed versions
        true_field_2d = smoothed_true_field_2d
        reconstructed_field_2d = smoothed_reconstructed_field_2d

        # Update 1D fields as well (for metrics calculation)
        true_field = true_field_2d.flatten()
        reconstructed_field = reconstructed_field_2d.flatten()

        logger.info("Gaussian smoothing applied to both true and reconstructed fields")

    # Calculate reconstruction quality metrics
    rmse = normalized_rmse(true_field, np.abs(reconstructed_field))
    corr = normalized_correlation(true_field, np.abs(reconstructed_field))

    # Print metrics
    logger.info("Reconstruction quality metrics:")
    logger.info(f"  Normalized RMSE: {rmse:.4f}")
    logger.info(f"  Correlation: {corr:.4f}")

    # Save results
    results_dict = {
        "points": points,
        "cluster_coefficients": cluster_coefficients,
        "reconstructed_field": reconstructed_field,
        "reconstructed_field_2d": reconstructed_field_2d,
        "measured_magnitude": measured_magnitude,
        "measured_magnitude_2d": measured_magnitude_2d,
        "stats": stats,
        "config": asdict(config),
    }

    # Add smoothing information if smoothing was applied
    if config.enable_smoothing:
        results_dict["smoothing_applied"] = True
        results_dict["smoothing_radius_mm"] = config.smoothing_radius_mm
        results_dict["smoothed_true_field"] = true_field
        results_dict["smoothed_true_field_2d"] = true_field_2d
        results_dict["smoothed_reconstructed_field"] = reconstructed_field
        results_dict["smoothed_reconstructed_field_2d"] = reconstructed_field_2d

    with open(f"{output_dir}/reconstruction_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    logger.info(f"Results saved to {output_dir}/reconstruction_results.pkl")

    # Generate field comparison visualization
    if not config.no_plot:
        logger.info("Generating field comparison visualization...")
        # With improved visualization functions, we can use the measurement plane directly
        # without any adaptation - the functions now detect plane type automatically

        visualize_fields(
            points,
            cluster_coefficients,  # Use reconstructed currents
            measurement_plane,  # Use measurement plane directly
            true_field_2d,
            measured_magnitude_2d,
            reconstructed_field_2d,
            rmse,
            corr,
            show_plot=config.show_plot,
            output_file=f"{output_dir}/cube_reconstruction.png",
        )

        # Create animation if history was returned
        if not config.no_anim:
            assert config.return_history is True, "History must be returned to create animations"
            logger.info("Creating GS iteration animation...")
            # Now our data is square (50x50), we can use the standard visualization functions
            logger.info(f"Field dimensions: {resolution_y}×{resolution_z}")

            # Use a single integer for resolution since our field is now square
            if resolution_y == resolution_z:
                resolution_param = resolution_y
            else:
                Exception("Field dimensions are not square")

            logger.info("Creating GS iteration animation...")
            visualize_iteration_history(
                points,
                H,
                coefficient_history,
                field_history,
                resolution_param,
                measurement_plane,  # Use measurement plane directly
                show_plot=config.show_plot,
                output_file=f"{output_dir}/gs_animation.gif",
                frame_skip=1,  # Lower frame skip for smoother animation
                perturbation_iterations=stats.get("perturbation_iterations", []),
                convergence_threshold=config.convergence_threshold,
                measured_magnitude=measured_magnitude,
            )

            logger.info("Creating enhanced 4-panel animation...")
            visualize_current_and_field_history(
                points,
                coefficient_history,
                field_history,
                true_field,
                resolution_param,
                measurement_plane,  # Use measurement plane directly
                show_plot=config.show_plot,
                output_file=f"{output_dir}/current_field_animation.gif",
                frame_skip=1,  # Lower frame skip for smoother animation
            )

    logger.info("Cube environment reconstruction complete!")


if __name__ == "__main__":
    main()
