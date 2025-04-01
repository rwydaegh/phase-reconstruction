import logging
import os
import pickle

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud
from src.io import load_measurement_data
from src.utils.field_utils import reconstruct_field
from src.utils.smoothing import (
    gaussian_convolution_local_support,
    physical_to_pixel_radius,
)
from src.visualization.field_plots import visualize_fields

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_measurement_plane(data):
    """
    Create measurement plane from continuous and discrete axes and convert to meters,
    ensuring the plane is centered at (0,0,0)
    """
    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]

    points_continuous = data["points_continuous"] / 1000.0 # Convert mm to m
    points_discrete = data["points_discrete"] / 1000.0 # Convert mm to m

    if continuous_axis == "z" and discrete_axis == "y":
        points_continuous_centered = points_continuous - np.mean(points_continuous)
        points_discrete_centered = points_discrete - np.mean(points_discrete)

        Z, Y = np.meshgrid(points_continuous_centered, points_discrete_centered)
        X = np.full_like(Z, 0.0)

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

    n_discrete = len(points_discrete)
    n_continuous = len(points_continuous)

    logger.info(f"Original data dimensions: {n_discrete}×{n_continuous}")

    assert target_resolution <= n_discrete, (
        f"Target resolution {target_resolution} must be less than or equal to "
        f"the original resolution of discrete axis {n_discrete}"
    )
    target_n_discrete = min(target_resolution, n_discrete)
    target_n_continuous = min(target_resolution, n_continuous)

    logger.info(f"Target dimensions: {target_n_discrete}×{target_n_continuous}")

    discrete_indices = np.linspace(0, n_discrete - 1, target_n_discrete, dtype=int)
    continuous_indices = np.linspace(0, n_continuous - 1, target_n_continuous, dtype=int)

    sampled_points_discrete = points_discrete[discrete_indices]
    sampled_points_continuous = points_continuous[continuous_indices]
    sampled_results = results[np.ix_(discrete_indices, continuous_indices)]

    logger.info(
        f"Sampled data dimensions: {len(sampled_points_discrete)}×{len(sampled_points_continuous)}"
    )

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

    return points[::factor]


# Removed duplicate functions physical_to_pixel_radius and apply_gaussian_smoothing
# These are now imported from src.utils.smoothing
@hydra.main(config_path="conf", config_name="measured_data", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))  # Log loaded config

    # Get Hydra output directory (current working directory)
    output_dir = os.getcwd() + "/plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("Starting point cloud processing...")
    logger.info(f"Hydra output directory: {output_dir}")

    if cfg.use_source_pointcloud:
        logger.info(f"Loading source point cloud from {cfg.source_pointcloud_path}...")
        with open(cfg.source_pointcloud_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            if data.shape[1] >= 3:
                full_points = downsample_pointcloud(data, cfg.pointcloud_downsample)

                logger.info(
                    f"Downsampled point cloud from {len(data)} to {len(full_points)} points "
                    f"(factor: {cfg.pointcloud_downsample})"
                )

                translation = np.array([-5.35211, -6.34833, -1.28819])
                points = full_points[:,:3] + translation # Adding negative values = subtraction
                logger.info(f"Translated point cloud by {translation} meters")

                if hasattr(
                    cfg, "max_distance_from_origin"
                ) and cfg.max_distance_from_origin  != -1:
                    # Always calculate distance from the origin based on translated points
                    # Use only the first 3 columns (X, Y, Z) for distance calculation
                    distances = np.sqrt(np.sum(points[:, :3]**2, axis=1))

                    orig_num_points = len(points)
                    points = points[distances <= cfg.max_distance_from_origin]

                    logger.info(
                        f"Filtered out {orig_num_points - len(points)} points "
                        f"more than {cfg.max_distance_from_origin}m from origin"
                    )
                    logger.info(
                        f"Retained {len(points)} points "
                        f"within {cfg.max_distance_from_origin}m from origin"
                    )
            else:
                raise ValueError(f"Source point cloud data has unexpected shape: {data.shape}")
        else:
            raise ValueError(f"Source point cloud data has unexpected type: {type(data)}")

        logger.info(f"Using real data point cloud with {len(points)} points, shape: {points.shape}")
    else:
        logger.info("Finished point cloud processing.")
        points = create_test_pointcloud(cfg)
        logger.info(
            f"Created cube environment with {len(points)} points, cube size: {cfg.room_size}m"
        )

    # Load and preprocess real measurement data
    measurement_file = "measurement_data/x400_zy.pickle"
    logger.info(f"Loading measurement data from {measurement_file}...")
    measurement_data = load_measurement_data(measurement_file)

    sampled_data = sample_measurement_data(measurement_data, target_resolution=cfg.resolution)

    measurement_plane = create_measurement_plane(sampled_data)
    logger.info("Finished creating measurement plane.")

    measured_magnitude = np.abs(sampled_data["results"]).flatten()

    measured_magnitude = np.nan_to_num(measured_magnitude, nan=np.nanmin(measured_magnitude))

    if (
        cfg.use_source_pointcloud
        and hasattr(cfg, "use_measurement_frequency")
        and cfg.use_measurement_frequency
    ):
        wavelength = 299792458 / (measurement_data["frequency"] / 1e9) / 1000 # meters
        k = 2 * np.pi / wavelength
        logger.info(f"Using measurement frequency: {measurement_data['frequency']/1e9:.2f} GHz")
        logger.info(f"Calculated wave number k: {k:.2f}, wavelength: {wavelength:.4f} m")
    else:
        k = 2 * np.pi / cfg.wavelength
        logger.info(f"Using config wave number k: {k:.2f}, wavelength: {cfg.wavelength:.4f} m")

    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, measurement_plane, k)

    initial_field_values = None
    if hasattr(cfg, "uniform_current_init") and cfg.uniform_current_init:
        logger.info(
            f"Initializing currents with uniform amplitude {cfg.initial_current_amplitude} "
            f"and random phase"
        )
        n_points = points.shape[0]
        random_phases = np.random.uniform(0, 2 * np.pi, n_points)
        initial_currents = cfg.initial_current_amplitude * np.exp(1j * random_phases)
        initial_field = H @ initial_currents

        from src.utils.phase_retrieval_utils import apply_magnitude_constraint
        initial_field_values = apply_magnitude_constraint(initial_field, measured_magnitude)

        logger.info(
            f"Initialized field values from uniform currents "
            f"with shape {initial_field_values.shape}"
        )

    # Run holographic phase retrieval
    logger.info("Starting holographic phase retrieval...")
    hpr_result = holographic_phase_retrieval(
        cfg,
        H,
        measured_magnitude,
        initial_field_values=initial_field_values,
    )
    logger.info("Finished holographic phase retrieval.")

    if cfg.return_history:
        cluster_coefficients, coefficient_history, field_history, stats = hpr_result
    else:
        cluster_coefficients, stats = hpr_result

    logger.info("Perturbation statistics:")
    logger.info(f"  Total iterations: {stats['iterations']}")
    logger.info(f"  Final RMSE: {stats['final_rmse']:.6f}")
    logger.info(f"  Best RMSE: {stats['best_rmse']:.6f}")
    logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

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
    from src.visualization.history_plots import (
        visualize_current_and_field_history,
        visualize_iteration_history,
    )

    reconstructed_field = reconstruct_field(H, cluster_coefficients)

    resolution_y = len(sampled_data["points_discrete"])
    resolution_z = len(sampled_data["points_continuous"])
    measured_magnitude_2d = measured_magnitude.reshape(resolution_y, resolution_z)
    reconstructed_field_2d = reconstructed_field.reshape(resolution_y, resolution_z)

    true_field = measured_magnitude.copy()
    true_field_2d = measured_magnitude_2d.copy()

    if cfg.enable_smoothing:
        logger.info(f"Applying Gaussian smoothing with radius {cfg.smoothing_radius_mm} mm")

        R_pixels = physical_to_pixel_radius(
            cfg.smoothing_radius_mm,
            sampled_data["points_continuous"],
            sampled_data["points_discrete"],
        )

        logger.info(
            f"Converted physical radius {cfg.smoothing_radius_mm} mm "
            f"to pixel radius {R_pixels:.2f}"
        )

        # Use the imported function
        smoothed_true_field_2d = gaussian_convolution_local_support(true_field_2d, R_pixels)

        reconstructed_field_abs_2d = np.abs(reconstructed_field_2d)
        # Use the imported function
        smoothed_reconstructed_field_abs_2d = gaussian_convolution_local_support(
            reconstructed_field_abs_2d, R_pixels
        )

        reconstructed_field_phase = np.angle(reconstructed_field_2d)
        # Combine smoothed magnitude and original phase
        smoothed_reconstructed_field_2d = smoothed_reconstructed_field_abs_2d * np.exp(
            1j * reconstructed_field_phase
        )

        true_field_2d = smoothed_true_field_2d
        reconstructed_field_2d = smoothed_reconstructed_field_2d

        true_field = true_field_2d.flatten()
        reconstructed_field = reconstructed_field_2d.flatten()

        logger.info("Gaussian smoothing applied to both true and reconstructed fields")

    rmse = normalized_rmse(true_field, np.abs(reconstructed_field))
    corr = normalized_correlation(true_field, np.abs(reconstructed_field))

    logger.info("Reconstruction quality metrics:")
    logger.info(f"  Normalized RMSE: {rmse:.4f}")
    logger.info(f"  Correlation: {corr:.4f}")

    results_dict = {
        "points": points,
        "cluster_coefficients": cluster_coefficients,
        "reconstructed_field": reconstructed_field,
        "reconstructed_field_2d": reconstructed_field_2d,
        "measured_magnitude": measured_magnitude,
        "measured_magnitude_2d": measured_magnitude_2d,
        "stats": stats,
        "config": cfg
    }

    if cfg.enable_smoothing:
        results_dict["smoothing_applied"] = True
        results_dict["smoothing_radius_mm"] = cfg.smoothing_radius_mm
        results_dict["smoothed_true_field"] = true_field
        results_dict["smoothed_true_field_2d"] = true_field_2d
        results_dict["smoothed_reconstructed_field"] = reconstructed_field
        results_dict["smoothed_reconstructed_field_2d"] = reconstructed_field_2d

    with open(f"{output_dir}/reconstruction_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    logger.info("Finished saving results.")
    logger.info(f"Results saved to {output_dir}/reconstruction_results.pkl")

    # Generate field comparison visualization
    logger.info("Starting visualization...")
    if not cfg.no_plot:
        logger.info("Generating field comparison visualization...")

        visualize_fields(
            points,                   # points
            cluster_coefficients,     # currents
            measurement_plane,        # measurement_plane
            true_field_2d,            # true_field_2d
            measured_magnitude_2d,    # measured_magnitude_2d
            reconstructed_field_2d,   # reconstructed_field_2d
            rmse,                     # rmse
            corr,                     # correlation
            show_plot=cfg.show_plot,
            output_dir=output_dir
        )

        if not cfg.no_anim:
            assert cfg.return_history is True, "History must be returned to create animations"
            logger.info("Creating GS iteration animation...")
            logger.info(f"Field dimensions: {resolution_y}×{resolution_z}")

            if resolution_y == resolution_z:
                resolution_param = resolution_y
            else:
                Exception("Field dimensions are not square")

            visualize_iteration_history(
                points,
                H,
                coefficient_history,
                field_history,
                resolution_param,
                measurement_plane,        # Added measurement_plane
                show_plot=cfg.show_plot,
                output_file=output_dir,
                animation_filename=os.path.join(output_dir, "gs_animation.gif"),
                frame_skip=1, # Lower frame skip for smoother animation
                perturbation_iterations=stats.get("perturbation_iterations", []),
                convergence_threshold=cfg.convergence_threshold,
                measured_magnitude=measured_magnitude,
            )

            logger.info("Creating enhanced 4-panel animation...")
            visualize_current_and_field_history(
                points,
                coefficient_history,
                field_history,
                true_field,
                resolution_param,
                measurement_plane,        # Added measurement_plane
                show_plot=cfg.show_plot,
                output_file=output_dir,
                animation_filename=os.path.join(output_dir, "current_field_animation.gif"),
                frame_skip=1, # Lower frame skip for smoother animation
            )

    logger.info("Finished visualization.")

if __name__ == "__main__":
    main()
