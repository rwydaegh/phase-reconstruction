import logging
import os
import pickle
import random

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

    points_continuous = data["points_continuous"] / 1000.0  # Convert mm to m
    points_discrete = data["points_discrete"] / 1000.0  # Convert mm to m

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

    # Set random seed for reproducibility
    seed = cfg.random_seed
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

    if cfg.use_source_pointcloud:
        logger.info(f"Loading source point cloud from {cfg.source_pointcloud_path}...")
        with open(cfg.source_pointcloud_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            if data.shape[1] >= 3:
                translation = np.array([0.0, 0.0, 0.0])
                points = data[:, :3] + translation  # Adding negative values = subtraction
                logger.info(f"Translated point cloud by {translation} meters")

                if hasattr(cfg, "max_distance_from_origin") and cfg.max_distance_from_origin != -1:
                    # Always calculate distance from the origin based on translated points
                    # Use only the first 3 columns (X, Y, Z) for distance calculation
                    distances = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))

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

        # Expecting preprocessed format: x,y,z, dist, nx,ny,nz, t1x,t1y,t1z, t2x,t2y,t2z (13 cols)
        if data.shape[1] == 13:
            # Extract tangents after downsampling and translation
            tangents1_full = data[:, 7:10]
            tangents2_full = data[:, 10:13]
            # Filter tangents based on distance filtering applied to points
            if hasattr(cfg, "max_distance_from_origin") and cfg.max_distance_from_origin != -1:
                tangents1 = tangents1_full[distances <= cfg.max_distance_from_origin]
                tangents2 = tangents2_full[distances <= cfg.max_distance_from_origin]
            else:
                tangents1 = tangents1_full
                tangents2 = tangents2_full

            logger.info(f"Loaded preprocessed point cloud with {len(points)} points.")
            logger.info(f"  Points shape: {points.shape}")
            logger.info(f"  Tangents1 shape: {tangents1.shape}")
            logger.info(f"  Tangents2 shape: {tangents2.shape}")

        elif data.shape[1] == 7:
            logger.warning(
                f"Loaded point cloud from {cfg.source_pointcloud_path} has only 7 columns. Calculating tangents and overwriting the file..."
            )
            # Calculate tangents based on the full (downsampled) point cloud normals
            normals_full = data[:, 4:7]  # Assuming columns 4, 5, 6 are nx, ny, nz
            from src.utils.preprocess_pointcloud import get_tangent_vectors

            tangents1_full, tangents2_full = get_tangent_vectors(normals_full)
            logger.info(
                f"Calculated tangents for full point cloud. t1: {tangents1_full.shape}, t2: {tangents2_full.shape}"
            )

            # Combine original full data with full tangents
            output_data_full = np.hstack((data, tangents1_full, tangents2_full))

            # Overwrite the original file with the full preprocessed data
            logger.info(
                f"Overwriting {cfg.source_pointcloud_path} with preprocessed data (13 columns)..."
            )
            try:
                # Need to reload the original *un-downsampled* data to save it correctly if downsampling happened
                # Or, more simply, save the *downsampled* 13-column data back. Let's do the latter.
                with open(cfg.source_pointcloud_path, "wb") as f_out:
                    # Save the downsampled but now 13-column data
                    pickle.dump(output_data_full, f_out)
                logger.info(
                    f"Successfully overwrote {cfg.source_pointcloud_path} with downsampled, 13-column data."
                )
            except Exception as e:
                logger.error(f"Failed to overwrite {cfg.source_pointcloud_path}: {e}")
                # Continue with calculated tangents, but log the error

            # Now filter the calculated tangents based on distance, same as points were filtered
            if hasattr(cfg, "max_distance_from_origin") and cfg.max_distance_from_origin != -1:
                tangents1 = tangents1_full[distances <= cfg.max_distance_from_origin]
                tangents2 = tangents2_full[distances <= cfg.max_distance_from_origin]
            else:
                tangents1 = tangents1_full
                tangents2 = tangents2_full
            logger.info(f"Filtered tangents. t1: {tangents1.shape}, t2: {tangents2.shape}")
        else:
            raise ValueError(
                f"Loaded point cloud data has unexpected shape: {data.shape}. Expected 7 or 13 columns."
            )

        logger.info(f"Using real data point cloud with {len(points)} points, shape: {points.shape}")
    else:
        logger.info("Finished point cloud processing.")
        points = create_test_pointcloud(cfg)
        # Calculate tangents for the test point cloud (assuming default normals)
        logger.info(
            "Calculating tangents for generated point cloud (assuming default normals [0, 0, 1])."
        )
        temp_normals = np.zeros_like(points)
        temp_normals[:, 2] = 1.0
        # Need the tangent calculation function from its new location
        from src.utils.preprocess_pointcloud import get_tangent_vectors

        tangents1, tangents2 = get_tangent_vectors(temp_normals)
        logger.info(
            f"Created cube environment with {len(points)} points, cube size: {cfg.room_size}m"
        )
        logger.info(f"Generated points shape: {points.shape}")
        logger.info(f"Generated tangents1 shape: {tangents1.shape}")
        logger.info(f"Generated tangents2 shape: {tangents2.shape}")

    logger.info(f"Final points shape: {points.shape}")
    # Ensure tangents are defined if vector model is selected
    # Note: 'tangents1' and 'tangents2' are only defined within the 'if cfg.use_source_pointcloud:' block
    # or the 'else:' block for generated points. Need to handle this.
    if cfg.use_vector_model:
        if "tangents1" not in locals() or "tangents2" not in locals():
            # This case implies use_source_pointcloud=False and vector model=True,
            # but tangents were not generated in the 'else' block above.
            # Let's calculate them here based on default normals for the generated cloud.
            if not cfg.use_source_pointcloud:
                logger.warning(
                    "Calculating tangents for generated point cloud as vector model is True."
                )
                temp_normals = np.zeros_like(points)
                temp_normals[:, 2] = 1.0
                from src.utils.preprocess_pointcloud import get_tangent_vectors

                tangents1, tangents2 = get_tangent_vectors(temp_normals)
            else:
                # This means use_source_pointcloud=True but tangents weren't loaded/calculated.
                raise RuntimeError(
                    "Vector model selected but tangents could not be loaded or calculated from source file."
                )

    # Downsample point cloud if requested
    if cfg.pointcloud_downsample > 1:
        original_point_count = len(points)
        points = downsample_pointcloud(points, cfg.pointcloud_downsample)
        logger.info(
            f"Downsampled point cloud from {original_point_count} to {len(points)} points "
            f"(factor: {cfg.pointcloud_downsample})"
        )
        # Downsample tangents if they exist and vector model is used
        if cfg.use_vector_model and "tangents1" in locals() and "tangents2" in locals():
            tangents1 = downsample_pointcloud(tangents1, cfg.pointcloud_downsample)
            tangents2 = downsample_pointcloud(tangents2, cfg.pointcloud_downsample)
            logger.info("Downsampled corresponding tangent vectors.")

    # Get measurement direction from config
    try:
        # Assuming cfg.measurement_direction = [x, y, z]
        measurement_direction = np.array(cfg.measurement_direction, dtype=float)
        if measurement_direction.shape != (3,):
            raise ValueError("measurement_direction must be a list/array of 3 numbers.")
        # Normalize
        norm_meas = np.linalg.norm(measurement_direction)
        if norm_meas < 1e-9:
            raise ValueError("measurement_direction cannot be a zero vector.")
        measurement_direction /= norm_meas
        logger.info(f"Using measurement direction: {measurement_direction}")
    except KeyError:
        logger.error("Config missing 'measurement_direction'. Using default [0, 1, 0].")
        measurement_direction = np.array([0.0, 1.0, 0.0])
    except Exception as e:
        logger.error(f"Error processing measurement_direction: {e}. Using default [0, 1, 0].")
        measurement_direction = np.array([0.0, 1.0, 0.0])

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
        wavelength = 299792458 / (measurement_data["frequency"] / 1e9) / 1000  # meters
        k = 2 * np.pi / wavelength
        logger.info(f"Using measurement frequency: {measurement_data['frequency']/1e9:.2f} GHz")
        logger.info(f"Calculated wave number k: {k:.2f}, wavelength: {wavelength:.4f} m")
    else:
        k = 2 * np.pi / cfg.wavelength
        logger.info(f"Using config wave number k: {k:.2f}, wavelength: {cfg.wavelength:.4f} m")

    logger.info("Creating channel matrix...")
    H = create_channel_matrix(
        points=points,
        measurement_plane=measurement_plane,
        k=k,
        use_vector_model=cfg.use_vector_model,
        # Pass tangents/direction only if using vector model
        tangents1=tangents1 if cfg.use_vector_model else None,
        tangents2=tangents2 if cfg.use_vector_model else None,
        measurement_direction=measurement_direction if cfg.use_vector_model else None,
    )

    initial_field_values = None
    if hasattr(cfg, "uniform_current_init") and cfg.uniform_current_init:
        logger.info(
            f"Initializing currents with uniform amplitude {cfg.initial_current_amplitude} "
            f"and random phase"
        )
        N_c = points.shape[0]
        random_phases = np.random.uniform(0, 2 * np.pi, N_c)
        base_currents = cfg.initial_current_amplitude * np.exp(1j * random_phases)

        if cfg.use_vector_model:
            # Initialize currents for 2 components per point, assign to first component
            initial_currents = np.zeros(2 * N_c, dtype=complex)
            initial_currents[0::2] = base_currents
            logger.info(f"Initialized vector currents shape: {initial_currents.shape}")
        else:
            # Scalar model uses N_c currents
            initial_currents = base_currents
            logger.info(f"Initialized scalar currents shape: {initial_currents.shape}")

        # Calculate initial field using H (N_m x 2N_c) and initial_currents (2N_c,)
        initial_field = H @ initial_currents

        from src.utils.phase_retrieval_utils import apply_magnitude_constraint

        initial_field_values = apply_magnitude_constraint(initial_field, measured_magnitude)

        logger.info(
            f"Initialized field values from uniform currents "
            f"with shape {initial_field_values.shape}"
        )

    # Run holographic phase retrieval
    logger.info("Starting holographic phase retrieval...")
    # Pass necessary info to HPR. Assume HPR handles H shape internally.
    # Create default train_plane_info for the single plane
    num_measurement_points = H.shape[0]
    train_plane_info = [("measured_plane", 0, num_measurement_points)]

    hpr_result = holographic_phase_retrieval(
        cfg=cfg,
        channel_matrix=H,
        measured_magnitude=measured_magnitude,
        train_plane_info=train_plane_info,  # Added required argument
        test_planes_data=None,  # Added optional argument
        initial_field_values=initial_field_values,
        # output_dir is not passed in legacy, HPR handles None
    )
    logger.info("Finished holographic phase retrieval.")

    if cfg.return_history:
        # Unpack results based on the new return signature (even if full_history is None)
        cluster_coefficients, full_history, stats = hpr_result
        # Extract legacy history variables if needed (and if history was returned)
        coefficient_history = (
            np.array([h["coefficients"] for h in full_history]) if full_history else None
        )
        # Field history is more complex now (segmented/test), maybe skip legacy field animation?
        # For now, let's try reconstructing the full field history from segments if possible
        # This assumes 'measured_plane' was the name used in train_plane_info
        field_history = (
            np.array([h["train_field_segments"]["measured_plane"] for h in full_history])
            if full_history and "measured_plane" in full_history[0]["train_field_segments"]
            else None
        )
    else:
        cluster_coefficients, stats = hpr_result

    logger.info("Perturbation statistics:")
    logger.info(f"  Total iterations: {stats['iterations']}")
    logger.info(f"  Final Overall RMSE: {stats['final_overall_rmse']:.6f}")  # Use correct key
    logger.info(f"  Best Overall RMSE: {stats['best_overall_rmse']:.6f}")  # Use correct key
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

    # H is (N_m, 2*N_c), cluster_coefficients is (2*N_c,)
    reconstructed_field = reconstruct_field(H, cluster_coefficients)  # Shape (N_m,)

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
        "config": cfg,
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
            points,  # points
            cluster_coefficients,  # currents (shape depends on model)
            measurement_plane,  # measurement_plane
            true_field_2d,  # true_field_2d
            measured_magnitude_2d,  # measured_magnitude_2d
            reconstructed_field_2d,  # reconstructed_field_2d
            rmse,  # rmse
            corr,  # correlation
            show_plot=cfg.show_plot,
            output_dir=output_dir,
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
                coefficient_history,  # Shape depends on model
                field_history if field_history is not None else [],  # Pass empty list if None
                resolution_param,
                measurement_plane,  # Added measurement_plane
                show_plot=cfg.show_plot,
                output_file=output_dir,
                animation_filename=os.path.join(output_dir, "gs_animation.gif"),
                frame_skip=1,  # Lower frame skip for smoother animation
                perturbation_iterations=stats.get("perturbation_iterations", []),
                convergence_threshold=cfg.convergence_threshold,
                measured_magnitude=measured_magnitude,
            )

            logger.info("Creating enhanced 4-panel animation...")
            visualize_current_and_field_history(
                points,
                coefficient_history,  # Shape depends on model
                field_history if field_history is not None else [],  # Pass empty list if None
                true_field,
                resolution_param,
                measurement_plane,  # Added measurement_plane
                show_plot=cfg.show_plot,
                output_file=output_dir,
                animation_filename=os.path.join(output_dir, "current_field_animation.gif"),
                frame_skip=1,  # Lower frame skip for smoother animation
            )

    logger.info("Finished visualization.")


if __name__ == "__main__":
    main()
