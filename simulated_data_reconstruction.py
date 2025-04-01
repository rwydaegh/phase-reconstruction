import logging
import os
import os.path
import pickle
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud
from src.utils.field_utils import compute_fields, reconstruct_field # These now expect tangents1, tangents2, meas_dir, and 2*N_c currents
from src.visualization.field_plots import visualize_fields
from src.visualization.history_plots import (
    visualize_current_and_field_history,
    visualize_iteration_history,
)
from src.visualization.vector_plots import visualize_vectors # Import the new function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Removed internal _get_cube_normals function



@hydra.main(config_path="conf", config_name="simulated_data", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main demonstration of field reconstruction using holographic phase retrieval."""
    output_dir = os.getcwd() + "/plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"use_source_pointcloud: {cfg.use_source_pointcloud}")
    logger.info(f"Hydra output directory: {output_dir}")


    if cfg.use_source_pointcloud:
        logger.info(f"Loading source point cloud from {cfg.source_pointcloud_path}...")
        with open(cfg.source_pointcloud_path, "rb") as f:
            data = pickle.load(f)

        # Expecting preprocessed format: x,y,z, dist, nx,ny,nz, t1x,t1y,t1z, t2x,t2y,t2z (13 cols)
        if data.shape[1] == 13:
            points = data[:, :3]
            # normals = data[:, 4:7] # Optional: if needed elsewhere
            tangents1 = data[:, 7:10]
            tangents2 = data[:, 10:13]
            logger.info(f"Loaded preprocessed point cloud with {len(points)} points.")
            logger.info(f"  Points shape: {points.shape}")
            logger.info(f"  Tangents1 shape: {tangents1.shape}")
            logger.info(f"  Tangents2 shape: {tangents2.shape}")
        elif data.shape[1] == 7:
            logger.warning(f"Loaded point cloud from {cfg.source_pointcloud_path} has only 7 columns. Calculating tangents and overwriting the file...")
            points = data[:, :3]
            normals = data[:, 4:7] # Assuming columns 4, 5, 6 are nx, ny, nz
            # Need the tangent calculation function
            from src.utils.preprocess_pointcloud import get_tangent_vectors
            tangents1, tangents2 = get_tangent_vectors(normals)
            logger.info(f"Calculated tangents. t1: {tangents1.shape}, t2: {tangents2.shape}")

            # Combine and save back to the original file
            output_data = np.hstack((data, tangents1, tangents2))
            logger.info(f"Overwriting {cfg.source_pointcloud_path} with preprocessed data (13 columns)...")
            try:
                with open(cfg.source_pointcloud_path, "wb") as f_out:
                    pickle.dump(output_data, f_out)
                logger.info(f"Successfully overwrote {cfg.source_pointcloud_path}.")
            except Exception as e:
                logger.error(f"Failed to overwrite {cfg.source_pointcloud_path}: {e}")
                # Continue with calculated tangents, but log the error
        else:
             raise ValueError(f"Loaded point cloud data has unexpected shape: {data.shape}. Expected 7 or 13 columns.")

    else:
        logger.info("Creating test point cloud...")
        points = create_test_pointcloud(cfg)  # Pass DictConfig, assuming compatibility
        # Get normals for the generated cube points using the utility function
        # Get normals for the generated cube points using the utility function
        from src.utils.geometry_utils import get_cube_normals
        # get_cube_normals now returns all points and their calculated inward normals
        # (normals might be zero for points not clearly on a single face)
        points, temp_normals = get_cube_normals(points, cfg.room_size)
        logger.info(f"Calculated normals for {points.shape[0]} points.")
        # No filtering needed here anymore
        logger.info("Calculating tangents for generated cube point cloud (using inward normals).")
        # Need the tangent calculation function from its new location
        from src.utils.preprocess_pointcloud import get_tangent_vectors
        tangents1, tangents2 = get_tangent_vectors(temp_normals)
        logger.info(f"Generated tangents1 shape: {tangents1.shape}")
        logger.info(f"Generated tangents2 shape: {tangents2.shape}")
        # Visualize the calculated vectors for verification (Moved here)
        if not cfg.no_plot: # Check if plotting is enabled
             visualize_vectors(
                  points=points,
                  normals=temp_normals, # Pass the calculated normals
                  tangents1=tangents1,
                  tangents2=tangents2,
                  num_vectors=50, # Plot vectors for ~50 random points
                  scale=0.1 * cfg.room_size, # Scale arrows relative to room size
                  output_dir=output_dir,
                  show_plot=cfg.show_plot,
                  filename="generated_cube_vectors.png"
             )


    logger.info(f"Using point cloud shape: {points.shape}")
    # logger.info(f"Using normals shape: {normals.shape}") # No longer primary input

    # Currents vector now needs 2 entries per point
    N_c = len(points)
    currents = np.zeros(2 * N_c, dtype=complex)

    num_sources = min(cfg.num_sources, N_c) # Number of points to activate
    source_indices = random.sample(range(N_c), num_sources) # Indices of points (0 to N_c-1)

    amplitudes = np.random.lognormal(mean=0, sigma=cfg.amplitude_sigma, size=num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    logger.info(f"Assigning random currents to {num_sources} source points (first tangent component)...")
    for i, point_idx in enumerate(source_indices):
        # Assign to the first component (index 2*point_idx)
        current_idx = 2 * point_idx
        currents[current_idx] = amplitudes[i] * np.exp(1j * phases[i])
        # Keep the second component (index 2*point_idx + 1) zero for simplicity
        # currents[current_idx + 1] = 0.0 # Already zero initialized

    # Ensure at least one source if num_sources was 0 or less
    if num_sources <= 0 and N_c > 0:
        logger.warning("num_sources <= 0, activating the first component of the first point.")
        currents[0] = 1.0 # Activate first component of first point

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


    logger.info("Creating measurement plane...")
    x = np.linspace(-cfg.plane_size / 2, cfg.plane_size / 2, cfg.resolution)
    y = np.linspace(-cfg.plane_size / 2, cfg.plane_size / 2, cfg.resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    k = 2 * np.pi / cfg.wavelength

    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, tangents1, tangents2, measurement_plane, measurement_direction, k)

    logger.info("Calculating ground truth field...")
    true_field = compute_fields(points, tangents1, tangents2, currents, measurement_plane, measurement_direction, k, H)

    true_field_2d = true_field.reshape(cfg.resolution, cfg.resolution)

    measured_magnitude = np.abs(true_field)  # Use magnitude from synthetic true field
    measured_magnitude_2d = measured_magnitude.reshape(cfg.resolution, cfg.resolution)

    logger.info("Running holographic phase retrieval with perturbation strategies...")

    # Note: holographic_phase_retrieval now receives H with shape (N_m, 2*N_c)
    # It needs to solve for cluster_coefficients of shape (2*N_c,)
    # We assume the implementation in gerchberg_saxton.py handles this correctly.
    hpr_result = holographic_phase_retrieval(cfg, H, measured_magnitude, output_dir=output_dir) # Pass output_dir

    if cfg.return_history:
        cluster_coefficients, coefficient_history, field_history, stats = hpr_result
        logger.info(f"Coefficient history shape: {coefficient_history.shape}")
        logger.info(f"Field history shape: {field_history.shape}")

        logger.info("Perturbation statistics:")
        logger.info(f"  Total iterations: {stats['iterations']}")
        logger.info(f"  Final RMSE: {stats['final_rmse']:.6f}")
        logger.info(f"  Best RMSE: {stats['best_rmse']:.6f}")
        logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

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
            logger.info("Creating GS iteration animation...")
            visualize_iteration_history(
                points,
                H,
                coefficient_history,
                field_history,
                cfg.resolution,
                measurement_plane,
                show_plot=(cfg.show_plot and not cfg.no_plot),
                output_file=None,
                animation_filename=os.path.join(output_dir, "gs_animation.gif"),
                frame_skip=10, # Increased frame skip for faster animation
                perturbation_iterations=stats.get("perturbation_iterations", []),
                # Get stats from HPR result
                restart_iterations=[], # No restarts in vanilla implementation
                convergence_threshold=cfg.convergence_threshold,
                measured_magnitude=measured_magnitude,
                # Pass the measured magnitude for true error calculation
            )

            logger.info("Creating enhanced 4-panel animation...")
            visualize_current_and_field_history(
                points,
                coefficient_history,
                field_history,
                true_field, # Pass the synthetic true field for comparison
                cfg.resolution,
                measurement_plane,
                show_plot=(cfg.show_plot and not cfg.no_plot),
                output_file=None,
                animation_filename=os.path.join(output_dir, "current_field_animation.gif"),
            )
        else:
            logger.info("Animation generation disabled via no_anim flag")
    else:
        cluster_coefficients, stats = hpr_result

        logger.info("Perturbation statistics:")
        logger.info(f"  Total iterations: {stats['iterations']}")
        logger.info(f"  Final RMSE: {stats['final_rmse']:.6f}")
        logger.info(f"  Best RMSE: {stats['best_rmse']:.6f}")
        logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

    reconstructed_field = reconstruct_field(H, cluster_coefficients)
    reconstructed_field_2d = reconstructed_field.reshape(cfg.resolution, cfg.resolution)

    def normalized_rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2)) / (np.max(a) - np.min(a))

    def normalized_correlation(a, b):
        a_norm = (a - np.mean(a)) / np.std(a)
        b_norm = (b - np.mean(b)) / np.std(b)
        return np.correlate(a_norm.flatten(), b_norm.flatten())[0] / len(a_norm.flatten())

    rmse = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))

    logger.info("Reconstruction quality metrics:")
    logger.info(f"  Normalized RMSE: {rmse:.4f}")
    logger.info(f"  Correlation: {corr:.4f}")

    if not cfg.no_plot:
        logger.info("Generating final field comparison visualization...")
        visualize_fields(
            points,
            currents, # Note: This 'currents' is the ground truth (2*N_c,), might need adjustment in visualize_fields if it expects (N_c,)
            measurement_plane,
            true_field_2d,
            measured_magnitude_2d,
            reconstructed_field_2d,
            rmse,  # Calculated RMSE
            corr,  # Calculated Correlation
            show_plot=cfg.show_plot,
            output_dir = output_dir
        )
    else:
        logger.info("Plot generation disabled via no_plot flag")


if __name__ == "__main__":
    main()  # Hydra automatically passes the config object (cfg)
