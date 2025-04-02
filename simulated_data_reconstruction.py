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
from src.utils.field_utils import (  # These now expect flag + conditional args
    compute_fields,
    reconstruct_field,
)
from src.visualization.field_plots import visualize_fields
from src.visualization.history_plots import (
    visualize_current_and_field_history,
    visualize_iteration_history,
)
from src.visualization.vector_plots import visualize_vectors  # Import the new function

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

    # Set random seed for reproducibility
    seed = cfg.random_seed
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

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
            logger.warning(
                f"Loaded point cloud from {cfg.source_pointcloud_path} has only 7 columns. Calculating tangents and overwriting the file..."
            )
            points = data[:, :3]
            normals = data[:, 4:7]  # Assuming columns 4, 5, 6 are nx, ny, nz
            # Need the tangent calculation function
            from src.utils.preprocess_pointcloud import get_tangent_vectors

            tangents1, tangents2 = get_tangent_vectors(normals)
            logger.info(f"Calculated tangents. t1: {tangents1.shape}, t2: {tangents2.shape}")

            # Combine and save back to the original file
            output_data = np.hstack((data, tangents1, tangents2))
            logger.info(
                f"Overwriting {cfg.source_pointcloud_path} with preprocessed data (13 columns)..."
            )
            try:
                with open(cfg.source_pointcloud_path, "wb") as f_out:
                    pickle.dump(output_data, f_out)
                logger.info(f"Successfully overwrote {cfg.source_pointcloud_path}.")
            except Exception as e:
                logger.error(f"Failed to overwrite {cfg.source_pointcloud_path}: {e}")
                # Continue with calculated tangents, but log the error
        else:
            raise ValueError(
                f"Loaded point cloud data has unexpected shape: {data.shape}. Expected 7 or 13 columns."
            )

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

    logger.info(f"Initial point cloud shape: {points.shape}")
    points_true = points.copy()  # This is the ground truth geometry
    tangents1_true = tangents1.copy() if tangents1 is not None else None
    tangents2_true = tangents2.copy() if tangents2 is not None else None

    if cfg.get("perturb_points", False):  # Check if perturbation is enabled (default False)
        perturbation_factor = cfg.get(
            "perturbation_factor", 0.01
        )  # Get factor from config, default 0.01
        logger.info(
            f"Perturbing points with distance-scaled uniform noise (factor: {perturbation_factor})..."
        )
        # Calculate distance from origin for each true point
        distances = np.sqrt(np.sum(points_true**2, axis=1))
        # Generate uniform random perturbations [-1, 1] for each coordinate
        random_perturbations = np.random.uniform(-1, 1, size=points_true.shape)
        # Scale perturbations by the factor and the distance
        scaled_perturbations = random_perturbations * perturbation_factor * distances[:, np.newaxis]
        # Apply perturbation
        points_perturbed = points_true + scaled_perturbations
        # For simplicity, we assume tangents don't change significantly with small perturbations
        # If tangent recalculation is needed, it would go here.
        tangents1_perturbed = tangents1_true
        tangents2_perturbed = tangents2_true
        logger.info(f"Perturbed point cloud shape: {points_perturbed.shape}")
    else:
        logger.info("Point perturbation disabled.")
        points_perturbed = points_true
        tangents1_perturbed = tangents1_true
        tangents2_perturbed = tangents2_true

    # logger.info(f"Using normals shape: {normals.shape}") # No longer primary input
    # Initialize currents based on the selected model
    N_c = len(points)
    if cfg.use_vector_model:
        currents = np.zeros(2 * N_c, dtype=complex)  # Vector model: 2 components per point
        logger.info(f"Initializing vector currents array shape: {currents.shape}")
    else:
        currents = np.zeros(N_c, dtype=complex)  # Scalar model: 1 component per point
        logger.info(f"Initializing scalar currents array shape: {currents.shape}")

    num_sources = min(cfg.num_sources, N_c)  # Number of points to activate
    source_indices = random.sample(range(N_c), num_sources)  # Indices of points (0 to N_c-1)

    amplitudes = np.random.lognormal(mean=0, sigma=cfg.amplitude_sigma, size=num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    logger.info(f"Assigning random currents to {num_sources} source points...")
    for i, point_idx in enumerate(source_indices):
        value = amplitudes[i] * np.exp(1j * phases[i])
        if cfg.use_vector_model:
            # Assign to the first component (index 2*point_idx) for vector model
            current_idx = 2 * point_idx
            currents[current_idx] = value
            # currents[current_idx + 1] = 0.0 # Keep second component zero
        else:
            # Assign directly to the point index for scalar model
            currents[point_idx] = value

    # Ensure at least one source if num_sources was 0 or less
    if num_sources <= 0 and N_c > 0:
        logger.warning("num_sources <= 0, activating the first component of the first point.")
        currents[0] = 1.0  # Activate first point (or first component if vector)

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

    logger.info("Creating channel matrix for ground truth field (H_true)...")
    H_true = create_channel_matrix(
        points=points_true,  # Use true points
        measurement_plane=measurement_plane,
        k=k,
        use_vector_model=cfg.use_vector_model,
        tangents1=tangents1_true if cfg.use_vector_model else None,
        tangents2=tangents2_true if cfg.use_vector_model else None,
        measurement_direction=measurement_direction if cfg.use_vector_model else None,
    )
    logger.info(f"H_true shape: {H_true.shape}")

    if cfg.get("perturb_points", False):
        logger.info(
            "Creating channel matrix for reconstruction (H_measured) using perturbed points..."
        )
        H_measured = create_channel_matrix(
            points=points_perturbed,  # Use perturbed points
            measurement_plane=measurement_plane,
            k=k,
            use_vector_model=cfg.use_vector_model,
            tangents1=tangents1_perturbed
            if cfg.use_vector_model
            else None,  # Use potentially perturbed tangents
            tangents2=tangents2_perturbed
            if cfg.use_vector_model
            else None,  # Use potentially perturbed tangents
            measurement_direction=measurement_direction if cfg.use_vector_model else None,
        )
        logger.info(f"H_measured shape: {H_measured.shape}")
    else:
        logger.info("Using H_true as H_measured (no perturbation).")
        H_measured = H_true  # If not perturbing, measured is same as true

    logger.info("Calculating ground truth field...")
    true_field = compute_fields(
        points=points_true,  # Use true points for ground truth field
        currents=currents,
        measurement_plane=measurement_plane,
        k=k,
        use_vector_model=cfg.use_vector_model,
        tangents1=tangents1_true if cfg.use_vector_model else None,  # Use true tangents
        tangents2=tangents2_true if cfg.use_vector_model else None,  # Use true tangents
        measurement_direction=measurement_direction if cfg.use_vector_model else None,
        channel_matrix=H_true,  # Use true channel matrix
    )

    true_field_2d = true_field.reshape(cfg.resolution, cfg.resolution)

    measured_magnitude = np.abs(true_field)  # Use magnitude from synthetic true field
    measured_magnitude_2d = measured_magnitude.reshape(cfg.resolution, cfg.resolution)

    logger.info("Running holographic phase retrieval...")

    # Pass necessary info to HPR. Assume HPR internally handles scalar/vector based on H shape.
    # Alternatively, HPR might need the flag explicitly. Assuming the former for now.
    # Create default train_plane_info for the single plane
    num_measurement_points = H_measured.shape[0]
    train_plane_info = [("simulated_plane", 0, num_measurement_points)]

    hpr_result = holographic_phase_retrieval(
        cfg=cfg,
        channel_matrix=H_measured,  # Use the (potentially perturbed) measured channel matrix
        measured_magnitude=measured_magnitude,  # Magnitude comes from the true field
        train_plane_info=train_plane_info,  # Added required argument
        test_planes_data=None,  # Added optional argument
        output_dir=output_dir,
    )

    if cfg.return_history:
        # Unpack results based on the new return signature (even if full_history is None)
        cluster_coefficients, full_history, stats = hpr_result
        # Extract legacy history variables if needed (and if history was returned)
        coefficient_history = (
            np.array([h["coefficients"] for h in full_history]) if full_history else None
        )
        # Field history is more complex now (segmented/test), maybe skip legacy field animation?
        # For now, let's try reconstructing the full field history from segments if possible
        # This assumes 'simulated_plane' was the name used in train_plane_info
        field_history = (
            np.array([h["train_field_segments"]["simulated_plane"] for h in full_history])
            if full_history and "simulated_plane" in full_history[0]["train_field_segments"]
            else None
        )
        logger.info(f"Coefficient history shape: {coefficient_history.shape}")
        logger.info(f"Field history shape: {field_history.shape}")

        logger.info("Perturbation statistics:")
        logger.info(f"  Total iterations: {stats['iterations']}")
        logger.info(f"  Final Overall RMSE: {stats['final_overall_rmse']:.6f}")  # Use correct key
        logger.info(f"  Best Overall RMSE: {stats['best_overall_rmse']:.6f}")  # Use correct key
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
                points_perturbed,  # Visualize based on the geometry used for reconstruction
                H_measured,  # Use the measured channel matrix
                coefficient_history,
                field_history if field_history is not None else [],  # Pass empty list if None
                cfg.resolution,
                measurement_plane,
                show_plot=(cfg.show_plot and not cfg.no_plot),
                output_file=None,
                animation_filename=os.path.join(output_dir, "gs_animation.gif"),
                frame_skip=10,  # Increased frame skip for faster animation
                perturbation_iterations=stats.get("perturbation_iterations", []),
                # Get stats from HPR result
                restart_iterations=[],  # No restarts in vanilla implementation
                convergence_threshold=cfg.convergence_threshold,
                measured_magnitude=measured_magnitude,
                # Pass the measured magnitude for true error calculation
            )

            logger.info("Creating enhanced 4-panel animation...")
            visualize_current_and_field_history(
                points_perturbed,  # Visualize currents relative to the perturbed points
                coefficient_history,  # Coefficients are relative to H_measured basis
                field_history if field_history is not None else [],  # Pass empty list if None
                true_field,  # Pass the synthetic true field for comparison
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
        logger.info(f"  Final Overall RMSE: {stats['final_overall_rmse']:.6f}")  # Use correct key
        logger.info(f"  Best Overall RMSE: {stats['best_overall_rmse']:.6f}")  # Use correct key
        logger.info(f"  Perturbations applied: {stats['num_perturbations']}")

    reconstructed_field = reconstruct_field(
        H_measured, cluster_coefficients
    )  # Reconstruct using H_measured
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
            points_perturbed,  # Show points used in reconstruction
            currents,  # Ground truth currents (defined on true points, but visualize magnitude)
            measurement_plane,
            true_field_2d,
            measured_magnitude_2d,
            reconstructed_field_2d,
            rmse,  # Calculated RMSE
            corr,  # Calculated Correlation
            show_plot=cfg.show_plot,
            output_dir=output_dir,
        )
    else:
        logger.info("Plot generation disabled via no_plot flag")


if __name__ == "__main__":
    main()  # Hydra automatically passes the config object (cfg)
