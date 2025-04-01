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
from src.utils.field_utils import compute_fields, reconstruct_field
from src.visualization.field_plots import visualize_fields
from src.visualization.history_plots import (
    visualize_current_and_field_history,
    visualize_iteration_history,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        points = data[:, :3]
        logger.info(f"Loaded point cloud with {len(points)} points, shape: {points.shape}")
    else:
        logger.info("Creating test point cloud...")
        points = create_test_pointcloud(cfg)  # Pass DictConfig, assuming compatibility

    logger.info(f"Using point cloud shape: {points.shape}")

    currents = np.zeros(len(points), dtype=complex)

    num_sources = min(cfg.num_sources, len(currents))
    source_indices = random.sample(range(len(currents)), num_sources)

    amplitudes = np.random.lognormal(mean=0, sigma=cfg.amplitude_sigma, size=num_sources)

    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    for i, idx in enumerate(source_indices):
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])
    else:
        currents[0] = 1.0

    logger.info("Creating measurement plane...")
    x = np.linspace(-cfg.plane_size / 2, cfg.plane_size / 2, cfg.resolution)
    y = np.linspace(-cfg.plane_size / 2, cfg.plane_size / 2, cfg.resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    k = 2 * np.pi / cfg.wavelength

    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, measurement_plane, k)

    logger.info("Calculating ground truth field...")
    true_field = compute_fields(points, currents, measurement_plane, k, H)

    true_field_2d = true_field.reshape(cfg.resolution, cfg.resolution)

    measured_magnitude = np.abs(true_field)  # Use magnitude from synthetic true field
    measured_magnitude_2d = measured_magnitude.reshape(cfg.resolution, cfg.resolution)

    logger.info("Running holographic phase retrieval with perturbation strategies...")

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
            currents,
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
