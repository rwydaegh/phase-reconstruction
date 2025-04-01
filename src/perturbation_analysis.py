#!/usr/bin/env python3
"""
Script to analyze the effect of point perturbation on field reconstruction using Hydra configuration.
This produces multiple comparisons to ensure our findings are correct.
"""

import logging
import math
import os
import random
import sys
from typing import List  # Import List for type hinting

import hydra
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig, OmegaConf

# Use the correct import path
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix
from src.create_test_pointcloud import create_test_pointcloud
from src.utils.normalized_correlation import normalized_correlation
from src.utils.normalized_rmse import normalized_rmse
from src.visualization.utils import visualize_field  # Keep if needed, maybe remove later

logger = logging.getLogger(__name__)

# Removed SimulationConfig dataclass
# Removed parse_args function
# Removed setup_simulation function


def create_measurement_plane(cfg: DictConfig):
    """Create measurement plane grid based on Hydra config"""
    resolution = int(cfg.resolution)
    x = np.linspace(
        cfg.room_size / 2 - cfg.plane_size / 2,
        cfg.room_size / 2 + cfg.plane_size / 2,
        resolution,
    )
    y = np.linspace(
        cfg.room_size / 2 - cfg.plane_size / 2,
        cfg.room_size / 2 + cfg.plane_size / 2,
        resolution,
    )
    X, Y = np.meshgrid(x, y)
    # Place plane at z = room_size / 2
    return np.stack([X, Y, np.ones_like(X) * cfg.room_size / 2], axis=-1), x, y


def generate_currents(points: np.ndarray, cfg: DictConfig):
    """Generate current sources based on model type using Hydra config"""
    N_c = len(points)
    num_sources = min(cfg.num_sources, N_c)  # Number of points to activate
    # Use numpy's random generator directly after seeding
    source_indices = np.random.choice(range(N_c), num_sources, replace=False)

    amplitudes = np.random.lognormal(mean=0, sigma=cfg.amplitude_sigma, size=num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    if cfg.use_vector_model:
        currents = np.zeros(2 * N_c, dtype=complex)  # Shape (2*N_c,)
        # Assign to the first component (index 2*point_idx)
        for i, point_idx in enumerate(source_indices):
            current_idx = 2 * point_idx
            currents[current_idx] = amplitudes[i] * np.exp(1j * phases[i])
            # currents[current_idx + 1] = 0.0 # Second component remains zero
    else:
        currents = np.zeros(N_c, dtype=complex)  # Shape (N_c,)
        for i, point_idx in enumerate(source_indices):
            currents[point_idx] = amplitudes[i] * np.exp(1j * phases[i])

    # Return currents (shape depends on model), point indices, amplitudes, phases
    return currents, source_indices, amplitudes, phases


def compute_field(
    points: np.ndarray,
    currents: np.ndarray,
    measurement_plane: np.ndarray,
    k: float,
    cfg: DictConfig,
    tangents1=None,
    tangents2=None,
):
    """Compute field component from currents using appropriate model based on Hydra config"""
    measurement_direction = None
    if cfg.use_vector_model:
        # Ensure measurement_direction is valid if using vector model
        if not hasattr(cfg, "measurement_direction") or len(cfg.measurement_direction) != 3:
            raise ValueError(
                "measurement_direction with 3 elements is required in config for vector model."
            )
        measurement_direction = np.array(cfg.measurement_direction)
        norm_meas = np.linalg.norm(measurement_direction)
        if norm_meas < 1e-9:
            raise ValueError("Measurement direction cannot be zero vector.")
        measurement_direction /= norm_meas

    H = create_channel_matrix(
        points=points,
        measurement_plane=measurement_plane,
        k=k,
        use_vector_model=cfg.use_vector_model,  # Pass flag directly
        tangents1=tangents1,  # Pass tangents if vector model
        tangents2=tangents2,  # Pass tangents if vector model
        measurement_direction=measurement_direction,  # Will be None if scalar
    )

    # Check current shape compatibility
    expected_current_len = H.shape[1]
    if currents.shape[0] != expected_current_len:
        raise ValueError(
            f"Shape mismatch: H columns {H.shape[1]} != currents length {currents.shape[0]}. Model: {'vector' if cfg.use_vector_model else 'scalar'}"
        )

    field = H @ currents
    return field, H


def reconstruct_field_from_magnitude(H: np.ndarray, field_magnitude: np.ndarray, cfg: DictConfig):
    """Reconstruct field from magnitude using holographic phase retrieval with Hydra config"""
    # Pass the cfg object directly to HPR
    hpr_result = holographic_phase_retrieval(
        cfg=cfg,
        channel_matrix=H,
        measured_magnitude=field_magnitude,
        # Assuming no initial field guess or output dir needed here for analysis script
    )

    # Handle potential tuple return if history/stats are included
    if isinstance(hpr_result, tuple):
        # Check if history was returned based on cfg (though default is False here)
        if cfg.return_history:
            reconstructed_coefficients = hpr_result[0]  # Coeffs are first if history returned
        else:
            reconstructed_coefficients = hpr_result[0]  # Coeffs are first if only stats returned
    else:
        # Should not happen if HPR always returns at least coeffs and stats
        raise TypeError(
            f"Unexpected return type from holographic_phase_retrieval: {type(hpr_result)}"
        )

    # Check coefficient shape compatibility
    if reconstructed_coefficients.shape[0] != H.shape[1]:
        raise ValueError(
            f"Shape mismatch after HPR: H columns {H.shape[1]} != "
            f"reconstructed coefficients length {reconstructed_coefficients.shape[0]}"
        )

    # Reconstruct field using H and the retrieved coefficients
    reconstructed_field = H @ reconstructed_coefficients
    return reconstructed_field, reconstructed_coefficients


def evaluate_reconstruction(true_field, reconstructed_field, title):
    """Evaluate reconstruction using RMSE and correlation"""
    rmse_val = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr_val = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))
    logger.debug(f"{title} - RMSE: {rmse_val:.6f}, Correlation: {corr_val:.6f}")
    return rmse_val, corr_val


def create_focused_perturbation_plots(all_results, perturbation_factors, cfg: DictConfig):
    """Create focused plots showing the relationship between perturbation amount
    and RMSE/correlation, using Hydra config for output."""

    output_dir = os.getcwd()  # Hydra sets the working directory to the output dir
    logger.info(f"Saving plots to Hydra output directory: {output_dir}")

    # Extract metrics
    rmse_real = [res["perturbed_forward_metrics"]["rmse"] for res in all_results]
    corr_real = [res["perturbed_forward_metrics"]["correlation"] for res in all_results]

    rmse_overfit = [res["perturbed_overfitting_metrics"]["rmse"] for res in all_results]
    corr_overfit = [res["perturbed_overfitting_metrics"]["correlation"] for res in all_results]

    rmse_self = [res["perturbed_vs_perturbed_metrics"]["rmse"] for res in all_results]
    corr_self = [res["perturbed_vs_perturbed_metrics"]["correlation"] for res in all_results]

    rmse_orig = [res["original_vs_original_metrics"]["rmse"] for res in all_results]
    corr_orig = [res["original_vs_original_metrics"]["correlation"] for res in all_results]

    # --- Plotting Logic (largely unchanged, uses output_dir and cfg.show_plots) ---

    # 1. Plot RMSE vs perturbation factor
    plt.figure(figsize=(10, 6))
    plt.plot(
        perturbation_factors,
        rmse_real,
        "o-",
        color="red",
        linewidth=2,
        label="Realistic: Perturbed geometry with true currents",
    )
    plt.plot(
        perturbation_factors,
        rmse_overfit,
        "x-",
        color="purple",
        linewidth=2,
        label="Overfitting: Perturbed geometry with optimized currents",
    )
    plt.plot(
        perturbation_factors,
        rmse_self,
        "s-",
        color="blue",
        linewidth=2,
        label="Self-reconstruction with perturbed geometry",
    )
    plt.plot(
        perturbation_factors,
        rmse_orig,
        "^-",
        color="green",
        linewidth=2,
        label="Original reconstruction",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Perturbation Factor", fontsize=12)
    plt.ylabel("Normalized RMSE", fontsize=12)
    plt.title("RMSE vs Perturbation Factor", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    rmse_filename = os.path.join(output_dir, "rmse_vs_perturbation.png")
    plt.savefig(rmse_filename, dpi=300)
    logger.info(f"Saved RMSE vs perturbation plot to {rmse_filename}")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()

    # 2. Plot correlation vs perturbation factor
    plt.figure(figsize=(10, 6))
    plt.plot(
        perturbation_factors,
        corr_real,
        "o-",
        color="red",
        linewidth=2,
        label="Realistic: Perturbed geometry with true currents",
    )
    plt.plot(
        perturbation_factors,
        corr_overfit,
        "x-",
        color="purple",
        linewidth=2,
        label="Overfitting: Perturbed geometry with optimized currents",
    )
    plt.plot(
        perturbation_factors,
        corr_self,
        "s-",
        color="blue",
        linewidth=2,
        label="Self-reconstruction with perturbed geometry",
    )
    plt.plot(
        perturbation_factors,
        corr_orig,
        "^-",
        color="green",
        linewidth=2,
        label="Original reconstruction",
    )
    plt.xscale("log")
    plt.xlabel("Perturbation Factor", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.title("Correlation vs Perturbation Factor", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    corr_filename = os.path.join(output_dir, "correlation_vs_perturbation.png")
    plt.savefig(corr_filename, dpi=300)
    logger.info(f"Saved correlation vs perturbation plot to {corr_filename}")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()

    # 3. Combined plot showing comparison between realistic and overfitting approaches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(
        perturbation_factors,
        rmse_real,
        "o-",
        color="red",
        linewidth=2,
        label="Realistic (fixed currents)",
    )
    ax1.plot(
        perturbation_factors,
        rmse_overfit,
        "x-",
        color="purple",
        linewidth=2,
        label="Overfitting (optimized currents)",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Perturbation Factor", fontsize=12)
    ax1.set_ylabel("Normalized RMSE (log scale)", fontsize=12)
    ax1.set_title("RMSE: Realistic vs Overfitting", fontsize=14)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    ax2.plot(
        perturbation_factors,
        corr_real,
        "o-",
        color="red",
        linewidth=2,
        label="Realistic (fixed currents)",
    )
    ax2.plot(
        perturbation_factors,
        corr_overfit,
        "x-",
        color="purple",
        linewidth=2,
        label="Overfitting (optimized currents)",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Perturbation Factor", fontsize=12)
    ax2.set_ylabel("Correlation", fontsize=12)
    ax2.set_title("Correlation: Realistic vs Overfitting", fontsize=14)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    comparison_filename = os.path.join(output_dir, "realistic_vs_overfitting.png")
    plt.savefig(comparison_filename, dpi=300)
    logger.info(f"Saved comparison plot to {comparison_filename}")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()

    # 4. Create a table of numerical values
    table_data = []
    for i, factor in enumerate(perturbation_factors):
        table_data.append(
            [
                factor,
                rmse_real[i],
                corr_real[i],
                rmse_overfit[i],
                corr_overfit[i],
                rmse_self[i],
                corr_self[i],
                rmse_orig[i],
                corr_orig[i],
            ]
        )
    fig, ax = plt.subplots(figsize=(16, len(perturbation_factors) + 2))
    ax.axis("off")
    column_labels = [
        "Perturb. Factor",
        "RMSE (Realistic)",
        "Corr (Realistic)",
        "RMSE (Overfitting)",
        "Corr (Overfitting)",
        "RMSE (Self-recon)",
        "Corr (Self-recon)",
        "RMSE (Original)",
        "Corr (Original)",
    ]
    table = ax.table(
        cellText=[[f"{val:.6f}" for val in row] for row in table_data],
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    for (i, _), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")
    plt.title("Numerical Results for Perturbation Effect Analysis", fontsize=14)
    plt.tight_layout()
    table_filename = os.path.join(output_dir, "perturbation_results_table.png")
    plt.savefig(table_filename, dpi=300)
    logger.info(f"Saved results table to {table_filename}")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()


@hydra.main(config_path="../conf", config_name="perturbation_analysis", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function to analyze perturbation effect using Hydra config"""
    output_dir = os.getcwd()  # Hydra sets cwd to the output directory
    logger.info(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        perturbation_factors = [float(f) for f in cfg.perturbation_factors.split(",")]
    except ValueError:
        logger.error("Error: Perturbation factors must be comma-separated numbers in config")
        return

    # Set random seed
    seed = cfg.random_seed
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

    # Calculate k from wavelength
    k = 2 * math.pi / cfg.wavelength

    all_results = []
    original_points = None
    original_currents = None
    original_source_indices = None
    original_tangents1 = None
    original_tangents2 = None

    for factor in perturbation_factors:
        logger.info(f"\n--- Testing perturbation factor {factor} ---")

        # Create a temporary config for create_test_pointcloud if needed
        # Or modify create_test_pointcloud to accept cfg directly
        # Assuming create_test_pointcloud now accepts cfg
        temp_cfg_for_pointcloud = cfg.copy()
        temp_cfg_for_pointcloud.perturb_points = factor > 0
        temp_cfg_for_pointcloud.perturbation_factor = factor if factor > 0 else 0.0

        perturbed_points = create_test_pointcloud(temp_cfg_for_pointcloud)

        # Calculate tangents for perturbed points (assuming default normals)
        temp_perturbed_normals = np.zeros_like(perturbed_points)
        temp_perturbed_normals[:, 2] = 1.0
        try:
            from src.utils.preprocess_pointcloud import get_tangent_vectors
        except ImportError:
            utils_dir = os.path.join(os.path.dirname(__file__), "utils")
            if utils_dir not in sys.path:
                sys.path.append(utils_dir)
            from preprocess_pointcloud import get_tangent_vectors
        perturbed_tangents1, perturbed_tangents2 = get_tangent_vectors(temp_perturbed_normals)

        if original_points is None:
            # Create original points (perturbation factor 0)
            temp_cfg_original = cfg.copy()
            temp_cfg_original.perturb_points = False
            temp_cfg_original.perturbation_factor = 0.0
            original_points = create_test_pointcloud(temp_cfg_original)

            # Calculate tangents for original points
            temp_original_normals = np.zeros_like(original_points)
            temp_original_normals[:, 2] = 1.0
            original_tangents1, original_tangents2 = get_tangent_vectors(temp_original_normals)

            # Generate original currents (shape depends on model)
            original_currents, original_source_indices, _, _ = generate_currents(
                original_points, cfg
            )

        # Create perturbed currents (shape depends on model) using same source indices/values
        if cfg.use_vector_model:
            perturbed_currents = np.zeros(2 * len(perturbed_points), dtype=complex)
            for _, point_idx in enumerate(original_source_indices):
                original_current_idx = 2 * point_idx
                if original_current_idx < len(original_currents):  # Check bounds
                    perturbed_current_idx = 2 * point_idx
                    if perturbed_current_idx < len(perturbed_currents):
                        perturbed_currents[perturbed_current_idx] = original_currents[
                            original_current_idx
                        ]
        else:
            perturbed_currents = np.zeros(len(perturbed_points), dtype=complex)
            for _, point_idx in enumerate(original_source_indices):
                if point_idx < len(original_currents) and point_idx < len(perturbed_currents):
                    perturbed_currents[point_idx] = original_currents[point_idx]

        measurement_plane, x, y = create_measurement_plane(cfg)

        # --- Compute fields ---
        original_true_field, H_original = compute_field(
            points=original_points,
            currents=original_currents,
            measurement_plane=measurement_plane,
            k=k,
            cfg=cfg,
            tangents1=original_tangents1,
            tangents2=original_tangents2,
        )
        perturbed_true_field, H_perturbed = compute_field(
            points=perturbed_points,
            currents=perturbed_currents,
            measurement_plane=measurement_plane,
            k=k,
            cfg=cfg,
            tangents1=perturbed_tangents1,
            tangents2=perturbed_tangents2,
        )

        original_true_field_2d = np.abs(original_true_field).reshape(cfg.resolution, cfg.resolution)
        perturbed_true_field_2d = np.abs(perturbed_true_field).reshape(
            cfg.resolution, cfg.resolution
        )

        # --- Scenario 1: Original reconstruction ---
        logger.info("SCENARIO 1: Original reconstruction")
        original_recon_field, _ = reconstruct_field_from_magnitude(
            H_original, np.abs(original_true_field), cfg
        )
        original_recon_field_2d = np.abs(original_recon_field).reshape(
            cfg.resolution, cfg.resolution
        )
        original_vs_original_metrics = {
            "rmse": normalized_rmse(np.abs(original_true_field), np.abs(original_recon_field)),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(original_recon_field)
            ),
        }
        logger.info(
            f"Original recon vs Original true - RMSE: {original_vs_original_metrics['rmse']:.6f}, Correlation: {original_vs_original_metrics['correlation']:.6f}"
        )

        # --- Scenario 2A: Realistic scenario - perturbed geometry, true currents ---
        logger.info("SCENARIO 2A: Realistic (perturbed geometry, true currents)")
        # Calculate forward field with perturbed H and original currents
        forward_field = H_perturbed @ original_currents
        perturbed_forward_field_2d = np.abs(forward_field).reshape(cfg.resolution, cfg.resolution)
        perturbed_forward_metrics = {
            "rmse": normalized_rmse(np.abs(original_true_field), np.abs(forward_field)),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(forward_field)
            ),
        }
        logger.info(
            f"Perturbed model (true currents) vs Original true - RMSE: {perturbed_forward_metrics['rmse']:.6f}, Correlation: {perturbed_forward_metrics['correlation']:.6f}"
        )

        # --- Scenario 2B: Overfitting scenario - perturbed geometry, optimized currents ---
        logger.info("SCENARIO 2B: Overfitting (perturbed geometry, optimized currents)")
        perturbed_overfitting_field, _ = reconstruct_field_from_magnitude(
            H_perturbed,
            np.abs(original_true_field),
            cfg,  # Reconstruct using original magnitude
        )
        perturbed_overfitting_field_2d = np.abs(perturbed_overfitting_field).reshape(
            cfg.resolution, cfg.resolution
        )
        perturbed_overfitting_metrics = {
            "rmse": normalized_rmse(
                np.abs(original_true_field), np.abs(perturbed_overfitting_field)
            ),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(perturbed_overfitting_field)
            ),
        }
        logger.info(
            f"Perturbed model (optimized currents) vs Original true - RMSE: {perturbed_overfitting_metrics['rmse']:.6f}, Correlation: {perturbed_overfitting_metrics['correlation']:.6f}"
        )

        # --- Scenario 3: Perturbed self-reconstruction ---
        logger.info("SCENARIO 3: Perturbed self-reconstruction")
        perturbed_self_recon_field, _ = reconstruct_field_from_magnitude(
            H_perturbed,
            np.abs(perturbed_true_field),
            cfg,  # Reconstruct using perturbed magnitude
        )
        perturbed_vs_perturbed_metrics = {
            "rmse": normalized_rmse(
                np.abs(perturbed_true_field), np.abs(perturbed_self_recon_field)
            ),
            "correlation": normalized_correlation(
                np.abs(perturbed_true_field), np.abs(perturbed_self_recon_field)
            ),
        }
        logger.info(
            f"Perturbed recon vs Perturbed true - RMSE: {perturbed_vs_perturbed_metrics['rmse']:.6f}, Correlation: {perturbed_vs_perturbed_metrics['correlation']:.6f}"
        )

        # Store all results for this perturbation factor
        results = {
            "factor": factor,
            "original_true_field_2d": original_true_field_2d,
            "perturbed_true_field_2d": perturbed_true_field_2d,
            "original_recon_field_2d": original_recon_field_2d,
            "perturbed_forward_field_2d": perturbed_forward_field_2d,
            "perturbed_overfitting_field_2d": perturbed_overfitting_field_2d,
            "original_vs_original_metrics": original_vs_original_metrics,
            "perturbed_forward_metrics": perturbed_forward_metrics,
            "perturbed_overfitting_metrics": perturbed_overfitting_metrics,
            "perturbed_vs_perturbed_metrics": perturbed_vs_perturbed_metrics,
        }
        all_results.append(results)

    # Create focused plots showing perturbation vs RMSE/correlation
    create_focused_perturbation_plots(all_results, perturbation_factors, cfg)

    logger.info(f"\nFocused perturbation analysis results saved to {output_dir}")


if __name__ == "__main__":
    # Setup basic logging before Hydra takes over if needed for early messages
    logging.basicConfig(level=logging.INFO)
    main()
