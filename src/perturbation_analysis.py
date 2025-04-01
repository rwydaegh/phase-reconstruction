#!/usr/bin/env python3
"""
Script to analyze the effect of point perturbation on field reconstruction.
This produces multiple comparisons to ensure our findings are correct.
"""

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List # Import List for type hinting

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Use the correct import path
from src.create_channel_matrix import create_channel_matrix
from create_test_pointcloud import create_test_pointcloud
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.visualization.utils import visualize_field
from utils.normalized_correlation import normalized_correlation
from utils.normalized_rmse import normalized_rmse


@dataclass
class SimulationConfig:
    wavelength: float = 10.7e-3
    plane_size: float = 1.0
    resolution: int = 30
    room_size: float = 2.0
    wall_points: int = 10
    num_sources: int = 50
    gs_iterations: int = 200
    convergence_threshold: float = 1e-3
    perturb_points: bool = False
    perturbation_factor: float = 0.0
    amplitude_sigma: float = 1.0 # Added default based on generate_currents usage
    # Added default based on reconstruct_field_from_magnitude usage
    adaptive_regularization: bool = False
    regularization: float = 1e-4 # Added default based on reconstruct_field_from_magnitude usage
    # Added default based on reconstruct_field_from_magnitude usage
    enable_perturbations: bool = False
    stagnation_window: int = 50 # Added default based on reconstruct_field_from_magnitude usage
    # Added default based on reconstruct_field_from_magnitude usage
    stagnation_threshold: float = 1e-5
    # Added default based on reconstruct_field_from_magnitude usage
    perturbation_intensity: float = 0.1
    perturbation_mode: str = "basic" # Added default based on reconstruct_field_from_magnitude usage
    # Added default based on reconstruct_field_from_magnitude usage
    constraint_skip_iterations: int = 0
    momentum_factor: float = 0.5 # Added default based on reconstruct_field_from_magnitude usage
    temperature: float = 1.0 # Added default based on reconstruct_field_from_magnitude usage
    verbose: bool = True
    no_plot: bool = False
    measurement_direction: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0]) # Default Y-direction
    k: float = field(init=False)

    def __post_init__(self):
        self.k = 2 * math.pi / self.wavelength


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze perturbation effect on field reconstruction"
    )

    parser.add_argument(
        "--perturbation-factors",
        type=str,
        default="0,0.001,0.01,0.05,0.1",
        help="Comma-separated list of perturbation factors to test",
    )
    parser.add_argument(
        "--wall-points", type=int, default=10, help="Points per wall edge (default: 10)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=30,
        help="Resolution of measurement plane grid (default: 30)",
    )
    parser.add_argument(
        "--num-sources", type=int, default=50, help="Number of sources (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="perturbation_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Show plots instead of just saving them"
    )

    return parser.parse_args()


def setup_simulation(args, factor):
    """Set up simulation with given perturbation factor"""
    perturb = factor > 0
    config = SimulationConfig(
        wavelength=10.7e-3,
        plane_size=1.0,
        resolution=args.resolution,
        room_size=2.0,
        wall_points=args.wall_points,
        num_sources=args.num_sources,
        gs_iterations=200,
        convergence_threshold=1e-3,
        perturb_points=perturb,
        perturbation_factor=factor if perturb else 0,
        verbose=True,
    )
    return config


def create_measurement_plane(config):
    """Create measurement plane grid"""
    resolution = int(config.resolution)
    x = np.linspace(
        config.room_size / 2 - config.plane_size / 2,
        config.room_size / 2 + config.plane_size / 2,
        resolution,
    )
    y = np.linspace(
        config.room_size / 2 - config.plane_size / 2,
        config.room_size / 2 + config.plane_size / 2,
        resolution,
    )
    X, Y = np.meshgrid(x, y)
    return np.stack([X, Y, np.ones_like(X) * config.room_size / 2], axis=-1), x, y


def generate_currents(points, num_sources, random_generator, config):
    """Generate current sources (2 components per point) with given random generator"""
    N_c = len(points)
    currents = np.zeros(2 * N_c, dtype=complex) # Shape (2*N_c,)
    num_sources = min(num_sources, N_c) # Number of points to activate

    # Indices of points (0 to N_c-1)
    source_indices = random_generator.sample(range(N_c), num_sources)

    # Use numpy's default random generator for amplitude/phase for consistency
    amplitudes = np.random.lognormal(mean=0, sigma=config.amplitude_sigma, size=num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    # Assign to the first component (index 2*point_idx)
    for i, point_idx in enumerate(source_indices):
        current_idx = 2 * point_idx
        currents[current_idx] = amplitudes[i] * np.exp(1j * phases[i])
        # currents[current_idx + 1] = 0.0 # Second component remains zero

    # Return currents (2*N_c,), point indices (N_c,), amplitudes (N_c,), phases (N_c,)
    return currents, source_indices, amplitudes, phases


def compute_field(points, tangents1, tangents2, currents, measurement_plane, measurement_direction, k):
    """Compute field component from currents using vectorized H and pre-calculated tangents"""
    # currents should have shape (2 * N_c,)
    H = create_channel_matrix(points, tangents1, tangents2, measurement_plane, measurement_direction, k)
    # H (N_m, 2*N_c) @ currents (2*N_c,) -> field (N_m,)
    if currents.shape[0] != H.shape[1]:
         raise ValueError(f"Shape mismatch: H columns {H.shape[1]} != currents length {currents.shape[0]}")
    field = H @ currents
    return field, H


def reconstruct_field_from_magnitude(H, field_magnitude, config: SimulationConfig):
    """Reconstruct field from magnitude using holographic phase retrieval"""
    # H has shape (N_m, 2*N_c)
    # field_magnitude has shape (N_m,)
    # holographic_phase_retrieval should return currents of shape (2*N_c,)
    hpr_result = holographic_phase_retrieval(
        config, # Pass the config object directly if HPR expects it
        H,
        field_magnitude,
        # initial_field_values=None, # Assuming no initial field guess here
        # output_dir=None # Assuming no output needed here
    )

    # Handle potential tuple return if history/stats are included
    if isinstance(hpr_result, tuple):
        reconstructed_currents = hpr_result[0] # Assume currents are the first element
    else:
        reconstructed_currents = hpr_result

    if reconstructed_currents.shape[0] != H.shape[1]:
        raise ValueError(
            f"Shape mismatch after HPR: H columns {H.shape[1]} != "
            f"reconstructed_currents length {reconstructed_currents.shape[0]}"
        )

    reconstructed_field = H @ reconstructed_currents
    return reconstructed_field, reconstructed_currents


def evaluate_reconstruction(true_field, reconstructed_field, title):
    """Evaluate reconstruction using RMSE and correlation"""
    rmse_val = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr_val = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))

    # print(f"{title} - RMSE: {rmse_val:.6f}, Correlation: {corr_val:.6f}")
    # Commented out for less verbosity
    return rmse_val, corr_val


def create_focused_perturbation_plots(all_results, perturbation_factors, args):
    """Create focused plots showing the relationship between perturbation amount
    and RMSE/correlation"""

    # Extract metrics
    rmse_real = [res["perturbed_forward_metrics"]["rmse"] for res in all_results]
    corr_real = [res["perturbed_forward_metrics"]["correlation"] for res in all_results]

    rmse_overfit = [res["perturbed_overfitting_metrics"]["rmse"] for res in all_results]
    corr_overfit = [res["perturbed_overfitting_metrics"]["correlation"] for res in all_results]

    rmse_self = [res["perturbed_vs_perturbed_metrics"]["rmse"] for res in all_results]
    corr_self = [res["perturbed_vs_perturbed_metrics"]["correlation"] for res in all_results]

    rmse_orig = [res["original_vs_original_metrics"]["rmse"] for res in all_results]
    corr_orig = [res["original_vs_original_metrics"]["correlation"] for res in all_results]

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

    rmse_filename = os.path.join(args.output_dir, "rmse_vs_perturbation.png")
    plt.savefig(rmse_filename, dpi=300)
    print(f"Saved RMSE vs perturbation plot to {rmse_filename}")

    if not args.show_plots:
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

    corr_filename = os.path.join(args.output_dir, "correlation_vs_perturbation.png")
    plt.savefig(corr_filename, dpi=300)
    print(f"Saved correlation vs perturbation plot to {corr_filename}")

    if not args.show_plots:
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
    comparison_filename = os.path.join(args.output_dir, "realistic_vs_overfitting.png")
    plt.savefig(comparison_filename, dpi=300)
    print(f"Saved comparison plot to {comparison_filename}")

    if not args.show_plots:
        plt.close()

    # Create a table of numerical values
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

    # Create figure and axis for table
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
        cellText=[
            [f"{row[0]:.6f}" if i == 0 else f"{row[i]:.6f}" for i in range(len(row))]
            for row in table_data
        ],
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for (i, _), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")

    plt.title("Numerical Results for Perturbation Effect Analysis", fontsize=14)
    plt.tight_layout()

    table_filename = os.path.join(args.output_dir, "perturbation_results_table.png")
    plt.savefig(table_filename, dpi=300)
    print(f"Saved results table to {table_filename}")

    if not args.show_plots:
        plt.close()


def main():
    """Main function to analyze perturbation effect"""
    args = parse_args()

    try:
        perturbation_factors = [float(f) for f in args.perturbation_factors.split(",")]
    except ValueError:
        print("Error: Perturbation factors must be comma-separated numbers")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    rng = random.Random(args.random_seed)

    all_results = []

    original_points = None
    original_currents = None
    original_source_indices = None

    for factor in perturbation_factors:
        # print(f"\n--- Testing perturbation factor {factor} ---")
        # Commented out for less verbosity

        config = setup_simulation(args, factor)

        # Create points (potentially perturbed)
        perturbed_points = create_test_pointcloud(config)
        # Calculate tangents for perturbed points (assuming default normals)
        temp_perturbed_normals = np.zeros_like(perturbed_points)
        temp_perturbed_normals[:, 2] = 1.0
        # Need the tangent calculation function from its new location
        try:
            # Adjust import path relative to this file's location
            from utils.preprocess_pointcloud import get_tangent_vectors
        except ImportError:
             # Add src/utils directory to path if needed
             utils_dir = os.path.join(os.path.dirname(__file__), 'utils')
             if utils_dir not in sys.path: sys.path.append(utils_dir)
             from preprocess_pointcloud import get_tangent_vectors # Now should work
        perturbed_tangents1, perturbed_tangents2 = get_tangent_vectors(temp_perturbed_normals)

        # Get measurement direction
        measurement_direction = np.array(config.measurement_direction)
        measurement_direction /= np.linalg.norm(measurement_direction) # Ensure unit vector

        if original_points is None:
            original_points = perturbed_points.copy()
            # Calculate tangents for original points using the imported function
            temp_original_normals = np.zeros_like(original_points)
            temp_original_normals[:, 2] = 1.0
            original_tangents1, original_tangents2 = get_tangent_vectors(temp_original_normals)

            # Generate original currents (shape 2*N_c)
            original_currents, original_source_indices, amplitudes, phases = generate_currents(
                original_points, args.num_sources, rng, config
            )

        # Create perturbed currents using the same source indices and amplitudes/phases
        # Create perturbed currents (shape 2*N_c) using the same source indices and values
        perturbed_currents = np.zeros(2 * len(perturbed_points), dtype=complex)
        for _, point_idx in enumerate(original_source_indices):
            original_current_idx = 2 * point_idx
            if original_current_idx < len(original_currents): # Check bounds for original
                 # Assign the value from the first component of the original current
                 perturbed_current_idx = 2 * point_idx
                 if perturbed_current_idx < len(perturbed_currents): # Check bounds for perturbed
                      perturbed_currents[perturbed_current_idx] = original_currents[original_current_idx]
                      # perturbed_currents[perturbed_current_idx + 1] = 0.0 # Keep second component zero

        measurement_plane, x, y = create_measurement_plane(config)


        # Compute fields using the updated function signature
        original_true_field, H_original = compute_field(
            original_points, original_tangents1, original_tangents2, original_currents,
            measurement_plane, measurement_direction, config.k
        )

        perturbed_true_field, H_perturbed = compute_field(
            perturbed_points, perturbed_tangents1, perturbed_tangents2, perturbed_currents,
            measurement_plane, measurement_direction, config.k
        )

        original_true_field_2d = np.abs(original_true_field).reshape(
            config.resolution, config.resolution
        )
        perturbed_true_field_2d = np.abs(perturbed_true_field).reshape(
            config.resolution, config.resolution
        )

        print("\nSCENARIO 1: Original reconstruction")
        original_recon_field, _ = reconstruct_field_from_magnitude(
            H_original, np.abs(original_true_field), config
        )

        original_recon_field_2d = np.abs(original_recon_field).reshape(
            config.resolution, config.resolution
        )

        original_vs_original_metrics = {
            "rmse": normalized_rmse(np.abs(original_true_field), np.abs(original_recon_field)),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(original_recon_field)
            ),
        }
        # print(
        #     f"Original recon vs Original true - "
        #     f"RMSE: {original_vs_original_metrics['rmse']:.6f}, "
        #     f"Correlation: {original_vs_original_metrics['correlation']:.6f}"
        # )
        # Commented out for less verbosity




        # Calculate forward field with perturbed H and original currents (shape 2*N_c)
        forward_field = H_perturbed @ original_currents

        perturbed_forward_field_2d = np.abs(forward_field).reshape(
            config.resolution, config.resolution
        )

        # Evaluate perturbed forward model against original true field
        perturbed_forward_metrics = {
            "rmse": normalized_rmse(np.abs(original_true_field), np.abs(forward_field)),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(forward_field)
            ),
        }
        # print(
        #     f"Perturbed model (true currents) vs Original true - "
        #     f"RMSE: {perturbed_forward_metrics['rmse']:.6f}, "
        #     f"Correlation: {perturbed_forward_metrics['correlation']:.6f}"
        # ) # Commented out for less verbosity



        print("\nSCENARIO 2B: Overfitting scenario - optimized currents")
        perturbed_overfitting_field, _ = reconstruct_field_from_magnitude(
            H_perturbed, np.abs(original_true_field), config
        )

        perturbed_overfitting_field_2d = np.abs(perturbed_overfitting_field).reshape(
            config.resolution, config.resolution
        )

        # Evaluate perturbed reconstruction against original true field
        perturbed_overfitting_metrics = {
            "rmse": normalized_rmse(
                np.abs(original_true_field), np.abs(perturbed_overfitting_field)
            ),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(perturbed_overfitting_field)
            ),
        }
        # print(
        #     f"Perturbed model (optimized currents) vs Original true - "
        #     f"RMSE: {perturbed_overfitting_metrics['rmse']:.6f}, "
        #     f"Correlation: {perturbed_overfitting_metrics['correlation']:.6f}"
        # ) # Commented out for less verbosity

        print("\nSCENARIO 3: Perturbed reconstruction vs Perturbed true field")
        perturbed_self_recon_field, _ = reconstruct_field_from_magnitude(
            H_perturbed, np.abs(perturbed_true_field), config
        )

        # Skip saving visualization, just evaluate
        perturbed_vs_perturbed_metrics = {
            "rmse": normalized_rmse(
                np.abs(perturbed_true_field), np.abs(perturbed_self_recon_field)
            ),
            "correlation": normalized_correlation(
                np.abs(perturbed_true_field), np.abs(perturbed_self_recon_field)
            ),
        }
        # print(
        #     f"Perturbed recon vs Perturbed true - "
        #     f"RMSE: {perturbed_vs_perturbed_metrics['rmse']:.6f}, "
        #     f"Correlation: {perturbed_vs_perturbed_metrics['correlation']:.6f}"
        # ) # Commented out for less verbosity

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
    create_focused_perturbation_plots(all_results, perturbation_factors, args)

    print(f"\nFocused perturbation analysis results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
