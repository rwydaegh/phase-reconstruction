#!/usr/bin/env python3
"""
Script to verify the effect of point perturbation on field reconstruction.
This produces multiple comparisons to ensure our findings are correct.
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the project
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_test_pointcloud import create_test_pointcloud
from src.simulation_config_real_data import SimulationConfig
from src.utils.field_utils import create_channel_matrix
from src.utils.normalized_correlation import normalized_correlation
from src.utils.normalized_rmse import normalized_rmse


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Verify perturbation effect on field reconstruction"
    )

    # Perturbation settings
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
        default="perturbation_analysis/plots",
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


def generate_currents(points, num_sources, random_generator):
    """Generate current sources with given random generator"""
    currents = np.zeros(len(points), dtype=complex)
    num_sources = min(num_sources, len(currents))

    # Sample indices using the provided random generator
    source_indices = random_generator.sample(range(len(currents)), num_sources)

    # Generate amplitudes and phases using numpy's random with the same seed
    amplitudes = np.random.lognormal(mean=0, sigma=3, size=num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=num_sources)

    # Set complex currents
    for i, idx in enumerate(source_indices):
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])

    return currents, source_indices, amplitudes, phases


def compute_field(points, currents, measurement_plane, k):
    """Compute field from currents"""
    H = create_channel_matrix(points, measurement_plane, k)
    field = H @ currents
    return field, H


def reconstruct_field_from_magnitude(H, field_magnitude, config):
    """Reconstruct field from magnitude using holographic phase retrieval"""
    reconstructed_currents = holographic_phase_retrieval(
        H,
        field_magnitude,
        adaptive_regularization=config.adaptive_regularization,
        num_iterations=config.gs_iterations,
        convergence_threshold=config.convergence_threshold,
        regularization=1e-3,
        return_history=False,
        debug=False,
    )

    reconstructed_field = H @ reconstructed_currents
    return reconstructed_field, reconstructed_currents


def evaluate_reconstruction(true_field, reconstructed_field, title):
    """Evaluate reconstruction using RMSE and correlation"""
    rmse_val = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr_val = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))

    print(f"{title} - RMSE: {rmse_val:.6f}, Correlation: {corr_val:.6f}")
    return rmse_val, corr_val


def visualize_field(field, x, y, title, filename=None, show=False):
    """Visualize field magnitude"""
    resolution = int(np.sqrt(field.size))
    field_2d = np.abs(field).reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        field_2d, cmap="viridis", origin="lower", extent=[x.min(), x.max(), y.min(), y.max()]
    )
    plt.colorbar(im, ax=ax, label="Field Magnitude")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Saved field visualization to {filename}")

    if not show:
        plt.close(fig)
    return field_2d


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
        linewidth=3,
        markersize=10,
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
        linewidth=3,
        markersize=10,
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
    """Main function to verify perturbation effect"""
    args = parse_args()

    # Parse perturbation factors
    try:
        perturbation_factors = [float(f) for f in args.perturbation_factors.split(",")]
    except ValueError:
        print("Error: Perturbation factors must be comma-separated numbers")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Create a random number generator for current source indices
    # This ensures we use the same source positions across all perturbation factors
    rng = random.Random(args.random_seed)

    # Results for each perturbation factor
    all_results = []

    # For the first factor (which should be 0), save the original points and currents
    original_points = None
    original_currents = None
    original_source_indices = None

    # Run simulations for each perturbation factor
    for factor in perturbation_factors:
        print(f"\n--- Testing perturbation factor {factor} ---")

        # Create configuration for this factor
        config = setup_simulation(args, factor)

        # Create perturbed point cloud
        perturbed_points = create_test_pointcloud(config)

        # For the first run (factor=0), save as original
        if original_points is None:
            original_points = perturbed_points.copy()

            # Generate currents for original points
            original_currents, original_source_indices, amplitudes, phases = generate_currents(
                original_points, args.num_sources, rng
            )

        # Create perturbed currents using the same source indices and amplitudes/phases
        perturbed_currents = np.zeros(len(perturbed_points), dtype=complex)
        for _, idx in enumerate(original_source_indices):
            if idx < len(perturbed_currents):
                # Actually use the same amplitude and phase as original
                perturbed_currents[idx] = original_currents[idx]

        # Create measurement plane (same for all simulations)
        measurement_plane, x, y = create_measurement_plane(config)

        # ----- COMPUTE TRUE FIELDS -----

        # Compute true field with original points
        original_true_field, H_original = compute_field(
            original_points, original_currents, measurement_plane, config.k
        )

        # Compute true field with perturbed points
        perturbed_true_field, H_perturbed = compute_field(
            perturbed_points, perturbed_currents, measurement_plane, config.k
        )

        # Visualize true fields - only storing the data, not saving visualization images
        original_true_field_2d = np.abs(original_true_field).reshape(
            config.resolution, config.resolution
        )
        perturbed_true_field_2d = np.abs(perturbed_true_field).reshape(
            config.resolution, config.resolution
        )

        # ----- SCENARIO 1: ORIGINAL RECONSTRUCTION -----
        # Reconstruct field using original points and original true field magnitude
        print("\nSCENARIO 1: Original reconstruction")
        original_recon_field, _ = reconstruct_field_from_magnitude(
            H_original, np.abs(original_true_field), config
        )

        # Store reshaped field data
        original_recon_field_2d = np.abs(original_recon_field).reshape(
            config.resolution, config.resolution
        )

        # Evaluate original reconstruction against original true field
        original_vs_original_metrics = {
            "rmse": normalized_rmse(np.abs(original_true_field), np.abs(original_recon_field)),
            "correlation": normalized_correlation(
                np.abs(original_true_field), np.abs(original_recon_field)
            ),
        }
        print(
            f"Original recon vs Original true - RMSE: {original_vs_original_metrics['rmse']:.6f}, "
            f"Correlation: {original_vs_original_metrics['correlation']:.6f}"
        )

        # ----- SCENARIO 2A: PERTURBED GEOMETRY WITH ORIGINAL CURRENTS -----
        # This is what happens in the real world:
        # 1. We have real field measurements (original_true_field)
        # 2. We have an imperfect model of the geometry (perturbed_points)
        # 3. We compute what field our imperfect model would produce WITH THE TRUE CURRENTS
        # 4. We compare this to the actual field measurements

        print("\nSCENARIO 2A: Realistic perturbation scenario - fixed currents")

        # Compute what field the perturbed point cloud would produce with original currents
        # This is our prediction from the perturbed model using the best estimate of currents
        # This simulates using our digital twin with the best estimate of the current sources
        forward_field = H_perturbed @ original_currents

        # Store reshaped field data
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
        print(
            f"Perturbed model (true currents) vs Original true - "
            f"RMSE: {perturbed_forward_metrics['rmse']:.6f}, "
            f"Correlation: {perturbed_forward_metrics['correlation']:.6f}"
        )

        # ----- SCENARIO 2B: PERTURBED RECONSTRUCTION WITH OPTIMIZED CURRENTS -----
        # This allows "overfitting":
        # 1. We have real field measurements (original_true_field)
        # 2. We have an imperfect model of the geometry (perturbed_points)
        # 3. We optimize currents using phase retrieval to match the field magnitude
        # This is like "adjusting" our source model to compensate for geometric errors

        print("\nSCENARIO 2B: Overfitting scenario - optimized currents")
        perturbed_overfitting_field, _ = reconstruct_field_from_magnitude(
            H_perturbed, np.abs(original_true_field), config
        )

        # Store reshaped field data
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
        print(
            f"Perturbed model (optimized currents) vs Original true - "
            f"RMSE: {perturbed_overfitting_metrics['rmse']:.6f}, "
            f"Correlation: {perturbed_overfitting_metrics['correlation']:.6f}"
        )

        # ----- SCENARIO 3: PERTURBED SELF-RECONSTRUCTION -----
        # Reconstruct field using perturbed points and perturbed true field magnitude
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
        print(
            f"Perturbed recon vs Perturbed true - "
            f"RMSE: {perturbed_vs_perturbed_metrics['rmse']:.6f}, "
            f"Correlation: {perturbed_vs_perturbed_metrics['correlation']:.6f}"
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
    create_focused_perturbation_plots(all_results, perturbation_factors, args)

    print(f"\nFocused perturbation analysis results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
