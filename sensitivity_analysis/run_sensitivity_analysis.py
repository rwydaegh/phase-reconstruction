import argparse
import logging
import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

# Add parent directory to path to find modules from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulation_config_real_data import SimulationConfig

# Import directly from sensitivity_analysis.py file in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensitivity_analysis import (
    ParameterRange,
    SensitivityAnalysisConfig,
    run_sensitivity_analysis,
)

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for field reconstruction"
    )

    # Add parameters for sensitivity analysis
    parser.add_argument(
        "--output-dir", type=str, default="sensitivity_results", help="Output directory for results"
    )
    parser.add_argument("--parallel", action="store_true", help="Run simulations in parallel")
    parser.add_argument(
        "--max-workers", type=int, default=12, help="Maximum number of parallel workers"
    )

    # Base simulation parameters
    parser.add_argument(
        "--resolution",
        type=int,
        default=30,
        help="Resolution of measurement plane grid (default: 30)",
    )
    parser.add_argument(
        "--gs-iterations",
        type=int,
        default=200,
        help="Maximum Gerchberg-Saxton iterations (default: 200)",
    )
    parser.add_argument(
        "--perturb-points", action="store_true", help="Enable point cloud perturbation"
    )
    parser.add_argument(
        "--perturbation-factor",
        type=float,
        default=0.01,
        help="Max perturbation as percentage of distance to origin (default: 0.01)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging (debug level)"
    )

    # Parse arguments
    return parser.parse_args()


def main():
    """Run sensitivity analysis with customizable parameters"""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create base configuration
    base_config = SimulationConfig(
        wavelength=10.7e-3,  # 28GHz wavelength in meters
        plane_size=1.0,  # 1m x 1m measurement plane
        resolution=args.resolution,
        room_size=2.0,  # 2m x 2m x 2m room
        wall_points=6,  # Points per wall edge
        num_sources=50,  # Number of sources to randomly select
        gs_iterations=args.gs_iterations,
        convergence_threshold=1e-3,
        perturb_points=args.perturb_points,  # Apply perturbation if enabled
        perturbation_factor=args.perturbation_factor,  # Perturbation scaling factor
        show_plot=False,  # Disable plotting for batch runs
        return_history=False,  # Disable history for faster execution
        verbose=args.verbose,  # Verbose output if requested
    )

    # Define parameter pairs to analyze
    # Focusing on the most important parameters for performance analysis
    parameter_ranges = [
        # Number of sources (how many active sources on walls)
        ParameterRange(param_name="num_sources", start=10, end=200, num_steps=10, log_scale=True),
        # Resolution of measurement plane
        ParameterRange(param_name="resolution", start=10, end=50, num_steps=8, log_scale=False),
        # Wall points per edge
        ParameterRange(param_name="wall_points", start=5, end=20, num_steps=8, log_scale=False),
    ]

    # Add perturbation factor range if perturbation is enabled
    if args.perturb_points:
        parameter_ranges.append(
            ParameterRange(
                param_name="perturbation_factor",
                start=0.001,  # 0.1% perturbation
                end=0.1,  # 10% perturbation
                num_steps=8,
                log_scale=True,
            )
        )

    # Create sensitivity analysis configuration
    analysis_config = SensitivityAnalysisConfig(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        output_dir=args.output_dir,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )

    # Run sensitivity analysis
    logger.info("Starting sensitivity analysis")
    run_sensitivity_analysis(analysis_config)
    logger.info(f"Sensitivity analysis complete. Results saved to {args.output_dir}")

    # Create visualizations directly in the output directory
    create_visualizations(args.output_dir, save_dir=args.output_dir)
    logger.info(f"Visualizations created in '{args.output_dir}' directory")


def create_rmse_plot(X, Y, rmse, param1_name, param2_name, save_dir):
    """Create an RMSE plot with outlier visualization."""
    # Identify outliers (values significantly larger than median)
    median_rmse = np.nanmedian(rmse)
    std_rmse = np.nanstd(rmse)
    outlier_threshold = median_rmse + 3 * std_rmse
    is_outlier = rmse > outlier_threshold

    # For visualization, cap extreme values to show color contrast better
    rmse_viz = rmse.copy()
    normal_max = np.nanpercentile(rmse[~is_outlier], 95)  # 95th percentile of non-outlier values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use diverging colormap for better contrast
    cmap = plt.cm.viridis.copy()

    # First plot non-outlier values with regular scale
    im = ax.pcolormesh(X, Y, rmse_viz, cmap=cmap, vmin=np.nanmin(rmse), vmax=normal_max)

    # Add outlier markers
    for ix in range(X.shape[0]):
        for iy in range(X.shape[1]):
            if is_outlier[ix, iy]:
                # Add a red marker for outliers
                rect = patches.Rectangle(
                    (X[ix, iy] - (X[0, 1] - X[0, 0]) / 2, Y[ix, iy] - (Y[1, 0] - Y[0, 0]) / 2),
                    X[0, 1] - X[0, 0],
                    Y[1, 0] - Y[0, 0],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add text with the actual value
                ax.text(
                    X[ix, iy],
                    Y[ix, iy],
                    f"{rmse[ix, iy]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "red"},
                )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE (normal range)", fontsize=12)

    # Add a note about outliers
    outlier_count = np.sum(is_outlier)
    if outlier_count > 0:
        ax.text(
            0.5,
            1.05,
            f"⚠️ {outlier_count} outlier(s) found - marked in red",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
            color="red",
        )

    # Set labels and title
    ax.set_xlabel(param1_name, fontsize=14)
    ax.set_ylabel(param2_name, fontsize=14)
    ax.set_title(f"RMSE Analysis: {param1_name} vs {param2_name}", fontsize=16)

    # Apply log scales if needed
    if np.all(np.diff(X[0, :]) > 0) and X[0, 1] / X[0, 0] > 1.5:
        ax.set_xscale("log")
    if np.all(np.diff(Y[:, 0]) > 0) and Y[1, 0] / Y[0, 0] > 1.5:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{param1_name}_vs_{param2_name}_rmse.png"), dpi=300)
    plt.close(fig)


def create_correlation_quality_map(X, Y, correlation, param1_name, param2_name, save_dir):
    """Create a quality classification map based on correlation values."""
    # Define quality thresholds
    quality_levels = {
        "Excellent": 0.999,  # Correlation > 0.999
        "Very Good": 0.99,  # Correlation > 0.99
        "Good": 0.9,  # Correlation > 0.9
        "Fair": 0.7,  # Correlation > 0.7
        "Poor": 0.5,  # Correlation > 0.5
        "Very Poor": 0.0,  # Correlation ≤ 0.5
    }

    # Create quality map
    quality_map = np.zeros_like(correlation, dtype=int)

    thresholds = list(quality_levels.values())
    for i in range(len(thresholds) - 1):
        mask = correlation > thresholds[i]
        quality_map[mask] = len(thresholds) - 1 - i

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a perceptually distinct colormap
    # Fix for matplotlib 3.7+ deprecation warning
    try:
        # New way in matplotlib 3.7+
        cmap = plt.colormaps["RdYlGn"].resampled(len(thresholds) - 1)
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        cmap = plt.cm.get_cmap("RdYlGn", len(thresholds) - 1)

    # Plot quality map
    im = ax.pcolormesh(X, Y, quality_map, cmap=cmap, vmin=0, vmax=len(thresholds) - 1)

    # Create custom colorbar with quality levels
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(thresholds) - 1))
    quality_labels = list(quality_levels.keys())[:-1]
    quality_labels.reverse()  # Reverse for correct order in colorbar
    cbar.set_ticklabels(quality_labels)
    cbar.set_label("Reconstruction Quality", fontsize=12)

    # Add correlation values as text
    for ix in range(X.shape[0]):
        for iy in range(X.shape[1]):
            # Only add text for lower quality levels
            if correlation[ix, iy] < 0.999:
                ax.text(
                    X[ix, iy],
                    Y[ix, iy],
                    f"{correlation[ix, iy]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.7},
                )

    # Set labels and title
    ax.set_xlabel(param1_name, fontsize=14)
    ax.set_ylabel(param2_name, fontsize=14)
    ax.set_title(f"Reconstruction Quality: {param1_name} vs {param2_name}", fontsize=16)

    # Apply log scales if needed
    if np.all(np.diff(X[0, :]) > 0) and X[0, 1] / X[0, 0] > 1.5:
        ax.set_xscale("log")
    if np.all(np.diff(Y[:, 0]) > 0) and Y[1, 0] / Y[0, 0] > 1.5:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"quality_{param1_name}_vs_{param2_name}.png"), dpi=300)
    plt.close(fig)


def plot_summary_subplot(ax, X, Y, rmse, correlation, param1_name, param2_name):
    """Create a subplot for the summary figure."""
    # Use a log scale for RMSE to show both large and small values
    log_rmse = np.log10(np.clip(rmse, 1e-10, np.inf))

    # Use a viridis colormap
    im = ax.pcolormesh(X, Y, log_rmse, cmap="viridis")

    # Apply log scales if needed
    if np.all(np.diff(X[0, :]) > 0) and X[0, 1] / X[0, 0] > 1.5:
        ax.set_xscale("log")
    if np.all(np.diff(Y[:, 0]) > 0) and Y[1, 0] / Y[0, 0] > 1.5:
        ax.set_yscale("log")

    # Mark locations with poor correlation
    poor_corr_mask = correlation < 0.9
    if np.any(poor_corr_mask):
        for ix in range(X.shape[0]):
            for iy in range(X.shape[1]):
                if poor_corr_mask[ix, iy]:
                    ax.plot(X[ix, iy], Y[ix, iy], "rx", markersize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(RMSE)")

    # Set labels and title
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(f"{param1_name} vs {param2_name}")


def create_visualizations(output_dir="sensitivity_results", save_dir=None):
    """Create visualizations of sensitivity analysis results."""
    # If no save_dir is provided, use the output_dir
    if save_dir is None:
        save_dir = output_dir

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # List of all npz files to process
    npz_files = [
        f
        for f in os.listdir(output_dir)
        if f.endswith("_data.npz") and f.startswith("sensitivity_")
    ]

    if not npz_files:
        logger.warning("No sensitivity analysis data files found for visualization")
        return

    # Create a figure for summary visualization
    fig_summary, ax_summary = plt.subplots(1, min(3, len(npz_files)), figsize=(18, 6))
    if len(npz_files) == 1:
        ax_summary = np.array([ax_summary])

    for i, npz_file in enumerate(npz_files):
        if i >= len(ax_summary):
            break

        file_path = os.path.join(output_dir, npz_file)
        if not os.path.exists(file_path):
            logger.warning(f"Warning: File not found: {file_path}")
            continue

        # Load data
        data = np.load(file_path)
        X = data["X"]
        Y = data["Y"]
        rmse = data["rmse"]
        correlation = data["correlation"]
        param1_name = str(data["param1_name"])
        param2_name = str(data["param2_name"])

        # Create RMSE visualization
        create_rmse_plot(X, Y, rmse, param1_name, param2_name, save_dir)

        # Create correlation quality map
        create_correlation_quality_map(X, Y, correlation, param1_name, param2_name, save_dir)

        # Add to summary plot
        plot_summary_subplot(ax_summary[i], X, Y, rmse, correlation, param1_name, param2_name)

    # Finalize and save summary plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sensitivity_summary.png"), dpi=300)
    plt.close(fig_summary)

    logger.info(f"Visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()
