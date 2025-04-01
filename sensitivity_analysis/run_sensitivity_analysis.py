import hydra
from omegaconf import DictConfig, OmegaConf

# import argparse # Replaced by Hydra
import logging
import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

# Add parent directory to path to find modules from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.simulation_config_real_data import SimulationConfig # Replaced by Hydra config

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


# Removed parse_args function (handled by Hydra)


@hydra.main(config_path="../conf", config_name="sensitivity_analysis", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run sensitivity analysis using Hydra configuration"""
    # args = parse_args() # Removed

    # Log the configuration
    logger.info("Starting sensitivity analysis with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Create output directory (Hydra handles the main output dir, but we might want a subdir)
    # Using Hydra's output dir: hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Or simply let run_sensitivity_analysis handle it based on cfg.output_dir
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create base configuration (Now part of the loaded cfg)
    # base_config = SimulationConfig(...) # Removed

    # Define parameter pairs to analyze (Now loaded from cfg)
    # parameter_ranges = [...] # Removed

    # Create sensitivity analysis configuration object from Hydra config
    # Need to instantiate ParameterRange objects from the list in cfg
    param_ranges_inst = [ParameterRange(**p_range) for p_range in cfg.parameter_ranges]

    analysis_config = SensitivityAnalysisConfig(
        base_config=cfg.base_simulation, # Access the composed base config
        parameter_ranges=param_ranges_inst,
        output_dir=output_dir, # Use output_dir from config
        parallel=cfg.parallel,
        max_workers=cfg.max_workers,
    )

    # Run sensitivity analysis
    logger.info("Starting sensitivity analysis")
    run_sensitivity_analysis(analysis_config)
    logger.info(f"Sensitivity analysis complete. Results saved to {output_dir}")

    # Create visualizations directly in the output directory
    create_visualizations(output_dir, save_dir=output_dir)
    logger.info(f"Visualizations created in '{output_dir}' directory")


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


# Removed __main__ block, entry point handled by Hydra decorator
