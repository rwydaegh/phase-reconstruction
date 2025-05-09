"""
Script to create improved visualizations for sensitivity analysis results,
highlighting outliers and providing better insight into algorithm behavior.
"""

import logging
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


def enhance_sensitivity_visualization(output_dir="sensitivity_results", save_dir="figs"):
    """Create enhanced visualizations of sensitivity analysis results."""
    # Configure logging inside the function
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Dynamically find all sensitivity data files in the output directory
    try:
        all_files = os.listdir(output_dir)
        npz_files = [
            f for f in all_files if f.startswith("sensitivity_") and f.endswith("_data.npz")
        ]
        if not npz_files:
            logger.warning(f"No sensitivity data files (*_data.npz) found in {output_dir}")
            return
        logger.info(f"Found {len(npz_files)} sensitivity data files to visualize.")
    except FileNotFoundError:
        logger.error(f"Output directory not found: {output_dir}")
        return

    # Create a figure for summary visualization
    # Adjust summary plot size based on the number of files found
    num_plots = len(npz_files)
    ncols = min(num_plots, 3)  # Max 3 columns
    nrows = (num_plots + ncols - 1) // ncols
    fig_summary, ax_summary = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
    )
    ax_summary_flat = ax_summary.flatten()

    for i, npz_file in enumerate(npz_files):
        file_path = os.path.join(output_dir, npz_file)
        if not os.path.exists(file_path):
            logger.warning(f"File not found, skipping: {file_path}")  # Adjusted message slightly
            continue

        # Load data
        data = np.load(file_path)
        X = data["X"]
        Y = data["Y"]
        rmse = data["rmse"]
        correlation = data["correlation"]
        param1_name = str(data["param1_name"])
        param2_name = str(data["param2_name"])

        # Create enhanced RMSE visualization
        create_enhanced_rmse_plot(X, Y, rmse, param1_name, param2_name, save_dir)

        # Create correlation quality map
        create_correlation_quality_map(X, Y, correlation, param1_name, param2_name, save_dir)

        # Add to summary plot
        # Add to summary plot using the flattened axes array
        if i < len(ax_summary_flat):
            plot_summary_subplot(
                ax_summary_flat[i], X, Y, rmse, correlation, param1_name, param2_name
            )

    # Hide any unused subplots in the summary figure
    for j in range(i + 1, len(ax_summary_flat)):
        ax_summary_flat[j].axis("off")

    # Finalize and save summary plot
    plt.tight_layout()
    summary_filename = os.path.join(save_dir, "enhanced_sensitivity_summary.png")
    plt.savefig(summary_filename, dpi=300)
    plt.close(fig_summary)
    logger.info(f"Summary plot saved to {summary_filename}")

    logger.info(f"Enhanced visualizations saved to {save_dir}")  # Use logger instead of print


def create_enhanced_rmse_plot(X, Y, rmse, param1_name, param2_name, save_dir):
    """Create an enhanced RMSE plot with better outlier visualization."""
    # Identify outliers
    median_rmse = np.nanmedian(rmse)
    std_rmse = np.nanstd(rmse)
    outlier_threshold = median_rmse + 3 * std_rmse
    is_outlier = rmse > outlier_threshold

    # Cap extreme values for visualization
    rmse_viz = rmse.copy()
    normal_max = np.nanpercentile(rmse[~is_outlier], 95)  # 95th percentile of non-outlier values

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a suitable colormap
    cmap = plt.cm.viridis.copy()

    # Plot non-outlier values
    im = ax.pcolormesh(X, Y, rmse_viz, cmap=cmap, vmin=np.nanmin(rmse), vmax=normal_max)

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

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE (normal range)", fontsize=12)

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

    ax.set_xlabel(param1_name, fontsize=14)
    ax.set_ylabel(param2_name, fontsize=14)
    ax.set_title(f"Enhanced RMSE Analysis: {param1_name} vs {param2_name}", fontsize=16)

    if np.all(np.diff(X[0, :]) > 0) and X[0, 1] / X[0, 0] > 1.5:
        ax.set_xscale("log")
    if np.all(np.diff(Y[:, 0]) > 0) and Y[1, 0] / Y[0, 0] > 1.5:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"enhanced_{param1_name}_vs_{param2_name}_rmse.png"), dpi=300
    )
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

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a distinct colormap
    cmap = plt.colormaps["RdYlGn"].resampled(
        len(thresholds) - 1
    )  # Use discrete colormap with specific number of levels

    im = ax.pcolormesh(X, Y, quality_map, cmap=cmap, vmin=0, vmax=len(thresholds) - 1)

    cbar = plt.colorbar(im, ax=ax, ticks=range(len(thresholds) - 1))
    quality_labels = list(quality_levels.keys())[:-1]
    quality_labels.reverse()  # Reverse for correct order in colorbar
    cbar.set_ticklabels(quality_labels)
    cbar.set_label("Reconstruction Quality", fontsize=12)

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

    ax.set_xlabel(param1_name, fontsize=14)
    ax.set_ylabel(param2_name, fontsize=14)
    ax.set_title(f"Reconstruction Quality: {param1_name} vs {param2_name}", fontsize=16)

    if np.all(np.diff(X[0, :]) > 0) and X[0, 1] / X[0, 0] > 1.5:
        ax.set_xscale("log")
    if np.all(np.diff(Y[:, 0]) > 0) and Y[1, 0] / Y[0, 0] > 1.5:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"quality_{param1_name}_vs_{param2_name}.png"), dpi=300)
    plt.close(fig)


def plot_summary_subplot(ax, X, Y, rmse, correlation, param1_name, param2_name):
    """Create a subplot for the summary figure."""
    # Use log scale for RMSE
    log_rmse = np.log10(np.clip(rmse, 1e-10, np.inf))

    im = ax.pcolormesh(X, Y, log_rmse, cmap="viridis")

    if np.all(np.diff(X[0, :]) > 0) and X[0, 1] / X[0, 0] > 1.5:
        ax.set_xscale("log")
    if np.all(np.diff(Y[:, 0]) > 0) and Y[1, 0] / Y[0, 0] > 1.5:
        ax.set_yscale("log")

    poor_corr_mask = correlation < 0.9
    if np.any(poor_corr_mask):
        for ix in range(X.shape[0]):
            for iy in range(X.shape[1]):
                if poor_corr_mask[ix, iy]:
                    ax.plot(X[ix, iy], Y[ix, iy], "rx", markersize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(RMSE)")

    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(f"{param1_name} vs {param2_name}")


def main():
    """Main function to run enhanced visualization."""
    enhance_sensitivity_visualization()


if __name__ == "__main__":
    main()
