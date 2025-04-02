import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def visualize_comparison(
    true_field: np.ndarray,
    reconstructed_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Field Comparison",
    metrics: Optional[Dict[str, float]] = None,
    filename: Optional[str] = None,
    show: bool = True,
    measurement_plane: Optional[np.ndarray] = None,
) -> None:
    """Create a side-by-side comparison of true vs reconstructed fields.

    Args:
        true_field: True complex field
        reconstructed_field: Reconstructed complex field
        x: X-coordinates of the field grid
        y: Y-coordinates of the field grid
        title: Main title for the plot
        metrics: Optional dictionary of metrics to display (e.g., {'RMSE': 0.05})
        filename: If provided, save the figure to this path
        show: Whether to display the figure
        measurement_plane: Optional measurement plane coordinates for proper axis labeling
    """
    resolution = int(np.sqrt(true_field.size))
    true_mag = np.abs(true_field).reshape(resolution, resolution)
    recon_mag = np.abs(reconstructed_field).reshape(resolution, resolution)

    # Calculate error map
    error = np.abs(true_mag - recon_mag)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Calculate global min and max for consistent colorbar scaling
    global_min = min(np.min(true_mag), np.min(recon_mag))
    global_max = max(np.max(true_mag), np.max(recon_mag))

    # Determine plane type and labels if measurement_plane is provided
    horizontal_label = "X (m)"
    vertical_label = "Y (m)"
    plane_type = "XY"
    extent = [x.min(), x.max(), y.min(), y.max()]

    if measurement_plane is not None:
        # Determine plane type and set appropriate axis labels
        x_min, _ = np.min(measurement_plane[:, :, 0]), np.max(measurement_plane[:, :, 0])
        # x_max unused
        y_min, _ = np.min(measurement_plane[:, :, 1]), np.max(measurement_plane[:, :, 1])
        # y_max unused
        _, _ = np.min(measurement_plane[:, :, 2]), np.max(measurement_plane[:, :, 2])
        # z_min, z_max unused

        # Determine plane type and set coordinates and labels
        if np.allclose(measurement_plane[:, :, 0], x_min):  # YZ plane
            horizontal_coords = measurement_plane[0, :, 1]  # Y values
            vertical_coords = measurement_plane[:, 0, 2]  # Z values
            horizontal_label = "Y (m)"
            vertical_label = "Z (m)"
            plane_type = "YZ"
            extent = [
                horizontal_coords.min(),
                horizontal_coords.max(),
                vertical_coords.min(),
                vertical_coords.max(),
            ]
        elif np.allclose(measurement_plane[:, :, 1], y_min):  # XZ plane
            horizontal_coords = measurement_plane[0, :, 0]  # X values
            vertical_coords = measurement_plane[:, 0, 2]  # Z values
            horizontal_label = "X (m)"
            vertical_label = "Z (m)"
            plane_type = "XZ"
            extent = [
                horizontal_coords.min(),
                horizontal_coords.max(),
                vertical_coords.min(),
                vertical_coords.max(),
            ]
        else:  # XY plane (default)
            horizontal_coords = measurement_plane[0, :, 0]  # X values
            vertical_coords = measurement_plane[:, 0, 1]  # Y values
            horizontal_label = "X (m)"
            vertical_label = "Y (m)"
            plane_type = "XY"
            extent = [
                horizontal_coords.min(),
                horizontal_coords.max(),
                vertical_coords.min(),
                vertical_coords.max(),
            ]

    # Plot true field with normalized colorbar
    im1 = axes[0].imshow(
        true_mag, cmap="jet", origin="lower", extent=extent, vmin=global_min, vmax=global_max
    )
    axes[0].set_title(f"True Field ({plane_type} Plane)")
    axes[0].set_xlabel(horizontal_label)
    axes[0].set_ylabel(vertical_label)
    plt.colorbar(im1, ax=axes[0], label="Field Magnitude")

    # Plot reconstructed field with normalized colorbar
    im2 = axes[1].imshow(
        recon_mag, cmap="jet", origin="lower", extent=extent, vmin=global_min, vmax=global_max
    )
    axes[1].set_title(f"Reconstructed Field ({plane_type} Plane)")
    axes[1].set_xlabel(horizontal_label)
    axes[1].set_ylabel(vertical_label)
    plt.colorbar(im2, ax=axes[1], label="Field Magnitude")

    # Plot error
    im3 = axes[2].imshow(error, cmap="jet", origin="lower", extent=extent)
    axes[2].set_title(f"Error ({plane_type} Plane)")
    axes[2].set_xlabel(horizontal_label)
    axes[2].set_ylabel(vertical_label)
    plt.colorbar(im3, ax=axes[2], label="Absolute Error")

    # Add metrics if provided
    if metrics is not None:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        fig.suptitle(f"{title}\n{metrics_str}", fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    if not show:
        plt.close(fig)
