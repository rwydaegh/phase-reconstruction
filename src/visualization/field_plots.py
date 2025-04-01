import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection

logger = logging.getLogger(__name__)


def visualize_fields(
    points: np.ndarray,
    currents: np.ndarray,
    measurement_plane: np.ndarray,
    true_field_2d: np.ndarray,
    measured_magnitude_2d: np.ndarray,
    reconstructed_field_2d: np.ndarray,
    rmse: float,
    correlation: float,
    show_plot: bool = True,
    output_dir: str = "/figs",
) -> None:
    """Visualize the true field, measured magnitude, and reconstructed field.

    Args:
        points: Point coordinates
        currents: Current amplitudes
        measurement_plane: Measurement plane coordinates
        true_field_2d: True complex field (2D)
        measured_magnitude_2d: Measured field magnitude (2D)
        reconstructed_field_2d: Reconstructed complex field (2D)
        rmse: Normalized RMSE between true and reconstructed fields
        correlation: Correlation between true and reconstructed fields
        show_plot: Whether to display the plot
        output_file: Path to save the output image
    """
    # Get resolution
    # resolution = true_field_2d.shape[0] # Unused

    # Determine plane type and set appropriate axis labels
    x_min, x_max = np.min(measurement_plane[:, :, 0]), np.max(measurement_plane[:, :, 0])
    y_min, y_max = np.min(measurement_plane[:, :, 1]), np.max(measurement_plane[:, :, 1])
    z_min, z_max = np.min(measurement_plane[:, :, 2]), np.max(measurement_plane[:, :, 2])

    # Calculate spread in each dimension to identify constant dimensions
    x_spread = x_max - x_min
    # y_spread = y_max - y_min # Unused
    # z_spread = z_max - z_min # Unused

    # Determine plane type and set coordinates and labels
    if np.isclose(x_spread, 0) or np.allclose(measurement_plane[:, :, 0], x_min):  # YZ plane
        # For YZ plane, Y is horizontal (dim 1) and Z is vertical (dim 2)
        # Get unique values to ensure we have the full range
        horizontal_coords = np.unique(measurement_plane[:, :, 1])  # Unique Y values
        vertical_coords = np.unique(measurement_plane[:, :, 2])  # Unique Z values
        horizontal_label = "Y (m)"
        vertical_label = "Z (m)"
        plane_type = "YZ"
    elif np.allclose(measurement_plane[:, :, 1], y_min):  # XZ plane
        # For XZ plane, X is horizontal (dim 0) and Z is vertical (dim 2)
        horizontal_coords = np.unique(measurement_plane[:, :, 0])  # Unique X values
        vertical_coords = np.unique(measurement_plane[:, :, 2])  # Unique Z values
        horizontal_label = "X (m)"
        vertical_label = "Z (m)"
        plane_type = "XZ"
    else:  # XY plane (default)
        # For XY plane, X is horizontal (dim 0) and Y is vertical (dim 1)
        horizontal_coords = np.unique(measurement_plane[:, :, 0])  # Unique X values
        vertical_coords = np.unique(measurement_plane[:, :, 1])  # Unique Y values
        horizontal_label = "X (m)"
        vertical_label = "Y (m)"
        plane_type = "XY"

    # Create figure
    fig = plt.figure(figsize=(15, 12))

    # Plot 1: Point cloud with currents - directly create as 3D
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")

    # Create the other 2D subplots manually
    ax1 = fig.add_subplot(2, 2, 2)  # Top right
    ax2 = fig.add_subplot(2, 2, 3)  # Bottom left
    ax3 = fig.add_subplot(2, 2, 4)  # Bottom right
    # Color points by current magnitude with improved visualization
    # Calculate combined magnitude per point from the two components in 'currents'
    if currents.shape[0] == 2 * points.shape[0]:
        coeffs_t1 = currents[0::2]  # Shape (N_c,)
        coeffs_t2 = currents[1::2]  # Shape (N_c,)
        current_mags_per_point = np.sqrt(
            np.abs(coeffs_t1) ** 2 + np.abs(coeffs_t2) ** 2
        )  # Shape (N_c,)
    elif currents.shape[0] == points.shape[0]:
        # Handle case where scalar currents might still be passed (e.g., from older data)
        logger.warning(
            "visualize_fields received currents with shape matching points, assuming scalar values."
        )
        current_mags_per_point = np.abs(currents)
    else:
        raise ValueError(
            f"Shape mismatch: currents shape {currents.shape} not compatible with points shape {points.shape}"
        )

    if current_mags_per_point.shape[0] != points.shape[0]:
        raise ValueError(
            f"Mismatch between points ({points.shape[0]}) and calculated current magnitudes ({current_mags_per_point.shape[0]})"
        )

    max_mag = np.max(current_mags_per_point)
    if max_mag > 1e-9:  # Use a small tolerance
        normalized_mags = current_mags_per_point / max_mag
        sizes = 0.5 + 150 * normalized_mags**2
        alphas = 0.2 + 0.8 * normalized_mags
    else:
        sizes = 0.5
        alphas = 0.2

    scatter = ax3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=current_mags_per_point,  # Use combined magnitude for color
        s=sizes,
        cmap="plasma",
        alpha=alphas,
    )

    # Add small black dots for all points to show the environment
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], color="black", s=1, alpha=0.3)

    ax3d.set_title("Point Cloud with Source Currents")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    # Add measurement plane visualization
    # Extract the extent of the measurement plane
    x_min, x_max = np.min(measurement_plane[:, :, 0]), np.max(measurement_plane[:, :, 0])
    y_min, y_max = np.min(measurement_plane[:, :, 1]), np.max(measurement_plane[:, :, 1])
    z_min, z_max = np.min(measurement_plane[:, :, 2]), np.max(measurement_plane[:, :, 2])

    # Create a rectangular face based on the extent
    if np.allclose(measurement_plane[:, :, 0], x_min):  # YZ plane
        xx, yy = np.meshgrid([x_min, x_min], [y_min, y_max])
        zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
    elif np.allclose(measurement_plane[:, :, 1], y_min):  # XZ plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_min])
        zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
    else:  # XY plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
        zz = np.ones_like(xx) * z_min

    # Plot the face with semi-transparency
    ax3d.plot_surface(xx, yy, zz, alpha=0.3, color="cyan", edgecolor="blue")

    # Set equal aspect ratio for 3D plot
    ax3d.set_box_aspect([1, 1, 1])

    plt.colorbar(scatter, ax=ax3d, label="Current Magnitude")

    # Find global min and max for consistent colorbar scaling across field plots
    true_field_abs = np.abs(true_field_2d)
    reconstructed_field_abs = np.abs(reconstructed_field_2d)

    # Calculate global min and max across all field arrays
    global_min = min(
        np.min(true_field_abs), np.min(measured_magnitude_2d), np.min(reconstructed_field_abs)
    )
    global_max = max(
        np.max(true_field_abs), np.max(measured_magnitude_2d), np.max(reconstructed_field_abs)
    )

    # Set extent for 2D plots using the detected coordinates
    extent = [
        horizontal_coords.min(),
        horizontal_coords.max(),
        vertical_coords.min(),
        vertical_coords.max(),
    ]

    # Plot 2: True field magnitude with normalized colorbar (top right)
    im1 = ax1.imshow(
        true_field_abs,
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=global_min,
        vmax=global_max,
    )
    ax1.set_title(f"True Field Magnitude ({plane_type} Plane)")
    ax1.set_xlabel(horizontal_label)
    ax1.set_ylabel(vertical_label)
    plt.colorbar(im1, ax=ax1, label="Field Magnitude")

    # Plot 3: Measured field magnitude with normalized colorbar (bottom left)
    im2 = ax2.imshow(
        measured_magnitude_2d,
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=global_min,
        vmax=global_max,
    )
    ax2.set_title(f"Measured Field Magnitude ({plane_type} Plane)")
    ax2.set_xlabel(horizontal_label)
    ax2.set_ylabel(vertical_label)
    plt.colorbar(im2, ax=ax2, label="Field Magnitude")

    # Plot 4: Reconstructed field magnitude with normalized colorbar (bottom right)
    im3 = ax3.imshow(
        reconstructed_field_abs,
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=global_min,
        vmax=global_max,
    )
    ax3.set_title(f"Reconstructed Field Magnitude ({plane_type} Plane)")
    ax3.set_xlabel(horizontal_label)
    ax3.set_ylabel(vertical_label)
    plt.colorbar(im3, ax=ax3, label="Field Magnitude")

    # Add overall title with metrics
    fig.suptitle(
        f"Field Reconstruction Comparison\nRMSE: {rmse:.4f}, Correlation: {correlation:.4f}",
        fontsize=16,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle

    # Save or show plot
    if output_dir:
        filename = Path(output_dir) / f"field_visualization_{plane_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {filename}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
