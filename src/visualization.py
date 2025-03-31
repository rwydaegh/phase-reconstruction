from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def visualize_field(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    filename: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """Visualize field magnitude.

    Args:
        field: Complex field to visualize
        x: X-coordinates of the field grid
        y: Y-coordinates of the field grid
        title: Plot title
        filename: If provided, save the figure to this path
        show: Whether to display the figure (if False, closes the figure after saving)

    Returns:
        2D field magnitude array
    """
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

    if not show:
        plt.close(fig)

    return field_2d


def visualize_point_cloud(
    points: np.ndarray,
    currents: Optional[np.ndarray] = None,
    title: str = "Point Cloud Visualization",
    filename: Optional[str] = None,
    show: bool = True,
    highlight_indices: Optional[List[int]] = None,
    room_size: float = 2.0,
    measurement_plane: Optional[np.ndarray] = None,
) -> None:
    """Visualize 3D point cloud with optional current magnitudes.

    Args:
        points: Point coordinates, shape (num_points, 3)
        currents: Optional complex currents, shape (num_points,)
        title: Plot title
        filename: If provided, save the figure to this path
        show: Whether to display the figure
        highlight_indices: Optional list of indices to highlight
        room_size: Size of the room for axis limits
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Default color and size
    colors = "blue"
    sizes = 20

    # If currents are provided, use them for coloring
    if currents is not None:
        colors = np.abs(currents)
        sizes = 20 + 100 * (colors / np.max(colors) if np.max(colors) > 0 else 0)

    # Plot all points
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=colors, s=sizes, cmap="viridis", alpha=0.7
    )

    # Highlight specific points if requested
    if highlight_indices is not None and len(highlight_indices) > 0:
        ax.scatter(
            points[highlight_indices, 0],
            points[highlight_indices, 1],
            points[highlight_indices, 2],
            color="red",
            s=50,
            marker="*",
        )

    # Add colorbar if currents are provided
    if currents is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Current Magnitude")

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # Set axis limits
    ax.set_xlim(0, room_size)
    ax.set_ylim(0, room_size)
    ax.set_zlim(0, room_size)

    # Add measurement plane if provided
    if measurement_plane is not None:
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
        ax.plot_surface(xx, yy, zz, alpha=0.3, color="cyan", edgecolor="blue")

    # Set equal aspect ratio to avoid distortion
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    if not show:
        plt.close(fig)


# Function visualize_fields moved to src/visualization/field_plots.py


# Function visualize_iteration_history moved to src/visualization/history_plots.py


# Function visualize_current_and_field_history moved to src/visualization/history_plots.py


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
        true_mag, cmap="viridis", origin="lower", extent=extent, vmin=global_min, vmax=global_max
    )
    axes[0].set_title(f"True Field ({plane_type} Plane)")
    axes[0].set_xlabel(horizontal_label)
    axes[0].set_ylabel(vertical_label)
    plt.colorbar(im1, ax=axes[0], label="Field Magnitude")

    # Plot reconstructed field with normalized colorbar
    im2 = axes[1].imshow(
        recon_mag, cmap="viridis", origin="lower", extent=extent, vmin=global_min, vmax=global_max
    )
    axes[1].set_title(f"Reconstructed Field ({plane_type} Plane)")
    axes[1].set_xlabel(horizontal_label)
    axes[1].set_ylabel(vertical_label)
    plt.colorbar(im2, ax=axes[1], label="Field Magnitude")

    # Plot error
    im3 = axes[2].imshow(error, cmap="hot", origin="lower", extent=extent)
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
