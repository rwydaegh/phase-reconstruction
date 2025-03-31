"""
General visualization utility functions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


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