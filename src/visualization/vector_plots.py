import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
# Ensure Axes3D is imported for 3D projection
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

def visualize_vectors(
    points: np.ndarray,
    normals: np.ndarray,
    tangents1: np.ndarray,
    tangents2: np.ndarray,
    num_vectors: int = 50, # Changed from 'step' to 'num_vectors'
    scale: float = 0.1,
    output_dir: Optional[str] = None,
    show_plot: bool = True,
    filename: str = "vector_visualization.png"
) -> None:
    """
    Visualizes points with their corresponding normal and tangent vectors using arrows.
    Selects a random subset of points to display vectors for clarity.

    Args:
        points: Point coordinates (N, 3).
        normals: Normal vectors (N, 3).
        tangents1: First tangent vectors (N, 3).
        tangents2: Second tangent vectors (N, 3).
        num_vectors: Approximate number of points for which to plot vectors.
        scale: Length scale for the vector arrows.
        output_dir: Directory to save the plot image.
        show_plot: Whether to display the plot interactively.
        filename: Name for the saved plot file.
    """
    num_total_points = points.shape[0]
    if num_total_points == 0:
        logger.warning("No points provided for vector visualization.")
        return

    # Determine the number of points to sample
    k = min(num_vectors, num_total_points) # Cannot sample more than available points
    if k <= 0:
         logger.warning(f"num_vectors requested ({num_vectors}) is zero or negative. No vectors will be plotted.")
         # Set indices to empty to skip quiver plots
         indices = np.array([], dtype=int)
    else:
        # Generate random indices without replacement
        indices = np.random.choice(num_total_points, size=k, replace=False)

    # Subsample points and vectors using random indices
    points_sub = points[indices]
    normals_sub = normals[indices]
    tangents1_sub = tangents1[indices]
    tangents2_sub = tangents2[indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all original points lightly
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='grey', marker='.', s=2, label='All Points', alpha=0.1)

    # Plot vectors only if indices were selected
    if indices.size > 0:
        # Plot normal vectors (blue)
        ax.quiver(
            points_sub[:, 0], points_sub[:, 1], points_sub[:, 2],
            normals_sub[:, 0], normals_sub[:, 1], normals_sub[:, 2],
            length=scale, normalize=True, color='blue', label='Normals (n)', alpha=0.8
        )

        # Plot first tangent vectors (red)
        ax.quiver(
            points_sub[:, 0], points_sub[:, 1], points_sub[:, 2],
            tangents1_sub[:, 0], tangents1_sub[:, 1], tangents1_sub[:, 2],
            length=scale, normalize=True, color='red', label='Tangent 1 (t1)', alpha=0.8
        )

        # Plot second tangent vectors (green)
        ax.quiver(
            points_sub[:, 0], points_sub[:, 1], points_sub[:, 2],
            tangents2_sub[:, 0], tangents2_sub[:, 1], tangents2_sub[:, 2],
            length=scale, normalize=True, color='green', label='Tangent 2 (t2)', alpha=0.8
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Normals and Tangents Visualization (Sampled {k} vectors, Scale={scale})")
    ax.legend()
    # Try to set equal aspect ratio
    try:
        ax.set_aspect('equal', adjustable='box')
    except NotImplementedError:
        logger.warning("3D aspect ratio 'equal' not fully supported. Plot may look distorted.")


    plt.tight_layout()

    # Save or show plot
    if output_dir:
        output_path = Path(output_dir) / filename
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved vector visualization to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)