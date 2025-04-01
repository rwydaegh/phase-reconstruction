import logging
import os

import matplotlib.pyplot as plt
import numpy as np

# Ensure Axes3D is imported for 3D projection
from mpl_toolkits.mplot3d import Axes3D

# Import necessary functions from the project
from src.create_channel_matrix import create_channel_matrix
from src.utils.field_utils import compute_fields
from src.utils.preprocess_pointcloud import get_tangent_vectors  # For calculating tangents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_two_source_simulation():
    """
    Runs a barebones simulation with two opposing sources in the measurement plane.
    Calculates and visualizes the resulting field magnitude and source vectors in a single 3D plot.
    """
    # --- Simulation Parameters ---
    wavelength = 10.7e-3  # 28GHz wavelength in meters
    k = 2 * np.pi / wavelength
    plane_size = 1.0  # 1m x 1m measurement plane
    resolution = 50  # Grid resolution (reduced for faster plotting)
    measurement_direction = np.array([0.0, 1.0, 0.0])  # Measure Y-component
    source_dist = 0.5  # Distance of sources from origin along X-axis
    output_dir = "plots"  # Simple output directory
    vector_scale = 0.2  # Scale for normal/tangent vectors

    logger.info("--- Two Source Simulation ---")
    logger.info(f"Wavelength: {wavelength:.4f} m, k: {k:.2f}")
    logger.info(
        f"Measurement Plane: {plane_size}x{plane_size} m, {resolution}x{resolution} points, Z=0"
    )
    logger.info(f"Sources at (+/-{source_dist}, 0, 0)")
    logger.info(f"Measurement Direction: {measurement_direction}")

    # --- Define Sources ---
    points = np.array([[-source_dist, 0.0, 0.0], [source_dist, 0.0, 0.0]])
    N_c = points.shape[0]
    logger.info(f"Source points shape: {points.shape}")

    # Define normals pointing towards the origin
    temp_normals = np.array(
        [
            [1.0, 0.0, 0.0],  # Normal for source at x=-0.5 points towards +X
            [-1.0, 0.0, 0.0],  # Normal for source at x=+0.5 points towards -X
        ]
    )
    logger.info("Calculating tangents (using inward-pointing normals)...")
    tangents1, tangents2 = get_tangent_vectors(temp_normals)
    logger.info(f"Tangents1 shape: {tangents1.shape}")
    logger.info(f"Tangents2 shape: {tangents2.shape}")

    # Define currents (shape 2*N_c)
    # Activate only the first component (t1) for both sources with amplitude 1
    currents = np.zeros(2 * N_c, dtype=complex)
    currents[0] = 1.0  # Source 1, t1 component
    # currents[1] = 0.0 # Source 1, t2 component (default zero)
    currents[2] = 1.0  # Source 2, t1 component (in phase)
    # currents[3] = 0.0 # Source 2, t2 component (default zero)
    logger.info(f"Currents shape: {currents.shape}, Values: {currents}")

    # --- Define Measurement Plane ---
    logger.info("Creating measurement plane...")
    x_mp = np.linspace(-plane_size / 2, plane_size / 2, resolution)
    y_mp = np.linspace(-plane_size / 2, plane_size / 2, resolution)
    X_mp, Y_mp = np.meshgrid(x_mp, y_mp)
    Z_mp = np.zeros_like(X_mp)  # Measurement plane at Z=0
    measurement_plane = np.stack([X_mp, Y_mp, Z_mp], axis=-1)

    # --- Calculate Field ---
    logger.info("Creating channel matrix...")
    H = create_channel_matrix(
        points, tangents1, tangents2, measurement_plane, measurement_direction, k
    )
    logger.info(f"Channel matrix H shape: {H.shape}")

    logger.info("Calculating ground truth field...")
    true_field = compute_fields(
        points=points,
        tangents1=tangents1,
        tangents2=tangents2,
        currents=currents,
        measurement_plane=measurement_plane,
        measurement_direction=measurement_direction,
        k=k,
        channel_matrix=H,
    )
    logger.info(f"Calculated field shape: {true_field.shape}")

    true_field_mag_2d = np.abs(true_field).reshape(resolution, resolution)

    # --- Visualize Combined 3D Plot ---
    logger.info("Visualizing combined 3D plot...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 1. Plot Field Magnitude on Measurement Plane (Z=0)
    # Use plot_surface with color mapping to magnitude
    # Normalize magnitude for color, ensuring scale starts at 0
    field_max = np.max(true_field_mag_2d)
    norm = plt.Normalize(
        vmin=0, vmax=field_max if field_max > 1e-9 else 1.0
    )  # Avoid error if max is 0
    colors = plt.cm.viridis(norm(true_field_mag_2d))

    ax.plot_surface(
        X_mp,
        Y_mp,
        Z_mp,
        facecolors=colors,
        rstride=1,
        cstride=1,  # Plot all points
        linewidth=0,
        antialiased=False,
        shade=False,  # Use facecolors directly
        alpha=0.7,  # Make slightly transparent
    )
    # Add a color bar manually with correct normalization
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    m.set_array([])  # Pass empty array, norm handles the scaling
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, label="Field Magnitude |Ey|")

    # 2. Plot Source Points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="red",
        marker="x",
        s=100,
        depthshade=False,
        label="Sources",
    )

    # 3. Plot Vectors for Sources
    # Normal vectors (blue)
    ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        temp_normals[:, 0],
        temp_normals[:, 1],
        temp_normals[:, 2],
        length=vector_scale,
        normalize=True,
        color="blue",
        label="Normals (n)",
        alpha=0.9,
    )
    # Tangent 1 vectors (red)
    ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        tangents1[:, 0],
        tangents1[:, 1],
        tangents1[:, 2],
        length=vector_scale,
        normalize=True,
        color="red",
        label="Tangent 1 (t1)",
        alpha=0.9,
    )
    # Tangent 2 vectors (green)
    ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        tangents2[:, 0],
        tangents2[:, 1],
        tangents2[:, 2],
        length=vector_scale,
        normalize=True,
        color="green",
        label="Tangent 2 (t2)",
        alpha=0.9,
    )

    # --- Plot Settings ---
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Two Sources: Field Magnitude and Source Vectors")
    ax.legend()
    # Set view angle
    ax.view_init(elev=20.0, azim=-65)
    # Set limits to keep plot focused
    ax.set_xlim([-plane_size / 2 - 0.1, plane_size / 2 + 0.1])
    ax.set_ylim([-plane_size / 2 - 0.1, plane_size / 2 + 0.1])
    ax.set_zlim([-vector_scale - 0.1, vector_scale + 0.1])  # Adjust Z limits based on vector scale

    plt.tight_layout()

    # Save 3D plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename_3d = os.path.join(output_dir, "two_source_combined_3D.png")
    plt.savefig(filename_3d, dpi=150)
    logger.info(f"Saved combined 3D visualization to {filename_3d}")
    plt.close(fig)  # Close the 3D plot window

    # --- Visualize 2D Field Plot ---
    logger.info("Visualizing 2D field magnitude...")
    fig_2d, ax_2d = plt.subplots(figsize=(8, 7))
    im_2d = ax_2d.imshow(
        true_field_mag_2d,
        cmap="viridis",
        origin="lower",
        extent=[-plane_size / 2, plane_size / 2, -plane_size / 2, plane_size / 2],
        interpolation="nearest",
        vmin=0,  # Explicitly set minimum for color scale
    )
    ax_2d.scatter(
        points[:, 0], points[:, 1], c="red", marker="x", s=50, label="Sources"
    )  # Show source locations
    ax_2d.set_title("Field Magnitude from Two Sources (Z=0 Plane)")
    ax_2d.set_xlabel("X (m)")
    ax_2d.set_ylabel("Y (m)")
    ax_2d.legend()
    plt.colorbar(im_2d, ax=ax_2d, label="Field Magnitude |Ey|")
    plt.tight_layout()

    # Save 2D plot
    filename_2d = os.path.join(output_dir, "two_source_field_2D.png")
    plt.savefig(filename_2d, dpi=150)
    logger.info(f"Saved 2D field visualization to {filename_2d}")
    plt.close(fig_2d)  # Close the 2D plot window


if __name__ == "__main__":
    run_two_source_simulation()
