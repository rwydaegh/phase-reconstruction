import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


def visualize_iteration_history(
    points: np.ndarray,
    H: np.ndarray,
    coefficient_history: np.ndarray,
    field_history: np.ndarray,
    resolution: int,
    measurement_plane: np.ndarray,
    show_plot: bool = True,
    output_file: Optional[str] = "gs_animation.gif",
    frame_skip: int = 3,
    perturbation_iterations: Optional[List[int]] = None,
    restart_iterations: Optional[List[int]] = None,
    convergence_threshold: float = 1e-3,
    measured_magnitude: Optional[np.ndarray] = None,
) -> None:
    """Create animation showing the evolution of field reconstruction over GS iterations.

    Args:
        points: Point coordinates
        H: Channel matrix
        coefficient_history: History of coefficients over iterations
        field_history: History of reconstructed fields over iterations
        resolution: Resolution of measurement grid
        measurement_plane: Measurement plane coordinates
        show_plot: Whether to display the plot
        output_file: Path to save the animation
        frame_skip: Number of frames to skip for faster animation
    """
    # Determine plane type and set appropriate axis labels
    x_min, x_max = np.min(measurement_plane[:, :, 0]), np.max(measurement_plane[:, :, 0])
    y_min, _ = np.min(measurement_plane[:, :, 1]), np.max(measurement_plane[:, :, 1]) # y_max unused
    _, _ = np.min(measurement_plane[:, :, 2]), np.max(measurement_plane[:, :, 2])
    # z_min, z_max unused

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
        horizontal_coords = measurement_plane[0, :, 0]  # X values
        vertical_coords = measurement_plane[:, 0, 2]  # Z values
        horizontal_label = "X (m)"
        vertical_label = "Z (m)"
        plane_type = "XZ"
    else:  # XY plane (default)
        horizontal_coords = measurement_plane[0, :, 0]  # X values
        vertical_coords = measurement_plane[:, 0, 1]  # Y values
        horizontal_label = "X (m)"
        vertical_label = "Y (m)"
        plane_type = "XY"

    # Create figure with a wider size for better visibility
    fig = plt.figure(figsize=(14, 8))

    # Three panels in a 1x3 layout: True Field, Reconstructed Field, Combined Errors
    ax3 = plt.subplot(1, 3, 1)  # True field (if available)
    ax1 = plt.subplot(1, 3, 2)  # Reconstructed field magnitude
    ax2 = plt.subplot(1, 3, 3)  # Combined error metrics

    # Calculate global max for consistent colormap scaling
    field_mag = np.abs(field_history[0]).reshape(resolution, resolution)
    global_max = np.max(
        [
            np.max(np.abs(field_history[i]).reshape(resolution, resolution))
            for i in range(len(field_history))
        ]
    )

    # Set extent for 2D plots using the detected coordinates
    extent = [
        horizontal_coords.min(),
        horizontal_coords.max(),
        vertical_coords.min(),
        vertical_coords.max(),
    ]

    # Setup for true field visualization (first panel)
    if measured_magnitude is not None:
        # If we have measured magnitude data, reshape it for display
        measured_field_2d = measured_magnitude.reshape(resolution, resolution)
        global_max = max(global_max, np.max(measured_field_2d))  # Update global max
        im3 = ax3.imshow(
            measured_field_2d,
            cmap="viridis",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=global_max,
        )  # Use consistent color scale
        ax3.set_title(f"True Field (Measured) ({plane_type} Plane)")
        ax3.set_xlabel(horizontal_label)
        ax3.set_ylabel(vertical_label)
        plt.colorbar(im3, ax=ax3, label="Field Magnitude")
    else:
        # If no measured data, show a placeholder
        im3 = ax3.imshow(
            np.zeros((resolution, resolution)),
            cmap="viridis",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=global_max,
        )
        ax3.set_title(f"No Reference Data Available ({plane_type} Plane)")
        ax3.set_xlabel(horizontal_label)
        ax3.set_ylabel(vertical_label)
        plt.colorbar(im3, ax=ax3, label="Field Magnitude")

    # Setup for reconstructed field magnitude (second panel)
    im1 = ax1.imshow(
        field_mag, cmap="viridis", origin="lower", extent=extent, vmin=0, vmax=global_max
    )  # Use consistent color scale
    ax1.set_title(f"Reconstructed Field ({plane_type} Plane)")
    ax1.set_xlabel(horizontal_label)
    ax1.set_ylabel(vertical_label)
    plt.colorbar(im1, ax=ax1, label="Field Magnitude")

    # Setup for the error metrics panel (second panel)
    ax2.grid(True)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error (log scale)")
    ax2.set_title("Error Metrics")

    # Compute two error metrics:
    # 1. Iteration-to-iteration change (convergence rate)
    iteration_changes = np.zeros(len(field_history))
    for i in range(len(field_history)):
        field_i = field_history[i]  # Current complex field (includes phase)
        field_prev = field_history[max(0, i - 1)]  # Previous complex field (includes phase)
        # RMSE between consecutive complex fields (captures both magnitude and phase changes)
        iteration_changes[i] = np.sqrt(np.mean(np.abs(field_i - field_prev) ** 2))

    # 2. Actual error (RMSE vs measured magnitude)
    actual_errors = np.zeros(len(field_history))
    if measured_magnitude is not None:
        measured_magnitude_norm = np.linalg.norm(measured_magnitude)
        for i in range(len(field_history)):
            simulated_magnitude = np.abs(field_history[i])
            actual_errors[i] = np.linalg.norm(simulated_magnitude - measured_magnitude) / measured_magnitude_norm
    else:
        # If no measured magnitude, use iteration changes as a proxy
        actual_errors = iteration_changes

    # Plot error metrics
    (line_changes,) = ax2.plot(range(len(field_history)), iteration_changes, label="Iter Change (RMSE)")
    (line_errors,) = ax2.plot(range(len(field_history)), actual_errors, label="Error vs True")
    ax2.set_yscale("log")

    # Add convergence threshold line
    ax2.axhline(
        y=convergence_threshold, color="r", linestyle="--", label=f"Threshold ({convergence_threshold:.1e})"
    )

    # Add vertical lines for perturbations and restarts
    perturbation_lines = []
    if perturbation_iterations:
        for p_iter in perturbation_iterations:
            perturbation_lines.append(ax2.axvline(x=p_iter, color="g", linestyle=":", alpha=0.7))

    restart_lines = []
    if restart_iterations:
        for r_iter in restart_iterations:
            restart_lines.append(ax2.axvline(x=r_iter, color="m", linestyle="-.", alpha=0.7))

    # Create legend for error lines and threshold
    legend_elements = [line_changes, line_errors, ax2.lines[-1]] # Get the threshold line
    legend_labels = ["Iter Change", "Error vs True", f"Threshold ({convergence_threshold:.1e})"]

    # Add perturbation/restart lines to legend if they exist
    if perturbation_iterations and perturbation_lines:
        legend_elements.append(perturbation_lines[0])
        legend_labels.append("Perturbation")

    if restart_iterations and restart_lines:
        legend_elements.append(restart_lines[0])
        legend_labels.append("Restart")

    # Add combined legend
    ax2.legend(legend_elements, legend_labels, loc="upper right")

    # Text for iteration number with more visible styling
    iter_text = ax1.text(
        0.02,
        0.98,
        "Iteration: 0",
        transform=ax1.transAxes,
        verticalalignment="top",
        color="white",
        fontweight="bold",
        bbox={"facecolor": "black", "alpha": 0.5},
    )

    # Smart adaptation of frame_skip based on total iteration count
    total_frames = len(field_history)

    # Calculate optimal frame_skip to target 60-100 frames
    # 1. For very short histories (<60), keep all frames
    # 2. For medium histories, aim for ~60-80 frames
    # 3. For very long histories (>1000), aim for ~100 frames
    if total_frames <= 60:
        # For short histories, keep all frames
        adaptive_frame_skip = 1
    elif total_frames <= 200:
        # For medium histories, skip a few frames to get around 60-80 frames
        adaptive_frame_skip = max(1, total_frames // 60)
    else:
        # For longer histories, be more aggressive with skipping
        # as iterations get higher, target ~100 frames
        adaptive_frame_skip = max(2, total_frames // 100)

    # If user specified a frame_skip, use the larger value
    final_frame_skip = max(frame_skip, adaptive_frame_skip)

    # Select frames based on final_frame_skip
    selected_frames = range(0, total_frames, final_frame_skip)
    selected_field_history = field_history[selected_frames]
    num_selected_frames = len(selected_frames)

    # Adapt animation interval based on frame count:
    # - Fewer frames -> slower animation (longer interval)
    # - More frames -> faster animation (shorter interval)
    if num_selected_frames < 30:
        # interval = 200  # Slower for fewer frames # Unused
        pass
    elif num_selected_frames < 60:
        # interval = 150  # Medium speed # Unused
        pass
    else:
        # interval = 100  # Faster for many frames # Unused
        pass

    # Create animation function
    def update(i):
        # Get actual frame index
        frame = selected_frames[i]

        # Update field magnitude
        field_mag = np.abs(selected_field_history[i]).reshape(resolution, resolution)
        im1.set_array(field_mag)

        # Update iteration-to-iteration change plot
        line_changes.set_data(range(frame + 1), iteration_changes[: frame + 1])

        # Update actual error plot
        line_errors.set_data(range(frame + 1), actual_errors[: frame + 1])

        # Update text
        iter_text.set_text(f"Iteration: {frame}")

        return im1, line_changes, line_errors, iter_text

    # Create animation with fewer frames and faster display interval
    anim = FuncAnimation(
        fig, update, frames=len(selected_frames), interval=100, blit=True
    )  # Faster frame rate (100ms vs 200ms)

    # Save animation if output file specified
    if output_file:
        # Use higher FPS for smoother, faster animation
        adjusted_fps = max(10, 30 // final_frame_skip)  # Higher base FPS (10-30 vs 5-15)
        anim.save(output_file, writer="pillow", fps=adjusted_fps)
        print(
            f"Saved animation to {output_file} (skipping {final_frame_skip-1} frames, fps={adjusted_fps})"
        )

    # Show plot if requested
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()