# src/visualization/multi_plane_plots.py
import logging
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D  # For custom legend

# Import necessary for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# Import utility for determining plane type/labels if available
# from .utils import get_plane_info # Example if refactored
# For now, duplicate or simplify the logic

logger = logging.getLogger(__name__)


def animate_convergence_errors(
    stats: Dict,  # GS stats dictionary containing history
    convergence_threshold: float,
    output_dir: str,
    show_plot: bool = False,
    animation_filename: str = "gs_convergence_errors.gif",
    frame_skip: int = 1,  # Default to showing more frames initially
) -> None:
    """
    Creates an animation showing the evolution of overall and per-training-plane RMSE.

    Args:
        stats: Dictionary returned by holographic_phase_retrieval, must contain
               'overall_rmse_history' and 'per_train_plane_rmse_history'.
        convergence_threshold: The convergence threshold value to plot.
        output_dir: Directory to save the animation.
        show_plot: Whether to display the plot interactively.
        animation_filename: Filename for the saved animation.
        frame_skip: Number of frames to skip for faster animation.
    """
    overall_rmse_history = stats.get("overall_rmse_history")
    per_plane_rmse_history = stats.get("per_train_plane_rmse_history")
    perturbation_iterations = stats.get("perturbation_iterations", [])

    if overall_rmse_history is None or per_plane_rmse_history is None:
        logger.warning("RMSE history not found in stats. Skipping convergence animation.")
        return

    num_iterations = len(overall_rmse_history)
    if num_iterations == 0:
        logger.warning("RMSE history is empty. Skipping convergence animation.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plotting Logic ---
    iterations = range(num_iterations)

    # Plot overall RMSE
    (line_overall,) = ax.plot(
        iterations, overall_rmse_history, label="Overall Train RMSE", linewidth=2, color="black"
    )

    # Plot per-plane RMSE
    plane_lines = {}
    colors = plt.cm.jet(np.linspace(0, 1, len(per_plane_rmse_history)))
    for idx, (plane_name, history) in enumerate(per_plane_rmse_history.items()):
        if len(history) == num_iterations:  # Ensure history length matches
            (line,) = ax.plot(
                iterations, history, label=f"{plane_name} RMSE", color=colors[idx], linestyle="--"
            )
            plane_lines[plane_name] = line
        else:
            logger.warning(
                f"Length mismatch for plane '{plane_name}' RMSE history ({len(history)}) vs overall ({num_iterations}). Skipping."
            )

    # Add convergence threshold line
    ax.axhline(
        y=convergence_threshold,
        color="r",
        linestyle=":",
        linewidth=2,
        label=f"Threshold ({convergence_threshold:.1e})",
    )

    # Add vertical lines for perturbations
    perturb_lines = []
    if perturbation_iterations:
        for p_iter in perturbation_iterations:
            if p_iter < num_iterations:  # Ensure marker is within plot range
                perturb_lines.append(
                    ax.axvline(x=p_iter, color="g", linestyle="-.", alpha=0.7, label="_nolegend_")
                )  # Use _nolegend_ initially

    # Create legend (handle perturbation marker separately if needed)
    handles, labels = ax.get_legend_handles_labels()
    if perturb_lines:
        # Add a single perturbation marker to the legend
        handles.append(perturb_lines[0])
        labels.append("Perturbation")
    ax.legend(handles, labels, loc="upper right")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized RMSE (log scale)")
    ax.set_title("GS Convergence - Training Set RMSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_xlim(0, num_iterations - 1 if num_iterations > 1 else 1)  # Adjust xlim

    # --- Animation Setup ---
    iter_text = ax.text(
        0.02,
        0.98,
        "Iteration: 0",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # Use frame skipping
    total_frames = num_iterations
    final_frame_skip = max(1, frame_skip)
    selected_indices = range(0, total_frames, final_frame_skip)
    num_anim_frames = len(selected_indices)

    def update(frame_idx):
        iter_index = selected_indices[frame_idx]  # Get the actual iteration index

        # Update overall line
        line_overall.set_data(iterations[: iter_index + 1], overall_rmse_history[: iter_index + 1])

        # Update per-plane lines
        for plane_name, line in plane_lines.items():
            line.set_data(
                iterations[: iter_index + 1], per_plane_rmse_history[plane_name][: iter_index + 1]
            )

        # Update iteration text
        iter_text.set_text(f"Iteration: {iter_index}")

        # Adjust y-limits dynamically? Optional, can be slow.
        # current_min = min(overall_rmse_history[:iter_index+1])
        # current_max = max(overall_rmse_history[:iter_index+1])
        # ax.set_ylim(max(1e-9, current_min * 0.5), current_max * 1.5) # Example dynamic limits

        artists = [line_overall, iter_text] + list(plane_lines.values())
        return artists

    anim = FuncAnimation(fig, update, frames=num_anim_frames, interval=100, blit=True)

    # Save animation
    output_path = os.path.join(output_dir, animation_filename)
    try:
        # Adjust FPS based on skip rate for consistent perceived speed
        adjusted_fps = max(5, 20 // final_frame_skip)
        anim.save(output_path, writer="pillow", fps=adjusted_fps)
        logger.info(
            f"Saved convergence animation to {output_path} (Frames: {num_anim_frames}, FPS: {adjusted_fps})"
        )
    except Exception as e:
        logger.error(f"Failed to save convergence animation: {e}", exc_info=True)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def animate_layout_and_currents(
    history: List[Dict],  # Detailed history from HPR
    points_perturbed: np.ndarray,
    measurement_planes: List[Dict],  # List of processed plane dicts
    output_dir: str,
    show_plot: bool = False,
    animation_filename: str = "layout_currents_history.gif",
    frame_skip: int = 3,
    rotation_speed_factor: float = 1.0,  # Adjust speed of rotation
) -> None:
    """
    Creates a rotating 3D animation showing source point currents and plane layout.

    Args:
        history: List of dictionaries per iteration from HPR, containing at least 'coefficients'.
        points_perturbed: Coordinates of the source points used for reconstruction (N, 3).
        measurement_planes: List of processed plane dictionaries, containing coordinates,
                              name, use_train, use_test flags.
        output_dir: Directory to save the animation.
        show_plot: Whether to display the plot interactively.
        animation_filename: Filename for the saved animation.
        frame_skip: Number of frames to skip for faster animation.
        rotation_speed_factor: Multiplier for the rotation speed (1.0 is default).
    """
    if not history:
        logger.warning("History is empty. Skipping layout/currents animation.")
        return

    num_points = points_perturbed.shape[0]

    # Determine model type from first coefficient entry
    first_coeffs = history[0]["coefficients"]
    use_vector_model = first_coeffs.shape[0] == 2 * num_points

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Determine plot limits ---
    all_coords = [points_perturbed] + [p["coordinates"] for p in measurement_planes]
    all_points = np.vstack(all_coords)
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    center = (max_coords + min_coords) / 2.0
    max_range = np.max(max_coords - min_coords) * 0.6  # Add some padding

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Currents & Plane Layout History")
    try:
        ax.set_aspect("equal", adjustable="box")
    except NotImplementedError:
        logger.warning("3D aspect ratio 'equal' not fully supported.")

    # --- Static elements (Planes) & Legend Setup ---
    plane_colors = {"train": "green", "test": "blue", "both": "purple"}
    plane_styles = {"train": "-", "test": "--", "both": "-."}
    legend_handles = []

    for plane in measurement_planes:
        coords = plane["coordinates"]
        name = plane.get("name", "Unnamed Plane")
        is_train = plane.get("use_train", False)
        is_test = plane.get("use_test", False)

        role = (
            "both"
            if is_train and is_test
            else ("train" if is_train else ("test" if is_test else "none"))
        )
        if role == "none":
            continue  # Don't draw planes not used for train or test

        color = plane_colors[role]
        linestyle = plane_styles[role]

        # Reshape coords if possible (assuming roughly grid-like)
        res_approx = int(np.sqrt(coords.shape[0]))
        if res_approx * res_approx == coords.shape[0]:
            coords_2d = coords.reshape(res_approx, res_approx, 3)
            x_p = coords_2d[:, :, 0]
            y_p = coords_2d[:, :, 1]
            z_p = coords_2d[:, :, 2]
            # Plot surface - use high stride to make it look solid, low alpha for transparency
            # Set linewidth=0 and antialiased=False for potentially smoother look
            ax.plot_surface(
                x_p,
                y_p,
                z_p,
                color=color,
                alpha=0.2,  # Reduced alpha for better transparency
                rstride=10,
                cstride=10,  # Use strides to control mesh density (optional)
                linewidth=0,
                antialiased=False,
            )
        else:
            # Fallback: plot boundary or just scatter?
            # For now, just skip surface plot if not easily reshapeable
            logger.warning(f"Could not reshape plane '{name}' for wireframe plot.")

        # Add handle for plane legend if this role hasn't been added yet
        role_label_map = {
            "train": "Training Plane",
            "test": "Test Plane",
            "both": "Train & Test Plane",
        }
        role_label = role_label_map.get(role, "Unknown Role")
        if role_label not in [h.get_label() for h in legend_handles]:
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle=linestyle, label=role_label)
            )

    # --- Animated elements (Scatter plot) ---
    # Initial scatter plot (frame 0)
    coeffs_0 = history[0]["coefficients"]
    if use_vector_model:
        mags_0 = np.sqrt(np.abs(coeffs_0[0::2]) ** 2 + np.abs(coeffs_0[1::2]) ** 2)
    else:
        mags_0 = np.abs(coeffs_0)

    # Find max coefficient magnitude across all frames for consistent scaling
    max_mag_overall = 0
    for h_entry in history:
        coeffs_iter = h_entry["coefficients"]
        if use_vector_model:
            mags_iter = np.sqrt(np.abs(coeffs_iter[0::2]) ** 2 + np.abs(coeffs_iter[1::2]) ** 2)
        else:
            mags_iter = np.abs(coeffs_iter)
        max_mag_overall = max(max_mag_overall, np.max(mags_iter) if mags_iter.size > 0 else 0)

    if max_mag_overall < 1e-9:
        max_mag_overall = 1.0  # Avoid division by zero

    sizes_0 = 1 + 100 * (mags_0 / max_mag_overall) ** 2
    scatter = ax.scatter(
        points_perturbed[:, 0],
        points_perturbed[:, 1],
        points_perturbed[:, 2],
        c=mags_0,
        s=sizes_0,
        cmap="jet",
        vmin=0,
        vmax=max_mag_overall,
        depthshade=True,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("Coefficient Magnitude")
    iter_text = ax.text2D(0.05, 0.95, "Iteration: 0", transform=ax.transAxes)

    # --- Animation Setup ---
    total_frames = len(history)
    final_frame_skip = max(1, frame_skip)
    selected_indices = range(0, total_frames, final_frame_skip)
    num_anim_frames = len(selected_indices)

    def update(frame_idx):
        iter_index = selected_indices[frame_idx]  # Get the actual iteration index
        coeffs = history[iter_index]["coefficients"]

        if use_vector_model:
            mags = np.sqrt(np.abs(coeffs[0::2]) ** 2 + np.abs(coeffs[1::2]) ** 2)
        else:
            mags = np.abs(coeffs)

        sizes = 1 + 100 * (mags / max_mag_overall) ** 2

        # Update scatter plot data
        # Note: Updating 3D scatter efficiently is tricky. Re-plotting might be simpler for FuncAnimation.
        # For now, try updating offsets and data directly.
        scatter._offsets3d = (
            points_perturbed[:, 0],
            points_perturbed[:, 1],
            points_perturbed[:, 2],
        )
        scatter.set_sizes(sizes)
        scatter.set_array(mags)

        # Update iteration text
        iter_text.set_text(f"Iteration: {iter_index}")

        # Update view angle for rotation
        azim = (frame_idx * rotation_speed_factor * 360 / num_anim_frames) % 360
        ax.view_init(elev=30, azim=azim)

        # Return list of artists to update
        # Note: Returning scatter might not be enough for full update in 3D
        return [scatter, iter_text]

    # Add legend entry for the scatter points
    # Create a dummy scatter plot point just for the legend handle
    scatter_legend = ax.scatter(
        [], [], [], c="black", s=50, cmap="jet", label="Source Point (Color/Size ~ Coeff. Mag.)"
    )
    legend_handles.append(scatter_legend)

    # Create and save animation
    anim = FuncAnimation(
        fig, update, frames=num_anim_frames, interval=100, blit=False
    )  # blit=False often needed for 3D
    ax.legend(handles=legend_handles)  # Display the combined legend
    output_path = os.path.join(output_dir, animation_filename)
    try:
        adjusted_fps = max(5, 20 // final_frame_skip)
        anim.save(output_path, writer="pillow", fps=adjusted_fps)
        logger.info(
            f"Saved layout/currents animation to {output_path} (Frames: {num_anim_frames}, FPS: {adjusted_fps})"
        )
    except Exception as e:
        logger.error(f"Failed to save layout/currents animation: {e}", exc_info=True)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def animate_plane_comparison(
    history: List[Dict],  # Detailed history from HPR
    planes_to_visualize: List[Dict],  # List of processed plane dicts to animate
    plots_dir: str,
    show_plot: bool = False,
    animation_filename_prefix: str = "comparison_history",
    frame_skip: int = 3,
) -> None:
    """
    Creates comparison animations (True vs Recon vs Error) for specified planes.

    Generates one animation file per plane.

    Args:
        history: List of dictionaries per iteration from HPR, containing 'test_fields'
                 and potentially 'train_field_segments'.
        planes_to_visualize: List of processed plane dictionaries for which to create animations.
                               Must contain 'name', 'coordinates', 'original_data_shape',
                               and either 'measured_magnitude' (if real) or be present
                               in history's 'test_fields'/'train_field_segments'.
        plots_dir: Directory to save the animations.
        show_plot: Whether to display the plots interactively (shows first animation only).
        animation_filename_prefix: Prefix for the saved animation filenames.
        frame_skip: Number of frames to skip for faster animation.
    """
    if not history:
        logger.warning("History is empty. Skipping plane comparison animations.")
        return
    if not planes_to_visualize:
        logger.info("No planes specified for comparison animation.")
        return

    num_iterations = len(history)

    # --- Animation Setup ---
    total_frames = num_iterations
    final_frame_skip = max(1, frame_skip)
    selected_indices = range(0, total_frames, final_frame_skip)
    num_anim_frames = len(selected_indices)

    # --- Loop through planes to generate animations ---
    for plane_idx, plane_data in enumerate(planes_to_visualize):
        plane_name = plane_data.get("name", f"plane_{plane_idx}")
        logger.info(f"Generating comparison animation for plane: {plane_name}")

        coords = plane_data.get("coordinates")
        shape = plane_data.get("original_data_shape")
        # Prioritize calculated GT for simulated, fallback to measured for real
        ground_truth_mag_flat = plane_data.get(
            "ground_truth_magnitude", plane_data.get("measured_magnitude")
        )

        if coords is None or shape is None:
            logger.warning(
                f"Skipping animation for '{plane_name}': Missing coordinates or original_data_shape."
            )
            continue

        # Determine extent and labels based on stored plane axes info
        # Use 'x' and 'y' as defaults if axes info is missing (e.g., older data)
        h_axis_name = plane_data.get("continuous_axis", "x")
        v_axis_name = plane_data.get("discrete_axis", "y")
        axis_map = {"x": 0, "y": 1, "z": 2}
        default_extent = [-0.5, 0.5, -0.5, 0.5]  # Fallback extent

        try:
            h_idx = axis_map.get(h_axis_name)
            v_idx = axis_map.get(v_axis_name)
            if h_idx is None or v_idx is None:
                raise ValueError(f"Invalid axis names stored: {h_axis_name}, {v_axis_name}")

            # Extract min/max directly from the flattened coordinates for robustness
            h_coords_all = coords[:, h_idx]
            v_coords_all = coords[:, v_idx]
            h_min, h_max = np.min(h_coords_all), np.max(h_coords_all)
            v_min, v_max = np.min(v_coords_all), np.max(v_coords_all)

            # Calculate extent [h_min, h_max, v_min, v_max]
            extent = [h_min, h_max, v_min, v_max]
            horizontal_label = f"{h_axis_name.upper()} (m)"
            vertical_label = f"{v_axis_name.upper()} (m)"

            # Check for degenerate extent
            if extent[0] >= extent[1] or extent[2] >= extent[3]:
                logger.warning(
                    f"Degenerate extent calculated for plane '{plane_name}': {extent}. Using default."
                )
                extent = default_extent
                horizontal_label = "Dim 1"  # Reset labels if using default extent
                vertical_label = "Dim 2"

        except Exception as e:
            logger.warning(
                f"Could not determine extent/labels for plane '{plane_name}': {e}. Using default."
            )
            extent = default_extent
            horizontal_label = "Dim 1"
            vertical_label = "Dim 2"
        # Remove duplicated/incorrect extent logic block

        # Get ground truth magnitude (already calculated/loaded and stored in plane_data)
        if ground_truth_mag_flat is None:
            logger.error(
                f"Cannot generate animation for '{plane_name}': Ground truth magnitude ('ground_truth_magnitude' or 'measured_magnitude') not found in plane_data."
            )
            continue

        ground_truth_mag_2d = ground_truth_mag_flat.reshape(shape)

        # Find max value across history for this plane for consistent scaling
        max_val = np.max(ground_truth_mag_2d)
        valid_history_indices = []  # Store indices where field data exists for this plane
        for idx in selected_indices:
            # Check test_fields first, then train_field_segments
            recon_field = history[idx].get("test_fields", {}).get(plane_name)
            if recon_field is None:
                recon_field = history[idx].get("train_field_segments", {}).get(plane_name)
            if recon_field is not None:
                max_val = max(max_val, np.max(np.abs(recon_field)))
                valid_history_indices.append(idx)  # Add index if data is valid

        if not valid_history_indices:
            logger.warning(
                f"No valid field history found for plane '{plane_name}'. Skipping animation."
            )
            continue

        if max_val < 1e-9:
            max_val = 1.0  # Avoid zero vmax

        # Determine Plane Role
        is_train = plane_data.get("use_train", False)
        is_test = plane_data.get("use_test", False)
        role = (
            "Training & Testing"
            if is_train and is_test
            else ("Training" if is_train else ("Testing" if is_test else "Unused"))
        )

        # --- Create Figure for this plane ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))  # Slightly taller for subtitle
        fig.suptitle(f"Plane: {plane_name} ({role}) - Reconstruction History", fontsize=16)

        # Panel 1: Ground Truth (Static)
        im1 = axes[0].imshow(
            ground_truth_mag_2d, cmap="jet", origin="lower", extent=extent, vmin=0, vmax=max_val
        )
        axes[0].set_title("Ground Truth Magnitude")
        axes[0].set_xlabel(horizontal_label)
        axes[0].set_ylabel(vertical_label)
        fig.colorbar(im1, ax=axes[0], label="Magnitude")

        # Panel 2: Reconstructed Magnitude (Animated)
        # Initial frame using the first valid index
        first_valid_idx = valid_history_indices[0]
        # Explicitly check test_fields then train_field_segments for initial frame
        recon_field_0 = history[first_valid_idx].get("test_fields", {}).get(plane_name)
        if recon_field_0 is None:
            recon_field_0 = history[first_valid_idx].get("train_field_segments", {}).get(plane_name)
        recon_mag_0 = np.abs(recon_field_0).reshape(shape)
        im2 = axes[1].imshow(
            recon_mag_0, cmap="jet", origin="lower", extent=extent, vmin=0, vmax=max_val
        )
        axes[1].set_title("Reconstructed Magnitude")
        axes[1].set_xlabel(horizontal_label)
        axes[1].set_ylabel(vertical_label)
        fig.colorbar(im2, ax=axes[1], label="Magnitude")  # Removed unused cbar2 assignment

        # Panel 3: Absolute Error (Animated)
        error_0 = np.abs(recon_mag_0 - ground_truth_mag_2d)
        im3 = axes[2].imshow(error_0, cmap="jet", origin="lower", extent=extent)
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel(horizontal_label)
        axes[2].set_ylabel(vertical_label)
        fig.colorbar(im3, ax=axes[2], label="Error")  # Removed unused cbar3 assignment

        # Add iteration text (similar to grid animation)
        iter_text = fig.text(
            0.5,
            0.96,
            f"Iteration: {first_valid_idx}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        # --- Animation Update Function ---
        # Use only the valid indices for animation frames
        anim_selected_indices = [idx for idx in selected_indices if idx in valid_history_indices]
        num_anim_frames = len(anim_selected_indices)

        def update_plane(
            frame_idx,
            # Pass loop variables as defaults to address B023
            anim_selected_indices=anim_selected_indices,
            plane_name=plane_name,
            history=history,
            im2=im2,
            im3=im3,
            iter_text=iter_text,
            shape=shape,
            ground_truth_mag_2d=ground_truth_mag_2d,
        ):
            # Map animation frame index back to original history iteration index
            iter_index = anim_selected_indices[frame_idx]
            history_entry = history[iter_index]

            # Check test_fields first, then train_field_segments (consistent with above)
            recon_field = history_entry.get("test_fields", {}).get(plane_name)
            if recon_field is None:
                recon_field = history_entry.get("train_field_segments", {}).get(plane_name)

            # This check should ideally not be needed due to pre-filtering, but safety first
            if recon_field is None:
                return im2, im3, iter_text

            recon_mag = np.abs(recon_field).reshape(shape)
            error = np.abs(recon_mag - ground_truth_mag_2d)

            im2.set_array(recon_mag)
            im3.set_array(error)
            # Update error colorbar limits dynamically
            im3.set_clim(vmin=np.min(error), vmax=max(np.max(error), 1e-9))  # Avoid zero range
            iter_text.set_text(f"Iteration: {iter_index}")

            return im2, im3, iter_text  # Return iter_text for blitting

        # --- Create and Save Animation ---
        anim = FuncAnimation(fig, update_plane, frames=num_anim_frames, interval=100, blit=True)
        output_path = os.path.join(plots_dir, f"{animation_filename_prefix}_{plane_name}.gif")
        try:
            adjusted_fps = max(5, 20 // final_frame_skip)
            anim.save(output_path, writer="pillow", fps=adjusted_fps)
            logger.info(
                f"Saved comparison animation for '{plane_name}' to {output_path} (Frames: {num_anim_frames}, FPS: {adjusted_fps})"
            )
        except Exception as e:
            logger.error(
                f"Failed to save comparison animation for '{plane_name}': {e}", exc_info=True
            )

        # Close the figure unless show_plot is True (only show first one if true)
        if show_plot and plane_idx == 0:
            plt.show()
        else:
            plt.close(fig)


# TODO: Refactor common plane coordinate/label logic into a utility function


def animate_multi_plane_comparison_grid(
    history: List[Dict],  # Detailed history from HPR
    planes_to_visualize: List[Dict],  # List of processed plane dicts to animate
    plots_dir: str,
    show_plot: bool = False,
    animation_filename: str = "comparison_grid_history.gif",
    frame_skip: int = 5,  # Increase default skip for potentially slower grid animation
    grid_cols: int = 1,  # Default to stacking planes vertically (1 plane per row)
) -> None:
    """
    Creates a single comparison animation (True vs Recon vs Error)
    showing multiple specified planes arranged in a grid.

    Args:
        history: List of dictionaries per iteration from HPR.
        planes_to_visualize: List of processed plane dictionaries to include.
        plots_dir: Directory to save the animation.
        show_plot: Whether to display the plot interactively.
        animation_filename: Filename for the saved animation.
        frame_skip: Number of frames to skip for faster animation.
        grid_cols: How many planes to show side-by-side horizontally.
                   Each plane still occupies 3 plot columns (True, Recon, Error).
    """
    if not history:
        logger.warning("History is empty. Skipping grid comparison animation.")
        return
    if not planes_to_visualize:
        logger.info("No planes specified for grid comparison animation.")
        return

    num_planes = len(planes_to_visualize)
    # Ensure grid_cols is at least 1
    grid_cols_planes = max(1, grid_cols)
    # Calculate rows needed based on the number of planes and planes per row
    grid_rows = math.ceil(num_planes / grid_cols_planes)
    # Total plot columns = 3 (True, Recon, Error) * number of planes per row
    num_plot_cols_total = 3 * grid_cols_planes

    # Adjust figsize: less wide, potentially taller depending on rows
    # Keep width reasonable even for grid_cols=1, make height proportional to rows
    fig_width = max(18, 6 * num_plot_cols_total)  # Base width for 3 plots, scale if more cols
    fig_height = 6 * grid_rows  # Height scales with rows
    fig, axes = plt.subplots(
        grid_rows, num_plot_cols_total, figsize=(fig_width, fig_height), squeeze=False
    )
    fig.suptitle("Multi-Plane Reconstruction History Grid", fontsize=16)

    num_iterations = len(history)
    total_frames = num_iterations
    final_frame_skip = max(1, frame_skip)
    selected_indices = range(0, total_frames, final_frame_skip)
    num_anim_frames = len(selected_indices)

    plot_elements = []  # Store elements to update: (im_recon, im_error, plane_data, h_label, v_label, extent, gt_mag_2d, valid_indices, plane_name)
    all_recon_max = 0  # Find max recon value across all planes/history for consistent scaling
    valid_history_indices_map = {}

    # --- Initial Setup Loop --- Find max values and valid indices
    logger.info("Setting up grid animation: Pre-calculating ranges...")
    for plane_idx, plane_data in enumerate(planes_to_visualize):
        plane_name = plane_data.get("name", f"plane_{plane_idx}")
        valid_history_indices_map[plane_name] = []
        ground_truth_mag_flat = plane_data.get("measured_magnitude")
        shape = plane_data.get("original_data_shape")

        if ground_truth_mag_flat is None:  # Simulated plane needs placeholder GT
            last_recon_field = history[-1].get("test_fields", {}).get(plane_name)
            if last_recon_field is None:
                last_recon_field = history[-1].get("train_field_segments", {}).get(plane_name)
            if last_recon_field is not None:
                ground_truth_mag_flat = np.abs(last_recon_field)
            else:
                logger.warning(
                    f"Cannot find any field data for plane '{plane_name}' in history. Skipping this plane."
                )
                continue  # Skip this plane if no data found at all

        if shape is None:
            logger.warning(f"Skipping plane '{plane_name}': Missing original_data_shape.")
            continue

        ground_truth_mag_2d = ground_truth_mag_flat.reshape(shape)
        plane_max = np.max(ground_truth_mag_2d)

        for idx in selected_indices:
            recon_field = history[idx].get("test_fields", {}).get(plane_name)
            if recon_field is None:
                recon_field = history[idx].get("train_field_segments", {}).get(plane_name)
            if recon_field is not None:
                plane_max = max(plane_max, np.max(np.abs(recon_field)))
                if idx not in valid_history_indices_map[plane_name]:
                    valid_history_indices_map[plane_name].append(idx)

        all_recon_max = max(all_recon_max, plane_max)

    if all_recon_max < 1e-9:
        all_recon_max = 1.0

    # --- Plotting Loop --- Create initial plots
    logger.info("Setting up grid animation: Creating initial plots...")
    processed_plane_count = 0
    for plane_idx, plane_data in enumerate(planes_to_visualize):
        plane_name = plane_data.get("name", f"plane_{plane_idx}")
        valid_indices = valid_history_indices_map.get(plane_name)

        if not valid_indices:  # Skip if no valid history found or skipped earlier
            logger.warning(
                f"Skipping plot setup for '{plane_name}' due to no valid history indices."
            )
            continue

        coords = plane_data.get("coordinates")
        shape = plane_data.get("original_data_shape")
        # Prioritize calculated GT for simulated, fallback to measured for real
        ground_truth_mag_flat = plane_data.get(
            "ground_truth_magnitude", plane_data.get("measured_magnitude")
        )

        if coords is None or shape is None:  # Should have been caught, but double check
            continue

        # Get ground truth magnitude (already calculated/loaded and stored in plane_data)
        if ground_truth_mag_flat is None:
            logger.error(
                f"Cannot generate grid animation plot for '{plane_name}': Ground truth magnitude ('ground_truth_magnitude' or 'measured_magnitude') not found in plane_data."
            )
            continue  # Skip setting up plots for this plane
        # Removed placeholder logic and is_placeholder_gt flag

        ground_truth_mag_2d = ground_truth_mag_flat.reshape(shape)

        # --- Calculate Extent and Labels (using corrected logic) ---
        h_axis_name = plane_data.get("continuous_axis", "x")
        v_axis_name = plane_data.get("discrete_axis", "y")
        axis_map = {"x": 0, "y": 1, "z": 2}
        default_extent = [-0.5, 0.5, -0.5, 0.5]
        try:
            h_idx = axis_map.get(h_axis_name)
            v_idx = axis_map.get(v_axis_name)
            if h_idx is None or v_idx is None:
                raise ValueError("Invalid axis names")
            h_coords_all = coords[:, h_idx]
            v_coords_all = coords[:, v_idx]
            h_min, h_max = np.min(h_coords_all), np.max(h_coords_all)
            v_min, v_max = np.min(v_coords_all), np.max(v_coords_all)
            extent = [h_min, h_max, v_min, v_max]
            horizontal_label = f"{h_axis_name.upper()} (m)"
            vertical_label = f"{v_axis_name.upper()} (m)"
            if extent[0] >= extent[1] or extent[2] >= extent[3]:
                logger.warning(f"Degenerate extent for '{plane_name}': {extent}. Using default.")
                extent = default_extent
                horizontal_label = "Dim 1"
                vertical_label = "Dim 2"
        except Exception as e:
            logger.warning(f"Extent/label error for '{plane_name}': {e}. Using default.")
            extent = default_extent
            horizontal_label = "Dim 1"
            vertical_label = "Dim 2"
        # --- End Extent Logic ---
        # Determine subplot row and starting column for this plane
        # Determine subplot row and starting column index for this plane's plots
        current_row = processed_plane_count // grid_cols_planes  # Row index
        current_col_start = (
            processed_plane_count % grid_cols_planes
        ) * 3  # Start column index (0, 3, 6...)

        # Get initial frame data
        first_valid_idx = valid_indices[0]
        recon_field_0 = history[first_valid_idx].get("test_fields", {}).get(plane_name)
        if recon_field_0 is None:
            recon_field_0 = history[first_valid_idx].get("train_field_segments", {}).get(plane_name)
        recon_mag_0 = np.abs(recon_field_0).reshape(shape)
        error_0 = np.abs(recon_mag_0 - ground_truth_mag_2d)

        # Plot Ground Truth (col 0)
        ax_gt = axes[current_row, current_col_start]
        im_gt = ax_gt.imshow(
            ground_truth_mag_2d,
            cmap="jet",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=all_recon_max,
        )
        gt_title = f"{plane_name}\nGround Truth Mag."
        # Removed placeholder title addition
        ax_gt.set_title(gt_title, fontsize=10)
        ax_gt.set_xlabel(horizontal_label)
        ax_gt.set_ylabel(vertical_label)
        fig.colorbar(im_gt, ax=ax_gt, label="Magnitude", shrink=0.8)

        # Plot Reconstructed Mag (col 1)
        ax_recon = axes[current_row, current_col_start + 1]
        im_recon = ax_recon.imshow(
            recon_mag_0, cmap="jet", origin="lower", extent=extent, vmin=0, vmax=all_recon_max
        )
        ax_recon.set_title(f"{plane_name}\nRecon Mag.", fontsize=10)
        ax_recon.set_xlabel(horizontal_label)
        ax_recon.set_ylabel(vertical_label)
        fig.colorbar(im_recon, ax=ax_recon, label="Magnitude", shrink=0.8)

        # Plot Error Map (col 2)
        ax_error = axes[current_row, current_col_start + 2]
        im_error = ax_error.imshow(error_0, cmap="jet", origin="lower", extent=extent)
        ax_error.set_title(f"{plane_name}\nAbs. Error", fontsize=10)
        ax_error.set_xlabel(horizontal_label)
        ax_error.set_ylabel(vertical_label)
        cbar_error = fig.colorbar(im_error, ax=ax_error, label="Error", shrink=0.8)

        plot_elements.append(
            {
                "im_recon": im_recon,
                "im_error": im_error,
                "cbar_error": cbar_error,
                "plane_data": plane_data,
                "gt_mag_2d": ground_truth_mag_2d,
                "valid_indices": valid_indices,
                "plane_name": plane_name,
                "shape": shape,
            }
        )
        processed_plane_count += 1

    # Hide unused axes
    # Hide unused axes at the end of the grid
    for r in range(grid_rows):
        for c in range(num_plot_cols_total):
            # Check if this subplot corresponds to a processed plane
            plane_index_for_col = (
                c // 3
            )  # Which plane this column belongs to (0, 0, 0, 1, 1, 1, ...)
            plane_overall_index = r * grid_cols_planes + plane_index_for_col
            if plane_overall_index >= processed_plane_count:
                axes[r, c].axis("off")

    iter_text = fig.text(
        0.5, 0.97, "Iteration: 0", ha="center", va="top", fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout

    # --- Animation Update Function ---
    def update(frame_idx):  # Correct function name for this scope
        iter_index = selected_indices[frame_idx]
        artists = [iter_text]
        iter_text.set_text(f"Iteration: {iter_index}")

        for elements in plot_elements:
            plane_name = elements["plane_name"]
            shape = elements["shape"]
            gt_mag_2d = elements["gt_mag_2d"]
            im_recon = elements["im_recon"]
            im_error = elements["im_error"]
            # cbar_error = elements["cbar_error"] # Removed unused variable
            valid_indices = elements["valid_indices"]

            if iter_index not in valid_indices:
                # If no data for this plane at this iter, skip update (keep previous frame)
                artists.extend([im_recon, im_error])  # Still need to return them for blitting
                continue

            history_entry = history[iter_index]
            recon_field = history_entry.get("test_fields", {}).get(plane_name)
            if recon_field is None:
                recon_field = history_entry.get("train_field_segments", {}).get(plane_name)

            if recon_field is not None:
                recon_mag = np.abs(recon_field).reshape(shape)
                error = np.abs(recon_mag - gt_mag_2d)

                im_recon.set_array(recon_mag)
                im_error.set_array(error)
                # Update error colorbar limits dynamically
                error_min, error_max = np.min(error), np.max(error)
                im_error.set_clim(vmin=error_min, vmax=max(error_max, 1e-9))  # Avoid zero range
                # cbar_error.update_normal(im_error) # Might be needed depending on matplotlib version

                artists.extend([im_recon, im_error])
            else:
                # Should not happen if valid_indices logic is correct, but handle defensively
                artists.extend([im_recon, im_error])

        return artists

    # --- Create and Save Animation ---
    logger.info("Creating grid animation...")
    # Blit=False might be more reliable for complex multi-subplot figures
    anim = FuncAnimation(fig, update, frames=num_anim_frames, interval=150, blit=False)
    output_path = os.path.join(plots_dir, animation_filename)
    try:
        adjusted_fps = max(5, 20 // final_frame_skip)
        anim.save(
            output_path, writer="pillow", fps=adjusted_fps, dpi=100
        )  # Lower dpi for faster save
        logger.info(
            f"Saved grid comparison animation to {output_path} (Frames: {num_anim_frames}, FPS: {adjusted_fps})"
        )
    except Exception as e:
        logger.error(f"Failed to save grid comparison animation: {e}", exc_info=True)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

        # Remove misplaced lines from previous function's update logic


# TODO: Refactor common plane coordinate/label logic into a utility function
