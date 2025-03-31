import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Dict, Any

def visualize_field(
    field: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    filename: Optional[str] = None, 
    show: bool = True
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
    im = ax.imshow(field_2d, cmap='viridis', origin='lower',
                 extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(im, ax=ax, label='Field Magnitude')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
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
    measurement_plane: Optional[np.ndarray] = None
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
    ax = fig.add_subplot(111, projection='3d')
    
    # Default color and size
    colors = 'blue'
    sizes = 20
    
    # If currents are provided, use them for coloring
    if currents is not None:
        colors = np.abs(currents)
        sizes = 20 + 100 * (colors / np.max(colors) if np.max(colors) > 0 else 0)
    
    # Plot all points
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=sizes, cmap='viridis', alpha=0.7
    )
    
    # Highlight specific points if requested
    if highlight_indices is not None and len(highlight_indices) > 0:
        ax.scatter(
            points[highlight_indices, 0], 
            points[highlight_indices, 1], 
            points[highlight_indices, 2],
            color='red', s=50, marker='*'
        )
    
    # Add colorbar if currents are provided
    if currents is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Current Magnitude')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(0, room_size)
    ax.set_ylim(0, room_size)
    ax.set_zlim(0, room_size)
    
    # Add measurement plane if provided
    if measurement_plane is not None:
        # Extract the extent of the measurement plane
        x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
        y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
        z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
        
        # Create a rectangular face based on the extent
        if np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
            xx, yy = np.meshgrid([x_min, x_min], [y_min, y_max])
            zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
        elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_min])
            zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
        else:  # XY plane
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
            zz = np.ones_like(xx) * z_min
        
        # Plot the face with semi-transparency
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan', edgecolor='blue')
    
    # Set equal aspect ratio to avoid distortion
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    if not show:
        plt.close(fig)

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
    output_file: Optional[str] = None
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
    resolution = true_field_2d.shape[0]
    
    # Determine plane type and set appropriate axis labels
    x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
    y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
    z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
    
    # Calculate spread in each dimension to identify constant dimensions
    x_spread = x_max - x_min
    y_spread = y_max - y_min
    z_spread = z_max - z_min
    
    # Determine plane type and set coordinates and labels
    if np.isclose(x_spread, 0) or np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
        # For YZ plane, Y is horizontal (dim 1) and Z is vertical (dim 2)
        # Get unique values to ensure we have the full range
        horizontal_coords = np.unique(measurement_plane[:, :, 1])  # Unique Y values
        vertical_coords = np.unique(measurement_plane[:, :, 2])    # Unique Z values
        horizontal_label = 'Y (m)'
        vertical_label = 'Z (m)'
        plane_type = 'YZ'
    elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
        # For XZ plane, X is horizontal (dim 0) and Z is vertical (dim 2)
        horizontal_coords = np.unique(measurement_plane[:, :, 0])  # Unique X values
        vertical_coords = np.unique(measurement_plane[:, :, 2])    # Unique Z values
        horizontal_label = 'X (m)'
        vertical_label = 'Z (m)'
        plane_type = 'XZ'
    else:  # XY plane (default)
        # For XY plane, X is horizontal (dim 0) and Y is vertical (dim 1)
        horizontal_coords = np.unique(measurement_plane[:, :, 0])  # Unique X values
        vertical_coords = np.unique(measurement_plane[:, :, 1])    # Unique Y values
        horizontal_label = 'X (m)'
        vertical_label = 'Y (m)'
        plane_type = 'XY'
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Point cloud with currents - directly create as 3D
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Create the other 2D subplots manually
    ax1 = fig.add_subplot(2, 2, 2)  # Top right
    ax2 = fig.add_subplot(2, 2, 3)  # Bottom left
    ax3 = fig.add_subplot(2, 2, 4)  # Bottom right
    # Color points by current magnitude with improved visualization
    current_mags = np.abs(currents)
    if np.max(current_mags) > 0:
        # More extreme scaling - almost invisible for zero values, larger for high values
        normalized_mags = current_mags / np.max(current_mags)
        
        # Make sizes very small (near zero) for zero current, larger for higher currents
        sizes = 0.5 + 150 * normalized_mags**2  # Square for more dramatic effect
        
        # Make colors more transparent for low values
        alphas = 0.2 + 0.8 * normalized_mags  # Range from 0.2 to 1.0 transparency
    else:
        sizes = 0.5  # Near-zero size for all points if all currents are zero
        alphas = 0.2  # Low alpha for all points if all currents are zero
    
    scatter = ax3d.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=current_mags, s=sizes, cmap='plasma', alpha=alphas
    )
    
    # Add small black dots for all points to show the environment
    ax3d.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        color='black', s=1, alpha=0.3
    )
    
    ax3d.set_title("Point Cloud with Source Currents")
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    # Add measurement plane visualization
    # Extract the extent of the measurement plane
    x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
    y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
    z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
    
    # Create a rectangular face based on the extent
    if np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
        xx, yy = np.meshgrid([x_min, x_min], [y_min, y_max])
        zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
    elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_min])
        zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
    else:  # XY plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
        zz = np.ones_like(xx) * z_min
    
    # Plot the face with semi-transparency
    ax3d.plot_surface(xx, yy, zz, alpha=0.3, color='cyan', edgecolor='blue')
    
    # Set equal aspect ratio for 3D plot
    ax3d.set_box_aspect([1, 1, 1])
    
    plt.colorbar(scatter, ax=ax3d, label='Current Magnitude')
    
    # Find global min and max for consistent colorbar scaling across field plots
    true_field_abs = np.abs(true_field_2d)
    reconstructed_field_abs = np.abs(reconstructed_field_2d)
    
    # Calculate global min and max across all field arrays
    global_min = min(np.min(true_field_abs), np.min(measured_magnitude_2d), np.min(reconstructed_field_abs))
    global_max = max(np.max(true_field_abs), np.max(measured_magnitude_2d), np.max(reconstructed_field_abs))
    
    # Set extent for 2D plots using the detected coordinates
    extent = [horizontal_coords.min(), horizontal_coords.max(), 
              vertical_coords.min(), vertical_coords.max()]
    
    # Plot 2: True field magnitude with normalized colorbar (top right)
    im1 = ax1.imshow(true_field_abs, cmap='viridis', origin='lower',
                   extent=extent,
                   vmin=global_min, vmax=global_max)
    ax1.set_title(f"True Field Magnitude ({plane_type} Plane)")
    ax1.set_xlabel(horizontal_label)
    ax1.set_ylabel(vertical_label)
    plt.colorbar(im1, ax=ax1, label='Field Magnitude')
    
    # Plot 3: Measured field magnitude with normalized colorbar (bottom left)
    im2 = ax2.imshow(measured_magnitude_2d, cmap='viridis', origin='lower',
                   extent=extent,
                   vmin=global_min, vmax=global_max)
    ax2.set_title(f"Measured Field Magnitude ({plane_type} Plane)")
    ax2.set_xlabel(horizontal_label)
    ax2.set_ylabel(vertical_label)
    plt.colorbar(im2, ax=ax2, label='Field Magnitude')
    
    # Plot 4: Reconstructed field magnitude with normalized colorbar (bottom right)
    im3 = ax3.imshow(reconstructed_field_abs, cmap='viridis', origin='lower',
                   extent=extent,
                   vmin=global_min, vmax=global_max)
    ax3.set_title(f"Reconstructed Field Magnitude ({plane_type} Plane)")
    ax3.set_xlabel(horizontal_label)
    ax3.set_ylabel(vertical_label)
    plt.colorbar(im3, ax=ax3, label='Field Magnitude')
    
    # Add metrics as text
    fig.suptitle(f"Field Reconstruction Results\nRMSE: {rmse:.4f}, Correlation: {correlation:.4f}", 
                fontsize=16)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved results to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

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
    measured_magnitude: Optional[np.ndarray] = None
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
    x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
    y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
    z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
    
    # Calculate spread in each dimension to identify constant dimensions
    x_spread = x_max - x_min
    y_spread = y_max - y_min
    z_spread = z_max - z_min
    
    # Determine plane type and set coordinates and labels
    if np.isclose(x_spread, 0) or np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
        # For YZ plane, Y is horizontal (dim 1) and Z is vertical (dim 2)
        # Get unique values to ensure we have the full range
        horizontal_coords = np.unique(measurement_plane[:, :, 1])  # Unique Y values
        vertical_coords = np.unique(measurement_plane[:, :, 2])    # Unique Z values
        horizontal_label = 'Y (m)'
        vertical_label = 'Z (m)'
        plane_type = 'YZ'
    elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
        horizontal_coords = measurement_plane[0, :, 0]  # X values
        vertical_coords = measurement_plane[:, 0, 2]    # Z values
        horizontal_label = 'X (m)'
        vertical_label = 'Z (m)'
        plane_type = 'XZ'
    else:  # XY plane (default)
        horizontal_coords = measurement_plane[0, :, 0]  # X values
        vertical_coords = measurement_plane[:, 0, 1]    # Y values
        horizontal_label = 'X (m)'
        vertical_label = 'Y (m)'
        plane_type = 'XY'
    
    # Create figure with a wider size for better visibility
    fig = plt.figure(figsize=(14, 8))
    
    # Three panels in a 1x3 layout: True Field, Reconstructed Field, Combined Errors
    ax3 = plt.subplot(1, 3, 1)  # True field (if available)
    ax1 = plt.subplot(1, 3, 2)  # Reconstructed field magnitude 
    ax2 = plt.subplot(1, 3, 3)  # Combined error metrics
    
    # Calculate global max for consistent colormap scaling
    field_mag = np.abs(field_history[0]).reshape(resolution, resolution)
    global_max = np.max([np.max(np.abs(field_history[i]).reshape(resolution, resolution)) 
                       for i in range(len(field_history))])
    
    # Set extent for 2D plots using the detected coordinates
    extent = [horizontal_coords.min(), horizontal_coords.max(), 
              vertical_coords.min(), vertical_coords.max()]
              
    # Setup for true field visualization (first panel)
    if measured_magnitude is not None:
        # If we have measured magnitude data, reshape it for display
        measured_field_2d = measured_magnitude.reshape(resolution, resolution)
        global_max = max(global_max, np.max(measured_field_2d))  # Update global max
        im3 = ax3.imshow(measured_field_2d, cmap='viridis', origin='lower',
                      extent=extent,
                      vmin=0, vmax=global_max)  # Use consistent color scale
        ax3.set_title(f"True Field (Measured) ({plane_type} Plane)")
        ax3.set_xlabel(horizontal_label)
        ax3.set_ylabel(vertical_label)
        plt.colorbar(im3, ax=ax3, label='Field Magnitude')
    else:
        # If no measured data, show a placeholder
        im3 = ax3.imshow(np.zeros((resolution, resolution)), cmap='viridis', origin='lower',
                      extent=extent,
                      vmin=0, vmax=global_max)
        ax3.set_title(f"No Reference Data Available ({plane_type} Plane)")
        ax3.set_xlabel(horizontal_label)
        ax3.set_ylabel(vertical_label)
        plt.colorbar(im3, ax=ax3, label='Field Magnitude')
    
    # Setup for reconstructed field magnitude (second panel)
    im1 = ax1.imshow(field_mag, cmap='viridis', origin='lower',
                   extent=extent,
                   vmin=0, vmax=global_max)  # Use consistent color scale
    ax1.set_title(f"Reconstructed Field ({plane_type} Plane)")
    ax1.set_xlabel(horizontal_label)
    ax1.set_ylabel(vertical_label)
    plt.colorbar(im1, ax=ax1, label='Field Magnitude')
    
    # Setup for the error metrics panel (second panel)
    ax2.grid(True)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error (log scale)')
    ax2.set_title('Error Metrics')
    
    # Compute two error metrics:
    # 1. Iteration-to-iteration change (convergence rate)
    iteration_changes = np.zeros(len(field_history))
    for i in range(len(field_history)):
        field_i = field_history[i]  # Current complex field (includes phase)
        field_prev = field_history[max(0, i-1)]  # Previous complex field (includes phase)
        # RMSE between consecutive complex fields (captures both magnitude and phase changes)
        iteration_changes[i] = np.sqrt(np.mean(np.abs(field_i - field_prev)**2))
    
    # 2. Actual error relative to the measured field (if provided)
    if measured_magnitude is not None:
        actual_errors = np.zeros(len(field_history))
        for i in range(len(field_history)):
            # Calculate error between simulated magnitude and measured magnitude
            simulated_magnitude = np.abs(field_history[i])
            measured_magnitude_norm = np.linalg.norm(measured_magnitude)
            mag_diff = simulated_magnitude - measured_magnitude
            actual_errors[i] = np.linalg.norm(mag_diff) / measured_magnitude_norm
    else:
        # Use iteration changes as a fallback
        actual_errors = iteration_changes.copy()
    
    # Plot both error metrics on the same axis with different colors/styles
    line_changes, = ax2.semilogy(range(1), [iteration_changes[0]], 'ro-', linewidth=2, markersize=4, 
                               label='Frame-to-Frame Change')
    line_errors, = ax2.semilogy(range(1), [actual_errors[0]], 'bo-', linewidth=2, markersize=4,
                              label='Actual Error')
    
    # Setup y-axis limits for the error plot
    min_error = max(1e-6, min(np.min(iteration_changes), np.min(actual_errors)))
    max_error = max(np.max(iteration_changes), np.max(actual_errors))
    ax2.set_ylim(min_error, max_error*1.1)
    ax2.set_xlim(0, len(field_history))
    
    # Add convergence threshold line
    threshold_line = ax2.axhline(y=convergence_threshold, color='r', linestyle='--')
    
    # Mark perturbations and restarts on the plot
    perturbation_lines = []
    restart_lines = []
    
    if perturbation_iterations:
        for iter_idx in perturbation_iterations:
            if iter_idx < len(iteration_changes):
                line = ax2.axvline(x=iter_idx, color='g', linestyle=':', alpha=0.7)
                perturbation_lines.append(line)
    
    if restart_iterations:
        for iter_idx in restart_iterations:
            if iter_idx < len(iteration_changes):
                line = ax2.axvline(x=iter_idx, color='m', linestyle='--', alpha=0.7)
                restart_lines.append(line)
    
    # Combine all legend elements
    legend_elements = [
        line_changes,
        line_errors,
        threshold_line,
    ]
    
    legend_labels = [
        'Frame-to-Frame Change',
        'Actual Error',
        f'Threshold ({convergence_threshold:.1e})',
    ]
    
    if perturbation_iterations and perturbation_lines:
        legend_elements.append(perturbation_lines[0])
        legend_labels.append('Perturbation')
    
    if restart_iterations and restart_lines:
        legend_elements.append(restart_lines[0])
        legend_labels.append('Restart')
    
    # Add combined legend
    ax2.legend(legend_elements, legend_labels, loc='upper right')
    
    # Text for iteration number with more visible styling
    iter_text = ax1.text(0.02, 0.98, f"Iteration: 0", transform=ax1.transAxes,
                      verticalalignment='top', color='white', fontweight='bold',
                      bbox=dict(facecolor='black', alpha=0.5))
    
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
        interval = 200  # Slower for fewer frames
    elif num_selected_frames < 60:
        interval = 150  # Medium speed
    else:
        interval = 100  # Faster for many frames
    
    # Create animation function
    def update(i):
        # Get actual frame index
        frame = selected_frames[i]
        
        # Update field magnitude
        field_mag = np.abs(selected_field_history[i]).reshape(resolution, resolution)
        im1.set_array(field_mag)
        
        # Update iteration-to-iteration change plot
        line_changes.set_data(range(frame+1), iteration_changes[:frame+1])
        
        # Update actual error plot
        line_errors.set_data(range(frame+1), actual_errors[:frame+1])
        
        # Update text
        iter_text.set_text(f"Iteration: {frame}")
        
        return im1, line_changes, line_errors, iter_text
    
    # Create animation with fewer frames and faster display interval
    anim = FuncAnimation(fig, update, frames=len(selected_frames), 
                         interval=100, blit=True)  # Faster frame rate (100ms vs 200ms)
    
    # Save animation if output file specified
    if output_file:
        # Use higher FPS for smoother, faster animation
        adjusted_fps = max(10, 30 // frame_skip)  # Higher base FPS (10-30 vs 5-15)
        anim.save(output_file, writer='pillow', fps=adjusted_fps)
        print(f"Saved animation to {output_file} (skipping {frame_skip-1} frames, fps={adjusted_fps})")
    
    # Show plot if requested
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

def visualize_current_and_field_history(
    points: np.ndarray,
    coefficient_history: np.ndarray,
    field_history: np.ndarray,
    true_field: np.ndarray,
    resolution: int,
    measurement_plane: np.ndarray,
    show_plot: bool = True,
    output_file: Optional[str] = "current_field_animation.gif",
    frame_skip: int = 3
) -> None:
    """Create animation showing current density history in 3D and field reconstruction history in 2D.
    
    Args:
        points: Point coordinates with shape (num_points, 3)
        coefficient_history: History of coefficients over iterations with shape (iterations, num_points)
        field_history: History of reconstructed fields over iterations with shape (iterations, resolution*resolution)
        true_field: True complex field to compare against, with shape (resolution*resolution)
        resolution: Resolution of measurement grid
        measurement_plane: Measurement plane coordinates
        show_plot: Whether to display the plot
        output_file: Path to save the animation
        frame_skip: Number of frames to skip for faster animation (higher values = faster animation with fewer frames)
    """
    # Determine plane type and set appropriate axis labels
    x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
    y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
    z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
    
    # Determine plane type and set coordinates and labels
    if np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
        # For YZ plane, Y is horizontal (dim 1) and Z is vertical (dim 2)
        # Get unique values to ensure we have the full range
        horizontal_coords = np.unique(measurement_plane[:, :, 1])  # Unique Y values
        vertical_coords = np.unique(measurement_plane[:, :, 2])    # Unique Z values
        horizontal_label = 'Y (m)'
        vertical_label = 'Z (m)'
        plane_type = 'YZ'
    elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
        horizontal_coords = measurement_plane[0, :, 0]  # X values
        vertical_coords = measurement_plane[:, 0, 2]    # Z values
        horizontal_label = 'X (m)'
        vertical_label = 'Z (m)'
        plane_type = 'XZ'
    else:  # XY plane (default)
        horizontal_coords = measurement_plane[0, :, 0]  # X values
        vertical_coords = measurement_plane[:, 0, 1]    # Y values
        horizontal_label = 'X (m)'
        vertical_label = 'Y (m)'
        plane_type = 'XY'
    
    # Reshape the true field for visualization
    true_field_2d = np.abs(true_field).reshape(resolution, resolution)
    
    # Calculate initial field magnitude and find global max for consistent color scaling
    field_mag = np.abs(field_history[0]).reshape(resolution, resolution)
    
    # Find global maximum value for consistent colormap scaling between plots
    global_max = max(np.max(true_field_2d), np.max([np.max(np.abs(field_history[i]).reshape(resolution, resolution)) 
                                                 for i in range(len(field_history))]))
    
    # Create figure with a 2x2 grid of subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Top left: 3D current density
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Top right: Field magnitude history
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Bottom left: True field
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Bottom right: Error/difference
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Initial 3D scatter plot of points with current densities (top left)
    current_mags = np.abs(coefficient_history[0])
    if np.max(current_mags) > 0:
        # More extreme scaling - almost invisible for zero values, larger for high values
        normalized_mags = current_mags / np.max(current_mags)
        
        # Make sizes very small (near zero) for zero current, larger for higher currents
        sizes = 0.5 + 150 * normalized_mags**2  # Square for more dramatic effect
        
        # Make colors more transparent for low values
        alphas = 0.2 + 0.8 * normalized_mags  # Range from 0.2 to 1.0 transparency
    else:
        sizes = 0.5  # Near-zero size for all points if all currents are zero
        alphas = 0.2  # Low alpha for all points if all currents are zero
    
    scatter = ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=current_mags, s=sizes, cmap='plasma', alpha=alphas
    )
    
    # Set 3D plot labels and title
    ax1.set_title("Current Density Distribution")
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # Add measurement plane visualization
    # Extract the extent of the measurement plane
    x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
    y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
    z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
    
    # Create a rectangular face based on the extent
    if np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
        xx, yy = np.meshgrid([x_min, x_min], [y_min, y_max])
        zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
    elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_min])
        zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
    else:  # XY plane
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
        zz = np.ones_like(xx) * z_min
    
    # Plot the face with semi-transparency
    ax1.plot_surface(xx, yy, zz, alpha=0.3, color='cyan', edgecolor='blue')
    
    # Add colorbar for current magnitude
    cbar1 = plt.colorbar(scatter, ax=ax1, label='Current Magnitude')
    
    # Set extent for 2D plots using the detected coordinates
    extent = [horizontal_coords.min(), horizontal_coords.max(), 
              vertical_coords.min(), vertical_coords.max()]
    
    # Initial 2D field magnitude plot (top right)
    im1 = ax2.imshow(field_mag, cmap='viridis', origin='lower',
                    extent=extent,
                    vmin=0, vmax=global_max)  # Use consistent color scale
    ax2.set_title(f"Reconstructed Field Magnitude ({plane_type} Plane)")
    ax2.set_xlabel(horizontal_label)
    ax2.set_ylabel(vertical_label)
    cbar2 = plt.colorbar(im1, ax=ax2, label='Field Magnitude')
    
    # True field magnitude plot (bottom left)
    im2 = ax3.imshow(true_field_2d, cmap='viridis', origin='lower',
                    extent=extent,
                    vmin=0, vmax=global_max)  # Use consistent color scale
    ax3.set_title(f"True Field Magnitude ({plane_type} Plane)")
    ax3.set_xlabel(horizontal_label)
    ax3.set_ylabel(vertical_label)
    cbar3 = plt.colorbar(im2, ax=ax3, label='Field Magnitude')
    
    # Initial error/difference plot (bottom right)
    error = np.abs(field_mag - true_field_2d)
    im3 = ax4.imshow(error, cmap='hot', origin='lower',
                    extent=extent)
    ax4.set_title(f"Error (Absolute Difference) ({plane_type} Plane)")
    ax4.set_xlabel(horizontal_label)
    ax4.set_ylabel(vertical_label)
    cbar4 = plt.colorbar(im3, ax=ax4, label='Error Magnitude')
    
    # Text for iteration number
    iter_text = fig.text(0.5, 0.98, f"Iteration: 0", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
    
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
    selected_coefficient_history = coefficient_history[selected_frames]
    selected_field_history = field_history[selected_frames]
    num_selected_frames = len(selected_frames)
    
    # Adapt animation interval based on frame count:
    # - Fewer frames -> slower animation (longer interval)
    # - More frames -> faster animation (shorter interval)
    if num_selected_frames < 30:
        interval = 250  # Slower for fewer frames - 3D needs more time
    elif num_selected_frames < 60:
        interval = 200  # Medium speed
    else:
        interval = 150  # Faster for many frames
    
    # Create animation function
    def update(i):
        # Get actual frame index
        frame = selected_frames[i]
        
        # Clear previous 3D scatter plot to prevent overcrowding
        ax1.clear()
        
        # Update current density scatter plot
        current_mags = np.abs(selected_coefficient_history[i])
        if np.max(current_mags) > 0:
            # More extreme scaling - almost invisible for zero values, larger for high values
            normalized_mags = current_mags / np.max(current_mags)
            
            # Make sizes very small (near zero) for zero current, larger for higher currents
            sizes = 0.5 + 150 * normalized_mags**2  # Square for more dramatic effect
            
            # Make colors more transparent for low values
            alphas = 0.2 + 0.8 * normalized_mags  # Range from 0.2 to 1.0 transparency
        else:
            sizes = 0.5  # Near-zero size for all points if all currents are zero
            alphas = 0.2  # Low alpha for all points if all currents are zero
        
        scatter = ax1.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=current_mags, s=sizes, cmap='plasma', alpha=alphas
        )
        
        # Restore 3D plot settings
        ax1.set_title("Current Density Distribution")
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        # Set equal aspect ratio for 3D plot
        ax1.set_box_aspect([1, 1, 1])
        
        # Add measurement plane visualization for each frame
        # Extract the extent of the measurement plane
        x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
        y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
        z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
        
        # Create a rectangular face based on the extent
        if np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
            xx, yy = np.meshgrid([x_min, x_min], [y_min, y_max])
            zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
        elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_min])
            zz = np.meshgrid([z_min, z_max], [z_min, z_max])[0]
        else:  # XY plane
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
            zz = np.ones_like(xx) * z_min
        
        # Plot the face with semi-transparency
        ax1.plot_surface(xx, yy, zz, alpha=0.3, color='cyan', edgecolor='blue')
        
        # Make the 3D view rotate by changing the azimuth angle in each frame
        rotation_speed = 360 / len(selected_frames)
        current_azim = (i * rotation_speed) % 360
        ax1.view_init(elev=30, azim=current_azim)
        
        # Update reconstructed field magnitude
        field_mag = np.abs(selected_field_history[i]).reshape(resolution, resolution)
        im1.set_array(field_mag)
        
        # Update error/difference plot
        error = np.abs(field_mag - true_field_2d)
        im3.set_array(error)
        
        # Update iteration text
        iter_text.set_text(f"Iteration: {frame}")
        
        return [scatter, im1, im3, iter_text]
    
    # Create animation with fewer frames
    # Disable blitting to ensure 3D rotation works properly
    anim = FuncAnimation(fig, update, frames=len(selected_frames), 
                        interval=200, blit=False)
    
    plt.tight_layout()
    
    # Save animation if output file specified
    if output_file:
        # Adjust FPS proportionally to maintain similar animation duration
        adjusted_fps = max(5, 15 // frame_skip)
        anim.save(output_file, writer='pillow', fps=adjusted_fps)
        print(f"Saved animation to {output_file} (skipping {frame_skip-1} frames, fps={adjusted_fps})")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_comparison(
    true_field: np.ndarray, 
    reconstructed_field: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray, 
    title: str = "Field Comparison",
    metrics: Optional[Dict[str, float]] = None,
    filename: Optional[str] = None,
    show: bool = True,
    measurement_plane: Optional[np.ndarray] = None
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
        x_min, x_max = np.min(measurement_plane[:,:,0]), np.max(measurement_plane[:,:,0])
        y_min, y_max = np.min(measurement_plane[:,:,1]), np.max(measurement_plane[:,:,1])
        z_min, z_max = np.min(measurement_plane[:,:,2]), np.max(measurement_plane[:,:,2])
        
        # Determine plane type and set coordinates and labels
        if np.allclose(measurement_plane[:,:,0], x_min):  # YZ plane
            horizontal_coords = measurement_plane[0, :, 1]  # Y values
            vertical_coords = measurement_plane[:, 0, 2]    # Z values
            horizontal_label = 'Y (m)'
            vertical_label = 'Z (m)'
            plane_type = 'YZ'
            extent = [horizontal_coords.min(), horizontal_coords.max(), 
                      vertical_coords.min(), vertical_coords.max()]
        elif np.allclose(measurement_plane[:,:,1], y_min):  # XZ plane
            horizontal_coords = measurement_plane[0, :, 0]  # X values
            vertical_coords = measurement_plane[:, 0, 2]    # Z values
            horizontal_label = 'X (m)'
            vertical_label = 'Z (m)'
            plane_type = 'XZ'
            extent = [horizontal_coords.min(), horizontal_coords.max(), 
                      vertical_coords.min(), vertical_coords.max()]
        else:  # XY plane (default)
            horizontal_coords = measurement_plane[0, :, 0]  # X values
            vertical_coords = measurement_plane[:, 0, 1]    # Y values
            horizontal_label = 'X (m)'
            vertical_label = 'Y (m)'
            plane_type = 'XY'
            extent = [horizontal_coords.min(), horizontal_coords.max(), 
                      vertical_coords.min(), vertical_coords.max()]
    
    # Plot true field with normalized colorbar
    im1 = axes[0].imshow(true_mag, cmap='viridis', origin='lower',
                       extent=extent,
                       vmin=global_min, vmax=global_max)
    axes[0].set_title(f"True Field ({plane_type} Plane)")
    axes[0].set_xlabel(horizontal_label)
    axes[0].set_ylabel(vertical_label)
    plt.colorbar(im1, ax=axes[0], label="Field Magnitude")
    
    # Plot reconstructed field with normalized colorbar
    im2 = axes[1].imshow(recon_mag, cmap='viridis', origin='lower',
                       extent=extent,
                       vmin=global_min, vmax=global_max)
    axes[1].set_title(f"Reconstructed Field ({plane_type} Plane)")
    axes[1].set_xlabel(horizontal_label)
    axes[1].set_ylabel(vertical_label)
    plt.colorbar(im2, ax=axes[1], label="Field Magnitude")
    
    # Plot error
    im3 = axes[2].imshow(error, cmap='hot', origin='lower',
                       extent=extent)
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
