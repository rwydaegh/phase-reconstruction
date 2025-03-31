import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


def load_measurement_data(file_path):
    """Load measurement data from pickle file"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def gaussian_kernel_2d(size, sigma):
    """
    Generate a 2D Gaussian kernel with specified size and sigma
    """
    x, y = np.meshgrid(
        np.linspace(-size / 2, size / 2, size), np.linspace(-size / 2, size / 2, size)
    )
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    return g / g.sum()  # Normalize


def gaussian_convolution_local_support(data, R_pixels):
    """
    Apply Gaussian convolution with local support R
    R_pixels: Radius in pixels
    """
    # Create a copy of the data to avoid modifying the original
    smoothed_data = np.copy(data)

    # Calculate sigma based on R (standard choice: sigma = R/3)
    sigma = R_pixels / 3.0

    # Use scipy's gaussian_filter for efficient implementation
    smoothed_data = gaussian_filter(smoothed_data, sigma=sigma, mode="nearest")

    return smoothed_data


def physical_to_pixel_radius(R_mm, points_continuous, points_discrete):
    """
    Convert physical radius (mm) to pixel radius
    """
    # Calculate average spacing in continuous and discrete dimensions
    cont_spacing = (np.max(points_continuous) - np.min(points_continuous)) / (
        len(points_continuous) - 1
    )

    # Handle case where points_discrete might be a list
    if isinstance(points_discrete, list):
        disc_points = np.array(points_discrete)
    else:
        disc_points = points_discrete

    disc_spacing = (np.max(disc_points) - np.min(disc_points)) / (len(disc_points) - 1)

    # Average spacing
    avg_spacing = (cont_spacing + disc_spacing) / 2

    # Convert R in mm to pixels
    R_pixels = R_mm / avg_spacing

    return R_pixels


def main():
    # Load measurement data
    file_path = "measurement_data/x0_zy.pickle"
    data = load_measurement_data(file_path)

    # Extract relevant data
    results = data["results"]
    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]
    frequency = data["frequency"]

    # Convert points_discrete to numpy array if it's a list
    if isinstance(points_discrete, list):
        points_discrete = np.array(points_discrete)

    # Create meshgrid for plotting
    cc, dd = np.meshgrid(points_continuous, points_discrete)

    # Define a set of increasing R values (in mm)
    R_values_mm = [0, 1, 3, 5, 10, 20]

    # Prepare a figure for comparison
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 3, figure=fig)

    # Plot each smoothed version
    for i, R_mm in enumerate(R_values_mm):
        ax = fig.add_subplot(gs[i // 3, i % 3])

        if R_mm == 0:
            # Original data without smoothing
            smoothed = results
            title = "Original (No Smoothing)"
        else:
            # Convert physical radius to pixel radius
            R_pixels = physical_to_pixel_radius(R_mm, points_continuous, points_discrete)

            # Apply Gaussian smoothing
            smoothed = gaussian_convolution_local_support(results, R_pixels)
            title = f"R = {R_mm} mm"

        # Plot the data
        im = ax.contourf(cc, dd, smoothed, levels=50, cmap="jet")
        ax.set_title(title)
        ax.set_xlabel(f"{continuous_axis} [mm]")
        ax.set_ylabel(f"{discrete_axis} [mm]")

        # Calculate and show metrics
        if R_mm > 0:
            rmse = np.sqrt(np.mean((smoothed - results) ** 2))
            max_diff = np.max(np.abs(smoothed - results))
            ax.text(
                0.03,
                0.97,
                f"RMSE: {rmse:.4f}\nMax Diff: {max_diff:.4f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7},
            )

    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="E-field magnitude [V/m]")

    # Add title for the entire figure
    plt.suptitle(
        f"Gaussian Smoothing with Increasing Support Radius\nEx-field at {frequency/1e9:.2f} GHz",
        fontsize=16,
    )

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    output_dir = "smoothing_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "gaussian_smoothing_comparison.png"), dpi=300)

    # Also create individual plots for each R value with the same colorbar scale
    print("Creating individual plots...")
    vmin, vmax = np.min(results), np.max(results)

    for _, R_mm in enumerate(R_values_mm):
        plt.figure(figsize=(10, 8))

        if R_mm == 0:
            smoothed = results
            title = "Original (No Smoothing)"
        else:
            R_pixels = physical_to_pixel_radius(R_mm, points_continuous, points_discrete)
            smoothed = gaussian_convolution_local_support(results, R_pixels)
            title = f"Gaussian Smoothing with R = {R_mm} mm"

        plt.contourf(cc, dd, smoothed, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
        plt.colorbar(label="E-field magnitude [V/m]")
        plt.title(title)
        plt.xlabel(f"{continuous_axis} [mm]")
        plt.ylabel(f"{discrete_axis} [mm]")

        filename = f"smoothed_R{R_mm}mm.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    print(f"All plots saved to {output_dir}/")
    return data, R_values_mm


if __name__ == "__main__":
    main()
