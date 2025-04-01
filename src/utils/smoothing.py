import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_kernel_2d(size, sigma):
    """
    Generate a 2D Gaussian kernel with specified size and sigma
    """
    x, y = np.meshgrid(
        np.linspace(-size / 2, size / 2, size), np.linspace(-size / 2, size / 2, size)
    )
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    # Add a small epsilon to prevent division by zero if the sum is zero
    g_sum = g.sum()
    if g_sum == 0:
        # Handle the case where the kernel sums to zero (e.g., size=0 or sigma=0)
        # Return a kernel with a single peak at the center or handle as appropriate
        # For simplicity, returning the current 'g' which might be all zeros or contain NaNs
        # A better approach might be to return an identity kernel or raise an error
        return g # Or handle appropriately, e.g., raise ValueError("Invalid kernel parameters")
    return g / g_sum  # Normalize


def gaussian_convolution_local_support(data, R_pixels):
    """
    Apply Gaussian convolution with local support R
    R_pixels: Radius in pixels
    """
    # Create a copy of the data to avoid modifying the original
    smoothed_data = np.copy(data)

    # Calculate sigma based on R (standard choice: sigma = R/3)
    # Ensure sigma is positive
    sigma = max(R_pixels / 3.0, 1e-9) # Use max to avoid non-positive sigma

    # Use scipy's gaussian_filter for efficient implementation
    # Use reflect mode to handle boundaries better for physical data
    smoothed_data = gaussian_filter(smoothed_data, sigma=sigma, mode="reflect")

    return smoothed_data


def physical_to_pixel_radius(R_mm, points_continuous, points_discrete):
    """
    Convert physical radius (mm) to pixel radius
    """
    # Ensure points are numpy arrays
    points_continuous = np.asarray(points_continuous)
    points_discrete = np.asarray(points_discrete)

    # Calculate average spacing in continuous and discrete dimensions
    # Handle cases with fewer than 2 points to avoid division by zero
    if len(points_continuous) > 1:
        cont_spacing = (np.max(points_continuous) - np.min(points_continuous)) / (
            len(points_continuous) - 1
        )
    else:
        cont_spacing = 1.0 # Assign a default or handle as error

    if len(points_discrete) > 1:
        disc_spacing = (np.max(points_discrete) - np.min(points_discrete)) / (len(points_discrete) - 1)
    else:
        disc_spacing = 1.0 # Assign a default or handle as error

    # Check for zero spacing
    if cont_spacing == 0 or disc_spacing == 0:
        # Handle cases where spacing is zero (e.g., all points are the same)
        # Returning 1 pixel radius or raising an error might be appropriate
        # print("Warning: Zero spacing detected in one or both dimensions.")
        # For now, use a small default if one is zero, or average if both > 0
        if cont_spacing == 0 and disc_spacing == 0:
             avg_spacing = 1.0 # Or raise error
        elif cont_spacing == 0:
             avg_spacing = disc_spacing
        elif disc_spacing == 0:
             avg_spacing = cont_spacing
        else: # Should not happen based on logic, but for completeness
             avg_spacing = (cont_spacing + disc_spacing) / 2
    else:
        # Average spacing
        avg_spacing = (cont_spacing + disc_spacing) / 2

    # Avoid division by zero if avg_spacing is somehow zero
    if avg_spacing == 0:
        # raise ValueError("Average spacing cannot be zero.")
        # Or return a default pixel value
        return 1.0 # Default to 1 pixel radius

    # Convert R in mm to pixels
    R_pixels = R_mm / avg_spacing

    return R_pixels