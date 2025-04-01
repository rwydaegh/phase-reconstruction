from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig


def create_test_pointcloud(config: DictConfig) -> np.ndarray:
    """Create a point cloud representing walls of a cubic room centered at origin (0,0,0).

    Args:
        config: Simulation configuration

    Returns:
        Array of (x,y,z) coordinates for points on the walls only

    Note:
        If config.perturb_points is True, the point positions will be randomly
        perturbed by a factor of config.perturbation_factor of their distance to origin.
    """
    points = []
    n = int(config.wall_points)
    size = config.room_size
    half_size = size / 2

    x, y, z = np.meshgrid(
        [-half_size], np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n)
    )
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))

    x, y, z = np.meshgrid(
        [half_size], np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n)
    )
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))

    x, y, z = np.meshgrid(
        np.linspace(-half_size, half_size, n), [-half_size], np.linspace(-half_size, half_size, n)
    )
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))

    x, y, z = np.meshgrid(
        np.linspace(-half_size, half_size, n), [half_size], np.linspace(-half_size, half_size, n)
    )
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))

    x, y, z = np.meshgrid(
        np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n), [-half_size]
    )
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))

    x, y, z = np.meshgrid(
        np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n), [half_size]
    )
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))

    points = np.vstack(points)

    # Removed incorrect filtering logic that removed interior face points.
    # The function should return all points generated on the faces.
    # Filtering for single-face points is now handled correctly in get_cube_normals.

    if config.perturb_points:
        distances = np.sqrt(np.sum(points**2, axis=1))

        random_perturbations = np.random.uniform(-1, 1, size=points.shape)

        # Scale perturbations by the perturbation factor and distance to origin
        scaled_perturbations = random_perturbations * config.perturbation_factor

        scaled_perturbations = scaled_perturbations * distances[:, np.newaxis]

        perturbed_points = points + scaled_perturbations

        points = perturbed_points

        if config.verbose:
            print(f"Applied random perturbations with factor {config.perturbation_factor}")

    return points
