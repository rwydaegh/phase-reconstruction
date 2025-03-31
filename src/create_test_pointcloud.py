import numpy as np
from dataclasses import dataclass
from src.simulation_config_real_data import SimulationConfig

def create_test_pointcloud(config: SimulationConfig) -> np.ndarray:
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
    
    # Create points on each of the 6 walls of the cube centered at (0,0,0)
    # Front wall (x=-half_size)
    x, y, z = np.meshgrid([-half_size], np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n))
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))
    
    # Back wall (x=half_size)
    x, y, z = np.meshgrid([half_size], np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n))
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))
    
    # Left wall (y=-half_size)
    x, y, z = np.meshgrid(np.linspace(-half_size, half_size, n), [-half_size], np.linspace(-half_size, half_size, n))
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))
    
    # Right wall (y=half_size)
    x, y, z = np.meshgrid(np.linspace(-half_size, half_size, n), [half_size], np.linspace(-half_size, half_size, n))
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))
    
    # Bottom wall (z=-half_size)
    x, y, z = np.meshgrid(np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n), [-half_size])
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))
    
    # Top wall (z=half_size)
    x, y, z = np.meshgrid(np.linspace(-half_size, half_size, n), np.linspace(-half_size, half_size, n), [half_size])
    points.append(np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1))
    
    # Combine all points
    points = np.vstack(points)
    
    # Verify no interior points
    half_size = config.room_size / 2
    non_edge_points = (-half_size < points[:, 0]) & (points[:, 0] < half_size) & \
                       (-half_size < points[:, 1]) & (points[:, 1] < half_size) & \
                       (-half_size < points[:, 2]) & (points[:, 2] < half_size)
    if np.any(non_edge_points):
        print(f"WARNING: {np.sum(non_edge_points)} interior points found and will be removed")
        points = points[~non_edge_points]
    
    # Apply random perturbation if enabled
    if config.perturb_points:
        # Calculate distance from each point to origin
        distances = np.sqrt(np.sum(points**2, axis=1))
        
        # Generate random perturbations in range [-1, 1] for each direction
        random_perturbations = np.random.uniform(-1, 1, size=points.shape)
        
        # Scale perturbations by the perturbation factor and distance to origin
        # This ensures that points further from origin can be perturbed more
        # while maintaining the relative perturbation percentage constant
        scaled_perturbations = random_perturbations * config.perturbation_factor
        
        # Broadcasting to multiply each point's perturbation by its distance to origin
        scaled_perturbations = scaled_perturbations * distances[:, np.newaxis]
        
        # Apply perturbations to original points
        perturbed_points = points + scaled_perturbations
        
        # Use the perturbed points
        points = perturbed_points
        
        if config.verbose:
            print(f"Applied random perturbations with factor {config.perturbation_factor}")

    return points
