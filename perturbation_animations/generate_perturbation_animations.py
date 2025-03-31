import numpy as np
import random
import logging
import os
import sys
from dataclasses import dataclass, fields

# Add parent directory to path to find src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from src directory
from src.create_test_pointcloud import create_test_pointcloud
from src.holographic_phase_retrieval import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix
from src.utils.field_utils import compute_fields, reconstruct_field
from src.visualization import visualize_current_and_field_history
from src.simulation_config_real_data import SimulationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_perturbation_animation(perturbation_factor, output_dir="perturbation_results"):
    """Run a simulation with perturbation and generate animation.
    
    Args:
        perturbation_factor: Factor to perturb points (0.001 = 0.1%)
        output_dir: Directory to save results
    """
    # Create configuration with default values
    config = SimulationConfig()
    
    # Set simulation parameters
    config.perturb_points = True
    config.perturbation_factor = perturbation_factor
    config.return_history = True
    config.show_plot = False
    
    logger.info(f"Running simulation with perturbation_factor={perturbation_factor} ({perturbation_factor*100:.1f}%)")
    
    # Create test environment
    logger.info("Creating test point cloud...")
    points = create_test_pointcloud(config)
    
    # Create simple current distribution (scalar amplitudes for each point)
    currents = np.zeros(len(points), dtype=complex)
    
    # Set random indices in the currents array to be active
    # with log-normal amplitude distribution and random phases
    num_sources = min(config.num_sources, len(currents))
    source_indices = random.sample(range(len(currents)), num_sources)
    
    # Generate log-normal amplitudes
    amplitudes = np.random.lognormal(mean=0, sigma=3, size=num_sources)
    
    # Generate random phases between 0 and 2π
    phases = np.random.uniform(0, 2*np.pi, size=num_sources)
    
    # Set complex currents with amplitude and phase
    for i, idx in enumerate(source_indices):
        # Convert amplitude and phase to complex number: A * e^(iθ)
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])
    else:
        # Fallback: use first point
        currents[0] = 1.0
    
    # Create measurement plane
    logger.info("Creating measurement plane...")
    x = np.linspace(config.room_size/2 - config.plane_size/2, config.room_size/2 + config.plane_size/2, config.resolution)
    y = np.linspace(config.room_size/2 - config.plane_size/2, config.room_size/2 + config.plane_size/2, config.resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.ones_like(X) * config.room_size / 2], axis=-1)
    
    # Create channel matrix for scalar fields
    logger.info("Creating channel matrix...")
    H = create_channel_matrix(points, measurement_plane, config.k)
    
    # Calculate ground truth field on measurement plane
    logger.info("Calculating ground truth field...")
    true_field = compute_fields(points, currents, measurement_plane, config.k, H)
    
    # Get field magnitude (what we would measure)
    measured_magnitude = np.abs(true_field)
    
    # Phase retrieval demonstration
    logger.info("Running holographic phase retrieval...")
    hpr_result = holographic_phase_retrieval(
        H, measured_magnitude,
        adaptive_regularization=config.adaptive_regularization,
        num_iterations=config.gs_iterations,
        convergence_threshold=config.convergence_threshold,
        regularization=1e-3,
        return_history=config.return_history,
        debug=config.debug
    )

    # Extract results
    cluster_coefficients, coefficient_history, field_history = hpr_result
    logger.info(f"Coefficient history shape: {coefficient_history.shape}")
    logger.info(f"Field history shape: {field_history.shape}")
    
    # Create enhanced animation with 3D current visualization and 2D field history with true field and error
    animation_filename = f"{output_dir}/current_field_animation_{perturbation_factor*100:.1f}percent.gif"
    logger.info(f"Creating enhanced 4-panel animation: {animation_filename}")
    
    visualize_current_and_field_history(
        points,
        coefficient_history,
        field_history,
        true_field,  # Pass the true field for comparison
        config.resolution,
        measurement_plane,
        show_plot=False,
        output_file=animation_filename
    )
    
    # Reconstruct field using estimated coefficients
    reconstructed_field = reconstruct_field(H, cluster_coefficients)
    
    # Calculate metrics
    def normalized_rmse(a, b):
        return np.sqrt(np.mean((a - b)**2)) / (np.max(a) - np.min(a))
        
    def normalized_correlation(a, b):
        a_norm = (a - np.mean(a)) / np.std(a)
        b_norm = (b - np.mean(b)) / np.std(b)
        return np.correlate(a_norm.flatten(), b_norm.flatten())[0] / len(a_norm.flatten())

    # Calculate reconstruction quality metrics
    rmse = normalized_rmse(np.abs(true_field), np.abs(reconstructed_field))
    corr = normalized_correlation(np.abs(true_field), np.abs(reconstructed_field))
    
    # Print metrics
    logger.info(f"Reconstruction quality metrics for perturbation {perturbation_factor*100:.1f}%:")
    logger.info(f"  Normalized RMSE: {rmse:.4f}")
    logger.info(f"  Correlation: {corr:.4f}")
    
    return rmse, corr

def main():
    """Run perturbation tests and generate animations."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directory for perturbation results if it doesn't exist
    output_dir = os.path.join(script_dir, "perturbation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the perturbation factors
    perturbation_factors = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1%
    results = []
    
    for factor in perturbation_factors:
        rmse, corr = run_perturbation_animation(factor, output_dir)
        results.append((factor, rmse, corr))
        logger.info(f"✓ Completed simulation with perturbation_factor={factor}")
    
    # Print summary of results
    logger.info("\nSummary of Results:")
    logger.info("-" * 50)
    logger.info("Perturbation Factor | Normalized RMSE | Correlation")
    logger.info("-" * 50)
    
    for factor, rmse, corr in results:
        logger.info(f"{factor*100:16.1f}% | {rmse:14.4f} | {corr:11.4f}")
    
    logger.info("\nAll simulations completed.")
    logger.info(f"Animation files are available in {output_dir}/")

if __name__ == "__main__":
    main()
