import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Optional, Any, Union
from src.utils.phase_retrieval_utils import (
    compute_pseudoinverse,
    initialize_field_values,
    calculate_error,
    apply_magnitude_constraint,
    create_convergence_plot
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def holographic_phase_retrieval(
    channel_matrix: np.ndarray,
    measured_magnitude: np.ndarray,
    num_iterations: int = 100,
    convergence_threshold: float = 1e-3,
    regularization: float = 1e-3,
    adaptive_regularization: bool = True,
    return_history: bool = False,
    verbose: bool = False,
    # Basic perturbation parameters
    enable_perturbations: bool = True,
    stagnation_window: int = 20,            # Window to detect stagnation
    stagnation_threshold: float = 1e-5,     # Threshold for meaningful improvements
    perturbation_mode: str = "basic",       # "none", "basic", "momentum", or "archived"
    perturbation_intensity: float = 0.2,    # Intensity of perturbations (added parameter)
    constraint_skip_iterations: int = 2,    # How many iterations to skip the constraint after perturbation
    momentum_factor: float = 0.8,           # Weight factor for momentum-based perturbation
    temperature: float = 5.0,               # Temperature for archived strategies
    # Visualization settings
    no_plot: bool = False,
    # Initialization parameters
    initial_field_values: Optional[np.ndarray] = None  # Optional custom field initialization
):
    """
    Basic holographic phase retrieval algorithm based on Gerchberg-Saxton with
    optional simple perturbation strategies.
    
    Args:
        channel_matrix: Matrix H relating clusters to measurement points
        measured_magnitude: Measured field magnitude
        num_iterations: Maximum number of iterations
        convergence_threshold: Convergence criterion
        regularization: Regularization parameter for SVD
        adaptive_regularization: Whether to use adaptive regularization
        return_history: Whether to return the history of cluster coefficients
        verbose: Whether to print verbose information
        enable_perturbations: Whether to enable perturbation strategies
        stagnation_window: Number of iterations to detect stagnation
        stagnation_threshold: Error improvement threshold to detect stagnation
        perturbation_mode: Which perturbation strategy to use
        perturbation_intensity: Intensity of perturbations (relative to field norm)
        constraint_skip_iterations: How many iterations to skip the constraint after perturbation
        momentum_factor: Weight factor for momentum-based perturbation
        temperature: Temperature parameter for archived strategies
        no_plot: Whether to create convergence plots
        
    Returns:
        Cluster coefficients and optionally history
    """
    # Convert to fortran-order arrays for better matrix multiplication performance
    measured_magnitude = np.asfortranarray(measured_magnitude)
    
    # Compute the pseudoinverse of the channel matrix with regularization
    H_pinv = compute_pseudoinverse(channel_matrix, regularization, adaptive_regularization)
    
    # Initialize with given field values or random phase if not provided
    if initial_field_values is not None:
        logger.info("Using provided initial field values")
        field_values = initial_field_values.copy()
    else:
        field_values = initialize_field_values(measured_magnitude)
    
    # Initialize error tracking
    errors = []
    best_coefficients = None
    best_error = float('inf')
    
    # Initialize history arrays if needed
    coefficient_history = [] if return_history else None
    field_history = [] if return_history else None
    
    # Precompute constant for normalization
    measured_magnitude_norm = np.linalg.norm(measured_magnitude)
    
    # Tracking variables for perturbations
    last_significant_improvement = 0
    perturbation_iterations = []
    previous_momentum = None  # For momentum-based perturbation
    
    # Variables to track post-perturbation progress
    post_perturbation_tracking = []
    current_tracking = None
    
    # Flag to control constraint skipping after perturbation
    skip_constraint_counter = 0
    
    for i in range(num_iterations):
        # Track state at start of iteration
        if i > 0 and verbose:
            try:
                start_magnitude = np.abs(channel_matrix @ cluster_coefficients)
                start_error = calculate_error(start_magnitude, measured_magnitude, measured_magnitude_norm)
                logger.debug(f"Iter {i}: START OF ITERATION. Error: {start_error:.4e}")
                if skip_constraint_counter > 0:
                    logger.info(f"Iter {i}: CONSTRAINT SKIPPING ACTIVE. Remaining: {skip_constraint_counter}")
            except:
                logger.debug(f"Iter {i}: START OF ITERATION. Cannot estimate start error.")
        
        # 1. Compute cluster coefficients
        cluster_coefficients = H_pinv @ field_values
        
        # 2. Forward transform from clusters to field
        simulated_field = channel_matrix @ cluster_coefficients
        
        if return_history:
            coefficient_history.append(cluster_coefficients.copy())
            field_history.append(simulated_field.copy())
        
        # 3. Calculate current error
        simulated_magnitude = np.abs(simulated_field)
        error = calculate_error(simulated_magnitude, measured_magnitude, measured_magnitude_norm)
        errors.append(error)
        
        # Print for every iteration
        logger.info(f"GS iteration {i}/{num_iterations}, error: {error:.6f}")
        
        # Save best coefficients
        if error < best_error:
            best_error = error
            best_coefficients = cluster_coefficients.copy()
            
            # Check if this is a significant improvement
            if i > 0 and (errors[last_significant_improvement] - error) > stagnation_threshold:
                last_significant_improvement = i
                
                # If we're tracking post-perturbation progress, update with success
                if current_tracking is not None and i - current_tracking["start_iter"] <= stagnation_window:
                    current_tracking["final_error"] = error
                    current_tracking["improvement"] = current_tracking["start_error"] - error
                    current_tracking["success"] = True
                    post_perturbation_tracking.append(current_tracking)
                    current_tracking = None
        
        # 4. Apply magnitude constraint (unless we're in constraint skipping mode)
        if skip_constraint_counter > 0:
            # Skip the constraint, but still preserve phase information
            # We will only update field_values in specific ways depending on the scenario
            skip_constraint_counter -= 1
            
            # Just use the current simulated field without constraint
            field_values = simulated_field.copy()
            
            if verbose:
                logger.info(f"Iter {i}: SKIPPING CONSTRAINT. Allowing perturbation to propagate further.")
        else:
            # Normal constraint application
            field_values = apply_magnitude_constraint(simulated_field, measured_magnitude)
        
        # 5. Check for convergence
        if error < convergence_threshold:
            if verbose:
                logger.info(f"Converged after {i+1} iterations with error {error:.6f}")
            break
        
        # 6. Check for stagnation and apply perturbation if needed
        if enable_perturbations and i - last_significant_improvement >= stagnation_window and skip_constraint_counter == 0:
            # We're stagnating - apply perturbation based on selected mode
            if perturbation_mode == "none":
                # Skip perturbation
                pass
            elif perturbation_mode == "basic":
                # Simple random perturbation
                field_values = apply_basic_perturbation(field_values, i, perturbation_intensity)
                perturbation_iterations.append(i)
                
                # Start tracking post-perturbation progress
                current_tracking = {
                    "start_iter": i,
                    "start_error": error,
                    "final_error": error,  # Initialize with current error
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "basic"
                }
                
                # Reset stagnation counter
                last_significant_improvement = i
                
                # Set the constraint skip counter
                skip_constraint_counter = constraint_skip_iterations
                
            elif perturbation_mode == "momentum":
                # Momentum-based perturbation
                field_values, previous_momentum = apply_momentum_perturbation(
                    field_values, error, previous_momentum, i, perturbation_intensity, momentum_factor
                )
                perturbation_iterations.append(i)
                
                # Start tracking post-perturbation progress
                current_tracking = {
                    "start_iter": i,
                    "start_error": error,
                    "final_error": error,  # Initialize with current error
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "momentum"
                }
                
                # Reset stagnation counter
                last_significant_improvement = i
                
                # Set the constraint skip counter
                skip_constraint_counter = constraint_skip_iterations
                
            elif perturbation_mode == "archived":
                # Use the archived complex strategies (via separate function)
                field_values = apply_archived_complex_strategies(
                    field_values, cluster_coefficients, error, i, perturbation_intensity, temperature
                )
                perturbation_iterations.append(i)
                
                # Start tracking post-perturbation progress
                current_tracking = {
                    "start_iter": i,
                    "start_error": error,
                    "final_error": error,  # Initialize with current error
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "archived"
                }
                
                # Reset stagnation counter
                last_significant_improvement = i
                
                # Set the constraint skip counter
                skip_constraint_counter = constraint_skip_iterations
        
        # If we're tracking post-perturbation progress and hit the end of tracking window
        if current_tracking is not None and i - current_tracking["start_iter"] >= stagnation_window:
            current_tracking["final_error"] = error
            current_tracking["improvement"] = current_tracking["start_error"] - error
            current_tracking["success"] = current_tracking["improvement"] > 0
            post_perturbation_tracking.append(current_tracking)
            current_tracking = None
    
    # Create convergence plot
    if not no_plot:
        create_convergence_plot(errors, perturbation_iterations, [], convergence_threshold)
    
    # Use the best coefficients found
    if best_error < error:
        if verbose:
            logger.info(f"Using best coefficients with error {best_error:.6f} instead of final error {error:.6f}")
        final_coefficients = best_coefficients
    else:
        final_coefficients = cluster_coefficients
    
    # Prepare statistics about perturbation effectiveness
    stats = {
        "iterations": i + 1,
        "final_error": error,
        "best_error": best_error,
        "num_perturbations": len(perturbation_iterations),
        "perturbation_iterations": perturbation_iterations,
        "post_perturbation_tracking": post_perturbation_tracking,
        "errors": errors
    }
    
    if verbose:
        logger.info(f"GS algorithm completed: {i+1} iterations, {len(perturbation_iterations)} perturbations, "
                   f"best error: {best_error:.6f}")
        
        # Report on perturbation effectiveness
        successful_perturbations = [p for p in post_perturbation_tracking if p["success"]]
        if post_perturbation_tracking:
            success_rate = len(successful_perturbations) / len(post_perturbation_tracking)
            avg_improvement = np.mean([p["improvement"] for p in successful_perturbations]) if successful_perturbations else 0
            logger.info(f"Perturbation success rate: {success_rate:.2f}, Average improvement: {avg_improvement:.6f}")
    
    if return_history:
        return final_coefficients, np.array(coefficient_history), np.array(field_history), stats
    else:
        return final_coefficients, stats


def apply_basic_perturbation(field_values: np.ndarray, iteration: int, intensity: float = 0.1) -> np.ndarray:
    """
    Apply a simple random perturbation to field values.
    
    Args:
        field_values: Current field values
        iteration: Current iteration number
        intensity: Perturbation intensity (relative to field norm)
    
    Returns:
        Perturbed field values
    """
    # Simple Gaussian perturbation
    field_norm = np.linalg.norm(field_values)
    perturbation_intensity = intensity
    
    # Scale perturbation relative to field norm
    perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(0, 1, field_values.shape)
    perturbation = perturbation * field_norm * perturbation_intensity / np.linalg.norm(perturbation)
    
    # Apply perturbation
    perturbed_values = field_values + perturbation
    
    # Log perturbation details
    perturbation_magnitude = np.linalg.norm(perturbation)
    relative_perturbation = perturbation_magnitude / field_norm
    logger.info(f"Iter {iteration}: Applied BASIC perturbation. Relative magnitude: {relative_perturbation*100:.2f}%")
    
    return perturbed_values


def apply_momentum_perturbation(
    field_values: np.ndarray, 
    current_error: float,
    previous_momentum: Optional[np.ndarray],
    iteration: int,
    intensity: float = 0.2,
    momentum_factor: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a momentum-based perturbation to help escape local minima.
    
    Args:
        field_values: Current field values
        current_error: Current error
        previous_momentum: Previous momentum direction (if any)
        iteration: Current iteration number
        intensity: Perturbation intensity (relative to field norm)
        momentum_factor: Weight factor for previous momentum
    
    Returns:
        Perturbed field values and new momentum
    """
    field_norm = np.linalg.norm(field_values)
    perturbation_intensity = intensity
    
    # Generate random perturbation
    random_perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(0, 1, field_values.shape)
    random_perturbation = random_perturbation * field_norm * perturbation_intensity / np.linalg.norm(random_perturbation)
    
    # Apply momentum if available
    if previous_momentum is not None:
        # Combine random perturbation with previous momentum
        combined_perturbation = random_perturbation + momentum_factor * previous_momentum
        # Normalize to desired intensity
        combined_perturbation = combined_perturbation * field_norm * perturbation_intensity / np.linalg.norm(combined_perturbation)
        new_momentum = combined_perturbation
    else:
        combined_perturbation = random_perturbation
        new_momentum = random_perturbation
    
    # Apply perturbation
    perturbed_values = field_values + combined_perturbation
    
    # Log perturbation details
    perturbation_magnitude = np.linalg.norm(combined_perturbation)
    relative_perturbation = perturbation_magnitude / field_norm
    logger.info(f"Iter {iteration}: Applied MOMENTUM perturbation. Relative magnitude: {relative_perturbation*100:.2f}%")
    
    return perturbed_values, new_momentum


def apply_archived_complex_strategies(
    field_values: np.ndarray, 
    cluster_coefficients: np.ndarray,
    current_error: float,
    iteration: int,
    intensity: float = 0.3,
    temperature: float = 5.0
) -> np.ndarray:
    """
    Apply the archived complex perturbation strategies from the original implementation.
    This function provides access to the complex strategies that were archived.
    
    Args:
        field_values: Current field values
        cluster_coefficients: Current cluster coefficients
        current_error: Current error
        iteration: Current iteration number
        intensity: Perturbation intensity (relative to field norm)
        temperature: Temperature parameter for simulated annealing-like strategies
    
    Returns:
        Perturbed field values
    """
    # Import the archived strategies module
    try:
        from src.utils.archived_strategies import apply_perturbation_strategy
        
        # Use a random strategy from the archived ones
        strategy = np.random.randint(0, 7)  # 7 different strategies in the archive
        memory_solutions = []  # We don't track memory solutions in this simple approach
        memory_errors = []
        best_coefficients = None
        best_error = current_error
        
        # Apply the selected strategy with the specified intensity
        perturbation_intensity = intensity
        # Use provided temperature parameter
        
        # Call the archived strategy function
        perturbed_values, skip_constraint, _, _, _, _ = apply_perturbation_strategy(
            field_values, 
            cluster_coefficients, 
            best_coefficients, 
            best_error,
            memory_solutions, 
            memory_errors, 
            strategy, 
            perturbation_intensity,
            temperature, 
            iteration, 
            field_values.copy(),  # Use current field as "previous" since we don't track
            verbose=True
        )
        
        logger.info(f"Iter {iteration}: Applied ARCHIVED STRATEGY {strategy}")
        return perturbed_values
        
    except ImportError:
        # Fallback if the archived strategies module is not available
        logger.warning("Archived strategies module not found. Using basic perturbation instead.")
        field_norm = np.linalg.norm(field_values)
        perturbation_intensity = intensity
        
        # Scale perturbation relative to field norm
        perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(0, 1, field_values.shape)
        perturbation = perturbation * field_norm * perturbation_intensity / np.linalg.norm(perturbation)
        
        # Apply perturbation
        perturbed_values = field_values + perturbation
        
        # Log perturbation details
        perturbation_magnitude = np.linalg.norm(perturbation)
        relative_perturbation = perturbation_magnitude / field_norm
        logger.info(f"Iter {iteration}: Applied basic perturbation as fallback. Relative magnitude: {relative_perturbation*100:.2f}%")
