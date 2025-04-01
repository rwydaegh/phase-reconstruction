import logging
import os

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def compute_pseudoinverse(channel_matrix, regularization, adaptive_regularization):
    """Compute regularized pseudoinverse of the channel matrix.

    Args:
        channel_matrix: Matrix H relating clusters to measurement points
        regularization: Regularization parameter for SVD
        adaptive_regularization: Whether to use adaptive regularization

    Returns:
        Regularized pseudoinverse of the channel matrix
    """
    # Convert to fortran-order arrays for better matrix multiplication performance
    channel_matrix = np.asfortranarray(channel_matrix)

    # Use standard SVD for better numerical stability
    U, S, Vh = np.linalg.svd(channel_matrix, full_matrices=False)

    if adaptive_regularization:
        # Adaptive Tikhonov regularization
        tau = regularization * S[0]  # adaptive threshold
        S_reg = S / (S**2 + tau**2)
    else:
        S_reg = S / (S**2 + regularization)

    # Optimized pseudoinverse calculation
    H_pinv = np.asfortranarray((Vh.conj().T * S_reg) @ U.conj().T)

    return H_pinv


def initialize_field_values(measured_magnitude):
    """Initialize field values with random phase.

    Args:
        measured_magnitude: Measured field magnitude

    Returns:
        Initial field values with random phase
    """
    # Initialize with random phase (using complex random values for better initial diversity)
    field_values = measured_magnitude * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, measured_magnitude.shape)
    )
    return field_values


def calculate_error(simulated_magnitude, measured_magnitude, measured_magnitude_norm):
    """Calculate relative error between simulated and measured magnitudes.

    Args:
        simulated_magnitude: Simulated field magnitude
        measured_magnitude: Measured field magnitude
        measured_magnitude_norm: Precomputed norm of the measured magnitude

    Returns:
        Relative error
    """
    mag_diff = simulated_magnitude - measured_magnitude
    measured_magnitude_norm = np.linalg.norm(measured_magnitude)
    if measured_magnitude_norm > 1e-10:
        error = np.linalg.norm(mag_diff) / measured_magnitude_norm
    else:
        error = np.inf
    return error


def apply_magnitude_constraint(
    simulated_field, measured_magnitude, skip_magnitude_constraint=False
):
    """Apply magnitude constraint to the field values.

    Args:
        simulated_field: Simulated field values
        measured_magnitude: Measured field magnitude
        skip_magnitude_constraint: Whether to skip the magnitude constraint

    Returns:
        Field values after applying magnitude constraint
    """
    if skip_magnitude_constraint:
        return simulated_field

    # Calculate magnitude
    simulated_magnitude = np.abs(simulated_field)

    # Handle division by zero or near-zero magnitudes
    # Create masks for zero and near-zero simulated magnitudes
    zero_mask = np.abs(simulated_field) < 1e-15 # Mask for exact zeros
    # Mask for near-zeros
    near_zero_mask = (np.abs(simulated_field) >= 1e-15) & (simulated_magnitude < 1e-10)

    # Initialize output field values
    field_values = np.zeros_like(simulated_field, dtype=np.complex128)

    # Case 1: Simulated field is exactly zero - assign measured magnitude with zero phase
    if np.any(zero_mask):
        field_values[zero_mask] = measured_magnitude[zero_mask] # Assign magnitude, phase is 0

    # Case 2: Simulated field is near-zero (but not exactly zero)
    # Use small value for stable division
    if np.any(near_zero_mask):
        # Calculate phase using the original simulated_field to avoid phase distortion
        phase_factor = simulated_field[near_zero_mask] / np.abs(simulated_field[near_zero_mask])
        field_values[near_zero_mask] = measured_magnitude[near_zero_mask] * phase_factor

    # Case 3: Normal case - simulated magnitude is sufficiently large
    normal_mask = ~(zero_mask | near_zero_mask)
    if np.any(normal_mask):
        # Apply magnitude constraint while preserving phase
        phase_factor = simulated_field[normal_mask] / simulated_magnitude[normal_mask]
        field_values[normal_mask] = measured_magnitude[normal_mask] * phase_factor

    return field_values


def check_stagnation(i, last_significant_improvement, stagnation_window):
    """Check if the algorithm is stagnating.

    Args:
        i: Current iteration
        last_significant_improvement: Iteration of the last significant improvement
        stagnation_window: Number of iterations to detect stagnation

    Returns:
        Whether stagnation is detected
    """
    return i - last_significant_improvement >= stagnation_window


def update_memory_solutions(
    best_coefficients, best_error, memory_solutions, memory_errors, memory_ensemble_size
):
    """Update memory of good solutions.

    Args:
        best_coefficients: Best cluster coefficients found
        best_error: Error associated with the best coefficients
        memory_solutions: Current memory of good solutions
        memory_errors: Errors associated with memory solutions
        memory_ensemble_size: Maximum number of solutions to remember

    Returns:
        Updated memory_solutions and memory_errors
    """
    if best_coefficients is not None:
        # Check if this solution is good enough to be in the memory
        if len(memory_solutions) < memory_ensemble_size:
            memory_solutions.append(best_coefficients.copy())
            memory_errors.append(best_error)
        else:
            # Find the worst solution in memory
            worst_idx = np.argmax(memory_errors)

            # Replace if this solution is better
            if best_error < memory_errors[worst_idx]:
                memory_solutions[worst_idx] = best_coefficients.copy()
                memory_errors[worst_idx] = best_error

    return memory_solutions, memory_errors


def apply_basic_perturbation(field_values, iteration, perturbation_intensity=0.1, verbose=False):
    """Apply a simple random perturbation to field values.

    Args:
        field_values: Current field values
        iteration: Current iteration number
        perturbation_intensity: Intensity of the perturbation
        verbose: Whether to print verbose information

    Returns:
        Perturbed field values
    """
    # Simple Gaussian perturbation
    field_norm = np.linalg.norm(field_values)

    # Scale perturbation relative to field norm
    perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(
        0, 1, field_values.shape
    )
    perturbation = perturbation * field_norm * perturbation_intensity / np.linalg.norm(perturbation)

    # Apply perturbation
    perturbed_values = field_values + perturbation

    # Log perturbation details
    perturbation_magnitude = np.linalg.norm(perturbation)
    relative_perturbation = perturbation_magnitude / field_norm
    if verbose:
        logger.info(
            f"Iter {iteration}: Applied BASIC perturbation. "
            f"Relative magnitude: {relative_perturbation*100:.2f}%"
        )

    return perturbed_values


def apply_momentum_perturbation(
    field_values,
    previous_momentum=None,
    perturbation_intensity=0.2,
    momentum_factor=0.8,
    iteration=0,
    verbose=False,
):
    """Apply a momentum-based perturbation to help escape local minima.

    Args:
        field_values: Current field values
        previous_momentum: Previous momentum direction (if any)
        perturbation_intensity: Intensity of the perturbation
        momentum_factor: Weight given to previous momentum
        iteration: Current iteration number
        verbose: Whether to print verbose information

    Returns:
        Tuple of (perturbed field values, new momentum)
    """
    field_norm = np.linalg.norm(field_values)

    # Generate random perturbation
    random_perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(
        0, 1, field_values.shape
    )
    random_perturbation = (
        random_perturbation
        * field_norm
        * perturbation_intensity
        / np.linalg.norm(random_perturbation)
    )

    # Apply momentum if available
    if previous_momentum is not None:
        # Combine random perturbation with previous momentum
        combined_perturbation = random_perturbation + momentum_factor * previous_momentum
        # Normalize to desired intensity
        combined_perturbation = (
            combined_perturbation
            * field_norm
            * perturbation_intensity
            / np.linalg.norm(combined_perturbation)
        )
        new_momentum = combined_perturbation
    else:
        combined_perturbation = random_perturbation
        new_momentum = random_perturbation

    # Apply perturbation
    perturbed_values = field_values + combined_perturbation

    # Log perturbation details
    perturbation_magnitude = np.linalg.norm(combined_perturbation)
    relative_perturbation = perturbation_magnitude / field_norm
    if verbose:
        logger.info(
            f"Iter {iteration}: Applied MOMENTUM perturbation. "
            f"Relative magnitude: {relative_perturbation*100:.2f}%"
        )

    return perturbed_values, new_momentum


def perform_intelligent_restart(
    measured_magnitude,
    channel_matrix,
    memory_solutions,
    memory_errors,
    temperature,
    temperature_initial,
    memory_contribution,
    verbose=False,
):
    """Perform an intelligent restart to escape persistent stagnation.

    Args:
        measured_magnitude: Measured field magnitude
        channel_matrix: Matrix H relating clusters to measurement points
        memory_solutions: Memory of good solutions
        memory_errors: Errors associated with memory solutions
        temperature: Current temperature for simulated annealing
        temperature_initial: Initial temperature for simulated annealing
        memory_contribution: Weight of ensemble solutions in restart
        verbose: Whether to print verbose information

    Returns:
        New field values after restart
    """
    if memory_solutions and np.random.rand() < 0.9:  # 90% chance to use memory
        # Create an ensemble solution using memory of good solutions
        ensemble_coefficients = np.zeros_like(memory_solutions[0])
        total_weight = 0

        # Weight solutions by their quality (inverse error)
        for soln, err in zip(memory_solutions, memory_errors):
            weight = 1.0 / (err + 1e-10)  # Avoid division by zero
            ensemble_coefficients += weight * soln
            total_weight += weight

        if total_weight > 0:
            ensemble_coefficients /= total_weight

        # Mix ensemble with randomization (weighted by temperature)
        random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, measured_magnitude.shape))
        random_field = measured_magnitude * random_phase

        # Compute field from ensemble coefficients
        ensemble_field = channel_matrix @ ensemble_coefficients
        # ensemble_magnitude = np.abs(ensemble_field) # Unused
        ensemble_phase = np.angle(ensemble_field)
        ensemble_field_normalized = measured_magnitude * np.exp(1j * ensemble_phase)

        # Mix based on temperature and memory contribution
        # Higher mix_ratio to prefer ensemble solutions over random ones
        mix_ratio = memory_contribution * (1 - temperature / temperature_initial)
        field_values = mix_ratio * ensemble_field_normalized + (1 - mix_ratio) * random_field

        # Add additional noise to escape local minima, but more controlled
        noise_level = 0.15 * (1.0 - mix_ratio)  # Less noise when using good ensemble solutions
        noise = np.random.normal(0, noise_level, field_values.shape) + 1j * np.random.normal(
            0, noise_level, field_values.shape
        )
        field_values += np.mean(np.abs(field_values)) * noise
    else:
        # Complete random restart with some diversity
        field_values = measured_magnitude * np.exp(
            1j * np.random.uniform(0, 2 * np.pi, measured_magnitude.shape)
        )

    if verbose:
        restart_type = "memory-based ensemble" if memory_solutions else "random initialization"
        logger.info(
            f"Performing intelligent restart with {restart_type}"
        )

    return field_values


def create_convergence_plot(
    errors,
    perturbation_iterations,
    restart_iterations,
    convergence_threshold,
    output_dir: str, # Add output directory argument
):
    """Create and save a plot showing the error evolution and perturbation points.

    Args:
        errors: List of errors at each iteration
        perturbation_iterations: Iterations where perturbations were applied
        restart_iterations: Iterations where restarts occurred
        convergence_threshold: Convergence criterion
        output_dir: Directory to save the plot file.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True) # Use the provided output directory

        # Calculate rate of change (first derivative of error)
        # Compute differences between consecutive errors, padded with 0 at the beginning
        error_rate_of_change = np.zeros_like(errors)
        error_rate_of_change[1:] = np.abs(np.array(errors[1:]) - np.array(errors[:-1]))

        # Set up figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Primary y-axis for error (left side)
        color1 = "blue"
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Relative Error (log scale)", color=color1)
        (error_line,) = ax1.semilogy(errors, color=color1, label="Error")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Secondary y-axis for rate of change (right side)
        ax2 = ax1.twinx()
        color2 = "red"
        ax2.set_ylabel("Rate of Change", color=color2)
        (rate_line,) = ax2.semilogy(
            error_rate_of_change, color=color2, linestyle="--", label="Rate of Change"
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        # Add title
        plt.title("GS Algorithm Convergence with Escape Mechanisms")

        # Mark perturbations on the plot
        perturbation_lines = []
        for iter_idx in perturbation_iterations:
            line = ax1.axvline(x=iter_idx, color="g", linestyle=":", alpha=0.7)
            if len(perturbation_lines) == 0:
                perturbation_lines.append(line)

        # Mark random restarts on the plot
        restart_lines = []
        for iter_idx in restart_iterations:
            line = ax1.axvline(x=iter_idx, color="m", linestyle="--", alpha=0.7)
            if len(restart_lines) == 0:
                restart_lines.append(line)

        # Add threshold line
        threshold_line = ax1.axhline(
            y=convergence_threshold,
            color="k",
            linestyle="-.",
            label=f"Threshold ({convergence_threshold})",
        )

        # Create combined legend
        legend_elements = [error_line, rate_line, threshold_line]
        legend_labels = ["Error", "Rate of Change", f"Threshold ({convergence_threshold:.1e})"]

        if perturbation_iterations and perturbation_lines:
            legend_elements.append(perturbation_lines[0])
            legend_labels.append("Perturbation Applied")

        if restart_iterations and restart_lines:
            legend_elements.append(restart_lines[0])
            legend_labels.append("Random Restart")

        fig.legend(legend_elements, legend_labels, loc="upper right", bbox_to_anchor=(0.95, 0.95))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gs_convergence.png"), dpi=300) # Save to output directory

        plt.close()

    except Exception as e:
        logger.warning(f"Could not create convergence plot: {str(e)}")


def compile_statistics(
    iterations,
    error,
    best_error,
    perturbation_iterations,
    restart_iterations,
    temperature,
    strategy_success_counts,
    strategy_attempt_counts,
    errors,
    evaluation_delay=8,
):
    """Compile statistics about the algorithm performance and escape strategies.

    Args:
        iterations: Number of iterations performed
        error: Final error
        best_error: Best error found
        perturbation_iterations: Iterations where perturbations were applied
        restart_iterations: Iterations where restarts occurred
        temperature: Final temperature
        strategy_success_counts: Success counts for each strategy
        strategy_attempt_counts: Attempt counts for each strategy
        errors: List of errors at each iteration
        evaluation_delay: Number of iterations to wait before evaluating perturbation effectiveness

    Returns:
        Dictionary of statistics
    """
    # Compile enhanced statistics about the escape strategies
    stats = {
        "total_iterations": iterations,
        "final_error": error,
        "best_error": best_error,
        "perturbations_applied": len(perturbation_iterations),
        "random_restarts": len(restart_iterations),
        "perturbation_iterations": perturbation_iterations,
        "restart_iterations": restart_iterations,
        "final_temperature": temperature,
    }

    # Add strategy effectiveness metrics
    strategy_success_rates = strategy_success_counts / strategy_attempt_counts
    for s in range(7):
        stats[f"strategy_{s}_success_rate"] = strategy_success_rates[s]
        stats[f"strategy_{s}_attempts"] = strategy_attempt_counts[s]

    # Add effectiveness metrics if perturbations were applied
    if perturbation_iterations:
        # Calculate average error reduction after perturbations
        error_before_perturbation_list = [
            errors[max(0, idx - 1)] for idx in perturbation_iterations
        ]
        error_after_perturbation_list = []

        for idx in perturbation_iterations:
            eval_idx = min(len(errors) - 1, idx + evaluation_delay)
            error_after_perturbation_list.append(errors[eval_idx])

        error_reduction = [
            before - after
            for before, after in zip(error_before_perturbation_list, error_after_perturbation_list)
        ]

        if error_reduction:
            stats["avg_error_reduction_after_perturbation"] = np.mean(error_reduction)
            stats["perturbation_effectiveness"] = np.sum([r > 0 for r in error_reduction]) / len(
                error_reduction
            )

    return stats
