"""
This module contains archived implementations of complex perturbation strategies
that were previously used in the holographic phase retrieval algorithm.

These strategies have been archived as they were causing excessive spiking in the
error convergence. They are preserved here for reference and potential future use
with more controlled parameters.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.phase_retrieval_utils import initialize_field_values

# Configure logging
logger = logging.getLogger(__name__)


def apply_perturbation_strategy(
    field_values: np.ndarray,
    cluster_coefficients: np.ndarray,
    best_coefficients: np.ndarray,
    best_error: float,
    memory_solutions: List[np.ndarray],
    memory_errors: List[float],
    strategy: int,
    perturbation_intensity: float,
    temperature: float,
    iteration: int,
    previous_field_values: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, bool, np.ndarray, float, List[np.ndarray], List[float]]:
    """
    ARCHIVED FUNCTION: Apply the selected perturbation strategy.
    This function contains the complex perturbation strategies from the original implementation.

    Args:
        field_values: Current field values
        cluster_coefficients: Current cluster coefficients
        best_coefficients: Best coefficients found so far
        best_error: Best error found so far
        memory_solutions: List of good solutions to remember
        memory_errors: List of errors for the good solutions
        strategy: Which perturbation strategy to use
        perturbation_intensity: Intensity of the perturbation
        temperature: Current temperature for simulated annealing
        iteration: Current iteration number
        previous_field_values: Field values from previous iteration
        verbose: Whether to print verbose information

    Returns:
        Perturbed field values, skip_magnitude_constraint flag,
        updated best coefficients, updated best error,
        updated memory solutions, updated memory errors
    """
    skip_magnitude_constraint = False
    # field_norm = np.linalg.norm(field_values) # Unused

    # Strategy 0: MODIFIED GS APPROACH - Change the fundamental algorithm
    if strategy == 0:
        # Generate completely new random coefficients
        random_coefficients = np.random.normal(
            0, 1, cluster_coefficients.shape
        ) + 1j * np.random.normal(0, 1, cluster_coefficients.shape)

        # Scale to have similar magnitude as original coefficients
        scale_factor = np.linalg.norm(cluster_coefficients) / np.linalg.norm(random_coefficients)
        random_coefficients *= scale_factor * (1.0 + np.random.random())

        # Replace the current coefficients and save as best
        best_coefficients = random_coefficients.copy()

        # Reset the best error to force algorithm to accept new direction
        best_error = best_error * 0.9

        # Reset the memory of good solutions
        memory_solutions = []
        memory_errors = []

        if verbose:
            logger.info(f"🔄 ALGORITHM RESET at iteration {iteration}")

    # Strategy 1: CATASTROPHIC phase inversion plus magnitude explosion
    elif strategy == 1:
        # Invert phases (add π) and multiply magnitudes by a large random factor
        phase = np.angle(field_values)
        magnitude = np.abs(field_values)
        inverted_phase = phase + np.pi  # 180-degree phase shift
        exploded_magnitude = magnitude * np.random.uniform(15, 40, magnitude.shape)
        field_values = exploded_magnitude * np.exp(1j * inverted_phase)

        # Skip applying magnitude constraint for this iteration
        skip_magnitude_constraint = True

        if verbose:
            logger.info(
                f"🔄 CATASTROPHIC phase inversion + magnitude explosion at iteration {iteration}"
            )

    # Strategy 2: CHAOTIC basin hopping with massive jump
    elif strategy == 2:
        # Create a completely random direction with no relation to current field
        random_direction = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(
            0, 1, field_values.shape
        )
        normalized_direction = random_direction / np.linalg.norm(random_direction)

        # Make an ENORMOUS jump in this direction
        jump_size = perturbation_intensity * np.random.uniform(200, 600)
        field_values = jump_size * normalized_direction * np.mean(np.abs(field_values))

        # Skip applying magnitude constraint
        skip_magnitude_constraint = True

        if verbose:
            logger.info(f"🌪️ CHAOTIC basin jump with size {jump_size:.4f} at iteration {iteration}")

    # Strategy 3: Fourier domain scrambling (extreme frequency perturbation)
    elif strategy == 3:
        # Reshape to 2D for FFT (assuming square measurement grid)
        grid_size = int(np.sqrt(field_values.shape[0]))
        reshaped = field_values.reshape(grid_size, grid_size)

        # Apply FFT
        freq_domain = np.fft.fft2(reshaped)

        # Apply STRONG perturbation in frequency domain - completely scramble phases
        freq_domain *= np.exp(1j * np.random.uniform(0, 2 * np.pi, freq_domain.shape))

        # Perturb magnitudes extremely aggressively (100-200%)
        mag_perturb = 1 + perturbation_intensity * 70 * np.random.normal(0, 1, freq_domain.shape)
        freq_domain *= mag_perturb

        # Back to spatial domain
        perturbed = np.fft.ifft2(freq_domain)

        # Extract phase and normalize magnitude
        magnitude = np.abs(field_values)
        perturbed_phase = np.angle(perturbed)
        field_values = magnitude.reshape(-1) * np.exp(1j * perturbed_phase.reshape(-1))

        if verbose:
            logger.info(f"Applied Fourier domain scrambling at iteration {iteration}")

    # Strategy 4: Simulated annealing with temperature-based jumps
    elif strategy == 4:
        # Calculate current "temperature" - higher at early iterations
        current_temp = temperature * 100

        # Temperature-dependent perturbation (stronger when temp is higher)
        phase = np.angle(field_values)
        magnitude = np.abs(field_values)
        phase_jump = current_temp * np.random.normal(0, np.pi / 2, phase.shape)
        mag_factor = 1 + current_temp * np.random.normal(0, 0.5, magnitude.shape)

        # Apply both phase and magnitude perturbation
        new_phase = phase + phase_jump
        new_magnitude = magnitude * np.abs(mag_factor)
        field_values = new_magnitude * np.exp(1j * new_phase)

        if verbose:
            logger.info(
                f"Applied simulated annealing jump with temperature {current_temp:.4f} "
                f"at iteration {iteration}"
            )

    # Strategy 5: Anti-correlation perturbation (move opposite to current gradient direction)
    elif strategy == 5:
        # Use the current gradient direction (change in field values)
        if previous_field_values is not None:
            # Approximate gradient using field value difference
            gradient = field_values - previous_field_values

            # Move in the opposite direction to escape local minimum
            anti_gradient = -gradient / (np.linalg.norm(gradient) + 1e-10)

            # Apply large step in anti-gradient direction
            jump_size = perturbation_intensity * np.random.uniform(20, 50)
            field_values = field_values + jump_size * anti_gradient * np.mean(np.abs(field_values))
        else:
            # Fallback if we don't have previous values
            magnitude = np.abs(field_values)
            phase = np.angle(field_values)
            field_values = magnitude * np.exp(
                1j * (phase + np.random.normal(0, np.pi, phase.shape))
            )

        if verbose:
            logger.info(f"Applied anti-correlation perturbation at iteration {iteration}")

    # Strategy 6: Mode-mixing (swap different spatial frequency components)
    elif strategy == 6:
        # Reshape to 2D for FFT
        grid_size = int(np.sqrt(field_values.shape[0]))
        reshaped = field_values.reshape(grid_size, grid_size)

        # FFT to get frequency domain representation
        freq_domain = np.fft.fft2(reshaped)

        # Create a permutation of the frequency domain
        flat_freq = freq_domain.flatten()
        perm_indices = np.random.permutation(len(flat_freq))

        # Permute a larger subset of the frequencies
        mix_ratio = np.random.uniform(0.7, 0.95)
        num_to_mix = int(mix_ratio * len(flat_freq))
        indices_to_mix = np.random.choice(len(flat_freq), num_to_mix, replace=False)

        # Create mixed frequency domain
        mixed_freq = flat_freq.copy()
        mixed_freq[indices_to_mix] = flat_freq[perm_indices[indices_to_mix]]

        # Reshape back and IFFT
        mixed_freq = mixed_freq.reshape(grid_size, grid_size)
        mixed_field = np.fft.ifft2(mixed_freq)

        # Keep original magnitude but use new phases
        magnitude = np.abs(field_values)
        mixed_phase = np.angle(mixed_field)
        field_values = magnitude.reshape(-1) * np.exp(1j * mixed_phase.reshape(-1))

        if verbose:
            logger.info(f"Applied mode-mixing with {mix_ratio:.2f} ratio at iteration {iteration}")

    # Update memory of good solutions
    memory_solutions, memory_errors = update_memory_solutions(
        cluster_coefficients, best_error, memory_solutions, memory_errors, 10
    )

    return (
        field_values,
        skip_magnitude_constraint,
        best_coefficients,
        best_error,
        memory_solutions,
        memory_errors,
    )


def update_memory_solutions(
    coefficients: np.ndarray,
    error: float,
    memory_solutions: List[np.ndarray],
    memory_errors: List[float],
    max_memory: int = 10,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    ARCHIVED FUNCTION: Update the memory of good solutions.

    Args:
        coefficients: Current cluster coefficients
        error: Current error
        memory_solutions: List of good solutions
        memory_errors: List of errors for good solutions
        max_memory: Maximum number of solutions to remember

    Returns:
        Updated memory solutions and errors
    """
    # Make copies to avoid reference issues
    solutions = [s.copy() for s in memory_solutions]
    errors = memory_errors.copy()

    # Check if current solution is worth remembering
    should_add = False

    # If we have less than max solutions, always add
    if len(solutions) < max_memory:
        should_add = True
    # Otherwise, only add if better than worst solution
    elif error < max(errors):
        should_add = True
        # Remove worst solution
        worst_idx = np.argmax(errors)
        solutions.pop(worst_idx)
        errors.pop(worst_idx)

    if should_add:
        solutions.append(coefficients.copy())
        errors.append(error)

    return solutions, errors


def perform_intelligent_restart(
    measured_magnitude: np.ndarray,
    channel_matrix: np.ndarray,
    memory_solutions: List[np.ndarray],
    memory_errors: List[float],
    temperature: float,
    temperature_initial: float,
    memory_contribution: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    ARCHIVED FUNCTION: Perform an intelligent restart using memory of good solutions.

    Args:
        measured_magnitude: Measured field magnitude
        channel_matrix: Matrix H relating clusters to measurement points
        memory_solutions: List of good solutions to remember
        memory_errors: List of errors for the good solutions
        temperature: Current temperature for simulated annealing
        temperature_initial: Initial temperature for simulated annealing
        memory_contribution: Weight of ensemble solutions in restart
        verbose: Whether to print verbose information

    Returns:
        New field values
    """
    # Generate completely new random field values
    new_field_values = initialize_field_values(measured_magnitude)

    # If we have memory solutions, incorporate them
    if memory_solutions:
        # Weight solutions by their error (lower error = higher weight)
        weights = 1.0 / (np.array(memory_errors) + 1e-10)
        weights = weights / np.sum(weights)

        # Create weighted combination of memory solutions
        ensemble_coefficients = np.zeros_like(memory_solutions[0])
        for i, sol in enumerate(memory_solutions):
            ensemble_coefficients += weights[i] * sol

        # Convert back to field values
        memory_field = channel_matrix @ ensemble_coefficients

        # Mix random and memory-based field values
        # Use temperature to control randomness: higher temp = more random
        t_factor = temperature / temperature_initial
        random_ratio = t_factor * 0.8 + 0.1  # Between 0.1 and 0.9

        # Adjust mix based on memory contribution parameter
        memory_ratio = (1 - random_ratio) * memory_contribution
        random_ratio = 1 - memory_ratio

        # Mix field values
        mixed_field = random_ratio * new_field_values + memory_ratio * memory_field

        # Apply magnitude constraint to ensure valid field
        magnitudes = np.abs(mixed_field)
        mixed_field = measured_magnitude * mixed_field / magnitudes

        if verbose:
            logger.info(
                f"Intelligent restart: {random_ratio*100:.1f}% random, "
                f"{memory_ratio*100:.1f}% memory-based"
            )
            logger.info(
                f"Using {len(memory_solutions)} memory solutions "
                f"with temperature factor {t_factor:.4f}"
            )

        return mixed_field
    else:
        # Just return the random initialization if no memory solutions
        if verbose:
            logger.info("Intelligent restart: 100% random (no memory solutions)")
        return new_field_values


def compile_statistics(
    iterations: int,
    final_error: float,
    best_error: float,
    perturbation_iterations: List[int],
    restart_iterations: List[int],
    temperature: float,
    strategy_success_counts: np.ndarray,
    strategy_attempt_counts: np.ndarray,
    errors: List[float],
) -> Dict[str, Any]:
    """
    ARCHIVED FUNCTION: Compile statistics about the algorithm's performance.

    Args:
        iterations: Number of iterations completed
        final_error: Final error achieved
        best_error: Best error found
        perturbation_iterations: List of iterations where perturbations were applied
        restart_iterations: List of iterations where restarts occurred
        temperature: Final temperature for simulated annealing
        strategy_success_counts: Count of successful applications of each strategy
        strategy_attempt_counts: Count of attempts of each strategy
        errors: List of errors at each iteration

    Returns:
        Dictionary of statistics
    """
    # Calculate strategy success rates
    strategy_success_rates = strategy_success_counts / strategy_attempt_counts

    # Compile statistics
    stats = {
        "iterations": iterations,
        "final_error": final_error,
        "best_error": best_error,
        "num_perturbations": len(perturbation_iterations),
        "num_restarts": len(restart_iterations),
        "perturbation_iterations": perturbation_iterations,
        "restart_iterations": restart_iterations,
        "final_temperature": temperature,
        "strategy_success_rates": strategy_success_rates.tolist(),
        "strategy_attempts": strategy_attempt_counts.tolist(),
        "strategy_successes": strategy_success_counts.tolist(),
        "errors": errors,
    }

    return stats
