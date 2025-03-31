import logging
import numpy as np

logger = logging.getLogger(__name__)


def apply_archived_complex_strategies(
    field_values: np.ndarray,
    cluster_coefficients: np.ndarray,
    current_error: float,
    iteration: int,
    intensity: float = 0.3,
    temperature: float = 5.0,
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
        # Assuming archived_strategies is in src.utils
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
        # Note: The original function returned skip_constraint, which is ignored here.
        #       The modular approach might need refinement if constraint skipping
        #       needs to be communicated back to the main algorithm.
        perturbed_values, _, _, _, _, _ = apply_perturbation_strategy(
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
            verbose=True, # Keep verbose logging from the archived function
        )

        logger.info(f"Iter {iteration}: Applied ARCHIVED STRATEGY {strategy}")
        return perturbed_values

    except ImportError:
        # Fallback if the archived strategies module is not available
        logger.warning("Archived strategies module not found. Using basic perturbation instead.")
        # Re-implement basic perturbation as fallback
        field_norm = np.linalg.norm(field_values)
        perturbation_intensity = intensity

        # Scale perturbation relative to field norm
        perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(
            0, 1, field_values.shape
        )
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > 1e-10:
             perturbation = (
                 perturbation * field_norm * perturbation_intensity / perturbation_norm
             )
        else:
             perturbation = np.zeros_like(field_values)

        # Apply perturbation
        perturbed_values = field_values + perturbation

        # Log perturbation details
        perturbation_magnitude = np.linalg.norm(perturbation)
        if field_norm > 1e-10:
            relative_perturbation = perturbation_magnitude / field_norm
        else:
            relative_perturbation = 0.0

        logger.info(
            f"Iter {iteration}: Applied basic perturbation as fallback. "
            f"Relative magnitude: {relative_perturbation*100:.2f}%"
        )
        return perturbed_values