import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def apply_momentum_perturbation(
    field_values: np.ndarray,
    current_error: float, # Note: current_error is passed but not used in this implementation
    previous_momentum: Optional[np.ndarray],
    iteration: int,
    intensity: float = 0.2,
    momentum_factor: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a momentum-based perturbation to help escape local minima.

    Args:
        field_values: Current field values
        current_error: Current error (currently unused)
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
    random_perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(
        0, 1, field_values.shape
    )
    # Normalize the random perturbation
    random_perturbation_norm = np.linalg.norm(random_perturbation)
    if random_perturbation_norm > 1e-10:
        random_perturbation = random_perturbation / random_perturbation_norm

    # Scale the random perturbation by the perturbation intensity
    random_perturbation = random_perturbation * perturbation_intensity


    # Apply momentum if available
    if previous_momentum is not None:
        # Combine random perturbation with previous momentum
        combined_perturbation = random_perturbation + momentum_factor * previous_momentum
        # Normalize to desired intensity
        combined_perturbation_norm = np.linalg.norm(combined_perturbation)
        if combined_perturbation_norm > 1e-10:
            combined_perturbation = (
                combined_perturbation
                * field_norm
                * perturbation_intensity
                / combined_perturbation_norm
            )
        else:
             combined_perturbation = np.zeros_like(field_values) # Avoid NaN if norm is zero

        new_momentum = combined_perturbation
    else:
        combined_perturbation = random_perturbation
        new_momentum = random_perturbation

    # Apply perturbation
    perturbed_values = field_values + combined_perturbation

    # Log perturbation details
    perturbation_magnitude = np.linalg.norm(combined_perturbation)
    if field_norm > 1e-10:
        relative_perturbation = perturbation_magnitude / field_norm
    else:
        relative_perturbation = 0.0

    logger.info(
        f"Iter {iteration}: Applied MOMENTUM perturbation. "
        f"Relative magnitude: {relative_perturbation*100:.2f}%"
    )

    return perturbed_values, new_momentum
