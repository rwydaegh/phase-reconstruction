import logging

import numpy as np

logger = logging.getLogger(__name__)


def apply_basic_perturbation(
    field_values: np.ndarray, iteration: int, intensity: float = 0.1
) -> np.ndarray:
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
    perturbation = np.random.normal(0, 1, field_values.shape) + 1j * np.random.normal(
        0, 1, field_values.shape
    )
    # Avoid division by zero if perturbation happens to be zero
    perturbation_norm = np.linalg.norm(perturbation)
    if perturbation_norm > 1e-10:
        perturbation = perturbation * field_norm * perturbation_intensity / perturbation_norm
    else:
        perturbation = np.zeros_like(field_values) # No perturbation if norm is zero

    # Apply perturbation
    perturbed_values = field_values + perturbation

    # Log perturbation details
    perturbation_magnitude = np.linalg.norm(perturbation)
    if field_norm > 1e-10:
        relative_perturbation = perturbation_magnitude / field_norm
    else:
        relative_perturbation = 0.0 # Avoid division by zero if field_norm is zero

    logger.info(
        f"Iter {iteration}: Applied BASIC perturbation. "
        f"Relative magnitude: {relative_perturbation*100:.2f}%"
    )

    return perturbed_values
