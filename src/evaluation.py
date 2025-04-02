# src/evaluation.py
import logging

import numpy as np
from omegaconf import DictConfig

from src.create_channel_matrix import create_channel_matrix
from src.utils.normalized_correlation import normalized_correlation
from src.utils.normalized_rmse import normalized_rmse

logger = logging.getLogger(__name__)


def evaluate_on_test_planes(
    test_planes: list,
    final_coefficients: np.ndarray,
    points_true: np.ndarray,
    tangents1_true: np.ndarray | None,
    tangents2_true: np.ndarray | None,
    points_perturbed: np.ndarray,
    tangents1_perturbed: np.ndarray | None,
    tangents2_perturbed: np.ndarray | None,
    original_currents: np.ndarray,  # Ground truth currents used for simulated planes
    config: DictConfig,  # Main config or relevant sub-config (e.g., global_params)
) -> dict:
    """
    Evaluates the reconstructed field against ground truth on specified test planes.

    Args:
        test_planes: List of processed test plane data dictionaries.
        final_coefficients: The cluster coefficients obtained from HPR.
        points_true: Ground truth source point coordinates.
        tangents1_true: Ground truth source tangent 1 vectors (or None).
        tangents2_true: Ground truth source tangent 2 vectors (or None).
        points_perturbed: Perturbed source point coordinates used for reconstruction H.
        tangents1_perturbed: Perturbed source tangent 1 vectors (or None).
        tangents2_perturbed: Perturbed source tangent 2 vectors (or None).
        original_currents: The ground truth currents corresponding to points_true.
        config: The main configuration object or global_params sub-object.

    Returns:
        Dictionary containing evaluation results per test plane.
        Example: {'plane_name': {'rmse': 0.1, 'correlation': 0.9, ...}, ...}
    """
    evaluation_results = {}
    global_params = config.get("global_params", config)  # Handle if full cfg or sub-cfg is passed
    k = 2 * np.pi / global_params.wavelength

    if not test_planes:
        logger.info("No test planes provided for evaluation.")
        return evaluation_results

    logger.info(f"Starting evaluation on {len(test_planes)} test plane(s)...")

    # Check coefficient shape compatibility once
    expected_coeffs_len = points_perturbed.shape[0] * (2 if global_params.use_vector_model else 1)
    if final_coefficients.shape[0] != expected_coeffs_len:
        raise ValueError(
            f"Shape mismatch for evaluation: final_coefficients length ({final_coefficients.shape[0]}) "
            f"!= expected from points_perturbed ({expected_coeffs_len})"
        )

    for i, plane in enumerate(test_planes):
        plane_name = plane.get("name", f"test_plane_{i}")
        logger.info(f"  Evaluating plane: {plane_name}")
        measurement_coords = plane["coordinates"]
        ground_truth_magnitude = None

        # 1. Get Ground Truth Magnitude for this plane
        if plane["is_real_plane"]:
            logger.info("    Using measured magnitude from real plane as ground truth.")
            ground_truth_magnitude = plane.get("measured_magnitude")
            if ground_truth_magnitude is None:
                logger.error(
                    f"    Cannot evaluate real plane '{plane_name}': 'measured_magnitude' missing."
                )
                continue
        else:
            # Calculate true field magnitude for simulated test plane
            logger.info("    Calculating true magnitude for simulated test plane...")
            try:
                H_true_sim_plane = create_channel_matrix(
                    points=points_true,  # Use true geometry
                    measurement_plane=measurement_coords,
                    k=k,
                    use_vector_model=global_params.use_vector_model,
                    tangents1=tangents1_true if global_params.use_vector_model else None,
                    tangents2=tangents2_true if global_params.use_vector_model else None,
                    measurement_direction=np.array(global_params.measurement_direction)
                    if global_params.use_vector_model
                    else None,
                )
                # Ensure current shape matches H_true_sim_plane columns
                expected_true_coeffs_len = points_true.shape[0] * (
                    2 if global_params.use_vector_model else 1
                )
                if H_true_sim_plane.shape[1] != expected_true_coeffs_len:
                    raise ValueError(
                        f"H_true columns ({H_true_sim_plane.shape[1]}) != expected from points_true ({expected_true_coeffs_len})"
                    )
                if original_currents.shape[0] != expected_true_coeffs_len:
                    raise ValueError(
                        f"original_currents shape ({original_currents.shape[0]}) != expected from points_true ({expected_true_coeffs_len})"
                    )

                true_field_sim_plane = H_true_sim_plane @ original_currents
                ground_truth_magnitude = np.abs(true_field_sim_plane)
                logger.info(
                    f"    Calculated ground truth magnitude shape: {ground_truth_magnitude.shape}"
                )
            except Exception as e:
                logger.error(
                    f"    Failed to calculate ground truth for simulated plane '{plane_name}': {e}",
                    exc_info=True,
                )
                continue

        # 2. Create H matrix using perturbed geometry for this test plane
        try:
            H_test_plane = create_channel_matrix(
                points=points_perturbed,
                measurement_plane=measurement_coords,
                k=k,
                use_vector_model=global_params.use_vector_model,
                tangents1=tangents1_perturbed if global_params.use_vector_model else None,
                tangents2=tangents2_perturbed if global_params.use_vector_model else None,
                measurement_direction=np.array(global_params.measurement_direction)
                if global_params.use_vector_model
                else None,
            )
            # Check H shape against coefficients
            if H_test_plane.shape[1] != final_coefficients.shape[0]:
                raise ValueError(
                    f"H_test_plane columns ({H_test_plane.shape[1]}) != final_coefficients length ({final_coefficients.shape[0]})"
                )

        except Exception as e:
            logger.error(
                f"    Failed to create H matrix for test plane '{plane_name}': {e}", exc_info=True
            )
            continue

        # 3. Calculate Reconstructed Field on this plane
        reconstructed_field_test = H_test_plane @ final_coefficients
        reconstructed_magnitude_test = np.abs(reconstructed_field_test)

        # Ensure shapes match for comparison
        if ground_truth_magnitude.shape != reconstructed_magnitude_test.shape:
            logger.error(
                f"    Shape mismatch between ground truth ({ground_truth_magnitude.shape}) "
                f"and reconstructed magnitude ({reconstructed_magnitude_test.shape}) for plane '{plane_name}'. Skipping metrics."
            )
            continue

        # 4. Calculate Metrics
        try:
            rmse = normalized_rmse(ground_truth_magnitude, reconstructed_magnitude_test)
            corr = normalized_correlation(ground_truth_magnitude, reconstructed_magnitude_test)
            logger.info(f"    Metrics - RMSE: {rmse:.6f}, Correlation: {corr:.6f}")

            # Store results
            evaluation_results[plane_name] = {
                "rmse": rmse,
                "correlation": corr,
                "ground_truth_magnitude": ground_truth_magnitude,  # Optional: store fields for visualization
                "reconstructed_magnitude": reconstructed_magnitude_test,  # Optional
                "reconstructed_field": reconstructed_field_test,  # Optional
                "original_data_shape": plane.get(
                    "original_data_shape"
                ),  # Store shape for reshaping later
            }
        except Exception as e:
            logger.error(
                f"    Failed to calculate metrics for plane '{plane_name}': {e}", exc_info=True
            )

    logger.info("Evaluation complete.")
    return evaluation_results


# TODO: Add unit tests for this module
