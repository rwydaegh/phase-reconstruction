import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig  # Import DictConfig

from src.perturbations.archived import apply_archived_complex_strategies
from src.perturbations.basic import apply_basic_perturbation
from src.perturbations.momentum import apply_momentum_perturbation
from src.utils.normalized_correlation import normalized_correlation
from src.utils.normalized_rmse import normalized_rmse
from src.utils.phase_retrieval_utils import (
    apply_magnitude_constraint,
    compute_pseudoinverse,
    create_convergence_plot,
    initialize_field_values,
)

logger = logging.getLogger(__name__)


def holographic_phase_retrieval(  # noqa C901 TODO: Refactor this function for complexity
    cfg: DictConfig,
    channel_matrix: np.ndarray,  # Combined H_train
    measured_magnitude: np.ndarray,  # Combined Mag_train
    train_plane_info: List[
        Tuple[str, int, int]
    ],  # List of (name, start_row, end_row) for training planes
    # Add arguments for test plane data needed for history/animations
    test_planes_data: Optional[
        Dict[str, Dict]
    ] = None,  # Dict mapping test_plane_name to {'H': H_test, 'coords': coords, ...}
    points_perturbed: Optional[
        np.ndarray
    ] = None,  # Needed if calculating H_test inside? No, pass H_test.
    k: Optional[float] = None,  # Needed if calculating H_test inside? No, pass H_test.
    initial_field_values: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Any]:  # Return type adjusted below
    """
    Holographic phase retrieval using Gerchberg-Saxton with enhancements.

    Optimizes cluster coefficients based on the combined magnitude measurements
    from one or more training planes. Optionally tracks history and calculates
    per-plane training errors.

    Args:
        cfg: Hydra configuration object (typically cfg.global_params).
             Contains algorithm settings (iterations, regularization, perturbations, etc.).
        channel_matrix: Combined channel matrix (H_train) for all training planes,
                        shape (N_m_total, N_coeffs).
        measured_magnitude: Combined measured field magnitude for all training planes,
                            shape (N_m_total,).
        train_plane_info: List of tuples `(name, start_row, end_row)` indicating the row
                          slices in `channel_matrix` and `measured_magnitude` for each training plane.
        test_planes_data: Optional dictionary mapping test plane names to their data dicts.
                          Each data dict should contain at least the pre-computed channel matrix
                          `'H_test'` for that plane (calculated using perturbed points).
                          Needed only if `cfg.return_history` is True for animations.
        points_perturbed: Perturbed source points (needed only if H_test is calculated inside, deprecated).
        k: Wavenumber (needed only if H_test is calculated inside, deprecated).
        initial_field_values: Optional initial guess for the complex field values on the
                              combined measurement plane, shape (N_m_total,).
        output_dir: Optional directory to save convergence plots.

    Returns:
        Tuple containing:
            - final_coefficients (np.ndarray): The optimized cluster coefficients.
            - full_history (Optional[List[Dict]]): If cfg.return_history is True, a list where
              each element is a dictionary containing data for that iteration:
              {
                  'iteration': int,
                  'coefficients': np.ndarray,
                  'train_field_segments': Dict[str, np.ndarray], # Complex fields on train planes
                  'test_fields': Dict[str, np.ndarray],       # Complex fields on test planes
                  'overall_train_rmse': float,
                  'per_train_plane_rmse': Dict[str, float]
              }
              Otherwise None.
            - stats (Dict): Dictionary containing final metrics (RMSEs), convergence info,
              perturbation tracking, and potentially aggregated history lists like
              'overall_rmse_history' and 'per_train_plane_rmse_history'.

        Return signature if cfg.return_history:
            (final_coefficients, full_history, stats)
        Return signature if not cfg.return_history:
            (final_coefficients, stats)

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    # Convert to fortran-order arrays for better matrix multiplication performance
    measured_magnitude = np.asfortranarray(measured_magnitude)

    # Compute the pseudoinverse of the channel matrix with regularization
    H_pinv = compute_pseudoinverse(channel_matrix, cfg.regularization, cfg.adaptive_regularization)

    # Initialize with given field values or random phase if not provided
    if initial_field_values is not None:
        logger.info("Using provided initial field values")
        field_values = initial_field_values.copy()
    else:
        field_values = initialize_field_values(measured_magnitude)

    # Initialize RMSE tracking
    # Initialize overall RMSE tracking
    overall_rmse_history = []
    best_coefficients = None
    best_overall_rmse = float("inf")

    # Initialize per-plane RMSE tracking
    per_train_plane_rmse_history: Dict[str, List[float]] = {
        name: [] for name, _, _ in train_plane_info
    }

    # Initialize detailed history list if needed
    full_history: Optional[List[Dict]] = [] if cfg.return_history else None

    # measured_magnitude_norm = np.linalg.norm(measured_magnitude) # No longer needed for error calc

    # Tracking variables for perturbations
    last_significant_improvement = 0
    perturbation_iterations = []
    previous_momentum = None  # For momentum-based perturbation

    # Variables to track post-perturbation progress
    post_perturbation_tracking = []
    current_tracking = None

    # Flag to control constraint skipping after perturbation
    skip_constraint_counter = 0

    for i in range(cfg.gs_iterations):  # Use cfg.gs_iterations
        # 1. Compute cluster coefficients
        cluster_coefficients = H_pinv @ field_values

        # Log state at start of iteration (moved here)
        if i > 0 and cfg.verbose:
            try:
                # Use the newly computed coefficients to estimate error before constraint
                # start_magnitude = np.abs(channel_matrix @ cluster_coefficients)
                # F841: Removed unused variable
                # Calculate start RMSE for debugging if needed (optional)
                # start_rmse = normalized_rmse(start_magnitude, measured_magnitude)
                # logger.debug(
                #     f"Iter {i}: START OF ITERATION (Post-Coeff Calc). "
                #     f"RMSE: {start_rmse:.4e}"
                # )
                pass  # Keep debug structure but remove old error calc for now
                if skip_constraint_counter > 0:
                    logger.info(
                        f"Iter {i}: CONSTRAINT SKIPPING ACTIVE. "
                        f"Remaining: {skip_constraint_counter}"
                    )
            except Exception:
                # This might happen if cluster_coefficients calculation failed, though unlikely
                logger.debug(
                    f"Iter {i}: START OF ITERATION (Post-Coeff Calc). "
                    f"Cannot estimate start RMSE."
                )

        # 2. Forward transform from clusters to field
        simulated_field = channel_matrix @ cluster_coefficients

        # --- Calculate current metrics (RMSE and Correlation) --- # Moved calculation before history storage
        simulated_magnitude = np.abs(simulated_field)
        rmse = normalized_rmse(simulated_magnitude, measured_magnitude)
        corr = normalized_correlation(simulated_magnitude, measured_magnitude)
        overall_rmse_history.append(rmse)

        # Calculate and store per-plane RMSE
        per_plane_rmse_this_iter = {}  # Store temporarily for history entry
        for name, start_row, end_row in train_plane_info:
            plane_sim_mag = simulated_magnitude[start_row:end_row]
            plane_meas_mag = measured_magnitude[start_row:end_row]
            plane_rmse = normalized_rmse(plane_sim_mag, plane_meas_mag)
            per_train_plane_rmse_history[name].append(plane_rmse)
            per_plane_rmse_this_iter[name] = plane_rmse  # Store for history dict

        # --- Store detailed history if requested ---
        if full_history is not None:
            history_entry: Dict[str, Any] = {
                "iteration": i,
                "coefficients": cluster_coefficients.copy(),
                "train_field_segments": {},  # Populated below
                "test_fields": {},
                "overall_train_rmse": rmse,
                "per_train_plane_rmse": {},
            }

            # Store segmented training fields and per-plane RMSEs
            for name, start_row, end_row in train_plane_info:
                history_entry["train_field_segments"][name] = simulated_field[
                    start_row:end_row
                ].copy()
                # Get the already calculated per-plane RMSE for this iteration
                history_entry["per_train_plane_rmse"][name] = per_train_plane_rmse_history[name][-1]

            # Calculate and store reconstructed fields on test planes
            if test_planes_data:
                for test_name, test_data in test_planes_data.items():
                    H_test = test_data.get("H_test")
                    if H_test is not None:
                        # Ensure H_test columns match coefficients length
                        if H_test.shape[1] == cluster_coefficients.shape[0]:
                            recon_field_test = H_test @ cluster_coefficients
                            history_entry["test_fields"][test_name] = recon_field_test.copy()
                        else:
                            logger.warning(
                                f"Iter {i}: Skipping test field calculation for '{test_name}' due to H shape mismatch "
                                f"(H: {H_test.shape[1]}, coeffs: {cluster_coefficients.shape[0]})"
                            )
                    else:
                        logger.warning(
                            f"Iter {i}: Skipping test field calculation for '{test_name}', H_test not found in test_planes_data."
                        )

            full_history.append(history_entry)
        # --- End History Storage ---

        # Print metrics for every iteration
        if cfg.verbose:
            logger.info(f"GS iteration {i}/{cfg.gs_iterations}, RMSE: {rmse:.6f}, Corr: {corr:.6f}")
        else:
            # Minimal progress update, overwriting the line
            progress = (i + 1) / cfg.gs_iterations * 100
            # Use sys.stdout.write for overwriting
            sys.stdout.write(f"\rGS Progress: {progress:.1f}%")
            sys.stdout.flush()  # Ensure it's written immediately

        # Save best coefficients
        # Save best coefficients based on overall RMSE
        if rmse < best_overall_rmse:
            best_overall_rmse = rmse
            best_coefficients = cluster_coefficients.copy()

            # Check if this is a significant improvement (only relevant if perturbations are enabled)
            if cfg.enable_perturbations and i > 0:
                improvement = overall_rmse_history[last_significant_improvement] - rmse
                if improvement > cfg.stagnation_threshold:
                    last_significant_improvement = i
            elif (
                i > 0
            ):  # If perturbations disabled, still update best_coefficients if RMSE improves
                # No need to check threshold, just track the best
                pass  # Logic for updating best_coefficients already exists above

            # The rest of the improvement tracking logic (for stats) should also be under the enable_perturbations check
            if cfg.enable_perturbations and i > 0 and improvement > cfg.stagnation_threshold:
                last_significant_improvement = i

                # If we're tracking post-perturbation progress, update with success
                if (
                    current_tracking is not None
                    and i - current_tracking["start_iter"]
                    <= cfg.stagnation_window  # This access is safe as current_tracking only exists if perturbations enabled
                ):
                    # Use RMSE for tracking improvement
                    current_tracking["final_rmse"] = rmse
                    current_tracking["improvement"] = current_tracking["start_rmse"] - rmse
                    # Improvement means RMSE decreased
                    current_tracking["success"] = current_tracking["improvement"] > 0
                    post_perturbation_tracking.append(current_tracking)
                    current_tracking = None

        # 4. Apply magnitude constraint (unless we're in constraint skipping mode)
        if skip_constraint_counter > 0:
            # Skip the constraint, but still preserve phase information
            # We will only update field_values in specific ways depending on the scenario
            skip_constraint_counter -= 1

            # Just use the current simulated field without constraint
            field_values = simulated_field.copy()

            if cfg.verbose:
                logger.info(
                    f"Iter {i}: SKIPPING CONSTRAINT. Allowing perturbation to propagate further."
                )
        else:
            # Normal constraint application
            field_values = apply_magnitude_constraint(simulated_field, measured_magnitude)

        # 5. Check for convergence
        if rmse < cfg.convergence_threshold:
            if cfg.verbose:
                logger.info(f"Converged after {i+1} iterations with RMSE {rmse:.6f}")
            break

        # 6. Check for stagnation and apply perturbation if needed
        if (
            cfg.enable_perturbations
            and i - last_significant_improvement >= cfg.stagnation_window
            and skip_constraint_counter == 0
        ):
            # We're stagnating - apply perturbation based on selected mode
            if cfg.perturbation_mode == "none":
                # Skip perturbation, but maybe reset stagnation counter anyway?
                # Or let it keep checking? For now, do nothing.
                pass
            elif cfg.perturbation_mode == "basic":
                # Simple random perturbation
                field_values = apply_basic_perturbation(field_values, i, cfg.perturbation_intensity)
                perturbation_iterations.append(i)
                current_tracking = {  # Start tracking post-perturbation progress
                    "start_iter": i,
                    "start_rmse": rmse,
                    "final_rmse": rmse,
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "basic",
                }
                last_significant_improvement = i  # Reset stagnation counter
                skip_constraint_counter = cfg.constraint_skip_iterations  # Set constraint skip
            elif cfg.perturbation_mode == "momentum":
                # Momentum-based perturbation
                field_values, previous_momentum = apply_momentum_perturbation(
                    field_values,
                    rmse,
                    previous_momentum,
                    i,
                    cfg.perturbation_intensity,
                    cfg.momentum_factor,
                )
                perturbation_iterations.append(i)
                current_tracking = {  # Start tracking
                    "start_iter": i,
                    "start_rmse": rmse,
                    "final_rmse": rmse,
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "momentum",
                }
                last_significant_improvement = i
                skip_constraint_counter = cfg.constraint_skip_iterations
            elif cfg.perturbation_mode == "archived":
                # Use the archived complex strategies
                field_values = apply_archived_complex_strategies(
                    field_values,
                    cluster_coefficients,
                    rmse,
                    i,
                    cfg.perturbation_intensity,
                    cfg.temperature,
                )
                perturbation_iterations.append(i)
                current_tracking = {  # Start tracking
                    "start_iter": i,
                    "start_rmse": rmse,
                    "final_rmse": rmse,
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "archived",
                }
                last_significant_improvement = i
                skip_constraint_counter = cfg.constraint_skip_iterations

        # If we're tracking post-perturbation progress and hit the end of tracking window
        is_tracking = current_tracking is not None
        tracking_window_elapsed = (
            i - current_tracking["start_iter"] >= cfg.stagnation_window
            if is_tracking
            else False  # Safe access
        )
        if is_tracking and tracking_window_elapsed:
            current_tracking["final_rmse"] = rmse
            current_tracking["improvement"] = current_tracking["start_rmse"] - rmse
            # Improvement means RMSE decreased
            current_tracking["success"] = current_tracking["improvement"] > 0
            post_perturbation_tracking.append(current_tracking)
            current_tracking = None

    # Print a newline to finalize progress if not verbose
    if not cfg.verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Create convergence plot if output directory is provided
    if not cfg.no_plot and output_dir:
        # Pass overall RMSE history to the convergence plot function
        create_convergence_plot(
            overall_rmse_history, perturbation_iterations, [], cfg.convergence_threshold, output_dir
        )
        # TODO: Consider enhancing create_convergence_plot to optionally show per-plane errors

    # Use the best coefficients found
    if best_overall_rmse < rmse:
        if cfg.verbose:
            logger.info(
                f"Using best coefficients with overall RMSE {best_overall_rmse:.6f} "
                f"instead of final RMSE {rmse:.6f}"
            )
        final_coefficients = best_coefficients
    else:
        final_coefficients = cluster_coefficients

    # Prepare statistics about perturbation effectiveness
    stats = {
        "iterations": i + 1,  # Use i+1 as loop might break early
        "final_overall_rmse": rmse,
        "best_overall_rmse": best_overall_rmse,
        "num_perturbations": len(perturbation_iterations),
        "perturbation_iterations": perturbation_iterations,
        "post_perturbation_tracking": post_perturbation_tracking,
        "overall_rmse_history": overall_rmse_history,
        "per_train_plane_rmse_history": per_train_plane_rmse_history,  # Add per-plane history
    }

    if cfg.verbose:
        logger.info(
            f"GS algorithm completed: {i+1} iterations, "
            f"{len(perturbation_iterations)} perturbations, "
            f"best overall RMSE: {best_overall_rmse:.6f}"
        )

        # Report on perturbation effectiveness
        successful_perturbations = [p for p in post_perturbation_tracking if p["success"]]
        if post_perturbation_tracking:
            success_rate = len(successful_perturbations) / len(post_perturbation_tracking)
            avg_improvement = (
                np.mean([p["improvement"] for p in successful_perturbations])
                if successful_perturbations
                else 0
            )
            logger.info(
                f"Perturbation success rate: {success_rate:.2f}, "
                f"Average improvement: {avg_improvement:.6f}"
            )

    # Adjust return value based on history flag
    if cfg.return_history:
        # Add aggregated histories to stats for convenience (e.g., for convergence plot)
        stats["overall_rmse_history"] = overall_rmse_history
        stats["per_train_plane_rmse_history"] = per_train_plane_rmse_history
        return final_coefficients, full_history, stats
    else:
        # Add aggregated histories to stats even if full_history isn't returned
        stats["overall_rmse_history"] = overall_rmse_history
        stats["per_train_plane_rmse_history"] = per_train_plane_rmse_history

        return final_coefficients, stats
