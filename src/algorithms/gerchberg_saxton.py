import logging
import sys
from typing import Optional, Tuple

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


def holographic_phase_retrieval(
    cfg: DictConfig,  # Use cfg object
    channel_matrix: np.ndarray,
    measured_magnitude: np.ndarray,
    initial_field_values: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,  # Add output directory parameter
):
    """
    Basic holographic phase retrieval algorithm based on Gerchberg-Saxton with
    optional simple perturbation strategies.

    Args:
        cfg: Configuration object containing parameters like:
            gs_iterations: Maximum number of iterations
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
            constraint_skip_iterations: How many iterations to skip the constraint
                                        after perturbation
            momentum_factor: Weight factor for momentum-based perturbation
            temperature: Temperature parameter for archived strategies
            no_plot: Whether to create convergence plots
        channel_matrix: Matrix H relating clusters to measurement points.
        measured_magnitude: Measured field magnitude.
        initial_field_values: Optional custom field initialization.
        output_dir: Optional directory to save plots like the convergence plot.

    Returns:
        Cluster coefficients and optionally history, along with stats.
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
    rmse_history = []
    best_coefficients = None
    best_rmse = float("inf")

    # Initialize history arrays if needed
    coefficient_history = [] if cfg.return_history else None
    field_history = [] if cfg.return_history else None

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

        if cfg.return_history:
            coefficient_history.append(cluster_coefficients.copy())
            field_history.append(simulated_field.copy())

        # 3. Calculate current metrics (RMSE and Correlation)
        simulated_magnitude = np.abs(simulated_field)
        rmse = normalized_rmse(simulated_magnitude, measured_magnitude)
        corr = normalized_correlation(simulated_magnitude, measured_magnitude)
        rmse_history.append(rmse)

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
        if rmse < best_rmse:
            best_rmse = rmse
            best_coefficients = cluster_coefficients.copy()

            # Check if this is a significant improvement
            improvement = rmse_history[last_significant_improvement] - rmse
            if i > 0 and improvement > cfg.stagnation_threshold:
                last_significant_improvement = i

                # If we're tracking post-perturbation progress, update with success
                if (
                    current_tracking is not None
                    and i - current_tracking["start_iter"] <= cfg.stagnation_window
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
                # Skip perturbation
                pass
            elif cfg.perturbation_mode == "basic":
                # Simple random perturbation
                field_values = apply_basic_perturbation(field_values, i, cfg.perturbation_intensity)
                perturbation_iterations.append(i)

                # Start tracking post-perturbation progress
                current_tracking = {
                    "start_iter": i,
                    "start_rmse": rmse,  # Track starting RMSE
                    # Initialize with current rmse
                    # F821: error undefined, use rmse
                    "final_rmse": rmse,
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "basic",
                }

                # Reset stagnation counter
                last_significant_improvement = i

                # Set the constraint skip counter
                skip_constraint_counter = cfg.constraint_skip_iterations

            elif cfg.perturbation_mode == "momentum":
                # Momentum-based perturbation
                field_values, previous_momentum = apply_momentum_perturbation(
                    field_values,
                    rmse,  # Pass current rmse as the error metric # F821: error undefined, use rmse
                    previous_momentum,
                    i,
                    cfg.perturbation_intensity,
                    cfg.momentum_factor,
                )
                perturbation_iterations.append(i)

                # Start tracking post-perturbation progress
                current_tracking = {
                    "start_iter": i,
                    "start_rmse": rmse,  # Track starting RMSE
                    # Initialize with current rmse
                    # F821: error undefined, use rmse
                    "final_rmse": rmse,
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "momentum",
                }

                # Reset stagnation counter
                last_significant_improvement = i

                # Set the constraint skip counter
                skip_constraint_counter = cfg.constraint_skip_iterations

            elif cfg.perturbation_mode == "archived":
                # Use the archived complex strategies (via separate function)
                field_values = apply_archived_complex_strategies(
                    field_values,
                    cluster_coefficients,
                    rmse,  # Pass current rmse as the error metric # F821: error undefined, use rmse
                    i,
                    cfg.perturbation_intensity,
                    cfg.temperature,
                )
                perturbation_iterations.append(i)

                # Start tracking post-perturbation progress
                current_tracking = {
                    "start_iter": i,
                    "start_rmse": rmse,  # Track starting RMSE
                    # Initialize with current rmse
                    # F821: error undefined, use rmse
                    "final_rmse": rmse,
                    "improvement": 0.0,
                    "success": False,
                    "perturbation_type": "archived",
                }

                # Reset stagnation counter
                last_significant_improvement = i

                # Set the constraint skip counter
                skip_constraint_counter = cfg.constraint_skip_iterations

        # If we're tracking post-perturbation progress and hit the end of tracking window
        is_tracking = current_tracking is not None
        tracking_window_elapsed = (
            i - current_tracking["start_iter"] >= cfg.stagnation_window if is_tracking else False
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
        create_convergence_plot(
            rmse_history, perturbation_iterations, [], cfg.convergence_threshold, output_dir
        )

    # Use the best coefficients found
    if best_rmse < rmse:
        if cfg.verbose:
            logger.info(
                f"Using best coefficients with RMSE {best_rmse:.6f} "
                f"instead of final RMSE {rmse:.6f}"
            )
        final_coefficients = best_coefficients
    else:
        final_coefficients = cluster_coefficients

    # Prepare statistics about perturbation effectiveness
    stats = {
        "iterations": i + 1,
        "final_rmse": rmse,  # Use RMSE
        "best_rmse": best_rmse,  # Use RMSE
        "num_perturbations": len(perturbation_iterations),
        "perturbation_iterations": perturbation_iterations,
        # Note: tracking dict now uses 'start_rmse', 'final_rmse'
        "post_perturbation_tracking": post_perturbation_tracking,
        "rmse_history": rmse_history,  # Use RMSE
    }

    if cfg.verbose:
        logger.info(
            f"GS algorithm completed: {i+1} iterations, "
            f"{len(perturbation_iterations)} perturbations, "
            f"best RMSE: {best_rmse:.6f}"
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

    if cfg.return_history:
        return final_coefficients, np.array(coefficient_history), np.array(field_history), stats
    else:
        return final_coefficients, stats
