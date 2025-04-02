# multi_plane_reconstruction.py
import logging
import os
import pickle
import random
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Module imports
from src import evaluation, plane_processing
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix
from src.source_handling import get_source_pointcloud
from src.visualization import (
    comparison_plots,
    field_plots,
    history_plots,
    multi_plane_plots,  # Import the new module
)

# Setup logger first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports for legacy script delegation
try:
    from measured_data_reconstruction import main as run_measured_legacy
except ImportError:
    run_measured_legacy = None
    logger.debug("measured_data_reconstruction.py not found, legacy measured mode disabled.")
try:
    from simulated_data_reconstruction import main as run_simulated_legacy
except ImportError:
    run_simulated_legacy = None
    logger.debug("simulated_data_reconstruction.py not found, legacy simulated mode disabled.")

# Removed placeholder run_legacy_workflow function
# Removed stray pass


def run_multi_plane_workflow(cfg: DictConfig, base_dir: str | None = None):
    """
    Orchestrates the new multi-plane reconstruction workflow.

    Args:
        cfg: The DictConfig object for the run.
        base_dir: Optional base directory path. If None, determined using hydra.utils.get_original_cwd().
                  This is useful for testing scenarios where Hydra's full context isn't initialized.
    """
    logger.info("Starting Multi-Plane Reconstruction Workflow...")
    output_dir = os.getcwd()  # Hydra sets cwd to the output directory for the run
    if base_dir is None:
        base_dir = hydra.utils.get_original_cwd()  # Get the original directory if not provided
    logger.info(f"Output directory: {output_dir}")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Plots will be saved to: {plots_dir}")

    # --- 1. Global Setup ---
    seed = cfg.global_params.random_seed
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using random seed: {seed}")

    k = 2 * np.pi / cfg.global_params.wavelength
    logger.info(f"Wave number k: {k:.4f} (wavelength: {cfg.global_params.wavelength:.4e} m)")

    # --- 2. Load/Generate Source Point Cloud ---
    logger.info("Processing Source Point Cloud...")
    try:
        (
            points_true,
            tangents1_true,
            tangents2_true,
            points_perturbed,
            tangents1_perturbed,
            tangents2_perturbed,
            original_currents,
        ) = get_source_pointcloud(
            cfg=cfg.source_pointcloud,  # Pass the source_pointcloud sub-config
            global_cfg=cfg.global_params,  # Pass the global_params sub-config
            base_dir=base_dir,
        )
        logger.info(
            f"Source points processed. True shape: {points_true.shape}, Perturbed shape: {points_perturbed.shape}, Currents shape: {original_currents.shape}"
        )
        # Verify current shape matches expectation from true points and model type
        num_coeffs_true = points_true.shape[0] * (2 if cfg.global_params.use_vector_model else 1)
        if original_currents.shape[0] != num_coeffs_true:
            logger.warning(
                f"Mismatch between generated currents shape ({original_currents.shape[0]}) and expected shape ({num_coeffs_true}). Check current generation logic."
            )
            # Decide how to handle: error out or try to proceed? For now, log warning.

    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Failed to process source point cloud: {e}", exc_info=True)
        return  # Cannot proceed without source points

    # --- 3. Process Measurement Plane Definitions ---
    logger.info("Processing Measurement Plane Definitions...")
    all_processed_planes = plane_processing.process_plane_definitions(
        plane_configs=cfg.measurement_planes, base_dir=base_dir
    )
    if not all_processed_planes:
        logger.warning("Plane processing resulted in no valid planes.")
        # Decide whether to exit or continue if possible (e.g., if only testing was intended but failed)
        # For now, let the check for train_planes handle the critical path.

    # --- 4. Separate Train/Test Planes ---
    train_planes = [p for p in all_processed_planes if p.get("use_train", False)]
    test_planes = [p for p in all_processed_planes if p.get("use_test", False)]
    logger.info(f"Found {len(train_planes)} training planes and {len(test_planes)} testing planes.")

    if not train_planes:
        logger.error("No training planes defined ('use_train: true'). Cannot proceed.")

    # --- Calculate and Store Ground Truth Magnitudes for Simulated Planes ---
    # This ensures the correct GT is available for animations.
    logger.info("Calculating ground truth magnitudes for simulated planes...")
    for plane in all_processed_planes:
        # Calculate GT for simulated planes
        if not plane.get("is_real_plane", False):
            plane_name = plane.get("name", "Unnamed Sim Plane")
            logger.debug(
                f"--> Starting GT calculation for simulated plane: {plane_name}"
            )  # DEBUG ADDED
            logger.info(f"  Calculating GT for: {plane_name}")
            try:
                H_true_sim = create_channel_matrix(
                    points=points_true,  # Use true geometry
                    measurement_plane=plane["coordinates"],
                    k=k,
                    use_vector_model=cfg.global_params.use_vector_model,
                    tangents1=tangents1_true if cfg.global_params.use_vector_model else None,
                    tangents2=tangents2_true if cfg.global_params.use_vector_model else None,
                    measurement_direction=np.array(cfg.global_params.measurement_direction)
                    if cfg.global_params.use_vector_model
                    else None,
                )
                # Ensure current shape matches H_true_sim columns
                if H_true_sim.shape[1] != original_currents.shape[0]:
                    raise ValueError(
                        f"Shape mismatch for GT calculation on plane {plane_name}: H cols ({H_true_sim.shape[1]}) != currents ({original_currents.shape[0]})"
                    )

                true_field_sim = H_true_sim @ original_currents
                plane["ground_truth_magnitude"] = np.abs(true_field_sim)  # Store GT mag in the dict
                logger.debug(
                    f"<-- Finished GT calculation for simulated plane: {plane_name}"
                )  # DEBUG ADDED
                logger.info(
                    f"    Stored GT magnitude shape: {plane['ground_truth_magnitude'].shape}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to calculate ground truth magnitude for simulated plane {plane_name}: {e}",
                    exc_info=True,
                )
                # Plane will proceed without GT magnitude, animation might fail later if needed
    logger.info("...Finished calculating ground truth magnitudes.")  # DEBUG ADDED

    # Removed stray return statement
    # --- 5. Create Training Matrices ---
    logger.info("--> Starting: Create Training Matrices (H_train, Mag_train)...")  # DEBUG ADDED
    H_train_list = []
    Mag_train_list = []
    train_plane_info_list = []  # To store (name, start_row, end_row)
    current_row_index = 0

    # Pre-calculate number of coefficients based on perturbed geometry
    num_coeffs_perturbed = points_perturbed.shape[0] * (
        2 if cfg.global_params.use_vector_model else 1
    )
    if original_currents.shape[0] != num_coeffs_true:
        logger.warning(
            f"Shape mismatch: original_currents ({original_currents.shape[0]}) vs expected from points_true ({num_coeffs_true}). Using dummy currents."
        )
        original_currents = np.random.rand(num_coeffs_true) + 1j * np.random.rand(
            num_coeffs_true
        )  # Re-dummy if mismatch

    for i, plane in enumerate(train_planes):
        plane_name = plane.get("name", f"train_plane_{i}")
        logger.info(f"  Processing training plane: {plane_name}")
        measurement_coords = plane["coordinates"]

        # Create H matrix using the perturbed geometry for this plane
        H_plane = create_channel_matrix(
            points=points_perturbed,
            measurement_plane=measurement_coords,
            k=k,
            use_vector_model=cfg.global_params.use_vector_model,
            tangents1=tangents1_perturbed if cfg.global_params.use_vector_model else None,
            tangents2=tangents2_perturbed if cfg.global_params.use_vector_model else None,
            measurement_direction=np.array(cfg.global_params.measurement_direction)
            if cfg.global_params.use_vector_model
            else None,
        )
        H_train_list.append(H_plane)
        num_rows_added = H_plane.shape[0]

        # Get or calculate the magnitude for this training plane
        if plane["is_real_plane"]:
            logger.info("    Using measured magnitude from real plane.")
            plane_mag = plane["measured_magnitude"]
            if plane_mag.shape[0] != num_rows_added:
                logger.warning(
                    f"Shape mismatch for real train plane {plane_name}: H rows ({num_rows_added}) != Mag length ({plane_mag.shape[0]})"
                )
                # Handle error? Skip plane? For now, assume it matches
            Mag_train_list.append(plane_mag)
        else:
            # Calculate true field magnitude for simulated training plane
            logger.info("    Calculating true magnitude for simulated training plane...")
            H_true_sim_plane = create_channel_matrix(
                points=points_true,  # Use true geometry
                measurement_plane=measurement_coords,
                k=k,
                use_vector_model=cfg.global_params.use_vector_model,
                tangents1=tangents1_true if cfg.global_params.use_vector_model else None,
                tangents2=tangents2_true if cfg.global_params.use_vector_model else None,
                measurement_direction=np.array(cfg.global_params.measurement_direction)
                if cfg.global_params.use_vector_model
                else None,
            )
            # Ensure current shape matches H_true_sim_plane columns
            if H_true_sim_plane.shape[1] != original_currents.shape[0]:
                raise ValueError(
                    f"Shape mismatch for true field calculation on plane {plane_name}: "
                    f"H columns ({H_true_sim_plane.shape[1]}) != "
                    f"original_currents ({original_currents.shape[0]})"
                )

            true_field_sim_plane = H_true_sim_plane @ original_currents
            true_magnitude_sim_plane = np.abs(true_field_sim_plane)
            if true_magnitude_sim_plane.shape[0] != num_rows_added:
                logger.warning(
                    f"Shape mismatch for sim train plane {plane_name}: H rows ({num_rows_added}) != Mag length ({true_magnitude_sim_plane.shape[0]})"
                )
                # Handle error? Skip plane? For now, assume it matches
            Mag_train_list.append(true_magnitude_sim_plane)
            logger.info(f"    Calculated magnitude shape: {true_magnitude_sim_plane.shape}")

        # Store info for this plane segment
        train_plane_info_list.append(
            (plane_name, current_row_index, current_row_index + num_rows_added)
        )
        current_row_index += num_rows_added

    # Combine all training matrices and magnitudes
    if not H_train_list:
        logger.error("H_train_list is empty. Cannot proceed.")
        # Or raise an error, depending on desired behavior if train_planes was somehow empty despite earlier check
        return

    H_train = np.vstack(H_train_list)
    Mag_train = np.concatenate(Mag_train_list)

    logger.info(
        f"Combined Training Matrices. H_train shape: {H_train.shape}, Mag_train shape: {Mag_train.shape}"
    )

    # --- 6. Pre-calculate H_test matrices (needed for history) ---
    test_planes_data = {}
    if test_planes and cfg.global_params.return_history:  # Only needed if history is requested
        logger.info("Pre-calculating H matrices for test planes (for history)...")
        for plane in test_planes:
            plane_name = plane.get("name", "unknown_test_plane")
            logger.info(f"  Calculating H_test for: {plane_name}")
            try:
                H_test = create_channel_matrix(
                    points=points_perturbed,  # Use perturbed points for H_test
                    measurement_plane=plane["coordinates"],
                    k=k,
                    use_vector_model=cfg.global_params.use_vector_model,
                    tangents1=tangents1_perturbed if cfg.global_params.use_vector_model else None,
                    tangents2=tangents2_perturbed if cfg.global_params.use_vector_model else None,
                    measurement_direction=np.array(cfg.global_params.measurement_direction)
                    if cfg.global_params.use_vector_model
                    else None,
                )
                # Store H_test along with other plane info needed by GS history calculation
                test_planes_data[plane_name] = {"H_test": H_test, **plane}
            except Exception as e:
                logger.error(f"Failed to create H_test for plane {plane_name}: {e}", exc_info=True)
                # Continue without this plane's H_test in history

    # --- 7. Run Gerchberg-Saxton Algorithm ---
    logger.info("Running Gerchberg-Saxton Algorithm...")
    hpr_result = holographic_phase_retrieval(
        cfg=cfg.global_params,
        channel_matrix=H_train,
        measured_magnitude=Mag_train,
        train_plane_info=train_plane_info_list,
        test_planes_data=test_planes_data
        if cfg.global_params.return_history
        else None,  # Pass test data if history needed
        output_dir=plots_dir,
        initial_field_values=None,
    )

    # Unpack results based on return_history flag (new signature)
    if cfg.global_params.return_history:
        final_coefficients, full_history, stats = hpr_result
        logger.info(f"GS returned detailed history (Length: {len(full_history)} iterations)")
        # Extract coefficient_history for saving if needed elsewhere (e.g., legacy plots or specific visualizations)
        # coefficient_history = np.array([h['coefficients'] for h in full_history]) if full_history else None
    else:
        final_coefficients, stats = hpr_result
        full_history = None
        coefficient_history = None  # Ensure defined

    logger.info(
        f"GS Algorithm completed. Final Overall Train RMSE: {stats.get('final_overall_rmse', 'N/A'):.6f}, Best Overall Train RMSE: {stats.get('best_overall_rmse', 'N/A'):.6f}"
    )

    # --- 8. Evaluate on Test Planes (using final coefficients) ---
    if test_planes:
        logger.info("Evaluating final coefficients on Test Planes...")
        # Pass the original processed test_planes list, not test_planes_data which might miss planes if H calc failed
        evaluation_results = evaluation.evaluate_on_test_planes(
            test_planes=test_planes,  # List of processed plane dicts
            final_coefficients=final_coefficients,
            points_true=points_true,
            tangents1_true=tangents1_true,
            tangents2_true=tangents2_true,
            points_perturbed=points_perturbed,
            tangents1_perturbed=tangents1_perturbed,
            tangents2_perturbed=tangents2_perturbed,
            original_currents=original_currents,
            config=cfg,  # Pass full config
        )
        # Log summary metrics from final evaluation results
        for plane_name, metrics in evaluation_results.items():
            logger.info(
                f"  FINAL Plane '{plane_name}': RMSE={metrics.get('rmse', 'N/A'):.4f}, Corr={metrics.get('correlation', 'N/A'):.4f}"
            )
    else:
        logger.info("No test planes defined ('use_test: true'). Skipping evaluation.")
        evaluation_results = {}

    # --- 9. Reporting & Visualization ---
    logger.info("Saving results and generating visualizations...")
    # Save key results to a pickle file
    results_to_save = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "gs_stats": stats,  # Includes aggregated RMSE history
        "final_evaluation_results": evaluation_results,  # Final metrics on test planes
        "final_coefficients": final_coefficients,
        "points_true": points_true,
        "points_perturbed": points_perturbed,
        "original_currents": original_currents,
        # Add full history if generated
        "full_history": full_history if cfg.global_params.return_history else None,
        # "tangents2_perturbed": tangents2_perturbed, # Example of adding more if needed
    }
    results_filename = os.path.join(output_dir, "multi_plane_results.pkl")
    try:
        with open(results_filename, "wb") as f:
            pickle.dump(results_to_save, f)
        logger.info(f"Saved detailed results to: {results_filename}")
    except Exception as e:
        logger.error(f"Failed to save results to {results_filename}: {e}", exc_info=True)

    # --- Static Visualization Calls ---
    if not cfg.global_params.no_plot:
        logger.info("Generating static visualizations...")
        # Visualize 3D Layout and Final Currents
        try:
            # Ensure final_coefficients is defined even if history wasn't returned
            if "final_coefficients" not in locals() and stats is not None:
                final_coefficients = stats.get(
                    "final_coefficients"
                )  # Attempt to get from stats if needed

            if final_coefficients is not None:
                # Check if visualize_layout_and_currents exists before calling
                if hasattr(multi_plane_plots, "visualize_layout_and_currents"):
                    multi_plane_plots.visualize_layout_and_currents(
                        points_perturbed=points_perturbed,
                        final_coefficients=final_coefficients,
                        measurement_planes=all_processed_planes,  # Pass all defined planes
                        output_dir=plots_dir,
                        show_plot=cfg.global_params.show_plot,
                    )
                else:
                    logger.warning(
                        "Function 'visualize_layout_and_currents' not found in multi_plane_plots."
                    )
            else:
                logger.warning("Skipping layout/currents plot: final_coefficients not available.")
        except Exception as e:
            logger.error(f"Failed to generate layout/currents plot: {e}", exc_info=True)

        # Visualize Per-Test-Plane Comparison (Static - using final results)
        if evaluation_results:
            try:
                # Check if visualize_per_test_plane_comparison exists before calling
                if hasattr(multi_plane_plots, "visualize_per_test_plane_comparison"):
                    multi_plane_plots.visualize_per_test_plane_comparison(
                        evaluation_results=evaluation_results,  # Contains final fields & metrics
                        plots_dir=plots_dir,
                        show_plot=cfg.global_params.show_plot,
                    )
                else:
                    logger.warning(
                        "Function 'visualize_per_test_plane_comparison' not found in multi_plane_plots."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to generate per-test-plane comparison plots: {e}", exc_info=True
                )
        else:
            logger.info("Skipping per-test-plane comparison plots as no evaluation results exist.")
    else:
        logger.info("Static plot generation disabled (no_plot=true).")

    # --- Animation Calls ---
    if cfg.global_params.return_history and not cfg.global_params.no_anim:
        logger.info("Generating history animations...")
        if full_history is not None:
            # Convergence Error Animation
            try:
                multi_plane_plots.animate_convergence_errors(
                    stats=stats,  # Stats dict now contains the aggregated histories
                    convergence_threshold=cfg.global_params.convergence_threshold,
                    output_dir=plots_dir,
                    show_plot=cfg.global_params.show_plot,
                )
            except Exception as e:
                logger.error(f"Failed to generate convergence animation: {e}", exc_info=True)

            # Layout and Currents Animation
            try:
                if hasattr(multi_plane_plots, "animate_layout_and_currents"):
                    multi_plane_plots.animate_layout_and_currents(
                        history=full_history,  # Pass the detailed history
                        points_perturbed=points_perturbed,
                        measurement_planes=all_processed_planes,
                        output_dir=plots_dir,
                        show_plot=cfg.global_params.show_plot,
                    )
                else:
                    logger.warning(
                        "Function 'animate_layout_and_currents' not found in multi_plane_plots."
                    )
            except Exception as e:
                logger.error(f"Failed to generate layout/currents animation: {e}", exc_info=True)

            # Per-Plane Comparison Animation
            # Determine which planes to animate based on config flags
            planes_to_animate = []
            for plane_cfg in cfg.measurement_planes:
                plane_name = plane_cfg.get("name")
                vis_cfg = plane_cfg.get("visualization", {})
                # Default to animating if use_test is true and flag isn't explicitly false
                should_animate = vis_cfg.get("animate_comparison", plane_cfg.get("use_test", False))
                if should_animate:
                    # Find the corresponding processed plane data
                    processed_plane = next(
                        (p for p in all_processed_planes if p.get("name") == plane_name), None
                    )
                    if processed_plane:
                        planes_to_animate.append(processed_plane)
                    else:
                        logger.warning(
                            f"Could not find processed data for plane '{plane_name}' requested for animation."
                        )

            if planes_to_animate:
                try:
                    if hasattr(multi_plane_plots, "animate_plane_comparison"):
                        multi_plane_plots.animate_plane_comparison(
                            history=full_history,  # Pass detailed history
                            planes_to_visualize=planes_to_animate,  # Pass list of processed plane dicts
                            plots_dir=plots_dir,
                            show_plot=cfg.global_params.show_plot,
                        )
                    else:
                        logger.warning(
                            "Function 'animate_plane_comparison' not found in multi_plane_plots."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to generate per-plane comparison animations: {e}", exc_info=True
                    )
            else:
                logger.info("No planes configured for comparison animation.")

            # Grid Comparison Animation (Optional)
            if cfg.global_params.get("animate_grid_comparison", False):
                if planes_to_animate:
                    try:
                        if hasattr(multi_plane_plots, "animate_multi_plane_comparison_grid"):
                            multi_plane_plots.animate_multi_plane_comparison_grid(
                                history=full_history,
                                planes_to_visualize=planes_to_animate,
                                plots_dir=plots_dir,
                                show_plot=cfg.global_params.show_plot,
                                # Add other params like grid_cols if needed, using defaults for now
                            )
                        else:
                            logger.warning(
                                "Function 'animate_multi_plane_comparison_grid' not found in multi_plane_plots."
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to generate grid comparison animation: {e}", exc_info=True
                        )
                else:
                    logger.info(
                        "Skipping grid comparison animation as no planes were selected for animation."
                    )

        else:
            logger.warning("Full history not available, skipping animations.")

    elif not cfg.global_params.return_history:
        logger.info("History not saved (return_history=false), skipping animations.")
    elif cfg.global_params.no_anim:
        logger.info("Animation generation disabled (no_anim=true).")

    logger.info("Multi-Plane Reconstruction Workflow Completed.")


def main_logic(cfg: DictConfig, base_dir_override: str | None = None) -> None:
    """Core logic previously in main(). Determines workflow based on config structure.

    Args:
        cfg: The DictConfig object.
        base_dir_override: If provided, overrides the base directory used for
                           multi-plane workflow. Useful for testing.
    """
    # Check for multi-plane config structure first
    if "measurement_planes" in cfg:
        logger.info("Detected multi-plane configuration structure.")
        logger.info(
            f"Running multi-plane workflow with config:\n{OmegaConf.to_yaml(cfg)}",
        )
        # Pass the override if provided, otherwise run_multi_plane_workflow will get it itself
        run_multi_plane_workflow(cfg, base_dir=base_dir_override)
    # Check for legacy measured config structure (using source_pointcloud_path as the key)
    elif "source_pointcloud_path" in cfg and cfg.get(
        "use_source_pointcloud", False
    ):  # Check for key and that it's used
        logger.info("Detected legacy measured data configuration structure.")
        if run_measured_legacy:
            logger.info("Delegating to measured_data_reconstruction.py...")
            run_measured_legacy(cfg)
        else:
            logger.error(
                "Legacy measured mode requested, but measured_data_reconstruction.py could not be imported."
            )
    # Assume legacy simulated config otherwise
    else:
        logger.info("Assuming legacy simulated data configuration structure.")
        if run_simulated_legacy:
            logger.info("Delegating to simulated_data_reconstruction.py...")
            run_simulated_legacy(cfg)
        else:
            logger.error(
                "Legacy simulated mode assumed, but simulated_data_reconstruction.py could not be imported."
            )


# Hydra entry point - simply calls the main logic
@hydra.main(config_path="conf", config_name="multi_plane_config", version_base="1.2")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    main_logic(cfg)  # base_dir_override is None, so get_original_cwd() will be used


if __name__ == "__main__":
    hydra_main()
