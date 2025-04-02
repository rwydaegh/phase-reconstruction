# multi_plane_reconstruction.py
import logging
import os
import random
import sys
import pickle

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Module imports
from src import plane_processing
from src.source_handling import get_source_pointcloud
from src import evaluation
from src.algorithms.gerchberg_saxton import holographic_phase_retrieval
from src.create_channel_matrix import create_channel_matrix
from src.visualization import field_plots, history_plots, comparison_plots # Add necessary visualization imports
# TODO: Add other necessary imports as functionality is filled in

# Placeholder for legacy script execution
# from measured_data_reconstruction import main as run_measured_legacy
# from simulated_data_reconstruction import main as run_simulated_legacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_legacy_workflow(cfg: DictConfig):
    """Delegates to the appropriate legacy script based on some logic."""
    logger.warning("Running in Legacy Mode. This is a placeholder.")
    # TODO: Determine which legacy script to run based on config name or content
    # Example logic (needs refinement):
    # if "measured_data" in cfg._metadata.config_name: # Requires loading original config name
    #     logger.info("Delegating to measured_data_reconstruction.py (legacy)...")
    #     # run_measured_legacy(cfg) # Need to adapt legacy main functions
    # elif "simulated_data" in cfg._metadata.config_name:
    #     logger.info("Delegating to simulated_data_reconstruction.py (legacy)...")
    #     # run_simulated_legacy(cfg) # Need to adapt legacy main functions
    # else:
    #     logger.error("Cannot determine which legacy script to run.")
    pass


def run_multi_plane_workflow(cfg: DictConfig):
    """Orchestrates the new multi-plane reconstruction workflow."""
    logger.info("Starting Multi-Plane Reconstruction Workflow...")
    output_dir = os.getcwd() # Hydra sets cwd to the output directory for the run
    base_dir = hydra.utils.get_original_cwd() # Get the original directory to resolve config paths
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
        points_true, tangents1_true, tangents2_true, \
        points_perturbed, tangents1_perturbed, tangents2_perturbed, \
        original_currents = get_source_pointcloud(
            cfg=cfg.source_pointcloud,         # Pass the source_pointcloud sub-config
            global_cfg=cfg.global_params,    # Pass the global_params sub-config
            base_dir=base_dir
        )
        logger.info(f"Source points processed. True shape: {points_true.shape}, Perturbed shape: {points_perturbed.shape}, Currents shape: {original_currents.shape}")
        # Verify current shape matches expectation from true points and model type
        num_coeffs_true = points_true.shape[0] * (2 if cfg.global_params.use_vector_model else 1)
        if original_currents.shape[0] != num_coeffs_true:
             logger.warning(f"Mismatch between generated currents shape ({original_currents.shape[0]}) and expected shape ({num_coeffs_true}). Check current generation logic.")
             # Decide how to handle: error out or try to proceed? For now, log warning.

    except (FileNotFoundError, ValueError, TypeError) as e:
        logger.error(f"Failed to process source point cloud: {e}", exc_info=True)
        return # Cannot proceed without source points


    # --- 3. Process Measurement Plane Definitions ---
    logger.info("Processing Measurement Plane Definitions...")
    all_processed_planes = plane_processing.process_plane_definitions(
        plane_configs=cfg.measurement_planes,
        base_dir=base_dir
    )
    if not all_processed_planes:
         logger.warning("Plane processing resulted in no valid planes.")
         # Decide whether to exit or continue if possible (e.g., if only testing was intended but failed)
         # For now, let the check for train_planes handle the critical path.


    # --- 4. Separate Train/Test Planes ---
    train_planes = [p for p in all_processed_planes if p.get('use_train', False)]
    test_planes = [p for p in all_processed_planes if p.get('use_test', False)]
    logger.info(f"Found {len(train_planes)} training planes and {len(test_planes)} testing planes.")

    if not train_planes:
        logger.error("No training planes defined ('use_train: true'). Cannot proceed.")
        return

    # --- 5. Create Training Matrices ---
    logger.info("Creating Training Matrices (H_train, Mag_train)...")
    H_train_list = []
    Mag_train_list = []

    # Pre-calculate number of coefficients based on perturbed geometry
    num_coeffs_perturbed = points_perturbed.shape[0] * (2 if cfg.global_params.use_vector_model else 1)
    if original_currents.shape[0] != num_coeffs_true:
         logger.warning(f"Shape mismatch: original_currents ({original_currents.shape[0]}) vs expected from points_true ({num_coeffs_true}). Using dummy currents.")
         original_currents = np.random.rand(num_coeffs_true) + 1j*np.random.rand(num_coeffs_true) # Re-dummy if mismatch

    for i, plane in enumerate(train_planes):
        plane_name = plane.get('name', f'train_plane_{i}')
        logger.info(f"  Processing training plane: {plane_name}")
        measurement_coords = plane['coordinates']

        # Create H matrix using the perturbed geometry for this plane
        H_plane = create_channel_matrix(
            points=points_perturbed,
            measurement_plane=measurement_coords,
            k=k,
            use_vector_model=cfg.global_params.use_vector_model,
            tangents1=tangents1_perturbed if cfg.global_params.use_vector_model else None,
            tangents2=tangents2_perturbed if cfg.global_params.use_vector_model else None,
            measurement_direction=np.array(cfg.global_params.measurement_direction) if cfg.global_params.use_vector_model else None,
        )
        H_train_list.append(H_plane)

        # Get or calculate the magnitude for this training plane
        if plane['is_real_plane']:
            logger.info(f"    Using measured magnitude from real plane.")
            Mag_train_list.append(plane['measured_magnitude'])
        else:
            # Calculate true field magnitude for simulated training plane
            logger.info(f"    Calculating true magnitude for simulated training plane...")
            H_true_sim_plane = create_channel_matrix(
                points=points_true, # Use true geometry
                measurement_plane=measurement_coords,
                k=k,
                use_vector_model=cfg.global_params.use_vector_model,
                tangents1=tangents1_true if cfg.global_params.use_vector_model else None,
                tangents2=tangents2_true if cfg.global_params.use_vector_model else None,
                measurement_direction=np.array(cfg.global_params.measurement_direction) if cfg.global_params.use_vector_model else None,
            )
            # Ensure current shape matches H_true_sim_plane columns
            if H_true_sim_plane.shape[1] != original_currents.shape[0]:
                 raise ValueError(f"Shape mismatch for true field calculation on plane {plane_name}: "
                                  f"H columns ({H_true_sim_plane.shape[1]}) != "
                                  f"original_currents ({original_currents.shape[0]})")

            true_field_sim_plane = H_true_sim_plane @ original_currents
            true_magnitude_sim_plane = np.abs(true_field_sim_plane)
            Mag_train_list.append(true_magnitude_sim_plane)
            logger.info(f"    Calculated magnitude shape: {true_magnitude_sim_plane.shape}")

    # Combine all training matrices and magnitudes
    if not H_train_list:
         logger.error("H_train_list is empty. Cannot proceed.")
         # Or raise an error, depending on desired behavior if train_planes was somehow empty despite earlier check
         return

    H_train = np.vstack(H_train_list)
    Mag_train = np.concatenate(Mag_train_list)

    logger.info(f"Combined Training Matrices. H_train shape: {H_train.shape}, Mag_train shape: {Mag_train.shape}")


    # --- 6. Run Gerchberg-Saxton Algorithm ---
    logger.info("Running Gerchberg-Saxton Algorithm...")
    hpr_result = holographic_phase_retrieval(
        cfg=cfg.global_params, # Pass the global params sub-config
        channel_matrix=H_train,
        measured_magnitude=Mag_train,
        output_dir=plots_dir, # Pass plots directory for convergence plot etc.
        # initial_field_values=None # Add if implementing initialization strategies
    )

    # Unpack results based on return_history flag
    if cfg.global_params.return_history:
        final_coefficients, coefficient_history, field_history, stats = hpr_result
        logger.info(f"GS returned history. Coeff shape: {coefficient_history.shape}, Field shape: {field_history.shape}")
    else:
        final_coefficients, stats = hpr_result
        coefficient_history, field_history = None, None # Ensure they are defined

    logger.info(f"GS Algorithm completed. Final RMSE: {stats.get('final_rmse', 'N/A'):.6f}, Best RMSE: {stats.get('best_rmse', 'N/A'):.6f}")

    # --- 7. Evaluate on Test Planes ---
    if test_planes:
        logger.info("Evaluating on Test Planes...")
        evaluation_results = evaluation.evaluate_on_test_planes(
            test_planes=test_planes,
            final_coefficients=final_coefficients,
            points_true=points_true,
            tangents1_true=tangents1_true,
            tangents2_true=tangents2_true,
            points_perturbed=points_perturbed,
            tangents1_perturbed=tangents1_perturbed,
            tangents2_perturbed=tangents2_perturbed,
            original_currents=original_currents, # Pass the original currents
            config=cfg # Pass full config (evaluation might need global_params)
        )
        # Log summary metrics from results
        for plane_name, metrics in evaluation_results.items():
             logger.info(f"  Plane '{plane_name}': RMSE={metrics.get('rmse', 'N/A'):.4f}, Corr={metrics.get('correlation', 'N/A'):.4f}")
    else:
        logger.info("No test planes defined ('use_test: true'). Skipping evaluation.")
        evaluation_results = {}

    # --- 8. Reporting & Visualization ---
    logger.info("Saving results and generating visualizations...")
    # Save key results to a pickle file
    results_to_save = {
        "config": OmegaConf.to_container(cfg, resolve=True), # Save resolved config
        "gs_stats": stats,
        "evaluation_results": evaluation_results,
        "final_coefficients": final_coefficients,
        "points_true": points_true,
        "points_perturbed": points_perturbed,
        "original_currents": original_currents,
        # Optionally add tangents if needed for reproducibility/analysis
        # "tangents1_true": tangents1_true,
        # "tangents2_true": tangents2_true,
        # "tangents1_perturbed": tangents1_perturbed,
        # "tangents2_perturbed": tangents2_perturbed,
    }
    results_filename = os.path.join(output_dir, "multi_plane_results.pkl")
    try:
        with open(results_filename, "wb") as f:
            pickle.dump(results_to_save, f)
        logger.info(f"Saved detailed results to: {results_filename}")
    except Exception as e:
        logger.error(f"Failed to save results to {results_filename}: {e}", exc_info=True)


    # --- Visualization Calls ---

    # Example: Visualize first test plane if available
    if test_planes and not cfg.global_params.no_plot:
         logger.info("Visualizing first test plane...")
         first_test_plane_name = test_planes[0].get('name')
         first_test_plane_data = evaluation_results.get(first_test_plane_name)
         if first_test_plane_data:
               # Reshape fields back to 2D using stored original_data_shape
               res_y, res_z = first_test_plane_data['original_data_shape'] # Example shape
               true_mag_2d = first_test_plane_data['ground_truth_magnitude'].reshape(res_y, res_z)
               recon_field_2d = first_test_plane_data['reconstructed_field'].reshape(res_y, res_z)
               # Need measured magnitude 2D if it was a real plane
               measured_mag_2d = test_planes[0].get('measured_magnitude', np.abs(true_mag_2d)).reshape(res_y, res_z) # Fallback

               # field_plots.visualize_fields(
               #      points=points_perturbed, # Show points used for reconstruction H
               #      currents=final_coefficients, # Show reconstructed coefficients
               #      measurement_plane=test_planes[0]['coordinates'].reshape(res_y, res_z, 3), # Reshape coords
               #      true_field_2d=true_mag_2d, # Ground truth magnitude
               #      measured_magnitude_2d=measured_mag_2d, # Use actual measured if available
               #      reconstructed_field_2d=recon_field_2d, # Complex reconstructed field
               #      rmse=first_test_plane_data['rmse'],
               #      correlation=first_test_plane_data['correlation'],
               #      show_plot=cfg.global_params.show_plot,
               #      output_dir=plots_dir,
               #      filename_suffix=f"_{first_test_plane_name}" # Add plane name to filename
               # )
               logger.info("Placeholder call to field_plots.visualize_fields")
         else:
              logger.warning(f"Could not find evaluation data for first test plane '{first_test_plane_name}' to visualize.")


    if cfg.global_params.return_history and not cfg.global_params.no_anim:
        logger.info("Generating history animations (if history was returned)...")
        if coefficient_history is not None and field_history is not None:
             # Determine representative resolution for animation title/display
             # Using the shape of the first training plane's magnitude might work
             first_train_plane = train_planes[0]
             res_y_anim, res_z_anim = first_train_plane.get('original_data_shape', (int(np.sqrt(field_history.shape[1])), -1))
             if res_y_anim != res_z_anim or res_y_anim == -1:
                  logger.warning("Cannot determine square resolution for animation title. Using placeholder.")
                  anim_res = 50 # Placeholder
             else:
                  anim_res = res_y_anim

             # history_plots.visualize_iteration_history(
             #      points=points_perturbed,
             #      channel_matrix=H_train, # Use the combined training matrix
             #      coefficient_history=coefficient_history,
             #      field_history=field_history, # Combined field history on training planes
             #      resolution=anim_res, # Representative resolution
             #      measurement_plane=None, # Difficult to show combined plane geometry easily
             #      show_plot=cfg.global_params.show_plot,
             #      output_dir=plots_dir,
             #      animation_filename=os.path.join(plots_dir, "gs_animation.gif"),
             #      perturbation_iterations=stats.get("perturbation_iterations", []),
             #      convergence_threshold=cfg.global_params.convergence_threshold,
             #      measured_magnitude=Mag_train # Use combined training magnitude for error calc
             # )
             logger.info("Placeholder call to history_plots.visualize_iteration_history")

             # visualize_current_and_field_history(...) # Needs significant adaptation for multi-plane
             logger.info("Placeholder call to history_plots.visualize_current_and_field_history (needs adaptation)")
        else:
             logger.info("History not available, skipping animations.")


    logger.info("Multi-Plane Reconstruction Workflow Completed.")


@hydra.main(config_path="conf", config_name="multi_plane_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Decide workflow based on legacy_mode flag
    if cfg.get("legacy_mode", False):
        logger.info("Legacy mode enabled. Attempting to run legacy workflow...")
        # Load the actual legacy config file specified (e.g., measured_data or simulated_data)
        # This requires knowing which one to load, which isn't directly in multi_plane_config
        # Option 1: Add a 'legacy_config_name' field to multi_plane_config
        # Option 2: Infer from the original command line override (tricky with Hydra)
        # Option 3: For now, just print a warning and exit or run a default legacy one.
        logger.warning("Legacy mode delegation is not fully implemented yet.")
        # Example: Load and run measured_data legacy config if available
        try:
            # Construct path relative to the original config path
            legacy_config_path = os.path.join(hydra.utils.get_original_cwd(), "conf")
            with hydra.initialize_config_dir(config_dir=legacy_config_path, version_base="1.2"):
                 legacy_cfg = hydra.compose(config_name="measured_data.yaml") # Or simulated_data.yaml
                 logger.info(f"Loaded legacy config: {legacy_cfg.pretty()}")
                 # run_legacy_workflow(legacy_cfg) # Pass the loaded legacy config
        except Exception as e:
            logger.error(f"Could not load or run legacy config: {e}")

    else:
        logger.info("Multi-plane mode enabled.")
        logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg)}")
        run_multi_plane_workflow(cfg)

if __name__ == "__main__":
    main()