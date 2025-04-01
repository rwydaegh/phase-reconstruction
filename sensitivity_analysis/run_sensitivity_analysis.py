import logging
import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path to find modules from root
# Ensure the root directory is in the path for src imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from sensitivity_analysis module within the same package
from .sensitivity_analysis import (
    ParameterRange,
    SensitivityAnalysisConfig,
    run_sensitivity_analysis,
)
# Import the enhanced visualization function
from .improved_sensitivity_visualization import enhance_sensitivity_visualization

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)




@hydra.main(config_path="../conf", config_name="sensitivity_analysis", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run sensitivity analysis using Hydra configuration"""

    # Log the configuration
    logger.info("Starting sensitivity analysis with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Create output directory (Hydra handles the main output dir, but we might want a subdir)
    # Using Hydra's output dir: hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Use Hydra's output directory mechanism
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Hydra output directory: {output_dir}")
    # Ensure the specific output subdir from config is created if needed,
    # but Hydra manages the main run directory.
    # If cfg.output_dir is meant to be relative *within* the Hydra dir:
    # specific_output_dir = os.path.join(output_dir, cfg.output_dir)
    # os.makedirs(specific_output_dir, exist_ok=True)
    # For now, assume run_sensitivity_analysis will use the main Hydra output dir.
    os.makedirs(output_dir, exist_ok=True)

    # Create base configuration (Now part of the loaded cfg)

    # Define parameter pairs to analyze (Now loaded from cfg)

    # Create sensitivity analysis configuration object from Hydra config

    # Instantiate ParameterRange objects
    param_ranges_inst = [ParameterRange(**p_range) for p_range in cfg.parameter_ranges]

    # Construct the base_config by filtering out sensitivity-specific keys from cfg
    # Convert DictConfig to a standard dictionary first
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Keys specific to sensitivity analysis run configuration
    sensitivity_keys = {"output_dir", "parallel", "max_workers", "parameter_ranges", "hydra", "defaults"}

    # Create the base_config dictionary
    base_config_dict = {
        key: value for key, value in cfg_dict.items() if key not in sensitivity_keys
    }
    # Convert back to DictConfig if SensitivityAnalysisConfig expects it,
    # or keep as dict if sensitivity_analysis.py handles it.
    # Assuming sensitivity_analysis.py now expects DictConfig based on previous edits.
    base_config_hydra = OmegaConf.create(base_config_dict)


    analysis_config = SensitivityAnalysisConfig(
        base_config=base_config_hydra, # Pass the filtered base config
        parameter_ranges=param_ranges_inst,
        output_dir=output_dir, # Use Hydra's runtime output directory
        parallel=cfg.parallel,
        max_workers=cfg.max_workers,
    )

    # Run sensitivity analysis (moved inside main)
    logger.info("Starting sensitivity analysis")
    run_sensitivity_analysis(analysis_config)
    logger.info(f"Sensitivity analysis complete. Results saved to {output_dir}")

    # Use the enhanced visualization script (moved inside main)
    logger.info("Creating enhanced visualizations...")
    enhance_sensitivity_visualization(output_dir=output_dir, save_dir=output_dir)
    logger.info(f"Visualizations created in '{output_dir}' directory")


if __name__ == "__main__":
    main()





